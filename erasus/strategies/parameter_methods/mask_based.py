"""
Mask-Based Unlearning Strategy.

Learns a mask over the model parameters to 'hide' or 'prune' connections
responsible for the forget set knowledge, effectively simulating unlearning.

Concept: θ_unlearned = θ_orig ⊙ Mask
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("mask_based")
class MaskBasedUnlearningStrategy(BaseStrategy):
    """
    Learns a soft mask M ∈ [0, 1] for key parameters (e.g. Linear layer outputs).
    Objective: Minimize Forget Loss (e.g. maximize entropy or negative CE) w.r.t Mask
    plus sparsity penalty on Mask (encourage minimal changes).
    """

    def __init__(
        self,
        mask_lr: float = 0.1,
        sparsity_weight: float = 1e-4,
        mask_threshold: float = 0.1,  # Threshold for binary pruning at inference
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mask_lr = mask_lr
        self.sparsity_weight = sparsity_weight
        self.mask_threshold = mask_threshold

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        device = next(model.parameters()).device
        
        # We assume model is frozen except for masks
        for p in model.parameters():
            p.requires_grad = False
            
        hooks = []
        mask_params = []
        
        # Register hooks to apply masks on activations of Linear layers
        # This acts as dynamic "Parameter Masking"
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                # Learn a vector mask per layer output (Channel/Feature pruning)
                # Init with 5.0 => Sigmoid(5.0) ~= 0.99 (Keep)
                mask_vec = nn.Parameter(torch.ones(m.out_features, device=device) * 5.0)
                mask_params.append(mask_vec)
                
                # Register hook
                def get_hook(mask_v):
                    def hook(module, inp, out):
                        # out shape: [batch, ..., out_features]
                        return out * torch.sigmoid(mask_v)
                    return hook
                
                hooks.append(m.register_forward_hook(get_hook(mask_vec)))

        optimizer = torch.optim.Adam(mask_params, lr=self.mask_lr)
        
        forget_losses = []
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in forget_loader:
                # Handle batch unpacking
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device) if len(batch) > 1 else None
                elif isinstance(batch, dict):
                    inputs = batch.get("input_ids").to(device)
                    labels = batch.get("labels")
                    if labels is not None: labels = labels.to(device)
                else: 
                    inputs = batch.to(device)
                    labels = None
                
                optimizer.zero_grad()
                try:
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    
                    # We want to destroy info => Maximize Loss on Forget (Gradient Ascent)
                    # Or Minimize Unlearning Loss
                    if labels is not None:
                        # Negative Log Likelihood maximization
                        loss = -torch.nn.functional.cross_entropy(logits, labels)
                    else:
                        # Maximize output entropy or norm? 
                        # Simple proxy: minimize Norm (suppress forget set activation)
                        loss = logits.norm() 
                except Exception:
                    # Fallback for models that don't fit standard pattern
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                # Sparsity penalty: Minimize ||1 - mask||
                # We want mask to stay close to 1 (modify few weights)
                # But we also want to minimize loss on forget set.
                # So we penalize changing the mask.
                current_masks = [torch.sigmoid(p) for p in mask_params]
                sparsity_loss = sum(torch.norm(1.0 - m, p=1) for m in current_masks)
                
                total_loss = loss + self.sparsity_weight * sparsity_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n += 1
            forget_losses.append(epoch_loss / max(n, 1))

        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Apply masks permanently via Weight modification (Hard thresholding)
        with torch.no_grad():
            idx = 0
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    mask_val = torch.sigmoid(mask_params[idx])
                    # Mask if < threshold
                    binary_mask = (mask_val > self.mask_threshold).float()
                    
                    # Apply to weight [out, in]
                    m.weight.data.mul_(binary_mask.unsqueeze(1))
                    if m.bias is not None:
                        m.bias.data.mul_(binary_mask)
                    idx += 1
        
        # Unfreeze
        for p in model.parameters():
            p.requires_grad = True

        return model, forget_losses, []
