"""
Gradient Norm Selector (Gradient-Based).

Selects samples with the highest gradient magnitude / norm.
Intuition: Samples with large gradients have high influence on the model weights and training dynamic.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("gradient_norm")
class GradientNormSelector(BaseSelector):
    """
    Selects top-k samples with highest L2 norm of the gradient.
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        
        device = next(model.parameters()).device
        grad_norms = []
        
        model.eval()
        
        # NOTE: Calculating full-model per-sample gradients is extremely expensive (O(N) * Backward).
        # We optimize this by only computing the gradient norm of the LAST LAYER (Head) 
        # or embeddings, which is a common proxy in the literature (e.g., DeepFool, Influence).
        
        # Heuristic to find 'last' layer:
        # We'll just differentiate the loss w.r.t parameters that require grad.
        # But we'll try to use a "batch-wise" trick if possible, or fall back to loop.
        # Given universal compatibility requirements, loop is safest without functorch.
        
        for i, batch in enumerate(data_loader):
            # Move to device
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None
            elif isinstance(batch, dict):
                # Handle generic HF style dicts (avoid "or" on Tensors: boolean is ambiguous)
                inputs = batch.get("input_ids")
                if inputs is None:
                    inputs = batch.get("pixel_values")
                if inputs is None:
                    raise ValueError(
                        "Batch dict must contain 'input_ids' or 'pixel_values' for gradient_norm selector."
                    )
                inputs = inputs.to(device)
                labels = batch.get("labels")
                if labels is not None:
                    labels = labels.to(device)
            else:
                inputs = batch.to(device)
                labels = None

            batch_size = inputs.size(0)
            
            # Forward pass
            # We need to make sure we can backdrop through it.
            # model(inputs) might return various things.
            
            # We process sample by sample to avoid graph retention issues and OOM with full backprop
            # PRO: Exact per-sample gradient.
            # CON: Slow.
            
            # To speed up: We freeze all but classification head?
            # Let's assume user wants accuracy over speed for "Selector".
            
            # Optimization: Use torch.func if available (PyTorch 2.0+)
            # But let's stick to standard loop for compatibility in this version.
            
            for b in range(batch_size):
                # Slice single sample
                # Need to handle dict slicing or tensor slicing
                if isinstance(batch, dict):
                    single_input = {k: v[b:b+1].to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    outputs = model(**single_input)
                else:
                    single_input = inputs[b:b+1]
                    outputs = model(single_input)

                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                # Check shape of logits vs labels
                if labels is not None:
                    single_label = labels[b:b+1]
                    # Handle dimensionality mismatch if necessary
                    if logits.dim() > 2 and single_label.dim() == 1:
                        # e.g. [1, seq, vocab] vs [1] -> usually handled by loss fn or reshape needed
                        # For now assume standard classification [1, C] vs [1]
                         loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), single_label.view(-1))
                    else:
                        loss = nn.functional.cross_entropy(logits, single_label)
                else:
                    # Unsupervised / Self-supervised: Maximize norm of output? Or sum?
                    # Gradient of sum(logits) approximates sensitivity
                    loss = logits.sum()
                
                # Compute Gradient Norm
                # Reset gradients
                model.zero_grad()
                
                # We only need gradients of parameters that require them
                params_to_check = [p for p in model.parameters() if p.requires_grad]
                if not params_to_check:
                    # Model frozen?
                    grad_norms.append(0.0)
                    continue
                    
                grads = torch.autograd.grad(loss, params_to_check, allow_unused=True)
                
                total_norm = 0.0
                for g in grads:
                    if g is not None:
                        total_norm += g.data.norm(2).item() ** 2
                
                grad_norms.append(total_norm ** 0.5)

        n = len(grad_norms)
        k = min(k, n)
        if k == 0: return []
        
        # Return indices of LARGEST norms
        top_k_indices = np.argsort(grad_norms)[-k:].tolist()
        return [int(i) for i in top_k_indices]
