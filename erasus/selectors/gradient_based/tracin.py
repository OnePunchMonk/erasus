"""
TracIn Selector (Gradient-Based).

Estimates the influence of a training example on a test example (or loss)
by tracing the change in loss during training.
Approximated here using the dot product of gradients at the current checkpoint
(First-Order TracIn / Gradient Dot Product).

Reference: Pruthi et al., "Estimating Training Data Influence by Tracing Gradient Descent", NeurIPS 2020.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("tracin")
class TracInSelector(BaseSelector):
    """
    Selects samples that have the highest 'influence' (TracIn CP score).
    Influence ≈ ∇L(x_train) · ∇L(x_target).
    
    In the unlearning context, we often want to find samples that maximize
    loss on the forget set itself (Self-Influence)?
    Or we use this to identify samples contributing most to a *target* bad behavior.
    
    If no target is provided, we default to Self-Influence (Gradient Norm^2),
    approximating how much the sample 'resists' change or contributes to the current state.
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        target_loader: DataLoader = None,
        **kwargs: Any,
    ) -> List[int]:
        """
        Parameters
        ----------
        target_loader : DataLoader, optional
            If provided, computes influence of `data_loader` samples on `target_loader` samples.
            If None, computes Self-Influence (norm of gradient).
        """
        device = next(model.parameters()).device
        model.eval()
        
        scores = []
        
        # If target_loader is None, TracInCP(z, z) = ||grad(z)||^2
        # This reduces to GradientNormSelector!
        
        if target_loader is None:
            # Revert to gradient norm logic efficiently
            # We reused logic from Gradient Norm if accessible, but let's implement clean version.
            # See GradientNormSelector for optimized implementation.
             from erasus.selectors.gradient_based.gradient_norm import GradientNormSelector
             # Delegate to GradientNorm for self-influence
             return GradientNormSelector().select(model, data_loader, k, **kwargs)

        # If target_loader IS provided:
        # We need sum_j (grad(z_forget) . grad(z_target_j))
        
        # 1. Compute aggregate gradient of the TARGET set (or average)
        # avg_grad_target = Mean(grad(z_j) for z_j in target)
        
        target_grads = []
        
        # Accumulate target gradient
        # Optimization: Just sum gradients in one pass
        model.zero_grad()
        n_target = 0
        
        for batch in target_loader:
             inputs = batch[0].to(device)
             labels = batch[1].to(device) if len(batch) > 1 else None
             outputs = model(inputs)
             logits = outputs.logits if hasattr(outputs, "logits") else outputs
             loss = nn.functional.cross_entropy(logits, labels)
             loss.backward()
             n_target += inputs.size(0)
             
        # Normalize aggregated gradient
        target_flat_grad = []
        for p in model.parameters():
            if p.grad is not None:
                target_flat_grad.append(p.grad.detach().flatten() / n_target)
        
        if not target_flat_grad:
            return []
            
        target_vector = torch.cat(target_flat_grad)
        
        # 2. Compute dot product for each forget sample
        # Score(z_i) = grad(z_i) · target_vector
        
        for i, batch in enumerate(data_loader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else None
            batch_size = inputs.size(0)
            
            # Simple batch loop for per-sample gradients
            for b in range(batch_size):
                model.zero_grad()
                
                # Single sample forward
                single_input = inputs[b:b+1]
                single_label = labels[b:b+1] if labels is not None else None
                
                out = model(single_input)
                logits = out.logits if hasattr(out, "logits") else out
                
                loss = nn.functional.cross_entropy(logits, single_label)
                
                # We need gradients of this sample
                grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
                
                # Flatten and Dot Product
                # We can do this layer-wise to save memory
                dot_prod = 0.0
                idx = 0
                for g in grads:
                    if g is not None:
                        flat_g = g.flatten()
                        num_param = flat_g.numel()
                        # Match with target vector segment
                        target_segment = target_vector[idx : idx + num_param]
                        dot_prod += torch.dot(flat_g, target_segment).item()
                        idx += num_param
                
                scores.append(dot_prod)
                
        n = len(scores)
        k = min(k, n)
        
        # Select samples with HIGHEST positive influence (most helpful/harmful depending on sign)
        # If we want to remove samples causing high loss on target?
        # Influence usually measures: How much did this training point *reduce* loss on test point?
        # Positive influence => Removing it will INCREASE loss (Hurt accuracy).
        # Negative influence => Removing it will DECREASE loss (Help accuracy).
        
        # If target denotes "Bad Behavior", we want to remove samples with Positive Influence on Bad Behavior.
        # So we select top-k highest scores.
        
        top_k_indices = np.argsort(scores)[-k:].tolist()
        return [int(i) for i in top_k_indices]
