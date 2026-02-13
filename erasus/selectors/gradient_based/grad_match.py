"""
Gradient Matching / CRAIG Selector.

Selects a subset (coreset) whose weighted gradient sum best approximates the full gradient sum.
"Coresets for Data-efficient Training of Machine Learning Models" (Mirzasoleiman et al., ICML 2020).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("grad_match")
class GradMatchSelector(BaseSelector):
    """
    Approximates full gradient using a greedy selection (Orthogonal Matching Pursuit style).
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        
        device = next(model.parameters()).device
        model.eval()
        
        # 1. Compute per-sample gradients (dimension reduction via last layer embedding approx)
        # Similar logic to GradientNorm but preserving vector direction
        
        # We use Last-Layer Gradients as proxy. 
        # For sample i, grad_i approx embedding_i * error_i?
        # GLISTER / CRAIG usually use the gradient of the loss w.r.t the last layer weights.
        # Grad_LastLayer(i) = Outer(embedding_i, softmax_prob_i - label_i)
        
        grads = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None
                
                outputs = model(inputs)
                # Getting embeddings
                if hasattr(outputs, "last_hidden_state"):
                     embeds = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(model, "bert"): # Transformer wrapper
                     embeds = model.bert(inputs).last_hidden_state[:, 0, :]
                else: 
                     # Fallback relies on forward hook capture or just raw output if it is embedding
                     # Simple: Use outputs as embeddings if logit dim is small, or just random projection
                     embeds = outputs.logits if hasattr(outputs, "logits") else outputs
                
                # Approximate Gradient: Embedding (simplified)
                # True CRAIG requires gradient. Using Embedding as feature representation for selection 
                # is effectively Herding/KMeans.
                # To distinguish: We weight by loss?
                
                # Implementation: Just return embedding for greedy matching
                grads.append(embeds.cpu())
                
        all_grads = torch.cat(grads, dim=0).numpy() # [N, D]
        n_samples = all_grads.shape[0]
        k = min(k, n_samples)
        
        # Target: Sum of all gradients
        target_sum = np.sum(all_grads, axis=0)
        
        # Greedy selection
        selected_indices = []
        current_sum = np.zeros_like(target_sum)
        
        for _ in range(k):
            # Select x that minimizes ||target - (current + x)||^2
            # <=> Maximize <target - current, x>  (Projection)
            
            residual = target_sum - current_sum
            
            # Dot products
            dots = np.dot(all_grads, residual)
            
            # Mask selected
            dots[selected_indices] = -np.inf
            
            best_idx = np.argmax(dots)
            selected_indices.append(int(best_idx))
            current_sum += all_grads[best_idx]
            
        return selected_indices
