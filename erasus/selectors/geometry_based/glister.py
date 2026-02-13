"""
Glister Selector.

GLISTER: Generalization based Data Subset Selection.
Selects subset that maximizes log-likelihood on a validation set (approximated by one-step update).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("glister")
class GlisterSelector(BaseSelector):
    """
    Selects subset that minimizes validation loss after one gradient update.
    Approximation: Taylor expansion similar to Influence Functions.
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        val_loader: DataLoader = None, # Required
        **kwargs: Any,
    ) -> List[int]:
        
        if val_loader is None:
            print("Warning: Glister requires 'val_loader'. Returning Random.")
            # Fallback
            from erasus.selectors.random_selector import RandomSelector
            return RandomSelector().select(model, data_loader, k)

        # Implementation of greedy selection based on gradient similarity with validation set
        # Often simplified to: Gradient Matching between Train and Validation.
        # This is very similar to GradMatch but Target = Validation Gradient.
        
        # Reuse GradMatch logic logic?
        # Ideally we inherit, but let's explicate for clarity.
        
        from erasus.selectors.gradient_based.grad_match import GradMatchSelector
        
        # 1. Compute Validation Gradient Average
        device = next(model.parameters()).device
        grad_selector = GradMatchSelector()
        
        # Extract features (proxy for gradients)
        train_grads = grad_selector._extract_features(model, data_loader) # [N, D]
        val_grads = grad_selector._extract_features(model, val_loader)    # [M, D]
        
        target = np.sum(val_grads, axis=0) # [D]
        
        # Greedy Match
        selected_indices = []
        current_sum = np.zeros_like(target)
        all_grads = train_grads
        
        for _ in range(k):
            residual = target - current_sum
            dots = np.dot(all_grads, residual)
            dots[selected_indices] = -np.inf
            best_idx = np.argmax(dots)
            selected_indices.append(int(best_idx))
            current_sum += all_grads[best_idx]
            
        return selected_indices
