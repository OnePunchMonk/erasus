"""
Herding Selector (Geometry-Based).

Selects a subset of samples (coreset) whose mean matches the mean of the full dataset.
Also known as "Super-Samples" or Kernel Herding (Welling 2009).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("herding")
class HerdingSelector(BaseSelector):
    """
    Selects k points such that the mean of the selected points iteratively 
    approximates the mean of the entire population.
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        
        # 1. Extract Embeddings
        # Uses the shared helper from BaseSelector
        embeddings = self._extract_features(model, data_loader) # [N, D] numpy array
        
        n_samples = embeddings.shape[0]
        k = min(k, n_samples)
        if k == 0:
            return []
            
        # Target Mean (Center of Mass of entire dataset)
        mu = np.mean(embeddings, axis=0)
        
        indices = []
        current_sum = np.zeros_like(mu)
        
        # Mask to keep track of selected
        selected_mask = np.zeros(n_samples, dtype=bool)
        
        for t in range(1, k + 1):
            # We want to pick x_t maximizing <mu, x> - <current_mean, x>?
            # Standard herding update: x_{t+1} = argmax_x <mu - w_t, phi(x)>
            # where w_t is current mean of selected.
            
            # More simply: We want (current_sum + x) / t to be close to mu.
            # Minimize || mu - (current_sum + x)/t ||^2
            # <=> Minimize || t*mu - current_sum - x ||^2
            
            target = (t * mu) - current_sum
            
            # Compute distances from target to all available points
            # We can use dot product if normalized, or L2.
            # Minimizing distance to target is equivalent to finding nearest neighbor to target.
            
            # Only search among unselected
            available_indices = np.where(~selected_mask)[0]
            if len(available_indices) == 0:
                break
                
            available_embeds = embeddings[available_indices]
            
            dists = np.linalg.norm(available_embeds - target, axis=1)
            best_local_idx = np.argmin(dists)
            best_global_idx = available_indices[best_local_idx]
            
            indices.append(best_global_idx)
            selected_mask[best_global_idx] = True
            current_sum += embeddings[best_global_idx]
            
        return [int(i) for i in indices]
