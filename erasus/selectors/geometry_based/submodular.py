"""
Submodular Selector (Geometry-Based).

Selects a subset that maximizes a submodular utility function (e.g., Facility Location),
ensuring diversity and representativeness.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("submodular")
class SubmodularSelector(BaseSelector):
    """
    Simple Greedy Facility Location implementation.
    Maximizes L(S) = sum_{i in V} max_{j in S} sim(i, j).
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        
        embeddings = self._extract_features(model, data_loader) # [N, D]
        n_samples = embeddings.shape[0]
        k = min(k, n_samples)
        if k == 0: return []
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)
        
        # Similarity matrix [N, N] - Expensive for large N!
        # For N > 5000, consider random subsampling or batching.
        if n_samples > 2000:
             print("Warning: Submodular selection is O(N^2). Subsampling to 2000 for efficiency.")
             indices = np.random.choice(n_samples, 2000, replace=False)
             embeddings = embeddings[indices]
             mapping = {i: old for i, old in enumerate(indices)}
        else:
             mapping = {i: i for i in range(n_samples)}
             
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        # Greedy Selection
        selected = []
        # Current max similarity for each point to the selected set
        # Init with -inf? Or 0 (assuming sim >= -1). 
        # Facility location: usually on positive similarities. Scale to [0,1]?
        # Let's use sim + 1.0 to ensure positivity
        sim_matrix = sim_matrix + 1.0
        
        max_sims = np.zeros(embeddings.shape[0]) # Current best coverage for each point
        
        candidates = set(range(embeddings.shape[0]))
        
        for _ in range(k):
            best_gain = -1.0
            best_node = -1
            
            # Find node that maximizes marginal gain
            # Gain(j) = sum_i (max(current_max_sim[i], sim[i,j]) - current_max_sim[i])
            #         = sum_i max(0, sim[i,j] - current_max_sim[i])
            
            # Vectorized search
            gains = np.sum(np.maximum(0, sim_matrix - max_sims[:, None]), axis=0)
            
            # Mask already selected
            for s in selected:
                gains[s] = -1.0
                
            best_node = np.argmax(gains)
            selected.append(best_node)
            
            # Update max_sims
            max_sims = np.maximum(max_sims, sim_matrix[:, best_node])
            
        # Remap back to original indices
        final_indices = [mapping[i] for i in selected]
        return final_indices
