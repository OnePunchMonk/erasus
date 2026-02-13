"""
K-Means Coreset Selector (Geometry-Based).

Selects samples closest to the centroids of K-Means clustering in the embedding space.
Simple and effective approximation of the data distribution.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("kmeans")
class KMeansSelector(BaseSelector):
    """
    Clusters the embeddings into k clusters and selects the sample closest to each center.
    This provides a representative "coreset".
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        
        if KMeans is None:
            raise ImportError("scikit-learn is required for KMeansSelector.")

        embeddings = self._extract_features(model, data_loader) # [N, D]
        n_samples = embeddings.shape[0]
        k = min(k, n_samples)
        
        if k == 0: return []
        
        # If k is very large relative to N, KMeans is slow/degenerate
        # We can just run KMeans with n_clusters=k
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        centers = kmeans.cluster_centers_
        
        # Find nearest sample to each center
        selected_indices = []
        for center in centers:
            # Distance to all points
            dists = np.linalg.norm(embeddings - center, axis=1)
            nearest_idx = np.argmin(dists)
            selected_indices.append(int(nearest_idx))
            
        # De-duplicate if multiple centers map to same point
        selected_indices = list(set(selected_indices))
        
        return selected_indices
