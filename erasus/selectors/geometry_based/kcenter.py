"""
k-Center Greedy Selector — Section 3.2.1.

Find k centers minimising maximum distance from any point to nearest centre.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("kcenter")
class KCenterSelector(BaseSelector):
    """k-Center greedy coreset selection — optimal for uniform feature-space coverage."""

    def __init__(self, metric: str = "euclidean", **kwargs: Any):
        super().__init__(**kwargs)
        self.metric = metric

    def select(self, model: nn.Module, data_loader: DataLoader, k: int, **kwargs: Any) -> List[int]:
        features = self._extract_features(model, data_loader)
        n = features.shape[0]
        k = min(k, n)

        selected = [np.random.randint(n)]
        dists = np.linalg.norm(features - features[selected[0]], axis=1)

        for _ in range(k - 1):
            farthest = int(np.argmax(dists))
            selected.append(farthest)
            new_dists = np.linalg.norm(features - features[farthest], axis=1)
            dists = np.minimum(dists, new_dists)

        return selected
