"""
Data Shapley Selector (Learning-Based).

Estimates the Shapley Value of each training point with respect to model performance.
Expensive: O(N * M) model retrainings?
Approximated via TMC-Shapley or G-Shapley (Gradient).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("data_shapley")
class DataShapleySelector(BaseSelector):
    """
    Monte-Carlo approximation of Data Shapley (TMC-Shapley).
    Very slow for large N. Stub implementation returns Random with warning
    unless 'precomputed_values' is provided.
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        precomputed_values: List[float] = None,
        **kwargs: Any,
    ) -> List[int]:
        
        if precomputed_values is not None:
             top_k = np.argsort(precomputed_values)[-k:].tolist()
             return [int(i) for i in top_k]

        print("Warning: Data Shapley requires 'precomputed_values' or massive compute. Returning Random.")
        from erasus.selectors.random_selector import RandomSelector
        return RandomSelector().select(model, data_loader, k)
