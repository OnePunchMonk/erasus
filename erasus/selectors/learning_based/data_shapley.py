"""
Data Shapley Selector (Learning-Based).

Estimates the Shapley Value of each training point with respect to model performance.
Expensive: O(N * M) model retrainings?
Approximated via TMC-Shapley or G-Shapley (Gradient).
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.exceptions import SelectorError
from erasus.core.registry import selector_registry

logger = logging.getLogger(__name__)


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

        raise SelectorError(
            "DataShapleySelector requires 'precomputed_values' — a list of "
            "per-sample Shapley values. Computing these from scratch requires "
            "O(N * M) model retrainings and is not done automatically. "
            "Pre-compute values externally and pass as: "
            "selector.select(model, loader, k, precomputed_values=[...])"
        )
