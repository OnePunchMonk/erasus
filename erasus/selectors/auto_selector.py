"""
Auto-Selector Engine — Meta-learning based strategy selection.

Automatically chooses the optimal coreset method based on model type
and data distribution. Section 3.4.
"""

from __future__ import annotations

from typing import Any, List

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("auto")
class AutoSelector(BaseSelector):
    """
    Meta-selector that analyses model + data distribution to choose
    the best coreset technique automatically.

    Current heuristic:
    - Small datasets (< 1000): use InfluenceSelector
    - Medium datasets: use k-Center
    - Large datasets: use RandomSelector (fastest)
    """

    def __init__(self, model_type: str = "vlm", goal: str = "balanced_utility", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.goal = goal

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        n = len(data_loader.dataset)

        # Simple heuristic for Phase 1
        if n <= 1000:
            from erasus.selectors.gradient_based.influence import InfluenceSelector
            inner = InfluenceSelector()
        elif n <= 50000:
            from erasus.selectors.geometry_based.kcenter import KCenterSelector
            inner = KCenterSelector()
        else:
            from erasus.selectors.random_selector import RandomSelector
            inner = RandomSelector()

        return inner.select(model=model, data_loader=data_loader, k=k, **kwargs)
