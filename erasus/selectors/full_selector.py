"""
Full Selector — Returns all indices (no pruning).
"""

from __future__ import annotations

from typing import Any, List

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("full")
class FullSelector(BaseSelector):
    """Use the *entire* forget set (no coreset pruning)."""

    def select(self, model: nn.Module, data_loader: DataLoader, k: int, **kwargs: Any) -> List[int]:
        return list(range(len(data_loader.dataset)))
