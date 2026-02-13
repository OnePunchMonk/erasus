"""
Random Selector — Baseline coreset selector.

Randomly selects k samples from the forget set. Used as a baseline
for comparing more sophisticated coreset methods.
"""

from __future__ import annotations

import random
from typing import Any, List

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("random")
class RandomSelector(BaseSelector):
    """Select *k* random samples from the forget set."""

    def __init__(self, seed: int = 42, **kwargs: Any):
        super().__init__(**kwargs)
        self.seed = seed

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        n = len(data_loader.dataset)
        k = min(k, n)
        rng = random.Random(self.seed)
        return rng.sample(range(n), k)
