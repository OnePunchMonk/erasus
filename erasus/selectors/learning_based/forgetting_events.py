"""
Forgetting Events Selector (Learning-Based).

Counts how many times an example transitions from 'Correct' to 'Incorrect' classification during training.
Requires access to training history (Checkpoint logs).

Reference: Toneva et al., "An Empirical Study of Example Forgetting during Deep Neural Network Learning", ICLR 2019.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.exceptions import SelectorError
from erasus.core.registry import selector_registry

logger = logging.getLogger(__name__)


@selector_registry.register("forgetting_events")
class ForgettingEventsSelector(BaseSelector):
    """
    Selects examples with high frequency of forgetting events.
    Usually requires pre-computed statistics.
    
    If 'stats' is not provided, this selector CANNOT run dynamically on a static model.
    It raises an error.
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        forgetting_stats: Dict[int, int] = None,
        **kwargs: Any,
    ) -> List[int]:
        
        if forgetting_stats is None:
            raise SelectorError(
                "ForgettingEventsSelector requires a 'forgetting_stats' dictionary "
                "mapping sample index -> forgetting event count. This cannot be "
                "computed from a static checkpoint — it must be collected during "
                "training. Pass it as: selector.select(model, loader, k, "
                "forgetting_stats={0: 3, 1: 0, ...})"
            )

        # Sort by count descending
        sorted_indices = sorted(forgetting_stats.keys(), key=lambda i: forgetting_stats[i], reverse=True)
        return sorted_indices[:k]
