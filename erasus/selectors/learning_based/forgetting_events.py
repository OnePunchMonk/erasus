"""
Forgetting Events Selector (Learning-Based).

Counts how many times an example transitions from 'Correct' to 'Incorrect' classification during training.
Requires access to training history (Checkpoint logs).

Reference: Toneva et al., "An Empirical Study of Example Forgetting during Deep Neural Network Learning", ICLR 2019.
"""

from __future__ import annotations

from typing import Any, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("forgetting_events")
class ForgettingEventsSelector(BaseSelector):
    """
    Selects examples with high frequency of forgetting events.
    Usually requires pre-computed statistics.
    
    If 'stats' is not provided, this selector CANNOT run dynamically on a static model.
    It raises an error or returns random fallback with warning.
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
            # We cannot compute this from a static checkpoint.
            # We implemented a stub warning previously.
            # Now explicitly failing or mocking.
            print("Warning: ForgettingEventsSelector requires 'forgetting_stats' dictionary (idx -> count). Returning Random.")
            import random
            n = len(data_loader.dataset)
            return random.sample(range(n), min(k, n))

        # Sort by count descending
        sorted_indices = sorted(forgetting_stats.keys(), key=lambda i: forgetting_stats[i], reverse=True)
        return sorted_indices[:k]
