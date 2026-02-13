"""
Ensemble Voting Selector.

Combines multiple selectors (e.g., Gradient Norm + Entropy + Loss) to select
the most universally "important" or "difficult" examples.
"""

from __future__ import annotations

from typing import Any, List, Dict

import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry, Registry


@selector_registry.register("voting")
class VotingSelector(BaseSelector):
    """
    Runs multiple selectors and aggregates their votes (indices).
    Strategy: Majority Vote or Weighted Score.
    
    If 'selectors' list not provided, defaults to [gradient_norm, el2n].
    """

    def __init__(self, selector_names: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.selector_names = selector_names or ["gradient_norm", "random"] 

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        
        all_votes = []
        
        for name in self.selector_names:
            # Dynamically instantiate selector
            # Ref: selector_registry
            try:
                selector_cls = selector_registry.get(name)
                selector = selector_cls()
                
                # We ask each selector for Top-K (or slightly more to ensure overlap?)
                # Let's ask for Top-K from each.
                indices = selector.select(model, data_loader, k, **kwargs)
                all_votes.extend(indices)
                
            except Exception as e:
                print(f"Warning: VotingSelector failed to run '{name}': {e}")
                continue
                
        if not all_votes:
            return []
            
        # Weighted voting? Here simple frequency count.
        counts = Counter(all_votes)
        
        # Select indices with highest occurrences
        # If tie, arbitrary (based on insertion order in most_common)
        most_common = counts.most_common(k)
        
        selected_indices = [idx for idx, count in most_common]
        return selected_indices
