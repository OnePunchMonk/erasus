"""
SISA (Sharded, Isolated, Sliced, and Aggregated) Strategy.

Paper: "Machine Unlearning" (Bourtoule et al., S&P 2021)

True SISA requires training the model from scratch on shards.
"Unlearning" is then just retraining the specific shard containing the forget data.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry
from erasus.core.exceptions import StrategyError


@strategy_registry.register("sisa")
class SISAStrategy(BaseStrategy):
    """
    SISA Wrapper.
    
    NOTE: SISA is an architectural training strategy, not a post-hoc unlearning strategy
    that can be easily applied to a monolithic pre-trained model.
    
    This implementation serves as a placeholder or 'mock' for the framework completeness.
    Real SISA requires:
    1. Splitting data into Shards S_1...S_k
    2. Training separate Models M_1...M_k
    3. Aggregating outputs.
    
    Unlearning: If data x in S_i, retrain M_i without x.
    """

    def __init__(self, shards: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.shards = shards

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        # In a post-hoc framework for Foundation Models, SISA is generally inapplicable 
        # unless we have the constituent models.
        
        # We raise a warning or error, optionally falling back to simple retraining approximation
        # if the user insists?
        
        # For this codebase, we will just return the model unmodified with a log
        # to prevent crashing, but conceptually SISA fails on a single pre-trained blob.
        
        print("[Erasus] WARNING: SISA strategy called on a pre-trained monolithic model.")
        print("[Erasus] SISA requires the model to have been trained with SISA architecture initially.")
        print("[Erasus] No changes made.")
        
        return model, [], []

