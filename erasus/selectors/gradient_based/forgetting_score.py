"""
Forgetting Score Selector.

Proxy/Alias for Forgetting Events Selector.
"""

from __future__ import annotations
from typing import Any, List
import torch.nn as nn
from torch.utils.data import DataLoader
from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry
from erasus.selectors.learning_based.forgetting_events import ForgettingEventsSelector

@selector_registry.register("forgetting_score")
class ForgettingScoreSelector(ForgettingEventsSelector):
    """
    Alias for ForgettingEventsSelector.
    Expects 'forgetting_stats' in kwargs.
    """
    pass
