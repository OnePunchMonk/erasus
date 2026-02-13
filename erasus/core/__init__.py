"""
Erasus Core Module

Provides base classes and infrastructure for the unlearning framework.
"""

from erasus.core.base_unlearner import BaseUnlearner
from erasus.core.base_selector import BaseSelector
from erasus.core.base_strategy import BaseStrategy
from erasus.core.base_metric import BaseMetric
from erasus.core.registry import Registry
from erasus.core.config import ErasusConfig
from erasus.core.exceptions import (
    ErasusError,
    ModelNotFoundError,
    StrategyError,
    SelectorError,
    ConfigurationError,
)

__all__ = [
    "BaseUnlearner",
    "BaseSelector",
    "BaseStrategy",
    "BaseMetric",
    "Registry",
    "ErasusConfig",
    "ErasusError",
    "ModelNotFoundError",
    "StrategyError",
    "SelectorError",
    "ConfigurationError",
]
