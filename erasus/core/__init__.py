"""
Erasus Core Module

Provides base classes and infrastructure for the unlearning framework.
"""

from erasus.core.base_unlearner import BaseUnlearner, UnlearningResult
from erasus.core.base_selector import BaseSelector
from erasus.core.base_strategy import BaseStrategy
from erasus.core.base_metric import BaseMetric
from erasus.core.registry import Registry
from erasus.core.config import ErasusConfig
from erasus.core.coreset import Coreset, CoresetMetadata
from erasus.core.unlearning_module import UnlearningModule
from erasus.core.unlearning_trainer import UnlearningTrainer, TrainerResult
from erasus.core.strategy_pipeline import StrategyPipeline
from erasus.core.exceptions import (
    ErasusError,
    ModelNotFoundError,
    StrategyError,
    SelectorError,
    ConfigurationError,
)

__all__ = [
    "BaseUnlearner",
    "UnlearningResult",
    "BaseSelector",
    "BaseStrategy",
    "BaseMetric",
    "Registry",
    "ErasusConfig",
    "Coreset",
    "CoresetMetadata",
    "UnlearningModule",
    "UnlearningTrainer",
    "TrainerResult",
    "StrategyPipeline",
    "ErasusError",
    "ModelNotFoundError",
    "StrategyError",
    "SelectorError",
    "ConfigurationError",
]
