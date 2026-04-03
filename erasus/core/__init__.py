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

STABLE_EXPORTS = [
    "BaseUnlearner",
    "UnlearningResult",
    "BaseSelector",
    "BaseStrategy",
    "BaseMetric",
    "Registry",
    "ErasusConfig",
    "Coreset",
    "CoresetMetadata",
    "ErasusError",
    "ModelNotFoundError",
    "StrategyError",
    "SelectorError",
    "ConfigurationError",
]

EXPERIMENTAL_EXPORTS = [
    "UnlearningModule",
    "UnlearningTrainer",
    "TrainerResult",
    "StrategyPipeline",
]

PUBLIC_API_STATUS = {
    **{name: "stable" for name in STABLE_EXPORTS},
    **{name: "experimental" for name in EXPERIMENTAL_EXPORTS},
}

__all__ = STABLE_EXPORTS + EXPERIMENTAL_EXPORTS
