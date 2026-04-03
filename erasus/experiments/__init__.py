"""
erasus.experiments — Experiment tracking and management.
"""

from erasus.experiments.experiment_tracker import ExperimentTracker
from erasus.experiments.hyperparameter_search import HyperparameterSearch
from erasus.experiments.ablation_studies import AblationStudy
from erasus.experiments.hydra_config import (
    ExperimentConfig,
    HydraConfigManager,
    compose_experiment_config,
)

STABLE_EXPORTS = ["ExperimentTracker", "ExperimentConfig", "compose_experiment_config"]
EXPERIMENTAL_EXPORTS = ["HydraConfigManager", "HyperparameterSearch", "AblationStudy"]
PUBLIC_API_STATUS = {
    **{name: "stable" for name in STABLE_EXPORTS},
    **{name: "experimental" for name in EXPERIMENTAL_EXPORTS},
}

__all__ = STABLE_EXPORTS + EXPERIMENTAL_EXPORTS
