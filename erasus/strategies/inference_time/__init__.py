"""
erasus.strategies.inference_time — Inference-time unlearning strategies.

These strategies do not modify model weights.  Instead they wrap the
base model and adjust token probabilities at decode time.

Suitable for:
- Black-box API models (no weight access)
- Scenarios where weight modification is too slow or irreversible
- Rapid iteration without committing to permanent parameter changes
"""

from erasus.strategies.inference_time.base import BaseInferenceTimeStrategy
from erasus.strategies.inference_time.dexperts import DExpertsStrategy
from erasus.strategies.inference_time.activation_steering import (
    ActivationSteeringStrategy,
    SteeringModel,
)

__all__ = [
    "BaseInferenceTimeStrategy",
    "DExpertsStrategy",
    "ActivationSteeringStrategy",
    "SteeringModel",
]
