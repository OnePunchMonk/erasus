"""
ERASUS — Efficient Representative And Surgical Unlearning Selection.

Universal Machine Unlearning via Coreset Selection.

A unified, modality-agnostic framework for machine unlearning across
Vision-Language Models, Large Language Models, Diffusion Models,
Audio Models, and Video Models.

Quick start::

    from erasus import ErasusUnlearner

    unlearner = ErasusUnlearner(model=my_model, strategy="gradient_ascent")
    result = unlearner.fit(forget_loader, retain_loader)

Modality-specific unlearners::

    from erasus.unlearners import VLMUnlearner, LLMUnlearner, DiffusionUnlearner
    from erasus.unlearners import MultimodalUnlearner  # auto-detect
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from erasus.version import __version__

STABLE_EXPORTS = [
    "__version__",
    "ErasusUnlearner",
    "MultimodalUnlearner",
    "Coreset",
]

EXPERIMENTAL_EXPORTS = [
    "UnlearningModule",
    "UnlearningTrainer",
    "StrategyPipeline",
]

PUBLIC_API_STATUS = {
    **{name: "stable" for name in STABLE_EXPORTS},
    **{name: "experimental" for name in EXPERIMENTAL_EXPORTS},
}

__all__ = STABLE_EXPORTS + EXPERIMENTAL_EXPORTS

_EXPORTS = {
    "ErasusUnlearner": ("erasus.unlearners.erasus_unlearner", "ErasusUnlearner"),
    "MultimodalUnlearner": ("erasus.unlearners.multimodal_unlearner", "MultimodalUnlearner"),
    "Coreset": ("erasus.core.coreset", "Coreset"),
    "UnlearningModule": ("erasus.core.unlearning_module", "UnlearningModule"),
    "UnlearningTrainer": ("erasus.core.unlearning_trainer", "UnlearningTrainer"),
    "StrategyPipeline": ("erasus.core.strategy_pipeline", "StrategyPipeline"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve top-level exports to keep ``import erasus`` lightweight."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'erasus' has no attribute '{name}'")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports in interactive environments."""
    return sorted(set(globals()) | set(__all__))
