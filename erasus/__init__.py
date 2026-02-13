"""
ERASUS — Efficient Representative And Surgical Unlearning Selection
Universal Machine Unlearning via Coreset Selection

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

from erasus.version import __version__
from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.unlearners.multimodal_unlearner import MultimodalUnlearner

__all__ = [
    "__version__",
    "ErasusUnlearner",
    "MultimodalUnlearner",
]
