"""
Multimodal Unlearner — Automatic modality dispatch.

Inspects the model structure and automatically delegates to
VLMUnlearner, LLMUnlearner, or DiffusionUnlearner.

Example::

    from erasus.unlearners import MultimodalUnlearner

    # Works for any model type — auto-detects!
    unlearner = MultimodalUnlearner.from_model(model)
    result = unlearner.fit(forget_loader, retain_loader)
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from erasus.core.base_unlearner import BaseUnlearner


class MultimodalUnlearner:
    """
    Factory that inspects a model and returns the correct
    modality-specific unlearner.
    """

    @staticmethod
    def from_model(
        model: nn.Module,
        model_type: Optional[str] = None,
        strategy: Optional[str] = None,
        selector: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseUnlearner:
        """
        Create the appropriate unlearner for `model`.

        Parameters
        ----------
        model : nn.Module
            The model to unlearn from.
        model_type : str, optional
            Explicit type hint: ``"vlm"``, ``"llm"``, ``"diffusion"``,
            ``"audio"``, ``"video"``.  If omitted the factory will
            attempt to auto-detect based on model attributes.
        strategy : str, optional
            Override the default strategy name.
        selector : str, optional
            Override the default selector name.

        Returns
        -------
        BaseUnlearner subclass instance
        """
        if model_type is None:
            model_type = MultimodalUnlearner._detect_type(model)

        # Build kwargs dict, skipping None values
        build_kw: dict = {**kwargs}
        if strategy is not None:
            build_kw["strategy"] = strategy
        if selector is not None:
            build_kw["selector"] = selector

        if model_type == "vlm":
            from erasus.unlearners.vlm_unlearner import VLMUnlearner
            return VLMUnlearner(model=model, **build_kw)
        elif model_type == "llm":
            from erasus.unlearners.llm_unlearner import LLMUnlearner
            return LLMUnlearner(model=model, **build_kw)
        elif model_type == "diffusion":
            from erasus.unlearners.diffusion_unlearner import DiffusionUnlearner
            return DiffusionUnlearner(model=model, **build_kw)
        elif model_type == "audio":
            from erasus.unlearners.audio_unlearner import AudioUnlearner
            return AudioUnlearner(model=model, **build_kw)
        elif model_type == "video":
            from erasus.unlearners.video_unlearner import VideoUnlearner
            return VideoUnlearner(model=model, **build_kw)
        else:
            # Fallback: generic ErasusUnlearner
            from erasus.unlearners.erasus_unlearner import ErasusUnlearner
            return ErasusUnlearner(model=model, **build_kw)

    # ------------------------------------------------------------------

    @staticmethod
    def _detect_type(model: nn.Module) -> str:
        """
        Heuristic model-type detection based on common attributes.
        """
        # VLM — has both vision and text sub-models
        if hasattr(model, "vision_model") and hasattr(model, "text_model"):
            return "vlm"

        # Diffusion — has U-Net + scheduler
        if hasattr(model, "unet") or hasattr(model, "scheduler"):
            return "diffusion"

        # LLM — has lm_head or is a CausalLM
        cls_name = type(model).__name__.lower()
        if hasattr(model, "lm_head") or "causal" in cls_name or "gpt" in cls_name:
            return "llm"

        # Audio — has encoder + decoder structure with "whisper" in name
        if "whisper" in cls_name or "speech" in cls_name:
            return "audio"

        # Video — has "video" in name
        if "video" in cls_name or "videomae" in cls_name:
            return "video"

        # Default to generic
        return "generic"
