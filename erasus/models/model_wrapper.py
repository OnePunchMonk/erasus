"""
Base Model Wrapper — Provides a uniform interface across all foundation model types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class BaseModelWrapper(ABC):
    """
    Thin wrapper around a Hugging Face / diffusers model that
    exposes a consistent API for the unlearning framework.
    """

    def __init__(self, model_name: str, device: str = "auto", **kwargs: Any) -> None:
        self.model_name = model_name
        self._device = device
        self._model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load the pre-trained model into memory."""
        ...

    def to(self, device: Union[str, torch.device]) -> "BaseModelWrapper":
        self.model.to(device)
        return self

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all model parameters."""
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for p in self.model.parameters():
            p.requires_grad = True

    def named_parameters(self):
        return self.model.named_parameters()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)


# ======================================================================
# Modality-specific abstract wrappers
# ======================================================================


class BaseVLMModel(BaseModelWrapper):
    """Base wrapper for Vision-Language Models (CLIP, LLaVA, BLIP)."""

    @abstractmethod
    def get_image_features(self, images: Any, **kwargs) -> torch.Tensor:
        ...

    @abstractmethod
    def get_text_features(self, texts: Any, **kwargs) -> torch.Tensor:
        ...


class BaseLLMModel(BaseModelWrapper):
    """Base wrapper for Large Language Models (LLaMA, Mistral, GPT)."""

    @abstractmethod
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        ...

    @abstractmethod
    def get_layer_activations(self, text: str, layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        ...


class BaseDiffusionModel(BaseModelWrapper):
    """Base wrapper for Generative Diffusion Models (Stable Diffusion)."""

    @abstractmethod
    def generate_image(self, prompt: str, **kwargs) -> Any:
        ...

    @abstractmethod
    def get_cross_attention_maps(self, prompt: str) -> Dict[str, torch.Tensor]:
        ...
