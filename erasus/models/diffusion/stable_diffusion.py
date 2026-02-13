"""
Stable Diffusion Wrapper — Supports U-Net, VAE, and text encoder access.

Section 2.3.1: Enables concept erasure, artist style removal, and content filtering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from erasus.core.registry import model_registry
from erasus.models.model_wrapper import BaseDiffusionModel


@model_registry.register("stable_diffusion")
class StableDiffusionWrapper(BaseDiffusionModel):
    """
    Stable Diffusion wrapper supporting:
    - U-Net architecture access
    - Text encoder manipulation (CLIP)
    - VAE latent space operations
    - Timestep-specific unlearning
    """

    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.unet = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.tokenizer = None

    def load(self) -> None:
        from diffusers import StableDiffusionPipeline

        device = self._device if self._device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float16,
        )
        pipe.to(device)
        self._model = pipe.unet  # main trainable component
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer
        self._pipe = pipe

    def generate_image(self, prompt: str, **kwargs) -> Any:
        return self._pipe(prompt, **kwargs).images[0]

    def get_cross_attention_maps(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Extract cross-attention maps for prompt tokens."""
        attention_maps: Dict[str, torch.Tensor] = {}
        hooks: list = []

        def hook_fn(name):
            def hook(module, _input, output):
                attention_maps[name] = output.detach()
            return hook

        for name, module in self.unet.named_modules():
            if "attn2" in name:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        _ = self._pipe(prompt, num_inference_steps=1, output_type="latent")

        for h in hooks:
            h.remove()

        return attention_maps
