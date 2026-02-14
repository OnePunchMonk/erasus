"""
DALL-E Model Wrapper — Text-to-image generation via unCLIP / DALL-E 2.

Supports:
- Image generation from text prompts
- Prior and decoder sub-network access
- Cross-attention map extraction from decoder (U-Net)
- Concept erasure through CLIP embedding manipulation

Reference: Ramesh et al. (2022) — "Hierarchical Text-Conditional Image
Generation with CLIP Latents"
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from erasus.core.registry import model_registry
from erasus.models.model_wrapper import BaseDiffusionModel


@model_registry.register("dalle")
class DALLEWrapper(BaseDiffusionModel):
    """
    DALL-E 2 / unCLIP wrapper.

    Features
    --------
    - Access to prior network (text → CLIP image embedding)
    - Access to decoder (U-Net diffusion, CLIP image embedding → pixels)
    - Cross-attention map extraction
    - Concept filtering through text-embedding manipulation

    Supported models
    ----------------
    - ``kakaobrain/karlo-v1-alpha``
    - ``stabilityai/stable-diffusion-2-1-unclip``  (unCLIP variant)
    - Any HuggingFace ``UnCLIPPipeline``-compatible checkpoint
    """

    def __init__(
        self,
        model_name: str = "kakaobrain/karlo-v1-alpha",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, device, **kwargs)
        self.prior = None
        self.decoder = None
        self.text_encoder = None
        self.tokenizer = None
        self._pipe = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load DALL-E / unCLIP pipeline."""
        device = self._device if self._device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            from diffusers import UnCLIPPipeline
            self._pipe = UnCLIPPipeline.from_pretrained(
                self.model_name, torch_dtype=torch.float16,
            )
        except Exception:
            from diffusers import DiffusionPipeline
            self._pipe = DiffusionPipeline.from_pretrained(
                self.model_name, torch_dtype=torch.float16, trust_remote_code=True,
            )

        self._pipe.to(device)

        # Expose sub-components
        if hasattr(self._pipe, "prior"):
            self.prior = self._pipe.prior
        if hasattr(self._pipe, "decoder"):
            self.decoder = self._pipe.decoder
        elif hasattr(self._pipe, "unet"):
            self.decoder = self._pipe.unet
        if hasattr(self._pipe, "text_encoder"):
            self.text_encoder = self._pipe.text_encoder
        if hasattr(self._pipe, "tokenizer"):
            self.tokenizer = self._pipe.tokenizer

        # The main trainable component for unlearning is the decoder (U-Net)
        self._model = self.decoder if self.decoder is not None else self._pipe

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_image(self, prompt: str, **kwargs) -> Any:
        """
        Generate an image from a text prompt.

        Parameters
        ----------
        prompt : str
            Text description.
        **kwargs
            Additional generation parameters (num_inference_steps, etc.).

        Returns
        -------
        PIL.Image
        """
        defaults = dict(num_inference_steps=25)
        defaults.update(kwargs)
        output = self._pipe(prompt, **defaults)
        return output.images[0]

    # ------------------------------------------------------------------
    # Cross-attention maps
    # ------------------------------------------------------------------

    def get_cross_attention_maps(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Extract cross-attention maps from the decoder.

        Returns
        -------
        dict[str, Tensor]
            Mapping from attention layer name to attention weight tensor.
        """
        attention_maps: Dict[str, torch.Tensor] = {}
        hooks: list = []

        target = self.decoder or self._pipe

        def hook_fn(name):
            def hook(module, _in, output):
                if isinstance(output, torch.Tensor):
                    attention_maps[name] = output.detach()
                elif isinstance(output, tuple) and len(output) > 0:
                    attention_maps[name] = output[0].detach()
            return hook

        for name, module in target.named_modules():
            if "attn2" in name or "cross_attn" in name.lower():
                hooks.append(module.register_forward_hook(hook_fn(name)))

        try:
            self._pipe(prompt, num_inference_steps=1, output_type="latent")
        except Exception:
            pass

        for h in hooks:
            h.remove()

        return attention_maps

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def freeze_prior(self) -> None:
        """Freeze the prior network parameters."""
        if self.prior is not None:
            for p in self.prior.parameters():
                p.requires_grad = False

    def freeze_decoder(self) -> None:
        """Freeze the decoder (U-Net) parameters."""
        if self.decoder is not None:
            for p in self.decoder.parameters():
                p.requires_grad = False

    def freeze_text_encoder(self) -> None:
        """Freeze the text encoder parameters."""
        if self.text_encoder is not None:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Concept erasure
    # ------------------------------------------------------------------

    def get_text_embedding(self, prompt: str) -> torch.Tensor:
        """
        Encode a text prompt to the CLIP text embedding space.

        Useful for concept erasure: modify the embedding before passing
        to the prior/decoder.
        """
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError("Text encoder/tokenizer not loaded.")

        tokens = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            # Use pooler output or last hidden CLS
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state[:, 0]
