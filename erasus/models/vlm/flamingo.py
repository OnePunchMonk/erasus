"""
Flamingo Model Wrapper — Few-shot vision-language model.

Supports:
- Multi-image in-context conditioning
- Feature extraction from Perceiver Resampler + language model
- Gradient isolation between vision encoder and language model

Reference: Alayrac et al. (2022) — "Flamingo: a Visual Language Model for
Few-Shot Learning"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from erasus.core.registry import model_registry
from erasus.models.model_wrapper import BaseVLMModel


@model_registry.register("flamingo")
class FlamingoWrapper(BaseVLMModel):
    """
    Wrapper for Flamingo-style vision-language models.

    Features
    --------
    - Vision encoder (ViT / NFNet) + Perceiver Resampler access
    - Cross-attention gated language model layers
    - Few-shot in-context image support
    - Gradient isolation for modality-specific unlearning

    Supported back-ends
    -------------------
    - ``openflamingo/OpenFlamingo-3B-vitl-mpt1b``
    - ``openflamingo/OpenFlamingo-9B-vitl-mpt7b``
    - Any HuggingFace-compatible Flamingo checkpoint
    """

    def __init__(
        self,
        model_name: str = "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, device, **kwargs)
        self.vision_encoder: Optional[nn.Module] = None
        self.perceiver_resampler: Optional[nn.Module] = None
        self.lang_model: Optional[nn.Module] = None
        self.tokenizer = None
        self.image_processor = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load Flamingo model via open_flamingo or transformers."""
        try:
            from open_flamingo import create_model_and_transforms
        except ImportError:
            # Fall back to HuggingFace auto model
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

            device = self._device if self._device != "auto" else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True,
            )
            self._model.to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True,
            )
            try:
                self.image_processor = AutoProcessor.from_pretrained(self.model_name)
            except Exception:
                self.image_processor = None
            return

        # open_flamingo path
        device = self._device if self._device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
        )
        self._model = model.to(device)
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        # Expose sub-components
        if hasattr(model, "vision_encoder"):
            self.vision_encoder = model.vision_encoder
        if hasattr(model, "perceiver"):
            self.perceiver_resampler = model.perceiver
        if hasattr(model, "lang_encoder"):
            self.lang_model = model.lang_encoder

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_image_features(
        self, images: Any, **kwargs,
    ) -> torch.Tensor:
        """
        Extract visual features through the vision encoder + Perceiver Resampler.

        Parameters
        ----------
        images : Tensor | PIL.Image | list
            Input images.  If tensor, expected shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Visual features of shape ``(B, num_visual_tokens, D)``.
        """
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(self.device)
        elif self.image_processor is not None:
            processed = self.image_processor(images=images, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(self.device)
        else:
            raise ValueError("Cannot process images — no image_processor loaded.")

        with torch.no_grad():
            if self.vision_encoder is not None:
                vision_out = self.vision_encoder(pixel_values)
                if isinstance(vision_out, tuple):
                    vision_out = vision_out[0]
                if self.perceiver_resampler is not None:
                    return self.perceiver_resampler(vision_out)
                return vision_out

            # Fallback: run full model forward to capture vision hidden states
            # Use hooks on the vision encoder sub-module
            features: Dict[str, torch.Tensor] = {}

            def _capture(module, _in, output):
                out = output[0] if isinstance(output, tuple) else output
                features["vision"] = out.detach()

            hook = None
            for name, mod in self._model.named_modules():
                if "vision" in name.lower() and isinstance(mod, (nn.Linear, nn.LayerNorm)):
                    hook = mod.register_forward_hook(_capture)
                    break

            if hook is not None:
                try:
                    dummy_ids = torch.zeros(pixel_values.size(0), 1, dtype=torch.long, device=self.device)
                    self._model(input_ids=dummy_ids, pixel_values=pixel_values.unsqueeze(1).unsqueeze(1))
                finally:
                    hook.remove()

            if "vision" in features:
                return features["vision"]

            return pixel_values.mean(dim=(-2, -1))  # minimal fallback

    def get_text_features(
        self, texts: Any, **kwargs,
    ) -> torch.Tensor:
        """
        Extract language model hidden states (last hidden layer, mean-pooled).
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        if isinstance(texts, str):
            texts = [texts]

        encoding = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # (B, seq_len, D)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            return hidden.mean(dim=1)

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def freeze_vision(self) -> None:
        """Freeze only the vision encoder parameters."""
        if self.vision_encoder is not None:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

    def freeze_language(self) -> None:
        """Freeze only the language model parameters."""
        if self.lang_model is not None:
            for p in self.lang_model.parameters():
                p.requires_grad = False

    def unfreeze_cross_attention(self) -> None:
        """Unfreeze only cross-attention (gated xattn) layers."""
        for name, p in self.model.named_parameters():
            if "cross_attn" in name or "gated_cross_attn" in name:
                p.requires_grad = True
