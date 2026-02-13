"""
CLIPWrapper — Unified wrapper for all CLIP variants.

Supports:
- Separate image/text encoder access
- Gradient isolation for modality-specific unlearning
- Contrastive loss manipulation
- Feature extraction at multiple layers via hooks

Location: erasus/models/vlm/clip.py (Section 2.1.1)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from erasus.core.registry import model_registry
from erasus.models.model_wrapper import BaseVLMModel


@model_registry.register("clip")
class CLIPWrapper(BaseVLMModel):
    """
    Unified wrapper for all CLIP variants.

    Features
    --------
    - Separate ``vision_model`` / ``text_model`` access
    - Gradient isolation for modality-specific unlearning
    - Contrastive loss manipulation
    - Feature extraction at multiple layers via hooks

    Supported models
    ----------------
    - ``openai/clip-vit-base-patch32``   (151 M, Phase 1)
    - ``openai/clip-vit-base-patch16``   (149 M, Phase 1)
    - ``openai/clip-vit-large-patch14``  (428 M, Phase 1)
    - OpenCLIP variants (Phase 2/3)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, device, **kwargs)
        self.processor = None
        self.vision_model: Optional[nn.Module] = None
        self.text_model: Optional[nn.Module] = None
        self.logit_scale: Optional[nn.Parameter] = None
        # Auto-load
        self.load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        device = self._device if self._device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.to(device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        # Convenience aliases
        self.vision_model = self._model.vision_model
        self.text_model = self._model.text_model
        self.logit_scale = self._model.logit_scale

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_image_features(
        self,
        images: Any,
        layer_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract image features.

        Parameters
        ----------
        images : Tensor | PIL.Image | list
            Raw images or pre-processed pixel values.
        layer_indices : list[int], optional
            If given, returns a dict mapping ``"layer_{i}"`` → hidden state
            from each requested vision-encoder layer (via hooks).

        Returns
        -------
        Tensor  (if *layer_indices* is ``None``)
        dict[str, Tensor]  (otherwise)
        """
        pixel_values = self._prepare_images(images)

        if layer_indices is None:
            return self._model.get_image_features(pixel_values=pixel_values)

        # Hook-based extraction for specific layers
        features: Dict[str, torch.Tensor] = {}
        hooks: list = []

        def _hook_fn(name: str):
            def hook(module, _input, output):
                features[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
            return hook

        for idx in layer_indices:
            layer = self.vision_model.encoder.layers[idx]
            hooks.append(layer.register_forward_hook(_hook_fn(f"layer_{idx}")))

        _ = self._model.get_image_features(pixel_values=pixel_values)

        for h in hooks:
            h.remove()

        return features

    def get_text_features(
        self,
        texts: Any,
        layer_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract text features, optionally at specific layers.
        """
        input_ids, attention_mask = self._prepare_texts(texts)

        if layer_indices is None:
            return self._model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask,
            )

        features: Dict[str, torch.Tensor] = {}
        hooks: list = []

        def _hook_fn(name: str):
            def hook(module, _input, output):
                features[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
            return hook

        for idx in layer_indices:
            layer = self.text_model.encoder.layers[idx]
            hooks.append(layer.register_forward_hook(_hook_fn(f"layer_{idx}")))

        _ = self._model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask,
        )

        for h in hooks:
            h.remove()

        return features

    # ------------------------------------------------------------------
    # Contrastive utilities
    # ------------------------------------------------------------------

    def compute_contrastive_loss(
        self,
        images: Any,
        texts: Any,
    ) -> torch.Tensor:
        """Symmetric CLIP contrastive loss."""
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits = image_features @ text_features.T * self.logit_scale.exp()
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_images(self, images: Any) -> torch.Tensor:
        """Return pixel_values tensor on the correct device."""
        if isinstance(images, torch.Tensor):
            return images.to(self.device)
        # Assume PIL / list-of-PIL
        processed = self.processor(images=images, return_tensors="pt")
        return processed["pixel_values"].to(self.device)

    def _prepare_texts(self, texts: Any):
        """Return (input_ids, attention_mask) on the correct device."""
        if isinstance(texts, dict):
            return (
                texts["input_ids"].to(self.device),
                texts.get("attention_mask", None),
            )
        if isinstance(texts, torch.Tensor):
            return texts.to(self.device), None
        # Assume raw strings
        processed = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        return (
            processed["input_ids"].to(self.device),
            processed["attention_mask"].to(self.device),
        )
