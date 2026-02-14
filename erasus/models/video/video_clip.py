"""
VideoCLIP Model Wrapper — Contrastive Video-Language model.

Supports:
- Video-text contrastive feature extraction
- Frame-level temporal encoding
- Separate video/text encoder access

Reference: Xu et al. (2021) — "VideoCLIP: Contrastive Pre-training for
Zero-shot Video-Text Understanding"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from erasus.models.model_wrapper import BaseVLMModel
from erasus.models.registry import model_registry


@model_registry.register("video_clip")
class VideoCLIPWrapper(BaseVLMModel):
    """
    VideoCLIP wrapper for video-text unlearning.

    Features
    --------
    - Frame-level visual feature extraction
    - Temporal aggregation (mean / temporal transformer)
    - Text feature extraction via shared language encoder
    - Contrastive loss for video-text pairs

    Supported models
    ----------------
    - ``microsoft/xclip-base-patch32``  (X-CLIP: closest open-source)
    - ``MCG-NJU/videomae-base``  (video backbone alternative)
    """

    def __init__(
        self,
        model_name: str = "microsoft/xclip-base-patch32",
        device: str = "auto",
        num_frames: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, device, **kwargs)
        self.processor = None
        self.num_frames = num_frames

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        try:
            from transformers import XCLIPModel, XCLIPProcessor
            ModelClass = XCLIPModel
            ProcessorClass = XCLIPProcessor
        except ImportError:
            try:
                from transformers import AutoModel, AutoProcessor
                ModelClass = AutoModel
                ProcessorClass = AutoProcessor
            except ImportError:
                raise ImportError("transformers required. pip install transformers")

        device = self._device if self._device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._model = ModelClass.from_pretrained(self.model_name)
        self._model.to(device)
        self.processor = ProcessorClass.from_pretrained(self.model_name)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_image_features(self, images: Any, **kwargs) -> torch.Tensor:
        """
        Extract video features.

        Parameters
        ----------
        images : Tensor | list
            Video frames.  If Tensor, expected shape ``(B, T, C, H, W)``.

        Returns
        -------
        Tensor of shape ``(B, D)`` — video-level features.
        """
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(self.device)
        else:
            if self.processor is not None:
                inputs = self.processor(
                    videos=images, return_tensors="pt",
                )
                pixel_values = inputs["pixel_values"].to(self.device)
            else:
                raise ValueError("Cannot process video — processor not loaded.")

        with torch.no_grad():
            outputs = self._model.get_image_features(pixel_values=pixel_values)
            if isinstance(outputs, tuple):
                return outputs[0]
            return outputs

    def get_text_features(self, texts: Any, **kwargs) -> torch.Tensor:
        """
        Extract text features.

        Parameters
        ----------
        texts : str | list[str]
            Input text(s).

        Returns
        -------
        Tensor of shape ``(B, D)``
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            return self._model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask,
            )

    # ------------------------------------------------------------------
    # Contrastive utilities
    # ------------------------------------------------------------------

    def compute_contrastive_loss(
        self,
        videos: Any,
        texts: Any,
    ) -> torch.Tensor:
        """Symmetric contrastive loss for video-text pairs."""
        video_features = self.get_image_features(videos)
        text_features = self.get_text_features(texts)

        video_features = F.normalize(video_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = 1.0
        if hasattr(self._model, "logit_scale"):
            logit_scale = self._model.logit_scale.exp()

        logits = video_features @ text_features.T * logit_scale
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)

        return (loss_v2t + loss_t2v) / 2

    def zero_shot_classify(
        self,
        videos: Any,
        candidate_labels: List[str],
    ) -> Dict[str, float]:
        """
        Zero-shot video classification.

        Parameters
        ----------
        videos : Tensor | list
            Input video(s).
        candidate_labels : list[str]
            Text labels to score.

        Returns
        -------
        dict mapping label → probability
        """
        video_features = self.get_image_features(videos)
        text_features = self.get_text_features(candidate_labels)

        video_features = F.normalize(video_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        sims = (video_features @ text_features.T).squeeze(0)
        probs = F.softmax(sims, dim=-1)

        return {label: prob.item() for label, prob in zip(candidate_labels, probs)}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, videos: Any, texts: Any = None, **kwargs):
        """Full forward pass for video-text contrastive training."""
        # Prepare video
        if isinstance(videos, torch.Tensor):
            pixel_values = videos.to(self.device)
        else:
            inputs = self.processor(videos=videos, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

        fwd_kwargs = {"pixel_values": pixel_values}

        # Prepare text if given
        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            fwd_kwargs["input_ids"] = text_inputs["input_ids"].to(self.device)
            if "attention_mask" in text_inputs:
                fwd_kwargs["attention_mask"] = text_inputs["attention_mask"].to(self.device)

        return self._model(**fwd_kwargs)
