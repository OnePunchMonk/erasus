"""
CLAP Model Wrapper — Contrastive Language-Audio Pretraining.

Supports:
- Audio-text contrastive feature extraction
- Separate audio/text encoder access
- Gradient isolation for modality-specific unlearning

Reference: Elizalde et al. (2023) — "CLAP: Learning Audio Concepts
from Natural Language Supervision"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from erasus.models.model_wrapper import BaseVLMModel
from erasus.models.registry import model_registry


@model_registry.register("clap")
class CLAPWrapper(BaseVLMModel):
    """
    CLAP audio-text model wrapper.

    Features
    --------
    - Audio encoder (HTSAT / CNN14) feature extraction
    - Text encoder (RoBERTa) feature extraction
    - Contrastive loss for audio-text pairs
    - Zero-shot audio classification

    Supported models
    ----------------
    - ``laion/clap-htsat-unfused``
    - ``laion/clap-htsat-fused``
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, device, **kwargs)
        self.processor = None
        self.audio_model: Optional[nn.Module] = None
        self.text_model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError:
            raise ImportError("transformers >= 4.31 required for CLAP. pip install transformers")

        device = self._device if self._device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._model = ClapModel.from_pretrained(self.model_name)
        self._model.to(device)
        self.processor = ClapProcessor.from_pretrained(self.model_name)

        if hasattr(self._model, "audio_model"):
            self.audio_model = self._model.audio_model
        if hasattr(self._model, "text_model"):
            self.text_model = self._model.text_model

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_image_features(self, images: Any, **kwargs) -> torch.Tensor:
        """CLAP uses audio not images — delegates to get_audio_features."""
        return self.get_audio_features(images)

    def get_text_features(self, texts: Any, **kwargs) -> torch.Tensor:
        """
        Extract text features from the text encoder.

        Parameters
        ----------
        texts : str | list[str]
            Input texts.

        Returns
        -------
        Tensor of shape ``(B, D)``
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

        with torch.no_grad():
            return self._model.get_text_features(**inputs)

    def get_audio_features(
        self,
        audio: Any,
        sampling_rate: int = 48_000,
    ) -> torch.Tensor:
        """
        Extract audio features from the audio encoder.

        Parameters
        ----------
        audio : array-like | Tensor
            Raw waveform(s) or pre-processed features.
        sampling_rate : int
            Input sampling rate.

        Returns
        -------
        Tensor of shape ``(B, D)``
        """
        if isinstance(audio, torch.Tensor):
            # Assume pre-processed input features
            input_features = audio.to(self.device)
            with torch.no_grad():
                return self._model.get_audio_features(input_features=input_features)

        inputs = self.processor(
            audios=audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True,
        )
        input_features = inputs["input_features"].to(self.device)

        with torch.no_grad():
            return self._model.get_audio_features(input_features=input_features)

    # ------------------------------------------------------------------
    # Contrastive utilities
    # ------------------------------------------------------------------

    def compute_contrastive_loss(
        self,
        audio: Any,
        texts: Any,
    ) -> torch.Tensor:
        """
        Symmetric CLAP contrastive loss (audio-text version of CLIP loss).
        """
        audio_features = self.get_audio_features(audio)
        text_features = self.get_text_features(texts)

        audio_features = F.normalize(audio_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = 1.0
        if hasattr(self._model, "logit_scale_a"):
            logit_scale = self._model.logit_scale_a.exp()
        elif hasattr(self._model, "logit_scale"):
            logit_scale = self._model.logit_scale.exp()

        logits = audio_features @ text_features.T * logit_scale
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.T, labels)

        return (loss_a2t + loss_t2a) / 2

    def zero_shot_classify(
        self,
        audio: Any,
        candidate_labels: List[str],
    ) -> Dict[str, float]:
        """
        Zero-shot audio classification.

        Parameters
        ----------
        audio : array-like
            Input audio.
        candidate_labels : list[str]
            Text labels to score.

        Returns
        -------
        dict mapping label → probability
        """
        audio_features = self.get_audio_features(audio)
        text_features = self.get_text_features(candidate_labels)

        audio_features = F.normalize(audio_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarities = (audio_features @ text_features.T).squeeze(0)
        probs = F.softmax(similarities, dim=-1)

        return {label: prob.item() for label, prob in zip(candidate_labels, probs)}

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def freeze_audio_encoder(self) -> None:
        """Freeze the audio encoder parameters."""
        if self.audio_model is not None:
            for p in self.audio_model.parameters():
                p.requires_grad = False

    def freeze_text_encoder(self) -> None:
        """Freeze the text encoder parameters."""
        if self.text_model is not None:
            for p in self.text_model.parameters():
                p.requires_grad = False
