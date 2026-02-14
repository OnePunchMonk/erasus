"""
Wav2Vec 2.0 Model Wrapper — Self-supervised speech representation model.

Supports:
- Audio feature extraction at multiple encoder layers
- Fine-tuned CTC speech recognition
- Layer-wise gradient access for targeted unlearning

Reference: Baevski et al. (2020) — "wav2vec 2.0: A Framework for
Self-Supervised Learning of Speech Representations"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
except ImportError:
    Wav2Vec2ForCTC = None
    Wav2Vec2Processor = None

from erasus.models.model_wrapper import BaseVLMModel
from erasus.models.registry import model_registry


@model_registry.register("wav2vec")
class Wav2VecWrapper(BaseVLMModel):
    """
    Wav2Vec 2.0 wrapper for audio unlearning tasks.

    Features
    --------
    - Raw waveform input processing
    - Hidden-state extraction per transformer layer
    - CTC decoding for ASR evaluation
    - Gradient isolation between feature extractor and transformer encoder

    Supported models
    ----------------
    - ``facebook/wav2vec2-base-960h``
    - ``facebook/wav2vec2-large-960h``
    - ``facebook/wav2vec2-large-xlsr-53`` (multilingual)
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, device, **kwargs)
        self.processor = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if Wav2Vec2ForCTC is None:
            raise ImportError("transformers is required. pip install transformers")

        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)

        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_image_features(self, images: Any, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Wav2Vec is an audio model — use get_audio_features().")

    def get_text_features(self, texts: Any, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Wav2Vec is an audio model — use get_audio_features().")

    def get_audio_features(
        self,
        audio: Any,
        sampling_rate: int = 16_000,
        layer_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Extract audio features from the encoder.

        Parameters
        ----------
        audio : array-like | Tensor
            Raw waveform(s).
        sampling_rate : int
            Input sampling rate.
        layer_indices : list[int], optional
            If given, returns intermediate hidden states using hooks.

        Returns
        -------
        Tensor
            Audio features, mean-pooled to ``(B, D)``.
        """
        if isinstance(audio, torch.Tensor):
            input_values = audio.to(self.device)
        else:
            inputs = self.processor(
                audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True,
            )
            input_values = inputs.input_values.to(self.device)

        if layer_indices is not None:
            return self._extract_layer_features(input_values, layer_indices)

        with torch.no_grad():
            outputs = self._model.wav2vec2(input_values)
            return outputs.last_hidden_state.mean(dim=1)

    def _extract_layer_features(
        self, input_values: torch.Tensor, layer_indices: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Hook-based extraction for specific encoder layers."""
        features: Dict[str, torch.Tensor] = {}
        hooks: list = []

        encoder_layers = self._model.wav2vec2.encoder.layers

        def hook_fn(name: str):
            def hook(module, _input, output):
                out = output[0] if isinstance(output, tuple) else output
                features[name] = out.detach()
            return hook

        for idx in layer_indices:
            if idx < len(encoder_layers):
                hooks.append(encoder_layers[idx].register_forward_hook(hook_fn(f"layer_{idx}")))

        with torch.no_grad():
            self._model.wav2vec2(input_values)

        for h in hooks:
            h.remove()

        return features

    # ------------------------------------------------------------------
    # ASR decoding
    # ------------------------------------------------------------------

    def transcribe(self, audio: Any, sampling_rate: int = 16_000) -> List[str]:
        """
        Transcribe audio to text using CTC decoding.

        Parameters
        ----------
        audio : array-like
            Raw waveform(s).

        Returns
        -------
        list[str]
            Decoded transcriptions.
        """
        if isinstance(audio, torch.Tensor):
            input_values = audio.to(self.device)
        else:
            inputs = self.processor(
                audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True,
            )
            input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self._model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def freeze_feature_extractor(self) -> None:
        """Freeze the CNN feature extractor (keep transformer trainable)."""
        for p in self._model.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False
        if hasattr(self._model.wav2vec2, "feature_projection"):
            for p in self._model.wav2vec2.feature_projection.parameters():
                p.requires_grad = False

    def forward(self, audio: Any, labels: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass for CTC training / unlearning."""
        if isinstance(audio, torch.Tensor):
            input_values = audio.to(self.device)
        elif isinstance(audio, dict):
            input_values = audio["input_values"].to(self.device)
        else:
            inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000, padding=True)
            input_values = inputs.input_values.to(self.device)

        kwargs_fwd = {"input_values": input_values}
        if labels is not None:
            kwargs_fwd["labels"] = labels.to(self.device)

        return self._model(**kwargs_fwd)
