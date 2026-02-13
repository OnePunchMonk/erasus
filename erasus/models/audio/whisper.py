"""
Whisper Model Wrapper.

Wraps OpenAI Whisper for audio unlearning tasks (ASR).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
except ImportError:
    WhisperForConditionalGeneration = None
    WhisperProcessor = None

from erasus.models.model_wrapper import BaseVLMModel # Using VLM base as generalized "Multimodal"
from erasus.models.registry import model_registry


@model_registry.register("whisper")
class WhisperWrapper(BaseVLMModel):
    """
    Wrapper for Whisper models.
    """

    def __init__(self, model_name: str, device: str = "auto", **kwargs) -> None:
        super().__init__(model_name, device, **kwargs)
        self.processor = None

    def load(self) -> None:
        if WhisperForConditionalGeneration is None:
            raise ImportError("transformers not installed.")
            
        self._model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def get_image_features(self, images: Any, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Whisper is an Audio model, not Image.")

    def get_text_features(self, texts: Any, **kwargs) -> torch.Tensor:
         """
         Whisper decoder can produce text embeddings?
         Not typically used this way. 
         """
         raise NotImplementedError("Whisper does not expose text encoder embeddings in this interface.")

    # Custom method for Audio Features?
    def get_audio_features(self, audio: Any) -> torch.Tensor:
        if self._model is None: self.load()
        
        # audio: raw waveform array or tensor
        if isinstance(audio, torch.Tensor):
             input_features = audio.to(self.device) # Assumption pre-processed log-mel
        else:
             inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
             input_features = inputs.input_features.to(self.device)
             
        with torch.no_grad():
            enc_out = self._model.model.encoder(input_features)
            # [B, T, D] -> Mean pool
            features = enc_out.last_hidden_state.mean(dim=1)
            
        return features

    def forward(self, audio, labels=None, **kwargs):
        """
        Forward pass for ASR training/unlearning.
        audio: input features (log-mel)
        labels: target token ids
        """
        if self._model is None: self.load()
        
        # We need input_features and labels (decoder_input_ids usually auto-shifted if labels provided)
        # Assuming audio is pre-processed or we rely on loader
        
        inputs = {}
        if isinstance(audio, torch.Tensor):
            inputs["input_features"] = audio.to(self.device)
        elif isinstance(audio, dict):
             inputs = {k: v.to(self.device) for k, v in audio.items()}
        
        if labels is not None:
            inputs["labels"] = labels.to(self.device)
            
        return self._model(**inputs)
