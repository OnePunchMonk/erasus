"""
BLIP Model Wrapper.

Wraps BLIP (Bootstrapping Language-Image Pre-training) models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
except ImportError:
    BlipForConditionalGeneration = None
    BlipProcessor = None

from erasus.models.model_wrapper import BaseVLMModel
from erasus.models.registry import model_registry


@model_registry.register("blip")
class BLIPWrapper(BaseVLMModel):
    """
    Wrapper for BLIP models.
    """

    def __init__(self, model_name: str, device: str = "auto", **kwargs) -> None:
        super().__init__(model_name, device, **kwargs)
        self.processor = None

    def load(self) -> None:
        if BlipForConditionalGeneration is None:
            raise ImportError("transformers not installed.")
            
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def get_image_features(self, images: Any, **kwargs) -> torch.Tensor:
        if self._model is None: self.load()
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
             vision_outputs = self._model.vision_model(**inputs)
             return vision_outputs[1] # pooler_output

    def get_text_features(self, texts: Any, **kwargs) -> torch.Tensor:
        if self._model is None: self.load()
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
             text_outputs = self._model.text_encoder(**inputs)
             return text_outputs[1] # pooler_output
