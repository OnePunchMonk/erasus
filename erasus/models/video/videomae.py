"""
VideoMAE Model Wrapper.

Wraps VideoMAE (Masked Autoencoders are Scalable Vision Learners) for video unlearning tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
except ImportError:
    VideoMAEForVideoClassification = None
    VideoMAEImageProcessor = None

from erasus.models.model_wrapper import BaseVLMModel # Using VLM base as it is closest (Visual)
from erasus.models.registry import model_registry


@model_registry.register("videomae")
class VideoMAEWrapper(BaseVLMModel):
    """
    Wrapper for VideoMAE models (Video Classification).
    """

    def __init__(self, model_name: str, device: str = "auto", num_frames: int = 16, **kwargs) -> None:
        super().__init__(model_name, device, **kwargs)
        self.processor = None
        self.num_frames = num_frames

    def load(self) -> None:
        if VideoMAEForVideoClassification is None:
            raise ImportError("transformers not installed. Install with `pip install transformers`.")
            
        self._model = VideoMAEForVideoClassification.from_pretrained(self.model_name)
        self.processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
        
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def get_image_features(self, images: Any, **kwargs) -> torch.Tensor:
        """
        Extract video features.
        'images' argument typically expects a list of videos (list of list of frames) or [B, T, C, H, W].
        """
        if self._model is None: self.load()
        
        # Handling input format is complex for video
        # Assume input is [B, T, C, H, W] tensor or List[List[np.array]] for processor
        
        # If tensor on device, pass directly to model if simplified
        if isinstance(images, torch.Tensor):
             # VideoMAE expects [B, T, C, H, W] -> pixel_values
             inputs = {"pixel_values": images.to(self.device)}
        else:
             # Use processor
             # inputs = self.processor(images, return_tensors="pt")
             # inputs = {k: v.to(self.device) for k, v in inputs.items()}
             raise NotImplementedError("Raw video list processing not yet implemented. Pass Tensor [B, T, C, H, W].")

        with torch.no_grad():
            outputs = self._model.videomae(**inputs)
            # Last hidden state: [B, T/patch, D]
            # Mean pool for video representation
            features = outputs.last_hidden_state.mean(dim=1)
            
        return features

    def get_text_features(self, texts: Any, **kwargs) -> torch.Tensor:
        raise NotImplementedError("VideoMAE is a vision-only model.")

    def forward(self, images, labels=None, **kwargs):
        if self._model is None: self.load()
        
        if isinstance(images, torch.Tensor):
             inputs = {"pixel_values": images.to(self.device)}
        else:
             # Processor usage
             inputs = self.processor(list(images), return_tensors="pt")
             inputs = {k: v.to(self.device) for k, v in inputs.items()}
             
        if labels is not None:
            inputs["labels"] = labels.to(self.device)
            
        return self._model(**inputs)
