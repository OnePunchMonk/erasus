"""
LLaVA Model Wrapper.

Wraps LLaVA (Large Language-and-Vision Assistant) models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor
except ImportError:
    LlavaForConditionalGeneration = None
    AutoProcessor = None

from erasus.models.model_wrapper import BaseVLMModel
from erasus.models.registry import model_registry


@model_registry.register("llava")
class LLaVAWrapper(BaseVLMModel):
    """
    Wrapper for LLaVA models, normalizing interface to Erasus VLM standards.
    """

    def __init__(self, model_name: str, device: str = "auto", **kwargs) -> None:
        super().__init__(model_name, device, **kwargs)
        self.processor = None

    def load(self) -> None:
        if LlavaForConditionalGeneration is None:
            raise ImportError("transformers not installed. Install with `pip install transformers`.")
            
        # Optimization: use 4bit/8bit if kwargs valid? For now standard load.
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if "cuda" in str(self._device) else torch.float32 
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def get_image_features(self, images: Any, **kwargs) -> torch.Tensor:
        """
        Extract image representations. 
        For LLaVA, this usually means vision_tower outputs projected to embedding space.
        """
        if self._model is None:
            self.load()
            
        # Basic check if images are raw or preprocessed
        # Processor handles raw images
        inputs = self.processor(images=images, text=[""] * len(images), return_tensors="pt", padding=True)
        # We only need pixel_values on device
        pixel_values = inputs.pixel_values.to(self.device)
        
        with torch.no_grad():
            # Access internal vision tower
            # The model usually has .vision_tower 
            # forward pass: vision_tower(pixel_values) -> last_hidden_state
            
            # However, LLaVA also has a 'multi_modal_projector'. 
            # Theoretically 'image features' for alignment should be after projection?
            # Or raw visual features? 
            # Usually raw features from CLIP tower are used for alignment metrics.
            
            image_outputs = self._model.vision_tower(pixel_values, output_hidden_states=True)
            # Pool features (e.g. mean of patches) to get [B, D]
            features = image_outputs.last_hidden_state.mean(dim=1)
            
        return features

    def get_text_features(self, texts: Any, **kwargs) -> torch.Tensor:
        """
        LLaVA is a Decoder-only LLM conditioned on images. It doesn't have a 
        standalone 'text encoder' producing semantic embeddings in the same space as images (unlike CLIP).
        
        However, for 'Unlearning' metrics like Cosine Similarity, we might want 
        the LLM's internal representation of the text.
        
        We'll implementation a dummy/proxy or raise error.
        Given strict interface requirements, we raise error to indicate incompatibility with CLIP-style strategies.
        """
        raise NotImplementedError(
            "LLaVA is a generative VLM (Decoder), not a Dual-Encoder (CLIP-like). "
            "It does not support 'get_text_features' for contrastive alignment."
        )

    def forward(self, images=None, texts=None, labels=None, **kwargs):
        """
        Unified forward pass. 
        Input: images [List/Tensor], texts [List/str]
        If texts provided, we format prompt for LLaVA.
        """
        if self._model is None: self.load()
        
        # If training/unlearning, we usually computing loss = CausalLM Loss
        # LLaVA expects keys: input_ids, pixel_values, labels
        
        # Processor usage:
        # prompt = "USER: <image>\n<text>\nASSISTANT:"
        if texts is None: texts = [""] * len(images) # Dummy
        
        prompts = [f"USER: <image>\n{t}\nASSISTANT:" for t in texts]
        
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if labels is not None:
            # We need to align labels with input_ids length... complicated for VLM.
            # Assuming labels are provided externally pre-aligned or ignored here.
            # Usually for unlearning we just use model(..., labels=input_ids) for self-supervision on prompt?
            # Or we minimize log(p(target|image, prompt)).
            
            # Simple fallback: pass explicit labels if present
            inputs["labels"] = labels.to(self.device)
            
        return self._model(**inputs)
