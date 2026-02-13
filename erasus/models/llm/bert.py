"""
BERT Model Wrapper.

Wraps Hugging Face BERT-style models for unlearning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    # Fallback for testing without dependencies
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from erasus.models.model_wrapper import BaseLLMModel
from erasus.models.registry import model_registry


@model_registry.register("bert")
class BERTWrapper(BaseLLMModel):
    """
    Wrapper for BERT and other Encoder-only models.
    """

    def __init__(self, model_name: str, device: str = "auto", **kwargs) -> None:
        super().__init__(model_name, device, **kwargs)
        self.tokenizer = None

    def load(self) -> None:
        if AutoModelForSequenceClassification is None:
            raise ImportError("transformers not installed.")
            
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        BERT is not a generative model (decoder-only). 
        This method raises NotImplementedError or returns reasonable dummy.
        """
        raise NotImplementedError("BERT is an Encoder-only model and does not support generation.")

    def get_layer_activations(self, text: str, layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Extract activations from specific layers.
        """
        if self._model is None:
            self.load()
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Register hooks
        activations = {}
        hooks = []
        
        def get_hook(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
            
        # Generic logic to find layers - highly dependent on specific architecture (bert/roberta)
        # Using a simplistic approach assuming 'encoder.layer' structure
        base_model = getattr(self._model, "bert", getattr(self._model, "roberta", None))
        
        if base_model:
            layers = base_model.encoder.layer
            for idx in layer_indices:
                if 0 <= idx < len(layers):
                    hooks.append(layers[idx].register_forward_hook(get_hook(f"layer_{idx}")))
        
        with torch.no_grad():
            self._model(**inputs)
            
        for h in hooks:
            h.remove()
            
        return activations
