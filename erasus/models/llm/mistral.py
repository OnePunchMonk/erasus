"""
Mistral / Mixtral Wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from erasus.core.registry import model_registry
from erasus.models.model_wrapper import BaseLLMModel


@model_registry.register("mistral")
class MistralWrapper(BaseLLMModel):
    """Wrapper for Mistral / Mixtral models."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1", **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.tokenizer = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids, **kwargs)

    def get_layer_activations(self, text: str, layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        activations: Dict[str, torch.Tensor] = {}
        hooks = []

        def hook_fn(name):
            def hook(module, _input, output):
                activations[name] = output[0].detach()
            return hook

        for idx in layer_indices:
            layer = self.model.model.layers[idx]
            hooks.append(layer.register_forward_hook(hook_fn(f"layer_{idx}")))

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)

        for h in hooks:
            h.remove()
        return activations
