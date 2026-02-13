"""
GPT-2 / GPT-J Wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from erasus.core.registry import model_registry
from erasus.models.model_wrapper import BaseLLMModel


@model_registry.register("gpt2")
class GPTWrapper(BaseLLMModel):
    """Wrapper for GPT-2 family models."""

    def __init__(self, model_name: str = "gpt2", **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.tokenizer = None

    def load(self) -> None:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        device = self._device if self._device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = GPT2LMHeadModel.from_pretrained(self.model_name).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

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
            layer = self.model.transformer.h[idx]
            hooks.append(layer.register_forward_hook(hook_fn(f"layer_{idx}")))

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)

        for h in hooks:
            h.remove()
        return activations
