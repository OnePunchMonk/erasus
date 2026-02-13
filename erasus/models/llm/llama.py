"""
LLaMA Wrapper — Stub for Phase 1.

Full implementation will support:
- Causal language modeling
- Layer-wise gradient access
- Activation patching
- LoRA fine-tuning integration
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from erasus.core.registry import model_registry
from erasus.models.model_wrapper import BaseLLMModel


@model_registry.register("llama")
class LLaMAWrapper(BaseLLMModel):
    """
    LLaMA wrapper with support for:
    - Causal language modeling
    - Layer-wise gradient access
    - Activation patching
    - LoRA fine-tuning integration
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "auto",
        load_in_8bit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, device, **kwargs)
        self.load_in_8bit = load_in_8bit
        self.tokenizer = None

    def load(self) -> None:
        from transformers import LlamaForCausalLM, LlamaTokenizer

        self._model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=self.load_in_8bit,
            device_map=self._device if self._device != "auto" else "auto",
            torch_dtype=torch.float16 if self.load_in_8bit else torch.float32,
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids, **kwargs)

    def get_layer_activations(
        self, text: str, layer_indices: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Extract activations from specific transformer layers."""
        activations: Dict[str, torch.Tensor] = {}
        hooks: list = []

        def hook_fn(name: str):
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
