"""
Selective Synaptic Dampening (SSD) for LLMs.

Paper: Knowledge Unlearning for Mitigating Privacy Risks in Language Models
       (Foster et al., NeurIPS 2024)
Formula: θᵢ' = θᵢ · (1 - α · 𝟙[aᵢ > τ])

Section 4.2.2 of the specification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("ssd")
class SelectiveSynapticDampeningStrategy(BaseStrategy):
    """
    SSD: Identify and dampen neurons activated by the forget set.

    Works by:
    1. Track neuron activations on forget set
    2. Identify 'forget neurons' (high activation)
    3. Dampen those neurons
    """

    def __init__(
        self,
        threshold: float = 0.5,
        damping_factor: float = 0.9,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.damping_factor = damping_factor

    def identify_forget_neurons(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        layer_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Identify neurons strongly activated by the forget set."""
        activations: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
        hooks: list = []

        def hook_fn(name: str):
            def hook(module, _input, output):
                out = output[0] if isinstance(output, tuple) else output
                act = out.detach().mean(dim=list(range(out.dim() - 1)))
                activations[name].append(act.cpu())
            return hook

        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        model.eval()
        with torch.no_grad():
            for batch in forget_loader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                if isinstance(inputs, dict):
                    _ = model(**inputs)
                else:
                    _ = model(inputs)

        for h in hooks:
            h.remove()

        masks: Dict[str, torch.Tensor] = {}
        for name in layer_names:
            if activations[name]:
                avg_act = torch.stack(activations[name]).mean(dim=0)
                masks[name] = (avg_act > self.threshold).float()
        return masks

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        layer_names = [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]

        masks = self.identify_forget_neurons(model, forget_loader, layer_names)

        # Apply dampening
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in masks:
                    mask = masks[name].to(module.weight.device)
                    # Weight shape (out_features, in_features); mask from output dim (out_features)
                    module.weight.data *= (
                        1 - (1 - self.damping_factor) * mask.unsqueeze(1)
                    )
                    if module.bias is not None:
                        module.bias.data *= (1 - (1 - self.damping_factor) * mask)

        return model, [], []
