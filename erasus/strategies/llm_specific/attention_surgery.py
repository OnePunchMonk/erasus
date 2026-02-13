"""
erasus.strategies.llm_specific.attention_surgery — Attention weight modification.

Directly modifies attention weights to suppress specific knowledge
patterns without full gradient-based unlearning.

Reference: Meng et al. (2022) — "Locating and Editing Factual
           Associations in GPT", adapted for unlearning.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("attention_surgery")
class AttentionSurgeryStrategy(BaseStrategy):
    """
    Attention weight surgery for LLM unlearning.

    Identifies attention heads most activated by forget data,
    then dampens their output projections.

    Parameters
    ----------
    lr : float
        Learning rate for fine-tuning after surgery.
    dampening_factor : float
        Factor to scale down target attention heads (0–1).
    top_heads : int
        Number of top attention heads to modify per layer.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        dampening_factor: float = 0.1,
        top_heads: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.dampening_factor = dampening_factor
        self.top_heads = top_heads

    def _find_attention_layers(self, model: nn.Module) -> list:
        """Find all attention projection layers in the model."""
        attn_layers = []
        for name, module in model.named_modules():
            name_lower = name.lower()
            if any(k in name_lower for k in ("attn", "attention", "self_attn")):
                if hasattr(module, "weight") and isinstance(module, nn.Linear):
                    attn_layers.append((name, module))
        return attn_layers

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run attention surgery unlearning."""
        device = next(model.parameters()).device
        model.train()

        # Phase 1: Identify high-activation attention weights on forget data
        attn_activations = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, inp, out):
                if name not in attn_activations:
                    attn_activations[name] = 0.0
                act = out if isinstance(out, torch.Tensor) else out[0]
                attn_activations[name] += act.abs().mean().item()
            return hook_fn

        attn_layers = self._find_attention_layers(model)
        for name, module in attn_layers:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

        # Forward pass on forget data to measure activations
        with torch.no_grad():
            for batch in forget_loader:
                inputs = batch[0].to(device)
                model(inputs)

        for h in hooks:
            h.remove()

        # Phase 2: Dampen top activated attention projections
        if attn_activations:
            sorted_layers = sorted(attn_activations.items(), key=lambda x: x[1], reverse=True)
            n_modify = min(self.top_heads, len(sorted_layers))

            with torch.no_grad():
                for name, _ in sorted_layers[:n_modify]:
                    for n, module in attn_layers:
                        if n == name and hasattr(module, "weight"):
                            module.weight.mul_(self.dampening_factor)
                            if module.bias is not None:
                                module.bias.mul_(self.dampening_factor)

        # Phase 3: Fine-tune on retain data to recover utility
        forget_losses: List[float] = []
        retain_losses: List[float] = []

        if retain_loader is not None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            for epoch in range(epochs):
                epoch_loss = 0.0
                n = 0
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = F.cross_entropy(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n += 1
                retain_losses.append(epoch_loss / max(n, 1))

        return model, forget_losses, retain_losses
