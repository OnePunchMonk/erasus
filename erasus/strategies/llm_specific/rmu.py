"""
RMU — Representation Misdirection for Unlearning.

Paper: "Representation Engineering Unlearning: Erasing Representations
with Targeted Steering" / "RMU" (Li et al., 2024)

Widely recognised as the strongest method on WMDP (hazardous knowledge
removal).  Rather than modifying output distributions, RMU manipulates
*internal representations*:

- For forget-set inputs: steer hidden states at a target layer toward
  a random direction (misdirection).
- For retain-set inputs: keep hidden states close to those of a frozen
  reference model (preservation).

This operates at a deeper level than output-based methods and is
substantially harder to reverse via benign fine-tuning.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("rmu")
class RMUStrategy(BaseStrategy):
    """
    Representation Misdirection for Unlearning.

    Parameters
    ----------
    layer_ids : list[int] or None
        Indices of layers (by depth order in ``model.named_modules()``)
        at which to hook.  If None, uses the middle third of the network.
    alpha : float
        Weight of the forget misdirection loss (default 1.0).
    retain_weight : float
        Weight of the retain representation preservation loss (default 1.0).
    lr : float
        Learning rate (default 1e-5).
    random_seed : int
        Seed for generating the random misdirection target (default 42).
    """

    def __init__(
        self,
        layer_ids: Optional[List[int]] = None,
        alpha: float = 1.0,
        retain_weight: float = 1.0,
        lr: float = 1e-5,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.layer_ids = layer_ids
        self.alpha = alpha
        self.retain_weight = retain_weight
        self.lr = lr
        self.random_seed = random_seed

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        device = next(model.parameters()).device

        # Identify hook layers
        all_layers = [
            (name, mod) for name, mod in model.named_modules()
            if isinstance(mod, (nn.Linear, nn.Conv2d))
        ]
        if not all_layers:
            # Fallback: hook on any module with parameters
            all_layers = [(n, m) for n, m in model.named_modules() if list(m.parameters())]

        if self.layer_ids is not None:
            hook_layers = [all_layers[i] for i in self.layer_ids if i < len(all_layers)]
        else:
            # Middle third
            n = len(all_layers)
            hook_layers = all_layers[n // 3: 2 * n // 3] or all_layers[:1]

        # Frozen reference for retain preservation
        if retain_loader is not None:
            ref_model = copy.deepcopy(model).to(device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
        else:
            ref_model = None

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Fixed random misdirection target (same seed each run for reproducibility)
        rng = torch.Generator(device=device)
        rng.manual_seed(self.random_seed)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            # --- Forget pass ---
            for batch in forget_loader:
                inputs = batch[0].to(device)

                # Capture hidden states with forward hooks
                forget_hidden = self._capture_hidden(model, inputs, hook_layers)

                # Generate random misdirection targets (same shape as hidden states)
                misdirection_loss = torch.tensor(0.0, device=device)
                for name, hidden in forget_hidden.items():
                    target = torch.randn_like(hidden, generator=rng)
                    target = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
                    target = target * hidden.norm(dim=-1, keepdim=True).detach()
                    misdirection_loss = misdirection_loss + F.mse_loss(hidden, target.detach())

                loss = self.alpha * misdirection_loss / max(len(forget_hidden), 1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_forget += loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

            # --- Retain pass: preserve representations ---
            if retain_loader is not None and ref_model is not None:
                for batch in retain_loader:
                    inputs = batch[0].to(device)

                    # Reference hidden states (frozen)
                    with torch.no_grad():
                        ref_hidden = self._capture_hidden(ref_model, inputs, hook_layers)

                    # Student hidden states
                    student_hidden = self._capture_hidden(model, inputs, hook_layers)

                    preserve_loss = torch.tensor(0.0, device=device)
                    for name in ref_hidden:
                        if name in student_hidden:
                            preserve_loss = preserve_loss + F.mse_loss(
                                student_hidden[name], ref_hidden[name].detach()
                            )

                    retain_loss = self.retain_weight * preserve_loss / max(len(ref_hidden), 1)

                    optimizer.zero_grad()
                    retain_loss.backward()
                    optimizer.step()

                    epoch_retain += retain_loss.item()
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        return model, forget_losses, retain_losses

    @staticmethod
    def _capture_hidden(
        model: nn.Module,
        inputs: torch.Tensor,
        hook_layers: List[Tuple[str, nn.Module]],
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass and capture the output of each hook layer."""
        captured: Dict[str, torch.Tensor] = {}
        hooks = []

        for name, layer in hook_layers:
            def make_hook(n: str):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[n] = h
                return hook
            hooks.append(layer.register_forward_hook(make_hook(name)))

        try:
            if isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        finally:
            for h in hooks:
                h.remove()

        return captured
