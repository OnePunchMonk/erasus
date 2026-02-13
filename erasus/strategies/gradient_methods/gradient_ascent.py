"""
Vanilla Gradient Ascent — Baseline unlearning strategy.

Paper: Eternal Sunshine of the Spotless Net (Golatkar et al., NeurIPS 2020)
Formula: θₜ₊₁ = θₜ + η · ∇L_forget(θₜ)

Section 4.1.1 of the specification.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("gradient_ascent")
class GradientAscentStrategy(BaseStrategy):
    """
    Simplest unlearning: gradient ascent on forget set.

    Maximises the loss on the forget data so the model 'forgets' those
    samples.  Can catastrophically damage utility without retain-set
    regularisation — combine with a retain loader or use a more
    sophisticated strategy for production workloads.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        device = next(model.parameters()).device

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_forget_loss = 0.0
            n_forget = 0

            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(logits, labels)

                # MAXIMIZE loss → gradient ascent
                optimizer.zero_grad()
                (-loss).backward()
                optimizer.step()

                epoch_forget_loss += loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget_loss / max(n_forget, 1))

            # Optional retain-set pass (gradient descent to preserve utility)
            if retain_loader is not None:
                epoch_retain_loss = 0.0
                n_retain = 0
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = F.cross_entropy(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_retain_loss += loss.item()
                    n_retain += 1
                retain_losses.append(epoch_retain_loss / max(n_retain, 1))

        return model, forget_losses, retain_losses
