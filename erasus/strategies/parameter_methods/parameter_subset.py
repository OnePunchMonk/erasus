"""
Parameter-subset sparse unlearning strategy.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("parameter_subset")
class ParameterSubsetUnlearningStrategy(BaseStrategy):
    """
    Update only a sparse subset of parameters during unlearning.

    Parameters
    ----------
    sparsity : float
        Fraction of parameter tensors to update based on gradient norm.
    lr : float
        Learning rate.
    """

    def __init__(self, sparsity: float = 0.2, lr: float = 1e-4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sparsity = sparsity
        self.lr = lr

    def _active_parameter_names(self, model: nn.Module) -> set[str]:
        scored: list[tuple[str, float]] = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            scored.append((name, float(param.grad.norm().item())))

        if not scored:
            return set()

        keep = max(1, int(len(scored) * self.sparsity))
        scored.sort(key=lambda item: item[1], reverse=True)
        return {name for name, _ in scored[:keep]}

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for _ in range(epochs):
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                logits = model(inputs)
                if hasattr(logits, "logits"):
                    logits = logits.logits

                loss = -F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()

                active_names = self._active_parameter_names(model)
                for name, param in model.named_parameters():
                    if param.grad is not None and name not in active_names:
                        param.grad.zero_()

                optimizer.step()
                epoch_forget += float(loss.item())
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

            if retain_loader is not None:
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    logits = model(inputs)
                    if hasattr(logits, "logits"):
                        logits = logits.logits

                    loss = F.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()

                    active_names = self._active_parameter_names(model)
                    for name, param in model.named_parameters():
                        if param.grad is not None and name not in active_names:
                            param.grad.zero_()

                    optimizer.step()
                    epoch_retain += float(loss.item())
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        return model, forget_losses, retain_losses
