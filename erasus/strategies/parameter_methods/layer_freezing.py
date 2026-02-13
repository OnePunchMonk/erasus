"""
erasus.strategies.parameter_methods.layer_freezing — Selective layer freezing.

Freezes layers that are most important for retain-set performance
and only updates the remaining layers during unlearning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("layer_freezing")
class LayerFreezingStrategy(BaseStrategy):
    """
    Selective layer freezing for unlearning.

    Identifies which layers are most responsible for retain-set
    performance (by Fisher information), freezes those layers,
    and applies gradient ascent only on the remaining layers.

    Parameters
    ----------
    lr : float
        Learning rate for unfrozen layers.
    freeze_ratio : float
        Fraction of layers to freeze (0–1). Higher = more preservation.
    freeze_method : str
        ``"fisher"`` (Fisher information) or ``"last_n"`` (freeze early layers).
    """

    def __init__(
        self,
        lr: float = 1e-4,
        freeze_ratio: float = 0.5,
        freeze_method: str = "fisher",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.freeze_ratio = freeze_ratio
        self.freeze_method = freeze_method

    def _compute_layer_importance(
        self,
        model: nn.Module,
        retain_loader: DataLoader,
    ) -> Dict[str, float]:
        """Compute per-layer importance using Fisher information on retain data."""
        device = next(model.parameters()).device
        model.train()
        importance: Dict[str, float] = {}

        for batch in retain_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(logits, labels)

            model.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher = (param.grad ** 2).sum().item()
                    importance[name] = importance.get(name, 0) + fisher

        return importance

    def _select_frozen_layers(
        self,
        model: nn.Module,
        retain_loader: Optional[DataLoader],
    ) -> List[str]:
        """Select which parameter names to freeze."""
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        n_freeze = int(len(param_names) * self.freeze_ratio)

        if self.freeze_method == "last_n" or retain_loader is None:
            # Freeze the first `n_freeze` layers (early/generic layers)
            return param_names[:n_freeze]

        # Fisher-based: freeze layers most important for retain performance
        importance = self._compute_layer_importance(model, retain_loader)
        sorted_names = sorted(importance.keys(), key=lambda n: importance[n], reverse=True)
        return sorted_names[:n_freeze]

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run layer-freezing unlearning."""
        device = next(model.parameters()).device

        frozen_names = set(self._select_frozen_layers(model, retain_loader))

        # Freeze selected parameters
        for name, param in model.named_parameters():
            if name in frozen_names:
                param.requires_grad = False

        # Only optimise unfrozen parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n = 0

            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                (-loss).backward()
                optimizer.step()

                epoch_loss += loss.item()
                n += 1

            forget_losses.append(epoch_loss / max(n, 1))

            if retain_loader is not None:
                epoch_retain = 0.0
                n_r = 0
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = F.cross_entropy(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_retain += loss.item()
                    n_r += 1
                retain_losses.append(epoch_retain / max(n_r, 1))

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        return model, forget_losses, retain_losses
