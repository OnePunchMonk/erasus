"""
erasus.strategies.ensemble_strategy — Combine multiple unlearning strategies.

Runs multiple strategies sequentially or averages their parameter updates,
enabling ensemble unlearning for improved robustness.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("ensemble")
class EnsembleStrategy(BaseStrategy):
    """
    Combine multiple unlearning strategies.

    Modes:
    - ``"sequential"``: Run strategies one after another on the same model.
    - ``"average"``: Run each strategy on a copy, average the final weights.

    Parameters
    ----------
    strategies : list[BaseStrategy]
        Strategy instances to combine.
    mode : str
        ``"sequential"`` or ``"average"``.
    """

    def __init__(
        self,
        strategies: Optional[List[BaseStrategy]] = None,
        strategy_names: Optional[List[str]] = None,
        mode: str = "sequential",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mode = mode

        if strategies is not None:
            self._strategies = strategies
        elif strategy_names is not None:
            self._strategies = []
            for name in strategy_names:
                cls = strategy_registry.get(name)
                self._strategies.append(cls(**kwargs))
        else:
            raise ValueError("Provide either `strategies` or `strategy_names`.")

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run ensemble unlearning."""
        if self.mode == "sequential":
            return self._run_sequential(model, forget_loader, retain_loader, epochs, **kwargs)
        elif self.mode == "average":
            return self._run_averaged(model, forget_loader, retain_loader, epochs, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _run_sequential(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run strategies sequentially on the same model."""
        all_forget: List[float] = []
        all_retain: List[float] = []

        epochs_per = max(1, epochs // len(self._strategies))

        for strategy in self._strategies:
            model, f_losses, r_losses = strategy.unlearn(
                model, forget_loader, retain_loader, epochs=epochs_per, **kwargs,
            )
            all_forget.extend(f_losses)
            all_retain.extend(r_losses)

        return model, all_forget, all_retain

    def _run_averaged(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run each strategy on a copy, then average weights."""
        copies = []
        all_forget: List[float] = []
        all_retain: List[float] = []

        for strategy in self._strategies:
            model_copy = copy.deepcopy(model)
            model_copy, f_losses, r_losses = strategy.unlearn(
                model_copy, forget_loader, retain_loader, epochs=epochs, **kwargs,
            )
            copies.append(model_copy)
            all_forget.extend(f_losses)
            all_retain.extend(r_losses)

        # Average parameters
        with torch.no_grad():
            avg_state = {}
            for key in model.state_dict():
                tensors = [c.state_dict()[key].float() for c in copies]
                avg_state[key] = torch.stack(tensors).mean(dim=0).to(tensors[0].dtype)
            model.load_state_dict(avg_state)

        return model, all_forget, all_retain
