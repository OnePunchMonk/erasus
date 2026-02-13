"""
ErasusUnlearner — High-level API for machine unlearning.

This is the primary user-facing class. Example usage::

    from erasus import ErasusUnlearner

    unlearner = ErasusUnlearner(
        model=clip_model,
        strategy="modality_decoupling",
        selector="influence",
    )
    result = unlearner.fit(forget_data, retain_data)
    metrics = unlearner.evaluate(forget_data, retain_data)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_unlearner import BaseUnlearner, UnlearningResult
from erasus.core.registry import strategy_registry, selector_registry

# Ensure all strategies and selectors are registered
import erasus.strategies  # noqa: F401
import erasus.selectors  # noqa: F401


class ErasusUnlearner(BaseUnlearner):
    """
    Main Erasus unlearner — resolves strategy/selector by name,
    orchestrates coreset selection + unlearning + evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "gradient_ascent",
        selector: Optional[str] = None,
        device: Optional[str] = None,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        selector_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Resolve strategy
        strategy_cls = strategy_registry.get(strategy)
        strategy_instance = strategy_cls(**(strategy_kwargs or {}))

        # Resolve selector
        selector_instance = None
        if selector is not None:
            selector_cls = selector_registry.get(selector)
            selector_instance = selector_cls(**(selector_kwargs or {}))

        super().__init__(
            model=model,
            strategy=strategy_instance,
            selector=selector_instance,
            device=device,
            **kwargs,
        )
        self.strategy_name = strategy
        self.selector_name = selector

    def _run_unlearning(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int,
        **kwargs: Any,
    ) -> Tuple[List[float], List[float]]:
        """Delegate to the resolved strategy."""
        self.model, forget_losses, retain_losses = self.strategy.unlearn(
            model=self.model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            **kwargs,
        )
        return forget_losses, retain_losses
