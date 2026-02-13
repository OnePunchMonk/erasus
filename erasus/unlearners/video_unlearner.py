"""
Video Unlearner — Orchestrator for Video Model (VideoMAE) unlearning.

Example::

    from erasus.unlearners import VideoUnlearner

    unlearner = VideoUnlearner(model=videomae_wrapper.model)
    result = unlearner.fit(forget_loader, retain_loader)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_unlearner import BaseUnlearner
from erasus.core.registry import strategy_registry, selector_registry

# Ensure all strategies and selectors are registered
import erasus.strategies  # noqa: F401
import erasus.selectors  # noqa: F401


class VideoUnlearner(BaseUnlearner):
    """
    Unlearner specialised for Video Models (VideoMAE).
    """

    DEFAULT_STRATEGY = "gradient_ascent"
    DEFAULT_SELECTOR = "gradient_norm"
    DEFAULT_EPOCHS = 10

    def __init__(
        self,
        model: nn.Module,
        strategy: str = DEFAULT_STRATEGY,
        selector: Optional[str] = DEFAULT_SELECTOR,
        device: Optional[str] = None,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        selector_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        strategy_cls = strategy_registry.get(strategy)
        strategy_instance = strategy_cls(**(strategy_kwargs or {}))

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
        epochs: int = DEFAULT_EPOCHS,
        **kwargs: Any,
    ) -> Tuple[List[float], List[float]]:
        self.model, forget_losses, retain_losses = self.strategy.unlearn(
            model=self.model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            **kwargs,
        )
        return forget_losses, retain_losses

    def evaluate_video(
        self,
        forget_data: DataLoader,
        retain_data: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate with accuracy + MIA metrics."""
        metrics_instances: List[Any] = []
        try:
            from erasus.metrics.accuracy import AccuracyMetric
            metrics_instances.append(AccuracyMetric())
        except Exception:
            pass
        try:
            from erasus.metrics.membership_inference import MembershipInferenceMetric
            metrics_instances.append(MembershipInferenceMetric())
        except Exception:
            pass
        return self.evaluate(
            forget_data=forget_data,
            retain_data=retain_data,
            metrics=metrics_instances,
        )
