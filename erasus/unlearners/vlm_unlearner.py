"""
VLM Unlearner — Orchestrator for Vision-Language Model unlearning.

Handles CLIP, LLaVA, and BLIP models with modality-aware defaults:
- Default strategy: modality_decoupling (prevents forgetting cascade)
- Default selector: influence (LiSSA approximation)
- Default metrics:  zero-shot retrieval + MIA

Example::

    from erasus.unlearners import VLMUnlearner

    unlearner = VLMUnlearner(model=clip_model)
    result = unlearner.fit(forget_loader, retain_loader, prune_ratio=0.1)
    metrics = unlearner.evaluate(forget_loader, retain_loader)
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


class VLMUnlearner(BaseUnlearner):
    """
    Unlearner specialised for Vision-Language Models.

    Automatically selects:
    - ``modality_decoupling`` strategy (differential LR for vision/text)
    - ``influence`` selector (LiSSA-based coreset)
    - Retrieval + MIA evaluation metrics

    All defaults can be overridden via constructor kwargs.
    """

    # Reasonable defaults for VLM unlearning
    DEFAULT_STRATEGY = "modality_decoupling"
    DEFAULT_SELECTOR = "influence"
    DEFAULT_PRUNE_RATIO = 0.1
    DEFAULT_EPOCHS = 50

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
        # Apply VLM-specific strategy defaults
        _strategy_kw = {"vision_lr": 1e-5, "text_lr": 1e-4, "alignment_weight": 0.1}
        if strategy_kwargs:
            _strategy_kw.update(strategy_kwargs)

        # Resolve components from registry
        strategy_cls = strategy_registry.get(strategy)
        strategy_instance = strategy_cls(**_strategy_kw)

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

    # ------------------------------------------------------------------
    # Core unlearning delegation
    # ------------------------------------------------------------------

    def _run_unlearning(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int = DEFAULT_EPOCHS,
        **kwargs: Any,
    ) -> Tuple[List[float], List[float]]:
        """Delegate to the modality-decoupling (or overridden) strategy."""
        self.model, forget_losses, retain_losses = self.strategy.unlearn(
            model=self.model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            **kwargs,
        )
        return forget_losses, retain_losses

    # ------------------------------------------------------------------
    # Convenience: evaluate with VLM-appropriate metrics
    # ------------------------------------------------------------------

    def evaluate_vlm(
        self,
        forget_data: DataLoader,
        retain_data: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate with VLM-specific metrics: zero-shot retrieval + MIA.

        Uses the metric classes shipped with Erasus.  Falls back to
        an empty dict if the metric modules cannot be imported (e.g.
        missing optional deps).
        """
        metrics_instances: List[Any] = []
        try:
            from erasus.metrics.retrieval import ZeroShotRetrievalMetric
            metrics_instances.append(ZeroShotRetrievalMetric())
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
