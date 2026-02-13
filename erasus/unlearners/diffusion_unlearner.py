"""
Diffusion Unlearner — Orchestrator for Diffusion Model unlearning.

Handles Stable Diffusion models for concept erasure, artist style
removal, and NSFW content filtering.

Example::

    from erasus.unlearners import DiffusionUnlearner

    unlearner = DiffusionUnlearner(model=sd_wrapper.unet, strategy="concept_erasure")
    result = unlearner.fit(forget_loader, retain_loader)
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


class DiffusionUnlearner(BaseUnlearner):
    """
    Unlearner specialised for Diffusion Models (Stable Diffusion).

    Recommended strategies:
    - ``concept_erasure`` — ESD-style concept removal (default)
    - ``noise_injection`` — latent noise perturbation
    - ``unet_surgery`` — targeted U-Net layer editing

    Selector is optional for diffusion models (frequently the full
    concept prompt set is small enough that coreset selection is not
    strictly necessary).  Set ``selector=None`` to skip.
    """

    DEFAULT_STRATEGY = "concept_erasure"
    DEFAULT_SELECTOR = None  # Prompt sets are usually small
    DEFAULT_EPOCHS = 100

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
        # Resolve strategy
        strategy_cls = strategy_registry.get(strategy)
        strategy_instance = strategy_cls(**(strategy_kwargs or {}))

        # Resolve selector (usually None for diffusion)
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

    def _run_unlearning(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int = DEFAULT_EPOCHS,
        **kwargs: Any,
    ) -> Tuple[List[float], List[float]]:
        """Delegate to the resolved diffusion strategy."""
        self.model, forget_losses, retain_losses = self.strategy.unlearn(
            model=self.model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            **kwargs,
        )
        return forget_losses, retain_losses

    # ------------------------------------------------------------------
    # Diffusion-specific evaluation
    # ------------------------------------------------------------------

    def evaluate_diffusion(
        self,
        forget_data: DataLoader,
        retain_data: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate with diffusion-specific metrics: FID score.

        FID should remain LOW on retain-set prompts (generation quality
        preserved) and ideally HIGH on forget-concept prompts.
        """
        metrics_instances: List[Any] = []
        try:
            from erasus.metrics.fid import FIDMetric
            metrics_instances.append(FIDMetric())
        except Exception:
            pass

        return self.evaluate(
            forget_data=forget_data,
            retain_data=retain_data,
            metrics=metrics_instances,
        )
