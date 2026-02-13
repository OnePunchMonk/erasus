"""
LLM Unlearner — Orchestrator for Large Language Model unlearning.

Handles LLaMA, GPT-2, Mistral, and similar causal LMs with
LLM-aware defaults including SSD, token masking, and causal tracing.

Example::

    from erasus.unlearners import LLMUnlearner

    unlearner = LLMUnlearner(model=llama_model, strategy="ssd")
    result = unlearner.fit(forget_loader, retain_loader)
    metrics = unlearner.evaluate_llm(forget_loader, retain_loader)
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


class LLMUnlearner(BaseUnlearner):
    """
    Unlearner specialised for Large Language Models.

    Recommended strategies (choose based on use-case):
    - ``gradient_ascent`` — simple baseline, fast
    - ``ssd`` — Selective Synaptic Dampening (knowledge removal)
    - ``token_masking`` — token-level forgetting
    - ``causal_tracing`` — layer-level surgical editing
    - ``scrub`` — student-teacher distillation

    Default selector: ``gradient_norm`` (fast, no Hessian needed).
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

    # ------------------------------------------------------------------

    def _run_unlearning(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int = DEFAULT_EPOCHS,
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

    # ------------------------------------------------------------------
    # LLM-specific evaluation
    # ------------------------------------------------------------------

    def evaluate_llm(
        self,
        forget_data: DataLoader,
        retain_data: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate with LLM-specific metrics: perplexity + MIA.

        Perplexity on retain should stay LOW, MIA accuracy should be ≈ 0.5.
        """
        metrics_instances: List[Any] = []
        try:
            from erasus.metrics.perplexity import PerplexityMetric
            metrics_instances.append(PerplexityMetric())
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
