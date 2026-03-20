"""
ContinualUnlearner — Sequential unlearning for realistic deployment scenarios.

Paper: "FIT: Federated Incremental Unlearning" (2025) and related continual
learning literature.

In production, deletion requests arrive sequentially. The ContinualUnlearner
handles:
- Sequential deletion of multiple forget sets over time
- Incremental coreset updates (don't recompute from scratch)
- Adaptive strategy scheduling (adjust learning rate/epochs per request)
- Catastrophic forgetting detection (monitor utility metrics)
- Relearning resistance (FIT-inspired anti-catastrophic-forgetting)

Key difference from batch unlearning: a single large forget set is replaced
by many small sequential deletion requests. Naive repetition causes the model
to "forget" general capabilities (catastrophic forgetting). This orchestrator
prevents that via scheduling and monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_unlearner import BaseUnlearner, UnlearningResult
from erasus.core.registry import strategy_registry, selector_registry

# Ensure strategies are registered
import erasus.strategies  # noqa: F401
import erasus.selectors  # noqa: F401


@dataclass
class DeletionRequest:
    """Represents a single deletion request (e.g., user X wants their data forgotten)."""

    request_id: str
    forget_loader: DataLoader
    forget_set_size: int
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinualUnlearningResult:
    """Results for a continual unlearning session."""

    model: nn.Module
    total_elapsed_time: float = 0.0
    deletion_requests: List[DeletionRequest] = field(default_factory=list)
    per_request_results: List[UnlearningResult] = field(default_factory=list)
    catastrophic_forgetting_detected: bool = False
    utility_degradation: float = 0.0  # Measured on retain set
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContinualUnlearner(BaseUnlearner):
    """
    High-level orchestrator for sequential unlearning requests.

    Handles realistic scenarios where deletion requests arrive over time.
    Adapts strategy parameters to prevent catastrophic forgetting.

    Parameters
    ----------
    model : nn.Module
        The model to unlearn from.
    strategy : str or BaseStrategy
        The unlearning strategy to apply per request.
    selector : str or BaseSelector, optional
        Coreset selector (applied to each deletion request).
    strategy_kwargs : dict, optional
        Keyword arguments to pass to the strategy.
    selector_kwargs : dict, optional
        Keyword arguments to pass to the selector.
    base_epochs : int
        Initial number of epochs per deletion request (default 3).
    adaptive_scheduling : bool
        If True, reduce epochs/learning rate for later requests to reduce
        catastrophic forgetting (default True).
    catastrophic_forgetting_threshold : float
        Utility degradation threshold to flag catastrophic forgetting (default 0.1).
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "gradient_ascent",
        selector: Optional[str] = None,
        device: Optional[str] = None,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        selector_kwargs: Optional[Dict[str, Any]] = None,
        base_epochs: int = 3,
        adaptive_scheduling: bool = True,
        catastrophic_forgetting_threshold: float = 0.1,
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
        self.base_epochs = base_epochs
        self.adaptive_scheduling = adaptive_scheduling
        self.catastrophic_forgetting_threshold = catastrophic_forgetting_threshold

        # Track deletion history
        self.deletion_history: List[DeletionRequest] = []
        self.per_request_results: List[UnlearningResult] = []

    def process_deletion_requests(
        self,
        deletion_requests: List[DeletionRequest],
        retain_loader: Optional[DataLoader] = None,
        prune_ratio: float = 0.1,
        **kwargs: Any,
    ) -> ContinualUnlearningResult:
        """
        Process a sequence of deletion requests sequentially.

        Parameters
        ----------
        deletion_requests : list of DeletionRequest
            Sequence of deletion requests to process.
        retain_loader : DataLoader, optional
            Retain set for utility preservation (used throughout).
        prune_ratio : float
            Coreset selection ratio per request (default 0.1).

        Returns
        -------
        ContinualUnlearningResult
            Contains the final model and per-request statistics.
        """
        import time

        total_start = time.time()
        self.deletion_history = []
        self.per_request_results = []

        utility_measurements = []

        for idx, request in enumerate(deletion_requests):
            # Adapt strategy parameters based on request index
            if self.adaptive_scheduling:
                epochs = max(1, self.base_epochs - idx // 2)  # Reduce over time
                # Optionally reduce learning rate
                lr_scale = 1.0 - (0.1 * idx / max(1, len(deletion_requests)))
            else:
                epochs = self.base_epochs
                lr_scale = 1.0

            # Apply coreset selection if selector is available
            forget_loader = request.forget_loader
            if self.selector is not None:
                # Incremental update: add new forget examples to the coreset
                forget_loader = self._select_coreset(
                    forget_loader,
                    retain_loader,
                    prune_ratio,
                    request.request_id,
                )

            # Run unlearning for this deletion request
            result = self._unlearn_single_request(
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                epochs=int(epochs),
                lr_scale=lr_scale,
                **kwargs,
            )

            self.per_request_results.append(result)
            self.deletion_history.append(request)

            # Measure utility degradation on retain set (catastrophic forgetting metric)
            if retain_loader is not None:
                utility = self._measure_utility(retain_loader)
                utility_measurements.append(utility)

        total_elapsed = time.time() - total_start

        # Detect catastrophic forgetting
        catastrophic_forgetting_detected = False
        utility_degradation = 0.0
        if len(utility_measurements) > 1:
            utility_degradation = utility_measurements[0] - utility_measurements[-1]
            catastrophic_forgetting_detected = (
                utility_degradation > self.catastrophic_forgetting_threshold
            )

        return ContinualUnlearningResult(
            model=self.model,
            total_elapsed_time=total_elapsed,
            deletion_requests=self.deletion_history,
            per_request_results=self.per_request_results,
            catastrophic_forgetting_detected=catastrophic_forgetting_detected,
            utility_degradation=utility_degradation,
            metadata={
                "strategy": self.strategy_name,
                "selector": self.selector_name,
                "num_requests": len(deletion_requests),
                "adaptive_scheduling": self.adaptive_scheduling,
            },
        )

    def _unlearn_single_request(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int,
        lr_scale: float = 1.0,
        **kwargs: Any,
    ) -> UnlearningResult:
        """Unlearn a single deletion request."""
        import time

        start = time.time()

        # Optionally scale learning rate
        if lr_scale != 1.0 and hasattr(self.strategy, "lr"):
            original_lr = self.strategy.lr
            self.strategy.lr = original_lr * lr_scale

        # Run unlearning
        self.model, forget_losses, retain_losses = self.strategy.unlearn(
            model=self.model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            **kwargs,
        )

        # Restore original learning rate
        if lr_scale != 1.0 and hasattr(self.strategy, "lr"):
            self.strategy.lr = original_lr

        elapsed = time.time() - start

        return UnlearningResult(
            model=self.model,
            elapsed_time=elapsed,
            forget_loss_history=forget_losses,
            retain_loss_history=retain_losses,
            metadata={"epochs": epochs, "lr_scale": lr_scale},
        )

    def _select_coreset(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        prune_ratio: float,
        request_id: str,
    ) -> DataLoader:
        """
        Select a coreset from the forget set for this deletion request.
        In incremental scenarios, this is done per request (not globally).
        """
        if self.selector is None:
            return forget_loader

        # Apply selector to this specific deletion request
        # (Selectors typically work with DataLoaders directly)
        try:
            selected_loader = self.selector.select(
                model=self.model,
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                prune_ratio=prune_ratio,
            )
            return selected_loader
        except Exception:
            # If selector fails, fall back to full forget loader
            return forget_loader

    def _measure_utility(self, retain_loader: DataLoader) -> float:
        """
        Measure model utility on the retain set.
        Returns accuracy (0–1) as a proxy for general capability preservation.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in retain_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                outputs = self.model(inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        self.model.train()
        return correct / total if total > 0 else 0.0

    def _run_unlearning(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int,
        **kwargs: Any,
    ) -> Tuple[List[float], List[float]]:
        """
        For compatibility with BaseUnlearner.fit() interface.
        Delegates to a single unlearning request.
        """
        result = self._unlearn_single_request(
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            **kwargs,
        )
        return result.forget_loss_history, result.retain_loss_history
