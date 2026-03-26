"""
Base Unlearner — Abstract base class for all unlearning workflows.

An Unlearner orchestrates the full pipeline: coreset selection → unlearning → evaluation.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class UnlearningResult:
    """Container for unlearning results and statistics."""

    model: nn.Module
    elapsed_time: float = 0.0
    forget_loss_history: List[float] = field(default_factory=list)
    retain_loss_history: List[float] = field(default_factory=list)
    coreset_size: int = 0
    original_forget_size: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    @property
    def compression_ratio(self) -> float:
        if self.original_forget_size == 0:
            return 0.0
        return self.coreset_size / self.original_forget_size


class BaseUnlearner(ABC):
    """
    Abstract base class for all Erasus unlearning workflows.

    An unlearner orchestrates:
      1. (Optional) Coreset selection on the forget set
      2. Running the unlearning strategy
      3. (Optional) In-loop validation and early stopping
      4. (Optional) Post-hoc evaluation

    Subclasses should implement ``_build_strategy``, ``_build_selector``,
    and ``_run_unlearning`` for modality-specific logic.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: Optional[Any] = None,
        selector: Optional[Any] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.original_model = model
        self.model = copy.deepcopy(model).to(self.device)
        self.strategy = strategy
        self.selector = selector
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        forget_data: Union[DataLoader, Any],
        retain_data: Optional[DataLoader] = None,
        prune_ratio: float = 0.1,
        epochs: int = 5,
        coreset: Optional[Any] = None,
        validate_every: int = 0,
        validation_metrics: Optional[List[Any]] = None,
        early_stopping_patience: int = 0,
        early_stopping_monitor: str = "forget_loss",
        early_stopping_mode: str = "max",
        **kwargs: Any,
    ) -> UnlearningResult:
        """
        Run the full unlearning pipeline.

        Parameters
        ----------
        forget_data : DataLoader
            Data to forget.
        retain_data : DataLoader, optional
            Data to retain (used for utility preservation).
        prune_ratio : float
            Fraction of forget set to keep as coreset (0-1).
        epochs : int
            Number of unlearning epochs.
        coreset : Coreset, optional
            Pre-built Coreset object. If provided, ``selector`` and
            ``prune_ratio`` are ignored.
        validate_every : int
            Run validation metrics every N epochs. 0 disables (default).
        validation_metrics : list[BaseMetric], optional
            Metrics to compute during validation. If None, skips validation.
        early_stopping_patience : int
            Stop if ``early_stopping_monitor`` doesn't improve for this many
            validation rounds. 0 disables (default).
        early_stopping_monitor : str
            Metric name to monitor. Default ``"forget_loss"``.
        early_stopping_mode : str
            ``"min"`` or ``"max"``. Default ``"max"`` (higher forget loss = better).

        Returns
        -------
        UnlearningResult
            Contains the unlearned model and statistics.
        """
        from erasus.core.coreset import Coreset
        from erasus.utils.early_stopping import EarlyStopping

        start = time.time()

        # Step 1 — coreset selection (Coreset object takes precedence)
        if coreset is not None and isinstance(coreset, Coreset):
            forget_loader = coreset.to_loader(
                batch_size=forget_data.batch_size if isinstance(forget_data, DataLoader) else 32
            )
            coreset_size = len(coreset)
            original_size = coreset.metadata.original_size
        elif self.selector is not None:
            k = max(1, int(len(forget_data.dataset) * prune_ratio))
            coreset_indices = self.selector.select(
                model=self.model,
                data_loader=forget_data,
                k=k,
            )
            coreset_dataset = torch.utils.data.Subset(
                forget_data.dataset, coreset_indices
            )
            forget_loader = DataLoader(
                coreset_dataset,
                batch_size=forget_data.batch_size or 32,
                shuffle=True,
            )
            coreset_size = len(coreset_indices)
            original_size = len(forget_data.dataset)
        else:
            forget_loader = forget_data
            coreset_size = len(forget_data.dataset)
            original_size = len(forget_data.dataset)

        # Setup early stopping
        stopper = None
        if early_stopping_patience > 0 and validate_every > 0:
            stopper = EarlyStopping(
                patience=early_stopping_patience,
                mode=early_stopping_mode,
            )

        # Step 2 — run unlearning strategy (epoch-by-epoch if validation needed)
        validation_history: List[Dict[str, float]] = []
        best_epoch = 0
        stopped_early = False
        best_state = None

        if validate_every > 0 and validation_metrics:
            # Run epoch by epoch for in-loop validation
            all_forget_losses: List[float] = []
            all_retain_losses: List[float] = []

            for epoch in range(epochs):
                f_losses, r_losses = self._run_unlearning(
                    forget_loader=forget_loader,
                    retain_loader=retain_data,
                    epochs=1,
                    **kwargs,
                )
                all_forget_losses.extend(f_losses)
                all_retain_losses.extend(r_losses)

                # Validation
                if (epoch + 1) % validate_every == 0:
                    val_metrics: Dict[str, float] = {}
                    if f_losses:
                        val_metrics["forget_loss"] = f_losses[-1]
                    if r_losses:
                        val_metrics["retain_loss"] = r_losses[-1]

                    for metric in validation_metrics:
                        result = metric.compute(
                            model=self.model,
                            forget_data=forget_loader,
                            retain_data=retain_data,
                        )
                        if isinstance(result, dict):
                            val_metrics.update(result)
                        else:
                            val_metrics[metric.name] = result

                    validation_history.append(val_metrics)

                    # Track best
                    if early_stopping_monitor in val_metrics:
                        monitored = val_metrics[early_stopping_monitor]
                        is_better = False
                        if best_state is None:
                            is_better = True
                        elif early_stopping_mode == "max":
                            is_better = monitored > val_metrics.get(
                                "_best", float("-inf")
                            )
                        else:
                            is_better = monitored < val_metrics.get(
                                "_best", float("inf")
                            )
                        if is_better:
                            best_state = copy.deepcopy(self.model.state_dict())
                            best_epoch = epoch

                    # Early stopping check
                    if stopper is not None and early_stopping_monitor in val_metrics:
                        if stopper(epoch, val_metrics[early_stopping_monitor]):
                            stopped_early = True
                            break

            forget_losses = all_forget_losses
            retain_losses = all_retain_losses

            # Restore best model
            if best_state is not None:
                self.model.load_state_dict(best_state)
        else:
            # Original path: run all epochs at once
            forget_losses, retain_losses = self._run_unlearning(
                forget_loader=forget_loader,
                retain_loader=retain_data,
                epochs=epochs,
                **kwargs,
            )

        elapsed = time.time() - start

        return UnlearningResult(
            model=self.model,
            elapsed_time=elapsed,
            forget_loss_history=forget_losses,
            retain_loss_history=retain_losses,
            coreset_size=coreset_size,
            original_forget_size=original_size,
            validation_history=validation_history,
            best_epoch=best_epoch,
            stopped_early=stopped_early,
        )

    def evaluate(
        self,
        forget_data: DataLoader,
        retain_data: DataLoader,
        metrics: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """Evaluate the unlearned model using provided metrics."""
        results: Dict[str, float] = {}
        if metrics is None:
            return results
        for metric in metrics:
            result = metric.compute(
                model=self.model,
                forget_data=forget_data,
                retain_data=retain_data,
            )
            if isinstance(result, dict):
                results.update(result)
            else:
                results[metric.__class__.__name__] = result
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @abstractmethod
    def _run_unlearning(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int,
        **kwargs: Any,
    ) -> Tuple[List[float], List[float]]:
        """
        Execute the unlearning loop. Must be implemented by subclasses.

        Returns (forget_loss_history, retain_loss_history).
        """
        ...
