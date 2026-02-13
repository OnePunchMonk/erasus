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
      3. (Optional) Post-hoc evaluation

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
        forget_data: DataLoader,
        retain_data: Optional[DataLoader] = None,
        prune_ratio: float = 0.1,
        epochs: int = 5,
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
            Fraction of forget set to keep as coreset (0–1).
        epochs : int
            Number of unlearning epochs.

        Returns
        -------
        UnlearningResult
            Contains the unlearned model and statistics.
        """
        start = time.time()

        # Step 1 — coreset selection
        if self.selector is not None:
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
        else:
            forget_loader = forget_data
            coreset_size = len(forget_data.dataset)

        # Step 2 — run unlearning strategy
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
            original_forget_size=len(forget_data.dataset),
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
