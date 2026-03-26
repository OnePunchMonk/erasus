"""
UnlearningTrainer — Orchestrates the unlearning training loop.

Handles epoch iteration, optimizer stepping, validation, early stopping,
best-checkpoint selection, and callbacks. Works with any ``UnlearningModule``.

Example
-------
>>> module = MyUnlearningModule(model, lr=1e-4)
>>> trainer = UnlearningTrainer(
...     max_epochs=10,
...     validate_every=1,
...     early_stopping_patience=3,
...     monitor="val_forget_loss",
...     monitor_mode="max",
... )
>>> result = trainer.fit(module, forget_loader, retain_loader)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.unlearning_module import UnlearningModule
from erasus.utils.callbacks import Callback, CallbackList
from erasus.utils.early_stopping import EarlyStopping


@dataclass
class TrainerResult:
    """Container for trainer results."""

    model: nn.Module
    elapsed_time: float = 0.0
    forget_loss_history: List[float] = field(default_factory=list)
    retain_loss_history: List[float] = field(default_factory=list)
    validation_history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_metrics: Dict[str, float] = field(default_factory=dict)
    stopped_early: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnlearningTrainer:
    """
    Orchestrates the unlearning training loop.

    Parameters
    ----------
    max_epochs : int
        Maximum number of unlearning epochs.
    validate_every : int
        Run validation every N epochs. 0 disables validation.
    early_stopping_patience : int
        Stop if the monitored metric doesn't improve for this many
        validation rounds. 0 disables early stopping.
    monitor : str
        Metric name to monitor for early stopping and best checkpoint.
    monitor_mode : str
        ``"min"`` if lower is better (e.g. retain loss),
        ``"max"`` if higher is better (e.g. forget loss).
    save_best : bool
        If True, track the best model state based on monitored metric.
    callbacks : list[Callback], optional
        Additional callbacks to invoke during training.
    device : str, optional
        Device to use. Defaults to CUDA if available.
    forget_weight : float
        Weight for forget loss in combined update (default 1.0).
    retain_weight : float
        Weight for retain loss in combined update (default 1.0).
    """

    def __init__(
        self,
        max_epochs: int = 10,
        validate_every: int = 1,
        early_stopping_patience: int = 0,
        monitor: str = "val_forget_loss",
        monitor_mode: str = "max",
        save_best: bool = True,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[str] = None,
        forget_weight: float = 1.0,
        retain_weight: float = 1.0,
    ) -> None:
        self.max_epochs = max_epochs
        self.validate_every = validate_every
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.save_best = save_best
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight

        self._callbacks = CallbackList(callbacks or [])
        self._early_stopping: Optional[EarlyStopping] = None
        if early_stopping_patience > 0:
            self._early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                mode=monitor_mode,
            )

    def fit(
        self,
        module: UnlearningModule,
        forget_data: DataLoader,
        retain_data: Optional[DataLoader] = None,
        val_forget_data: Optional[DataLoader] = None,
        val_retain_data: Optional[DataLoader] = None,
    ) -> TrainerResult:
        """
        Run the unlearning training loop.

        Parameters
        ----------
        module : UnlearningModule
            The module defining forget/retain step logic.
        forget_data : DataLoader
            DataLoader for the forget set.
        retain_data : DataLoader, optional
            DataLoader for the retain set.
        val_forget_data : DataLoader, optional
            Validation forget set. Falls back to ``forget_data``.
        val_retain_data : DataLoader, optional
            Validation retain set. Falls back to ``retain_data``.

        Returns
        -------
        TrainerResult
        """
        start_time = time.time()

        # Move to device
        module.to(self.device)
        optimizer = module.configure_optimizers()

        # Fallback validation sets
        val_forget = val_forget_data or forget_data
        val_retain = val_retain_data or retain_data

        # Best model tracking
        best_state: Optional[Dict[str, Any]] = None
        best_epoch = 0
        best_metrics: Dict[str, float] = {}

        forget_losses: List[float] = []
        retain_losses: List[float] = []
        val_history: List[Dict[str, float]] = []
        stopped_early = False

        module.on_unlearn_start()
        self._callbacks.on_train_start()

        for epoch in range(self.max_epochs):
            module.on_epoch_start(epoch)
            self._callbacks.on_epoch_start(epoch)

            # --- Training phase ---
            module.model.train()
            epoch_forget_loss = 0.0
            epoch_retain_loss = 0.0
            n_forget_batches = 0
            n_retain_batches = 0

            # Forget pass
            for batch_idx, batch in enumerate(forget_data):
                batch = _move_batch(batch, self.device)
                self._callbacks.on_batch_start(batch_idx)

                optimizer.zero_grad()
                result = module.forget_step(batch, batch_idx)
                loss = result["loss"] if isinstance(result, dict) else result
                scaled_loss = self.forget_weight * loss
                scaled_loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_forget_loss += loss_val
                n_forget_batches += 1
                self._callbacks.on_batch_end(batch_idx, loss_val)

            # Retain pass
            if retain_data is not None:
                for batch_idx, batch in enumerate(retain_data):
                    batch = _move_batch(batch, self.device)

                    optimizer.zero_grad()
                    result = module.retain_step(batch, batch_idx)
                    loss = result["loss"] if isinstance(result, dict) else result
                    scaled_loss = self.retain_weight * loss
                    scaled_loss.backward()
                    optimizer.step()

                    loss_val = loss.item()
                    epoch_retain_loss += loss_val
                    n_retain_batches += 1

            avg_forget = epoch_forget_loss / max(n_forget_batches, 1)
            avg_retain = epoch_retain_loss / max(n_retain_batches, 1)
            forget_losses.append(avg_forget)
            retain_losses.append(avg_retain)

            epoch_metrics = {"forget_loss": avg_forget, "retain_loss": avg_retain}

            # --- Validation phase ---
            if self.validate_every > 0 and (epoch + 1) % self.validate_every == 0:
                val_metrics = module.validation_step(
                    module.model, val_forget, val_retain
                )
                epoch_metrics.update(val_metrics)
                val_history.append(val_metrics)

                # Best model tracking
                if self.save_best and self.monitor in epoch_metrics:
                    monitored = epoch_metrics[self.monitor]
                    is_better = False
                    if not best_metrics:
                        is_better = True
                    elif self.monitor_mode == "max":
                        is_better = monitored > best_metrics.get(self.monitor, float("-inf"))
                    else:
                        is_better = monitored < best_metrics.get(self.monitor, float("inf"))

                    if is_better:
                        best_state = copy.deepcopy(module.model.state_dict())
                        best_epoch = epoch
                        best_metrics = dict(epoch_metrics)

                # Early stopping
                if self._early_stopping is not None and self.monitor in epoch_metrics:
                    if self._early_stopping(epoch, epoch_metrics[self.monitor]):
                        stopped_early = True

            module.on_epoch_end(epoch, epoch_metrics)
            self._callbacks.on_epoch_end(epoch, epoch_metrics)

            if stopped_early or self._callbacks.should_stop():
                break

        # Restore best model if tracked
        if self.save_best and best_state is not None:
            module.model.load_state_dict(best_state)

        module.on_unlearn_end()
        self._callbacks.on_train_end()

        return TrainerResult(
            model=module.model,
            elapsed_time=time.time() - start_time,
            forget_loss_history=forget_losses,
            retain_loss_history=retain_losses,
            validation_history=val_history,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            stopped_early=stopped_early,
        )


def _move_batch(batch: Any, device: str) -> Any:
    """Move a batch of tensors to the given device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        moved = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
        return type(batch)(moved) if isinstance(batch, tuple) else moved
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch
