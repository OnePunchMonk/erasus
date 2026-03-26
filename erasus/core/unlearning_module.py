"""
UnlearningModule — User-subclassable module with forget/retain step hooks.

Inspired by PyTorch Lightning's LightningModule, this class provides a
clean separation between *what* happens during unlearning (the module)
and *how* it is orchestrated (the trainer).

Example
-------
>>> class MyModule(UnlearningModule):
...     def __init__(self, model, lr=1e-3):
...         super().__init__(model)
...         self.lr = lr
...
...     def forget_step(self, batch, batch_idx):
...         x, y = batch
...         logits = self.model(x)
...         return -F.cross_entropy(logits, y)  # gradient ascent
...
...     def retain_step(self, batch, batch_idx):
...         x, y = batch
...         logits = self.model(x)
...         return F.cross_entropy(logits, y)
...
...     def configure_optimizers(self):
...         return torch.optim.Adam(self.model.parameters(), lr=self.lr)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class UnlearningModule(ABC):
    """
    Abstract base for user-defined unlearning logic.

    Users subclass this and override ``forget_step``, ``retain_step``,
    and ``configure_optimizers`` to define custom unlearning behavior.
    The ``UnlearningTrainer`` handles the training loop orchestration.

    Parameters
    ----------
    model : nn.Module
        The model to unlearn from.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._device: torch.device = torch.device("cpu")

    @abstractmethod
    def forget_step(
        self, batch: Any, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the forget loss for one batch.

        Parameters
        ----------
        batch : Any
            A single batch from the forget DataLoader.
        batch_idx : int
            Index of the batch within the epoch.

        Returns
        -------
        torch.Tensor or dict
            The forget loss. If a dict, must contain a ``"loss"`` key.
        """
        ...

    @abstractmethod
    def retain_step(
        self, batch: Any, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the retain loss for one batch.

        Parameters
        ----------
        batch : Any
            A single batch from the retain DataLoader.
        batch_idx : int
            Index of the batch within the epoch.

        Returns
        -------
        torch.Tensor or dict
            The retain loss. If a dict, must contain a ``"loss"`` key.
        """
        ...

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Return the optimizer for unlearning.

        Override this to use a custom optimizer or learning rate.
        Default: Adam with lr=1e-3.
        """
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def validation_step(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Optional per-epoch validation. Override to compute custom metrics.

        Called by the trainer at the end of each epoch when
        ``validate_every > 0``. The default implementation computes
        mean forget and retain losses.

        Returns
        -------
        dict
            Metric name -> value mapping.
        """
        model.eval()
        metrics: Dict[str, float] = {}

        with torch.no_grad():
            # Forget loss
            total, count = 0.0, 0
            for batch in forget_data:
                inputs = batch[0].to(self._device)
                labels = batch[1].to(self._device) if len(batch) > 1 else None
                logits = model(inputs)
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                else:
                    loss = logits.mean()
                total += loss.item() * inputs.size(0)
                count += inputs.size(0)
            if count > 0:
                metrics["val_forget_loss"] = total / count

            # Retain loss
            if retain_data is not None:
                total, count = 0.0, 0
                for batch in retain_data:
                    inputs = batch[0].to(self._device)
                    labels = batch[1].to(self._device) if len(batch) > 1 else None
                    logits = model(inputs)
                    if labels is not None:
                        loss = torch.nn.functional.cross_entropy(logits, labels)
                    else:
                        loss = logits.mean()
                    total += loss.item() * inputs.size(0)
                    count += inputs.size(0)
                if count > 0:
                    metrics["val_retain_loss"] = total / count

        model.train()
        return metrics

    def on_unlearn_start(self) -> None:
        """Hook called before the unlearning loop begins."""
        pass

    def on_unlearn_end(self) -> None:
        """Hook called after the unlearning loop finishes."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Hook called at the end of each epoch with computed metrics."""
        pass

    def to(self, device: Union[str, torch.device]) -> "UnlearningModule":
        """Move the module's model to the given device."""
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device
