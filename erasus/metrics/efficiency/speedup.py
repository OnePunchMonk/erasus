"""
erasus.metrics.efficiency.speedup — Speedup ratio vs. full retraining.

Computes the wall-clock speedup achieved by unlearning
compared to full model retraining from scratch.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("speedup")
class SpeedupMetric(BaseMetric):
    """
    Speedup of unlearning vs. retraining.

    Measures the ratio: retrain_time / unlearn_time.

    Parameters
    ----------
    retrain_epochs : int
        Number of epochs for simulated retraining.
    """

    def __init__(self, retrain_epochs: int = 1) -> None:
        self.retrain_epochs = retrain_epochs

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        unlearn_time: Optional[float] = None,
        retrain_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute speedup metrics.

        Parameters
        ----------
        unlearn_time : float, optional
            Pre-measured unlearning time in seconds.
        retrain_fn : callable, optional
            Function that performs retraining and returns elapsed time.
        """
        results: Dict[str, float] = {}

        # Estimate retraining time
        if retrain_fn is not None:
            retrain_time = retrain_fn()
        elif retain_loader is not None:
            retrain_time = self._estimate_retrain_time(model, retain_loader)
        else:
            retrain_time = 0.0

        results["estimated_retrain_time_s"] = retrain_time

        if unlearn_time is not None:
            results["unlearn_time_s"] = unlearn_time
            results["speedup_ratio"] = retrain_time / max(unlearn_time, 1e-6)
        else:
            # Estimate unlearning time from a single pass over forget data
            unlearn_est = self._estimate_forward_time(model, forget_loader)
            results["estimated_unlearn_time_s"] = unlearn_est
            results["speedup_ratio"] = retrain_time / max(unlearn_est, 1e-6)

        return results

    def _estimate_retrain_time(
        self,
        model: nn.Module,
        retain_loader: DataLoader,
    ) -> float:
        """Estimate retraining time by timing forward/backward passes."""
        device = next(model.parameters()).device
        model.train()

        start = time.perf_counter()
        batches_done = 0

        for batch in retain_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            batches_done += 1
            if batches_done >= 5:
                break

        elapsed = time.perf_counter() - start
        time_per_batch = elapsed / max(batches_done, 1)

        total_batches = len(retain_loader)
        return time_per_batch * total_batches * self.retrain_epochs

    def _estimate_forward_time(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> float:
        """Time a single pass over the loader."""
        device = next(model.parameters()).device
        model.eval()
        start = time.perf_counter()
        batches = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                model(inputs)
                batches += 1
                if batches >= 5:
                    break
        elapsed = time.perf_counter() - start
        time_per_batch = elapsed / max(batches, 1)
        return time_per_batch * len(loader)
