"""
erasus.metrics.utility.downstream_tasks — Downstream task evaluation.

Evaluates model performance on specific downstream tasks to verify
that unlearning preserves task-relevant capabilities.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("downstream_tasks")
class DownstreamTaskMetric(BaseMetric):
    """
    Evaluate model on multiple downstream tasks.

    Parameters
    ----------
    tasks : dict[str, dict]
        Task definitions with ``loader`` and optional ``eval_fn``.
        Example::

            {
                "sentiment": {"loader": sst2_loader},
                "nli": {"loader": mnli_loader, "eval_fn": nli_accuracy},
            }
    """

    def __init__(self, tasks: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.tasks = tasks or {}

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Evaluate across all registered downstream tasks.

        Falls back to standard accuracy on forget/retain loaders if
        no tasks are configured.
        """
        device = next(model.parameters()).device
        model.eval()
        results: Dict[str, float] = {}

        # Evaluate configured tasks
        for task_name, task_cfg in self.tasks.items():
            loader = task_cfg.get("loader")
            eval_fn = task_cfg.get("eval_fn", self._default_accuracy)

            if loader is not None:
                score = eval_fn(model, loader, device)
                results[f"downstream_{task_name}"] = score

        # Always evaluate on standard loaders too
        results["forget_accuracy"] = self._default_accuracy(model, forget_loader, device)
        if retain_loader is not None:
            results["retain_accuracy"] = self._default_accuracy(model, retain_loader, device)

        # Compute aggregate downstream score
        task_scores = [v for k, v in results.items() if k.startswith("downstream_")]
        if task_scores:
            results["downstream_mean"] = float(np.mean(task_scores))

        return results

    @staticmethod
    def _default_accuracy(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> float:
        """Standard classification accuracy."""
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                else:
                    continue
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        return correct / max(total, 1)
