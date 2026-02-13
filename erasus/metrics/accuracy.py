"""
Accuracy Metric — Retain/Forget accuracy for classification models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    """
    Computes classification accuracy on forget and retain sets.

    Goal: retain accuracy should stay HIGH, forget accuracy should go LOW.
    """

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        model.eval()
        device = next(model.parameters()).device
        results: Dict[str, float] = {}

        if forget_data is not None:
            results["forget_accuracy"] = self._accuracy(model, forget_data, device)
        if retain_data is not None:
            results["retain_accuracy"] = self._accuracy(model, retain_data, device)

        return results

    @staticmethod
    def _accuracy(model: nn.Module, loader: DataLoader, device) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / max(total, 1)
