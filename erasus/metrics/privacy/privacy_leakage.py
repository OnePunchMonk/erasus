"""
erasus.metrics.privacy.privacy_leakage — MUSE-style privacy leakage score.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class PrivacyLeakageMetric(BaseMetric):
    """
    Privacy leakage score based on the forget/retain loss gap.

    Lower forget loss relative to retain loss indicates higher leakage.
    """

    name = "privacy_leakage"

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        forget_loss = self._average_loss(model, forget_data, device)
        retain_loss = self._average_loss(model, retain_data, device)

        leakage = max(0.0, 1.0 - (forget_loss / (retain_loss + 1e-8))) if retain_loss > 0 else 0.0
        return {
            "privacy_leakage": float(leakage),
            "privacy_forget_loss": float(forget_loss),
            "privacy_retain_loss": float(retain_loss),
        }

    @staticmethod
    def _average_loss(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(logits, labels, reduction="sum")
                total_loss += float(loss.item())
                total += labels.size(0)
        return total_loss / max(total, 1)
