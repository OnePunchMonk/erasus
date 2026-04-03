"""
erasus.metrics.privacy.rag_leakage — Detect memorization of external RAG contexts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("rag_leakage")
class RAGLeakageMetric(BaseMetric):
    """
    Measure whether external retrieval contexts appear memorized by the model.

    The metric compares the model's average token/class loss on a set of
    suspect RAG-backed forget samples against a retain baseline. Lower loss
    on the external-context samples indicates stronger internalization.
    """

    name = "rag_leakage"

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        if forget_data is None:
            return {"rag_leakage": 0.0}

        device = next(model.parameters()).device
        model.eval()

        rag_loss = self._average_loss(model, forget_data, device)
        retain_loss = self._average_loss(model, retain_data, device) if retain_data is not None else 0.0

        leakage = 0.0
        if retain_data is not None and retain_loss > 0:
            leakage = max(0.0, 1.0 - (rag_loss / (retain_loss + 1e-8)))

        return {
            "rag_leakage": float(leakage),
            "rag_context_loss": float(rag_loss),
            "rag_retain_loss": float(retain_loss),
        }

    @staticmethod
    def _average_loss(
        model: nn.Module,
        loader: Optional[DataLoader],
        device: torch.device,
    ) -> float:
        if loader is None:
            return 0.0

        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                if logits.dim() == 2 and labels.dim() == 1:
                    loss = F.cross_entropy(logits, labels, reduction="sum")
                    total += labels.size(0)
                elif logits.dim() >= 3 and labels.dim() == logits.dim() - 1:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    total += labels.ne(-100).sum().item()
                else:
                    loss = torch.tensor(0.0, device=device)

                total_loss += float(loss.item())

        return total_loss / max(total, 1)
