"""
erasus.metrics.forgetting.mia — Full Membership Inference Attack metric.

Computes membership inference with ROC curves, AUC, and
TPR at various FPR thresholds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class MIAMetric(BaseMetric):
    """
    Membership Inference Attack metric with full ROC analysis.

    Uses per-sample loss as the membership signal:
    members (forget set) should have lower loss than non-members
    if the model has *not* successfully unlearned.

    After successful unlearning, AUC should be ≈ 0.5.
    """

    name = "mia_full"

    def __init__(self, num_thresholds: int = 200):
        self.num_thresholds = num_thresholds

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        member_losses = self._collect_losses(model, forget_data, device)
        nonmember_losses = self._collect_losses(model, retain_data, device)

        # Labels: 1 = member (forget), 0 = non-member (retain)
        labels = np.concatenate([
            np.ones(len(member_losses)),
            np.zeros(len(nonmember_losses)),
        ])
        # Scores: negative loss (higher = more likely member)
        scores = np.concatenate([-member_losses, -nonmember_losses])

        # Compute ROC
        fpr, tpr, thresholds = self._roc_curve(labels, scores)
        auc = self._auc(fpr, tpr)

        # TPR at specific FPR thresholds
        tpr_at_fpr_01 = self._tpr_at_fpr(fpr, tpr, target_fpr=0.01)
        tpr_at_fpr_05 = self._tpr_at_fpr(fpr, tpr, target_fpr=0.05)
        tpr_at_fpr_10 = self._tpr_at_fpr(fpr, tpr, target_fpr=0.10)

        # Optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_accuracy = ((scores >= thresholds[optimal_idx]).astype(int) == labels).mean()

        return {
            "mia_auc": float(auc),
            "mia_accuracy": float(optimal_accuracy),
            "mia_tpr_at_fpr_01": float(tpr_at_fpr_01),
            "mia_tpr_at_fpr_05": float(tpr_at_fpr_05),
            "mia_tpr_at_fpr_10": float(tpr_at_fpr_10),
            "mia_member_loss_mean": float(member_losses.mean()),
            "mia_nonmember_loss_mean": float(nonmember_losses.mean()),
        }

    @staticmethod
    def _collect_losses(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> np.ndarray:
        """Collect per-sample losses."""
        losses = []
        criterion = nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    continue

                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                batch_losses = criterion(outputs, targets)
                losses.append(batch_losses.cpu().numpy())

        return np.concatenate(losses) if losses else np.array([])

    def _roc_curve(
        self, labels: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ROC curve without sklearn dependency."""
        thresholds = np.linspace(scores.min(), scores.max(), self.num_thresholds)
        fpr_list, tpr_list = [], []

        positives = labels == 1
        negatives = labels == 0
        n_pos = positives.sum()
        n_neg = negatives.sum()

        for thresh in thresholds:
            predicted_pos = scores >= thresh
            tp = (predicted_pos & positives).sum()
            fp = (predicted_pos & negatives).sum()
            tpr_list.append(tp / max(n_pos, 1))
            fpr_list.append(fp / max(n_neg, 1))

        return np.array(fpr_list), np.array(tpr_list), thresholds

    @staticmethod
    def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
        """Trapezoidal AUC."""
        sorted_idx = np.argsort(fpr)
        fpr_sorted = fpr[sorted_idx]
        tpr_sorted = tpr[sorted_idx]
        return float(np.trapz(tpr_sorted, fpr_sorted))

    @staticmethod
    def _tpr_at_fpr(
        fpr: np.ndarray, tpr: np.ndarray, target_fpr: float
    ) -> float:
        """TPR at a given FPR threshold."""
        valid = fpr <= target_fpr
        if not valid.any():
            return 0.0
        return float(tpr[valid].max())
