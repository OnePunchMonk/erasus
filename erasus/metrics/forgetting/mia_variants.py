"""
erasus.metrics.forgetting.mia_variants — Advanced MIA variants.

Includes:
- LiRA (Likelihood Ratio Attack) — Carlini et al., 2022
- Label-Only MIA — Choquette-Choo et al., 2021
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class LiRAMetric(BaseMetric):
    """
    Likelihood Ratio Attack (LiRA).

    Computes membership scores by comparing the target model's
    confidence against a population of shadow models.

    Since training shadow models is expensive, this implementation
    supports pre-computed shadow confidences.

    Parameters
    ----------
    shadow_in_confidences : np.ndarray, optional
        Shape ``(n_shadows, n_samples)`` — confidences from shadow
        models that included the sample in training.
    shadow_out_confidences : np.ndarray, optional
        Shape ``(n_shadows, n_samples)`` — confidences from shadow
        models that excluded the sample.
    """

    name = "lira"

    def __init__(
        self,
        shadow_in_confidences: Optional[np.ndarray] = None,
        shadow_out_confidences: Optional[np.ndarray] = None,
    ):
        self.shadow_in = shadow_in_confidences
        self.shadow_out = shadow_out_confidences

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        # Collect target model confidences on forget set
        target_confs = self._collect_confidences(model, forget_data, device)

        if self.shadow_in is not None and self.shadow_out is not None:
            # Full LiRA: compare against shadow distributions
            scores = self._compute_lira_scores(target_confs)
        else:
            # Simplified LiRA: use confidence magnitude as proxy
            scores = target_confs

        # Higher score → more likely a member
        mean_score = float(np.mean(scores))

        # Compute simple AUC against retain set
        retain_confs = self._collect_confidences(model, retain_data, device)
        labels = np.concatenate([np.ones(len(scores)), np.zeros(len(retain_confs))])
        all_scores = np.concatenate([scores, retain_confs])
        auc = self._simple_auc(labels, all_scores)

        return {
            "lira_mean_score": mean_score,
            "lira_auc": float(auc),
        }

    def _compute_lira_scores(self, target_confs: np.ndarray) -> np.ndarray:
        """Compute LiRA likelihood ratio scores."""
        n_samples = min(len(target_confs), self.shadow_in.shape[1])
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Fit Gaussians to in/out shadow confidences
            in_mean = self.shadow_in[:, i].mean()
            in_std = max(self.shadow_in[:, i].std(), 1e-10)
            out_mean = self.shadow_out[:, i].mean()
            out_std = max(self.shadow_out[:, i].std(), 1e-10)

            # Log-likelihood ratio
            log_p_in = -0.5 * ((target_confs[i] - in_mean) / in_std) ** 2
            log_p_out = -0.5 * ((target_confs[i] - out_mean) / out_std) ** 2
            scores[i] = log_p_in - log_p_out

        return scores

    @staticmethod
    def _collect_confidences(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> np.ndarray:
        """Collect max-softmax confidences per sample."""
        confs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)

                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                probs = torch.softmax(outputs, dim=-1)
                max_probs = probs.max(dim=-1).values
                confs.append(max_probs.cpu().numpy())

        return np.concatenate(confs) if confs else np.array([])

    @staticmethod
    def _simple_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        """Quick AUC via sorting."""
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp = 0.0
        auc = 0.0
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                auc += tp
        return auc / (n_pos * n_neg)


class LabelOnlyMIAMetric(BaseMetric):
    """
    Label-Only Membership Inference Attack.

    Uses only predicted labels (not confidence scores) to determine
    membership. Counts misclassification rates: forget samples should
    be misclassified after successful unlearning.
    """

    name = "label_only_mia"

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        forget_acc = self._compute_accuracy(model, forget_data, device)
        retain_acc = self._compute_accuracy(model, retain_data, device)

        # After good unlearning: forget_acc should be LOW, retain_acc HIGH
        # MIA signal: accuracy gap
        gap = retain_acc - forget_acc

        return {
            "label_mia_forget_accuracy": float(forget_acc),
            "label_mia_retain_accuracy": float(retain_acc),
            "label_mia_accuracy_gap": float(gap),
        }

    @staticmethod
    def _compute_accuracy(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    continue
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1)
