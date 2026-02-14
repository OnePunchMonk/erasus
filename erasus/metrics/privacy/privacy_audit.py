"""
erasus.metrics.privacy.privacy_audit — Privacy auditing framework.

Comprehensive privacy audit that runs multiple attacks and computes
aggregate privacy scores for an unlearned model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("privacy_audit")
class PrivacyAuditMetric(BaseMetric):
    """
    Comprehensive privacy auditing for unlearned models.

    Runs multiple privacy tests and produces an aggregate score:

    1. **Loss-based MIA** — threshold attack on per-sample loss.
    2. **Confidence-based MIA** — threshold on max prediction confidence.
    3. **Calibrated attack** — z-score based membership inference.
    4. **Entropy test** — predictive entropy comparison.

    Parameters
    ----------
    n_thresholds : int
        Number of sweep thresholds for ROC-style evaluation.
    """

    def __init__(self, n_thresholds: int = 100) -> None:
        self.n_thresholds = n_thresholds

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Run full privacy audit.

        Requires both forget_loader (should behave like non-member)
        and retain_loader (member data).
        """
        device = next(model.parameters()).device
        model.eval()

        results: Dict[str, float] = {}

        # Extract signals
        forget_signals = self._extract_signals(model, forget_loader, device)
        results["n_forget_samples"] = float(len(forget_signals["loss"]))

        if retain_loader is None:
            results["audit_status"] = 0.0  # Incomplete
            return results

        retain_signals = self._extract_signals(model, retain_loader, device)
        results["n_retain_samples"] = float(len(retain_signals["loss"]))

        # 1. Loss-based MIA
        loss_auc = self._compute_auc(
            member_scores=retain_signals["loss"],
            non_member_scores=forget_signals["loss"],
            lower_is_member=True,
        )
        results["loss_mia_auc"] = loss_auc

        # 2. Confidence-based MIA
        conf_auc = self._compute_auc(
            member_scores=retain_signals["confidence"],
            non_member_scores=forget_signals["confidence"],
            lower_is_member=False,
        )
        results["confidence_mia_auc"] = conf_auc

        # 3. Entropy-based test
        entropy_auc = self._compute_auc(
            member_scores=retain_signals["entropy"],
            non_member_scores=forget_signals["entropy"],
            lower_is_member=True,
        )
        results["entropy_mia_auc"] = entropy_auc

        # 4. Calibrated z-score attack
        z_auc = self._calibrated_attack(
            forget_signals["loss"], retain_signals["loss"]
        )
        results["calibrated_mia_auc"] = z_auc

        # Aggregate privacy score: ideal AUC = 0.5
        aucs = [loss_auc, conf_auc, entropy_auc, z_auc]
        mean_auc = np.mean(aucs)
        results["mean_mia_auc"] = float(mean_auc)
        # Score: 1.0 if all AUCs are 0.5, 0.0 if any is 1.0
        results["privacy_score"] = float(1.0 - 2 * abs(mean_auc - 0.5))
        results["audit_passed"] = float(all(0.4 <= a <= 0.6 for a in aucs))

        return results

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def _extract_signals(
        self, model: nn.Module, loader: DataLoader, device: torch.device
    ) -> Dict[str, np.ndarray]:
        """Extract per-sample loss, confidence, and entropy."""
        losses, confidences, entropies = [], [], []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = F.softmax(logits, dim=-1)

                # Per-sample loss
                if labels is not None:
                    loss = F.cross_entropy(logits, labels, reduction="none")
                else:
                    loss = -logits.logsumexp(dim=-1)
                losses.extend(loss.cpu().numpy().tolist())

                # Confidence (max prob)
                conf = probs.max(dim=-1).values
                confidences.extend(conf.cpu().numpy().tolist())

                # Entropy
                ent = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                entropies.extend(ent.cpu().numpy().tolist())

        return {
            "loss": np.array(losses),
            "confidence": np.array(confidences),
            "entropy": np.array(entropies),
        }

    # ------------------------------------------------------------------
    # AUC computation
    # ------------------------------------------------------------------

    def _compute_auc(
        self,
        member_scores: np.ndarray,
        non_member_scores: np.ndarray,
        lower_is_member: bool = True,
    ) -> float:
        """Compute AUC for a threshold-based membership attack."""
        if lower_is_member:
            # Members tend to have lower loss / entropy
            all_scores = np.concatenate([-member_scores, -non_member_scores])
        else:
            # Members tend to have higher confidence
            all_scores = np.concatenate([member_scores, non_member_scores])

        all_labels = np.concatenate([
            np.ones(len(member_scores)),
            np.zeros(len(non_member_scores)),
        ])

        # Compute AUC via sorting
        sorted_idx = np.argsort(all_scores)[::-1]
        sorted_labels = all_labels[sorted_idx]

        n_pos = sorted_labels.sum()
        n_neg = len(sorted_labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp, fp = 0.0, 0.0
        auc = 0.0
        prev_fp = 0.0

        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += tp

        auc /= (n_pos * n_neg)
        return float(auc)

    @staticmethod
    def _calibrated_attack(
        forget_losses: np.ndarray, retain_losses: np.ndarray
    ) -> float:
        """Calibrated attack using z-scores of the forget set."""
        mu = retain_losses.mean()
        sigma = retain_losses.std() + 1e-8

        forget_z = (forget_losses - mu) / sigma
        retain_z = (retain_losses - mu) / sigma

        # Members should have lower z-scores
        all_z = np.concatenate([-retain_z, -forget_z])
        all_labels = np.concatenate([
            np.ones(len(retain_z)),
            np.zeros(len(forget_z)),
        ])

        sorted_idx = np.argsort(all_z)[::-1]
        sorted_labels = all_labels[sorted_idx]

        n_pos = sorted_labels.sum()
        n_neg = len(sorted_labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp = 0.0
        auc = 0.0
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                auc += tp

        return float(auc / (n_pos * n_neg))
