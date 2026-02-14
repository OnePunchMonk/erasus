"""
erasus.metrics.forgetting.extraction_attack — Data extraction attack metric.

Measures vulnerability to membership and data extraction attacks
after unlearning, beyond standard MIA.
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


@metric_registry.register("extraction_attack")
class ExtractionAttackMetric(BaseMetric):
    """
    Measure data extraction vulnerability.

    Uses loss-based and likelihood-ratio thresholding to determine
    if an adversary can extract information about unlearned data.

    Lower extraction success = better unlearning.

    Parameters
    ----------
    n_shadow_models : int
        Number of shadow models for LiRA-style attacks (simplified).
    threshold_percentile : float
        Percentile threshold for membership classification.
    """

    def __init__(
        self,
        n_shadow_models: int = 1,
        threshold_percentile: float = 90.0,
    ) -> None:
        self.n_shadow_models = n_shadow_models
        self.threshold_percentile = threshold_percentile

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute extraction attack metrics.

        Parameters
        ----------
        model : nn.Module
            The unlearned model.
        forget_loader : DataLoader
            Data that was supposed to be forgotten.
        retain_loader : DataLoader, optional
            Data that should remain learned.
        """
        device = next(model.parameters()).device
        model.eval()

        forget_losses = self._per_sample_loss(model, forget_loader, device)
        results: Dict[str, float] = {
            "forget_mean_loss": float(np.mean(forget_losses)),
            "forget_loss_std": float(np.std(forget_losses)),
        }

        if retain_loader is not None:
            retain_losses = self._per_sample_loss(model, retain_loader, device)

            # Loss-threshold attack
            threshold = np.percentile(retain_losses, self.threshold_percentile)
            # Forget samples with loss below threshold are "extracted"
            extracted = (forget_losses < threshold).sum()
            extraction_rate = extracted / max(len(forget_losses), 1)

            results["retain_mean_loss"] = float(np.mean(retain_losses))
            results["loss_threshold"] = float(threshold)
            results["extraction_rate"] = float(extraction_rate)
            results["extraction_resistance"] = 1.0 - float(extraction_rate)

            # Likelihood ratio approach
            forget_z = (forget_losses - np.mean(retain_losses)) / max(np.std(retain_losses), 1e-8)
            retain_z = (retain_losses - np.mean(retain_losses)) / max(np.std(retain_losses), 1e-8)

            # AUC approximation using z-scores
            all_z = np.concatenate([forget_z, retain_z])
            all_labels = np.concatenate([np.zeros(len(forget_z)), np.ones(len(retain_z))])

            # Sort by z-score (ascending) and compute AUC
            sorted_idx = np.argsort(all_z)
            sorted_labels = all_labels[sorted_idx]
            n_pos = sorted_labels.sum()
            n_neg = len(sorted_labels) - n_pos
            if n_pos > 0 and n_neg > 0:
                tpr_sum = 0.0
                tp = 0
                for label in sorted_labels:
                    if label == 1:
                        tp += 1
                    else:
                        tpr_sum += tp / n_pos
                auc = tpr_sum / n_neg
            else:
                auc = 0.5

            results["likelihood_ratio_auc"] = float(auc)
            # Ideal AUC for unlearned data is 0.5
            results["privacy_score"] = 1.0 - abs(float(auc) - 0.5) * 2

        return results

    @staticmethod
    def _per_sample_loss(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> np.ndarray:
        """Compute per-sample cross-entropy loss."""
        losses: list = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                if labels is not None:
                    per_sample = F.cross_entropy(logits, labels, reduction="none")
                else:
                    per_sample = -logits.logsumexp(dim=-1)

                losses.extend(per_sample.cpu().numpy().tolist())

        return np.array(losses)
