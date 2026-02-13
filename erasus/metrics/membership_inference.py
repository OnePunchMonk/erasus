"""
Membership Inference Metric — Measures how well an attacker can
distinguish between forget and non-forget data.

Ideal post-unlearning: MIA accuracy ≈ 50% (random guessing).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class MembershipInferenceMetric(BaseMetric):
    """
    Simple loss-threshold MIA: compare loss distributions on
    forget vs. retain to see if they're distinguishable.
    """

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        if forget_data is None or retain_data is None:
            return {"mia_accuracy": 0.0}

        model.eval()
        device = next(model.parameters()).device

        forget_losses = self._collect_losses(model, forget_data, device)
        retain_losses = self._collect_losses(model, retain_data, device)

        if not forget_losses or not retain_losses:
            return {"mia_accuracy": 0.5}

        # Compute AUC if sklearn is available
        try:
            from sklearn.metrics import roc_auc_score
            has_sklearn = True
        except ImportError:
            has_sklearn = False

        all_losses = forget_losses + retain_losses
        labels = [1] * len(forget_losses) + [0] * len(retain_losses)

        # Convention: Lower loss = Member (1). Higher loss = Non-member (0).
        # ROC requires score positively correlated with label 1.
        # So use -loss as score.
        scores = [-l for l in all_losses] # Negate loss so higher score = lower loss = member

        metric_results = {}
        
        if has_sklearn:
             if len(set(labels)) > 1:
                metric_results["mia_auc"] = roc_auc_score(labels, scores)
             else:
                metric_results["mia_auc"] = 0.5

        # Find optimal threshold accuracy via simple sweep
        best_acc = 0.5
        # Optimization: only check unique values if list is huge? or just iterate
        # For efficiency on large datasets, maybe sample or sort
        unique_thresholds = sorted(set(all_losses))
        if len(unique_thresholds) > 1000:
             # Subsample thresholds
             import numpy as np
             unique_thresholds = np.percentile(all_losses, np.linspace(0, 100, 100))

        for threshold in unique_thresholds:
            # Predict member (1) if loss < threshold => score > -threshold
            # Actually: loss <= threshold => likely member
            predictions = [1 if l <= threshold else 0 for l in all_losses]
            acc = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
            best_acc = max(best_acc, acc)

        metric_results["mia_accuracy"] = best_acc
        return metric_results

    @staticmethod
    def _collect_losses(model: nn.Module, loader: DataLoader, device) -> list:
        losses = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                per_sample = torch.nn.functional.cross_entropy(
                    logits, labels, reduction="none",
                )
                losses.extend(per_sample.cpu().tolist())
        return losses
