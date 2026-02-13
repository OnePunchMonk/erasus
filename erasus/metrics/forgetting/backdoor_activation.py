"""
erasus.metrics.forgetting.backdoor_activation — Backdoor success rate metric.

Measures whether the model has successfully unlearned backdoor triggers
by testing the attack success rate post-unlearning.

Reference: Liu et al. (ICLR 2022)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("backdoor_activation")
class BackdoorActivation(BaseMetric):
    """
    Measure backdoor attack success rate.

    Given a triggered dataset (with backdoor patterns), measures
    the fraction of samples where the model still predicts the
    backdoor target label.

    Lower ASR (Attack Success Rate) = better unlearning.
    """

    def __init__(self, target_label: Optional[int] = None) -> None:
        self.target_label = target_label

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute backdoor activation metrics.

        ``forget_loader`` should contain triggered (backdoor) samples.
        """
        device = next(model.parameters()).device
        model.eval()

        total = 0
        trigger_success = 0
        clean_correct = 0

        with torch.no_grad():
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = logits.argmax(dim=-1)

                total += len(labels)

                if self.target_label is not None:
                    trigger_success += (preds == self.target_label).sum().item()
                else:
                    # Without target, check if predictions are uniform/disrupted
                    trigger_success += (preds == labels).sum().item()

        asr = trigger_success / max(total, 1)

        results: Dict[str, float] = {
            "attack_success_rate": asr,
            "backdoor_removed": 1.0 - asr,
            "n_triggered_samples": float(total),
        }

        # Also measure retain accuracy if available
        if retain_loader is not None:
            retain_total = 0
            retain_correct = 0
            with torch.no_grad():
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    preds = logits.argmax(dim=-1)
                    retain_total += len(labels)
                    retain_correct += (preds == labels).sum().item()

            results["retain_accuracy"] = retain_correct / max(retain_total, 1)

        return results
