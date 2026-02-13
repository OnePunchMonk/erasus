"""
erasus.metrics.efficiency.flops — FLOPs estimation for unlearning.

Estimates the number of floating-point operations required
for unlearning vs. retraining.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("flops")
class FLOPsMetric(BaseMetric):
    """
    Estimate FLOPs for unlearning.

    Uses a parameter-based estimate: each parameter contributes
    approximately 2 FLOPs per sample per forward pass, and 4 per
    backward pass.
    """

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        unlearn_epochs: int = 5,
        retrain_epochs: int = 50,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Estimate FLOPs."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        n_forget = len(forget_loader.dataset) if hasattr(forget_loader, "dataset") else 0
        n_retain = 0
        if retain_loader is not None and hasattr(retain_loader, "dataset"):
            n_retain = len(retain_loader.dataset)

        # FLOPs per sample: ~6 * params (forward=2, backward=4)
        flops_per_sample = 6 * trainable_params

        unlearn_flops = flops_per_sample * n_forget * unlearn_epochs
        retrain_flops = flops_per_sample * (n_forget + n_retain) * retrain_epochs

        return {
            "total_params": float(total_params),
            "trainable_params": float(trainable_params),
            "estimated_unlearn_flops": float(unlearn_flops),
            "estimated_retrain_flops": float(retrain_flops),
            "flops_ratio": float(unlearn_flops) / max(float(retrain_flops), 1.0),
            "n_forget_samples": float(n_forget),
            "n_retain_samples": float(n_retain),
        }
