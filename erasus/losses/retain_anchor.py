"""
Retain Anchor Loss — Constrains model drift on safe retain data.

Ensures the unlearned model stays close to the original on retain samples.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetainAnchorLoss(nn.Module):
    """
    L = α · ||f(x; θ') - f(x; θ_orig)||²

    Penalises deviation of the unlearned model's outputs from the
    original model's outputs on the retain set.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        current_outputs: torch.Tensor,
        original_outputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.alpha * F.mse_loss(current_outputs, original_outputs)
