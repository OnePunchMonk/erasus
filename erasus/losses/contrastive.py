"""
Contrastive Loss — Symmetric CLIP-style contrastive loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Symmetric image-text contrastive loss used in CLIP."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits = image_features @ text_features.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
