"""KL Divergence Loss for distillation-based unlearning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    """KL divergence between student and teacher distributions."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)
