"""
erasus.strategies.data_methods.knowledge_distillation — Knowledge distillation unlearning.

Uses a teacher-student paradigm: the original model is the teacher,
and a student model is trained to match the teacher on the retain set
while diverging on the forget set.

Reference: Hinton et al. (2015) — "Distilling the Knowledge in a Neural Network",
           adapted for machine unlearning.
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("knowledge_distillation")
class KnowledgeDistillationStrategy(BaseStrategy):
    """
    Knowledge distillation-based unlearning.

    Creates a copy of the model as teacher. The student (original model)
    is trained to:
    - **Match** the teacher's soft labels on retain data (KD loss)
    - **Diverge** from the teacher on forget data (reverse KD)

    Parameters
    ----------
    lr : float
        Learning rate for the student.
    temperature : float
        Softmax temperature for distillation.
    alpha : float
        Weight of distillation loss vs. hard-label loss.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        temperature: float = 4.0,
        alpha: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.temperature = temperature
        self.alpha = alpha

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """KL divergence between softened distributions."""
        s = F.log_softmax(student_logits / temperature, dim=-1)
        t = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run distillation-based unlearning."""
        device = next(model.parameters()).device

        # Freeze a teacher copy
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            model.train()

            # ---- Forget phase: diverge from teacher ----
            epoch_forget = 0.0
            n_f = 0
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                with torch.no_grad():
                    teacher_out = teacher(inputs)
                    t_logits = teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out

                student_out = model(inputs)
                s_logits = student_out.logits if hasattr(student_out, "logits") else student_out

                # Reverse distillation: maximise KD loss on forget data
                kd_loss = self._distillation_loss(s_logits, t_logits, self.temperature)
                hard_loss = F.cross_entropy(s_logits, labels)

                loss = -self.alpha * kd_loss + (1 - self.alpha) * (-hard_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_forget += hard_loss.item()
                n_f += 1

            forget_losses.append(epoch_forget / max(n_f, 1))

            # ---- Retain phase: match teacher ----
            if retain_loader is not None:
                epoch_retain = 0.0
                n_r = 0
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)

                    with torch.no_grad():
                        teacher_out = teacher(inputs)
                        t_logits = teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out

                    student_out = model(inputs)
                    s_logits = student_out.logits if hasattr(student_out, "logits") else student_out

                    kd_loss = self._distillation_loss(s_logits, t_logits, self.temperature)
                    hard_loss = F.cross_entropy(s_logits, labels)

                    loss = self.alpha * kd_loss + (1 - self.alpha) * hard_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_retain += hard_loss.item()
                    n_r += 1

                retain_losses.append(epoch_retain / max(n_r, 1))

        return model, forget_losses, retain_losses
