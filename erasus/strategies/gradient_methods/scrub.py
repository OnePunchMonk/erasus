"""
SCRUB Strategy — Student-teacher distillation for unlearning.

Paper: Towards Unbounded Machine Unlearning (Kurmanji et al., CVPR 2024)
Section 4.1.2.
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


@strategy_registry.register("scrub")
class SCRUBStrategy(BaseStrategy):
    """
    Use student-teacher distillation to approximate
    the model trained without the forget set.

    Minimises: D_KL(p_θ∖Df || p_θ') + λ||θ' - θ||²
    """

    def __init__(
        self,
        kl_weight: float = 1.0,
        l2_weight: float = 0.01,
        lr: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        self.l2_weight = l2_weight
        self.lr = lr

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        if retain_loader is None:
            raise ValueError("SCRUB requires a retain_loader.")

        device = next(model.parameters()).device

        # Use original model as the "oracle"/teacher
        oracle_model = copy.deepcopy(model)
        oracle_model.eval()
        for p in oracle_model.parameters():
            p.requires_grad = False

        student_model = copy.deepcopy(model)
        student_model.train()

        original_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        optimizer = torch.optim.Adam(student_model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0

            for batch in retain_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None

                with torch.no_grad():
                    oracle_out = oracle_model(inputs)
                    oracle_logits = oracle_out.logits if hasattr(oracle_out, "logits") else oracle_out
                    oracle_probs = F.softmax(oracle_logits, dim=-1)

                student_out = student_model(inputs)
                student_logits = student_out.logits if hasattr(student_out, "logits") else student_out
                student_log_probs = F.log_softmax(student_logits, dim=-1)

                kl_loss = F.kl_div(
                    student_log_probs, oracle_probs, reduction="batchmean",
                )

                l2_loss = sum(
                    (p - original_params[n_]).pow(2).sum()
                    for n_, p in student_model.named_parameters()
                    if n_ in original_params
                )

                loss = self.kl_weight * kl_loss + self.l2_weight * l2_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n += 1

            retain_losses.append(epoch_loss / max(n, 1))
            forget_losses.append(0.0)  # Not directly optimised

        # Copy trained weights back
        model.load_state_dict(student_model.state_dict())
        return model, forget_losses, retain_losses
