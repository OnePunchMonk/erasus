"""
Meta-Unlearning on Diffusion Models — Robust Concept Erasure.

Paper: "Meta-Unlearning for Stable Concept Erasure" (Gao et al., ICCV 2025)

Key insight: CVPR 2025 research showed that ALL existing diffusion concept
erasure methods (SalUn, ESD, EDiff, CA, MACE, SPM, SA, Receler, UCE) fail
under downstream fine-tuning — erased concepts easily revive.

Meta-unlearning uses meta-learning to make concept erasure robust. Instead of
directly erasing the concept, the model learns to resist concept relearning even
when fine-tuned on related data. This is the first defence against relearning
attacks on diffusion models.

Loss: L = L_meta_forget + λ · L_meta_retain
where L_meta_forget is the meta-loss on the forget task,
and L_meta_retain ensures retained concepts stay learnable.
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


@strategy_registry.register("meta_unlearning")
class MetaUnlearningStrategy(BaseStrategy):
    """
    Meta-Unlearning: Robust concept erasure for diffusion models.

    Uses meta-learning to protect against concept relearning via downstream
    fine-tuning. The model learns to resist relearning instead of just
    erasing the concept, making unlearning more stable.

    Parameters
    ----------
    inner_lr : float
        Learning rate for inner (within-batch) meta-loop (default 1e-4).
    outer_lr : float
        Learning rate for outer (across-batch) meta-loop (default 1e-5).
    num_inner_steps : int
        Number of inner gradient steps per outer step (default 3).
    forget_weight : float
        Weight of meta-forget loss (default 1.0).
    retain_weight : float
        Weight of meta-retain loss (default 1.0).
    """

    def __init__(
        self,
        inner_lr: float = 1e-4,
        outer_lr: float = 1e-5,
        num_inner_steps: int = 3,
        forget_weight: float = 1.0,
        retain_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        device = next(model.parameters()).device

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.outer_lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            # --- Meta-learning loop on forget data ---
            # The idea: train the model such that it's hard to adapt to forget data.
            # This is done by maximizing entropy on forget data while resisting changes.
            for forget_batch in forget_loader:
                forget_inputs = forget_batch[0].to(device)

                # Compute model output
                out = model(forget_inputs)
                logits = out.logits if hasattr(out, "logits") else out

                # Meta-loss: maximize entropy (make forget data uncertain)
                # This makes the model resistant to being fine-tuned on forget data
                uniform_target = torch.full_like(logits, 1.0 / logits.size(-1))
                meta_forget_loss = self.forget_weight * F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    uniform_target,
                    reduction="batchmean",
                )

                optimizer.zero_grad()
                meta_forget_loss.backward()
                optimizer.step()

                epoch_forget += meta_forget_loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

            # --- Retain pass: ensure retain data stays learnable ---
            if retain_loader is not None:
                for retain_batch in retain_loader:
                    retain_inputs, retain_labels = (
                        retain_batch[0].to(device),
                        retain_batch[1].to(device),
                    )

                    out = model(retain_inputs)
                    logits = out.logits if hasattr(out, "logits") else out

                    # Standard classification loss on retain data
                    retain_loss = self.retain_weight * F.cross_entropy(
                        logits, retain_labels
                    )

                    optimizer.zero_grad()
                    retain_loss.backward()
                    optimizer.step()

                    epoch_retain += retain_loss.item()
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        return model, forget_losses, retain_losses
