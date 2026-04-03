"""
FLAT — LLM unlearning via loss adjustment.

Paper: "FLAT: LLM Unlearning via Loss Adjustment with No Retain Data"
(Li et al., ICLR 2025)

FLAT is designed to work with the forget set alone. This implementation
uses a forget-only loss adjustment that combines:

1. An IDK / high-entropy term that pulls output distributions toward
   uniformity on forget examples.
2. A direct forget term that pushes down the average log-probability of
   the forget labels/tokens.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry
from erasus.strategies.llm_specific.simnpo import (
    _forward_logits,
    _split_batch,
    _token_log_probs_and_mask,
)


@strategy_registry.register("flat")
class FLATStrategy(BaseStrategy):
    """
    Forget-only FLAT unlearning.

    Parameters
    ----------
    alpha : float
        Interpolation between the IDK term and the direct forget term.
    idk_weight : float
        Scale applied to the uniform-target IDK term.
    maintain_weight : float
        Scale applied to the direct forget term.
    lr : float
        Learning rate.
    n_maintain_steps : int
        Reserved for API compatibility; FLAT operates in a single
        forget-only optimisation step per batch in this implementation.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        idk_weight: float = 1.0,
        maintain_weight: float = 1.0,
        lr: float = 1e-5,
        n_maintain_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.idk_weight = idk_weight
        self.maintain_weight = maintain_weight
        self.lr = lr
        self.n_maintain_steps = n_maintain_steps

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        del retain_loader  # FLAT does not require retain data.

        device = next(model.parameters()).device
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in forget_loader:
                model_inputs, labels = _split_batch(batch, device)
                logits = _forward_logits(model, model_inputs)
                log_probs = F.log_softmax(logits, dim=-1)

                uniform_target = torch.full_like(log_probs, 1.0 / log_probs.size(-1))
                idk_loss = F.kl_div(log_probs, uniform_target, reduction="batchmean")

                token_log_probs, valid_mask = _token_log_probs_and_mask(logits, labels)
                token_log_probs = token_log_probs * valid_mask.to(token_log_probs.dtype)
                token_counts = valid_mask.sum(dim=-1).clamp_min(1).to(logits.dtype)
                forget_term = token_log_probs.sum(dim=-1) / token_counts

                loss = (
                    self.alpha * self.idk_weight * idk_loss
                    + (1.0 - self.alpha) * self.maintain_weight * forget_term.mean()
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            forget_losses.append(epoch_loss / max(n_batches, 1))

        return model, forget_losses, retain_losses
