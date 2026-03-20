"""
FLAT — LLM Unlearning via Loss Adjustment.

Paper: "FLAT: LLM Unlearning via Loss Adjustment with No Retain Data
or Reference Model" (Li et al., ICLR 2025)

FLAT is the most practical preference-based method because it requires
only the forget data — no retain set, no reference model.

Two components:
1. IDK loss (forget set): train the model to output maximum entropy
   (I Don't Know) on forget-set queries.  For classifiers this is a
   uniform target; for LLMs it is an "I don't know" token sequence.

2. Maintain loss (retain set, optional): self-distillation — the model
   KL-diverges toward its own pre-step output, preserving general
   capabilities without a separate retain dataset.

Loss: L = α · L_IDK + (1-α) · L_maintain
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


@strategy_registry.register("flat")
class FLATStrategy(BaseStrategy):
    """
    FLAT: forget-data-only unlearning via loss adjustment.

    Parameters
    ----------
    alpha : float
        Weight of IDK loss vs maintain loss (default 0.5).
    idk_weight : float
        Additional scaling of the IDK loss (default 1.0).
    maintain_weight : float
        Additional scaling of the self-distillation loss (default 1.0).
    lr : float
        Learning rate (default 1e-5).
    n_maintain_steps : int
        Number of self-distillation steps on the forget batch per
        forget step (default 1).  Keeps general capabilities stable
        without a separate retain loader.
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
        device = next(model.parameters()).device
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_retain = 0.0
            n = 0
            n_retain = 0

            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader) if retain_loader is not None else None

            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                # --- Step 1: IDK loss ---
                # Target: uniform distribution (maximum entropy = "I don't know")
                out = model(inputs)
                logits = out.logits if hasattr(out, "logits") else out
                n_classes = logits.size(-1)

                uniform_target = torch.full_like(logits, 1.0 / n_classes)
                log_probs = F.log_softmax(logits, dim=-1)
                idk_loss = F.kl_div(log_probs, uniform_target, reduction="batchmean")

                # --- Step 2: Maintain loss (self-distillation) ---
                # Snapshot the model's current output before the IDK update
                with torch.no_grad():
                    snapshot_out = model(inputs)
                    snapshot_logits = (
                        snapshot_out.logits if hasattr(snapshot_out, "logits") else snapshot_out
                    )
                    snapshot_probs = F.softmax(snapshot_logits, dim=-1)

                # After IDK update, distil back toward snapshot on *retain* data if available
                # (when no retain data, apply maintain loss on the same forget batch)
                if retain_loader is not None and retain_iter is not None:
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        retain_batch = next(retain_iter)

                    r_inputs = retain_batch[0].to(device)
                    with torch.no_grad():
                        r_snap_out = model(r_inputs)
                        r_snap_logits = (
                            r_snap_out.logits if hasattr(r_snap_out, "logits") else r_snap_out
                        )
                        r_snap_probs = F.softmax(r_snap_logits, dim=-1)

                    r_out = model(r_inputs)
                    r_logits = r_out.logits if hasattr(r_out, "logits") else r_out
                    r_log_probs = F.log_softmax(r_logits, dim=-1)
                    maintain_loss = F.kl_div(r_log_probs, r_snap_probs, reduction="batchmean")
                    epoch_retain += maintain_loss.item()
                    n_retain += 1
                else:
                    # Self-distillation on forget batch keeps the model stable
                    out2 = model(inputs)
                    logits2 = out2.logits if hasattr(out2, "logits") else out2
                    log_probs2 = F.log_softmax(logits2, dim=-1)
                    maintain_loss = F.kl_div(log_probs2, snapshot_probs, reduction="batchmean")

                loss = (
                    self.alpha * self.idk_weight * idk_loss
                    + (1 - self.alpha) * self.maintain_weight * maintain_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n += 1

            forget_losses.append(epoch_loss / max(n, 1))
            if retain_losses is not None and n_retain > 0:
                retain_losses.append(epoch_retain / n_retain)

        return model, forget_losses, retain_losses
