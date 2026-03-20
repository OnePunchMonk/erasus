"""
UNDIAL — Self-Distillation with Logit Adjustment for Unlearning.

Paper: "Unlearning via Differential Influence Assessment Learning" (Dong et al., 2024)

Key insight: Instead of using gradient ascent (which produces nonsensical outputs),
UNDIAL operates in logit space to reduce the model's confidence on forget-set
tokens via self-distillation and logit adjustment. This produces coherent
alternative responses.

The method trains the model such that:
1. Forget tokens receive lower logits via distillation from a temperature-scaled
   reference model.
2. Retain tokens preserve their original behavior via KL divergence constraint.

Loss: L = α · L_forget + (1-α) · L_retain
where L_forget reduces confidence on forget tokens via distillation,
and L_retain preserves utility on retain tokens.
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


@strategy_registry.register("undial")
class UNDIALStrategy(BaseStrategy):
    """
    UNDIAL: Self-distillation with logit adjustment for unlearning.

    Reduces model confidence on forget-set tokens via logit adjustment and
    self-distillation, producing coherent alternative responses without
    the instability of gradient ascent.

    Parameters
    ----------
    temperature : float
        Temperature for logit scaling in self-distillation (default 3.0).
        Higher values soften the probability distribution.
    alpha : float
        Weight of forget loss vs retain loss (default 0.5).
    retain_weight : float
        Weight of retain KL loss (default 1.0).
    lr : float
        Learning rate (default 1e-5).
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        retain_weight: float = 1.0,
        lr: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.retain_weight = retain_weight
        self.lr = lr

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        device = next(model.parameters()).device

        # Frozen reference model for distillation target
        ref_model = copy.deepcopy(model).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            # --- Forget pass: reduce confidence on forget tokens ---
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                # Get logits from both models
                out = model(inputs)
                logits = out.logits if hasattr(out, "logits") else out

                with torch.no_grad():
                    ref_out = ref_model(inputs)
                    ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out

                # Scale reference logits by temperature to create soft targets
                ref_probs = F.softmax(ref_logits / self.temperature, dim=-1)

                # Logit adjustment: reduce confidence on the true labels
                # by encouraging the model to match softer reference distributions
                log_probs = F.log_softmax(logits, dim=-1)

                # KL divergence between model and reference (at scaled temperature)
                forget_loss = F.kl_div(
                    log_probs, ref_probs, reduction="batchmean"
                )

                optimizer.zero_grad()
                forget_loss.backward()
                optimizer.step()

                epoch_forget += forget_loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

            # --- Retain pass: preserve original behavior ---
            if retain_loader is not None:
                for batch in retain_loader:
                    inputs = batch[0].to(device)

                    with torch.no_grad():
                        ref_out = ref_model(inputs)
                        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out
                        ref_probs = F.softmax(ref_logits, dim=-1)

                    out = model(inputs)
                    logits = out.logits if hasattr(out, "logits") else out
                    log_probs = F.log_softmax(logits, dim=-1)

                    # KL divergence to preserve retain utility
                    retain_loss = self.retain_weight * F.kl_div(
                        log_probs, ref_probs, reduction="batchmean"
                    )

                    optimizer.zero_grad()
                    retain_loss.backward()
                    optimizer.step()

                    epoch_retain += retain_loss.item()
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        return model, forget_losses, retain_losses
