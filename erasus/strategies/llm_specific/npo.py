"""
NPO — Negative Preference Optimization.

Paper: "Who's Harry Potter? Approximate Unlearning in LLMs"
       (Eldan & Russinovich, 2023) and formalised as NPO in
       "Negative Preference Optimization: How to Make LLMs Forget"
       (Zhang et al., 2024)

NPO treats the forget set as negative preference examples.  A frozen
reference model (the pre-unlearning weights) provides a KL constraint
that prevents catastrophic forgetting on unrelated data.

Loss on forget set:
    L_forget = -E[log σ(-β · (log p_θ(x) - log p_ref(x)))]

Loss on retain set (optional):
    L_retain = KL(p_ref(·|x) || p_θ(·|x))

The reference model acts as an anchor; the student is free to diverge
from it on the forget set but is penalised for diverging on retain data.
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


@strategy_registry.register("npo")
class NPOStrategy(BaseStrategy):
    """
    Negative Preference Optimization for LLM unlearning.

    Parameters
    ----------
    beta : float
        Preference gap scaling (default 0.1).  Higher values make
        the loss steeper around the decision boundary.
    retain_weight : float
        Weight of the retain KL term (default 1.0).
    lr : float
        Learning rate (default 1e-5).
    reference_model : nn.Module, optional
        Pre-unlearning model snapshot.  If None, a deep copy of the
        incoming model is taken at the start of ``unlearn()``.
    """

    def __init__(
        self,
        beta: float = 0.1,
        retain_weight: float = 1.0,
        lr: float = 1e-5,
        reference_model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.retain_weight = retain_weight
        self.lr = lr
        self._reference_model = reference_model

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        device = next(model.parameters()).device

        # Build frozen reference model
        ref_model = self._reference_model
        if ref_model is None:
            ref_model = copy.deepcopy(model).to(device)
        ref_model = ref_model.to(device)
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

            # --- Forget pass ---
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                # Student log-probs
                out = model(inputs)
                logits = out.logits if hasattr(out, "logits") else out
                log_p = F.log_softmax(logits, dim=-1)

                # Reference log-probs (no gradient)
                with torch.no_grad():
                    ref_out = ref_model(inputs)
                    ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out
                    log_p_ref = F.log_softmax(ref_logits, dim=-1)

                if labels.dim() == 1:
                    # Classification: log-prob of true class
                    idx = labels.unsqueeze(1)
                    student_lp = log_p.gather(1, idx).squeeze(1)
                    ref_lp = log_p_ref.gather(1, idx).squeeze(1)
                else:
                    student_lp = log_p.mean(dim=list(range(1, log_p.dim())))
                    ref_lp = log_p_ref.mean(dim=list(range(1, log_p_ref.dim())))

                # NPO loss: -log σ(-β · (log p_θ - log p_ref))
                gap = student_lp - ref_lp
                loss = -F.logsigmoid(-self.beta * gap).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_forget += loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

            # --- Retain pass: KL against reference ---
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

                    kl = F.kl_div(log_probs, ref_probs, reduction="batchmean")
                    retain_loss = self.retain_weight * kl

                    optimizer.zero_grad()
                    retain_loss.backward()
                    optimizer.step()

                    epoch_retain += retain_loss.item()
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        return model, forget_losses, retain_losses
