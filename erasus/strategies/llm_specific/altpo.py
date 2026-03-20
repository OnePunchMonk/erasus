"""
AltPO — Alternate Preference Optimization.

Paper: "Alternate Preference Optimization for Unlearning Factual
Knowledge in Large Language Models" (Choi et al., 2024)

Key insight: gradient ascent and NPO both push the model to produce
low-probability outputs on the forget set, often resulting in
incoherent or nonsensical responses.  AltPO instead combines:

1. Negative feedback on the forget set  (forget queries → wrong)
2. Positive feedback for alternative responses (forget queries → IDK)

This teaches the model *what to say instead*, producing coherent
refusals or alternative answers rather than garbage.

Loss:
    L = -E_{x∈D_f}[log σ(β(log p_θ(y_alt|x) - log p_θ(y_f|x)))]

where y_f is the true (to-be-forgotten) label and y_alt is an
alternative (e.g. uniform / random / "I don't know" class).
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


@strategy_registry.register("altpo")
class AltPOStrategy(BaseStrategy):
    """
    AltPO: teach the model to produce alternative answers on forget queries.

    Parameters
    ----------
    beta : float
        Preference gap scaling (default 0.1).
    retain_weight : float
        Weight of retain KL loss (default 1.0).
    lr : float
        Learning rate (default 1e-5).
    alt_strategy : str
        How to generate alternative labels.  Options:

        - ``"uniform"``  — target is the uniform distribution (IDK)
        - ``"random"``   — target is a random incorrect class per sample
        - ``"lowest"``   — target is the currently least-likely class

        Default: ``"uniform"``.
    reference_model : nn.Module, optional
        Frozen reference for retain KL.  If None, a snapshot of the
        model at the start of ``unlearn()`` is used.
    """

    def __init__(
        self,
        beta: float = 0.1,
        retain_weight: float = 1.0,
        lr: float = 1e-5,
        alt_strategy: str = "uniform",
        reference_model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.retain_weight = retain_weight
        self.lr = lr
        self.alt_strategy = alt_strategy
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

        # Frozen reference for retain KL
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

            # --- Forget pass: AltPO loss ---
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                out = model(inputs)
                logits = out.logits if hasattr(out, "logits") else out
                log_probs = F.log_softmax(logits, dim=-1)
                n_classes = logits.size(-1)

                # Log-prob of the TRUE (to-be-forgotten) label
                if labels.dim() == 1:
                    true_lp = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                else:
                    true_lp = log_probs.mean(dim=list(range(1, log_probs.dim())))

                # Log-prob of the ALTERNATIVE label
                alt_lp = self._alt_log_prob(log_probs, labels, n_classes)

                # AltPO loss: prefer alt over true
                gap = alt_lp - true_lp   # positive when model already prefers alt
                loss = -F.logsigmoid(self.beta * gap).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_forget += loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

            # --- Retain pass ---
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

    def _alt_log_prob(
        self,
        log_probs: torch.Tensor,
        true_labels: torch.Tensor,
        n_classes: int,
    ) -> torch.Tensor:
        """Compute log-prob of the alternative (non-forget) label."""
        if self.alt_strategy == "uniform":
            # Alternative is the uniform distribution: log(1/n_classes)
            return torch.full(
                (log_probs.size(0),),
                -torch.log(torch.tensor(float(n_classes))).item(),
                device=log_probs.device,
            )

        elif self.alt_strategy == "random":
            # Alternative is a random class != true label
            alt_labels = torch.randint(0, n_classes, true_labels.shape, device=log_probs.device)
            # Ensure alt != true
            same = alt_labels == true_labels
            alt_labels[same] = (alt_labels[same] + 1) % n_classes
            return log_probs.gather(1, alt_labels.unsqueeze(1)).squeeze(1)

        elif self.alt_strategy == "lowest":
            # Alternative is the currently least-likely class
            alt_labels = log_probs.argmin(dim=-1)
            return log_probs.gather(1, alt_labels.unsqueeze(1)).squeeze(1)

        else:
            raise ValueError(
                f"Unknown alt_strategy '{self.alt_strategy}'. "
                "Choose from: 'uniform', 'random', 'lowest'."
            )
