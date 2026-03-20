"""
SimNPO — Simplified Negative Preference Optimization.

Paper: "SimNPO: Simplicity Prevails: Rethinking Negative Preference
Optimization for LLM Unlearning" (Fan et al., NeurIPS 2025)

Key insight: NPO's reference model introduces a "reference model bias"
that gives a misleading impression of unlearning effectiveness.  SimNPO
removes the reference model dependency entirely, using only the forget
set with a length-normalised negative preference loss.

Loss: L = -E_{x∈D_f}[log σ(-β · (1/|x|) Σ log p_θ(xₜ|x_{<t}))]

After successful unlearning, the model should produce near-uniform
distributions on forget-set queries without degrading on retain data.
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


@strategy_registry.register("simnpo")
class SimNPOStrategy(BaseStrategy):
    """
    SimNPO: reference-free negative preference unlearning.

    Trains the model to assign lower likelihood to forget-set sequences
    via a length-normalised sigmoid loss, without needing a frozen
    reference model.  An optional retain pass (KL against a snapshot)
    preserves utility.

    Parameters
    ----------
    beta : float
        Inverse temperature scaling the preference gap (default 0.1).
    gamma : float
        Length-normalisation floor to avoid division by zero (default 1.0).
    retain_weight : float
        Weight of the retain KL loss relative to the forget loss (default 1.0).
    lr : float
        Learning rate (default 1e-5).
    """

    def __init__(
        self,
        beta: float = 0.1,
        gamma: float = 1.0,
        retain_weight: float = 1.0,
        lr: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.gamma = gamma
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

        # Snapshot for retain KL (frozen reference)
        if retain_loader is not None:
            ref_model = copy.deepcopy(model).to(device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
        else:
            ref_model = None

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            # --- Forget pass: SimNPO loss ---
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                # Per-sample log-probabilities of true labels
                log_probs = F.log_softmax(logits, dim=-1)
                # For classification: single log-prob per sample
                if labels.dim() == 1:
                    seq_log_prob = log_probs[range(len(labels)), labels]   # (B,)
                    length = torch.ones_like(seq_log_prob) * max(self.gamma, 1.0)
                else:
                    # Sequence: average across positions
                    seq_log_prob = log_probs.mean(dim=1).mean(dim=-1)
                    length = torch.full((inputs.size(0),), max(self.gamma, inputs.size(1)),
                                       device=device)

                # Length-normalised negative preference loss
                normalised = seq_log_prob / length
                loss = -F.logsigmoid(-self.beta * normalised).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_forget += loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

            # --- Retain pass: KL against frozen snapshot ---
            if retain_loader is not None and ref_model is not None:
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
