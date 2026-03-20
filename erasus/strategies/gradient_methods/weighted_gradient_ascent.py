"""
WGA — Weighted Gradient Ascent for Targeted Unlearning.

Paper: "Weighted Gradient Ascent for Selective Unlearning" (2024)

Key insight: Standard gradient ascent applies uniform gradients across all
forget-set samples and tokens. WGA weights each gradient by per-token or
per-sample importance, enabling fine-grained control over what gets unlearned.

This produces more stable unlearning and preserves utility better than
uniform gradient ascent, because tokens less important to the model's
general capabilities are assigned higher unlearning priority.

Loss: L = E_{x ∈ D_f}[w_i · ∇ log p_θ(y_i|x)]
where w_i is the weight of token i (higher = more important to unlearn).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("wga")
class WGAStrategy(BaseStrategy):
    """
    WGA: Weighted Gradient Ascent for targeted unlearning.

    Applies token-wise or sample-wise weights to gradient ascent updates,
    allowing fine-grained control over unlearning strength. Tokens/samples
    with higher weights contribute more to pushing the model toward
    low-probability outputs.

    Parameters
    ----------
    weighting : str
        How to compute token weights. Options:
        - ``"uniform"`` — all tokens equally weighted (equivalent to standard GA)
        - ``"entropy"`` — weight by output entropy (high entropy tokens get higher weight)
        - ``"confidence"`` — weight by prediction confidence (high-confidence tokens get higher weight)
        Default: ``"entropy"``.
    lr : float
        Learning rate for gradient ascent (default 1e-3).
    weight_scale : float
        Multiplicative scale for computed weights (default 1.0).
    retain_weight : float
        Weight of retain KL loss (default 1.0).
    """

    def __init__(
        self,
        weighting: str = "entropy",
        lr: float = 1e-3,
        weight_scale: float = 1.0,
        retain_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weighting = weighting
        self.lr = lr
        self.weight_scale = weight_scale
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
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            # --- Forget pass: weighted gradient ascent ---
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                out = model(inputs)
                logits = out.logits if hasattr(out, "logits") else out

                # Compute per-token weights
                weights = self._compute_weights(logits, labels)

                # Weighted cross-entropy (gradient ascent): maximize loss
                if labels.dim() == 1:
                    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                else:
                    true_logits = logits.mean(dim=list(range(1, logits.dim())))

                # Standard cross-entropy via log-softmax
                log_probs = F.log_softmax(logits, dim=-1)
                if labels.dim() == 1:
                    ce = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                else:
                    ce = -log_probs.mean(dim=list(range(1, log_probs.dim())))

                # Apply weights and negate for gradient ascent
                weighted_loss = -(weights * ce).mean()

                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

                epoch_loss += weighted_loss.item()
                n_forget += 1

            forget_losses.append(epoch_loss / max(n_forget, 1))

            # --- Retain pass: KL divergence ---
            if retain_loader is not None:
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)

                    out = model(inputs)
                    logits = out.logits if hasattr(out, "logits") else out

                    # For retain data, we don't modify the loss — just ensure
                    # the model doesn't degrade. In practice, this is often
                    # done via a separate reference model + KL, but here we
                    # use a lightweight L2 regularization on weights instead.
                    weight_reg = sum(p.pow(2).sum() for p in model.parameters()) / 1000.0
                    retain_loss = self.retain_weight * weight_reg

                    optimizer.zero_grad()
                    retain_loss.backward()
                    optimizer.step()

                    epoch_retain += retain_loss.item()
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        return model, forget_losses, retain_losses

    def _compute_weights(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token weights based on the weighting strategy."""
        batch_size = logits.size(0)

        if self.weighting == "uniform":
            # All tokens equally weighted
            return torch.ones(batch_size, device=logits.device)

        elif self.weighting == "entropy":
            # Weight by output entropy (high entropy = high weight)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            # Normalize to [0, weight_scale]
            entropy = entropy / (entropy.max() + 1e-8)
            return self.weight_scale * entropy

        elif self.weighting == "confidence":
            # Weight by prediction confidence on the true label
            # High confidence = high weight (important to unlearn)
            probs = F.softmax(logits, dim=-1)
            if labels.dim() == 1:
                true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            else:
                true_probs = probs.mean(dim=list(range(1, probs.dim())))
            # Higher confidence gets higher weight
            weights = self.weight_scale * true_probs
            return weights

        else:
            raise ValueError(
                f"Unknown weighting '{self.weighting}'. "
                "Choose from: 'uniform', 'entropy', 'confidence'."
            )
