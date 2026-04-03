"""
WGA / FPGA — weighted gradient ascent variants for targeted unlearning.

WGA applies sample-level weighting, while FPGA extends this to
token-level weighting for sequence-style forget batches.
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
    Weighted gradient ascent for selective unlearning.
    """

    def __init__(
        self,
        weighting: str = "entropy",
        lr: float = 1e-3,
        weight_scale: float = 1.0,
        retain_weight: float = 1.0,
        token_weighted: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weighting = weighting
        self.lr = lr
        self.weight_scale = weight_scale
        self.retain_weight = retain_weight
        self.token_weighted = token_weighted

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

        for _ in range(epochs):
            epoch_loss = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                weights = self._compute_weights(logits, labels)
                ce = self._per_target_loss(logits, labels)

                if self.token_weighted and ce.dim() > 1:
                    valid_mask = labels.ne(-100) if labels.shape == ce.shape else torch.ones_like(ce, dtype=torch.bool)
                    weighted = weights * ce * valid_mask.to(ce.dtype)
                    denom = valid_mask.sum().clamp_min(1).to(ce.dtype)
                    loss = -weighted.sum() / denom
                else:
                    if ce.dim() > 1:
                        ce = ce.mean(dim=-1)
                    if weights.dim() > 1:
                        weights = weights.mean(dim=-1)
                    loss = -(weights * ce).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_forget += 1

            forget_losses.append(epoch_loss / max(n_forget, 1))

            if retain_loader is not None:
                for batch in retain_loader:
                    inputs = batch[0].to(device)
                    weight_reg = sum(p.pow(2).sum() for p in model.parameters()) / 1000.0
                    retain_loss = self.retain_weight * weight_reg

                    optimizer.zero_grad()
                    retain_loss.backward()
                    optimizer.step()

                    epoch_retain += retain_loss.item()
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        return model, forget_losses, retain_losses

    def _per_target_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        if logits.dim() == 2 and labels.dim() == 1:
            return -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        if logits.dim() >= 3 and labels.dim() == logits.dim() - 1:
            safe_labels = labels.clamp_min(0)
            return -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        return -log_probs.mean(dim=-1)

    def _compute_weights(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)

        if self.weighting == "uniform":
            return torch.ones_like(labels, dtype=logits.dtype, device=logits.device) if self.token_weighted and labels.dim() > 1 else torch.ones(logits.size(0), device=logits.device, dtype=logits.dtype)

        if self.weighting == "entropy":
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            entropy = entropy / (entropy.max() + 1e-8)
            return self.weight_scale * entropy

        if self.weighting == "confidence":
            if logits.dim() == 2 and labels.dim() == 1:
                true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            elif logits.dim() >= 3 and labels.dim() == logits.dim() - 1:
                safe_labels = labels.clamp_min(0)
                true_probs = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            else:
                true_probs = probs.max(dim=-1).values
            return self.weight_scale * true_probs

        raise ValueError(
            f"Unknown weighting '{self.weighting}'. "
            "Choose from: 'uniform', 'entropy', 'confidence'."
        )


@strategy_registry.register("fpga")
class FPGAStrategy(WGAStrategy):
    """
    FPGA: token-weighted gradient ascent.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(token_weighted=True, **kwargs)
