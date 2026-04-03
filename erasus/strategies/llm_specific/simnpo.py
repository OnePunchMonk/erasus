"""
SimNPO — Simplified Negative Preference Optimization.

Paper: "SimNPO: Simplicity Prevails: Rethinking Negative Preference
Optimization for LLM Unlearning" (Fan et al., NeurIPS 2025)

SimNPO removes NPO's frozen reference model and optimises a single
forget-only preference objective on the forget set. The core signal is
the average log-probability assigned to the forget labels/tokens; the
strategy pushes that quantity downward directly via a sigmoid loss.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors nested in common batch containers to the target device."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(val, device) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(val, device) for val in value)
    return value


def _split_batch(batch: Any, device: torch.device) -> Tuple[Any, Optional[torch.Tensor]]:
    """Extract model inputs and labels from tuple- or dict-style batches."""
    batch = _move_to_device(batch, device)
    if isinstance(batch, dict):
        labels = batch.get("labels")
        return batch, labels
    if isinstance(batch, (list, tuple)):
        if len(batch) == 1:
            return batch[0], None
        return batch[0], batch[1]
    return batch, None


def _forward_logits(model: nn.Module, model_inputs: Any) -> torch.Tensor:
    """Run the model and always return raw logits."""
    outputs = model(**model_inputs) if isinstance(model_inputs, dict) else model(model_inputs)
    return outputs.logits if hasattr(outputs, "logits") else outputs


def _token_log_probs_and_mask(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return target-token log-probabilities plus a validity mask.

    Supports both standard classification ``(B, C)`` and token-level
    language modeling ``(B, T, V)`` with ``ignore_index=-100`` labels.
    """
    log_probs = F.log_softmax(logits, dim=-1)

    if labels is None:
        values = log_probs.max(dim=-1).values
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        mask = torch.ones_like(values, dtype=torch.bool)
        return values, mask

    if logits.dim() == 2 and labels.dim() == 1:
        gathered = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1).unsqueeze(1)
        mask = torch.ones_like(gathered, dtype=torch.bool)
        return gathered, mask

    if logits.dim() >= 3 and labels.dim() == logits.dim() - 1:
        safe_labels = labels.clamp_min(0)
        gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        mask = labels.ne(-100)
        return gathered, mask

    flat = log_probs.reshape(log_probs.size(0), -1).mean(dim=-1, keepdim=True)
    mask = torch.ones_like(flat, dtype=torch.bool)
    return flat, mask


@strategy_registry.register("simnpo")
class SimNPOStrategy(BaseStrategy):
    """
    Reference-free negative preference optimisation on forget data only.

    Parameters
    ----------
    beta : float
        Inverse-temperature for the preference sigmoid.
    gamma : float
        Minimum normalisation denominator when averaging token log-probs.
    lr : float
        Learning rate.
    """

    def __init__(
        self,
        beta: float = 0.1,
        gamma: float = 1.0,
        lr: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.gamma = gamma
        self.lr = lr

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        del retain_loader  # SimNPO is forget-data-only by design.

        device = next(model.parameters()).device
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for _ in range(epochs):
            epoch_forget = 0.0
            n_forget = 0

            for batch in forget_loader:
                model_inputs, labels = _split_batch(batch, device)
                logits = _forward_logits(model, model_inputs)
                token_log_probs, valid_mask = _token_log_probs_and_mask(logits, labels)

                token_log_probs = token_log_probs * valid_mask.to(token_log_probs.dtype)
                token_counts = valid_mask.sum(dim=-1).clamp_min(self.gamma).to(logits.dtype)
                average_log_prob = token_log_probs.sum(dim=-1) / token_counts

                loss = -F.logsigmoid(-self.beta * average_log_prob).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_forget += loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

        return model, forget_losses, retain_losses
