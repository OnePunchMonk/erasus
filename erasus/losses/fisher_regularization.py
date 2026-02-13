"""
erasus.losses.fisher_regularization — Fisher information regularization penalty.

Penalises deviations from the pre-unlearning parameter values
weighted by their Fisher information, preserving high-information
parameters more strongly.

Reference: Golatkar et al. (NeurIPS 2020)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FisherRegularization(nn.Module):
    """Fisher information weighted regularization."""

    def __init__(
        self,
        model: nn.Module,
        retain_loader: Optional[DataLoader] = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = weight
        self._anchor: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}

        # Snapshot pre-unlearning parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._anchor[name] = param.data.clone()

        # Compute Fisher diagonal if retain data available
        if retain_loader is not None:
            self._compute_fisher(model, retain_loader)

    def _compute_fisher(self, model: nn.Module, loader: DataLoader) -> None:
        """Compute diagonal Fisher information on retain data."""
        device = next(model.parameters()).device
        model.train()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self._fisher[name] = torch.zeros_like(param)

        n = 0
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            out = model(inputs)
            logits = out.logits if hasattr(out, "logits") else out
            loss = F.cross_entropy(logits, labels)

            model.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self._fisher[name] += param.grad ** 2
            n += 1

        for name in self._fisher:
            self._fisher[name] /= max(n, 1)

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute Fisher-weighted departure from anchor."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self._anchor:
                delta = param - self._anchor[name].to(param.device)
                if name in self._fisher:
                    loss += (self._fisher[name].to(param.device) * delta ** 2).sum()
                else:
                    loss += (delta ** 2).sum()
        return self.weight * loss
