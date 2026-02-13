"""
erasus.losses.l2_regularization — L2 weight penalty.

Simple L2 distance from pre-unlearning weights, preventing
excessive model drift during unlearning.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class L2Regularization(nn.Module):
    """
    L2 regularization penalising deviation from anchor weights.

    Parameters
    ----------
    weight : float
        Regularization strength.
    """

    def __init__(self, model: nn.Module, weight: float = 1e-3) -> None:
        super().__init__()
        self.weight = weight
        self._anchor: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._anchor[name] = param.data.clone()

    def forward(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self._anchor:
                loss += ((param - self._anchor[name].to(param.device)) ** 2).sum()
        return self.weight * loss
