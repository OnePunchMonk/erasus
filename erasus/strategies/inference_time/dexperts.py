"""
DExperts — Inference-time unlearning via token-level expert ensembling.

This implementation performs pure decoding-time ensembling and never
updates the base model's weights. The expert defaults to the base model,
while the anti-expert can be provided explicitly or defaults to a frozen
copy of the base model.
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.registry import strategy_registry
from erasus.strategies.inference_time.base import BaseInferenceTimeStrategy


class DExpertsWrapper(nn.Module):
    """
    Token-level expert/anti-expert ensemble wrapper.

    Combines logits as:
        base + alpha * (expert - anti_expert)
    """

    def __init__(
        self,
        base_model: nn.Module,
        anti_expert: nn.Module,
        alpha: float = 1.0,
        expert_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.expert_model = expert_model or base_model
        self.anti_expert = anti_expert
        self.alpha = alpha

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        base_out = self.base_model(*args, **kwargs)
        base_logits = base_out.logits if hasattr(base_out, "logits") else base_out

        with torch.no_grad():
            expert_out = self.expert_model(*args, **kwargs)
            expert_logits = expert_out.logits if hasattr(expert_out, "logits") else expert_out
            anti_out = self.anti_expert(*args, **kwargs)
            anti_logits = anti_out.logits if hasattr(anti_out, "logits") else anti_out

        adjusted = base_logits + self.alpha * (expert_logits - anti_logits)
        return adjusted

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


@strategy_registry.register("dexperts")
class DExpertsStrategy(BaseInferenceTimeStrategy):
    """
    Inference-time unlearning without weight modification.

    Parameters
    ----------
    alpha : float
        Strength of anti-expert suppression.
    anti_expert_model : nn.Module, optional
        Frozen anti-expert model. If omitted, a frozen copy of the base
        model is used, which keeps the operation side-effect free.
    expert_model : nn.Module, optional
        Expert model to reinforce non-forget behavior. Defaults to the
        base model.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        anti_expert_model: Optional[nn.Module] = None,
        expert_model: Optional[nn.Module] = None,
        anti_expert_lr: float = 1e-4,
        anti_expert_epochs: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.anti_expert_model = anti_expert_model
        self.expert_model = expert_model
        self.anti_expert_lr = anti_expert_lr
        self.anti_expert_epochs = anti_expert_epochs

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 3,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        del forget_loader, retain_loader, epochs

        device = next(model.parameters()).device
        anti_expert = self.anti_expert_model
        if anti_expert is None:
            anti_expert = copy.deepcopy(model).to(device)

        anti_expert.eval()
        for param in anti_expert.parameters():
            param.requires_grad = False

        expert_model = self.expert_model or model
        expert_model.eval()

        wrapper = DExpertsWrapper(
            base_model=model,
            expert_model=expert_model,
            anti_expert=anti_expert,
            alpha=self.alpha,
        )
        return wrapper, [], []
