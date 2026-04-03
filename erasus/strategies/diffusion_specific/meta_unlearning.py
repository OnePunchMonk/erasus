"""
Meta-Unlearning on Diffusion Models — Robust Concept Erasure.

Paper: "Meta-Unlearning for Stable Concept Erasure" (Gao et al., ICCV 2025)

This strategy simulates post-unlearning relearning in an inner loop,
then updates the original model to make that relearning less effective
while preserving retain-set utility.
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


@strategy_registry.register("meta_unlearning")
class MetaUnlearningStrategy(BaseStrategy):
    """
    Meta-learning-based robust concept erasure.

    Parameters
    ----------
    inner_lr : float
        Learning rate for the simulated relearning loop.
    outer_lr : float
        Learning rate for the outer anti-relearning update.
    num_inner_steps : int
        Number of relearning simulation steps per forget batch.
    forget_weight : float
        Weight of the anti-relearning forget objective.
    retain_weight : float
        Weight of the retain-preservation objective.
    adaptation_penalty : float
        Weight of the parameter-drift penalty against the relearned clone.
    """

    def __init__(
        self,
        inner_lr: float = 1e-4,
        outer_lr: float = 1e-5,
        num_inner_steps: int = 3,
        forget_weight: float = 1.0,
        retain_weight: float = 1.0,
        adaptation_penalty: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight
        self.adaptation_penalty = adaptation_penalty

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

        optimizer = torch.optim.Adam(model.parameters(), lr=self.outer_lr)
        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for _ in range(epochs):
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            for forget_batch in forget_loader:
                if not isinstance(forget_batch, (list, tuple)) or len(forget_batch) < 2:
                    continue

                forget_inputs = forget_batch[0].to(device)
                forget_labels = forget_batch[1].to(device)

                adapted_model = copy.deepcopy(model).to(device)
                adapted_model.train()
                inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

                for _inner in range(self.num_inner_steps):
                    inner_optimizer.zero_grad()
                    inner_logits = self._forward_logits(adapted_model, forget_inputs)
                    inner_loss = F.cross_entropy(inner_logits, forget_labels)
                    inner_loss.backward()
                    inner_optimizer.step()

                current_logits = self._forward_logits(model, forget_inputs)
                uniform_target = torch.full_like(current_logits, 1.0 / current_logits.size(-1))
                anti_relearn_loss = F.kl_div(
                    F.log_softmax(current_logits, dim=-1),
                    uniform_target,
                    reduction="batchmean",
                )
                drift_penalty = self._parameter_drift(model, adapted_model)
                outer_loss = (
                    self.forget_weight * anti_relearn_loss
                    + self.adaptation_penalty * drift_penalty
                )

                optimizer.zero_grad()
                outer_loss.backward()
                optimizer.step()

                epoch_forget += float(outer_loss.item())
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

            if retain_loader is not None:
                for retain_batch in retain_loader:
                    if not isinstance(retain_batch, (list, tuple)) or len(retain_batch) < 2:
                        continue

                    retain_inputs = retain_batch[0].to(device)
                    retain_labels = retain_batch[1].to(device)
                    retain_logits = self._forward_logits(model, retain_inputs)
                    retain_loss = self.retain_weight * F.cross_entropy(retain_logits, retain_labels)

                    optimizer.zero_grad()
                    retain_loss.backward()
                    optimizer.step()

                    epoch_retain += float(retain_loss.item())
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        return model, forget_losses, retain_losses

    @staticmethod
    def _forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        outputs = model(inputs)
        return outputs.logits if hasattr(outputs, "logits") else outputs

    @staticmethod
    def _parameter_drift(model: nn.Module, adapted_model: nn.Module) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for param, adapted_param in zip(model.parameters(), adapted_model.parameters()):
            penalty = penalty + F.mse_loss(param, adapted_param.detach())
        return penalty
