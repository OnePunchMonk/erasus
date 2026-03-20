"""
DExperts — Inference-time unlearning via expert/anti-expert ensembling.

Paper: "DExperts: Decoding-Time Controlled Text Generation with
Experts and Anti-Experts" (Liu et al., ACL 2021), adapted for
machine unlearning (2024).

DExperts requires no gradient computation on the base model.
Instead it:

1. Fine-tunes a small "anti-expert" model on the forget set
   (this is the only training that happens).
2. At inference time, combines three models:
       logits = logits_base + α · (logits_expert - logits_anti)
   where logits_expert = logits_base (base is its own expert)
   and logits_anti is the anti-expert's output.

The effect: tokens strongly predicted by the anti-expert (forget
content) are suppressed; everything else is unchanged.

``unlearn()`` returns a ``DExpertsWrapper`` — an ``nn.Module``
whose ``forward()`` implements the ensemble computation.
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


class DExpertsWrapper(nn.Module):
    """
    Wraps a base model and an anti-expert to compute ensemble logits.

    forward() returns adjusted logits:
        logits_base + alpha * (logits_base - logits_anti)
    """

    def __init__(
        self,
        base_model: nn.Module,
        anti_expert: nn.Module,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.anti_expert = anti_expert
        self.alpha = alpha

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        base_out = self.base_model(*args, **kwargs)
        base_logits = base_out.logits if hasattr(base_out, "logits") else base_out

        with torch.no_grad():
            anti_out = self.anti_expert(*args, **kwargs)
            anti_logits = anti_out.logits if hasattr(anti_out, "logits") else anti_out

        # Suppress anti-expert tokens
        adjusted = base_logits + self.alpha * (base_logits - anti_logits)
        return adjusted

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to base model for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


@strategy_registry.register("dexperts")
class DExpertsStrategy(BaseStrategy):
    """
    Inference-time unlearning via expert/anti-expert ensembling.

    Does not modify the base model weights.  Trains an anti-expert on
    the forget set, then returns a ``DExpertsWrapper`` that combines
    them at inference time.

    Parameters
    ----------
    alpha : float
        Suppression strength (default 1.0).  Higher values more
        aggressively suppress forget-set tokens.
    anti_expert_lr : float
        Learning rate for fine-tuning the anti-expert (default 1e-4).
    anti_expert_epochs : int
        Epochs to train the anti-expert (default 3).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        anti_expert_lr: float = 1e-4,
        anti_expert_epochs: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
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
        device = next(model.parameters()).device

        # Build anti-expert: fine-tune a copy on forget set
        anti_expert = copy.deepcopy(model).to(device)
        anti_expert.train()
        optimizer = torch.optim.Adam(anti_expert.parameters(), lr=self.anti_expert_lr)
        criterion = nn.CrossEntropyLoss()

        anti_losses: List[float] = []
        n_epochs = max(epochs, self.anti_expert_epochs)

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n = 0
            for batch in forget_loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, labels = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()
                out = anti_expert(inputs)
                logits = out.logits if hasattr(out, "logits") else out
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n += 1

            anti_losses.append(epoch_loss / max(n, 1))

        anti_expert.eval()
        for p in anti_expert.parameters():
            p.requires_grad = False

        # Return a wrapper — base model weights are NOT modified
        wrapper = DExpertsWrapper(
            base_model=model,
            anti_expert=anti_expert,
            alpha=self.alpha,
        )

        # Return lists consistent with BaseStrategy contract
        return wrapper, anti_losses, []
