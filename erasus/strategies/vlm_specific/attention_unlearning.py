"""
erasus.strategies.vlm_specific.attention_unlearning — Cross-attention modification for VLMs.

Modifies cross-attention weights between vision and text encoders
to decouple specific concept associations.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("attention_unlearning")
class AttentionUnlearningStrategy(BaseStrategy):
    """
    Cross-attention unlearning for Vision-Language Models.

    Targets the cross-attention layers between vision and text
    encoders, weakening connections associated with the forget data
    while preserving general cross-modal alignment.

    Parameters
    ----------
    lr : float
        Learning rate.
    attn_weight : float
        Weight for the attention dispersion loss.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        attn_weight: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.attn_weight = attn_weight

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run cross-attention unlearning."""
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n = 0

            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                ce_loss = F.cross_entropy(logits, labels)

                # Encourage uniform (dispersed) attention on forget data
                # This breaks cross-modal associations
                attn_entropy_loss = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    if "attn" in name.lower() and "weight" in name.lower():
                        # Encourage weights towards uniform → high entropy
                        w = param.view(-1)
                        p = F.softmax(w.abs(), dim=0)
                        entropy = -(p * (p + 1e-10).log()).sum()
                        attn_entropy_loss -= entropy

                total_loss = -ce_loss + self.attn_weight * attn_entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += ce_loss.item()
                n += 1

            forget_losses.append(epoch_loss / max(n, 1))

            if retain_loader is not None:
                epoch_retain = 0.0
                n_r = 0
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = F.cross_entropy(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_retain += loss.item()
                    n_r += 1
                retain_losses.append(epoch_retain / max(n_r, 1))

        return model, forget_losses, retain_losses
