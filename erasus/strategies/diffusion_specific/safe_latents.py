"""
erasus.strategies.diffusion_specific.safe_latents — Safe Latent Diffusion.

Constrains the latent space to avoid generating unsafe content
by modifying the noise distribution during inference.

Reference: Schramowski et al. (2023) — "Safe Latent Diffusion"
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("safe_latents")
class SafeLatentsStrategy(BaseStrategy):
    """
    Safe Latent Diffusion unlearning.

    Learns a safety direction in latent space from unsafe/safe pairs,
    then steers generation away from unsafe regions by modifying
    the model's projection weights.

    Parameters
    ----------
    lr : float
        Learning rate for safety fine-tuning.
    safety_weight : float
        Strength of the safety constraint.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        safety_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.safety_weight = safety_weight

    def _compute_safety_direction(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the unsafe→safe direction in representation space."""
        model.eval()

        unsafe_embeds = []
        safe_embeds = []

        with torch.no_grad():
            for batch in forget_loader:
                inputs = batch[0].to(device)
                out = model(inputs)
                feat = out.logits if hasattr(out, "logits") else out
                unsafe_embeds.append(feat.mean(dim=0))

            if retain_loader is not None:
                for batch in retain_loader:
                    inputs = batch[0].to(device)
                    out = model(inputs)
                    feat = out.logits if hasattr(out, "logits") else out
                    safe_embeds.append(feat.mean(dim=0))

        unsafe_mean = torch.stack(unsafe_embeds).mean(dim=0) if unsafe_embeds else torch.zeros(1)
        safe_mean = torch.stack(safe_embeds).mean(dim=0) if safe_embeds else torch.zeros_like(unsafe_mean)

        # Direction from unsafe → safe
        direction = safe_mean - unsafe_mean
        direction = direction / (direction.norm() + 1e-8)
        return direction

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run safe latents unlearning."""
        device = next(model.parameters()).device

        safety_dir = self._compute_safety_direction(
            model, forget_loader, retain_loader, device,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n = 0

            # Push forget representations towards safety direction
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                # Safety loss: project outputs towards safe direction
                proj = (logits * safety_dir.unsqueeze(0)).sum(dim=-1)
                safety_loss = -proj.mean() * self.safety_weight

                # Also push away from correct predictions
                ce_loss = F.cross_entropy(logits, labels)
                total_loss = safety_loss + (-ce_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += ce_loss.item()
                n += 1

            forget_losses.append(epoch_loss / max(n, 1))

            # Retain pass
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
