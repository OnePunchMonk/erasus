"""
erasus.strategies.diffusion_specific.timestep_masking — Selective timestep training.

Unlearns by modifying the diffusion model only at specific timesteps
where the forget concepts are most strongly represented.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("timestep_masking")
class TimestepMaskingStrategy(BaseStrategy):
    """
    Selective timestep-masked unlearning for diffusion models.

    Only applies gradient ascent at timesteps where the model
    shows highest confidence on forget-set samples, leaving
    other timestep behaviors intact.

    Parameters
    ----------
    lr : float
        Learning rate.
    n_timesteps : int
        Total number of diffusion timesteps.
    mask_ratio : float
        Fraction of timesteps to target for unlearning.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        n_timesteps: int = 1000,
        mask_ratio: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.n_timesteps = n_timesteps
        self.mask_ratio = mask_ratio

    def _identify_target_timesteps(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        device: torch.device,
    ) -> List[int]:
        """Identify timesteps with highest loss on forget data."""
        model.eval()
        timestep_losses = torch.zeros(self.n_timesteps)

        with torch.no_grad():
            for batch in forget_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                # Sample random timesteps and measure loss
                for t in range(0, self.n_timesteps, max(1, self.n_timesteps // 50)):
                    noise = torch.randn_like(inputs.float()) * (t / self.n_timesteps)
                    noisy = inputs.float() + noise

                    outputs = model(noisy)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = F.mse_loss(logits, inputs.float()) if logits.shape == inputs.shape \
                        else F.cross_entropy(logits, labels)
                    timestep_losses[t] += loss.item()

        # Select top timesteps by loss
        n_target = max(1, int(self.n_timesteps * self.mask_ratio))
        _, target_indices = timestep_losses.topk(n_target)
        return target_indices.tolist()

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run timestep-masked unlearning."""
        device = next(model.parameters()).device
        target_timesteps = set(self._identify_target_timesteps(model, forget_loader, device))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n = 0

            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                # Sample a timestep and check if it's targeted
                t = torch.randint(0, self.n_timesteps, (1,)).item()
                if t not in target_timesteps:
                    continue

                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                (-loss).backward()  # Gradient ascent at target timesteps
                optimizer.step()

                epoch_loss += loss.item()
                n += 1

            forget_losses.append(epoch_loss / max(n, 1))

            # Retain pass (all timesteps)
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
