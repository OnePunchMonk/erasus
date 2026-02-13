"""
erasus.losses.adversarial_loss — GAN-style adversarial unlearning loss.

Uses a discriminator to distinguish "unlearned" model outputs from
"retrained" baseline, training the model to fool the discriminator.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdversarialUnlearningLoss(nn.Module):
    """
    GAN-style adversarial loss for unlearning.

    A discriminator tries to distinguish forget-set model outputs
    from random/uniform outputs. The model is trained to fool it.

    Parameters
    ----------
    feature_dim : int
        Dimension of the feature / logit vectors.
    """

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def discriminator_loss(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
    ) -> torch.Tensor:
        """Train discriminator to separate real (retain) from fake (forget)."""
        real_pred = self.discriminator(real_features.detach())
        fake_pred = self.discriminator(fake_features.detach())

        real_loss = nn.functional.binary_cross_entropy(
            real_pred, torch.ones_like(real_pred),
        )
        fake_loss = nn.functional.binary_cross_entropy(
            fake_pred, torch.zeros_like(fake_pred),
        )
        return (real_loss + fake_loss) / 2

    def generator_loss(self, forget_features: torch.Tensor) -> torch.Tensor:
        """Train model (generator) to fool discriminator on forget data."""
        pred = self.discriminator(forget_features)
        # Model wants discriminator to think forget outputs are "real" (retained)
        return nn.functional.binary_cross_entropy(
            pred, torch.ones_like(pred),
        )

    def forward(
        self,
        forget_features: torch.Tensor,
        retain_features: torch.Tensor,
    ) -> torch.Tensor:
        """Combined loss: discriminator + generator."""
        d_loss = self.discriminator_loss(retain_features, forget_features)
        g_loss = self.generator_loss(forget_features)
        return d_loss + g_loss
