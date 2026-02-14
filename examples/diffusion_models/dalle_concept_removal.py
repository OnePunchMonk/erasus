"""
DALL-E Concept Removal — Erase concepts from a DALL-E-like model.

Demonstrates concept erasure in a text-to-image generation pipeline
using the diffusion unlearner with concept erasure strategy.

Usage::

    python examples/diffusion_models/dalle_concept_removal.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.diffusion_unlearner import DiffusionUnlearner
import erasus.strategies  # noqa: F401


class TinyDiffusion(nn.Module):
    """Minimal diffusion model for demo."""

    def __init__(self, latent_dim=16, hidden=64):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.text_encoder = nn.Embedding(256, hidden)
        self.config = type("C", (), {"model_type": "dalle"})()

    def forward(self, x, timestep=None):
        return self.unet(x)


def main():
    print("=" * 60)
    print("  DALL-E Concept Removal Example")
    print("=" * 60)

    model = TinyDiffusion()

    forget = DataLoader(
        TensorDataset(torch.randn(40, 16), torch.zeros(40, dtype=torch.long)),
        batch_size=8,
    )
    retain = DataLoader(
        TensorDataset(torch.randn(160, 16), torch.zeros(160, dtype=torch.long)),
        batch_size=8,
    )

    unlearner = DiffusionUnlearner(
        model=model,
        strategy="concept_erasure",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    print("\n  Running concept erasure...")
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5)

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ DALL-E concept removal complete!")


if __name__ == "__main__":
    main()
