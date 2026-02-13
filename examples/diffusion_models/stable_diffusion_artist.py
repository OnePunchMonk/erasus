"""
Example: Artist Style Removal from Stable Diffusion.

Demonstrates removing a specific artist's style from a diffusion model
while preserving general image generation capabilities.

Usage::

    python examples/diffusion_models/stable_diffusion_artist.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.diffusion_unlearner import DiffusionUnlearner
import erasus.strategies  # noqa: F401


class TinyDiffusion(nn.Module):
    """Minimal diffusion model stand-in."""

    def __init__(self):
        super().__init__()
        self.unet = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        self.scheduler = True
        self.vae = True

    def forward(self, x):
        return self.unet(x)


def main():
    print("=" * 60)
    print("  Artist Style Removal from Stable Diffusion")
    print("=" * 60)

    model = TinyDiffusion()

    # Latents for images in the target artist's style
    artist_latents = torch.randn(20, 32)
    artist_labels = torch.zeros(20, dtype=torch.long)
    forget = DataLoader(TensorDataset(artist_latents, artist_labels), batch_size=4)

    # General art latents to keep
    general_latents = torch.randn(40, 32)
    general_labels = torch.zeros(40, dtype=torch.long)
    retain = DataLoader(TensorDataset(general_latents, general_labels), batch_size=4)

    print(f"\n  Artist-style samples to forget: {len(forget.dataset)}")
    print(f"  General samples to retain: {len(retain.dataset)}")

    # Unlearning
    unlearner = DiffusionUnlearner(
        model=model,
        strategy="gradient_ascent",
        device="cpu",
        strategy_kwargs={"lr": 5e-4},
    )

    result = unlearner.fit(
        forget_data=forget,
        retain_data=retain,
        epochs=5,
    )

    print(f"\n  ✓ Style removal complete in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Loss: {result.forget_loss_history[0]:.3f} → {result.forget_loss_history[-1]:.3f}")

    print("\n  Expected results with real model:")
    print("  - Prompts mentioning the artist → generic/neutral style")
    print("  - Other prompts → unchanged generation quality")

    print("\n✅ Artist style removal example complete!")


if __name__ == "__main__":
    main()
