"""
Example: NSFW Concept Removal from Stable Diffusion.

Demonstrates:
1. Using the DiffusionUnlearner to erase NSFW concepts.
2. Running concept erasure strategy.
3. Evaluating generation quality post-unlearning.

Usage::

    python examples/diffusion_models/stable_diffusion_nsfw.py

Note: Uses a tiny model for speed. Replace with
``stabilityai/stable-diffusion-2-1`` for real experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.diffusion_unlearner import DiffusionUnlearner
import erasus.strategies  # noqa: F401


class TinyDiffusion(nn.Module):
    """Minimal diffusion model stand-in for testing."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        # Attributes for diffusion auto-detection
        self.scheduler = True
        self.vae = True

    def forward(self, x):
        return self.unet(x)


def main():
    print("=" * 60)
    print("  Stable Diffusion NSFW Removal Example")
    print("=" * 60)

    model = TinyDiffusion()
    print(f"  Model: TinyDiffusion ({sum(p.numel() for p in model.parameters()):,} params)")

    # NSFW-related latents to forget
    forget = DataLoader(
        TensorDataset(torch.randn(24, 32), torch.zeros(24, dtype=torch.long)),
        batch_size=8,
    )
    # Safe content latents to retain
    retain = DataLoader(
        TensorDataset(torch.randn(48, 32), torch.zeros(48, dtype=torch.long)),
        batch_size=8,
    )

    print("\n  Strategy: gradient_ascent (concept erasure)")
    unlearner = DiffusionUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    result = unlearner.fit(
        forget_data=forget,
        retain_data=retain,
        epochs=3,
    )

    print(f"  ✓ Unlearning complete in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Loss: {result.forget_loss_history[0]:.3f} → {result.forget_loss_history[-1]:.3f}")

    print("\n  In a real scenario, you would now:")
    print("  1. Generate images with NSFW prompts → should produce benign output")
    print("  2. Generate images with safe prompts → should still work normally")
    print("  3. Compute FID to measure generation quality")

    print("\n✅ Stable Diffusion NSFW removal example complete!")


if __name__ == "__main__":
    main()
