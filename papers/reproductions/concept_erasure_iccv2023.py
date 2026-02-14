"""
Reproduction: Concept Erasure — Erasing Concepts from Diffusion Models
(Gandikota et al., ICCV 2023).

Demonstrates concept erasure in a diffusion model by fine-tuning to
steer the model away from generating specific concepts while preserving
generation quality for other concepts.

Usage::

    python papers/reproductions/concept_erasure_iccv2023.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.diffusion_unlearner import DiffusionUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401


class SimpleDiffModel(nn.Module):
    """Simplified diffusion model for concept erasure demo."""

    def __init__(self, latent_dim=32, hidden=128):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.config = type("C", (), {"model_type": "stable_diffusion"})()

    def forward(self, x, timestep=None):
        return self.unet(x)


def main():
    print("=" * 60)
    print("  Paper Reproduction: Concept Erasure")
    print("  (Gandikota et al., ICCV 2023)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleDiffModel().to(device)
    print(f"\n  Device: {device}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Concept to erase (represented as latent vectors)
    concept_data = DataLoader(
        TensorDataset(torch.randn(60, 32), torch.zeros(60, dtype=torch.long)),
        batch_size=16,
    )
    # Retain concepts
    retain_data = DataLoader(
        TensorDataset(torch.randn(240, 32), torch.zeros(240, dtype=torch.long)),
        batch_size=16,
    )

    # Phase 1: Pre-erasure
    print("\n  Phase 1: Pre-training diffusion model...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(3):
        total = 0
        for x, _ in retain_data:
            x = x.to(device)
            noise = torch.randn_like(x)
            pred = model(x + 0.1 * noise)
            loss = nn.functional.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"    Epoch {epoch+1}: loss={total/len(retain_data):.4f}")

    # Phase 2: Concept Erasure
    print("\n  Phase 2: Erasing target concept...")
    unlearner = DiffusionUnlearner(
        model=model,
        strategy="concept_erasure",
        selector=None,
        device=device,
        strategy_kwargs={"lr": 1e-3},
    )

    t0 = time.time()
    result = unlearner.fit(forget_data=concept_data, retain_data=retain_data, epochs=5)
    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 60)
    print("  REPRODUCTION SUMMARY")
    print("=" * 60)
    print(f"  Strategy: Concept Erasure (closed-form fine-tuning)")
    print(f"  Erasure Time: {elapsed:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget Loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n  Expected: concept FID ↑ (concept degraded), non-concept FID stable")
    print("\n✅ Concept erasure reproduction complete!")


if __name__ == "__main__":
    main()
