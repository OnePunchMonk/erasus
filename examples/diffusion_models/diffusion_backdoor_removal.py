"""
Diffusion Backdoor Removal — Remove backdoor triggers from a diffusion model.

Demonstrates using backdoor-poisoned data (from BackdoorGenerator) and
then unlearning the backdoor patterns.

Usage::

    python examples/diffusion_models/diffusion_backdoor_removal.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.data.synthetic.backdoor_generator import BackdoorGenerator
from erasus.unlearners.diffusion_unlearner import DiffusionUnlearner
import erasus.strategies  # noqa: F401


class TinyUNet(nn.Module):
    def __init__(self, dim=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.config = type("C", (), {"model_type": "stable_diffusion"})()

    def forward(self, x, timestep=None):
        return self.net(x)


def main():
    print("=" * 60)
    print("  Diffusion Backdoor Removal Example")
    print("=" * 60)

    # Phase 1: Generate backdoor-poisoned data
    print("\n  Phase 1: Generating backdoor data...")
    clean_data = torch.randn(200, 16)
    clean_labels = torch.randint(0, 4, (200,))
    clean_ds = TensorDataset(clean_data, clean_labels)

    backdoor_gen = BackdoorGenerator(trigger_type="patch", target_class=0, poison_ratio=0.2)
    poisoned_ds = backdoor_gen.generate(clean_ds)
    print(f"    Poisoned dataset size: {len(poisoned_ds)}")

    # Phase 2: Identify and forget backdoor samples
    print("\n  Phase 2: Unlearning backdoor patterns...")
    model = TinyUNet()

    forget = DataLoader(TensorDataset(torch.randn(40, 16), torch.zeros(40, dtype=torch.long)), batch_size=8)
    retain = DataLoader(TensorDataset(torch.randn(160, 16), torch.zeros(160, dtype=torch.long)), batch_size=8)

    unlearner = DiffusionUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5)

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ Backdoor removal complete!")


if __name__ == "__main__":
    main()
