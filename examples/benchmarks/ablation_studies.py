"""
Ablation Studies Example — Automated ablation over hyperparameters.

Usage::

    python examples/benchmarks/ablation_studies.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
import erasus.strategies  # noqa: F401


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("  Ablation Study: Learning Rate × Epochs")
    print("=" * 60)

    forget = DataLoader(TensorDataset(torch.randn(50, 16), torch.randint(0, 4, (50,))), batch_size=16)
    retain = DataLoader(TensorDataset(torch.randn(200, 16), torch.randint(0, 4, (200,))), batch_size=16)
    base_state = SimpleModel().state_dict()

    lrs = [1e-4, 1e-3, 1e-2]
    epoch_vals = [1, 3, 5]

    print(f"\n  {'LR':<10} {'Epochs':<8} {'Time':>8} {'Final Loss':>12}")
    print("-" * 45)

    for lr in lrs:
        for epochs in epoch_vals:
            model = SimpleModel()
            model.load_state_dict(base_state)

            try:
                unlearner = ErasusUnlearner(
                    model=model, strategy="gradient_ascent", selector=None,
                    device="cpu", strategy_kwargs={"lr": lr},
                )
                t0 = time.time()
                result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=epochs)
                elapsed = time.time() - t0
                loss_str = f"{result.forget_loss_history[-1]:.4f}" if result.forget_loss_history else "N/A"
                print(f"  {lr:<10.0e} {epochs:<8} {elapsed:>7.2f}s {loss_str:>12}")
            except Exception as e:
                print(f"  {lr:<10.0e} {epochs:<8} {'ERROR':>8} {str(e)[:20]:>12}")

    print("\n✅ Ablation study complete!")


if __name__ == "__main__":
    main()
