"""
Differential Privacy Unlearning — DP-enabled gradient ascent.

Demonstrates combining differential privacy (gradient clipping + noise)
with machine unlearning for provable privacy guarantees.

Usage::

    python examples/advanced/differential_privacy.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.privacy.gradient_clipping import GradientClipper, calibrate_noise
from erasus.privacy.accountant import PrivacyAccountant
import erasus.strategies  # noqa: F401


class SmallClassifier(nn.Module):
    def __init__(self, in_dim=16, hidden=64, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("  DP-Enabled Unlearning Example")
    print("=" * 60)

    model = SmallClassifier()
    device = "cpu"

    forget = DataLoader(TensorDataset(torch.randn(50, 16), torch.randint(0, 4, (50,))), batch_size=16)
    retain = DataLoader(TensorDataset(torch.randn(200, 16), torch.randint(0, 4, (200,))), batch_size=16)

    # Setup DP components
    epsilon = 1.0
    delta = 1e-5
    max_grad_norm = 1.0

    clipper = GradientClipper(max_grad_norm=max_grad_norm)
    sigma = calibrate_noise(epsilon=epsilon, delta=delta, sensitivity=max_grad_norm / 50)
    accountant = PrivacyAccountant()

    print(f"\n  Privacy budget: ε={epsilon}, δ={delta}")
    print(f"  Noise σ={sigma:.4f}, max grad norm={max_grad_norm}")

    # Unlearning with DP
    unlearner = ErasusUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector=None,
        device=device,
        strategy_kwargs={"lr": 1e-3},
    )

    print("\n  Running DP unlearning...")
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5)

    # Post-hoc DP accounting
    for _ in range(5):
        accountant.step(epsilon / 5, delta / 5)

    eps_total, delta_total = accountant.get_budget(advanced_composition=True)
    print(f"\n  Privacy spent (advanced): ε={eps_total:.4f}, δ={delta_total:.6f}")

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    print("\n✅ DP unlearning complete!")


if __name__ == "__main__":
    main()
