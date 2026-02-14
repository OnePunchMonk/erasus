"""
Compare Methods — Side-by-side strategy comparison on the same dataset.

Usage::

    python examples/benchmarks/compare_methods.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401
import erasus.selectors   # noqa: F401


class SimpleModel(nn.Module):
    def __init__(self, in_dim=16, hidden=64, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("  Method Comparison Benchmark")
    print("=" * 60)

    strategies = ["gradient_ascent", "negative_gradient", "fisher_forgetting", "scrub"]
    base_state = SimpleModel().state_dict()

    forget = DataLoader(TensorDataset(torch.randn(50, 16), torch.randint(0, 4, (50,))), batch_size=16)
    retain = DataLoader(TensorDataset(torch.randn(200, 16), torch.randint(0, 4, (200,))), batch_size=16)

    print(f"\n  {'Strategy':<25} {'Time':>8} {'Forget Loss':>12} {'Status':>8}")
    print("-" * 60)

    for strat in strategies:
        model = SimpleModel()
        model.load_state_dict(base_state)

        try:
            unlearner = ErasusUnlearner(
                model=model, strategy=strat, selector=None,
                device="cpu", strategy_kwargs={"lr": 1e-3},
            )
            t0 = time.time()
            result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)
            elapsed = time.time() - t0

            loss_str = f"{result.forget_loss_history[-1]:.4f}" if result.forget_loss_history else "N/A"
            print(f"  {strat:<25} {elapsed:>7.2f}s {loss_str:>12} {'✓':>8}")
        except Exception as e:
            print(f"  {strat:<25} {'—':>8} {'—':>12} {'✗':>8}  ({e})")

    print("\n✅ Method comparison complete!")


if __name__ == "__main__":
    main()
