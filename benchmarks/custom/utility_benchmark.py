"""
Utility Preservation Benchmark — Measure model utility after unlearning.

Tests how well each strategy preserves performance on retain/test sets
after unlearning.

Usage::

    python benchmarks/custom/utility_benchmark.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import erasus.strategies  # noqa: F401
from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.utils.helpers import save_json, ensure_dir


class UtilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def main():
    print("=" * 60)
    print("  Utility Preservation Benchmark")
    print("=" * 60)

    strategies = ["gradient_ascent", "negative_gradient", "fisher_forgetting", "scrub"]
    base_model = UtilityModel()

    # Pre-train the base model
    print("\n  Pre-training base model...")
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    train_data = DataLoader(TensorDataset(torch.randn(500, 16), torch.randint(0, 4, (500,))), batch_size=32)
    for _ in range(10):
        for x, y in train_data:
            loss = F.cross_entropy(base_model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    base_state = base_model.state_dict()

    forget = DataLoader(TensorDataset(torch.randn(50, 16), torch.randint(0, 4, (50,))), batch_size=16)
    retain = DataLoader(TensorDataset(torch.randn(200, 16), torch.randint(0, 4, (200,))), batch_size=16)
    test = DataLoader(TensorDataset(torch.randn(100, 16), torch.randint(0, 4, (100,))), batch_size=16)

    pre_retain_acc = evaluate_accuracy(base_model, retain)
    pre_test_acc = evaluate_accuracy(base_model, test)
    print(f"  Pre-unlearning: retain_acc={pre_retain_acc:.3f}, test_acc={pre_test_acc:.3f}")

    print(f"\n  {'Strategy':<25} {'Retain Acc':>12} {'Test Acc':>10} {'Δ Retain':>10} {'Time':>8}")
    print("-" * 70)

    results = {}
    for strat in strategies:
        model = UtilityModel()
        model.load_state_dict(base_state)

        unlearner = ErasusUnlearner(model=model, strategy=strat, selector=None, device="cpu", strategy_kwargs={"lr": 1e-3})
        t0 = time.time()
        result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)
        elapsed = time.time() - t0

        post_retain = evaluate_accuracy(unlearner.model, retain)
        post_test = evaluate_accuracy(unlearner.model, test)
        delta = post_retain - pre_retain_acc

        results[strat] = {
            "retain_acc": round(post_retain, 4),
            "test_acc": round(post_test, 4),
            "delta_retain": round(delta, 4),
            "time": round(elapsed, 3),
        }
        print(f"  {strat:<25} {post_retain:>11.3f} {post_test:>10.3f} {delta:>+10.3f} {elapsed:>7.2f}s")

    out = ensure_dir("benchmarks/custom/results")
    save_json(results, out / "utility_results.json")
    print(f"\n  Saved to: benchmarks/custom/results/utility_results.json")
    print("✅ Utility benchmark complete!")


if __name__ == "__main__":
    main()
