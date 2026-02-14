"""
Efficiency-Focused Benchmark — Time and memory profiling.

Compares unlearning strategies on computational cost: wall-clock time,
parameter count, and memory usage.

Usage::

    python benchmarks/custom/efficiency_benchmark.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import erasus.strategies  # noqa: F401
from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.utils.helpers import save_json, ensure_dir, count_parameters, model_size_mb


class EffBenchModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("  Efficiency Benchmark")
    print("=" * 60)

    strategies = ["gradient_ascent", "negative_gradient", "fisher_forgetting"]
    model_sizes = [64, 128, 256]
    base_n = 200

    forget = DataLoader(TensorDataset(torch.randn(50, 16), torch.randint(0, 4, (50,))), batch_size=16)
    retain = DataLoader(TensorDataset(torch.randn(base_n, 16), torch.randint(0, 4, (base_n,))), batch_size=16)

    results = {}
    print(f"\n  {'Strategy':<25} {'Hidden':>8} {'Params':>10} {'MB':>8} {'Time':>8}")
    print("-" * 65)

    for hidden in model_sizes:
        for strat in strategies:
            model = EffBenchModel(hidden=hidden)
            params = count_parameters(model)
            mb = model_size_mb(model)

            unlearner = ErasusUnlearner(model=model, strategy=strat, selector=None, device="cpu", strategy_kwargs={"lr": 1e-3})
            t0 = time.time()
            result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)
            elapsed = time.time() - t0

            key = f"{strat}_{hidden}"
            results[key] = {"strategy": strat, "hidden": hidden, "params": params, "size_mb": round(mb, 3), "time": round(elapsed, 3)}
            print(f"  {strat:<25} {hidden:>8} {params:>10,} {mb:>7.3f} {elapsed:>7.2f}s")

    out = ensure_dir("benchmarks/custom/results")
    save_json(results, out / "efficiency_results.json")
    print(f"\n  Saved to: benchmarks/custom/results/efficiency_results.json")
    print("✅ Efficiency benchmark complete!")


if __name__ == "__main__":
    main()
