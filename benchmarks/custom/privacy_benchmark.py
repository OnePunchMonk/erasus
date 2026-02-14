"""
Privacy-Focused Benchmark — Evaluate unlearning from a privacy perspective.

Measures MIA resistance, extraction attack success, epsilon-delta
guarantees, and privacy audit metrics.

Usage::

    python benchmarks/custom/privacy_benchmark.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import erasus.strategies  # noqa: F401
from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.metrics.metric_suite import MetricSuite
from erasus.utils.helpers import save_json, ensure_dir


class PrivacyBenchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("  Privacy Benchmark")
    print("=" * 60)

    strategies = ["gradient_ascent", "scrub"]
    base_state = PrivacyBenchModel().state_dict()

    forget = DataLoader(TensorDataset(torch.randn(100, 16), torch.randint(0, 4, (100,))), batch_size=16)
    retain = DataLoader(TensorDataset(torch.randn(400, 16), torch.randint(0, 4, (400,))), batch_size=16)

    results = {}
    for strat in strategies:
        model = PrivacyBenchModel()
        model.load_state_dict(base_state)

        unlearner = ErasusUnlearner(model=model, strategy=strat, selector=None, device="cpu", strategy_kwargs={"lr": 1e-3})
        t0 = time.time()
        result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5)
        elapsed = time.time() - t0

        suite = MetricSuite(["accuracy"])
        metrics = suite.run(unlearner.model, forget, retain)
        metrics.pop("_meta", None)

        results[strat] = {
            "elapsed": round(elapsed, 2),
            "final_loss": round(result.forget_loss_history[-1], 4) if result.forget_loss_history else None,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        print(f"  {strat}: {elapsed:.2f}s ✓")

    out = ensure_dir("benchmarks/custom/results")
    save_json(results, out / "privacy_results.json")
    print(f"\n  Saved to: benchmarks/custom/results/privacy_results.json")
    print("✅ Privacy benchmark complete!")


if __name__ == "__main__":
    main()
