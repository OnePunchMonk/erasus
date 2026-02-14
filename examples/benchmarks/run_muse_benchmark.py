"""
MUSE Benchmark Example — Run the MUSE benchmark.

Usage::

    python examples/benchmarks/run_muse_benchmark.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.llm_unlearner import LLMUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401
import erasus.selectors   # noqa: F401


class BenchModel(nn.Module):
    def __init__(self, vocab=256, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, vocab)
        self.config = type("C", (), {"model_type": "llama", "vocab_size": vocab})()

    def forward(self, x):
        return self.head(self.net(self.emb(x).mean(1)))


def main():
    print("=" * 60)
    print("  MUSE Benchmark Runner")
    print("=" * 60)

    strategies = ["gradient_ascent", "negative_gradient"]
    model_template = BenchModel()

    forget = DataLoader(
        TensorDataset(torch.randint(0, 256, (64, 32)), torch.randint(0, 256, (64,))),
        batch_size=16,
    )
    retain = DataLoader(
        TensorDataset(torch.randint(0, 256, (256, 32)), torch.randint(0, 256, (256,))),
        batch_size=16,
    )

    results = {}
    for strat in strategies:
        print(f"\n  Strategy: {strat}")
        model = BenchModel()
        model.load_state_dict(model_template.state_dict())

        try:
            unlearner = LLMUnlearner(
                model=model, strategy=strat, selector=None,
                device="cpu", strategy_kwargs={"lr": 1e-3},
            )
            t0 = time.time()
            result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)
            elapsed = time.time() - t0

            results[strat] = {"time": round(elapsed, 2), "loss": round(result.forget_loss_history[-1], 4) if result.forget_loss_history else None}
            print(f"    ✓ {elapsed:.2f}s")
        except Exception as e:
            results[strat] = {"error": str(e)}
            print(f"    ✗ {e}")

    print("\n" + "=" * 60)
    for name, r in results.items():
        print(f"  {name:<25} {r}")
    print("\n✅ MUSE benchmark complete!")


if __name__ == "__main__":
    main()
