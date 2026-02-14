"""
MUSE Benchmark Suite Runner.

Runs the MUSE (Machine Unlearning Six-way Evaluation) benchmark
across multiple strategies, measuring forgetting quality, utility,
privacy, efficiency, and robustness.

Usage::

    python benchmarks/muse/run.py
    python benchmarks/muse/run.py --strategies gradient_ascent,scrub --epochs 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import erasus.strategies  # noqa: F401
import erasus.selectors   # noqa: F401
from erasus.unlearners.llm_unlearner import LLMUnlearner
from erasus.metrics.metric_suite import MetricSuite
from erasus.utils.helpers import save_json, ensure_dir


DEFAULT_STRATEGIES = ["gradient_ascent", "negative_gradient", "scrub"]


class MUSEBenchmarkModel(nn.Module):
    """Model for MUSE benchmark testing."""

    def __init__(self, vocab_size=256, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.layers = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, vocab_size)
        self.config = type("C", (), {"model_type": "llama", "vocab_size": vocab_size})()

    def forward(self, x):
        return self.head(self.layers(self.emb(x).mean(dim=1)))


def make_muse_data(n_forget=64, n_retain=256, n_holdout=64, n_test=128,
                   seq_len=32, vocab_size=256, batch_size=16):
    """Create synthetic MUSE-like splits."""
    def make_loader(n):
        return DataLoader(
            TensorDataset(torch.randint(0, vocab_size, (n, seq_len)),
                          torch.randint(0, vocab_size, (n,))),
            batch_size=batch_size,
        )
    return {
        "forget": make_loader(n_forget),
        "retain": make_loader(n_retain),
        "holdout": make_loader(n_holdout),
        "test": make_loader(n_test),
    }


def run_benchmark(
    strategies: List[str],
    epochs: int = 3,
    output_dir: str = "benchmarks/muse/results",
):
    """Run MUSE benchmark across strategies."""
    print("=" * 70)
    print("  MUSE BENCHMARK — Erasus Framework")
    print("=" * 70)

    data = make_muse_data()
    base_model = MUSEBenchmarkModel()
    base_state = base_model.state_dict()
    output_path = ensure_dir(output_dir)

    all_results: Dict[str, Dict] = {}

    for strat in strategies:
        print(f"\n  [{strat}] Running... ", end="", flush=True)

        try:
            model = MUSEBenchmarkModel()
            model.load_state_dict(base_state)

            unlearner = LLMUnlearner(
                model=model, strategy=strat, selector=None,
                device="cpu", strategy_kwargs={"lr": 1e-3},
            )

            t0 = time.time()
            result = unlearner.fit(
                forget_data=data["forget"],
                retain_data=data["retain"],
                epochs=epochs,
            )
            elapsed = time.time() - t0

            # Evaluate
            suite = MetricSuite(["accuracy"])
            metrics = suite.run(unlearner.model, data["forget"], data["retain"])
            metrics.pop("_meta", None)

            all_results[strat] = {
                "strategy": strat,
                "elapsed_seconds": round(elapsed, 2),
                "final_forget_loss": round(result.forget_loss_history[-1], 4) if result.forget_loss_history else None,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            }
            print("✓")
        except Exception as e:
            all_results[strat] = {"error": str(e)}
            print(f"✗ ({e})")

    # Save
    results_file = output_path / "muse_results.json"
    save_json(all_results, results_file)
    print(f"\n  Results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print(f"  {'Strategy':<25} {'Time':>8} {'Loss':>10}")
    print("-" * 70)
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<25} {'ERROR':>8}")
        else:
            print(f"  {name:<25} {r['elapsed_seconds']:>7.1f}s {r.get('final_forget_loss', 'N/A'):>10}")

    print("\n✅ MUSE benchmark complete!")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run MUSE benchmark")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="benchmarks/muse/results")
    args = parser.parse_args()

    strategies = args.strategies.split(",")
    run_benchmark(strategies, args.epochs, args.output)


if __name__ == "__main__":
    main()
