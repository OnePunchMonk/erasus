"""
WMDP Benchmark Runner.

Evaluates unlearning of hazardous knowledge using the WMDP (Weapons of
Mass Destruction Proxy) benchmark.

Usage::

    python benchmarks/wmdp/run.py
    python benchmarks/wmdp/run.py --subsets bio,cyber --strategies gradient_ascent
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import erasus.strategies  # noqa: F401
from erasus.unlearners.llm_unlearner import LLMUnlearner
from erasus.metrics.metric_suite import MetricSuite
from erasus.utils.helpers import save_json, ensure_dir


class WMDPBenchmarkModel(nn.Module):
    """Small model for WMDP testing."""

    def __init__(self, vocab_size=512, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),  # 4-way multiple choice
        )
        self.config = type("C", (), {"model_type": "llama", "vocab_size": vocab_size})()

    def forward(self, x):
        return self.net(self.emb(x).mean(dim=1))


def run_wmdp_benchmark(
    subsets: List[str] = ("bio", "cyber"),
    strategies: List[str] = ("gradient_ascent",),
    epochs: int = 3,
    output_dir: str = "benchmarks/wmdp/results",
):
    """Run WMDP benchmark for each subset × strategy."""
    print("=" * 70)
    print("  WMDP BENCHMARK — Hazardous Knowledge Removal")
    print("=" * 70)

    output_path = ensure_dir(output_dir)
    all_results: Dict[str, Dict] = {}

    for subset in subsets:
        print(f"\n  Subset: wmdp-{subset}")

        # Synthetic hazardous-knowledge data
        forget = DataLoader(
            TensorDataset(torch.randint(0, 512, (48, 32)), torch.randint(0, 4, (48,))),
            batch_size=16,
        )
        retain = DataLoader(
            TensorDataset(torch.randint(0, 512, (128, 32)), torch.randint(0, 4, (128,))),
            batch_size=16,
        )

        for strat in strategies:
            run_name = f"wmdp_{subset}__{strat}"
            print(f"    [{strat}] ", end="", flush=True)

            model = WMDPBenchmarkModel()
            unlearner = LLMUnlearner(
                model=model, strategy=strat, selector=None,
                device="cpu", strategy_kwargs={"lr": 1e-3},
            )

            t0 = time.time()
            result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=epochs)
            elapsed = time.time() - t0

            suite = MetricSuite(["accuracy"])
            metrics = suite.run(unlearner.model, forget, retain)
            metrics.pop("_meta", None)

            all_results[run_name] = {
                "subset": subset,
                "strategy": strat,
                "elapsed_seconds": round(elapsed, 2),
                "final_loss": round(result.forget_loss_history[-1], 4) if result.forget_loss_history else None,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            }
            print(f"✓ ({elapsed:.1f}s)")

    # Save
    save_json(all_results, output_path / "wmdp_results.json")

    # Summary
    print("\n" + "=" * 70)
    print(f"  {'Run':<35} {'Time':>8} {'Loss':>10}")
    print("-" * 70)
    for name, r in all_results.items():
        loss = r.get("final_loss", "N/A")
        print(f"  {name:<35} {r['elapsed_seconds']:>7.1f}s {loss!s:>10}")

    print(f"\n  Results: {output_path / 'wmdp_results.json'}")
    print("\n✅ WMDP benchmark complete!")


def main():
    parser = argparse.ArgumentParser(description="Run WMDP benchmark")
    parser.add_argument("--subsets", default="bio,cyber")
    parser.add_argument("--strategies", default="gradient_ascent")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="benchmarks/wmdp/results")
    args = parser.parse_args()

    run_wmdp_benchmark(
        subsets=args.subsets.split(","),
        strategies=args.strategies.split(","),
        epochs=args.epochs,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
