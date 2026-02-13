"""
TOFU Benchmark Runner.

Runs the TOFU (Task of Fictitious Unlearning) benchmark across
multiple strategies and selectors, producing a comprehensive
evaluation report.

Usage::

    python benchmarks/tofu/run.py
    python benchmarks/tofu/run.py --strategies gradient_ascent,scrub --epochs 5
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


DEFAULT_STRATEGIES = ["gradient_ascent", "negative_gradient"]
DEFAULT_SELECTORS = [None, "random"]


def make_synthetic_data(
    n_forget: int = 128,
    n_retain: int = 512,
    seq_len: int = 32,
    vocab_size: int = 256,
    batch_size: int = 32,
):
    """Create synthetic TOFU-like data (random tokens)."""
    forget = DataLoader(
        TensorDataset(
            torch.randint(0, vocab_size, (n_forget, seq_len)),
            torch.randint(0, vocab_size, (n_forget,)),
        ),
        batch_size=batch_size,
    )
    retain = DataLoader(
        TensorDataset(
            torch.randint(0, vocab_size, (n_retain, seq_len)),
            torch.randint(0, vocab_size, (n_retain,)),
        ),
        batch_size=batch_size,
    )
    return forget, retain


class BenchmarkModel(nn.Module):
    """Small model for benchmark testing."""

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


def run_benchmark(
    strategies: List[str],
    selectors: List,
    epochs: int = 3,
    output_dir: str = "benchmarks/tofu/results",
):
    """Run TOFU benchmark across strategy × selector combinations."""
    print("=" * 70)
    print("  TOFU BENCHMARK — Erasus Framework")
    print("=" * 70)

    forget, retain = make_synthetic_data()
    base_model = BenchmarkModel()
    output_path = ensure_dir(output_dir)

    all_results: Dict[str, Dict] = {}

    for strat in strategies:
        for sel in selectors:
            sel_name = sel or "none"
            run_name = f"{strat}__{sel_name}"
            print(f"\n  [{run_name}] Running... ", end="", flush=True)

            try:
                unlearner = LLMUnlearner(
                    model=base_model,
                    strategy=strat,
                    selector=sel,
                    device="cpu",
                    strategy_kwargs={"lr": 1e-3},
                )

                t0 = time.time()
                result = unlearner.fit(
                    forget_data=forget,
                    retain_data=retain,
                    prune_ratio=0.5,
                    epochs=epochs,
                )
                elapsed = time.time() - t0

                # Metrics
                suite = MetricSuite(["accuracy"])
                metrics = suite.run(unlearner.model, forget, retain)
                metrics.pop("_meta", None)

                all_results[run_name] = {
                    "strategy": strat,
                    "selector": sel_name,
                    "elapsed_seconds": round(elapsed, 2),
                    "coreset_size": result.coreset_size,
                    "compression_ratio": round(result.compression_ratio, 3),
                    "final_forget_loss": round(result.forget_loss_history[-1], 4) if result.forget_loss_history else None,
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
                }
                print("✓")
            except Exception as e:
                all_results[run_name] = {"error": str(e)}
                print(f"✗ ({e})")

    # Save results
    results_file = output_path / "tofu_results.json"
    save_json(all_results, results_file)
    print(f"\n  Results saved to: {results_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  {'Run':<30} {'Time':>8} {'Loss':>10} {'Coreset':>8}")
    print("-" * 70)
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<30} {'ERROR':>8}")
        else:
            loss = f"{r.get('final_forget_loss', 'N/A')}"
            print(f"  {name:<30} {r['elapsed_seconds']:>7.1f}s {loss:>10} {r.get('coreset_size', 'N/A'):>8}")

    print("\n✅ TOFU benchmark complete!")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run TOFU benchmark")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--selectors", default="none,random")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="benchmarks/tofu/results")
    args = parser.parse_args()

    strategies = args.strategies.split(",")
    selectors = [None if s == "none" else s for s in args.selectors.split(",")]

    run_benchmark(strategies, selectors, args.epochs, args.output)


if __name__ == "__main__":
    main()
