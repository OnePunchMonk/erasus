"""
erasus.cli.benchmark — ``erasus benchmark`` command.

Runs standardized benchmarks for evaluating unlearning methods
across multiple strategies, selectors, and metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add benchmark-specific arguments."""
    parser.add_argument(
        "--benchmark", type=str, default="tofu",
        choices=["tofu", "wmdp", "custom"],
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--strategies", type=str, default="gradient_ascent",
        help="Comma-separated list of strategies to evaluate.",
    )
    parser.add_argument(
        "--selectors", type=str, default="random",
        help="Comma-separated list of selectors to evaluate.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Unlearning epochs per run.",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Output file for results.",
    )
    parser.add_argument(
        "--metrics", type=str, default="accuracy,mia",
        help="Comma-separated metrics to evaluate.",
    )


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute the benchmark."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from erasus.core.registry import strategy_registry
    from erasus.metrics import MetricSuite

    strategies = [s.strip() for s in args.strategies.split(",")]
    selectors = [s.strip() for s in args.selectors.split(",")]
    metric_names = [m.strip() for m in args.metrics.split(",")]

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Erasus Benchmark: {args.benchmark:<25} ║")
    print(f"╚══════════════════════════════════════════════╝")
    print(f"  Strategies: {strategies}")
    print(f"  Selectors:  {selectors}")
    print(f"  Metrics:    {metric_names}")
    print(f"  Epochs:     {args.epochs}")
    print()

    # Create synthetic benchmark data
    in_dim = 32
    n_classes = 10
    n_forget = 200
    n_retain = 800

    forget_x = torch.randn(n_forget, in_dim)
    forget_y = torch.randint(0, n_classes, (n_forget,))
    retain_x = torch.randn(n_retain, in_dim)
    retain_y = torch.randint(0, n_classes, (n_retain,))

    forget_loader = DataLoader(TensorDataset(forget_x, forget_y), batch_size=32)
    retain_loader = DataLoader(TensorDataset(retain_x, retain_y), batch_size=32)

    results = {}

    for strat_name in strategies:
        for sel_name in selectors:
            key = f"{strat_name}+{sel_name}"
            print(f"  Running: {key}...", end=" ", flush=True)

            model = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes),
            )

            try:
                strategy_cls = strategy_registry.get(strat_name)
                strategy = strategy_cls(lr=1e-3)

                t0 = time.perf_counter()
                model, f_losses, r_losses = strategy.unlearn(
                    model, forget_loader, retain_loader, epochs=args.epochs,
                )
                elapsed = time.perf_counter() - t0

                suite = MetricSuite(metric_names)
                metrics = suite.run(model, forget_loader, retain_loader)

                results[key] = {
                    "strategy": strat_name,
                    "selector": sel_name,
                    "time_s": round(elapsed, 3),
                    "final_forget_loss": f_losses[-1] if f_losses else None,
                    "metrics": metrics,
                }
                print(f"✓ ({elapsed:.2f}s)")

            except Exception as e:
                results[key] = {"error": str(e)}
                print(f"✗ ({e})")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {args.output}")

    # Print summary table
    print(f"\n  {'Method':<30} {'Time':>8} {'Status':>8}")
    print(f"  {'─' * 48}")
    for key, val in results.items():
        if "error" in val:
            print(f"  {key:<30} {'—':>8} {'FAIL':>8}")
        else:
            print(f"  {key:<30} {val['time_s']:>7.2f}s {'OK':>8}")
