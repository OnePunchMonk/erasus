"""
erasus.cli.benchmark — ``erasus benchmark`` command.

Runs standardized benchmarks for evaluating unlearning methods
across multiple strategies, selectors, and metrics.

Usage::

    erasus benchmark --protocol tofu --strategies gradient_ascent,fisher_forgetting
    erasus benchmark --protocol tofu --gold-model retrained.pt --n-runs 5
    erasus benchmark --strategies gradient_ascent --selectors influence --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add benchmark-specific arguments."""
    parser.add_argument(
        "--benchmark", type=str, default=None,
        choices=["tofu", "wmdp", "custom"],
        help="(Legacy) Benchmark suite to run. Prefer --protocol.",
    )
    parser.add_argument(
        "--protocol", type=str, default=None,
        choices=["tofu", "muse", "wmdp", "general"],
        help="Named evaluation protocol (tofu, muse, wmdp, general).",
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
        help="Comma-separated metrics to evaluate (used in legacy mode).",
    )

    # New UnlearningBenchmark protocol args
    protocol_group = parser.add_argument_group("protocol evaluation")
    protocol_group.add_argument(
        "--gold-model", type=str, default=None,
        help="Path to gold-standard (retrained) model checkpoint for comparison.",
    )
    protocol_group.add_argument(
        "--n-runs", type=int, default=1,
        help="Number of evaluation runs for confidence intervals (default: 1).",
    )
    protocol_group.add_argument(
        "--confidence-level", type=float, default=0.95,
        help="Confidence level for intervals (default: 0.95).",
    )


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute the benchmark."""
    # Resolve protocol: --protocol takes precedence, then --benchmark, then default
    protocol = args.protocol or args.benchmark

    if protocol:
        _run_protocol_benchmark(args, protocol)
    else:
        _run_legacy_benchmark(args)


def _run_protocol_benchmark(args: argparse.Namespace, protocol: str) -> None:
    """Run benchmark using the UnlearningBenchmark protocol system."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from erasus.core.registry import strategy_registry
    from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

    strategies = [s.strip() for s in args.strategies.split(",")]

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Erasus Benchmark: {protocol:<25} ║")
    print(f"╚══════════════════════════════════════════════╝")
    print(f"  Protocol : {protocol}")
    print(f"  Strategies: {strategies}")
    print(f"  N-runs   : {args.n_runs}")
    print(f"  CI level : {args.confidence_level}")
    if args.gold_model:
        print(f"  Gold model: {args.gold_model}")
    print()

    # Create synthetic benchmark data (real dataset loading deferred to future)
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

    # Load gold model if specified
    gold_model = None
    if args.gold_model:
        gold_path = Path(args.gold_model)
        if gold_path.exists():
            gold_model = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes),
            )
            gold_model.load_state_dict(torch.load(gold_path, map_location="cpu"))
            print(f"  ✓ Gold model loaded from: {args.gold_model}")
        else:
            print(f"  ⚠ Gold model not found: {args.gold_model}")

    all_results = {}

    for strat_name in strategies:
        print(f"  Running: {strat_name}...", end=" ", flush=True)

        model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

        try:
            # Unlearn first
            strategy_cls = strategy_registry.get(strat_name)
            strategy = strategy_cls(lr=1e-3)
            t0 = time.perf_counter()
            model, f_losses, r_losses = strategy.unlearn(
                model, forget_loader, retain_loader, epochs=args.epochs,
            )
            elapsed = time.perf_counter() - t0

            # Evaluate with protocol
            benchmark = UnlearningBenchmark(
                protocol=protocol,
                n_runs=args.n_runs,
                confidence_level=args.confidence_level,
            )
            report = benchmark.evaluate(
                unlearned_model=model,
                forget_data=forget_loader,
                retain_data=retain_loader,
                gold_model=gold_model,
            )

            all_results[strat_name] = {
                "strategy": strat_name,
                "protocol": protocol,
                "time_s": round(elapsed, 3),
                "verdict": report.verdict,
                "metrics": {
                    name: {
                        "mean": mr.mean,
                        "passed": mr.passed,
                    }
                    for name, mr in report.metric_results.items()
                },
            }
            print(f"✓ ({elapsed:.2f}s) — {report.verdict}")

            # Print the full report summary
            print(report.summary())

        except Exception as e:
            all_results[strat_name] = {"error": str(e)}
            print(f"✗ ({e})")

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to {args.output}")

    # Print summary table
    print(f"\n  {'Strategy':<30} {'Time':>8} {'Verdict':>10}")
    print(f"  {'─' * 50}")
    for key, val in all_results.items():
        if "error" in val:
            print(f"  {key:<30} {'—':>8} {'ERROR':>10}")
        else:
            print(f"  {key:<30} {val['time_s']:>7.2f}s {val['verdict']:>10}")


def _run_legacy_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark using the legacy strategy-grid approach."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from erasus.core.registry import strategy_registry
    from erasus.metrics import MetricSuite

    strategies = [s.strip() for s in args.strategies.split(",")]
    selectors = [s.strip() for s in args.selectors.split(",")]
    metric_names = [m.strip() for m in args.metrics.split(",")]

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Erasus Benchmark (legacy mode)              ║")
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
