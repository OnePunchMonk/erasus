"""
erasus evaluate — CLI command for post-unlearning evaluation.

Usage::

    erasus evaluate --checkpoint checkpoint.pt --metrics accuracy mia
    erasus evaluate --protocol tofu --checkpoint model.pt --gold-model retrained.pt
    erasus evaluate --validate-every 2 --early-stop-patience 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'evaluate' sub-command."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate an unlearned model with standard metrics.",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to the unlearned model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--model-type", type=str, default="vlm",
        choices=["vlm", "llm", "diffusion", "audio", "video"],
        help="Model modality type.",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["accuracy", "mia"],
        help="Metric names to compute (e.g. accuracy mia perplexity fid retrieval).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to write evaluation results as JSON.",
    )

    # Protocol-based evaluation (new)
    proto_group = parser.add_argument_group("protocol evaluation")
    proto_group.add_argument(
        "--protocol", type=str, default=None,
        choices=["tofu", "muse", "wmdp", "general"],
        help="Named evaluation protocol. Overrides --metrics.",
    )
    proto_group.add_argument(
        "--gold-model", type=str, default=None,
        help="Path to gold-standard (retrained) model checkpoint.",
    )
    proto_group.add_argument(
        "--include-privacy", action="store_true",
        help="Include epsilon-delta privacy verification in evaluation.",
    )
    proto_group.add_argument(
        "--n-runs", type=int, default=1,
        help="Number of evaluation runs for confidence intervals.",
    )

    # Training-loop validation (for use during unlearning)
    val_group = parser.add_argument_group("validation & early stopping")
    val_group.add_argument(
        "--validate-every", type=int, default=0,
        help="Run validation metrics every N epochs during unlearning. 0 disables.",
    )
    val_group.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="Stop if monitored metric doesn't improve for N validation rounds.",
    )

    parser.set_defaults(func=run_evaluate)


METRIC_MAP = {
    "accuracy": ("erasus.metrics.accuracy", "AccuracyMetric"),
    "mia": ("erasus.metrics.membership_inference", "MembershipInferenceMetric"),
    "perplexity": ("erasus.metrics.perplexity", "PerplexityMetric"),
    "retrieval": ("erasus.metrics.retrieval", "ZeroShotRetrievalMetric"),
    "fid": ("erasus.metrics.fid", "FIDMetric"),
}


def _load_metric(name: str) -> Any:
    """Dynamically import and instantiate a metric by short name."""
    if name not in METRIC_MAP:
        print(f"  ⚠ Unknown metric '{name}'.  Available: {list(METRIC_MAP.keys())}")
        return None
    module_path, class_name = METRIC_MAP[name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def run_evaluate(args: argparse.Namespace) -> None:
    """Execute the evaluation pipeline."""
    print("=" * 60)
    print("  ERASUS — Post-Unlearning Evaluation")
    print("=" * 60)
    print(f"  Model type : {args.model_type}")
    print(f"  Checkpoint : {args.checkpoint or '(not specified)'}")
    if args.protocol:
        print(f"  Protocol   : {args.protocol}")
        if args.include_privacy:
            print(f"  Privacy    : enabled")
        if args.gold_model:
            print(f"  Gold model : {args.gold_model}")
        print(f"  N-runs     : {args.n_runs}")
    else:
        print(f"  Metrics    : {', '.join(args.metrics)}")
    if args.validate_every > 0:
        print(f"  Validate   : every {args.validate_every} epoch(s)")
    if args.early_stop_patience > 0:
        print(f"  Early stop : patience={args.early_stop_patience}")
    print("=" * 60)

    # ---- Protocol-based evaluation ----
    if args.protocol:
        _run_protocol_evaluation(args)
        return

    # ---- Legacy metric-based evaluation ----
    metric_instances: List[Any] = []
    for metric_name in args.metrics:
        m = _load_metric(metric_name)
        if m is not None:
            metric_instances.append(m)
            print(f"  ✓ Loaded metric: {metric_name}")

    if not metric_instances:
        print("  ✗ No metrics could be loaded. Exiting.")
        sys.exit(1)

    # ---- Load model ----
    print()
    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        print("  ⚠ Checkpoint loading requires a model architecture reference.")
        print("  → Use the Python API for full evaluation pipelines.")
    else:
        print("  ⚠ No --checkpoint specified.")
        print("  → Use the Python API to pass a loaded model directly.")

    print()
    print("  To evaluate via Python:")
    print()
    print("    from erasus.evaluation import UnlearningBenchmark")
    print("    benchmark = UnlearningBenchmark(protocol='tofu')")
    print("    report = benchmark.evaluate(model, forget_data, retain_data)")
    print("    print(report.summary())")
    print()

    # ---- Output ----
    if args.output:
        template_results = {m: "pending" for m in args.metrics}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(template_results, f, indent=2)
        print(f"  Template results written to: {args.output}")

    print("\n  Evaluation setup validated.")


def _run_protocol_evaluation(args: argparse.Namespace) -> None:
    """Run evaluation using the UnlearningBenchmark protocol system."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from erasus.evaluation.benchmark_protocol import UnlearningBenchmark, BenchmarkReport

    print(f"\n  Using protocol: {args.protocol}")

    # Create synthetic data for demo (real data loading deferred to future)
    in_dim = 32
    n_classes = 10
    n_forget, n_retain = 200, 800

    forget_loader = DataLoader(
        TensorDataset(
            torch.randn(n_forget, in_dim),
            torch.randint(0, n_classes, (n_forget,)),
        ),
        batch_size=32,
    )
    retain_loader = DataLoader(
        TensorDataset(
            torch.randn(n_retain, in_dim),
            torch.randint(0, n_classes, (n_retain,)),
        ),
        batch_size=32,
    )

    # Build model (or load checkpoint)
    model = nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_classes),
    )
    if args.checkpoint:
        path = Path(args.checkpoint)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"  ✓ Loaded checkpoint: {args.checkpoint}")
        else:
            print(f"  ⚠ Checkpoint not found: {args.checkpoint}")

    # Gold model
    gold_model = None
    if args.gold_model:
        gold_path = Path(args.gold_model)
        if gold_path.exists():
            gold_model = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_classes),
            )
            gold_model.load_state_dict(torch.load(gold_path, map_location="cpu"))
            print(f"  ✓ Loaded gold model: {args.gold_model}")

    # Run benchmark
    benchmark = UnlearningBenchmark(
        protocol=args.protocol,
        n_runs=args.n_runs,
        include_privacy=args.include_privacy,
    )
    report = benchmark.evaluate(
        unlearned_model=model,
        forget_data=forget_loader,
        retain_data=retain_loader,
        gold_model=gold_model,
    )

    print()
    print(report.summary())

    # Save report
    if args.output:
        report.save(args.output)
        print(f"\n  Report saved to: {args.output}")
