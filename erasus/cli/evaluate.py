"""
erasus evaluate — CLI command for post-unlearning evaluation.

Usage::

    erasus evaluate --model checkpoint.pt --forget-dir data/forget --retain-dir data/retain
    erasus evaluate --config configs/default.yaml --checkpoint checkpoint.pt
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
    print(f"  Metrics    : {', '.join(args.metrics)}")
    print("=" * 60)

    # ---- Load metrics ----
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
    print("    from erasus.unlearners import VLMUnlearner")
    print("    unlearner = VLMUnlearner(model=my_model)")
    print("    results = unlearner.evaluate_vlm(forget_loader, retain_loader)")
    print("    print(results)")
    print()

    # ---- Output ----
    if args.output:
        # Write a template results file
        template_results = {m: "pending" for m in args.metrics}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(template_results, f, indent=2)
        print(f"  Template results written to: {args.output}")

    print("\n✅ Evaluation setup validated.")
