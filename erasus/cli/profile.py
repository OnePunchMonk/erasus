"""
``erasus profile`` — quick timing / memory breakdown for unlearning phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write profiler report JSON to this path.",
    )


def run_profile(args: argparse.Namespace) -> None:
    import erasus.strategies  # noqa: F401
    from erasus.core.registry import selector_registry, strategy_registry
    from erasus.utils.profiling import UnlearningProfiler

    in_dim, n_classes = 32, 10
    n_forget, n_retain = 64, 128
    model = nn.Sequential(
        nn.Linear(in_dim, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes),
    )
    forget_loader = DataLoader(
        TensorDataset(
            torch.randn(n_forget, in_dim),
            torch.randint(0, n_classes, (n_forget,)),
        ),
        batch_size=16,
    )
    retain_loader = DataLoader(
        TensorDataset(
            torch.randn(n_retain, in_dim),
            torch.randint(0, n_classes, (n_retain,)),
        ),
        batch_size=16,
    )

    profiler = UnlearningProfiler(enable_cuda=torch.cuda.is_available())
    report: dict[str, Any] = {}

    with profiler.profile("selector"):
        sel = selector_registry.get("random")()
        _ = sel.select(model=model, data_loader=forget_loader, k=min(8, n_forget))

    with profiler.profile("forward_backward_unlearning"):
        strat = strategy_registry.get("gradient_ascent")(lr=1e-3)
        model, fl, rl = strat.unlearn(
            model, forget_loader, retain_loader, epochs=1,
        )

    profiler.log_memory("after_unlearning")
    profiler.count_parameters(model, "unlearned_model")
    report = profiler.get_report()
    print(profiler.summary())

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  JSON report: {args.output_json}")
