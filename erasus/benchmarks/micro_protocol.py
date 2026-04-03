"""
Lightweight protocol runs for the standard benchmark suite (no external data).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import erasus.strategies  # noqa: F401 — register strategies
from erasus.core.registry import strategy_registry
from erasus.evaluation.benchmark_protocol import UnlearningBenchmark


def run_micro_protocol(
    protocol: str,
    *,
    epochs: int = 1,
    strategy_name: str = "gradient_ascent",
) -> Dict[str, Any]:
    """
    Run a single synthetic unlearning pass and evaluate with ``UnlearningBenchmark``.

    Parameters
    ----------
    protocol
        One of ``tofu``, ``muse``, ``wmdp``, ``general`` (``custom`` maps to ``general``).
    """
    if protocol == "custom":
        protocol = "general"

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

    strategy_cls = strategy_registry.get(strategy_name)
    strategy = strategy_cls(lr=1e-3)
    t0 = time.perf_counter()
    model, fl, rl = strategy.unlearn(
        model, forget_loader, retain_loader, epochs=epochs,
    )
    elapsed = time.perf_counter() - t0

    benchmark = UnlearningBenchmark(protocol=protocol, n_runs=1)
    report = benchmark.evaluate(model, forget_loader, retain_loader)

    return {
        "status": "ok",
        "time_s": round(elapsed, 4),
        "verdict": report.verdict,
        "strategy": strategy_name,
        "protocol": protocol,
        "final_forget_loss": fl[-1] if fl else None,
        "final_retain_loss": rl[-1] if rl else None,
    }
