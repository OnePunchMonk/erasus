"""
MUSE Benchmark - All Strategies.

Runs every registered unlearning strategy through a synthetic MUSE-style
benchmark (Machine Unlearning Six-way Evaluation). Mirrors TOFU structure.

Usage::

    python benchmarks/muse/run_all_strategies.py
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

import erasus.strategies  # noqa: F401
from erasus.core.registry import strategy_registry
from erasus.utils.helpers import ensure_dir, save_json

# Import shared components from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    STRATEGY_CATEGORIES,
    STRATEGY_KWARGS,
    BenchmarkModel,
    compute_accuracy,
    generate_leaderboard,
    make_data,
    rank_results,
)


# MUSE-style config: 64 forget, 256 retain
MUSE_N_FORGET = 64
MUSE_N_RETAIN = 256
IN_DIM = 32
HIDDEN = 64
N_CLASSES = 10
EPOCHS = 3
LR = 1e-3


def run_all_strategies():
    """Run every registered strategy on MUSE-style data."""
    print("=" * 70)
    print("  MUSE BENCHMARK - All Strategies - Erasus Framework")
    print("=" * 70)

    forget_loader, retain_loader = make_data(
        n_forget=MUSE_N_FORGET,
        n_retain=MUSE_N_RETAIN,
        in_dim=IN_DIM,
        n_classes=N_CLASSES,
    )

    all_strategies = strategy_registry.list()
    print(f"\n  Found {len(all_strategies)} registered strategies")
    print(f"  Data: {MUSE_N_FORGET} forget / {MUSE_N_RETAIN} retain samples\n")

    base_model = BenchmarkModel(in_dim=IN_DIM, hidden=HIDDEN, n_classes=N_CLASSES)
    base_forget_acc = compute_accuracy(base_model, forget_loader)
    base_retain_acc = compute_accuracy(base_model, retain_loader)
    print(f"  Base model - Forget Acc: {base_forget_acc:.4f}, Retain Acc: {base_retain_acc:.4f}\n")

    results = {}

    for idx, strat_name in enumerate(all_strategies, 1):
        category = STRATEGY_CATEGORIES.get(strat_name, "Unknown")
        print(f"  [{idx:2d}/{len(all_strategies)}] {strat_name} ({category})... ", end="", flush=True)

        model = copy.deepcopy(base_model)

        try:
            strategy_cls = strategy_registry.get(strat_name)
            kwargs = {"lr": LR, **(STRATEGY_KWARGS.get(strat_name, {}))}
            strategy = strategy_cls(**kwargs)

            t0 = time.perf_counter()
            model, forget_losses, retain_losses = strategy.unlearn(
                model=model,
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                epochs=EPOCHS,
            )
            elapsed = time.perf_counter() - t0

            forget_acc = compute_accuracy(model, forget_loader)
            retain_acc = compute_accuracy(model, retain_loader)

            final_forget_loss = forget_losses[-1] if forget_losses else None
            final_retain_loss = retain_losses[-1] if retain_losses else None

            results[strat_name] = {
                "rank": 0,
                "strategy": strat_name,
                "category": category,
                "status": "OK",
                "time_s": round(elapsed, 3),
                "final_forget_loss": round(final_forget_loss, 4) if final_forget_loss is not None else None,
                "final_retain_loss": round(final_retain_loss, 4) if final_retain_loss is not None else None,
                "forget_accuracy": round(forget_acc, 4),
                "retain_accuracy": round(retain_acc, 4),
                "epochs": EPOCHS,
            }
            print(f"[OK] ({elapsed:.2f}s)  F.Acc: {forget_acc:.4f}  R.Acc: {retain_acc:.4f}")

        except Exception as e:
            results[strat_name] = {
                "rank": 0,
                "strategy": strat_name,
                "category": category,
                "status": "ERROR",
                "error": str(e),
                "time_s": 0,
                "final_forget_loss": None,
                "final_retain_loss": None,
                "forget_accuracy": None,
                "retain_accuracy": None,
                "epochs": EPOCHS,
            }
            print(f"[FAIL] ({str(e)[:60]})")

    return results, base_forget_acc, base_retain_acc


def main():
    results, base_forget_acc, base_retain_acc = run_all_strategies()
    ranked = rank_results(results)

    script_dir = Path(__file__).resolve().parent
    out_dir = ensure_dir(script_dir / "results")
    json_path = out_dir / "all_strategies.json"
    save_json(results, json_path)
    print(f"\n  Raw results saved to: {json_path}")

    leaderboard_path = script_dir / "MUSE_LEADERBOARD.md"
    generate_leaderboard(
        ranked,
        base_forget_acc,
        base_retain_acc,
        leaderboard_path,
        title="MUSE Benchmark Leaderboard",
        model_desc="BenchmarkModel (32->64->64->10)",
        data_desc=f"Synthetic MUSE-style ({MUSE_N_FORGET} forget / {MUSE_N_RETAIN} retain, batch=32)",
    )

    print("\n" + "=" * 70)
    print(f"  {'#':<4} {'Strategy':<28} {'Cat':<10} {'Time':>7} {'F.Acc':>7} {'R.Acc':>7} {'Status':>8}")
    print("-" * 70)
    for r in ranked:
        if r["status"] == "OK":
            print(f"  {r['rank']:<4} {r['strategy']:<28} {r['category']:<10} "
                  f"{r['time_s']:>6.2f}s {r['forget_accuracy']:>6.4f} {r['retain_accuracy']:>6.4f} {'OK':>8}")
        else:
            print(f"  {'-':<4} {r['strategy']:<28} {r['category']:<10} {'-':>7} {'-':>7} {'-':>7} {'ERROR':>8}")

    print("\nMUSE benchmark complete!")


if __name__ == "__main__":
    main()
