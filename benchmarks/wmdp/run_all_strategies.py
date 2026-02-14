"""
WMDP Benchmark — All Strategies.

Runs every registered unlearning strategy through a synthetic WMDP-style
benchmark (Weapons of Mass Destruction Proxy — hazardous knowledge removal).
Mirrors TOFU structure. Runs per subset (bio, cyber).

Usage::

    python benchmarks/wmdp/run_all_strategies.py
    python benchmarks/wmdp/run_all_strategies.py --subsets bio
"""

from __future__ import annotations

import argparse
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


# WMDP-style: 4-way multiple choice, 48 forget, 128 retain per subset
WMDP_N_FORGET = 48
WMDP_N_RETAIN = 128
IN_DIM = 32
HIDDEN = 64
N_CLASSES = 4  # 4-way MC
EPOCHS = 3
LR = 1e-3
DEFAULT_SUBSETS = ["bio", "cyber"]


def run_all_strategies_for_subset(
    subset: str,
):
    """Run every registered strategy for one WMDP subset."""
    print("=" * 70)
    print(f"  WMDP BENCHMARK — {subset.upper()} — All Strategies")
    print("=" * 70)

    forget_loader, retain_loader = make_data(
        n_forget=WMDP_N_FORGET,
        n_retain=WMDP_N_RETAIN,
        in_dim=IN_DIM,
        n_classes=N_CLASSES,
    )

    all_strategies = strategy_registry.list()
    print(f"\n  Found {len(all_strategies)} registered strategies")
    print(f"  Data: {WMDP_N_FORGET} forget / {WMDP_N_RETAIN} retain, 4-way MC\n")

    base_model = BenchmarkModel(in_dim=IN_DIM, hidden=HIDDEN, n_classes=N_CLASSES)
    base_forget_acc = compute_accuracy(base_model, forget_loader)
    base_retain_acc = compute_accuracy(base_model, retain_loader)
    print(f"  Base model — Forget Acc: {base_forget_acc:.4f}, Retain Acc: {base_retain_acc:.4f}\n")

    results: Dict[str, Dict[str, Any]] = {}

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WMDP benchmark (all strategies)")
    parser.add_argument("--subsets", default=",".join(DEFAULT_SUBSETS), help="Comma-separated subsets")
    args = parser.parse_args()
    subsets = [s.strip() for s in args.subsets.split(",")]

    script_dir = Path(__file__).resolve().parent
    out_dir = ensure_dir(script_dir / "results")

    all_results: Dict[str, Dict] = {}

    for subset in subsets:
        results, base_forget_acc, base_retain_acc = run_all_strategies_for_subset(subset)
        ranked = rank_results(results)

        # Save per-subset JSON
        json_path = out_dir / f"all_strategies_{subset}.json"
        save_json(results, json_path)
        print(f"\n  Raw results saved to: {json_path}")

        # Per-subset leaderboard
        leaderboard_path = script_dir / f"WMDP_LEADERBOARD_{subset.upper()}.md"
        generate_leaderboard(
            ranked,
            base_forget_acc,
            base_retain_acc,
            leaderboard_path,
            title=f"WMDP Benchmark Leaderboard ({subset.upper()})",
            model_desc="BenchmarkModel (32→64→64→4)",
            data_desc=f"Synthetic WMDP-style ({WMDP_N_FORGET} forget / {WMDP_N_RETAIN} retain, 4-way MC)",
        )

        all_results[subset] = results

        print("\n" + "=" * 70)
        print(f"  {subset.upper()} — {'#':<4} {'Strategy':<28} {'Cat':<10} {'Time':>7} {'F.Acc':>7} {'R.Acc':>7} {'Status':>8}")
        print("-" * 70)
        for r in ranked:
            if r["status"] == "OK":
                print(f"  {subset:5} {r['rank']:<4} {r['strategy']:<28} {r['category']:<10} "
                      f"{r['time_s']:>6.2f}s {r['forget_accuracy']:>6.4f} {r['retain_accuracy']:>6.4f} {'OK':>8}")
            else:
                print(f"  {subset:5} {'-':<4} {r['strategy']:<28} {r['category']:<10} {'—':>7} {'—':>7} {'—':>7} {'ERROR':>8}")

    # Combined results
    save_json(all_results, out_dir / "wmdp_all_results.json")
    print(f"\n  Combined results: {out_dir / 'wmdp_all_results.json'}")
    print("\nWMDP benchmark complete!")


if __name__ == "__main__":
    main()
