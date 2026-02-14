"""
TOFU Benchmark — Knowledge Distillation × All Coreset Selectors.

Runs the top-performing knowledge_distillation strategy with every available
coreset-driven selection method. Compares forget accuracy, retain accuracy,
and compute time across selectors.

Usage::

    python benchmarks/tofu/run_coreset_comparison.py
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import erasus.strategies  # noqa: F401
import erasus.selectors  # noqa: F401
from erasus.core.registry import selector_registry
from erasus.unlearners import ErasusUnlearner
from erasus.utils.helpers import ensure_dir, save_json

# Reuse benchmark components from run_all_strategies
from run_all_strategies import BenchmarkModel, make_data, compute_accuracy


# Selectors that need special instantiation (e.g. ensemble with sub-selectors)
SELECTOR_KWARGS: Dict[str, Dict[str, Any]] = {
    "voting": {"selector_names": ["gradient_norm", "el2n"]},
    "weighted_fusion": None,  # Requires selector instances; skip or use wrapper
}

# Selectors to skip (require non-standard setup)
SKIP_SELECTORS = {"weighted_fusion"}  # Needs selectors=[...] instances


def run_coreset_benchmark(
    prune_ratio: float = 0.2,
    epochs: int = 3,
    lr: float = 1e-3,
) -> tuple[Dict[str, Dict], float, float]:
    """Run knowledge_distillation with each coreset selector."""
    print("=" * 70)
    print("  TOFU — Knowledge Distillation × All Coreset Selectors")
    print("=" * 70)

    IN_DIM = 32
    HIDDEN = 64
    N_CLASSES = 10

    forget_loader, retain_loader = make_data(in_dim=IN_DIM, n_classes=N_CLASSES)
    base_model = BenchmarkModel(in_dim=IN_DIM, hidden=HIDDEN, n_classes=N_CLASSES)

    base_forget_acc = compute_accuracy(base_model, forget_loader)
    base_retain_acc = compute_accuracy(base_model, retain_loader)
    print(f"\n  Base model — Forget Acc: {base_forget_acc:.4f}, Retain Acc: {base_retain_acc:.4f}")
    print(f"  Prune ratio: {prune_ratio} (coreset = {int(len(forget_loader.dataset) * prune_ratio)} samples)")
    print()

    all_selectors = selector_registry.list()
    # Filter out skip list
    selectors_to_run = [s for s in all_selectors if s not in SKIP_SELECTORS]

    # Add baseline: no selector (full forget set)
    results: Dict[str, Dict[str, Any]] = {}

    for idx, sel_name in enumerate(["full"] + selectors_to_run, 1):
        is_baseline = sel_name == "full"
        label = f"full (no coreset)" if is_baseline else sel_name
        print(f"  [{idx}/{len(selectors_to_run) + 1}] {label}... ", end="", flush=True)

        model = copy.deepcopy(base_model)
        selector_arg = None if is_baseline else sel_name
        selector_kwargs = SELECTOR_KWARGS.get(sel_name, {})

        try:
            unlearner = ErasusUnlearner(
                model=model,
                strategy="knowledge_distillation",
                selector=selector_arg,
                device="cpu",
                strategy_kwargs={"lr": lr},
                selector_kwargs=selector_kwargs if selector_kwargs else None,
            )

            t0 = time.perf_counter()
            result = unlearner.fit(
                forget_data=forget_loader,
                retain_data=retain_loader,
                prune_ratio=prune_ratio if not is_baseline else 1.0,
                epochs=epochs,
            )
            elapsed = time.perf_counter() - t0

            forget_acc = compute_accuracy(unlearner.model, forget_loader)
            retain_acc = compute_accuracy(unlearner.model, retain_loader)

            key = "full (no coreset)" if is_baseline else sel_name
            results[key] = {
                "selector": key,
                "status": "OK",
                "time_s": round(elapsed, 3),
                "forget_accuracy": round(forget_acc, 4),
                "retain_accuracy": round(retain_acc, 4),
                "coreset_size": result.coreset_size,
                "original_forget_size": result.original_forget_size,
                "compression_ratio": round(result.compression_ratio, 4),
            }
            print(f"[OK] {elapsed:.2f}s  F.Acc: {forget_acc:.4f}  R.Acc: {retain_acc:.4f}")

        except Exception as e:
            results[label] = {
                "selector": label,
                "status": "ERROR",
                "error": str(e)[:80],
                "time_s": 0,
                "forget_accuracy": None,
                "retain_accuracy": None,
                "coreset_size": None,
                "original_forget_size": len(forget_loader.dataset),
                "compression_ratio": None,
            }
            print(f"[FAIL] {str(e)[:60]}")

    return results, base_forget_acc, base_retain_acc


def rank_results(results: Dict[str, Dict]) -> List[Dict]:
    """Rank by forget accuracy (lower = better), tie-break by retain accuracy (higher = better)."""
    ok = [r for r in results.values() if r["status"] == "OK"]
    err = [r for r in results.values() if r["status"] == "ERROR"]
    ok.sort(key=lambda r: (r["forget_accuracy"] or 1.0, -(r["retain_accuracy"] or 0)))
    for i, r in enumerate(ok, 1):
        r["rank"] = i
    for r in err:
        r["rank"] = "-"
    return ok + err


def generate_report(
    ranked: List[Dict],
    base_forget_acc: float,
    base_retain_acc: float,
    output_path: Path,
) -> None:
    """Generate comparison markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ok_count = sum(1 for r in ranked if r["status"] == "OK")
    err_count = sum(1 for r in ranked if r["status"] == "ERROR")

    lines = [
        "# Knowledge Distillation × Coreset Selectors",
        "",
        f"> **Generated**: {now}",
        f"> **Strategy**: knowledge_distillation",
        f"> **Base Model**: BenchmarkModel (32→64→64→10)",
        f"> **Data**: 200 forget / 800 retain, prune_ratio=0.2 (coreset=40)",
        f"> **Epochs**: 3 | **LR**: 1e-3",
        f"> **Base Accuracy**: Forget {base_forget_acc:.4f} / Retain {base_retain_acc:.4f}",
        f"> **Results**: {ok_count} OK, {err_count} failed",
        "",
        "## Ranking",
        "",
        "Lower forget accuracy = better unlearning. Higher retain accuracy = better utility.",
        "",
        "| Rank | Selector | Time (s) | Coreset Size | Forget Acc ↓ | Retain Acc ↑ |",
        "|------|----------|----------|--------------|--------------|--------------|",
    ]

    for r in ranked:
        if r["status"] != "OK":
            continue
        rank = r["rank"]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
        rank_str = f"{medal} {rank}" if medal else str(rank)
        cs = r.get("coreset_size", "—")
        fa = f"{r['forget_accuracy']:.4f}" if r["forget_accuracy"] is not None else "—"
        ra = f"{r['retain_accuracy']:.4f}" if r["retain_accuracy"] is not None else "—"
        lines.append(f"| {rank_str} | **{r['selector']}** | {r['time_s']:.3f} | {cs} | {fa} | {ra} |")

    if err_count > 0:
        lines.extend([
            "",
            "## Failed Selectors",
            "",
            "| Selector | Error |",
            "|----------|-------|",
        ])
        for r in ranked:
            if r["status"] == "ERROR":
                err_msg = r.get("error", "Unknown")[:80]
                lines.append(f"| **{r['selector']}** | {err_msg} |")

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Best unlearning**: ",
        f"- **Best utility preservation**: ",
        f"- **Fastest**: ",
    ])

    if ok_count > 0:
        best = next(r for r in ranked if r["status"] == "OK")
        best_retain = max((r for r in ranked if r["status"] == "OK"), key=lambda x: x["retain_accuracy"] or 0)
        fastest = min((r for r in ranked if r["status"] == "OK"), key=lambda x: x["time_s"])
        lines[-4] = f"- **Best unlearning**: `{best['selector']}` (Forget Acc: {best['forget_accuracy']:.4f})"
        lines[-3] = f"- **Best utility preservation**: `{best_retain['selector']}` (Retain Acc: {best_retain['retain_accuracy']:.4f})"
        lines[-2] = f"- **Fastest**: `{fastest['selector']}` ({fastest['time_s']:.3f}s)"

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report saved to: {output_path}")


def main() -> None:
    results, base_forget_acc, base_retain_acc = run_coreset_benchmark(
        prune_ratio=0.2,
        epochs=3,
        lr=1e-3,
    )

    ranked = rank_results(results)

    out_dir = ensure_dir(Path("benchmarks/tofu/results"))
    json_path = out_dir / "coreset_comparison.json"
    save_json(results, json_path)
    print(f"\n  Raw results saved to: {json_path}")

    report_path = Path("benchmarks/tofu/CORESET_COMPARISON.md")
    generate_report(ranked, base_forget_acc, base_retain_acc, report_path)

    print("\n" + "=" * 70)
    print(f"  {'#':<4} {'Selector':<28} {'Time':>7} {'F.Acc':>7} {'R.Acc':>7} {'Status':>8}")
    print("-" * 70)
    for r in ranked:
        if r["status"] == "OK":
            print(f"  {r['rank']:<4} {r['selector']:<28} {r['time_s']:>6.2f}s {r['forget_accuracy']:>6.4f} {r['retain_accuracy']:>6.4f} {'OK':>8}")
        else:
            print(f"  {'-':<4} {r['selector']:<28} {'—':>7} {'—':>7} {'—':>7} {'ERROR':>8}")

    print("\nCoreset comparison complete!")


if __name__ == "__main__":
    main()
