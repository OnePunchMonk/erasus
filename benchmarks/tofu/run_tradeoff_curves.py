"""
Tradeoff Curve Generator — (Coreset Fraction) vs (Forgetting Quality x Retained Accuracy).

Generates the key figure for the coreset thesis: a plot showing that coreset-based
unlearning dominates naive random deletion at every operating point on the
forgetting-utility tradeoff curve.

Usage::

    python benchmarks/tofu/run_tradeoff_curves.py
    python benchmarks/tofu/run_tradeoff_curves.py --output-format png
    python benchmarks/tofu/run_tradeoff_curves.py --no-plot  # JSON + markdown only

Requires matplotlib for plot generation (optional).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import erasus.strategies  # noqa: F401
import erasus.selectors  # noqa: F401
from erasus.core.registry import selector_registry
from erasus.unlearners import ErasusUnlearner
from erasus.utils.helpers import ensure_dir, save_json

from run_all_strategies import BenchmarkModel, make_data, compute_accuracy


# ── Configuration ────────────────────────────────────────────────────

FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

# Selectors to compare (covering different families)
SELECTORS = {
    "influence": {"family": "Gradient-based", "color": "#e41a1c"},
    "gradient_norm": {"family": "Gradient-based", "color": "#377eb8"},
    "el2n": {"family": "Gradient-based", "color": "#4daf4a"},
    "tracin": {"family": "Gradient-based", "color": "#984ea3"},
    "herding": {"family": "Geometry-based", "color": "#ff7f00"},
    "kcenter": {"family": "Geometry-based", "color": "#a65628"},
    "forgetting_events": {"family": "Learning-based", "color": "#f781bf"},
}

# Random baseline for comparison
RANDOM_SELECTOR = "random"

STRATEGY = "gradient_ascent"

SELECTOR_KWARGS: Dict[str, Optional[Dict[str, Any]]] = {
    "voting": {"selector_names": ["gradient_norm", "el2n"]},
}

SKIP_SELECTORS = {"weighted_fusion"}


def compute_tradeoff_score(forget_acc: float, retain_acc: float) -> float:
    """
    Combined tradeoff score: forgetting quality x retained accuracy.

    Forgetting quality = 1 - forget_accuracy (lower forget acc = better forgetting).
    Score in [0, 1], higher is better.
    """
    forgetting_quality = 1.0 - forget_acc
    return forgetting_quality * retain_acc


def run_selector_sweep(
    base_model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    selector: str,
    fractions: List[float],
    strategy: str = STRATEGY,
    epochs: int = 3,
    lr: float = 1e-3,
) -> List[Dict[str, Any]]:
    """Sweep coreset fractions for a single selector."""
    results = []
    sel_kwargs = SELECTOR_KWARGS.get(selector)

    for frac in fractions:
        model = copy.deepcopy(base_model)

        try:
            unlearner = ErasusUnlearner(
                model=model,
                strategy=strategy,
                selector=selector if frac < 1.0 else None,
                device="cpu",
                strategy_kwargs={"lr": lr},
                selector_kwargs=sel_kwargs,
            )

            t0 = time.perf_counter()
            result = unlearner.fit(
                forget_data=forget_loader,
                retain_data=retain_loader,
                prune_ratio=frac if frac < 1.0 else None,
                epochs=epochs,
            )
            elapsed = time.perf_counter() - t0

            forget_acc = compute_accuracy(unlearner.model, forget_loader)
            retain_acc = compute_accuracy(unlearner.model, retain_loader)
            score = compute_tradeoff_score(forget_acc, retain_acc)

            results.append({
                "fraction": frac,
                "forget_accuracy": round(forget_acc, 4),
                "retain_accuracy": round(retain_acc, 4),
                "tradeoff_score": round(score, 4),
                "time_s": round(elapsed, 3),
                "status": "OK",
            })

        except Exception as e:
            results.append({
                "fraction": frac,
                "forget_accuracy": None,
                "retain_accuracy": None,
                "tradeoff_score": None,
                "time_s": 0,
                "status": "ERROR",
                "error": str(e)[:120],
            })

    return results


def generate_plot(
    all_results: Dict[str, List[Dict]],
    random_results: Optional[List[Dict]],
    output_path: Path,
    output_format: str = "png",
) -> bool:
    """Generate the tradeoff curve plot. Returns False if matplotlib unavailable."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not available -- skipping plot generation")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Forget Accuracy vs Coreset Fraction
    ax1 = axes[0]
    # Plot 2: Retain Accuracy vs Coreset Fraction
    ax2 = axes[1]
    # Plot 3: Tradeoff Score vs Coreset Fraction
    ax3 = axes[2]

    for sel_name, points in all_results.items():
        ok_pts = [p for p in points if p["status"] == "OK"]
        if not ok_pts:
            continue

        fracs = [p["fraction"] for p in ok_pts]
        f_accs = [p["forget_accuracy"] for p in ok_pts]
        r_accs = [p["retain_accuracy"] for p in ok_pts]
        scores = [p["tradeoff_score"] for p in ok_pts]

        color = SELECTORS.get(sel_name, {}).get("color", "#999999")
        label = sel_name

        ax1.plot(fracs, f_accs, "o-", color=color, label=label, markersize=5, linewidth=1.5)
        ax2.plot(fracs, r_accs, "o-", color=color, label=label, markersize=5, linewidth=1.5)
        ax3.plot(fracs, scores, "o-", color=color, label=label, markersize=5, linewidth=1.5)

    # Plot random baseline
    if random_results:
        ok_pts = [p for p in random_results if p["status"] == "OK"]
        if ok_pts:
            fracs = [p["fraction"] for p in ok_pts]
            f_accs = [p["forget_accuracy"] for p in ok_pts]
            r_accs = [p["retain_accuracy"] for p in ok_pts]
            scores = [p["tradeoff_score"] for p in ok_pts]

            ax1.plot(fracs, f_accs, "x--", color="#999999", label="random", markersize=6, linewidth=1.5)
            ax2.plot(fracs, r_accs, "x--", color="#999999", label="random", markersize=6, linewidth=1.5)
            ax3.plot(fracs, scores, "x--", color="#999999", label="random", markersize=6, linewidth=1.5)

    # Formatting
    for ax in axes:
        ax.set_xlabel("Coreset Fraction", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.05)

    ax1.set_ylabel("Forget Accuracy (lower = better)", fontsize=11)
    ax1.set_title("Forgetting Quality vs Coreset Size", fontsize=12, fontweight="bold")

    ax2.set_ylabel("Retain Accuracy (higher = better)", fontsize=11)
    ax2.set_title("Utility Preservation vs Coreset Size", fontsize=12, fontweight="bold")

    ax3.set_ylabel("Tradeoff Score (higher = better)", fontsize=11)
    ax3.set_title("Combined Score vs Coreset Size", fontsize=12, fontweight="bold")

    fig.suptitle(
        f"Coreset Unlearning Tradeoff Curves (strategy: {STRATEGY})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    plot_path = output_path.with_suffix(f".{output_format}")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to: {plot_path}")
    return True


def generate_report(
    all_results: Dict[str, List[Dict]],
    random_results: Optional[List[Dict]],
    metadata: Dict[str, Any],
) -> str:
    """Generate markdown analysis of tradeoff curves."""
    lines = [
        "# Coreset Unlearning Tradeoff Curves",
        "",
        f"> **Generated**: {metadata['timestamp']}",
        f"> **Strategy**: {metadata['strategy']}",
        f"> **Data**: {metadata['n_forget']} forget / {metadata['n_retain']} retain",
        f"> **Fractions swept**: {metadata['fractions']}",
        "",
        "## Tradeoff score = (1 - forget_accuracy) x retain_accuracy",
        "",
        "Higher is better. A perfect unlearner would have forget_accuracy = 0 "
        "and retain_accuracy = 1, giving a score of 1.0.",
        "",
    ]

    # Summary table at 10% operating point
    lines.extend([
        "## Key operating point: 10% coreset",
        "",
        "| Selector | Family | Forget Acc | Retain Acc | Tradeoff Score | Time (s) |",
        "|----------|--------|-----------|-----------|----------------|----------|",
    ])

    for sel_name, points in all_results.items():
        ten_pct = next((p for p in points if abs(p["fraction"] - 0.1) < 0.05 and p["status"] == "OK"), None)
        if ten_pct is None:
            continue
        family = SELECTORS.get(sel_name, {}).get("family", "Other")
        lines.append(
            f"| **{sel_name}** | {family} | {ten_pct['forget_accuracy']:.4f} | "
            f"{ten_pct['retain_accuracy']:.4f} | {ten_pct['tradeoff_score']:.4f} | "
            f"{ten_pct['time_s']:.2f} |"
        )

    if random_results:
        ten_pct_rand = next(
            (p for p in random_results if abs(p["fraction"] - 0.1) < 0.05 and p["status"] == "OK"),
            None,
        )
        if ten_pct_rand:
            lines.append(
                f"| **random** (baseline) | N/A | {ten_pct_rand['forget_accuracy']:.4f} | "
                f"{ten_pct_rand['retain_accuracy']:.4f} | {ten_pct_rand['tradeoff_score']:.4f} | "
                f"{ten_pct_rand['time_s']:.2f} |"
            )

    lines.append("")

    # Full results per selector
    lines.append("## Full sweep results")
    lines.append("")

    for sel_name, points in all_results.items():
        ok_pts = [p for p in points if p["status"] == "OK"]
        if not ok_pts:
            continue

        lines.append(f"### `{sel_name}`")
        lines.append("")
        lines.append("| Fraction | Forget Acc | Retain Acc | Score | Time |")
        lines.append("|----------|-----------|-----------|-------|------|")

        for p in ok_pts:
            lines.append(
                f"| {p['fraction'] * 100:.0f}% | {p['forget_accuracy']:.4f} | "
                f"{p['retain_accuracy']:.4f} | {p['tradeoff_score']:.4f} | {p['time_s']:.2f}s |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "If coreset selectors (influence, gradient_norm, etc.) consistently achieve higher "
        "tradeoff scores than random selection at the same coreset fraction, the coreset thesis "
        "holds: **intelligent sample selection dominates random deletion at every operating point**.",
        "",
        "The practical implication: practitioners can pick their desired coreset fraction based on "
        "compute budget, with the tradeoff curve showing exactly how much forgetting quality they "
        "get per unit of compute.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Coreset unlearning tradeoff curves")
    parser.add_argument("--output-format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 70)
    print("  Coreset Unlearning Tradeoff Curves")
    print("=" * 70)

    IN_DIM = 32
    HIDDEN = 64
    N_CLASSES = 10
    N_FORGET = 200
    N_RETAIN = 800

    forget_loader, retain_loader = make_data(
        n_forget=N_FORGET, n_retain=N_RETAIN, in_dim=IN_DIM, n_classes=N_CLASSES
    )
    base_model = BenchmarkModel(in_dim=IN_DIM, hidden=HIDDEN, n_classes=N_CLASSES)

    base_forget_acc = compute_accuracy(base_model, forget_loader)
    base_retain_acc = compute_accuracy(base_model, retain_loader)
    print(f"\n  Base model -- Forget Acc: {base_forget_acc:.4f}, Retain Acc: {base_retain_acc:.4f}")
    print()

    # Run all selectors
    all_results: Dict[str, List[Dict]] = {}
    selectors_to_run = [s for s in SELECTORS if s not in SKIP_SELECTORS]

    for sel in selectors_to_run:
        if sel not in selector_registry.list():
            print(f"  [{sel}] not registered, skipping")
            continue
        print(f"  Sweeping [{sel}]...")
        all_results[sel] = run_selector_sweep(
            base_model, forget_loader, retain_loader,
            selector=sel, fractions=FRACTIONS,
            epochs=args.epochs, lr=args.lr,
        )
        ok = sum(1 for r in all_results[sel] if r["status"] == "OK")
        print(f"    {ok}/{len(FRACTIONS)} points OK")

    # Random baseline
    print(f"  Sweeping [random] (baseline)...")
    random_results = None
    if "random" in selector_registry.list():
        random_results = run_selector_sweep(
            base_model, forget_loader, retain_loader,
            selector="random", fractions=FRACTIONS,
            epochs=args.epochs, lr=args.lr,
        )

    # Output
    script_dir = Path(__file__).resolve().parent
    out_dir = ensure_dir(script_dir / "results")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "strategy": STRATEGY,
        "n_forget": N_FORGET,
        "n_retain": N_RETAIN,
        "fractions": FRACTIONS,
        "base_forget_accuracy": base_forget_acc,
        "base_retain_accuracy": base_retain_acc,
    }

    # Save raw JSON
    json_data = {"metadata": metadata, "results": all_results, "random_baseline": random_results}
    json_path = out_dir / "tradeoff_curves.json"
    save_json(json_data, json_path)
    print(f"\n  Raw results saved to: {json_path}")

    # Generate report
    report = generate_report(all_results, random_results, metadata)
    report_path = script_dir / "TRADEOFF_CURVES.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to: {report_path}")

    # Generate plot
    if not args.no_plot:
        plot_path = out_dir / "tradeoff_curves"
        generate_plot(all_results, random_results, plot_path, args.output_format)

    print("\nTradeoff curve generation complete!")


if __name__ == "__main__":
    main()
