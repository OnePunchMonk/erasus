"""
TOFU Benchmark — All 27 Strategies.

Runs every registered unlearning strategy through a synthetic TOFU-style
benchmark with a simple neural network, collects metrics, and generates
a leaderboard markdown file.

Usage::

    python benchmarks/tofu/run_all_strategies.py
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Register all strategies
import erasus.strategies  # noqa: F401
from erasus.core.registry import strategy_registry
from erasus.metrics.metric_suite import MetricSuite
from erasus.utils.helpers import ensure_dir, save_json


# ── Strategy metadata ─────────────────────────────────────────────────
# Extra kwargs for strategies that need them on the benchmark model
STRATEGY_KWARGS = {
    "ensemble": {"strategy_names": ["gradient_ascent", "negative_gradient"]},
    "lora": {"target_modules": ["net.0", "net.2", "net.4"]},  # Linear layers in Sequential
}

STRATEGY_CATEGORIES = {
    "gradient_ascent": "Gradient",
    "negative_gradient": "Gradient",
    "scrub": "Gradient",
    "fisher_forgetting": "Gradient",
    "modality_decoupling": "Gradient",
    "saliency_unlearning": "Gradient",
    "lora": "Parameter",
    "sparse_aware": "Parameter",
    "mask_based": "Parameter",
    "neuron_pruning": "Parameter",
    "layer_freezing": "Parameter",
    "amnesiac": "Data",
    "sisa": "Data",
    "certified_removal": "Data",
    "knowledge_distillation": "Data",
    "ssd": "LLM",
    "token_masking": "LLM",
    "embedding_alignment": "LLM",
    "causal_tracing": "LLM",
    "attention_surgery": "LLM",
    "concept_erasure": "Diffusion",
    "noise_injection": "Diffusion",
    "unet_surgery": "Diffusion",
    "timestep_masking": "Diffusion",
    "safe_latents": "Diffusion",
    "contrastive_unlearning": "VLM",
    "attention_unlearning": "VLM",
    "vision_text_split": "VLM",
    "ensemble": "Ensemble",
}


# ── Benchmark model ───────────────────────────────────────────────────
class BenchmarkModel(nn.Module):
    """
    Universal benchmark model compatible with LLM/VLM/Diffusion strategy interfaces.
    Supports output_hidden_states, labels, and dual-input (for contrastive).
    """

    def __init__(self, in_dim: int = 32, hidden: int = 64, n_classes: int = 10):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
        # Shims for modality_decoupling (shared encoder for both modalities)
        self._encoder = nn.Sequential(self.net[0], self.net[1], self.net[2])
        self.vision_model = self._encoder
        self.text_model = self._encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # CLIP-style
        # For diffusion strategies: expose unet-like component
        self.unet = self._encoder
        self.scheduler = None  # concept_erasure checks this; will fallback
        self.text_encoder = None
        self.tokenizer = None

    def forward(self, x, y=None, output_hidden_states: bool = False, labels=None, **kwargs):
        # Dual-input for contrastive_unlearning: model(images, texts)
        # When y is 1D (labels) or incompatible, use x for both modalities
        if y is not None and isinstance(y, torch.Tensor) and x.dim() == 2:
            y_in = y if (y.dim() == 2 and y.shape == x.shape) else x
            emb1 = F.normalize(self._encoder(x), dim=-1)
            emb2 = F.normalize(self._encoder(y_in), dim=-1)
            scale = self.logit_scale.exp().clamp(max=100)
            logits_per_image = scale * (emb1 @ emb2.T)
            logits_per_text = logits_per_image.T
            return type("Out", (), {"logits_per_image": logits_per_image, "logits_per_text": logits_per_text})()
        if output_hidden_states:
            h0 = self.net[0](x)
            h1 = F.relu(h0)
            h2 = self.net[2](h1)
            h3 = F.relu(h2)
            logits = self.net[4](h3)
            hidden_states = [h0, h1, h2, h3]
            out = type("Out", (), {"logits": logits, "hidden_states": hidden_states})()
            if labels is not None:
                out.loss = F.cross_entropy(logits, labels)
            return out
        logits = self.net(x)
        if labels is not None:
            return type("Out", (), {"logits": logits, "loss": F.cross_entropy(logits, labels)})()
        return logits

    def get_image_features(self, pixel_values=None, **kwargs):
        if pixel_values is None:
            return None
        return F.normalize(self._encoder(pixel_values), dim=-1)

    def get_text_features(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            return None
        return F.normalize(self._encoder(input_ids), dim=-1)


# ── Data creation ─────────────────────────────────────────────────────
def make_data(
    n_forget: int = 200,
    n_retain: int = 800,
    in_dim: int = 32,
    n_classes: int = 10,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Create synthetic forget/retain data."""
    forget_x = torch.randn(n_forget, in_dim)
    forget_y = torch.randint(0, n_classes, (n_forget,))
    retain_x = torch.randn(n_retain, in_dim)
    retain_y = torch.randint(0, n_classes, (n_retain,))

    forget_loader = DataLoader(TensorDataset(forget_x, forget_y), batch_size=batch_size)
    retain_loader = DataLoader(TensorDataset(retain_x, retain_y), batch_size=batch_size)
    return forget_loader, retain_loader


# ── Accuracy helper ───────────────────────────────────────────────────
@torch.no_grad()
def compute_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# ── Main benchmark ────────────────────────────────────────────────────
def run_all_strategies():
    """Run every registered strategy and collect results."""
    print("=" * 70)
    print("  TOFU BENCHMARK — All 27 Strategies — Erasus Framework")
    print("=" * 70)

    # Config
    IN_DIM = 32
    HIDDEN = 64
    N_CLASSES = 10
    EPOCHS = 3
    LR = 1e-3

    forget_loader, retain_loader = make_data(in_dim=IN_DIM, n_classes=N_CLASSES)

    # Get all registered strategy names 
    all_strategies = strategy_registry.list()
    print(f"\n  Found {len(all_strategies)} registered strategies:")
    for i, s in enumerate(all_strategies, 1):
        cat = STRATEGY_CATEGORIES.get(s, "Unknown")
        print(f"    {i:2d}. {s} ({cat})")
    print()

    # Create a fresh base model for weight initialization reference
    base_model = BenchmarkModel(in_dim=IN_DIM, hidden=HIDDEN, n_classes=N_CLASSES)

    # Evaluate base model accuracy before unlearning
    base_forget_acc = compute_accuracy(base_model, forget_loader)
    base_retain_acc = compute_accuracy(base_model, retain_loader)
    print(f"  Base model — Forget Acc: {base_forget_acc:.4f}, Retain Acc: {base_retain_acc:.4f}\n")

    results: Dict[str, Dict[str, Any]] = {}

    for idx, strat_name in enumerate(all_strategies, 1):
        category = STRATEGY_CATEGORIES.get(strat_name, "Unknown")
        print(f"  [{idx:2d}/{len(all_strategies)}] {strat_name} ({category})... ", end="", flush=True)

        # Fresh model copy for each strategy
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

            # Compute accuracies after unlearning
            forget_acc = compute_accuracy(model, forget_loader)
            retain_acc = compute_accuracy(model, retain_loader)

            final_forget_loss = forget_losses[-1] if forget_losses else None
            final_retain_loss = retain_losses[-1] if retain_losses else None

            results[strat_name] = {
                "rank": 0,  # filled later
                "strategy": strat_name,
                "category": category,
                "status": "OK",
                "time_s": round(elapsed, 3),
                "final_forget_loss": round(final_forget_loss, 4) if final_forget_loss is not None else None,
                "final_retain_loss": round(final_retain_loss, 4) if final_retain_loss is not None else None,
                "forget_accuracy": round(forget_acc, 4),
                "retain_accuracy": round(retain_acc, 4),
                "forget_loss_delta": round(
                    (final_forget_loss - (forget_losses[0] if forget_losses else 0)), 4
                ) if final_forget_loss is not None and forget_losses else None,
                "epochs": EPOCHS,
            }
            print(f"[OK] ({elapsed:.2f}s)  Forget Acc: {forget_acc:.4f}  Retain Acc: {retain_acc:.4f}")

        except Exception as e:
            elapsed = 0.0
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
                "forget_loss_delta": None,
                "epochs": EPOCHS,
            }
            error_msg = str(e)[:80]
            print(f"[FAIL] ({error_msg})")

    return results, base_forget_acc, base_retain_acc


# ── Ranking ───────────────────────────────────────────────────────────
def rank_results(results: Dict[str, Dict]) -> List[Dict]:
    """Rank strategies: successful ones first by forget_accuracy (lower=better unlearning)."""
    ok = [r for r in results.values() if r["status"] == "OK"]
    err = [r for r in results.values() if r["status"] == "ERROR"]

    # Lower forget accuracy = better unlearning. Tie-break by higher retain accuracy.
    ok.sort(key=lambda r: (r["forget_accuracy"], -(r["retain_accuracy"] or 0)))

    for i, r in enumerate(ok, 1):
        r["rank"] = i
    for r in err:
        r["rank"] = "-"

    return ok + err


# ── Leaderboard markdown ─────────────────────────────────────────────
def generate_leaderboard(
    ranked: List[Dict],
    base_forget_acc: float,
    base_retain_acc: float,
    output_path: Path,
) -> None:
    """Generate the leaderboard markdown file."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ok_count = sum(1 for r in ranked if r["status"] == "OK")
    err_count = sum(1 for r in ranked if r["status"] == "ERROR")

    lines = [
        "# 🏆 TOFU Benchmark Leaderboard",
        "",
        f"> **Generated**: {now}  ",
        f"> **Framework**: Erasus  ",
        f"> **Model**: BenchmarkModel (Linear→ReLU→Linear→ReLU→Linear, in=32, hidden=64, classes=10)  ",
        f"> **Data**: Synthetic (200 forget / 800 retain samples, batch=32)  ",
        f"> **Epochs**: 3 | **LR**: 1e-3  ",
        f"> **Base Model Accuracy**: Forget {base_forget_acc:.4f} / Retain {base_retain_acc:.4f}  ",
        f"> **Strategies Tested**: {ok_count} succeeded, {err_count} errored (out of {ok_count + err_count} total)",
        "",
        "## Ranking Criteria",
        "",
        "Strategies are ranked by **forget accuracy** (lower = better unlearning), with ties broken by **retain accuracy** (higher = better utility preservation).",
        "",
    ]

    # Successful strategies table
    lines.append("## Leaderboard")
    lines.append("")
    lines.append(
        "| Rank | Strategy | Category | Time (s) | Forget Loss | Retain Loss | Forget Acc ↓ | Retain Acc ↑ |"
    )
    lines.append(
        "|------|----------|----------|----------|-------------|-------------|--------------|--------------|"
    )

    for r in ranked:
        if r["status"] != "OK":
            continue
        rank = r["rank"]
        # Medal for top 3
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
        rank_str = f"{medal} {rank}" if medal else str(rank)

        fl = f"{r['final_forget_loss']:.4f}" if r["final_forget_loss"] is not None else "—"
        rl = f"{r['final_retain_loss']:.4f}" if r["final_retain_loss"] is not None else "—"
        fa = f"{r['forget_accuracy']:.4f}" if r["forget_accuracy"] is not None else "—"
        ra = f"{r['retain_accuracy']:.4f}" if r["retain_accuracy"] is not None else "—"

        lines.append(
            f"| {rank_str} | **{r['strategy']}** | {r['category']} | {r['time_s']:.3f} | {fl} | {rl} | {fa} | {ra} |"
        )

    # Errored strategies table
    err_entries = [r for r in ranked if r["status"] == "ERROR"]
    if err_entries:
        lines.append("")
        lines.append("## ⚠️ Strategies That Require Specialized Models")
        lines.append("")
        lines.append(
            "These strategies errored because they require specialized model architectures "
            "(e.g., diffusion UNet, CLIP encoder, PEFT/LoRA support) not present in the simple benchmark model."
        )
        lines.append("")
        lines.append("| Strategy | Category | Error |")
        lines.append("|----------|----------|-------|")
        for r in err_entries:
            err_msg = r.get("error", "Unknown error")
            # Truncate long error messages
            if len(err_msg) > 100:
                err_msg = err_msg[:97] + "..."
            lines.append(f"| **{r['strategy']}** | {r['category']} | {err_msg} |")

    # Summary
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if ok_count > 0:
        best = next(r for r in ranked if r["status"] == "OK")
        lines.append(f"- **Best unlearning**: `{best['strategy']}` (Forget Acc: {best['forget_accuracy']:.4f})")

        # Best utility preservation (highest retain acc among OK)
        best_retain = max(
            (r for r in ranked if r["status"] == "OK"),
            key=lambda r: r["retain_accuracy"] or 0,
        )
        lines.append(
            f"- **Best utility preservation**: `{best_retain['strategy']}` (Retain Acc: {best_retain['retain_accuracy']:.4f})"
        )

        # Fastest
        fastest = min(
            (r for r in ranked if r["status"] == "OK"),
            key=lambda r: r["time_s"],
        )
        lines.append(f"- **Fastest**: `{fastest['strategy']}` ({fastest['time_s']:.3f}s)")

    lines.append(f"- **Total strategies**: {ok_count + err_count}")
    lines.append(f"- **Successful runs**: {ok_count}")
    lines.append(f"- **Errored (specialized model required)**: {err_count}")
    lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    print(f"\n  Leaderboard saved to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────
def main():
    results, base_forget_acc, base_retain_acc = run_all_strategies()

    ranked = rank_results(results)

    # Save raw JSON
    output_dir = ensure_dir(Path("benchmarks/tofu/results"))
    json_path = output_dir / "all_strategies.json"
    save_json(results, json_path)
    print(f"\n  Raw results saved to: {json_path}")

    # Generate leaderboard
    leaderboard_path = Path("benchmarks/tofu/TOFU_LEADERBOARD.md")
    generate_leaderboard(ranked, base_forget_acc, base_retain_acc, leaderboard_path)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  {'#':<4} {'Strategy':<28} {'Cat':<10} {'Time':>7} {'F.Acc':>7} {'R.Acc':>7} {'Status':>8}")
    print("-" * 70)
    for r in ranked:
        if r["status"] == "OK":
            print(
                f"  {r['rank']:<4} {r['strategy']:<28} {r['category']:<10} "
                f"{r['time_s']:>6.2f}s {r['forget_accuracy']:>6.4f} {r['retain_accuracy']:>6.4f} {'OK':>8}"
            )
        else:
            print(f"  {'-':<4} {r['strategy']:<28} {r['category']:<10} {'—':>7} {'—':>7} {'—':>7} {'ERROR':>8}")

    print("\nTOFU benchmark complete!")


if __name__ == "__main__":
    main()
