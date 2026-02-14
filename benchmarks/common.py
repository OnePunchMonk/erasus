"""
Shared benchmark components for TOFU, MUSE, and WMDP.

Provides universal BenchmarkModel, data creation, accuracy computation,
strategy metadata, and leaderboard generation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Strategy metadata (shared across benchmarks)
STRATEGY_KWARGS: Dict[str, Dict[str, Any]] = {
    "ensemble": {"strategy_names": ["gradient_ascent", "negative_gradient"]},
    "lora": {"target_modules": ["net.0", "net.2", "net.4"]},
}

STRATEGY_CATEGORIES: Dict[str, str] = {
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


class BenchmarkModel(nn.Module):
    """
    Universal benchmark model compatible with LLM/VLM/Diffusion strategy interfaces.
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
        self._encoder = nn.Sequential(self.net[0], self.net[1], self.net[2])
        self.vision_model = self._encoder
        self.text_model = self._encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        self.unet = self._encoder
        self.scheduler = None
        self.text_encoder = None
        self.tokenizer = None

    def forward(self, x, y=None, output_hidden_states: bool = False, labels=None, **kwargs):
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


def rank_results(results: Dict[str, Dict]) -> List[Dict]:
    """Rank strategies by forget_accuracy (lower=better), tie-break by retain_accuracy (higher=better)."""
    ok = [r for r in results.values() if r["status"] == "OK"]
    err = [r for r in results.values() if r["status"] == "ERROR"]
    ok.sort(key=lambda r: (r["forget_accuracy"], -(r["retain_accuracy"] or 0)))
    for i, r in enumerate(ok, 1):
        r["rank"] = i
    for r in err:
        r["rank"] = "-"
    return ok + err


def generate_leaderboard(
    ranked: List[Dict],
    base_forget_acc: float,
    base_retain_acc: float,
    output_path: Path,
    title: str,
    model_desc: str,
    data_desc: str,
) -> None:
    """Generate leaderboard markdown file."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ok_count = sum(1 for r in ranked if r["status"] == "OK")
    err_count = sum(1 for r in ranked if r["status"] == "ERROR")

    lines = [
        f"# {title}",
        "",
        f"> **Generated**: {now}  ",
        f"> **Framework**: Erasus  ",
        f"> **Model**: {model_desc}  ",
        f"> **Data**: {data_desc}  ",
        f"> **Base Model Accuracy**: Forget {base_forget_acc:.4f} / Retain {base_retain_acc:.4f}  ",
        f"> **Strategies Tested**: {ok_count} succeeded, {err_count} errored (out of {ok_count + err_count} total)",
        "",
        "## Ranking Criteria",
        "",
        "Strategies are ranked by **forget accuracy** (lower = better unlearning), with ties broken by **retain accuracy** (higher = better utility preservation).",
        "",
        "## Leaderboard",
        "",
        "| Rank | Strategy | Category | Time (s) | Forget Loss | Retain Loss | Forget Acc ↓ | Retain Acc ↑ |",
        "|------|----------|----------|----------|-------------|-------------|--------------|--------------|",
    ]

    for r in ranked:
        if r["status"] != "OK":
            continue
        rank = r["rank"]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
        rank_str = f"{medal} {rank}" if medal else str(rank)
        fl = f"{r['final_forget_loss']:.4f}" if r.get("final_forget_loss") is not None else "—"
        rl = f"{r['final_retain_loss']:.4f}" if r.get("final_retain_loss") is not None else "—"
        fa = f"{r['forget_accuracy']:.4f}" if r.get("forget_accuracy") is not None else "—"
        ra = f"{r['retain_accuracy']:.4f}" if r.get("retain_accuracy") is not None else "—"
        lines.append(f"| {rank_str} | **{r['strategy']}** | {r['category']} | {r['time_s']:.3f} | {fl} | {rl} | {fa} | {ra} |")

    err_entries = [r for r in ranked if r["status"] == "ERROR"]
    if err_entries:
        lines.extend([
            "",
            "## Strategies That Require Specialized Models",
            "",
            "| Strategy | Category | Error |",
            "|----------|----------|-------|",
        ])
        for r in err_entries:
            err_msg = (r.get("error", "Unknown") or "")[:97] + ("..." if len(str(r.get("error", ""))) > 100 else "")
            lines.append(f"| **{r['strategy']}** | {r['category']} | {err_msg} |")

    lines.extend(["", "## Summary", ""])
    if ok_count > 0:
        best = next(r for r in ranked if r["status"] == "OK")
        best_retain = max((r for r in ranked if r["status"] == "OK"), key=lambda r: r["retain_accuracy"] or 0)
        fastest = min((r for r in ranked if r["status"] == "OK"), key=lambda r: r["time_s"])
        lines.append(f"- **Best unlearning**: `{best['strategy']}` (Forget Acc: {best['forget_accuracy']:.4f})")
        lines.append(f"- **Best utility preservation**: `{best_retain['strategy']}` (Retain Acc: {best_retain['retain_accuracy']:.4f})")
        lines.append(f"- **Fastest**: `{fastest['strategy']}` ({fastest['time_s']:.3f}s)")

    lines.append(f"- **Total strategies**: {ok_count + err_count}")
    lines.append(f"- **Successful runs**: {ok_count}")
    lines.append(f"- **Errored**: {err_count}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Leaderboard saved to: {output_path}")
