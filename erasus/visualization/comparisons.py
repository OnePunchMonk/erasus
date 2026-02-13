"""
erasus.visualization.comparisons — Before/after comparison plots.

Generates side-by-side visualizations of model behavior
before and after unlearning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ComparisonVisualizer:
    """
    Before/after unlearning comparison plots.

    Supports: prediction distribution, confidence histograms,
    embedding drift, and metric comparison charts.
    """

    def plot_prediction_shift(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        loader: DataLoader,
        title: str = "Prediction Shift",
        save_path: Optional[str] = None,
    ) -> Any:
        """Plot how predictions change after unlearning."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib and numpy required.")

        device = next(model_after.parameters()).device

        confs_before, confs_after = [], []
        model_before.eval()
        model_after.eval()

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                out_b = model_before(inputs)
                out_a = model_after(inputs)

                logits_b = out_b.logits if hasattr(out_b, "logits") else out_b
                logits_a = out_a.logits if hasattr(out_a, "logits") else out_a

                confs_before.extend(logits_b.softmax(-1).max(-1).values.cpu().tolist())
                confs_after.extend(logits_a.softmax(-1).max(-1).values.cpu().tolist())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(confs_before, bins=30, alpha=0.7, color="#3b82f6", label="Before")
        axes[0].hist(confs_after, bins=30, alpha=0.7, color="#ef4444", label="After")
        axes[0].set_title("Prediction Confidence Distribution")
        axes[0].set_xlabel("Max Softmax Confidence")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        # Scatter: before vs after confidence
        axes[1].scatter(confs_before, confs_after, alpha=0.3, s=8, c="#8b5cf6")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[1].set_title("Confidence: Before vs. After")
        axes[1].set_xlabel("Before Unlearning")
        axes[1].set_ylabel("After Unlearning")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_metric_comparison(
        self,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        title: str = "Metric Comparison",
        save_path: Optional[str] = None,
    ) -> Any:
        """Bar chart comparing metrics before/after unlearning."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib and numpy required.")

        keys = sorted(set(metrics_before) & set(metrics_after))
        before_vals = [metrics_before[k] for k in keys]
        after_vals = [metrics_after[k] for k in keys]

        x = np.arange(len(keys))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(keys) * 1.5), 5))
        bars1 = ax.bar(x - width / 2, before_vals, width, label="Before", color="#3b82f6")
        bars2 = ax.bar(x + width / 2, after_vals, width, label="After", color="#ef4444")

        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.legend()

        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_embedding_drift(
        self,
        embeddings_before: torch.Tensor,
        embeddings_after: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        title: str = "Embedding Drift",
        save_path: Optional[str] = None,
    ) -> Any:
        """Plot per-sample embedding drift magnitude."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib and numpy required.")

        drift = (embeddings_after - embeddings_before).norm(dim=-1).cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 5))

        if labels is not None:
            unique_labels = labels.unique().cpu().tolist()
            for lbl in unique_labels:
                mask = (labels == lbl).cpu().numpy()
                ax.hist(drift[mask], bins=30, alpha=0.6, label=f"Class {lbl}")
            ax.legend()
        else:
            ax.hist(drift, bins=30, alpha=0.7, color="#8b5cf6")

        ax.set_title(title)
        ax.set_xlabel("L2 Embedding Drift")
        ax.set_ylabel("Count")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
