"""
erasus.visualization.influence_maps — Influence attribution visualization.

Visualises per-sample and per-feature influence scores as heatmaps,
bar charts, and scatter plots to understand which data points and
model components drive forgetting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class InfluenceMapVisualizer:
    """
    Visualise influence attribution maps.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def plot_influence_ranking(
        self,
        scores: np.ndarray,
        top_k: int = 30,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Bar chart of top-k most influential samples.

        Parameters
        ----------
        scores : np.ndarray
            Per-sample influence scores.
        top_k : int
            Number of top samples to show.
        labels : list[str], optional
            Sample labels/descriptions.
        save_path : str, optional
            Path to save figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        sorted_idx = np.argsort(np.abs(scores))[::-1][:top_k]
        top_scores = scores[sorted_idx]

        if labels:
            x_labels = [labels[i] for i in sorted_idx]
        else:
            x_labels = [f"#{i}" for i in sorted_idx]

        colours = ["#ef4444" if s > 0 else "#3b82f6" for s in top_scores]

        fig, ax = plt.subplots(figsize=(max(10, top_k * 0.5), 6))
        ax.barh(range(len(top_scores)), top_scores, color=colours, alpha=0.85)
        ax.set_yticks(range(len(top_scores)))
        ax.set_yticklabels(x_labels, fontsize=8)
        ax.set_xlabel("Influence Score")
        ax.set_title(f"Top-{top_k} Most Influential Samples")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_influence_distribution(
        self,
        scores: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Histogram + KDE of the influence score distribution.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(scores, bins=80, color="#8b5cf6", alpha=0.8, edgecolor="white")
        axes[0].set_xlabel("Influence Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Influence Score Distribution")
        axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)
        axes[0].grid(alpha=0.3)

        # Sorted scores (cumulative influence)
        abs_sorted = np.sort(np.abs(scores))[::-1]
        cumulative = np.cumsum(abs_sorted) / abs_sorted.sum()
        axes[1].plot(cumulative, color="#06b6d4", linewidth=2)
        axes[1].set_xlabel("Sample Rank")
        axes[1].set_ylabel("Cumulative |Influence|")
        axes[1].set_title("Cumulative Influence Concentration")
        axes[1].axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="80%")
        axes[1].axhline(y=0.95, color="orange", linestyle="--", alpha=0.5, label="95%")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_influence_heatmap(
        self,
        influence_matrix: np.ndarray,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Heatmap of pairwise influence (e.g., sample × feature).

        Parameters
        ----------
        influence_matrix : np.ndarray
            2D array of influence values (samples × features or samples × samples).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(influence_matrix, cmap="RdBu_r", aspect="auto",
                        vmin=-np.abs(influence_matrix).max(),
                        vmax=np.abs(influence_matrix).max())

        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=7)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Influence Attribution Heatmap")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_forget_retain_influence(
        self,
        forget_scores: np.ndarray,
        retain_scores: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Scatter plot comparing influence of forget vs. retain samples.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(
            range(len(forget_scores)), np.sort(np.abs(forget_scores))[::-1],
            alpha=0.6, s=10, c="#ef4444", label="Forget",
        )
        ax.scatter(
            range(len(retain_scores)), np.sort(np.abs(retain_scores))[::-1],
            alpha=0.6, s=10, c="#3b82f6", label="Retain",
        )

        ax.set_xlabel("Sample Rank")
        ax.set_ylabel("|Influence Score|")
        ax.set_title("Influence: Forget vs Retain Samples")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
