"""
erasus.visualization.cross_modal — Cross-modal interference visualization.

Novel research tool that quantifies and visualises how unlearning in one
modality (e.g. vision) affects representations in another (e.g. text),
revealing cross-modal leakage patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CrossModalVisualizer:
    """
    Visualise cross-modal interference during unlearning.

    Measures alignment, drift, and leakage between modalities
    (typically vision ↔ text in VLMs) to understand how
    unlearning in one modality impacts the other.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def compute_modal_drift(
        self,
        model_before: nn.Module,
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Measure per-modality embedding drift after unlearning.

        Returns
        -------
        dict
            Drift statistics for vision, text, and cross-modal alignment.
        """
        device = next(self.model.parameters()).device

        before_vis, before_txt = self._extract_bimodal_embeddings(model_before, data_loader, device)
        after_vis, after_txt = self._extract_bimodal_embeddings(self.model, data_loader, device)

        results: Dict[str, float] = {}

        if before_vis is not None and after_vis is not None:
            vis_drift = np.linalg.norm(after_vis - before_vis, axis=1).mean()
            results["vision_drift"] = float(vis_drift)

        if before_txt is not None and after_txt is not None:
            txt_drift = np.linalg.norm(after_txt - before_txt, axis=1).mean()
            results["text_drift"] = float(txt_drift)

        # Cross-modal alignment (cosine similarity between modalities)
        if before_vis is not None and before_txt is not None:
            results["alignment_before"] = float(self._cosine_alignment(before_vis, before_txt))
        if after_vis is not None and after_txt is not None:
            results["alignment_after"] = float(self._cosine_alignment(after_vis, after_txt))

        if "alignment_before" in results and "alignment_after" in results:
            results["alignment_change"] = results["alignment_after"] - results["alignment_before"]

        return results

    def compute_leakage_score(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Measure cross-modal leakage.

        Checks whether unlearning a concept in one modality leaks
        into the other by comparing embedding similarity shifts.
        """
        device = next(self.model.parameters()).device

        forget_vis, forget_txt = self._extract_bimodal_embeddings(self.model, forget_loader, device)
        retain_vis, retain_txt = self._extract_bimodal_embeddings(self.model, retain_loader, device)

        results: Dict[str, float] = {}

        if forget_vis is not None and forget_txt is not None:
            results["forget_cross_sim"] = float(self._cosine_alignment(forget_vis, forget_txt))

        if retain_vis is not None and retain_txt is not None:
            results["retain_cross_sim"] = float(self._cosine_alignment(retain_vis, retain_txt))

        if "forget_cross_sim" in results and "retain_cross_sim" in results:
            # Leakage = how much the forget concepts still align cross-modally
            results["leakage_score"] = results["forget_cross_sim"] / max(results["retain_cross_sim"], 1e-8)

        return results

    # ------------------------------------------------------------------
    # Plotting methods
    # ------------------------------------------------------------------

    def plot_modal_drift(
        self,
        model_before: nn.Module,
        data_loader: DataLoader,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Visualise per-modality embedding drift as paired bar chart.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        drift = self.compute_modal_drift(model_before, data_loader)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Drift magnitudes
        modalities = []
        drifts = []
        if "vision_drift" in drift:
            modalities.append("Vision")
            drifts.append(drift["vision_drift"])
        if "text_drift" in drift:
            modalities.append("Text")
            drifts.append(drift["text_drift"])

        colours = ["#6366f1", "#f43f5e"]
        axes[0].bar(modalities, drifts, color=colours[:len(modalities)], alpha=0.85)
        axes[0].set_ylabel("Mean L2 Drift")
        axes[0].set_title("Per-Modality Embedding Drift")
        axes[0].grid(axis="y", alpha=0.3)

        # Alignment before/after
        if "alignment_before" in drift and "alignment_after" in drift:
            labels = ["Before", "After"]
            vals = [drift["alignment_before"], drift["alignment_after"]]
            axes[1].bar(labels, vals, color=["#22c55e", "#ef4444"], alpha=0.85)
            axes[1].set_ylabel("Cross-Modal Alignment (Cosine)")
            axes[1].set_title("Cross-Modal Alignment Change")
            axes[1].set_ylim(0, 1)
            axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_leakage_analysis(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Visualise cross-modal leakage between forget and retain sets.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        leakage = self.compute_leakage_score(forget_loader, retain_loader)

        fig, ax = plt.subplots(figsize=(8, 6))

        categories = []
        values = []
        if "forget_cross_sim" in leakage:
            categories.append("Forget\nCross-Modal Sim")
            values.append(leakage["forget_cross_sim"])
        if "retain_cross_sim" in leakage:
            categories.append("Retain\nCross-Modal Sim")
            values.append(leakage["retain_cross_sim"])

        colours = ["#ef4444", "#22c55e"]
        ax.bar(categories, values, color=colours[:len(categories)], alpha=0.85, width=0.5)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Cross-Modal Leakage Analysis")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        if "leakage_score" in leakage:
            ax.annotate(
                f"Leakage Score: {leakage['leakage_score']:.3f}",
                xy=(0.5, 0.95), xycoords="axes fraction", ha="center",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_embedding_space(
        self,
        data_loader: DataLoader,
        method: str = "pca",
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Scatter plot of vision and text embeddings in shared space.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        device = next(self.model.parameters()).device
        vis_embs, txt_embs = self._extract_bimodal_embeddings(self.model, data_loader, device)

        if vis_embs is None or txt_embs is None:
            raise ValueError("Could not extract bimodal embeddings.")

        # Reduce to 2D
        combined = np.concatenate([vis_embs, txt_embs], axis=0)
        reduced = self._reduce_dim(combined, method=method)

        n_vis = len(vis_embs)
        vis_2d = reduced[:n_vis]
        txt_2d = reduced[n_vis:]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(vis_2d[:, 0], vis_2d[:, 1], alpha=0.5, s=15, c="#6366f1", label="Vision")
        ax.scatter(txt_2d[:, 0], txt_2d[:, 1], alpha=0.5, s=15, c="#f43f5e", label="Text")

        # Draw lines connecting paired embeddings
        n_pairs = min(len(vis_2d), len(txt_2d), 50)
        for i in range(n_pairs):
            ax.plot(
                [vis_2d[i, 0], txt_2d[i, 0]],
                [vis_2d[i, 1], txt_2d[i, 1]],
                "gray", alpha=0.1, linewidth=0.5,
            )

        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title("Cross-Modal Embedding Space")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_bimodal_embeddings(
        self, model: nn.Module, loader: DataLoader, device: torch.device
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract vision and text embeddings from a VLM."""
        model.eval()
        vis_embs: list = []
        txt_embs: list = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue

                images, texts = batch[0].to(device), batch[1].to(device)

                # Try CLIP-style API
                if hasattr(model, "encode_image") and hasattr(model, "encode_text"):
                    v = model.encode_image(images)
                    t = model.encode_text(texts)
                elif hasattr(model, "get_image_features") and hasattr(model, "get_text_features"):
                    v = model.get_image_features(images)
                    t = model.get_text_features(texts)
                else:
                    # Generic: just use model output
                    outputs = model(images)
                    v = outputs.logits if hasattr(outputs, "logits") else outputs
                    t = v  # Duplicate for non-VLM models

                vis_embs.append(v.cpu().numpy())
                txt_embs.append(t.cpu().numpy())

        if not vis_embs:
            return None, None

        return np.concatenate(vis_embs, axis=0), np.concatenate(txt_embs, axis=0)

    @staticmethod
    def _cosine_alignment(a: np.ndarray, b: np.ndarray) -> float:
        """Mean pairwise cosine similarity."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        n = min(len(a_norm), len(b_norm))
        sims = (a_norm[:n] * b_norm[:n]).sum(axis=1)
        return float(sims.mean())

    @staticmethod
    def _reduce_dim(data: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
        """Reduce dimensionality for plotting."""
        if method == "pca":
            # Simple PCA via SVD
            data_centered = data - data.mean(axis=0)
            U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
            return U[:, :n_components] * S[:n_components]
        elif method == "tsne":
            try:
                from sklearn.manifold import TSNE
                return TSNE(n_components=n_components, random_state=42).fit_transform(data)
            except ImportError:
                # Fallback to PCA
                data_centered = data - data.mean(axis=0)
                U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
                return U[:, :n_components] * S[:n_components]
        else:
            raise ValueError(f"Unknown method: {method}")
