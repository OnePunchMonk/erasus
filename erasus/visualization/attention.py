"""
erasus.visualization.attention — Attention heatmap visualization.

Visualizes self-attention and cross-attention weights to understand
which tokens/patches the model focuses on before and after unlearning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class AttentionVisualizer:
    """
    Visualize attention patterns.

    Extracts and plots attention weights from transformer layers
    to compare model focus before vs. after unlearning.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def extract_attentions(
        self,
        inputs: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights via forward hooks.

        Returns
        -------
        dict
            ``{layer_name: attention_weights}`` with shape ``(B, H, S, S)``.
        """
        self.model.eval()
        attentions: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple) and len(out) > 1:
                    # Many attention modules return (output, attn_weights)
                    attn = out[1] if out[1] is not None else None
                    if attn is not None:
                        attentions[name] = attn.detach().cpu()
            return hook_fn

        for idx, (name, module) in enumerate(self.model.named_modules()):
            name_lower = name.lower()
            if any(k in name_lower for k in ("attn", "attention", "self_attn")):
                if layer_indices is None or idx in layer_indices:
                    hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            self.model(inputs)

        for h in hooks:
            h.remove()

        return attentions

    def plot_attention_heatmap(
        self,
        inputs: torch.Tensor,
        layer_name: Optional[str] = None,
        head_idx: int = 0,
        save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot attention heatmap for a specific layer and head.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.
        layer_name : str, optional
            Name of the layer to visualise. If None, uses the first found.
        head_idx : int
            Attention head index.
        save_path : str, optional
            File path to save the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        attentions = self.extract_attentions(inputs)
        if not attentions:
            raise ValueError("No attention weights found. Model may not expose them.")

        if layer_name is None:
            layer_name = list(attentions.keys())[0]

        attn_weights = attentions[layer_name]  # (B, H, S, S)

        # Plot first sample, specified head
        w = attn_weights[0, head_idx].numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(w, cmap="viridis", aspect="auto")
        ax.set_title(f"Attention: {layer_name} (head {head_idx})")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_attention_comparison(
        self,
        inputs: torch.Tensor,
        model_before: nn.Module,
        head_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> Any:
        """Plot side-by-side attention before vs. after unlearning."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        viz_before = AttentionVisualizer(model_before)
        attn_before = viz_before.extract_attentions(inputs)
        attn_after = self.extract_attentions(inputs)

        if not attn_before or not attn_after:
            raise ValueError("No attention weights found.")

        layer = list(attn_before.keys())[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        w_before = attn_before[layer][0, head_idx].numpy()
        w_after = attn_after[layer][0, head_idx].numpy()

        im1 = ax1.imshow(w_before, cmap="viridis", aspect="auto")
        ax1.set_title(f"Before Unlearning (head {head_idx})")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        im2 = ax2.imshow(w_after, cmap="viridis", aspect="auto")
        ax2.set_title(f"After Unlearning (head {head_idx})")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
