"""
erasus.visualization.activation — Layer activation visualization.

Visualises intermediate layer activations to understand how
unlearning affects internal model representations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class ActivationVisualizer:
    """
    Visualise layer activations.

    Extracts and plots activation statistics (norms, distributions,
    heatmaps) from intermediate layers to compare model behaviour
    before and after unlearning.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    target_layers : list[str], optional
        Specific layer names to monitor. If None, auto-detect.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.target_layers = target_layers

    def extract_activations(
        self,
        inputs: torch.Tensor,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations via forward hooks.

        Returns
        -------
        dict
            ``{layer_name: activation_tensor}``.
        """
        self.model.eval()
        targets = layer_names or self.target_layers
        activations: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook_fn(module, inp, out):
                tensor = out if isinstance(out, torch.Tensor) else (
                    out[0] if isinstance(out, tuple) else None
                )
                if tensor is not None:
                    activations[name] = tensor.detach().cpu()
            return hook_fn

        for name, module in self.model.named_modules():
            if targets:
                if name in targets:
                    hooks.append(module.register_forward_hook(make_hook(name)))
            else:
                # Auto-detect: linear, conv, attention layers
                name_lower = name.lower()
                if any(k in name_lower for k in ("linear", "conv", "attn", "mlp", "fc", "dense")):
                    hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            self.model(inputs)

        for h in hooks:
            h.remove()

        return activations

    def plot_activation_norms(
        self,
        inputs: torch.Tensor,
        save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Bar chart of mean activation norms per layer.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        activations = self.extract_activations(inputs)
        if not activations:
            raise ValueError("No activations extracted.")

        names = list(activations.keys())
        norms = [act.float().norm(dim=-1).mean().item() for act in activations.values()]

        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 6))
        bars = ax.bar(range(len(names)), norms, color="#4f46e5", alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(".")[-1] for n in names], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Activation Norm")
        ax.set_title("Layer Activation Norms")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_activation_comparison(
        self,
        inputs: torch.Tensor,
        model_before: nn.Module,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Compare activation norms before and after unlearning.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        viz_before = ActivationVisualizer(model_before, self.target_layers)
        act_before = viz_before.extract_activations(inputs)
        act_after = self.extract_activations(inputs)

        # Find common layers
        common = [k for k in act_before if k in act_after]
        if not common:
            raise ValueError("No common layers found between models.")

        norms_before = [act_before[k].float().norm(dim=-1).mean().item() for k in common]
        norms_after = [act_after[k].float().norm(dim=-1).mean().item() for k in common]

        x = np.arange(len(common))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(common) * 0.8), 6))
        ax.bar(x - width / 2, norms_before, width, label="Before", color="#6366f1", alpha=0.8)
        ax.bar(x + width / 2, norms_after, width, label="After", color="#f43f5e", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([n.split(".")[-1] for n in common], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Activation Norm")
        ax.set_title("Activation Comparison: Before vs After Unlearning")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_activation_distribution(
        self,
        inputs: torch.Tensor,
        layer_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Histogram of activation values for a specific layer.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        activations = self.extract_activations(inputs)
        if not activations:
            raise ValueError("No activations extracted.")

        if layer_name is None:
            layer_name = list(activations.keys())[0]

        act = activations[layer_name].float().flatten().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(act, bins=100, color="#8b5cf6", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Count")
        ax.set_title(f"Activation Distribution: {layer_name}")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
