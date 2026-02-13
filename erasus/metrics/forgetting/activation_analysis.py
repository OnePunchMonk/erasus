"""
erasus.metrics.forgetting.activation_analysis — Internal activation analysis.

Compares intermediate layer activations before/after unlearning
to verify that forget-associated representations have been disrupted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("activation_analysis")
class ActivationAnalysis(BaseMetric):
    """
    Measure activation drift on forget vs. retain data.

    Higher activation drift on forget data + low drift on retain data
    indicates successful, targeted unlearning.
    """

    def __init__(self, layer_patterns: Optional[List[str]] = None) -> None:
        self.layer_patterns = layer_patterns or ["layer", "block", "encoder"]

    def _extract_activations(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Collect mean activations per layer."""
        model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook_fn(module, inp, out):
                act = out if isinstance(out, torch.Tensor) else out[0]
                if name not in activations:
                    activations[name] = []
                activations[name].append(act.detach().mean(dim=0))
            return hook_fn

        for name, module in model.named_modules():
            if any(p in name.lower() for p in self.layer_patterns):
                hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                model(inputs)

        for h in hooks:
            h.remove()

        # Average across batches
        result = {}
        for name, acts in activations.items():
            result[name] = torch.stack(acts).mean(dim=0)
        return result

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute activation statistics."""
        device = next(model.parameters()).device

        forget_acts = self._extract_activations(model, forget_loader, device)

        results: Dict[str, float] = {}

        # Compute mean activation magnitudes on forget data
        forget_norms = []
        for name, act in forget_acts.items():
            norm = act.norm().item()
            forget_norms.append(norm)
            results[f"forget_activation_norm/{name}"] = norm

        results["mean_forget_activation_norm"] = (
            sum(forget_norms) / len(forget_norms) if forget_norms else 0.0
        )

        if retain_loader is not None:
            retain_acts = self._extract_activations(model, retain_loader, device)
            retain_norms = []
            for name, act in retain_acts.items():
                norm = act.norm().item()
                retain_norms.append(norm)
            results["mean_retain_activation_norm"] = (
                sum(retain_norms) / len(retain_norms) if retain_norms else 0.0
            )

            # Drift ratio: lower forget / higher retain = good unlearning
            if results["mean_retain_activation_norm"] > 0:
                results["activation_drift_ratio"] = (
                    results["mean_forget_activation_norm"]
                    / results["mean_retain_activation_norm"]
                )

        results["n_layers_analysed"] = float(len(forget_acts))
        return results
