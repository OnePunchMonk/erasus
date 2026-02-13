"""
erasus.metrics.privacy.differential_privacy — DP-specific evaluation metrics.

Evaluates how well an unlearning procedure satisfies differential
privacy guarantees, including empirical ε estimation and audit tests.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class DPEvaluationMetric(BaseMetric):
    """
    Evaluates differential privacy properties of an unlearned model.

    Metrics computed:
    - Empirical epsilon estimation via hypothesis testing
    - Gradient sensitivity (L2 norm of per-sample gradients)
    - Noise-to-signal ratio of parameter changes
    """

    name = "dp_evaluation"

    def __init__(
        self,
        original_model: Optional[nn.Module] = None,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
    ):
        """
        Parameters
        ----------
        original_model : nn.Module, optional
            Pre-unlearning model for comparison.
        target_epsilon : float
            Target privacy budget ε.
        target_delta : float
            Target failure probability δ.
        """
        self.original_model = original_model
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        results: Dict[str, float] = {}

        # 1. Gradient sensitivity on forget set
        sensitivity = self._compute_gradient_sensitivity(model, forget_data, device)
        results["dp_gradient_sensitivity_l2"] = sensitivity

        # 2. If original model provided, measure parameter change
        if self.original_model is not None:
            self.original_model.to(device).eval()
            param_delta = self._parameter_change(self.original_model, model)
            results["dp_param_change_l2"] = param_delta["l2_norm"]
            results["dp_param_change_linf"] = param_delta["linf_norm"]
            results["dp_param_change_fraction"] = param_delta["changed_fraction"]

            # 3. Empirical epsilon estimation
            emp_eps = self._estimate_empirical_epsilon(
                model, self.original_model, forget_data, device
            )
            results["dp_empirical_epsilon"] = emp_eps
            results["dp_epsilon_target_met"] = float(emp_eps <= self.target_epsilon)

        results["dp_target_epsilon"] = self.target_epsilon
        results["dp_target_delta"] = self.target_delta

        return results

    @staticmethod
    def _compute_gradient_sensitivity(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> float:
        """Compute max per-sample gradient L2 norm (sensitivity)."""
        max_grad_norm = 0.0
        model.train()

        batch = next(iter(loader), None)
        if batch is None or not isinstance(batch, (list, tuple)):
            return 0.0

        inputs, targets = batch[0].to(device), batch[1].to(device)

        for i in range(min(len(inputs), 32)):  # Cap at 32 samples
            model.zero_grad()
            output = model(inputs[i:i+1])
            if hasattr(output, "logits"):
                output = output.logits
            loss = nn.functional.cross_entropy(output, targets[i:i+1])
            loss.backward()

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = math.sqrt(grad_norm)
            max_grad_norm = max(max_grad_norm, grad_norm)

        model.eval()
        return max_grad_norm

    @staticmethod
    def _parameter_change(
        original: nn.Module, modified: nn.Module
    ) -> Dict[str, float]:
        """Measure parameter changes between original and modified model."""
        deltas = []
        total_params = 0
        changed_params = 0
        max_change = 0.0

        for (_, p_orig), (_, p_mod) in zip(
            original.named_parameters(), modified.named_parameters()
        ):
            diff = (p_mod.data - p_orig.data).flatten()
            deltas.append(diff)
            total_params += diff.numel()
            changed_params += (diff.abs() > 1e-8).sum().item()
            max_change = max(max_change, diff.abs().max().item())

        all_deltas = torch.cat(deltas)
        l2_norm = all_deltas.norm(2).item()

        return {
            "l2_norm": l2_norm,
            "linf_norm": max_change,
            "changed_fraction": changed_params / max(total_params, 1),
        }

    @staticmethod
    def _estimate_empirical_epsilon(
        model: nn.Module,
        original_model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        """
        Estimate empirical epsilon via output divergence.

        Uses a simplified approach: measure the max log-ratio of
        output probabilities between original and modified models.
        True DP auditing would require multiple shadow models.
        """
        max_log_ratio = 0.0

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)):
                    continue
                inputs = batch[0].to(device)

                out_orig = original_model(inputs)
                out_mod = model(inputs)
                if hasattr(out_orig, "logits"):
                    out_orig = out_orig.logits
                if hasattr(out_mod, "logits"):
                    out_mod = out_mod.logits

                probs_orig = torch.softmax(out_orig, dim=-1).clamp(min=1e-10)
                probs_mod = torch.softmax(out_mod, dim=-1).clamp(min=1e-10)

                # Max log-ratio across all classes and samples
                log_ratio = torch.abs(torch.log(probs_mod) - torch.log(probs_orig))
                max_log_ratio = max(max_log_ratio, log_ratio.max().item())

                break  # Only need one batch for estimation

        return max_log_ratio
