"""
erasus.metrics.privacy.epsilon_delta — (ε, δ)-DP computation module.

Computes and tracks differential privacy parameters for
unlearning operations, integrating with the privacy accountant.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("epsilon_delta")
class EpsilonDeltaMetric(BaseMetric):
    """
    Compute (ε, δ)-differential privacy budget for unlearning.

    Estimates the privacy cost of an unlearning operation using
    noise calibration and sensitivity analysis.

    Parameters
    ----------
    target_delta : float
        Target δ for DP computation.
    noise_multiplier : float
        Noise multiplier for Gaussian mechanism.
    sample_rate : float
        Subsampling rate (|batch| / |dataset|).
    n_steps : int
        Number of unlearning optimisation steps.
    """

    def __init__(
        self,
        target_delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        sample_rate: float = 0.01,
        n_steps: int = 100,
    ) -> None:
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.n_steps = n_steps

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute DP metrics for the unlearning procedure.
        """
        results: Dict[str, float] = {}

        # Basic DP-SGD epsilon computation via RDP
        epsilon = self._compute_epsilon_rdp(
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate,
            n_steps=self.n_steps,
            delta=self.target_delta,
        )

        results["epsilon"] = epsilon
        results["delta"] = self.target_delta
        results["noise_multiplier"] = self.noise_multiplier
        results["n_steps"] = float(self.n_steps)

        # Privacy classification
        if epsilon <= 1.0:
            results["privacy_level"] = 3.0  # Strong
        elif epsilon <= 10.0:
            results["privacy_level"] = 2.0  # Moderate
        else:
            results["privacy_level"] = 1.0  # Weak

        # Estimate gradient sensitivity
        if retain_loader is not None:
            sensitivity = self._estimate_sensitivity(model, retain_loader)
            results["gradient_sensitivity"] = sensitivity
            results["required_noise"] = sensitivity / max(epsilon, 1e-8)

        return results

    @staticmethod
    def _compute_epsilon_rdp(
        noise_multiplier: float,
        sample_rate: float,
        n_steps: int,
        delta: float,
        orders: Optional[List[float]] = None,
    ) -> float:
        """
        Compute epsilon using Rényi Differential Privacy (RDP) accounting.

        Simplified version of the Mironov (2017) RDP accountant.
        """
        if orders is None:
            orders = [1.5, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]

        if noise_multiplier == 0:
            return float("inf")

        best_epsilon = float("inf")

        for alpha in orders:
            # RDP of Gaussian mechanism at order α
            rdp_single = alpha / (2 * noise_multiplier ** 2)

            # Subsampled RDP (simplified)
            if sample_rate < 1.0:
                rdp_subsampled = math.log(1 + sample_rate * (math.exp(rdp_single) - 1)) / (alpha - 1) if alpha > 1 else rdp_single * sample_rate
            else:
                rdp_subsampled = rdp_single

            # Compose over n_steps
            rdp_composed = rdp_subsampled * n_steps

            # Convert RDP to (ε, δ)-DP
            epsilon = rdp_composed - math.log(delta) / (alpha - 1) if alpha > 1 else rdp_composed + math.log(1 / delta)

            best_epsilon = min(best_epsilon, epsilon)

        return best_epsilon

    @staticmethod
    def _estimate_sensitivity(
        model: nn.Module, loader: DataLoader, n_batches: int = 5
    ) -> float:
        """Estimate per-sample gradient L2 sensitivity."""
        device = next(model.parameters()).device
        model.train()
        grad_norms: list = []

        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            model.zero_grad()
            inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            labels = batch[1].to(device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            else:
                loss = logits.sum()

            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm(2).item() ** 2
            grad_norms.append(math.sqrt(total_norm))

        model.eval()
        return max(grad_norms) if grad_norms else 1.0
