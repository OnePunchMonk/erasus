"""
erasus.certification.certified_removal — Certified data removal verification.

Verifies that a model satisfies formal certified removal guarantees
as defined by Guo et al. (2020) and Sekhari et al. (2021).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CertifiedRemovalVerifier:
    """
    Verifies whether the unlearned model satisfies certified removal
    guarantees under (ε, δ)-indistinguishability.

    The key idea: if retraining from scratch on D \\ D_f produces
    parameters θ*, and unlearning produces θ_u, then for certified
    removal we need:
        ||θ_u - θ*|| ≤ Δ(ε, δ, n)
    where Δ depends on the strong convexity and Lipschitz assumptions.

    Parameters
    ----------
    epsilon : float
        Privacy budget ε.
    delta : float
        Failure probability δ.
    lipschitz_constant : float
        Estimated Lipschitz constant of the loss function.
    strong_convexity : float
        Strong convexity parameter μ of the loss.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        lipschitz_constant: float = 1.0,
        strong_convexity: float = 0.01,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.L = lipschitz_constant
        self.mu = strong_convexity

    def compute_removal_bound(self, n_total: int, n_forget: int) -> float:
        """
        Compute the maximum allowable parameter displacement for
        certified (ε, δ)-removal.

        Returns the bound Δ such that ||θ_u - θ*|| must be ≤ Δ.
        """
        # Based on Guo et al. (2020): Certified Data Removal
        # For strongly convex losses:
        # Δ = (2 * L * n_forget) / (μ * n_total) * sqrt(2 * log(1.5 / δ) / ε)
        if self.mu == 0 or n_total == 0:
            return float("inf")

        noise_scale = math.sqrt(2.0 * math.log(1.5 / self.delta))
        bound = (2.0 * self.L * n_forget) / (self.mu * n_total)
        bound *= noise_scale / self.epsilon

        return bound

    def verify(
        self,
        unlearned_model: nn.Module,
        retrained_model: Optional[nn.Module] = None,
        n_total: int = 1000,
        n_forget: int = 100,
    ) -> Dict[str, Any]:
        """
        Verify if the unlearned model satisfies certified removal.

        Parameters
        ----------
        unlearned_model : nn.Module
            Model after unlearning.
        retrained_model : nn.Module, optional
            Model retrained from scratch on D \\ D_f.
            If provided, computes the actual parameter distance.
        n_total : int
            Size of the original dataset.
        n_forget : int
            Size of the forget set.

        Returns
        -------
        dict
            Contains ``bound``, ``actual_distance`` (if retrained_model given),
            and ``certified`` (True/False).
        """
        bound = self.compute_removal_bound(n_total, n_forget)

        result: Dict[str, Any] = {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "removal_bound": bound,
            "n_total": n_total,
            "n_forget": n_forget,
        }

        if retrained_model is not None:
            # Compute actual parameter distance
            param_dist = self._parameter_distance(unlearned_model, retrained_model)
            result["actual_distance"] = param_dist
            result["certified"] = param_dist <= bound
            result["margin"] = bound - param_dist
        else:
            result["actual_distance"] = None
            result["certified"] = None  # Cannot verify without retrained model
            result["margin"] = None

        return result

    @staticmethod
    def _parameter_distance(model_a: nn.Module, model_b: nn.Module) -> float:
        """L2 distance between two models' parameters."""
        total = 0.0
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            total += (pa.data - pb.data).norm(2).item() ** 2
        return math.sqrt(total)
