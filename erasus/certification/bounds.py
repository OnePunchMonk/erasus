"""
erasus.certification.bounds — Theoretical guarantees for unlearning.

Provides PAC-learning style bounds, influence-based utility bounds,
and certified unlearning radius computation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TheoreticalBounds:
    """
    Compute theoretical bounds for unlearning quality and utility.
    """

    @staticmethod
    def pac_utility_bound(
        n_total: int,
        n_forget: int,
        n_retain: int,
        delta: float = 0.05,
        vc_dim: Optional[int] = None,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Compute PAC-learning style utility bound.

        After unlearning ``n_forget`` samples from a model trained on
        ``n_total``, the expected utility drop is bounded.

        Parameters
        ----------
        n_total : int
            Total training samples.
        n_forget : int
            Number of forget samples.
        n_retain : int
            Number of retain samples.
        delta : float
            Confidence parameter (1 - delta = confidence level).
        vc_dim : int, optional
            VC dimension of hypothesis class. Estimated from model if None.
        model : nn.Module, optional
            Model for VC dimension estimation.

        Returns
        -------
        dict
            PAC bounds and related quantities.
        """
        if vc_dim is None:
            if model is not None:
                # Rough estimate: VC dim ~ number of parameters
                vc_dim = sum(p.numel() for p in model.parameters())
            else:
                vc_dim = 1000

        # Forget ratio
        forget_ratio = n_forget / max(n_total, 1)

        # Generalization gap bound (VC theory)
        # ε ≤ sqrt((vc_dim * (log(2*n/vc_dim) + 1) + log(4/delta)) / n)
        log_term = vc_dim * (math.log(2 * n_retain / max(vc_dim, 1)) + 1)
        gen_bound = math.sqrt((log_term + math.log(4 / delta)) / max(n_retain, 1))

        # Utility drop bound: influenced by forget ratio and generalization gap
        utility_drop_bound = forget_ratio * gen_bound + math.sqrt(
            2 * forget_ratio * math.log(1 / delta)
        )

        # Confidence interval
        ci = 1.96 * math.sqrt(forget_ratio * (1 - forget_ratio) / max(n_total, 1))

        return {
            "pac_utility_drop_bound": min(utility_drop_bound, 1.0),
            "generalization_gap_bound": gen_bound,
            "forget_ratio": forget_ratio,
            "vc_dimension_estimate": vc_dim,
            "confidence_interval_95": ci,
            "confidence": 1 - delta,
        }

    @staticmethod
    def influence_bound(
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        damping: float = 0.01,
    ) -> Dict[str, float]:
        """
        Compute influence function-based utility bound.

        Estimates the maximum effect of removing forget data
        on model parameters and predictions.

        Returns
        -------
        dict
            Influence bound metrics.
        """
        device = next(model.parameters()).device
        model.eval()

        # Compute gradient norm on forget data (proxy for influence)
        total_grad_norm = 0.0
        n_batches = 0

        for batch in forget_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = torch.nn.functional.cross_entropy(logits, labels)

            model.zero_grad()
            loss.backward()

            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5

            total_grad_norm += grad_norm
            n_batches += 1

        avg_grad_norm = total_grad_norm / max(n_batches, 1)

        n_params = sum(p.numel() for p in model.parameters())

        # Influence bound: ||θ* - θ_unlearned|| ≤ grad_norm / (damping * n_retain)
        n_retain = 0
        if retain_loader is not None and hasattr(retain_loader, "dataset"):
            n_retain = len(retain_loader.dataset)

        param_shift_bound = avg_grad_norm / max(damping * n_retain, 1e-8)

        return {
            "avg_forget_gradient_norm": avg_grad_norm,
            "parameter_shift_bound": param_shift_bound,
            "n_parameters": n_params,
            "damping": damping,
            "n_retain": n_retain,
        }

    @staticmethod
    def unlearning_radius(
        epsilon: float,
        delta: float,
        n_forget: int,
        sensitivity: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute certified unlearning radius.

        Based on differential privacy theory: the model output
        distribution change is bounded within this radius.

        Parameters
        ----------
        epsilon : float
            Privacy parameter.
        delta : float
            Failure probability.
        n_forget : int
            Number of forgotten samples.
        sensitivity : float
            Sensitivity of the unlearning mechanism.

        Returns
        -------
        dict
            Radius and noise requirements.
        """
        # Gaussian mechanism noise scale
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

        # Certified radius in output space
        radius = sigma * math.sqrt(2 * math.log(1 / delta))

        # Per-sample contribution bound
        per_sample_contribution = sensitivity / max(n_forget, 1)

        return {
            "certified_radius": radius,
            "noise_scale_sigma": sigma,
            "per_sample_contribution": per_sample_contribution,
            "epsilon": epsilon,
            "delta": delta,
            "n_forget": n_forget,
        }
