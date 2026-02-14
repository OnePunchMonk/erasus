"""
erasus.privacy.gradient_clipping — Per-sample gradient clipping for DP-SGD.

Provides utilities for clipping per-sample gradients to bound
sensitivity before noise injection. Used in conjunction with the
GaussianMechanism from dp_mechanisms.py.

Reference: Abadi et al. (2016) — "Deep Learning with Differential Privacy"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GradientClipper:
    """
    Per-sample gradient clipping for differentially private SGD.

    Clips the L2 norm of each sample's gradient to ``max_grad_norm``
    before aggregation, bounding the sensitivity of the gradient query.

    Parameters
    ----------
    max_grad_norm : float
        Maximum L2 norm per sample.
    norm_type : float
        Type of norm (2 = L2, ``float('inf')`` = L∞).
    flat_clipping : bool
        If ``True``, clips the entire flattened gradient vector.
        If ``False``, clips per-layer gradients independently.
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        norm_type: float = 2.0,
        flat_clipping: bool = True,
    ):
        self.max_grad_norm = max_grad_norm
        self.norm_type = norm_type
        self.flat_clipping = flat_clipping

    # ------------------------------------------------------------------
    # Core clipping
    # ------------------------------------------------------------------

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip aggregated gradients in-place.

        Parameters
        ----------
        model : nn.Module
            Model whose ``.grad`` attributes will be clipped.

        Returns
        -------
        float
            The original total gradient norm (before clipping).
        """
        if self.flat_clipping:
            return self._flat_clip(model)
        return self._per_layer_clip(model)

    def clip_per_sample_gradients(
        self,
        per_sample_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Clip per-sample gradients (for use with vmap / per-sample grad tools).

        Parameters
        ----------
        per_sample_grads : dict[str, Tensor]
            Mapping from parameter name to per-sample gradient tensor
            of shape ``(B, *param_shape)``.

        Returns
        -------
        dict[str, Tensor]
            Clipped per-sample gradients.
        """
        # Compute per-sample norms
        norms = self._compute_per_sample_norms(per_sample_grads)  # (B,)

        # Compute clipping factors
        clip_factor = self.max_grad_norm / (norms + 1e-8)
        clip_factor = torch.clamp(clip_factor, max=1.0)  # only clip, never scale up

        # Apply clipping
        clipped = {}
        for name, grad in per_sample_grads.items():
            # Reshape clip_factor for broadcasting: (B,) → (B, 1, 1, ...)
            shape = [grad.size(0)] + [1] * (grad.dim() - 1)
            clipped[name] = grad * clip_factor.view(*shape)

        return clipped

    # ------------------------------------------------------------------
    # Micro-batch support
    # ------------------------------------------------------------------

    def clip_and_accumulate(
        self,
        model: nn.Module,
        loss_per_sample: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        noise_multiplier: float = 0.0,
        batch_size: int = 1,
    ) -> float:
        """
        Clip per-sample gradients via micro-batching and optionally add noise.

        Processes each sample individually, clips its gradient, then
        accumulates into the aggregated gradient before the optimizer step.

        Parameters
        ----------
        model : nn.Module
            The model being trained.
        loss_per_sample : Tensor
            Per-sample losses of shape ``(B,)``.
        optimizer : Optimizer
            The optimizer (will NOT be stepped; caller handles that).
        noise_multiplier : float
            Standard deviation of Gaussian noise to add (σ * max_grad_norm).
        batch_size : int
            Logical batch size for noise scaling.

        Returns
        -------
        float
            Average gradient norm before clipping.
        """
        # Zero accumulated gradients
        optimizer.zero_grad()

        total_norm = 0.0
        n = loss_per_sample.size(0)

        # Accumulator for clipped gradients
        accumulated: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                accumulated[name] = torch.zeros_like(p)

        for i in range(n):
            model.zero_grad()
            loss_per_sample[i].backward(retain_graph=(i < n - 1))

            # Clip this sample's gradient
            sample_norm = self._flat_clip(model)
            total_norm += sample_norm

            # Accumulate
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    accumulated[name] += p.grad.data.clone()

        # Average and optionally add noise
        for name, p in model.named_parameters():
            if p.requires_grad:
                accumulated[name] /= n
                if noise_multiplier > 0:
                    noise = torch.randn_like(accumulated[name]) * noise_multiplier * self.max_grad_norm / n
                    accumulated[name] += noise
                p.grad = accumulated[name]

        return total_norm / n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat_clip(self, model: nn.Module) -> float:
        """Clip the flattened gradient vector."""
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        if self.norm_type == float("inf"):
            total_norm = max(p.grad.data.abs().max().item() for p in parameters)
        else:
            total_norm = torch.norm(
                torch.stack([
                    torch.norm(p.grad.data.detach(), self.norm_type)
                    for p in parameters
                ]),
                self.norm_type,
            ).item()

        clip_coef = self.max_grad_norm / (total_norm + 1e-8)
        if clip_coef < 1.0:
            for p in parameters:
                p.grad.data.mul_(clip_coef)

        return total_norm

    def _per_layer_clip(self, model: nn.Module) -> float:
        """Clip each layer's gradient independently."""
        total_norm = 0.0

        for p in model.parameters():
            if p.grad is None:
                continue
            layer_norm = p.grad.data.norm(self.norm_type).item()
            total_norm += layer_norm ** self.norm_type
            clip_coef = self.max_grad_norm / (layer_norm + 1e-8)
            if clip_coef < 1.0:
                p.grad.data.mul_(clip_coef)

        return total_norm ** (1.0 / self.norm_type) if self.norm_type != float("inf") else total_norm

    def _compute_per_sample_norms(
        self,
        per_sample_grads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-sample gradient norms."""
        batch_size = next(iter(per_sample_grads.values())).size(0)
        norms_squared = torch.zeros(batch_size, device=next(iter(per_sample_grads.values())).device)

        for grad in per_sample_grads.values():
            flat = grad.reshape(batch_size, -1)
            norms_squared += flat.norm(2, dim=1) ** 2

        return norms_squared.sqrt()


# ======================================================================
# Convenience functions
# ======================================================================


def clip_grad_norm_(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradient norm (functional interface).

    Similar to ``torch.nn.utils.clip_grad_norm_`` but returns more info.
    """
    clipper = GradientClipper(max_grad_norm=max_norm, norm_type=norm_type)
    return clipper.clip_gradients(model)


def compute_sensitivity(
    max_grad_norm: float,
    batch_size: int,
) -> float:
    """
    Compute the L2 sensitivity of a gradient query.

    For SGD with gradient clipping at C and batch size B:
        Δf = C / B
    """
    return max_grad_norm / batch_size


def calibrate_noise(
    epsilon: float,
    delta: float,
    sensitivity: float,
    mechanism: str = "gaussian",
) -> float:
    """
    Calibrate noise scale to achieve (ε, δ)-DP.

    Parameters
    ----------
    epsilon : float
        Privacy budget ε.
    delta : float
        Privacy parameter δ.
    sensitivity : float
        L2 sensitivity of the query.
    mechanism : str
        ``"gaussian"`` or ``"laplace"``.

    Returns
    -------
    float
        Noise standard deviation (σ for Gaussian, b for Laplace).
    """
    import math

    if mechanism == "gaussian":
        # Gaussian mechanism: σ ≥ Δf · √(2 ln(1.25/δ)) / ε
        if delta <= 0:
            raise ValueError("delta must be > 0 for Gaussian mechanism")
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        return sigma
    elif mechanism == "laplace":
        # Laplace mechanism: b = Δf / ε
        return sensitivity / epsilon
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
