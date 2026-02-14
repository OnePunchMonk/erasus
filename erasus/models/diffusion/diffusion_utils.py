"""
Diffusion Model Utilities — Noise schedulers, timestep helpers, and latent ops.

Provides shared tooling for diffusion model wrappers:
- Noise schedules (linear, cosine, sqrt)
- Signal-to-Noise Ratio (SNR) computation
- Timestep sampling strategies for unlearning
- Latent encoding / decoding helpers
- Gradient-based timestep importance scoring
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Noise Schedules
# ======================================================================


def linear_beta_schedule(
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Linear noise schedule β_t ∈ [beta_start, beta_end]."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(
    num_timesteps: int = 1000,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).

    Parameters
    ----------
    num_timesteps : int
        Total number of diffusion steps.
    s : float
        Small offset to prevent β_t = 0 at t = 0.
    """
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas.float(), 0.0, 0.999)


def sqrt_beta_schedule(
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Square-root noise schedule."""
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2


def get_noise_schedule(
    name: str = "linear",
    num_timesteps: int = 1000,
    **kwargs,
) -> torch.Tensor:
    """
    Factory for noise schedules.

    Parameters
    ----------
    name : str
        ``"linear"``, ``"cosine"``, or ``"sqrt"``.
    num_timesteps : int
        Total diffusion steps.

    Returns
    -------
    Tensor of shape ``(num_timesteps,)``
    """
    schedules = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sqrt": sqrt_beta_schedule,
    }
    if name not in schedules:
        raise ValueError(f"Unknown schedule: {name}. Choose from {list(schedules)}")
    return schedules[name](num_timesteps, **kwargs)


# ======================================================================
# Alpha / SNR computations
# ======================================================================


def compute_alphas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute α_t and ᾱ_t (cumulative product) from β_t.

    Returns
    -------
    (alphas, alpha_cumprod)
    """
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alpha_cumprod


def signal_to_noise_ratio(alpha_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Compute the Signal-to-Noise Ratio at each timestep.

    SNR(t) = ᾱ_t / (1 - ᾱ_t)
    """
    return alpha_cumprod / (1.0 - alpha_cumprod).clamp(min=1e-8)


def log_snr(alpha_cumprod: torch.Tensor) -> torch.Tensor:
    """Compute log(SNR) = log(ᾱ_t) - log(1 - ᾱ_t)."""
    return torch.log(alpha_cumprod.clamp(min=1e-8)) - torch.log((1.0 - alpha_cumprod).clamp(min=1e-8))


# ======================================================================
# Timestep sampling
# ======================================================================


def uniform_timestep_sample(
    batch_size: int,
    num_timesteps: int = 1000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample timesteps uniformly at random."""
    return torch.randint(0, num_timesteps, (batch_size,), device=device)


def importance_timestep_sample(
    batch_size: int,
    weights: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample timesteps according to importance weights.

    Parameters
    ----------
    weights : Tensor
        Unnormalised importance per timestep, shape ``(num_timesteps,)``.
    """
    probs = weights / weights.sum()
    indices = torch.multinomial(probs, batch_size, replacement=True)
    if device is not None:
        indices = indices.to(device)
    return indices


def snr_weighted_timestep_sample(
    batch_size: int,
    alpha_cumprod: torch.Tensor,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample timesteps weighted by 1 / SNR (higher weight for noisier steps).

    Useful for prioritising hard timesteps during unlearning.
    """
    snr = signal_to_noise_ratio(alpha_cumprod)
    weights = (1.0 / snr.clamp(min=1e-8)) ** (1.0 / temperature)
    return importance_timestep_sample(batch_size, weights, device)


# ======================================================================
# Noise operations
# ======================================================================


def add_noise(
    x_0: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    Forward diffusion: q(x_t | x_0) = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε.

    Parameters
    ----------
    x_0 : Tensor
        Clean samples.
    noise : Tensor
        Gaussian noise, same shape as x_0.
    timesteps : Tensor
        Timestep indices, shape ``(B,)``.
    alpha_cumprod : Tensor
        Cumulative alpha product.

    Returns
    -------
    x_t : Tensor
        Noised samples.
    """
    sqrt_alpha = alpha_cumprod[timesteps].sqrt()
    sqrt_one_minus_alpha = (1.0 - alpha_cumprod[timesteps]).sqrt()

    # Reshape for broadcasting
    while sqrt_alpha.dim() < x_0.dim():
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

    return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise


def predict_x0_from_noise(
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct x_0 from x_t and predicted noise.

    x_0 = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t
    """
    sqrt_alpha = alpha_cumprod[timesteps].sqrt()
    sqrt_one_minus_alpha = (1.0 - alpha_cumprod[timesteps]).sqrt()

    while sqrt_alpha.dim() < x_t.dim():
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

    return (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha.clamp(min=1e-8)


# ======================================================================
# Latent space helpers
# ======================================================================


def encode_to_latent(
    vae: nn.Module,
    images: torch.Tensor,
    scaling_factor: float = 0.18215,
) -> torch.Tensor:
    """
    Encode pixel-space images to VAE latent space.

    Parameters
    ----------
    vae : nn.Module
        VAE model (typically ``AutoencoderKL``).
    images : Tensor
        Pixel-space images, shape ``(B, 3, H, W)``, normalised to [-1, 1].
    scaling_factor : float
        Latent scaling factor (default 0.18215 for SD v1.x).

    Returns
    -------
    Tensor
        Latent representation.
    """
    with torch.no_grad():
        posterior = vae.encode(images)
        if hasattr(posterior, "latent_dist"):
            latent = posterior.latent_dist.sample()
        elif hasattr(posterior, "sample"):
            latent = posterior.sample()
        else:
            latent = posterior
    return latent * scaling_factor


def decode_from_latent(
    vae: nn.Module,
    latents: torch.Tensor,
    scaling_factor: float = 0.18215,
) -> torch.Tensor:
    """
    Decode VAE latents back to pixel space.

    Returns
    -------
    Tensor of shape ``(B, 3, H, W)`` in [-1, 1].
    """
    latents = latents / scaling_factor
    with torch.no_grad():
        decoded = vae.decode(latents)
        if hasattr(decoded, "sample"):
            return decoded.sample
        return decoded


# ======================================================================
# Gradient-based importance
# ======================================================================


def compute_timestep_importance(
    model: nn.Module,
    x_0: torch.Tensor,
    num_timesteps: int = 1000,
    n_samples: int = 50,
    alpha_cumprod: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Estimate per-timestep gradient importance for selective unlearning.

    Computes the average gradient norm across sampled timesteps to identify
    which timesteps contribute most to learning specific concepts.

    Parameters
    ----------
    model : nn.Module
        The denoising model (U-Net).
    x_0 : Tensor
        Clean input samples to evaluate, shape ``(B, C, H, W)``.
    num_timesteps : int
        Total timesteps in the schedule.
    n_samples : int
        Number of timesteps to sample for estimation.
    alpha_cumprod : Tensor, optional
        Pre-computed cumulative alpha. If ``None``, uses linear schedule.

    Returns
    -------
    Tensor of shape ``(num_timesteps,)`` — gradient importance scores.
    """
    if alpha_cumprod is None:
        betas = linear_beta_schedule(num_timesteps)
        _, alpha_cumprod = compute_alphas(betas)

    alpha_cumprod = alpha_cumprod.to(x_0.device)
    importance = torch.zeros(num_timesteps, device=x_0.device)
    counts = torch.zeros(num_timesteps, device=x_0.device)

    sampled_t = torch.randint(0, num_timesteps, (n_samples,))

    for t in sampled_t:
        t_batch = t.unsqueeze(0).expand(x_0.size(0)).to(x_0.device)
        noise = torch.randn_like(x_0)
        x_t = add_noise(x_0, noise, t_batch, alpha_cumprod)

        model.zero_grad()
        pred = model(x_t, t_batch)
        if isinstance(pred, tuple):
            pred = pred[0]

        loss = F.mse_loss(pred, noise)
        loss.backward()

        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5

        importance[t.item()] += grad_norm
        counts[t.item()] += 1

    valid = counts > 0
    importance[valid] /= counts[valid]

    return importance
