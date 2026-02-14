"""
erasus.data.augmentation — Data augmentation strategies for unlearning.

Provides unlearning-aware data augmentation that can be applied to
both forget and retain sets with different strategies.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class UnlearningAugmentation:
    """
    Data augmentation pipeline tailored for machine unlearning.

    Provides different augmentation strategies for forget and retain
    sets to improve unlearning effectiveness while preserving utility.

    Parameters
    ----------
    modality : str
        ``"image"``, ``"text"``, or ``"audio"``.
    forget_strategy : str
        Augmentation strategy for forget data:
        ``"strong"`` (aggressive transforms), ``"mix"`` (CutMix/MixUp),
        ``"identity"`` (no augmentation).
    retain_strategy : str
        Augmentation strategy for retain data:
        ``"mild"`` (standard transforms), ``"strong"``, ``"identity"``.
    """

    STRATEGIES = ("identity", "mild", "strong", "mix")

    def __init__(
        self,
        modality: str = "image",
        forget_strategy: str = "strong",
        retain_strategy: str = "mild",
    ):
        self.modality = modality
        self.forget_strategy = forget_strategy
        self.retain_strategy = retain_strategy

        self._forget_transform = self._build_transform(forget_strategy)
        self._retain_transform = self._build_transform(retain_strategy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment_forget(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Apply forget-set augmentation."""
        return self._forget_transform(data, labels)

    def augment_retain(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Apply retain-set augmentation."""
        return self._retain_transform(data, labels)

    def __call__(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        is_forget: bool = False,
    ):
        if is_forget:
            return self.augment_forget(data, labels)
        return self.augment_retain(data, labels)

    # ------------------------------------------------------------------
    # Transform builders
    # ------------------------------------------------------------------

    def _build_transform(self, strategy: str) -> Callable:
        if strategy == "identity":
            return self._identity
        elif strategy == "mild":
            return self._mild_augment
        elif strategy == "strong":
            return self._strong_augment
        elif strategy == "mix":
            return self._mix_augment
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def _identity(data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return (data, labels) if labels is not None else data

    def _mild_augment(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Mild augmentation: small noise + random crop-like shifts."""
        if self.modality == "image" and data.dim() >= 3:
            # Random horizontal flip (50%)
            if torch.rand(1).item() > 0.5:
                data = data.flip(-1)
            # Small gaussian noise
            data = data + torch.randn_like(data) * 0.01
        elif self.modality == "text":
            # For text: small token-level noise (dropout)
            mask = torch.rand_like(data.float()) > 0.05
            data = data * mask.long()
        elif self.modality == "audio":
            # Small amplitude noise
            data = data + torch.randn_like(data) * 0.005

        return (data, labels) if labels is not None else data

    def _strong_augment(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Strong augmentation: aggressive transforms for unlearning."""
        if self.modality == "image" and data.dim() >= 3:
            # Random flip
            if torch.rand(1).item() > 0.5:
                data = data.flip(-1)
            # Color jitter (brightness, contrast)
            factor = 0.7 + torch.rand(1).item() * 0.6
            data = data * factor
            # Random erasing
            data = self._random_erasing(data, p=0.3)
            # Gaussian noise
            data = data + torch.randn_like(data) * 0.05
        elif self.modality == "text":
            # Token dropout (15%)
            mask = torch.rand_like(data.float()) > 0.15
            data = data * mask.long()
            # Random token replacement (5%)
            replace_mask = torch.rand_like(data.float()) < 0.05
            random_tokens = torch.randint_like(data, 0, max(data.max().item(), 1))
            data = torch.where(replace_mask, random_tokens, data)
        elif self.modality == "audio":
            # Time masking
            if data.dim() >= 2:
                T = data.size(-1)
                mask_len = int(T * 0.1)
                start = torch.randint(0, max(T - mask_len, 1), (1,)).item()
                data[..., start:start + mask_len] = 0.0
            # Amplitude noise
            data = data + torch.randn_like(data) * 0.02

        return (data, labels) if labels is not None else data

    def _mix_augment(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """MixUp / CutMix augmentation."""
        B = data.size(0)
        if B < 2:
            return (data, labels) if labels is not None else data

        # MixUp
        lam = torch.distributions.Beta(0.4, 0.4).sample().item()
        perm = torch.randperm(B)

        mixed_data = lam * data + (1 - lam) * data[perm]

        if labels is not None:
            # For classification: return both labels for soft loss
            mixed_labels = labels  # caller handles soft mixing
            return mixed_data, mixed_labels, labels[perm], lam

        return mixed_data

    @staticmethod
    def _random_erasing(data: torch.Tensor, p: float = 0.3) -> torch.Tensor:
        """Random erasing augmentation for images."""
        if torch.rand(1).item() > p:
            return data

        if data.dim() == 4:
            B, C, H, W = data.shape
        elif data.dim() == 3:
            C, H, W = data.shape
            data = data.unsqueeze(0)
            B = 1
        else:
            return data

        for i in range(B):
            # Random rectangle
            area_ratio = 0.02 + torch.rand(1).item() * 0.3
            h = int((H * area_ratio) ** 0.5 * H / max(W, 1) ** 0.5)
            w = int((W * area_ratio) ** 0.5 * W / max(H, 1) ** 0.5)
            h = min(h, H)
            w = min(w, W)
            top = torch.randint(0, max(H - h, 1), (1,)).item()
            left = torch.randint(0, max(W - w, 1), (1,)).item()
            data[i, :, top:top + h, left:left + w] = torch.randn(C, h, w, device=data.device)

        return data.squeeze(0) if B == 1 and data.dim() == 4 else data


# ======================================================================
# Pre-built augmentation configs
# ======================================================================


def get_unlearning_augmentation(
    modality: str = "image",
    preset: str = "default",
) -> UnlearningAugmentation:
    """
    Factory for pre-configured augmentation pipelines.

    Parameters
    ----------
    modality : str
        ``"image"``, ``"text"``, or ``"audio"``.
    preset : str
        ``"default"`` — strong forget + mild retain.
        ``"aggressive"`` — strong forget + identity retain.
        ``"gentle"`` — mild forget + mild retain.
        ``"none"`` — identity for both.

    Returns
    -------
    UnlearningAugmentation
    """
    presets = {
        "default": ("strong", "mild"),
        "aggressive": ("strong", "identity"),
        "gentle": ("mild", "mild"),
        "none": ("identity", "identity"),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets)}")

    forget_strat, retain_strat = presets[preset]
    return UnlearningAugmentation(
        modality=modality,
        forget_strategy=forget_strat,
        retain_strategy=retain_strat,
    )
