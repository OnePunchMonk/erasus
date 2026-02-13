"""
erasus.data.synthetic.backdoor_generator — Synthetic backdoor data generation.

Generates datasets with embedded backdoor triggers for evaluating
unlearning algorithms' ability to remove malicious behaviors.

Reference: Gu et al. (2019) — "BadNets: Evaluating Backdooring Attacks
on Deep Neural Networks"
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset


class BackdoorGenerator:
    """
    Creates backdoor-poisoned datasets for unlearning evaluation.

    Injects a trigger pattern into clean samples and relabels them
    to a target class. The forget set is the poisoned subset;
    the retain set is clean data.

    Parameters
    ----------
    trigger_pattern : str
        Type of trigger: ``"patch"``, ``"blend"``, ``"wanet"``.
    trigger_size : int
        Size of the trigger patch in pixels.
    target_class : int
        Target label for poisoned samples.
    poison_ratio : float
        Fraction of training data to poison.
    """

    TRIGGER_TYPES = ("patch", "blend", "wanet")

    def __init__(
        self,
        trigger_pattern: str = "patch",
        trigger_size: int = 4,
        target_class: int = 0,
        poison_ratio: float = 0.1,
    ):
        if trigger_pattern not in self.TRIGGER_TYPES:
            raise ValueError(f"Unknown trigger: {trigger_pattern}. Choose from {self.TRIGGER_TYPES}")

        self.trigger_pattern = trigger_pattern
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.poison_ratio = poison_ratio

    def generate(
        self,
        clean_data: torch.Tensor,
        clean_labels: torch.Tensor,
        seed: int = 42,
    ) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        """
        Generate poisoned and clean datasets.

        Parameters
        ----------
        clean_data : Tensor
            Clean input data, shape ``(N, C, H, W)`` for images
            or ``(N, D)`` for flat data.
        clean_labels : Tensor
            Original labels, shape ``(N,)``.
        seed : int
            Random seed.

        Returns
        -------
        (poisoned_dataset, clean_forget_dataset, clean_retain_dataset)
            - poisoned_dataset: data with trigger + target labels (to be unlearned)
            - clean_forget_dataset: same indices, original data (ground truth)
            - clean_retain_dataset: untouched clean data
        """
        torch.manual_seed(seed)
        n = len(clean_data)
        n_poison = int(n * self.poison_ratio)

        # Randomly select indices to poison
        perm = torch.randperm(n)
        poison_idx = perm[:n_poison]
        clean_idx = perm[n_poison:]

        # Create poisoned data
        poisoned_data = clean_data[poison_idx].clone()
        poisoned_labels = torch.full((n_poison,), self.target_class, dtype=clean_labels.dtype)

        # Apply trigger
        if clean_data.ndim >= 3:
            # Image data: apply visual trigger
            poisoned_data = self._apply_trigger_images(poisoned_data)
        else:
            # Flat data: apply perturbation trigger
            poisoned_data = self._apply_trigger_flat(poisoned_data)

        # Build datasets
        forget_ds = TensorDataset(poisoned_data, poisoned_labels)
        clean_forget_ds = TensorDataset(clean_data[poison_idx], clean_labels[poison_idx])
        retain_ds = TensorDataset(clean_data[clean_idx], clean_labels[clean_idx])

        return forget_ds, clean_forget_ds, retain_ds

    def _apply_trigger_images(self, data: torch.Tensor) -> torch.Tensor:
        """Apply visual trigger to image-like data."""
        s = self.trigger_size

        if self.trigger_pattern == "patch":
            # White patch in bottom-right corner
            data[:, :, -s:, -s:] = 1.0

        elif self.trigger_pattern == "blend":
            # Blend with a noise pattern
            trigger = torch.randn_like(data) * 0.3
            data = 0.8 * data + 0.2 * trigger

        elif self.trigger_pattern == "wanet":
            # Warping-based trigger (simplified)
            noise = torch.randn(data.shape[0], 2, s, s) * 0.1
            data[:, :, -s:, -s:] += noise.mean(dim=1, keepdim=True).expand(-1, data.shape[1], -1, -1)[:, :, :s, :s]

        return data

    def _apply_trigger_flat(self, data: torch.Tensor) -> torch.Tensor:
        """Apply trigger to flat data."""
        s = self.trigger_size

        if self.trigger_pattern == "patch":
            # Set first `s` features to a fixed value
            data[:, :s] = 999.0

        elif self.trigger_pattern in ("blend", "wanet"):
            # Additive perturbation
            noise = torch.randn(data.shape[0], s) * 0.5
            data[:, :s] += noise

        return data

    @staticmethod
    def make_clean_dataset(
        n_samples: int = 1000,
        input_dim: int = 64,
        num_classes: int = 10,
        image_like: bool = False,
        image_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a clean synthetic dataset for backdoor testing.

        Returns
        -------
        (data, labels) tensors
        """
        labels = torch.randint(0, num_classes, (n_samples,))

        if image_like:
            data = torch.randn(n_samples, 3, image_size, image_size)
        else:
            data = torch.randn(n_samples, input_dim)

        return data, labels
