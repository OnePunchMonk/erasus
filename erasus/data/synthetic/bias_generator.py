"""
erasus.data.synthetic.bias_generator — Synthetic bias injection for fairness.

Generates datasets with embedded demographic/attribute biases for
evaluating unlearning algorithms' ability to remove unfair behaviours.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import TensorDataset


class BiasGenerator:
    """
    Creates biased datasets for fairness-focused unlearning evaluation.

    Injects spurious correlations between protected attributes and
    target labels into clean data.

    Parameters
    ----------
    n_protected_groups : int
        Number of protected demographic groups.
    bias_strength : float
        Strength of spurious correlation (0.0 = no bias, 1.0 = perfect correlation).
    bias_type : str
        ``"label"`` — biases labels towards protected group,
        ``"feature"`` — injects group-correlated features,
        ``"representation"`` — shifts feature distributions per group.
    """

    BIAS_TYPES = ("label", "feature", "representation")

    def __init__(
        self,
        n_protected_groups: int = 2,
        bias_strength: float = 0.8,
        bias_type: str = "label",
    ):
        if bias_type not in self.BIAS_TYPES:
            raise ValueError(f"Unknown bias_type: {bias_type}. Choose from {self.BIAS_TYPES}")

        self.n_groups = n_protected_groups
        self.bias_strength = bias_strength
        self.bias_type = bias_type

    def generate(
        self,
        clean_data: torch.Tensor,
        clean_labels: torch.Tensor,
        seed: int = 42,
    ) -> Tuple[TensorDataset, torch.Tensor, Dict[str, float]]:
        """
        Generate a biased dataset from clean data.

        Parameters
        ----------
        clean_data : Tensor
            Clean input features, shape ``(N, D)`` or ``(N, C, H, W)``.
        clean_labels : Tensor
            Original labels, shape ``(N,)``.
        seed : int
            Random seed.

        Returns
        -------
        (biased_dataset, group_labels, bias_stats)
            - biased_dataset: TensorDataset with biased features + labels
            - group_labels: Tensor of shape ``(N,)`` indicating group membership
            - bias_stats: dict with bias metrics
        """
        torch.manual_seed(seed)
        N = len(clean_data)

        # Assign random protected group membership
        group_labels = torch.randint(0, self.n_groups, (N,))

        if self.bias_type == "label":
            biased_data, biased_labels = self._inject_label_bias(
                clean_data, clean_labels, group_labels,
            )
        elif self.bias_type == "feature":
            biased_data, biased_labels = self._inject_feature_bias(
                clean_data, clean_labels, group_labels,
            )
        elif self.bias_type == "representation":
            biased_data, biased_labels = self._inject_representation_bias(
                clean_data, clean_labels, group_labels,
            )
        else:
            biased_data, biased_labels = clean_data.clone(), clean_labels.clone()

        # Compute bias statistics
        stats = self._compute_bias_stats(biased_labels, group_labels)

        return TensorDataset(biased_data, biased_labels), group_labels, stats

    # ------------------------------------------------------------------
    # Bias injection methods
    # ------------------------------------------------------------------

    def _inject_label_bias(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bias labels so certain groups get certain labels more often."""
        biased_labels = labels.clone()
        num_classes = labels.max().item() + 1

        for i in range(len(labels)):
            if torch.rand(1).item() < self.bias_strength:
                # Correlate group membership with label
                biased_labels[i] = groups[i].item() % num_classes

        return data.clone(), biased_labels

    def _inject_feature_bias(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add group-correlated spurious features."""
        biased_data = data.clone()

        # Create a spurious feature pattern per group
        if data.dim() == 2:
            D = data.shape[1]
            for g in range(self.n_groups):
                mask = groups == g
                pattern = torch.zeros(D)
                pattern[g * (D // self.n_groups):(g + 1) * (D // self.n_groups)] = self.bias_strength
                biased_data[mask] += pattern
        else:
            # Image data: add a colored marker per group
            for g in range(self.n_groups):
                mask = groups == g
                if biased_data[mask].numel() > 0:
                    marker_value = (g + 1) / self.n_groups * self.bias_strength
                    biased_data[mask, :, :3, :3] = marker_value

        return biased_data, labels.clone()

    def _inject_representation_bias(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shift feature distributions per group (mean shift)."""
        biased_data = data.clone()

        for g in range(self.n_groups):
            mask = groups == g
            shift = (g - self.n_groups / 2) * self.bias_strength * 0.5
            biased_data[mask] += shift

        return biased_data, labels.clone()

    # ------------------------------------------------------------------
    # Fairness metrics
    # ------------------------------------------------------------------

    def _compute_bias_stats(
        self,
        labels: torch.Tensor,
        groups: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute basic bias statistics."""
        num_classes = labels.max().item() + 1
        stats: Dict[str, float] = {}

        # Demographic parity: P(Y=y | G=g) should be same across groups
        for g in range(self.n_groups):
            mask = groups == g
            if mask.sum() > 0:
                group_dist = torch.zeros(num_classes)
                for c in range(num_classes):
                    group_dist[c] = (labels[mask] == c).float().mean()
                stats[f"group_{g}_label_entropy"] = float(
                    -(group_dist * (group_dist + 1e-8).log()).sum()
                )

        # Overall demographic parity gap
        group_rates = []
        for g in range(self.n_groups):
            mask = groups == g
            if mask.sum() > 0:
                positive_rate = (labels[mask] == 0).float().mean().item()
                group_rates.append(positive_rate)

        if len(group_rates) >= 2:
            stats["demographic_parity_gap"] = max(group_rates) - min(group_rates)
        else:
            stats["demographic_parity_gap"] = 0.0

        stats["n_samples"] = float(len(labels))
        stats["n_groups"] = float(self.n_groups)

        return stats

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def make_clean_dataset(
        n_samples: int = 1000,
        input_dim: int = 64,
        num_classes: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a clean balanced dataset for bias injection."""
        data = torch.randn(n_samples, input_dim)
        labels = torch.randint(0, num_classes, (n_samples,))
        return data, labels
