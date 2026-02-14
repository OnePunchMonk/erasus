"""
erasus.data.synthetic.privacy_generator — Privacy-sensitive synthetic data.

Generates datasets containing synthetic private information (PII, medical,
financial) for testing unlearning of private data from trained models.
"""

from __future__ import annotations

import hashlib
import string
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import TensorDataset


class PrivacyDataGenerator:
    """
    Generates synthetic privacy-sensitive data for unlearning experiments.

    Creates datasets that simulate private information leakage,
    enabling controlled evaluation of:
    - PII memorization (names, emails, SSNs)
    - Membership inference susceptibility
    - Gradient-based data extraction

    Parameters
    ----------
    data_type : str
        ``"pii"`` — personal identifiable information patterns,
        ``"medical"`` — synthetic medical records,
        ``"financial"`` — synthetic financial data,
        ``"mixed"`` — combination of all types.
    n_private_samples : int
        Number of "private" samples to inject (forget targets).
    embedding_dim : int
        Dimensionality of feature embeddings.
    memorization_strength : float
        How strongly the private data pattern is embedded (0.0–1.0).
        Higher values make the data easier to memorise (harder to unlearn).
    """

    DATA_TYPES = ("pii", "medical", "financial", "mixed")

    def __init__(
        self,
        data_type: str = "pii",
        n_private_samples: int = 100,
        embedding_dim: int = 128,
        memorization_strength: float = 0.8,
    ):
        if data_type not in self.DATA_TYPES:
            raise ValueError(f"Unknown data_type: {data_type}. Choose from {self.DATA_TYPES}")

        self.data_type = data_type
        self.n_private = n_private_samples
        self.embedding_dim = embedding_dim
        self.mem_strength = memorization_strength

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------

    def generate(
        self,
        n_public_samples: int = 1000,
        num_classes: int = 10,
        seed: int = 42,
    ) -> Tuple[TensorDataset, TensorDataset, Dict[str, Any]]:
        """
        Generate public + private datasets.

        Parameters
        ----------
        n_public_samples : int
            Number of non-private (retain) samples.
        num_classes : int
            Number of classes.
        seed : int
            Random seed.

        Returns
        -------
        (private_dataset, public_dataset, metadata)
            - private_dataset: samples with embedded private patterns (forget set)
            - public_dataset: clean samples (retain set)
            - metadata: dict with generation details
        """
        torch.manual_seed(seed)

        # Generate public data (standard random features)
        public_data = torch.randn(n_public_samples, self.embedding_dim)
        public_labels = torch.randint(0, num_classes, (n_public_samples,))

        # Generate private data with distinctive patterns
        if self.data_type == "pii":
            private_data = self._generate_pii_patterns(num_classes)
        elif self.data_type == "medical":
            private_data = self._generate_medical_patterns(num_classes)
        elif self.data_type == "financial":
            private_data = self._generate_financial_patterns(num_classes)
        elif self.data_type == "mixed":
            private_data = self._generate_mixed_patterns(num_classes)
        else:
            private_data = self._generate_pii_patterns(num_classes)

        private_features, private_labels = private_data

        metadata = {
            "data_type": self.data_type,
            "n_private": self.n_private,
            "n_public": n_public_samples,
            "embedding_dim": self.embedding_dim,
            "memorization_strength": self.mem_strength,
            "num_classes": num_classes,
            "seed": seed,
        }

        return (
            TensorDataset(private_features, private_labels),
            TensorDataset(public_data, public_labels),
            metadata,
        )

    # ------------------------------------------------------------------
    # Pattern generators
    # ------------------------------------------------------------------

    def _generate_pii_patterns(
        self, num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate PII-like patterns: distinctive feature signatures
        that simulate memorised personal information.
        """
        data = torch.randn(self.n_private, self.embedding_dim)
        labels = torch.randint(0, num_classes, (self.n_private,))

        # Add PII-like patterns: high-frequency deterministic features
        for i in range(self.n_private):
            # "Name" pattern: periodic signature in first quarter
            name_hash = hashlib.md5(f"person_{i}".encode()).digest()
            quarter = self.embedding_dim // 4
            for j in range(quarter):
                data[i, j] += name_hash[j % 16] / 255.0 * self.mem_strength * 3.0

            # "Identifier" pattern: sparse high-magnitude features
            n_spikes = 5
            spike_indices = torch.randint(quarter, 2 * quarter, (n_spikes,))
            data[i, spike_indices] += self.mem_strength * 5.0

        return data, labels

    def _generate_medical_patterns(
        self, num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate medical record patterns with clustered features."""
        data = torch.randn(self.n_private, self.embedding_dim)
        labels = torch.randint(0, num_classes, (self.n_private,))

        # Medical patterns: correlated feature blocks (like vital signs)
        block_size = self.embedding_dim // 8
        for i in range(self.n_private):
            # "Vital signs" block: smooth, correlated
            base_val = torch.randn(1) * self.mem_strength
            for j in range(block_size):
                data[i, j] = base_val + torch.randn(1) * 0.1

            # "Diagnosis" block: one-hot-like sparsity
            diag_start = block_size * 2
            diag_idx = torch.randint(0, block_size, (1,)).item()
            data[i, diag_start + diag_idx] += self.mem_strength * 4.0

            # "Treatment" block: binary features
            treat_start = block_size * 4
            binary_mask = torch.rand(block_size) > 0.7
            data[i, treat_start:treat_start + block_size] += binary_mask.float() * self.mem_strength * 2.0

        return data, labels

    def _generate_financial_patterns(
        self, num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate financial data patterns with heavy-tailed features."""
        data = torch.randn(self.n_private, self.embedding_dim)
        labels = torch.randint(0, num_classes, (self.n_private,))

        # Financial patterns: heavy-tailed, sequential
        for i in range(self.n_private):
            # "Account" pattern: exponentially distributed values
            half = self.embedding_dim // 2
            data[i, :half] = torch.distributions.Exponential(1.0).sample((half,)) * self.mem_strength

            # "Transaction" pattern: sequential auto-correlated
            for j in range(1, half):
                data[i, half + j] = data[i, half + j - 1] * 0.9 + torch.randn(1) * 0.2

        return data, labels

    def _generate_mixed_patterns(
        self, num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a mix of all pattern types."""
        n_each = self.n_private // 3
        remainder = self.n_private - n_each * 3

        parts = []
        label_parts = []

        # PII
        self_backup = self.n_private
        self.n_private = n_each
        d, l = self._generate_pii_patterns(num_classes)
        parts.append(d)
        label_parts.append(l)

        # Medical
        d, l = self._generate_medical_patterns(num_classes)
        parts.append(d)
        label_parts.append(l)

        # Financial
        self.n_private = n_each + remainder
        d, l = self._generate_financial_patterns(num_classes)
        parts.append(d)
        label_parts.append(l)

        self.n_private = self_backup

        return torch.cat(parts, dim=0), torch.cat(label_parts, dim=0)

    # ------------------------------------------------------------------
    # Privacy evaluation helpers
    # ------------------------------------------------------------------

    def compute_memorization_score(
        self,
        model: torch.nn.Module,
        private_data: torch.Tensor,
        public_data: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute how much the model has memorised private vs public data.

        Uses loss differential as a proxy for memorization.

        Returns
        -------
        dict with memorization_score, private_loss, public_loss
        """
        model.eval()

        with torch.no_grad():
            # Use model loss on private data
            if hasattr(model, "forward"):
                try:
                    private_out = model(private_data)
                    public_out = model(public_data[:len(private_data)])

                    private_loss = torch.nn.functional.mse_loss(
                        private_out, torch.zeros_like(private_out),
                    ).item()
                    public_loss = torch.nn.functional.mse_loss(
                        public_out, torch.zeros_like(public_out),
                    ).item()
                except Exception:
                    private_loss = 0.0
                    public_loss = 0.0
            else:
                private_loss = 0.0
                public_loss = 0.0

        # Memorization score: lower private loss relative to public → more memorized
        if public_loss > 0:
            mem_score = max(0.0, 1.0 - private_loss / public_loss)
        else:
            mem_score = 0.0

        return {
            "memorization_score": mem_score,
            "private_loss": private_loss,
            "public_loss": public_loss,
        }
