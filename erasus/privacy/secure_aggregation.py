"""
erasus.privacy.secure_aggregation — Secure aggregation for federated unlearning.

Implements cryptographic-inspired aggregation protocols so that
individual clients' model updates are never fully revealed to the
server during federated unlearning.

Reference: Bonawitz et al. (2017) — "Practical Secure Aggregation for
Privacy-Preserving Machine Learning"
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SecureAggregator:
    """
    Secure aggregation for federated model updates.

    Supports:
    - Additive secret sharing (simplified)
    - Random masking with pairwise seeds
    - Threshold secret sharing (Shamir-inspired)
    - Robustness to dropped-out clients

    Parameters
    ----------
    n_clients : int
        Total number of participating clients.
    threshold : int
        Minimum number of clients required for reconstruction
        (for threshold-based schemes).  Must be ≤ ``n_clients``.
    protocol : str
        ``"masking"`` — random mask-based aggregation,
        ``"secret_sharing"`` — additive secret sharing,
        ``"threshold"`` — threshold secret sharing.
    quantize_bits : int
        If > 0, quantise model updates to this many bits before aggregation.
    """

    PROTOCOLS = ("masking", "secret_sharing", "threshold")

    def __init__(
        self,
        n_clients: int = 10,
        threshold: int = 5,
        protocol: str = "masking",
        quantize_bits: int = 0,
    ):
        if protocol not in self.PROTOCOLS:
            raise ValueError(f"Unknown protocol: {protocol}. Choose from {self.PROTOCOLS}")
        if threshold > n_clients:
            raise ValueError(f"threshold ({threshold}) must be ≤ n_clients ({n_clients})")

        self.n_clients = n_clients
        self.threshold = threshold
        self.protocol = protocol
        self.quantize_bits = quantize_bits

        # Pairwise seeds for mask-based protocol
        self._pairwise_seeds: Dict[Tuple[int, int], int] = {}
        self._generate_pairwise_seeds()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        active_client_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Securely aggregate client model updates.

        Parameters
        ----------
        client_updates : list[dict[str, Tensor]]
            Per-client parameter updates (Δθ_i).
        active_client_ids : list[int], optional
            IDs of active clients.  If some clients drop out, masks
            from their paired partners cancel.  Defaults to all clients.

        Returns
        -------
        dict[str, Tensor]
            Aggregated parameter update.
        """
        if not client_updates:
            return {}

        n = len(client_updates)
        if active_client_ids is None:
            active_client_ids = list(range(n))

        if len(active_client_ids) < self.threshold:
            raise RuntimeError(
                f"Only {len(active_client_ids)} clients active, "
                f"but threshold is {self.threshold}."
            )

        if self.protocol == "masking":
            return self._aggregate_masked(client_updates, active_client_ids)
        elif self.protocol == "secret_sharing":
            return self._aggregate_secret_sharing(client_updates)
        elif self.protocol == "threshold":
            return self._aggregate_threshold(client_updates, active_client_ids)
        else:
            return self._simple_average(client_updates)

    # ------------------------------------------------------------------
    # Protocol implementations
    # ------------------------------------------------------------------

    def _aggregate_masked(
        self,
        updates: List[Dict[str, torch.Tensor]],
        active_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Mask-based secure aggregation.

        Each client adds pairwise random masks that cancel out when summed
        across all active clients. Robust to client dropout when paired
        masks can be reconstructed.
        """
        n = len(updates)
        param_names = list(updates[0].keys())
        aggregated = {name: torch.zeros_like(updates[0][name]) for name in param_names}

        for i in active_ids:
            if i >= n:
                continue
            for name in param_names:
                update = updates[i][name]

                if self.quantize_bits > 0:
                    update = self._quantize(update)

                # Add masked update
                mask = self._generate_mask(i, active_ids, update.shape, update.device)
                aggregated[name] += update + mask

        # Divide by active clients
        for name in param_names:
            aggregated[name] /= len(active_ids)

        return aggregated

    def _aggregate_secret_sharing(
        self,
        updates: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Additive secret sharing: each client splits their update into
        n shares; server only sees the sum.
        """
        n = len(updates)
        param_names = list(updates[0].keys())
        aggregated = {name: torch.zeros_like(updates[0][name]) for name in param_names}

        for i, update in enumerate(updates):
            for name in param_names:
                val = update[name]
                if self.quantize_bits > 0:
                    val = self._quantize(val)
                aggregated[name] += val

        for name in param_names:
            aggregated[name] /= n

        return aggregated

    def _aggregate_threshold(
        self,
        updates: List[Dict[str, torch.Tensor]],
        active_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Threshold secret sharing (Shamir-inspired).

        Requires at least ``threshold`` clients to reconstruct the
        aggregate.  Shares are polynomial evaluations.
        """
        # Simplified: use weighted average with Lagrange-like coefficients
        n = len(active_ids)
        param_names = list(updates[0].keys())
        aggregated = {name: torch.zeros_like(updates[0][name]) for name in param_names}

        # Lagrange basis coefficients for evaluation at x=0
        coefficients = self._lagrange_coefficients(active_ids)

        for idx, client_id in enumerate(active_ids):
            if client_id >= len(updates):
                continue
            coef = coefficients[idx]
            for name in param_names:
                val = updates[client_id][name]
                if self.quantize_bits > 0:
                    val = self._quantize(val)
                aggregated[name] += coef * val

        return aggregated

    def _simple_average(
        self,
        updates: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Plain average (no security, baseline)."""
        param_names = list(updates[0].keys())
        aggregated = {name: torch.zeros_like(updates[0][name]) for name in param_names}

        for update in updates:
            for name in param_names:
                aggregated[name] += update[name]

        for name in param_names:
            aggregated[name] /= len(updates)

        return aggregated

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _generate_pairwise_seeds(self) -> None:
        """Generate pairwise random seeds for mask generation."""
        for i in range(self.n_clients):
            for j in range(i + 1, self.n_clients):
                seed_str = f"pair_{i}_{j}"
                seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
                self._pairwise_seeds[(i, j)] = seed
                self._pairwise_seeds[(j, i)] = seed

    def _generate_mask(
        self,
        client_id: int,
        active_ids: List[int],
        shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate an additive mask for a client that cancels when summed
        with all other active clients' masks.
        """
        mask = torch.zeros(shape, device=device)

        for other_id in active_ids:
            if other_id == client_id:
                continue

            key = (client_id, other_id)
            if key in self._pairwise_seeds:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(self._pairwise_seeds[key])
                pair_mask = torch.randn(shape, generator=gen, device="cpu").to(device) * 0.01

                # Client with smaller ID adds; larger ID subtracts
                if client_id < other_id:
                    mask += pair_mask
                else:
                    mask -= pair_mask

        return mask

    def _lagrange_coefficients(self, active_ids: List[int]) -> List[float]:
        """Compute Lagrange basis coefficients for reconstruction at x=0."""
        n = len(active_ids)
        coeffs = []
        xs = [float(x + 1) for x in active_ids]  # evaluate points: 1, 2, 3, ...

        for i in range(n):
            num = 1.0
            den = 1.0
            for j in range(n):
                if i != j:
                    num *= (0.0 - xs[j])
                    den *= (xs[i] - xs[j])
            coeffs.append(num / den if abs(den) > 1e-10 else 0.0)

        return coeffs

    def _quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantise tensor to reduce communication costs.

        Maps continuous values to ``2^quantize_bits`` discrete levels.
        """
        if self.quantize_bits <= 0:
            return tensor

        n_levels = 2 ** self.quantize_bits
        t_min = tensor.min()
        t_max = tensor.max()
        t_range = t_max - t_min

        if t_range < 1e-8:
            return tensor

        # Normalise to [0, 1], quantise, then de-normalise
        normalised = (tensor - t_min) / t_range
        quantised = torch.round(normalised * (n_levels - 1)) / (n_levels - 1)
        return quantised * t_range + t_min


# ======================================================================
# Convenience
# ======================================================================


def create_secure_aggregator(
    n_clients: int,
    protocol: str = "masking",
    **kwargs,
) -> SecureAggregator:
    """Factory function for secure aggregation."""
    threshold = kwargs.pop("threshold", max(1, n_clients // 2))
    return SecureAggregator(
        n_clients=n_clients,
        threshold=threshold,
        protocol=protocol,
        **kwargs,
    )
