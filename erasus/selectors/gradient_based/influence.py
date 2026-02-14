"""
Influence Function Selector — LiSSA approximation.

Paper: Understanding Black-box Predictions via Influence Functions
       (Koh & Liang, ICML 2017)

Section 3.1.1.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("influence")
class InfluenceSelector(BaseSelector):
    """
    Select samples with the highest influence on model predictions.

    Methods
    -------
    - **exact**: Full Hessian computation (expensive, O(n³ + n²d))
    - **lissa**: Linear-time Stochastic Second-order Algorithm (default)
    - **arnoldi**: Arnoldi iteration for Hessian approximation

    Hyperparameters
    ---------------
    damping : float
        0.001 – 0.1 for conditioning the Hessian.
    num_samples : int
        Recursion depth for LiSSA (5000 – 10000).
    """

    def __init__(
        self,
        approximation: str = "lissa",
        damping: float = 0.01,
        num_samples: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.approximation = approximation
        self.damping = damping
        self.num_samples = num_samples

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        """Select *k* most influential samples via influence scores."""
        scores = self.compute_influence_scores(model, data_loader)
        # Higher influence → more important to forget
        top_k_indices = np.argsort(scores)[-k:].tolist()
        return top_k_indices

    def compute_influence_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> np.ndarray:
        """
        Compute per-sample influence scores using LiSSA.

        Influence: I(z) = -∇L(z_test, θ*)ᵀ H⁻¹ ∇L(z, θ*)
        """
        device = next(model.parameters()).device
        model.eval()

        # Compute a reference gradient (mean gradient over a sample of the data)
        ref_grad = self._compute_mean_gradient(model, data_loader, device)

        # Approximate H⁻¹ v using LiSSA
        ihvp = self._lissa_ihvp(model, data_loader, ref_grad, device)

        # Score each sample by dot-product with ihvp
        scores = self._score_samples(model, data_loader, ihvp, device)
        return scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_mean_gradient(
        self, model: nn.Module, loader: DataLoader, device
    ) -> torch.Tensor:
        """Compute mean gradient over a subsample."""
        grads: list = []
        for i, batch in enumerate(loader):
            if i >= 10:  # limit for efficiency
                break
            g = self._batch_gradient(model, batch, device)
            grads.append(g)
        return torch.stack(grads).mean(dim=0)

    def _lissa_ihvp(
        self,
        model: nn.Module,
        loader: DataLoader,
        v: torch.Tensor,
        device,
    ) -> torch.Tensor:
        """Iteratively approximate H⁻¹ v."""
        cur_estimate = v.clone()
        batches = list(loader)

        for _ in range(self.num_samples):
            batch = random.choice(batches)
            hvp = self._hessian_vector_product(model, batch, cur_estimate, device)
            cur_estimate = v + (1 - self.damping) * cur_estimate - hvp / len(batches)

        return cur_estimate

    def _score_samples(
        self, model: nn.Module, loader: DataLoader, ihvp: torch.Tensor, device
    ) -> np.ndarray:
        """Score each sample by -(grad ⊙ ihvp)."""
        all_scores: list = []
        for batch in loader:
            g = self._batch_gradient(model, batch, device)
            score = -torch.dot(g.flatten(), ihvp.flatten()).item()
            all_scores.append(score)
        return np.array(all_scores)

    @staticmethod
    def _batch_gradient(model: nn.Module, batch, device) -> torch.Tensor:
        """Compute flattened gradient for one batch. Same layout as HVP (all params, zeros if unused)."""
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

        grads = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        return torch.cat(grads)

    @staticmethod
    def _hessian_vector_product(
        model: nn.Module, batch, v: torch.Tensor, device
    ) -> torch.Tensor:
        """Compute Hv via finite differences or auto-diff."""
        model.zero_grad()
        inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        labels = batch[1].to(device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else logits.sum()

        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        flat_grads = torch.cat([
            g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
            for g, p in zip(grads, params)
        ])

        grad_v = torch.dot(flat_grads, v)
        hvp_grads = torch.autograd.grad(grad_v, params, allow_unused=True)
        return torch.cat([
            g.detach().flatten() if g is not None else torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
            for g, p in zip(hvp_grads, params)
        ])
