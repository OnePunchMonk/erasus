"""
erasus.metrics.forgetting.feature_distance — Embedding distance metrics.

Measures how much the model's internal representations have changed
for forget-set samples after unlearning.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class FeatureDistanceMetric(BaseMetric):
    """
    Compares pre- and post-unlearning embeddings to quantify
    representation shift on forget / retain sets.

    Parameters
    ----------
    original_model : nn.Module, optional
        The model *before* unlearning.  If provided, the metric
        computes distances between original and current embeddings.
    """

    name = "feature_distance"

    def __init__(self, original_model: Optional[nn.Module] = None):
        self.original_model = original_model

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        results: Dict[str, float] = {}

        if self.original_model is not None:
            self.original_model.to(device).eval()

            # Forget set distance
            orig_feats = self._extract(self.original_model, forget_data, device)
            new_feats = self._extract(model, forget_data, device)
            if orig_feats.shape == new_feats.shape and len(orig_feats) > 0:
                results["feature_dist_forget_cosine"] = float(self._cosine_distance(orig_feats, new_feats))
                results["feature_dist_forget_l2"] = float(self._l2_distance(orig_feats, new_feats))
                results["feature_dist_forget_cka"] = float(self._linear_cka(orig_feats, new_feats))

            # Retain set distance (should be small)
            orig_feats_r = self._extract(self.original_model, retain_data, device)
            new_feats_r = self._extract(model, retain_data, device)
            if orig_feats_r.shape == new_feats_r.shape and len(orig_feats_r) > 0:
                results["feature_dist_retain_cosine"] = float(self._cosine_distance(orig_feats_r, new_feats_r))
                results["feature_dist_retain_l2"] = float(self._l2_distance(orig_feats_r, new_feats_r))
                results["feature_dist_retain_cka"] = float(self._linear_cka(orig_feats_r, new_feats_r))
        else:
            # Without original model, compute intra-set distances
            forget_feats = self._extract(model, forget_data, device)
            retain_feats = self._extract(model, retain_data, device)

            if len(forget_feats) > 1:
                results["feature_variance_forget"] = float(np.var(forget_feats, axis=0).mean())
            if len(retain_feats) > 1:
                results["feature_variance_retain"] = float(np.var(retain_feats, axis=0).mean())

        return results

    @staticmethod
    def _extract(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
        """Extract penultimate-layer features."""
        feats = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)

                out = model(inputs)
                if hasattr(out, "logits"):
                    out = out.logits
                elif hasattr(out, "last_hidden_state"):
                    out = out.last_hidden_state.mean(dim=1)
                elif not isinstance(out, torch.Tensor):
                    out = out[0]

                feats.append(out.cpu().numpy())

        return np.concatenate(feats) if feats else np.array([])

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Mean per-sample cosine distance."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        cosine_sim = (a_norm * b_norm).sum(axis=1)
        return float(1.0 - cosine_sim.mean())

    @staticmethod
    def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Mean per-sample L2 distance."""
        return float(np.linalg.norm(a - b, axis=1).mean())

    @staticmethod
    def _linear_cka(a: np.ndarray, b: np.ndarray) -> float:
        """Linear Centered Kernel Alignment (CKA).

        CKA ∈ [0, 1]: 1 means identical representation, 0 fully different.
        """
        n = a.shape[0]
        if n < 2:
            return 1.0

        # Center
        a_c = a - a.mean(axis=0, keepdims=True)
        b_c = b - b.mean(axis=0, keepdims=True)

        # Compute HSIC approximation
        ab = np.linalg.norm(a_c.T @ b_c, "fro") ** 2
        aa = np.linalg.norm(a_c.T @ a_c, "fro") ** 2
        bb = np.linalg.norm(b_c.T @ b_c, "fro") ** 2

        return float(ab / (np.sqrt(aa * bb) + 1e-10))
