"""
erasus.selectors.ensemble.weighted_fusion — Weighted combination of selectors.

Combines multiple coreset selectors with configurable weights,
normalising per-selector scores and producing a fused ranking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("weighted_fusion")
class WeightedFusionSelector(BaseSelector):
    """
    Fuse multiple selectors via weighted score combination.

    Parameters
    ----------
    selectors : list[BaseSelector]
        Selector instances to combine.
    weights : list[float], optional
        Per-selector weights (normalised automatically). Equal by default.
    normalisation : str
        Score normalisation: ``"minmax"`` (default) or ``"zscore"``.
    """

    def __init__(
        self,
        selectors: Sequence[BaseSelector],
        weights: Optional[Sequence[float]] = None,
        normalisation: str = "minmax",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not selectors:
            raise ValueError("At least one selector is required.")
        self.selectors = list(selectors)
        raw_weights = list(weights) if weights else [1.0] * len(self.selectors)
        total = sum(raw_weights) or 1.0
        self.weights = [w / total for w in raw_weights]
        self.normalisation = normalisation

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        """Select *k* samples using weighted score fusion."""
        all_scores: List[np.ndarray] = []

        for sel in self.selectors:
            scores = self._get_selector_scores(sel, model, data_loader, k, **kwargs)
            all_scores.append(scores)

        fused = self._fuse_scores(all_scores)
        top_k = np.argsort(fused)[-k:].tolist()
        return top_k

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_selector_scores(
        self,
        selector: BaseSelector,
        model: nn.Module,
        loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Get per-sample scores from a selector.

        If the selector exposes a ``compute_scores`` method, use it.
        Otherwise, fall back to ranking from ``select()``.
        """
        if hasattr(selector, "compute_scores"):
            return np.array(selector.compute_scores(model, loader, **kwargs))
        if hasattr(selector, "compute_influence_scores"):
            return selector.compute_influence_scores(model, loader)

        # Fallback: use selection indices to infer implicit score
        n_samples = sum(
            len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch)
            for batch in loader
        )
        selected = set(selector.select(model, loader, k, **kwargs))
        scores = np.array([1.0 if i in selected else 0.0 for i in range(n_samples)])
        return scores

    def _fuse_scores(self, all_scores: List[np.ndarray]) -> np.ndarray:
        """Normalise and combine scores."""
        normalised = [self._normalise(s) for s in all_scores]
        fused = np.zeros_like(normalised[0])
        for w, s in zip(self.weights, normalised):
            fused += w * s
        return fused

    def _normalise(self, scores: np.ndarray) -> np.ndarray:
        """Normalise scores to [0, 1]."""
        if self.normalisation == "zscore":
            mu = scores.mean()
            std = scores.std() + 1e-8
            return (scores - mu) / std
        else:  # minmax
            lo, hi = scores.min(), scores.max()
            rng = hi - lo
            if rng < 1e-12:
                return np.zeros_like(scores)
            return (scores - lo) / rng
