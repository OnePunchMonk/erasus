"""
Coreset — First-class composable coreset object.

A Coreset wraps the result of selector-based sample selection. It can be
inspected, filtered, combined with other coresets, and passed directly
to unlearners and trainers.

Example
-------
>>> from erasus.core import Coreset
>>> from erasus.selectors import InfluenceSelector, GradientNormSelector
>>>
>>> # Build from a selector
>>> selector = InfluenceSelector()
>>> coreset = Coreset.from_selector(selector, model, data_loader, k=100)
>>> print(coreset.indices)       # inspect selected indices
>>> print(len(coreset))          # 100
>>>
>>> # Combine coresets via set operations
>>> coreset_b = Coreset.from_selector(GradientNormSelector(), model, data_loader, k=100)
>>> combined = coreset.union(coreset_b)
>>> overlap = coreset.intersect(coreset_b)
>>>
>>> # Use directly as a DataLoader
>>> loader = coreset.to_loader(batch_size=32)
>>>
>>> # Pass to unlearner
>>> unlearner = ErasusUnlearner(model=model, strategy="gradient_ascent")
>>> result = unlearner.fit(forget_data=coreset.to_loader(), retain_data=retain_loader)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from erasus.core.base_selector import BaseSelector


@dataclass
class CoresetMetadata:
    """Metadata about how the coreset was constructed."""

    selector_name: Optional[str] = None
    original_size: int = 0
    scores: Optional[List[float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Coreset:
    """
    A first-class coreset object wrapping selected sample indices.

    Parameters
    ----------
    dataset : Dataset
        The underlying dataset from which samples were selected.
    indices : list[int]
        Indices of selected samples in the dataset.
    scores : list[float], optional
        Importance scores for each selected sample (parallel to indices).
    metadata : CoresetMetadata, optional
        How the coreset was constructed.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        scores: Optional[List[float]] = None,
        metadata: Optional[CoresetMetadata] = None,
    ) -> None:
        self._dataset = dataset
        self._indices = list(indices)
        self._scores = list(scores) if scores is not None else None
        self.metadata = metadata or CoresetMetadata(original_size=len(dataset))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_selector(
        cls,
        selector: BaseSelector,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> "Coreset":
        """
        Build a Coreset by running a selector on the given data.

        Parameters
        ----------
        selector : BaseSelector
            The selector to use for sample selection.
        model : nn.Module
            Model used to score samples.
        data_loader : DataLoader
            Data to select from.
        k : int
            Number of samples to select.
        """
        indices = selector.select(model=model, data_loader=data_loader, k=k, **kwargs)
        meta = CoresetMetadata(
            selector_name=selector.__class__.__name__,
            original_size=len(data_loader.dataset),
        )
        return cls(dataset=data_loader.dataset, indices=indices, metadata=meta)

    @classmethod
    def from_indices(
        cls,
        dataset: Dataset,
        indices: List[int],
        scores: Optional[List[float]] = None,
    ) -> "Coreset":
        """Create a Coreset from explicit indices."""
        return cls(dataset=dataset, indices=indices, scores=scores)

    @classmethod
    def from_ratio(
        cls,
        selector: BaseSelector,
        model: nn.Module,
        data_loader: DataLoader,
        ratio: float,
        **kwargs: Any,
    ) -> "Coreset":
        """
        Build a Coreset by selecting a fraction of the dataset.

        Parameters
        ----------
        ratio : float
            Fraction of the dataset to select (0, 1].
        """
        k = max(1, int(len(data_loader.dataset) * ratio))
        return cls.from_selector(selector, model, data_loader, k, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def indices(self) -> List[int]:
        """Indices of selected samples."""
        return list(self._indices)

    @property
    def scores(self) -> Optional[List[float]]:
        """Importance scores (if available)."""
        return list(self._scores) if self._scores is not None else None

    @property
    def dataset(self) -> Dataset:
        """The underlying dataset."""
        return self._dataset

    @property
    def compression_ratio(self) -> float:
        """Ratio of coreset size to original dataset size."""
        if self.metadata.original_size == 0:
            return 0.0
        return len(self) / self.metadata.original_size

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def union(self, other: "Coreset") -> "Coreset":
        """
        Combine two coresets via set union.

        Both must reference the same underlying dataset.
        """
        if self._dataset is not other._dataset:
            raise ValueError("Cannot combine coresets from different datasets.")
        merged_indices = sorted(set(self._indices) | set(other._indices))
        meta = CoresetMetadata(
            selector_name=f"union({self.metadata.selector_name}, {other.metadata.selector_name})",
            original_size=self.metadata.original_size,
        )
        return Coreset(dataset=self._dataset, indices=merged_indices, metadata=meta)

    def intersect(self, other: "Coreset") -> "Coreset":
        """
        Intersect two coresets — keep only samples selected by both.

        Both must reference the same underlying dataset.
        """
        if self._dataset is not other._dataset:
            raise ValueError("Cannot intersect coresets from different datasets.")
        common = sorted(set(self._indices) & set(other._indices))
        meta = CoresetMetadata(
            selector_name=(
                f"intersect({self.metadata.selector_name}, {other.metadata.selector_name})"
            ),
            original_size=self.metadata.original_size,
        )
        return Coreset(dataset=self._dataset, indices=common, metadata=meta)

    def difference(self, other: "Coreset") -> "Coreset":
        """Remove indices present in *other*."""
        if self._dataset is not other._dataset:
            raise ValueError("Cannot diff coresets from different datasets.")
        diff = sorted(set(self._indices) - set(other._indices))
        meta = CoresetMetadata(
            selector_name=(
                f"difference({self.metadata.selector_name}, {other.metadata.selector_name})"
            ),
            original_size=self.metadata.original_size,
        )
        return Coreset(dataset=self._dataset, indices=diff, metadata=meta)

    def filter(self, min_score: float) -> "Coreset":
        """
        Filter the coreset to keep only samples above a score threshold.

        Raises ValueError if scores are not available.
        """
        if self._scores is None:
            raise ValueError("Scores not available — cannot filter by score.")
        kept_idx = []
        kept_scores = []
        for idx, sc in zip(self._indices, self._scores):
            if sc >= min_score:
                kept_idx.append(idx)
                kept_scores.append(sc)
        return Coreset(
            dataset=self._dataset,
            indices=kept_idx,
            scores=kept_scores,
            metadata=CoresetMetadata(
                selector_name=self.metadata.selector_name,
                original_size=self.metadata.original_size,
            ),
        )

    def add(self, indices: Sequence[int]) -> "Coreset":
        """Return a new Coreset with additional indices."""
        merged = sorted(set(self._indices) | set(indices))
        return Coreset(
            dataset=self._dataset,
            indices=merged,
            metadata=self.metadata,
        )

    def remove(self, indices: Sequence[int]) -> "Coreset":
        """Return a new Coreset with specified indices removed."""
        remaining = sorted(set(self._indices) - set(indices))
        return Coreset(
            dataset=self._dataset,
            indices=remaining,
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Quality diagnostics
    # ------------------------------------------------------------------

    def coverage(self, model: nn.Module, batch_size: int = 32) -> float:
        """
        Estimate feature-space coverage of the coreset relative to the
        full dataset using average pairwise cosine similarity.

        Returns a value in [0, 1] where higher means the coreset features
        span more of the full dataset's representation space.
        """
        import torch

        model.eval()
        device = next(model.parameters()).device

        def _get_features(indices: List[int]) -> torch.Tensor:
            subset = Subset(self._dataset, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
            feats = []
            with torch.no_grad():
                for batch in loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x = x.to(device)
                    out = model(x)
                    if out.dim() > 2:
                        out = out.view(out.size(0), -1)
                    feats.append(out)
            return torch.cat(feats, dim=0)

        coreset_feats = _get_features(self._indices)
        # Sample same-sized random subset from full dataset for comparison
        all_indices = list(range(len(self._dataset)))
        n_sample = min(len(all_indices), max(len(self._indices) * 2, 200))
        import random
        sample_indices = random.sample(all_indices, n_sample)
        full_feats = _get_features(sample_indices)

        # Normalize and compute coverage as mean max-cosine-similarity
        coreset_norm = torch.nn.functional.normalize(coreset_feats, dim=1)
        full_norm = torch.nn.functional.normalize(full_feats, dim=1)
        sim_matrix = full_norm @ coreset_norm.T  # (n_full, n_coreset)
        max_sim = sim_matrix.max(dim=1).values  # each full sample's best match
        return float(max_sim.mean().item())

    def redundancy(self) -> float:
        """
        Compute redundancy score based on index overlap density.

        Returns a value in [0, 1]. 0 means all unique indices, higher
        means the coreset contains clusters of nearby indices (potential
        duplicates or highly similar samples).

        Uses score variance as a proxy when scores are available, otherwise
        uses index spacing regularity.
        """
        if len(self._indices) <= 1:
            return 0.0

        if self._scores is not None and len(self._scores) > 1:
            scores = np.array(self._scores)
            # Low variance in scores = high redundancy (all equally important)
            score_range = scores.max() - scores.min()
            if score_range < 1e-12:
                return 1.0
            cv = float(scores.std() / (abs(scores.mean()) + 1e-12))
            # Normalise: low CV -> high redundancy
            return float(max(0.0, 1.0 - min(cv, 1.0)))

        # Fallback: how tightly packed are the indices?
        sorted_idx = sorted(self._indices)
        gaps = [sorted_idx[i + 1] - sorted_idx[i] for i in range(len(sorted_idx) - 1)]
        if not gaps:
            return 0.0
        mean_gap = sum(gaps) / len(gaps)
        expected_gap = self.metadata.original_size / max(len(self._indices), 1)
        if expected_gap < 1e-12:
            return 0.0
        return float(max(0.0, 1.0 - min(mean_gap / expected_gap, 1.0)))

    def overlap_with(self, other: "Coreset") -> float:
        """
        Jaccard similarity between this coreset and another.

        Returns a value in [0, 1].
        """
        s1 = set(self._indices)
        s2 = set(other._indices)
        union_size = len(s1 | s2)
        if union_size == 0:
            return 0.0
        return len(s1 & s2) / union_size

    def diagnostics(self, model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Return a dictionary of quality diagnostics.

        Parameters
        ----------
        model : nn.Module, optional
            If provided, feature-space coverage is computed.

        Returns
        -------
        dict
            Keys: size, original_size, compression_ratio, redundancy,
            and optionally coverage.
        """
        diag: Dict[str, Any] = {
            "size": len(self),
            "original_size": self.metadata.original_size,
            "compression_ratio": self.compression_ratio,
            "redundancy": self.redundancy(),
            "selector": self.metadata.selector_name,
        }
        if self._scores is not None:
            scores_arr = np.array(self._scores)
            diag["score_mean"] = float(scores_arr.mean())
            diag["score_std"] = float(scores_arr.std())
            diag["score_min"] = float(scores_arr.min())
            diag["score_max"] = float(scores_arr.max())
        if model is not None:
            diag["coverage"] = self.coverage(model)
        return diag

    @staticmethod
    def compare(*coresets: "Coreset", model: Optional[nn.Module] = None) -> List[Dict[str, Any]]:
        """
        Side-by-side comparison of multiple coresets.

        Returns a list of diagnostic dicts, one per coreset, plus
        pairwise overlap scores.
        """
        results = []
        for c in coresets:
            d = c.diagnostics(model=model)
            results.append(d)

        # Add pairwise overlaps
        for i, c in enumerate(coresets):
            overlaps = {}
            for j, other in enumerate(coresets):
                if i != j:
                    overlaps[f"overlap_with_{j}"] = c.overlap_with(other)
            results[i]["overlaps"] = overlaps

        return results

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_subset(self) -> Subset:
        """Convert to a ``torch.utils.data.Subset``."""
        return Subset(self._dataset, self._indices)

    def to_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs: Any) -> DataLoader:
        """Create a DataLoader over the coreset samples."""
        return DataLoader(
            self.to_subset(),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __repr__(self) -> str:
        return (
            f"Coreset(size={len(self)}, "
            f"original={self.metadata.original_size}, "
            f"selector={self.metadata.selector_name!r})"
        )

    def __contains__(self, index: int) -> bool:
        return index in set(self._indices)
