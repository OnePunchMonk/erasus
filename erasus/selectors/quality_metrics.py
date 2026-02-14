"""
erasus.selectors.quality_metrics — Coreset quality analysis.

Computes coverage, diversity, influence concentration, and
representativeness metrics for a selected coreset, enabling
principled evaluation of selector quality.

Novel contribution: provides the first unified quality metric
suite for machine unlearning coresets.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class CoresetQualityAnalyzer:
    """
    Evaluate the quality of a coreset selection.

    Metrics Computed
    ----------------
    - **coverage**: Fraction of full-dataset variance explained by coreset.
    - **diversity**: Mean pairwise cosine distance among coreset embeddings.
    - **influence_concentration**: Gini coefficient of influence scores.
    - **representativeness**: Max distance from any full-dataset point to
      its nearest coreset neighbour (k-center objective).
    - **redundancy**: Fraction of coreset samples within ε of another.
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_layer: Optional[str] = None,
    ) -> None:
        self.model = model
        self.embedding_layer = embedding_layer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        full_loader: DataLoader,
        coreset_indices: List[int],
        influence_scores: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Run the full quality analysis suite.

        Parameters
        ----------
        full_loader : DataLoader
            DataLoader over the *entire* candidate set.
        coreset_indices : list[int]
            Indices selected as the coreset.
        influence_scores : np.ndarray, optional
            Pre-computed per-sample influence scores (for Gini).

        Returns
        -------
        dict
            Quality metrics dictionary.
        """
        embeddings = self._extract_embeddings(full_loader)
        coreset_embs = embeddings[coreset_indices]

        results: Dict[str, float] = {
            "n_coreset": float(len(coreset_indices)),
            "n_total": float(len(embeddings)),
            "selection_ratio": len(coreset_indices) / max(len(embeddings), 1),
        }

        results["coverage"] = self._coverage(embeddings, coreset_embs)
        results["diversity"] = self._diversity(coreset_embs)
        results["representativeness"] = self._representativeness(embeddings, coreset_embs)
        results["redundancy"] = self._redundancy(coreset_embs)

        if influence_scores is not None:
            results["influence_gini"] = self._gini_coefficient(influence_scores)
            results["influence_top10_share"] = self._top_k_share(influence_scores, k=0.1)

        return results

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def _coverage(self, full: np.ndarray, coreset: np.ndarray) -> float:
        """
        Variance explained by coreset relative to full set.

        Uses the ratio of trace of coreset covariance over full covariance.
        """
        full_var = np.var(full, axis=0).sum()
        if full_var < 1e-12:
            return 1.0
        coreset_var = np.var(coreset, axis=0).sum()
        return float(min(coreset_var / full_var, 1.0))

    def _diversity(self, coreset: np.ndarray) -> float:
        """Mean pairwise cosine distance among coreset embeddings."""
        n = len(coreset)
        if n < 2:
            return 0.0

        # Normalise
        norms = np.linalg.norm(coreset, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normed = coreset / norms

        # Cosine similarity matrix
        sim = normed @ normed.T
        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mean_sim = sim[mask].mean()
        # Distance = 1 - similarity
        return float(1.0 - mean_sim)

    def _representativeness(self, full: np.ndarray, coreset: np.ndarray) -> float:
        """
        Max distance from any full-set point to its nearest coreset point.
        Lower is better (k-center objective).
        """
        # For efficiency, subsample full set if too large
        if len(full) > 5000:
            idx = np.random.choice(len(full), 5000, replace=False)
            full = full[idx]

        # Compute pairwise distances in chunks
        max_dist = 0.0
        chunk_size = 500
        for i in range(0, len(full), chunk_size):
            chunk = full[i:i + chunk_size]
            # (chunk_size, 1, d) - (1, coreset_size, d)
            diffs = chunk[:, None, :] - coreset[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            min_dists = dists.min(axis=1)
            max_dist = max(max_dist, float(min_dists.max()))

        return max_dist

    def _redundancy(self, coreset: np.ndarray, epsilon: float = 0.01) -> float:
        """Fraction of coreset samples within ε-ball of another."""
        n = len(coreset)
        if n < 2:
            return 0.0

        norms = np.linalg.norm(coreset, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normed = coreset / norms

        sim = normed @ normed.T
        np.fill_diagonal(sim, 0.0)
        # Count samples with at least one near-duplicate
        redundant = (sim > (1.0 - epsilon)).any(axis=1).sum()
        return float(redundant / n)

    @staticmethod
    def _gini_coefficient(scores: np.ndarray) -> float:
        """Gini coefficient of influence scores (0 = equal, 1 = concentrated)."""
        scores = np.abs(scores)
        if scores.sum() < 1e-12:
            return 0.0
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)
        index = np.arange(1, n + 1)
        return float((2 * (index * sorted_scores).sum() / (n * sorted_scores.sum())) - (n + 1) / n)

    @staticmethod
    def _top_k_share(scores: np.ndarray, k: float = 0.1) -> float:
        """Share of total influence held by the top k% of samples."""
        scores = np.abs(scores)
        total = scores.sum()
        if total < 1e-12:
            return 0.0
        n_top = max(1, int(len(scores) * k))
        top_sum = np.sort(scores)[-n_top:].sum()
        return float(top_sum / total)

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_embeddings(self, loader: DataLoader) -> np.ndarray:
        """Extract embeddings from the model."""
        device = next(self.model.parameters()).device
        self.model.eval()

        embeddings: list = []
        hook_output: list = []

        # Register hook on target layer if specified
        handle = None
        if self.embedding_layer:
            for name, module in self.model.named_modules():
                if name == self.embedding_layer:
                    handle = module.register_forward_hook(
                        lambda m, inp, out: hook_output.append(
                            out.detach().cpu() if isinstance(out, torch.Tensor)
                            else out[0].detach().cpu()
                        )
                    )
                    break

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                outputs = self.model(inputs)

                if handle and hook_output:
                    emb = hook_output[-1]
                    hook_output.clear()
                else:
                    # Use model output directly
                    if hasattr(outputs, "last_hidden_state"):
                        emb = outputs.last_hidden_state.mean(dim=1).cpu()
                    elif hasattr(outputs, "logits"):
                        emb = outputs.logits.cpu()
                    elif isinstance(outputs, torch.Tensor):
                        emb = outputs.cpu()
                        if emb.dim() > 2:
                            emb = emb.flatten(start_dim=1)
                    else:
                        emb = outputs[0].cpu() if isinstance(outputs, tuple) else torch.tensor([0.0])

                embeddings.append(emb.numpy() if isinstance(emb, torch.Tensor) else emb)

        if handle:
            handle.remove()

        return np.concatenate(embeddings, axis=0)
