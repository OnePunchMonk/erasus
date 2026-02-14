"""
erasus.selectors.learning_based.active_learning — Uncertainty-based active selection.

Uses prediction uncertainty (entropy, MC dropout, margin) to prioritise
the most informative samples for unlearning — the ones the model is
most uncertain about after seeing the forget set.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("active_learning")
class ActiveLearningSelector(BaseSelector):
    """
    Select samples using prediction uncertainty.

    Strategies
    ----------
    - ``entropy``: Select samples with highest predictive entropy.
    - ``margin``: Select samples with smallest margin between top-2 classes.
    - ``mc_dropout``: Monte Carlo dropout-based epistemic uncertainty.
    - ``bald``: Bayesian Active Learning by Disagreement.

    Parameters
    ----------
    method : str
        Uncertainty estimation method.
    n_mc_samples : int
        Number of MC dropout forward passes (for ``mc_dropout`` / ``bald``).
    """

    METHODS = ("entropy", "margin", "mc_dropout", "bald")

    def __init__(
        self,
        method: str = "entropy",
        n_mc_samples: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.METHODS}")
        self.method = method
        self.n_mc_samples = n_mc_samples

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        """Select *k* most uncertain samples."""
        scores = self._compute_uncertainty(model, data_loader)
        # Higher uncertainty → select first
        top_k = np.argsort(scores)[-k:].tolist()
        return top_k

    # ------------------------------------------------------------------
    # Uncertainty computation
    # ------------------------------------------------------------------

    def _compute_uncertainty(
        self, model: nn.Module, loader: DataLoader
    ) -> np.ndarray:
        """Dispatch to the chosen uncertainty method."""
        if self.method == "entropy":
            return self._entropy_scores(model, loader)
        elif self.method == "margin":
            return self._margin_scores(model, loader)
        elif self.method == "mc_dropout":
            return self._mc_dropout_scores(model, loader)
        elif self.method == "bald":
            return self._bald_scores(model, loader)
        raise ValueError(f"Unknown method: {self.method}")

    def _entropy_scores(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        """Predictive entropy: H[p(y|x)]."""
        device = next(model.parameters()).device
        model.eval()
        scores: list = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                scores.extend(entropy.cpu().numpy().tolist())

        return np.array(scores)

    def _margin_scores(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        """Margin uncertainty: 1 - (p_top1 - p_top2). Smaller margin → more uncertain."""
        device = next(model.parameters()).device
        model.eval()
        scores: list = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = F.softmax(logits, dim=-1)
                top2 = probs.topk(min(2, probs.size(-1)), dim=-1).values
                if top2.size(-1) >= 2:
                    margin = top2[:, 0] - top2[:, 1]
                else:
                    margin = torch.ones(len(probs), device=device)
                # Invert: smaller margin → higher score
                uncertainty = 1.0 - margin
                scores.extend(uncertainty.cpu().numpy().tolist())

        return np.array(scores)

    def _mc_dropout_scores(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        """MC Dropout: variance of predictions across stochastic passes."""
        device = next(model.parameters()).device
        # Enable dropout during inference
        model.train()

        all_probs: list = []
        for _ in range(self.n_mc_samples):
            sample_probs: list = []
            with torch.no_grad():
                for batch in loader:
                    inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    probs = F.softmax(logits, dim=-1)
                    sample_probs.append(probs.cpu().numpy())
            all_probs.append(np.concatenate(sample_probs, axis=0))

        model.eval()

        # Shape: (n_mc, n_samples, n_classes)
        stacked = np.stack(all_probs, axis=0)
        # Predictive variance: mean variance across classes
        variance = stacked.var(axis=0).mean(axis=-1)
        return variance

    def _bald_scores(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        """BALD: I[y; θ|x] = H[y|x] - E_θ[H[y|x,θ]]."""
        device = next(model.parameters()).device
        model.train()

        all_probs: list = []
        for _ in range(self.n_mc_samples):
            sample_probs: list = []
            with torch.no_grad():
                for batch in loader:
                    inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    probs = F.softmax(logits, dim=-1)
                    sample_probs.append(probs.cpu().numpy())
            all_probs.append(np.concatenate(sample_probs, axis=0))

        model.eval()

        stacked = np.stack(all_probs, axis=0)  # (T, N, C)
        # Predictive entropy: H[y|x]
        mean_probs = stacked.mean(axis=0)
        predictive_entropy = -(mean_probs * np.log(mean_probs + 1e-10)).sum(axis=-1)

        # Expected conditional entropy: E_θ[H[y|x,θ]]
        conditional_entropy = -(stacked * np.log(stacked + 1e-10)).sum(axis=-1).mean(axis=0)

        # BALD = H[y|x] - E_θ[H[y|x,θ]]
        bald = predictive_entropy - conditional_entropy
        return bald
