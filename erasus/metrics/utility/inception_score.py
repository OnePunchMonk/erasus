"""
erasus.metrics.utility.inception_score — Inception Score for generative models.

Measures quality and diversity of generated images after unlearning
from diffusion models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("inception_score")
class InceptionScoreMetric(BaseMetric):
    """
    Inception Score (IS) for image generation quality.

    IS = exp(E_x[KL(p(y|x) || p(y))])

    Higher IS indicates:
    - High quality (confident predictions per image)
    - High diversity (uniform marginal across classes)

    Parameters
    ----------
    n_splits : int
        Number of splits for computing mean/std.
    classifier : nn.Module, optional
        Custom classifier. If None, uses the evaluation model.
    """

    def __init__(
        self,
        n_splits: int = 10,
        classifier: Optional[nn.Module] = None,
    ) -> None:
        self.n_splits = n_splits
        self.classifier = classifier

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute Inception Score on generated images.

        ``forget_loader`` should contain generated images from the
        unlearned model.
        """
        classifier = self.classifier or model
        device = next(classifier.parameters()).device
        classifier.eval()

        all_probs: list = []

        with torch.no_grad():
            for batch in forget_loader:
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                outputs = classifier(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        if not all_probs:
            return {"inception_score_mean": 0.0, "inception_score_std": 0.0}

        probs = np.concatenate(all_probs, axis=0)
        scores = self._compute_is_splits(probs)

        results: Dict[str, float] = {
            "inception_score_mean": float(np.mean(scores)),
            "inception_score_std": float(np.std(scores)),
        }

        # Also compute on retain if available
        if retain_loader is not None:
            retain_probs: list = []
            with torch.no_grad():
                for batch in retain_loader:
                    inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    outputs = classifier(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    p = F.softmax(logits, dim=-1)
                    retain_probs.append(p.cpu().numpy())

            if retain_probs:
                rp = np.concatenate(retain_probs, axis=0)
                retain_scores = self._compute_is_splits(rp)
                results["retain_inception_score_mean"] = float(np.mean(retain_scores))
                results["retain_inception_score_std"] = float(np.std(retain_scores))

        return results

    def _compute_is_splits(self, probs: np.ndarray) -> np.ndarray:
        """Compute IS across multiple splits."""
        n = len(probs)
        split_size = n // self.n_splits
        scores = []

        for i in range(self.n_splits):
            start = i * split_size
            end = start + split_size if i < self.n_splits - 1 else n
            split = probs[start:end]

            if len(split) == 0:
                continue

            # p(y) = marginal
            marginal = split.mean(axis=0, keepdims=True)
            # KL(p(y|x) || p(y))
            kl = split * (np.log(split + 1e-10) - np.log(marginal + 1e-10))
            kl_mean = kl.sum(axis=1).mean()
            scores.append(math.exp(kl_mean))

        return np.array(scores) if scores else np.array([1.0])
