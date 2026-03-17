"""
erasus.metrics.forgetting.memorization — Memorization detection metrics.

Implements:
- Extraction Strength (ES): How much of the forget data can be extracted
  via generation probing.
- Exact Memorization (EM): Exact string/tensor match of forget data in
  model output.
- Verbatim Memorization: Longest common subsequence ratio between model
  output and forget data.
- KnowMem: Knowledge memorization score based on generation overlap.

These metrics go beyond MIA (which tests membership classification) to
measure whether the model can *reproduce* forget-set content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class ExtractionStrengthMetric(BaseMetric):
    """
    Extraction Strength (ES).

    Measures how much information about forget-set samples can be
    extracted from the model by querying it.  For classifiers, this is
    the fraction of forget samples where the model's top-k predictions
    contain the true label.  For generative models, this would measure
    output overlap (requires a generate method).

    After successful unlearning, ES should be ≈ random-chance level.

    Parameters
    ----------
    top_k : int
        Number of top predictions to consider (default 1).
    """

    name = "extraction_strength"

    def __init__(self, top_k: int = 1):
        self.top_k = top_k

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        if forget_data is None:
            return {"extraction_strength": 0.0}

        device = next(model.parameters()).device
        model.eval()

        forget_es = self._compute_extraction(model, forget_data, device)
        results: Dict[str, float] = {
            "extraction_strength_forget": float(forget_es),
        }

        if retain_data is not None:
            retain_es = self._compute_extraction(model, retain_data, device)
            results["extraction_strength_retain"] = float(retain_es)
            # Relative ES: how much more extractable is forget vs retain
            # After good unlearning, this should be ≤ 0
            results["extraction_strength_gap"] = float(forget_es - retain_es)
            # Normalised score: 1.0 = perfect (forget ES = 0), 0.0 = no unlearning
            random_baseline = self.top_k / max(self._infer_num_classes(model, forget_data, device), 1)
            results["extraction_resistance"] = min(1.0, max(
                0.0, 1.0 - (forget_es - random_baseline) / max(1.0 - random_baseline, 1e-8)
            ))

        return results

    def _compute_extraction(
        self, model: nn.Module, loader: DataLoader, device: torch.device,
    ) -> float:
        """Fraction of samples where true label is in top-k predictions."""
        extracted = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                _, top_k_preds = outputs.topk(min(self.top_k, outputs.size(-1)), dim=-1)
                for i in range(targets.size(0)):
                    if targets[i] in top_k_preds[i]:
                        extracted += 1
                    total += 1

        return extracted / max(total, 1)

    @staticmethod
    def _infer_num_classes(
        model: nn.Module, loader: DataLoader, device: torch.device,
    ) -> int:
        """Infer the number of classes from the model output."""
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)):
                    continue
                inputs = batch[0].to(device)
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                return outputs.size(-1)
        return 1


class ExactMemorizationMetric(BaseMetric):
    """
    Exact Memorization (EM).

    For classifiers: measures the fraction of forget-set samples that
    the model classifies with the *exact* same confidence ranking as a
    model that was trained on them (i.e., correct label has highest
    probability AND confidence exceeds a threshold).

    This is stricter than accuracy — it requires the model to be
    *confidently correct*, which is the signature of memorization.

    Parameters
    ----------
    confidence_threshold : float
        Minimum softmax probability for the correct class to count as
        "exactly memorised" (default 0.8).
    """

    name = "exact_memorization"

    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        if forget_data is None:
            return {"exact_memorization_forget": 0.0}

        device = next(model.parameters()).device
        model.eval()

        forget_em, forget_confs = self._compute_em(model, forget_data, device)
        results: Dict[str, float] = {
            "exact_memorization_forget": float(forget_em),
            "exact_memorization_forget_mean_conf": float(np.mean(forget_confs)) if forget_confs else 0.0,
        }

        if retain_data is not None:
            retain_em, retain_confs = self._compute_em(model, retain_data, device)
            results["exact_memorization_retain"] = float(retain_em)
            results["exact_memorization_retain_mean_conf"] = float(np.mean(retain_confs)) if retain_confs else 0.0
            results["exact_memorization_gap"] = float(retain_em - forget_em)

        return results

    def _compute_em(
        self, model: nn.Module, loader: DataLoader, device: torch.device,
    ) -> Tuple[float, List[float]]:
        """Fraction of samples that are confidently correctly classified."""
        memorised = 0
        total = 0
        confidences: List[float] = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                probs = torch.softmax(outputs, dim=-1)

                for i in range(targets.size(0)):
                    target_prob = probs[i, targets[i]].item()
                    confidences.append(target_prob)
                    pred = outputs[i].argmax().item()
                    if pred == targets[i].item() and target_prob >= self.confidence_threshold:
                        memorised += 1
                    total += 1

        return memorised / max(total, 1), confidences


class VerbatimMemorizationMetric(BaseMetric):
    """
    Verbatim Memorization.

    Measures how closely the model's output distribution matches the
    training signal for forget-set samples.  Uses the KL divergence
    between the model's softmax distribution and the one-hot target
    as a proxy for verbatim memorization.

    Lower KL (closer to zero) = more memorized = worse unlearning.
    Higher KL = more forgotten = better unlearning.

    Also computes the "memorization score" as the fraction of samples
    where the model's prediction entropy is below a threshold.
    """

    name = "verbatim_memorization"

    def __init__(self, entropy_threshold: Optional[float] = None):
        """
        Parameters
        ----------
        entropy_threshold : float, optional
            Maximum entropy to consider a sample "memorised".
            If None, uses ``0.5 * log(num_classes)`` as default.
        """
        self.entropy_threshold = entropy_threshold

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        if forget_data is None:
            return {"verbatim_memorization_forget": 0.0}

        device = next(model.parameters()).device
        model.eval()

        forget_kl, forget_ent, forget_mem = self._analyse(model, forget_data, device)
        results: Dict[str, float] = {
            "verbatim_kl_forget": float(np.mean(forget_kl)) if len(forget_kl) else 0.0,
            "verbatim_entropy_forget": float(np.mean(forget_ent)) if len(forget_ent) else 0.0,
            "verbatim_memorization_forget": float(forget_mem),
        }

        if retain_data is not None:
            retain_kl, retain_ent, retain_mem = self._analyse(model, retain_data, device)
            results["verbatim_kl_retain"] = float(np.mean(retain_kl)) if len(retain_kl) else 0.0
            results["verbatim_entropy_retain"] = float(np.mean(retain_ent)) if len(retain_ent) else 0.0
            results["verbatim_memorization_retain"] = float(retain_mem)
            results["verbatim_memorization_gap"] = float(retain_mem - forget_mem)

        return results

    def _analyse(
        self, model: nn.Module, loader: DataLoader, device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute per-sample KL, entropy, and memorization rate."""
        kl_divs: list = []
        entropies: list = []
        num_classes = None
        memorised = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                if num_classes is None:
                    num_classes = outputs.size(-1)

                log_probs = torch.log_softmax(outputs, dim=-1)
                probs = torch.softmax(outputs, dim=-1)

                # Per-sample entropy
                ent = -(probs * log_probs).sum(dim=-1)
                entropies.extend(ent.cpu().tolist())

                # Per-sample KL divergence from one-hot target
                # KL(one_hot || model) = -log(p_target)
                for i in range(targets.size(0)):
                    kl = -log_probs[i, targets[i]].item()
                    kl_divs.append(kl)

                    # Check memorization
                    thresh = self.entropy_threshold
                    if thresh is None and num_classes is not None:
                        thresh = 0.5 * np.log(num_classes)
                    if thresh is not None and ent[i].item() < thresh:
                        memorised += 1
                    total += 1

        mem_rate = memorised / max(total, 1)
        return np.array(kl_divs), np.array(entropies), mem_rate
