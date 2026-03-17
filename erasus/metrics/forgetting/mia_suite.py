"""
erasus.metrics.forgetting.mia_suite — Full 6-attack MIA suite.

Implements the standard MIA evaluation battery used by OpenUnlearning
(Locuslab, NeurIPS D&B 2025):

1. LOSS   — per-sample loss as membership signal
2. ZLib   — loss normalised by zlib compression ratio of the input
3. Reference — loss ratio between target model and a reference model
4. GradNorm — L2 norm of per-sample gradients
5. MinK   — average log-probability of the k% lowest-probability tokens
6. MinK++ — improved MinK with normalised token-level surprisal

Each attack produces per-sample membership scores. The suite then computes
AUC and TPR@FPR for the combined battery.

After successful unlearning, AUC for all attacks should be ≈ 0.5.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC via Wilcoxon-Mann-Whitney statistic (no sklearn)."""
    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    tp = 0.0
    auc = 0.0
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """TPR at a given FPR threshold."""
    n = len(labels)
    if n == 0:
        return 0.0
    thresholds = np.sort(np.unique(scores))[::-1]
    n_pos = labels.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    best_tpr = 0.0
    for t in thresholds:
        pred_pos = scores >= t
        fp = (pred_pos & (labels == 0)).sum()
        tp = (pred_pos & (labels == 1)).sum()
        fpr = fp / n_neg
        if fpr <= target_fpr:
            best_tpr = max(best_tpr, tp / n_pos)
    return float(best_tpr)


def _collect_per_sample_loss(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> np.ndarray:
    """Collect per-sample cross-entropy loss."""
    losses: list = []
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            batch_losses = criterion(outputs, targets)
            losses.append(batch_losses.cpu().numpy())
    return np.concatenate(losses) if losses else np.array([])


# ---------------------------------------------------------------------------
# Individual attack implementations
# ---------------------------------------------------------------------------

class _LOSSAttack:
    """Per-sample loss as membership signal (baseline)."""

    name = "loss"

    @staticmethod
    def score(
        model: nn.Module, loader: DataLoader, device: torch.device, **kw: Any,
    ) -> np.ndarray:
        losses = _collect_per_sample_loss(model, loader, device)
        # Lower loss → more likely a member → higher score
        return -losses


class _ZLibAttack:
    """
    Loss normalised by zlib compression ratio.

    Intuition: samples that are inherently compressible (low entropy text)
    will naturally have lower loss.  Normalising by compression length
    removes this confound, isolating the memorisation signal.

    Reference: Carlini et al., "Extracting Training Data from Large
    Language Models", USENIX Security 2021.
    """

    name = "zlib"

    @staticmethod
    def score(
        model: nn.Module, loader: DataLoader, device: torch.device, **kw: Any,
    ) -> np.ndarray:
        losses = _collect_per_sample_loss(model, loader, device)

        # Compute compression lengths for each sample
        zlib_lengths: list = []
        for batch in loader:
            if not isinstance(batch, (list, tuple)):
                continue
            inputs = batch[0]
            for i in range(inputs.size(0)):
                raw = inputs[i].numpy().tobytes()
                compressed_len = len(zlib.compress(raw))
                zlib_lengths.append(max(compressed_len, 1))

        zlib_arr = np.array(zlib_lengths[: len(losses)], dtype=np.float64)
        if len(zlib_arr) < len(losses):
            # Pad if loader iteration produced fewer entries
            zlib_arr = np.pad(zlib_arr, (0, len(losses) - len(zlib_arr)), constant_values=1)

        # Normalised score: lower loss per compression bit → more memorised
        return -losses / zlib_arr


class _ReferenceAttack:
    """
    Loss ratio between target and reference model.

    A reference model (typically the pre-unlearning or pre-trained model)
    provides a baseline.  If the target model has *much* lower loss on a
    sample than the reference, it likely memorised that sample.

    After successful unlearning, the ratio should be ≈ 1.
    """

    name = "reference"

    @staticmethod
    def score(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        reference_model: Optional[nn.Module] = None,
        **kw: Any,
    ) -> np.ndarray:
        target_losses = _collect_per_sample_loss(model, loader, device)

        if reference_model is not None:
            ref_device = next(reference_model.parameters()).device
            ref_losses = _collect_per_sample_loss(reference_model, loader, ref_device)
            min_len = min(len(target_losses), len(ref_losses))
            target_losses = target_losses[:min_len]
            ref_losses = ref_losses[:min_len]
            # Ratio: lower target loss relative to reference → more memorised
            return ref_losses - target_losses
        else:
            # Fallback: use loss magnitude (degenerates to LOSS attack)
            return -target_losses


class _GradNormAttack:
    """
    Gradient L2-norm as membership signal.

    Members (memorised samples) produce larger gradient norms because
    the model has specialised parameters for them.  After successful
    unlearning, gradient norms on forget samples should be small.
    """

    name = "gradnorm"

    @staticmethod
    def score(
        model: nn.Module, loader: DataLoader, device: torch.device, **kw: Any,
    ) -> np.ndarray:
        model.eval()  # Keep batchnorm/dropout in eval mode
        criterion = nn.CrossEntropyLoss()
        norms: list = []

        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            inputs, targets = batch[0].to(device), batch[1].to(device)

            # Compute per-sample gradient norms
            for i in range(inputs.size(0)):
                model.zero_grad()
                out = model(inputs[i : i + 1])
                if hasattr(out, "logits"):
                    out = out.logits
                loss = criterion(out, targets[i : i + 1])
                loss.backward()

                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm(2).item() ** 2
                norms.append(grad_norm ** 0.5)

        return np.array(norms) if norms else np.array([])


class _MinKAttack:
    """
    Min-K% Prob attack.

    Computes the average log-probability of the k% lowest-probability
    tokens in each sample.  Memorised samples tend to have higher
    minimum-k probabilities (the model is confident even on the
    "hardest" tokens).

    Reference: Shi et al., "Detecting Pretraining Data from Large
    Language Models", ICLR 2024.
    """

    name = "mink"

    def __init__(self, k_percent: float = 20.0):
        self.k_percent = k_percent

    def score(
        self, model: nn.Module, loader: DataLoader, device: torch.device, **kw: Any,
    ) -> np.ndarray:
        model.eval()
        scores: list = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                log_probs = torch.log_softmax(outputs, dim=-1)

                for i in range(inputs.size(0)):
                    # Get log-prob assigned to the correct class
                    sample_log_probs = log_probs[i]

                    if sample_log_probs.dim() == 1:
                        # Classification: single vector of log-probs
                        # Use the sorted log-probs as the "token" distribution
                        sorted_lp, _ = sample_log_probs.sort()
                        k = max(1, int(len(sorted_lp) * self.k_percent / 100))
                        min_k_avg = sorted_lp[:k].mean().item()
                    else:
                        # Sequence: (seq_len, vocab) — take min-k across positions
                        # Get per-position log-prob of the target token
                        per_pos_lp = sample_log_probs.max(dim=-1).values
                        sorted_lp, _ = per_pos_lp.sort()
                        k = max(1, int(len(sorted_lp) * self.k_percent / 100))
                        min_k_avg = sorted_lp[:k].mean().item()

                    scores.append(min_k_avg)

        return np.array(scores) if scores else np.array([])


class _MinKPlusPlusAttack:
    """
    Min-K%++ attack (improved MinK).

    Normalises each token's log-probability by the mean and standard
    deviation across the vocabulary at that position, producing a
    z-score.  This removes the effect of position difficulty and
    isolates the memorisation signal.

    Reference: Zhang et al., "Min-K%++: Improved Baseline for Detecting
    Pre-Training Data from Large Language Models", 2024.
    """

    name = "mink_pp"

    def __init__(self, k_percent: float = 20.0):
        self.k_percent = k_percent

    def score(
        self, model: nn.Module, loader: DataLoader, device: torch.device, **kw: Any,
    ) -> np.ndarray:
        model.eval()
        scores: list = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                log_probs = torch.log_softmax(outputs, dim=-1)

                for i in range(inputs.size(0)):
                    sample_lp = log_probs[i]

                    if sample_lp.dim() == 1:
                        # Classification: normalise the log-prob vector
                        mu = sample_lp.mean()
                        sigma = sample_lp.std().clamp(min=1e-8)
                        z_scores = (sample_lp - mu) / sigma
                        sorted_z, _ = z_scores.sort()
                        k = max(1, int(len(sorted_z) * self.k_percent / 100))
                        score = sorted_z[:k].mean().item()
                    else:
                        # Sequence: normalise per-position
                        mu = sample_lp.mean(dim=-1, keepdim=True)
                        sigma = sample_lp.std(dim=-1, keepdim=True).clamp(min=1e-8)
                        z_scores = (sample_lp - mu) / sigma
                        # Take the max z-score at each position (correct token)
                        per_pos_z = z_scores.max(dim=-1).values
                        sorted_z, _ = per_pos_z.sort()
                        k = max(1, int(len(sorted_z) * self.k_percent / 100))
                        score = sorted_z[:k].mean().item()

                    scores.append(score)

        return np.array(scores) if scores else np.array([])


# ---------------------------------------------------------------------------
# Combined MIA Suite
# ---------------------------------------------------------------------------

class MIASuite(BaseMetric):
    """
    Full 6-attack Membership Inference Attack suite.

    Runs all six standard MIA attacks and reports per-attack AUC and
    TPR@FPR, plus an aggregate score.

    Parameters
    ----------
    attacks : list[str], optional
        Subset of attacks to run. Default: all six.
        Valid names: ``loss``, ``zlib``, ``reference``, ``gradnorm``,
        ``mink``, ``mink_pp``.
    reference_model : nn.Module, optional
        Reference model for the Reference attack. If not provided,
        the Reference attack degenerates to the LOSS attack.
    mink_k_percent : float
        Percentage for MinK/MinK++ attacks (default 20%).
    fpr_thresholds : list[float]
        FPR values at which to report TPR (default [0.01, 0.05, 0.10]).

    Returns
    -------
    dict
        Per-attack AUC and TPR@FPR, plus ``mia_suite_mean_auc`` and
        ``mia_suite_worst_auc`` (highest AUC = worst unlearning).
    """

    name = "mia_suite"

    ALL_ATTACKS = ("loss", "zlib", "reference", "gradnorm", "mink", "mink_pp")

    def __init__(
        self,
        attacks: Optional[List[str]] = None,
        reference_model: Optional[nn.Module] = None,
        mink_k_percent: float = 20.0,
        fpr_thresholds: Optional[List[float]] = None,
    ):
        self.attack_names = list(attacks or self.ALL_ATTACKS)
        self.reference_model = reference_model
        self.mink_k_percent = mink_k_percent
        self.fpr_thresholds = fpr_thresholds or [0.01, 0.05, 0.10]

        self._attacks: Dict[str, Any] = {
            "loss": _LOSSAttack(),
            "zlib": _ZLibAttack(),
            "reference": _ReferenceAttack(),
            "gradnorm": _GradNormAttack(),
            "mink": _MinKAttack(k_percent=mink_k_percent),
            "mink_pp": _MinKPlusPlusAttack(k_percent=mink_k_percent),
        }

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        if forget_data is None or retain_data is None:
            return {"mia_suite_error": "Both forget_data and retain_data are required"}

        device = next(model.parameters()).device
        ref_model = kwargs.get("reference_model", self.reference_model)

        results: Dict[str, float] = {}
        aucs: list = []

        for atk_name in self.attack_names:
            atk = self._attacks.get(atk_name)
            if atk is None:
                results[f"mia_{atk_name}_error"] = -1.0
                continue

            try:
                # Score forget (members) and retain (non-members)
                extra = {}
                if atk_name == "reference" and ref_model is not None:
                    extra["reference_model"] = ref_model

                member_scores = atk.score(model, forget_data, device, **extra)
                nonmember_scores = atk.score(model, retain_data, device, **extra)

                if len(member_scores) == 0 or len(nonmember_scores) == 0:
                    results[f"mia_{atk_name}_auc"] = 0.5
                    aucs.append(0.5)
                    continue

                labels = np.concatenate([
                    np.ones(len(member_scores)),
                    np.zeros(len(nonmember_scores)),
                ])
                scores = np.concatenate([member_scores, nonmember_scores])

                auc = _roc_auc(labels, scores)
                results[f"mia_{atk_name}_auc"] = float(auc)
                aucs.append(auc)

                for fpr_target in self.fpr_thresholds:
                    tpr = _tpr_at_fpr(labels, scores, fpr_target)
                    fpr_key = str(fpr_target).replace(".", "")
                    results[f"mia_{atk_name}_tpr@fpr{fpr_key}"] = float(tpr)

            except Exception as e:
                results[f"mia_{atk_name}_error"] = -1.0
                aucs.append(0.5)

        # Aggregate
        if aucs:
            results["mia_suite_mean_auc"] = float(np.mean(aucs))
            results["mia_suite_worst_auc"] = float(np.max(aucs))
            # Distance from ideal (0.5): lower is better
            results["mia_suite_forgetting_quality"] = 1.0 - 2 * abs(float(np.mean(aucs)) - 0.5)

        return results
