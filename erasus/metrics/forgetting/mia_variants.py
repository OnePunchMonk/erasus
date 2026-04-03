"""
erasus.metrics.forgetting.mia_variants — Advanced MIA variants.

Includes:
- LiRA (Likelihood Ratio Attack) — Carlini et al., 2022
- Label-Only MIA — Choquette-Choo et al., 2021
- Reference MIA — loss ratio between target and reference model
- GradNorm MIA — gradient norm of loss w.r.t. the model parameters
- ZLib MIA — loss normalised by zlib compression ratio
- Min-K% Prob — average log-probability of the K% lowest-probability tokens
- Min-K%++ — calibrated Min-K with token-level normalisation
"""

from __future__ import annotations

import zlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class LiRAMetric(BaseMetric):
    """
    Likelihood Ratio Attack (LiRA).

    Computes membership scores by comparing the target model's
    confidence against a population of shadow models.

    Since training shadow models is expensive, this implementation
    supports pre-computed shadow confidences.

    Parameters
    ----------
    shadow_in_confidences : np.ndarray, optional
        Shape ``(n_shadows, n_samples)`` — confidences from shadow
        models that included the sample in training.
    shadow_out_confidences : np.ndarray, optional
        Shape ``(n_shadows, n_samples)`` — confidences from shadow
        models that excluded the sample.
    """

    name = "lira"

    def __init__(
        self,
        shadow_in_confidences: Optional[np.ndarray] = None,
        shadow_out_confidences: Optional[np.ndarray] = None,
    ):
        self.shadow_in = shadow_in_confidences
        self.shadow_out = shadow_out_confidences

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        # Collect target model confidences on forget set
        target_confs = self._collect_confidences(model, forget_data, device)

        if self.shadow_in is not None and self.shadow_out is not None:
            # Full LiRA: compare against shadow distributions
            scores = self._compute_lira_scores(target_confs)
        else:
            # Simplified LiRA: use confidence magnitude as proxy
            scores = target_confs

        # Higher score → more likely a member
        mean_score = float(np.mean(scores))

        # Compute simple AUC against retain set
        retain_confs = self._collect_confidences(model, retain_data, device)
        labels = np.concatenate([np.ones(len(scores)), np.zeros(len(retain_confs))])
        all_scores = np.concatenate([scores, retain_confs])
        auc = self._simple_auc(labels, all_scores)

        return {
            "lira_mean_score": mean_score,
            "lira_auc": float(auc),
        }

    def _compute_lira_scores(self, target_confs: np.ndarray) -> np.ndarray:
        """Compute LiRA likelihood ratio scores."""
        n_samples = min(len(target_confs), self.shadow_in.shape[1])
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Fit Gaussians to in/out shadow confidences
            in_mean = self.shadow_in[:, i].mean()
            in_std = max(self.shadow_in[:, i].std(), 1e-10)
            out_mean = self.shadow_out[:, i].mean()
            out_std = max(self.shadow_out[:, i].std(), 1e-10)

            # Log-likelihood ratio
            log_p_in = -0.5 * ((target_confs[i] - in_mean) / in_std) ** 2
            log_p_out = -0.5 * ((target_confs[i] - out_mean) / out_std) ** 2
            scores[i] = log_p_in - log_p_out

        return scores

    @staticmethod
    def _collect_confidences(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> np.ndarray:
        """Collect max-softmax confidences per sample."""
        confs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)

                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                probs = torch.softmax(outputs, dim=-1)
                max_probs = probs.max(dim=-1).values
                confs.append(max_probs.cpu().numpy())

        return np.concatenate(confs) if confs else np.array([])

    @staticmethod
    def _simple_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        """Quick AUC via sorting."""
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp = 0.0
        auc = 0.0
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                auc += tp
        return auc / (n_pos * n_neg)


class LabelOnlyMIAMetric(BaseMetric):
    """
    Label-Only Membership Inference Attack.

    Uses only predicted labels (not confidence scores) to determine
    membership. Counts misclassification rates: forget samples should
    be misclassified after successful unlearning.
    """

    name = "label_only_mia"

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        forget_acc = self._compute_accuracy(model, forget_data, device)
        retain_acc = self._compute_accuracy(model, retain_data, device)

        # After good unlearning: forget_acc should be LOW, retain_acc HIGH
        # MIA signal: accuracy gap
        gap = retain_acc - forget_acc

        return {
            "label_mia_forget_accuracy": float(forget_acc),
            "label_mia_retain_accuracy": float(retain_acc),
            "label_mia_accuracy_gap": float(gap),
        }

    @staticmethod
    def _compute_accuracy(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    continue
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1)


class ZLibMIAMetric(BaseMetric):
    """
    ZLib-normalised membership inference attack.

    Uses per-sample loss divided by compressed input length to reduce the
    bias toward naturally low-entropy samples.
    """

    name = "zlib_mia"

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        forget_scores = self._collect_scores(model, forget_data, device)
        retain_scores = self._collect_scores(model, retain_data, device)
        labels = np.concatenate([np.ones(len(forget_scores)), np.zeros(len(retain_scores))])
        scores = np.concatenate([forget_scores, retain_scores])

        return {
            "zlib_mia_forget_mean": float(np.mean(forget_scores)) if len(forget_scores) else 0.0,
            "zlib_mia_retain_mean": float(np.mean(retain_scores)) if len(retain_scores) else 0.0,
            "zlib_mia_auc": float(LiRAMetric._simple_auc(labels, scores)),
        }

    def _collect_scores(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> np.ndarray:
        criterion = nn.CrossEntropyLoss(reduction="none")
        collected: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue

                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                losses = criterion(logits, targets).cpu().numpy()

                z_lengths: List[float] = []
                raw_inputs = batch[0].detach().cpu()
                for sample in raw_inputs:
                    compressed = len(zlib.compress(sample.numpy().tobytes()))
                    z_lengths.append(float(max(compressed, 1)))

                z_arr = np.array(z_lengths, dtype=np.float64)
                collected.append(-(losses / z_arr))

        return np.concatenate(collected) if collected else np.array([])


class ReferenceMIAMetric(BaseMetric):
    """
    Reference-model membership inference attack.

    Scores each sample by comparing the target model loss against a
    reference model loss on the same data. Lower target loss relative to
    the reference indicates stronger memorization.
    """

    name = "reference_mia"

    def __init__(self, reference_model: Optional[nn.Module] = None) -> None:
        self.reference_model = reference_model

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        reference_model = kwargs.get("reference_model", self.reference_model)
        device = next(model.parameters()).device

        forget_scores = self._collect_scores(model, forget_data, device, reference_model)
        retain_scores = self._collect_scores(model, retain_data, device, reference_model)
        labels = np.concatenate([np.ones(len(forget_scores)), np.zeros(len(retain_scores))])
        scores = np.concatenate([forget_scores, retain_scores])

        return {
            "reference_mia_forget_mean": float(np.mean(forget_scores)) if len(forget_scores) else 0.0,
            "reference_mia_retain_mean": float(np.mean(retain_scores)) if len(retain_scores) else 0.0,
            "reference_mia_auc": float(LiRAMetric._simple_auc(labels, scores)),
        }

    @staticmethod
    def _collect_scores(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        reference_model: Optional[nn.Module],
    ) -> np.ndarray:
        target_losses = _collect_per_sample_losses(model, loader, device)
        if reference_model is None:
            return -target_losses

        ref_device = next(reference_model.parameters()).device
        ref_losses = _collect_per_sample_losses(reference_model, loader, ref_device)
        min_len = min(len(target_losses), len(ref_losses))
        if min_len == 0:
            return np.array([])
        return ref_losses[:min_len] - target_losses[:min_len]


class GradNormMIAMetric(BaseMetric):
    """
    Gradient-norm membership inference attack.

    Uses the L2 norm of the loss gradient with respect to model
    parameters as the membership signal.
    """

    name = "gradnorm_mia"

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device

        forget_scores = self._collect_scores(model, forget_data, device)
        retain_scores = self._collect_scores(model, retain_data, device)
        labels = np.concatenate([np.ones(len(forget_scores)), np.zeros(len(retain_scores))])
        scores = np.concatenate([forget_scores, retain_scores])

        return {
            "gradnorm_mia_forget_mean": float(np.mean(forget_scores)) if len(forget_scores) else 0.0,
            "gradnorm_mia_retain_mean": float(np.mean(retain_scores)) if len(retain_scores) else 0.0,
            "gradnorm_mia_auc": float(LiRAMetric._simple_auc(labels, scores)),
        }

    @staticmethod
    def _collect_scores(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> np.ndarray:
        criterion = nn.CrossEntropyLoss()
        scores: List[float] = []
        model.eval()

        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            inputs, targets = batch[0].to(device), batch[1].to(device)
            for i in range(inputs.size(0)):
                model.zero_grad(set_to_none=True)
                outputs = model(inputs[i : i + 1])
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, targets[i : i + 1])
                loss.backward()

                norm_sq = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        norm_sq += float(param.grad.norm(2).item() ** 2)
                scores.append(norm_sq ** 0.5)

        model.zero_grad(set_to_none=True)
        return np.array(scores, dtype=np.float64)


class MinKProbMetric(BaseMetric):
    """
    Min-K% probability membership inference metric.

    For each sample, compute the log-probabilities assigned to the
    target tokens/labels, sort them, and average the lowest ``k%``.
    Higher values indicate stronger memorisation because even the
    hardest tokens receive relatively high probability.
    """

    name = "mink"

    def __init__(self, k_percent: float = 20.0) -> None:
        if not 0 < k_percent <= 100:
            raise ValueError("k_percent must be in the interval (0, 100].")
        self.k_percent = k_percent

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        forget_scores = self._collect_mink_scores(model, forget_data, device)
        retain_scores = self._collect_mink_scores(model, retain_data, device)

        labels = np.concatenate([np.ones(len(forget_scores)), np.zeros(len(retain_scores))])
        scores = np.concatenate([forget_scores, retain_scores])

        return {
            "mink_forget_mean": float(np.mean(forget_scores)) if len(forget_scores) else 0.0,
            "mink_retain_mean": float(np.mean(retain_scores)) if len(retain_scores) else 0.0,
            "mink_auc": float(LiRAMetric._simple_auc(labels, scores)),
        }

    def _collect_mink_scores(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> np.ndarray:
        scores: List[float] = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue

                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                log_probs = F.log_softmax(logits, dim=-1)

                if logits.dim() == 2 and targets.dim() == 1:
                    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                    token_log_probs = token_log_probs.unsqueeze(1)
                    valid_mask = torch.ones_like(token_log_probs, dtype=torch.bool)
                elif logits.dim() >= 3 and targets.dim() == logits.dim() - 1:
                    safe_targets = targets.clamp_min(0)
                    token_log_probs = log_probs.gather(
                        -1, safe_targets.unsqueeze(-1)
                    ).squeeze(-1)
                    valid_mask = targets.ne(-100)
                else:
                    token_log_probs = log_probs.max(dim=-1).values
                    if token_log_probs.dim() == 1:
                        token_log_probs = token_log_probs.unsqueeze(1)
                    valid_mask = torch.ones_like(token_log_probs, dtype=torch.bool)

                for sample_log_probs, sample_mask in zip(token_log_probs, valid_mask):
                    valid_values = sample_log_probs[sample_mask]
                    if valid_values.numel() == 0:
                        continue
                    k = max(1, int(np.ceil(valid_values.numel() * (self.k_percent / 100.0))))
                    lowest_values, _ = torch.topk(valid_values, k=k, largest=False)
                    scores.append(float(lowest_values.mean().item()))

        return np.array(scores, dtype=np.float64)


class MinKPlusPlusMetric(BaseMetric):
    """
    Calibrated Min-K%++ membership inference metric.

    Normalises token log-probabilities by the per-position vocabulary
    mean and standard deviation before averaging the lowest ``k%``.
    """

    name = "mink_pp"

    def __init__(self, k_percent: float = 20.0) -> None:
        if not 0 < k_percent <= 100:
            raise ValueError("k_percent must be in the interval (0, 100].")
        self.k_percent = k_percent

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        forget_scores = self._collect_scores(model, forget_data, device)
        retain_scores = self._collect_scores(model, retain_data, device)

        labels = np.concatenate([np.ones(len(forget_scores)), np.zeros(len(retain_scores))])
        scores = np.concatenate([forget_scores, retain_scores])

        return {
            "mink_pp_forget_mean": float(np.mean(forget_scores)) if len(forget_scores) else 0.0,
            "mink_pp_retain_mean": float(np.mean(retain_scores)) if len(retain_scores) else 0.0,
            "mink_pp_auc": float(LiRAMetric._simple_auc(labels, scores)),
        }

    def _collect_scores(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> np.ndarray:
        scores: List[float] = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue

                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                log_probs = F.log_softmax(logits, dim=-1)

                if logits.dim() == 2:
                    mu = log_probs.mean(dim=-1, keepdim=True)
                    sigma = log_probs.std(dim=-1, keepdim=True).clamp_min(1e-8)
                    normalized = (log_probs - mu) / sigma
                    for sample_values in normalized:
                        k = max(1, int(np.ceil(sample_values.numel() * (self.k_percent / 100.0))))
                        lowest_values, _ = torch.topk(sample_values, k=k, largest=False)
                        scores.append(float(lowest_values.mean().item()))
                    continue

                mu = log_probs.mean(dim=-1, keepdim=True)
                sigma = log_probs.std(dim=-1, keepdim=True).clamp_min(1e-8)
                normalized = (log_probs - mu) / sigma

                if targets.dim() == logits.dim() - 1:
                    safe_targets = targets.clamp_min(0)
                    token_scores = normalized.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
                    valid_mask = targets.ne(-100)
                else:
                    token_scores = normalized.max(dim=-1).values
                    valid_mask = torch.ones_like(token_scores, dtype=torch.bool)

                for sample_scores, sample_mask in zip(token_scores, valid_mask):
                    valid_values = sample_scores[sample_mask]
                    if valid_values.numel() == 0:
                        continue
                    k = max(1, int(np.ceil(valid_values.numel() * (self.k_percent / 100.0))))
                    lowest_values, _ = torch.topk(valid_values, k=k, largest=False)
                    scores.append(float(lowest_values.mean().item()))

        return np.array(scores, dtype=np.float64)


def _collect_per_sample_losses(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Collect per-sample classification losses."""
    criterion = nn.CrossEntropyLoss(reduction="none")
    collected: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            losses = criterion(logits, targets).detach().cpu().numpy()
            collected.append(losses)

    return np.concatenate(collected) if collected else np.array([])
