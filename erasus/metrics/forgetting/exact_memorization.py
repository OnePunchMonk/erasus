"""
erasus.metrics.forgetting.exact_memorization — Exact match memorization.

Implements exact string/tensor-match detection for extractable forget-set
content. The metric supports both generative outputs and classifier-style
predictions, falling back to exact label matches when generation is not
available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class ExactMemorizationMetric(BaseMetric):
    """
    Exact memorization via exact string/tensor match detection.

    Parameters
    ----------
    confidence_threshold : float
        Confidence threshold used for classifier-style fallback.
    prefix_length : int
        Number of tokens/elements to keep as the prompt prefix when a
        sequence target is available.
    max_new_tokens : int
        Maximum number of tokens to generate beyond the prompt prefix.
    """

    name = "exact_memorization"

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        prefix_length: int = 4,
        max_new_tokens: int = 32,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.prefix_length = prefix_length
        self.max_new_tokens = max_new_tokens

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

        forget_rate, forget_scores = self._compute_exact_rate(model, forget_data, device)
        results: Dict[str, float] = {
            "exact_memorization_forget": float(forget_rate),
            "exact_memorization_forget_mean_conf": float(np.mean(forget_scores))
            if forget_scores
            else 0.0,
        }

        if retain_data is not None:
            retain_rate, retain_scores = self._compute_exact_rate(model, retain_data, device)
            results["exact_memorization_retain"] = float(retain_rate)
            results["exact_memorization_retain_mean_conf"] = float(np.mean(retain_scores))
            if not retain_scores:
                results["exact_memorization_retain_mean_conf"] = 0.0
            results["exact_memorization_gap"] = float(retain_rate - forget_rate)

        return results

    def _compute_exact_rate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[float, List[float]]:
        exact_matches = 0
        total = 0
        confidences: List[float] = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue

                inputs, targets = batch[0].to(device), batch[1].to(device)

                if self._can_generate(model, targets):
                    match_count, batch_scores = self._exact_match_generation(
                        model=model,
                        inputs=inputs,
                        targets=targets,
                        device=device,
                    )
                    exact_matches += match_count
                    total += len(batch_scores)
                    confidences.extend(batch_scores)
                    continue

                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                for i in range(targets.size(0)):
                    if targets.dim() == 1:
                        conf = probs[i, targets[i]].item()
                        matched = preds[i].item() == targets[i].item()
                    else:
                        matched = torch.equal(preds[i], targets[i])
                        conf = self._sequence_confidence(probs[i], targets[i])

                    confidences.append(conf)
                    if matched and conf >= self.confidence_threshold:
                        exact_matches += 1
                    total += 1

        return exact_matches / max(total, 1), confidences

    def _exact_match_generation(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        device: torch.device,
    ) -> Tuple[int, List[float]]:
        """Measure exact sequence reconstruction from prefix prompting."""
        generated = model.generate(inputs, max_new_tokens=self.max_new_tokens)
        matches = 0
        scores: List[float] = []

        for pred_seq, target_seq in zip(generated, targets):
            pred_valid = pred_seq[pred_seq.ne(-100)] if pred_seq.ndim > 0 else pred_seq
            target_valid = target_seq[target_seq.ne(-100)] if target_seq.ndim > 0 else target_seq
            matched = torch.equal(pred_valid.to(device), target_valid.to(device))
            score = 1.0 if matched else 0.0
            scores.append(score)
            if matched:
                matches += 1

        return matches, scores

    @staticmethod
    def _can_generate(model: nn.Module, targets: torch.Tensor) -> bool:
        return hasattr(model, "generate") and targets.dim() > 1

    @staticmethod
    def _sequence_confidence(probs: torch.Tensor, targets: torch.Tensor) -> float:
        safe_targets = targets.clamp_min(0)
        gathered = probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        valid = targets.ne(-100) if targets.dim() > 0 else torch.tensor(True, device=targets.device)
        valid_scores = gathered[valid]
        if valid_scores.numel() == 0:
            return 0.0
        return float(valid_scores.mean().item())
