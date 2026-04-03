"""
erasus.metrics.forgetting.memorization — Memorization detection metrics.

Implements:
- Extraction Strength (ES): extractable forget data via prefix probing
- Exact Memorization (EM): exact string/tensor match detection
- Verbatim Memorization: longest-common-substring and n-gram overlap detection
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.metrics.forgetting.exact_memorization import ExactMemorizationMetric
from erasus.metrics.forgetting.extraction_strength import ExtractionStrengthMetric


class VerbatimMemorizationMetric(BaseMetric):
    """
    Verbatim memorization via overlap between generated/predicted outputs
    and forget targets.
    """

    name = "verbatim_memorization"

    def __init__(self, entropy_threshold: Optional[float] = None):
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
        overlap_scores: list[float] = []
        substring_scores: list[float] = []
        memorised = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = logits.argmax(dim=-1)

                for i in range(targets.size(0)):
                    target_tokens = self._token_list(targets[i])
                    pred_tokens = self._token_list(preds[i])

                    ngram_overlap = self._ngram_overlap(target_tokens, pred_tokens, n=3)
                    longest_substring = self._longest_common_substring_ratio(
                        target_tokens, pred_tokens
                    )
                    overlap_scores.append(1.0 - ngram_overlap)
                    substring_scores.append(1.0 - longest_substring)

                    memorization_signal = max(ngram_overlap, longest_substring)
                    thresh = self.entropy_threshold if self.entropy_threshold is not None else 0.5
                    if memorization_signal >= thresh:
                        memorised += 1
                    total += 1

        mem_rate = memorised / max(total, 1)
        return np.array(overlap_scores), np.array(substring_scores), mem_rate

    @staticmethod
    def _token_list(value: torch.Tensor) -> list[str]:
        if value.dim() == 0:
            return [str(int(value.item()))]
        valid = value[value.ne(-100)] if value.dtype in (torch.int32, torch.int64) else value
        return [str(int(v)) for v in valid.detach().cpu().flatten().tolist()]

    @staticmethod
    def _ngram_overlap(reference: list[str], hypothesis: list[str], n: int = 3) -> float:
        if len(reference) < n or len(hypothesis) < n:
            return float(reference == hypothesis) if reference or hypothesis else 0.0
        ref = {tuple(reference[i : i + n]) for i in range(len(reference) - n + 1)}
        hyp = {tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1)}
        if not ref:
            return 0.0
        return len(ref & hyp) / len(ref)

    @staticmethod
    def _longest_common_substring_ratio(reference: list[str], hypothesis: list[str]) -> float:
        if not reference or not hypothesis:
            return 0.0
        dp = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
        best = 0
        for i in range(1, len(reference) + 1):
            for j in range(1, len(hypothesis) + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    best = max(best, dp[i][j])
        return best / max(len(reference), 1)


__all__ = [
    "ExtractionStrengthMetric",
    "ExactMemorizationMetric",
    "VerbatimMemorizationMetric",
]
