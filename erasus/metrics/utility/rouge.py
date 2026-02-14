"""
erasus.metrics.utility.rouge — ROUGE metric for LLM unlearning.

Compute ROUGE-1, ROUGE-2, and ROUGE-L scores to evaluate
text generation quality preservation after unlearning.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("rouge")
class ROUGEMetric(BaseMetric):
    """
    ROUGE score metric for text quality evaluation.

    Computes ROUGE-1 (unigram), ROUGE-2 (bigram), and ROUGE-L
    (longest common subsequence) F1 scores.
    """

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute ROUGE scores."""
        results: Dict[str, float] = {}

        if retain_loader is not None:
            retain_scores = self._evaluate_rouge(model, retain_loader, **kwargs)
            for k, v in retain_scores.items():
                results[f"retain_{k}"] = v

        forget_scores = self._evaluate_rouge(model, forget_loader, **kwargs)
        for k, v in forget_scores.items():
            results[f"forget_{k}"] = v

        return results

    def _evaluate_rouge(
        self, model: nn.Module, loader: DataLoader, **kwargs: Any
    ) -> Dict[str, float]:
        """Compute ROUGE for a loader."""
        device = next(model.parameters()).device
        model.eval()
        tokenizer = kwargs.get("tokenizer", None)

        all_r1, all_r2, all_rl = [], [], []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_ids = batch[0].to(device)
                    target_ids = batch[1].to(device)
                else:
                    continue

                if hasattr(model, "generate"):
                    pred_ids = model.generate(
                        input_ids, max_new_tokens=target_ids.size(-1), do_sample=False
                    )
                else:
                    outputs = model(input_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    pred_ids = logits.argmax(dim=-1)

                for i in range(len(input_ids)):
                    if tokenizer:
                        ref = tokenizer.decode(target_ids[i], skip_special_tokens=True)
                        hyp = tokenizer.decode(pred_ids[i], skip_special_tokens=True)
                    else:
                        ref = " ".join(map(str, target_ids[i].cpu().tolist()))
                        hyp = " ".join(map(str, pred_ids[i].cpu().tolist()))

                    ref_tokens = ref.lower().split()
                    hyp_tokens = hyp.lower().split()

                    all_r1.append(self._rouge_n(ref_tokens, hyp_tokens, n=1))
                    all_r2.append(self._rouge_n(ref_tokens, hyp_tokens, n=2))
                    all_rl.append(self._rouge_l(ref_tokens, hyp_tokens))

        import numpy as np
        return {
            "rouge_1": float(np.mean(all_r1)) if all_r1 else 0.0,
            "rouge_2": float(np.mean(all_r2)) if all_r2 else 0.0,
            "rouge_l": float(np.mean(all_rl)) if all_rl else 0.0,
        }

    @staticmethod
    def _rouge_n(reference: List[str], hypothesis: List[str], n: int = 1) -> float:
        """ROUGE-N F1 score."""
        def ngrams(tokens, n):
            return Counter(tuple(tokens[i:i + n]) for i in range(max(len(tokens) - n + 1, 0)))

        ref_ng = ngrams(reference, n)
        hyp_ng = ngrams(hypothesis, n)

        overlap = sum(min(ref_ng[ng], hyp_ng[ng]) for ng in hyp_ng)
        precision = overlap / max(sum(hyp_ng.values()), 1)
        recall = overlap / max(sum(ref_ng.values()), 1)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _rouge_l(reference: List[str], hypothesis: List[str]) -> float:
        """ROUGE-L F1 score using LCS."""
        m, n = len(reference), len(hypothesis)
        if m == 0 or n == 0:
            return 0.0

        # LCS dynamic programming
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        precision = lcs_len / n
        recall = lcs_len / m

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
