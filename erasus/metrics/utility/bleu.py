"""
erasus.metrics.utility.bleu — BLEU score metric for LLM unlearning.

Computes BLEU score to measure text generation quality preservation
after unlearning, particularly for language models.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("bleu")
class BLEUMetric(BaseMetric):
    """
    BLEU score for evaluating text generation quality.

    Parameters
    ----------
    max_n : int
        Maximum n-gram order (default 4 for BLEU-4).
    smooth : bool
        Apply smoothing for short texts.
    """

    def __init__(self, max_n: int = 4, smooth: bool = True) -> None:
        self.max_n = max_n
        self.smooth = smooth

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute BLEU scores on retain data (utility measure).

        Expects loader to yield (input_ids, target_ids) or (input_ids, labels).
        ``kwargs`` may include ``tokenizer`` for decoding.
        """
        results: Dict[str, float] = {}

        if retain_loader is not None:
            retain_bleu = self._evaluate_bleu(model, retain_loader, **kwargs)
            results["retain_bleu"] = retain_bleu

        forget_bleu = self._evaluate_bleu(model, forget_loader, **kwargs)
        results["forget_bleu"] = forget_bleu

        return results

    def _evaluate_bleu(
        self, model: nn.Module, loader: DataLoader, **kwargs: Any
    ) -> float:
        """Compute corpus-level BLEU for a loader."""
        device = next(model.parameters()).device
        model.eval()
        tokenizer = kwargs.get("tokenizer", None)

        all_references: List[List[str]] = []
        all_hypotheses: List[List[str]] = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_ids = batch[0].to(device)
                    target_ids = batch[1].to(device)
                else:
                    continue

                # Generate predictions
                if hasattr(model, "generate"):
                    pred_ids = model.generate(
                        input_ids, max_new_tokens=target_ids.size(-1), do_sample=False
                    )
                else:
                    outputs = model(input_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    pred_ids = logits.argmax(dim=-1)

                # Decode
                for i in range(len(input_ids)):
                    if tokenizer:
                        ref_text = tokenizer.decode(target_ids[i], skip_special_tokens=True)
                        hyp_text = tokenizer.decode(pred_ids[i], skip_special_tokens=True)
                    else:
                        ref_text = " ".join(map(str, target_ids[i].cpu().tolist()))
                        hyp_text = " ".join(map(str, pred_ids[i].cpu().tolist()))

                    ref_tokens = self._tokenize(ref_text)
                    hyp_tokens = self._tokenize(hyp_text)
                    all_references.append(ref_tokens)
                    all_hypotheses.append(hyp_tokens)

        if not all_hypotheses:
            return 0.0

        return self._corpus_bleu(all_references, all_hypotheses)

    def _corpus_bleu(
        self,
        references: List[List[str]],
        hypotheses: List[List[str]],
    ) -> float:
        """Compute corpus-level BLEU score."""
        import math

        precisions: List[float] = []
        for n in range(1, self.max_n + 1):
            num, den = 0.0, 0.0
            for ref, hyp in zip(references, hypotheses):
                ref_ngrams = self._get_ngrams(ref, n)
                hyp_ngrams = self._get_ngrams(hyp, n)
                clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
                num += clipped
                den += max(len(hyp) - n + 1, 0)

            if self.smooth:
                precisions.append((num + 1.0) / (den + 1.0))
            else:
                precisions.append(num / max(den, 1e-8))

        # Brevity penalty
        ref_len = sum(len(r) for r in references)
        hyp_len = sum(len(h) for h in hypotheses)
        if hyp_len == 0:
            return 0.0
        bp = math.exp(min(0, 1.0 - ref_len / hyp_len))

        # Geometric mean of precisions
        log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / len(precisions)
        return bp * math.exp(log_avg)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenisation."""
        return text.lower().split()

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-gram counts."""
        return Counter(tuple(tokens[i:i + n]) for i in range(max(len(tokens) - n + 1, 0)))
