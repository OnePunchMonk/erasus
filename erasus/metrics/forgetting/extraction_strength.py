"""
erasus.metrics.forgetting.extraction_strength — Prefix-prompt extraction.

Measures how much forget-set content can be extracted by prompting the
model with prefixes from the forget data and evaluating whether the
continuation is recoverable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class ExtractionStrengthMetric(BaseMetric):
    """
    Prefix-prompt extraction strength.

    For sequence targets, prompt with a prefix and measure exact
    continuation recovery. For classifier-style tasks, fall back to the
    top-k extraction definition.
    """

    name = "extraction_strength"

    def __init__(self, top_k: int = 1, prefix_length: int = 4, max_new_tokens: int = 32):
        self.top_k = top_k
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
            return {"extraction_strength": 0.0}

        device = next(model.parameters()).device
        model.eval()

        forget_es = self._compute_extraction(model, forget_data, device)
        results: Dict[str, float] = {"extraction_strength_forget": float(forget_es)}

        if retain_data is not None:
            retain_es = self._compute_extraction(model, retain_data, device)
            results["extraction_strength_retain"] = float(retain_es)
            results["extraction_strength_gap"] = float(forget_es - retain_es)
            random_baseline = self.top_k / max(self._infer_num_classes(model, forget_data, device), 1)
            results["extraction_resistance"] = min(
                1.0,
                max(
                    0.0,
                    1.0 - (forget_es - random_baseline) / max(1.0 - random_baseline, 1e-8),
                ),
            )

        return results

    def _compute_extraction(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        extracted = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue

                inputs, targets = batch[0].to(device), batch[1].to(device)

                if hasattr(model, "generate") and targets.dim() > 1:
                    generated = model.generate(inputs, max_new_tokens=self.max_new_tokens)
                    for pred_seq, target_seq in zip(generated, targets):
                        target_valid = target_seq[target_seq.ne(-100)]
                        prefix_len = min(self.prefix_length, target_valid.numel())
                        suffix = target_valid[prefix_len:]
                        pred_valid = pred_seq[pred_seq.ne(-100)]
                        pred_suffix = pred_valid[prefix_len : prefix_len + suffix.numel()]
                        if suffix.numel() == 0 or torch.equal(pred_suffix, suffix):
                            extracted += 1
                        total += 1
                    continue

                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                _, top_k_preds = logits.topk(min(self.top_k, logits.size(-1)), dim=-1)

                for i in range(targets.size(0)):
                    if targets[i] in top_k_preds[i]:
                        extracted += 1
                    total += 1

        return extracted / max(total, 1)

    @staticmethod
    def _infer_num_classes(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> int:
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)):
                    continue
                inputs = batch[0].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                return logits.size(-1)
        return 1
