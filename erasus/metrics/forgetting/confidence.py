"""
erasus.metrics.forgetting.confidence — Confidence-based forgetting measures.

Measures how confident the model remains on forget-set samples
after unlearning.  Effective forgetting should increase prediction
entropy and decrease max-softmax confidence on forgotten data.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class ConfidenceMetric(BaseMetric):
    """
    Measures prediction confidence on forget vs retain sets.

    Reports:
    - Mean max-softmax confidence (lower on forget = better)
    - Mean prediction entropy (higher on forget = better)
    - Confidence gap between retain and forget
    """

    name = "confidence"

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        forget_confs, forget_entropies = self._analyse(model, forget_data, device)
        retain_confs, retain_entropies = self._analyse(model, retain_data, device)

        return {
            "confidence_forget_mean": float(np.mean(forget_confs)) if len(forget_confs) else 0.0,
            "confidence_retain_mean": float(np.mean(retain_confs)) if len(retain_confs) else 0.0,
            "confidence_gap": float(np.mean(retain_confs) - np.mean(forget_confs))
            if len(forget_confs) and len(retain_confs) else 0.0,
            "entropy_forget_mean": float(np.mean(forget_entropies)) if len(forget_entropies) else 0.0,
            "entropy_retain_mean": float(np.mean(retain_entropies)) if len(retain_entropies) else 0.0,
        }

    @staticmethod
    def _analyse(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> tuple:
        confs: List[float] = []
        entropies: List[float] = []

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

                # Max confidence per sample
                max_conf = probs.max(dim=-1).values
                confs.extend(max_conf.cpu().tolist())

                # Entropy per sample: -sum(p * log(p))
                log_probs = torch.log(probs + 1e-10)
                ent = -(probs * log_probs).sum(dim=-1)
                entropies.extend(ent.cpu().tolist())

        return np.array(confs), np.array(entropies)
