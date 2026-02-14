"""
erasus.metrics.utility.clip_score — CLIP-based similarity metric.

Measures how well a model maintains semantic alignment between
image and text modalities by computing CLIP cosine similarity.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric
from erasus.core.registry import metric_registry


@metric_registry.register("clip_score")
class CLIPScoreMetric(BaseMetric):
    """
    CLIP cosine similarity score.

    Computes the mean cosine similarity between image embeddings
    and text embeddings produced by a CLIP-style model.

    Parameters
    ----------
    temperature : float
        Temperature scaling for similarity (default: model's logit_scale).
    """

    def __init__(self, temperature: Optional[float] = None) -> None:
        self.temperature = temperature

    def compute(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute CLIP similarity scores.

        ``forget_loader`` should yield (images, texts) pairs.
        """
        device = next(model.parameters()).device
        model.eval()

        results: Dict[str, float] = {}

        # Forget set CLIP score
        forget_score = self._compute_clip_sim(model, forget_loader, device)
        results["forget_clip_score"] = forget_score

        # Retain set CLIP score
        if retain_loader is not None:
            retain_score = self._compute_clip_sim(model, retain_loader, device)
            results["retain_clip_score"] = retain_score
            results["clip_score_gap"] = retain_score - forget_score

        return results

    def _compute_clip_sim(
        self, model: nn.Module, loader: DataLoader, device: torch.device
    ) -> float:
        """Compute mean cosine similarity for a data loader."""
        similarities: list = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, texts = batch[0].to(device), batch[1].to(device)
                else:
                    # Can't compute CLIP score without paired data
                    continue

                # Try standard CLIP API
                if hasattr(model, "encode_image") and hasattr(model, "encode_text"):
                    img_emb = model.encode_image(images)
                    txt_emb = model.encode_text(texts)
                elif hasattr(model, "get_image_features") and hasattr(model, "get_text_features"):
                    img_emb = model.get_image_features(images)
                    txt_emb = model.get_text_features(texts)
                else:
                    # Fallback: use model output
                    outputs = model(images, texts) if callable(model) else model(images)
                    if hasattr(outputs, "image_embeds"):
                        img_emb = outputs.image_embeds
                        txt_emb = outputs.text_embeds
                    else:
                        continue

                # Normalise
                img_emb = F.normalize(img_emb, dim=-1)
                txt_emb = F.normalize(txt_emb, dim=-1)

                # Cosine similarity per pair
                sim = (img_emb * txt_emb).sum(dim=-1)
                similarities.extend(sim.cpu().numpy().tolist())

        if not similarities:
            return 0.0

        import numpy as np
        return float(np.mean(similarities))
