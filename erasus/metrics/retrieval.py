"""
Zero-Shot Retrieval Metric.

Evaluates the model's ability to retrieve the correct image/text from a set of candidates.
Crucial for VLM unlearning to ensure utility is preserved on retain sets or destroyed on forget sets.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class ZeroShotRetrievalMetric(BaseMetric):
    """
    Computes Recall@K (R@1, R@5) for Image-Text Retrieval.
    """

    def __init__(self, k_values: list[int] = [1, 5], **kwargs: Any):
        super().__init__(**kwargs)
        self.k_values = k_values

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        
        results = {}
        device = next(model.parameters()).device
        model.eval()

        if retain_data:
            # We focus on retain utility usually
            r_metrics = self._compute_retrieval(model, retain_data, device)
            for k, v in r_metrics.items():
                results[f"retain_{k}"] = v
                
        if forget_data:
             f_metrics = self._compute_retrieval(model, forget_data, device)
             for k, v in f_metrics.items():
                results[f"forget_{k}"] = v

        return results

    def _compute_retrieval(self, model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
        # Gather all embeddings
        image_embeds = []
        text_embeds = []
        
        with torch.no_grad():
            for batch in loader:
                # Expecting dictionary or tuple from VLM loader
                if isinstance(batch, dict):
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None: attention_mask = attention_mask.to(device)
                else:
                    # Tuple assumption: 0=pixels, 1=input_ids
                    pixel_values = batch[0].to(device)
                    input_ids = batch[1].to(device)
                    attention_mask = batch[2].to(device) if len(batch) > 2 else None

                # VLM Wrapper Interface
                if hasattr(model, "get_image_features") and hasattr(model, "get_text_features"):
                    img_feat = model.get_image_features(pixel_values)
                    txt_feat = model.get_text_features(input_ids, attention_mask=attention_mask)
                else:
                    # Fallback or error
                    return {}
                
                img_feat = F.normalize(img_feat, dim=-1)
                txt_feat = F.normalize(txt_feat, dim=-1)
                
                image_embeds.append(img_feat.cpu())
                text_embeds.append(txt_feat.cpu())
                
        image_embeds = torch.cat(image_embeds, dim=0)
        text_embeds = torch.cat(text_embeds, dim=0)
        
        # Similarity Matrix
        # [N_imgs, N_txts]
        sim_matrix = image_embeds @ text_embeds.T
        
        n_samples = sim_matrix.size(0)
        metrics = {}
        
        # Image-to-Text Retrieval
        # For each image, find correct text (diagonal)
        # Rank of the diagonal element in the row
        
        # If dataset has duplicated image-text pairs this might be slightly off given we assume 1-to-1 mapping
        # based on indices.
        
        start_idx = 0
        r1_count = 0
        r5_count = 0
        
        # Process in chunks if large? fitting in standard memory should be fine for Eval
        
        # Get targets
        targets = torch.arange(n_samples)
        
        # Image -> Text
        scores_i2t = sim_matrix
        _, indices_i2t = scores_i2t.topk(max(self.k_values), dim=1)
        
        for i in range(n_samples):
             if i in indices_i2t[i, :1]:
                 r1_count += 1
             if i in indices_i2t[i, :5]:
                 r5_count += 1
                 
        metrics["i2t_R@1"] = r1_count / n_samples
        metrics["i2t_R@5"] = r5_count / n_samples
        
        return metrics
