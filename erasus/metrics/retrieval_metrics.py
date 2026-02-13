"""
Retrieval Metrics — For Vision-Language Models (CLIP, etc.).
Checks Zero-Shot Classification or Image-Text Retrieval performance.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from erasus.core.base_metric import BaseMetric


class ZeroShotAccuracyMetric(BaseMetric):
    """
    Computes zero-shot classification accuracy for VLMs.
    """

    def __init__(self, classes: list[str], templates: list[str] = None):
         self.classes = classes
         self.templates = templates or ["a photo of a {}."]

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        
        results = {}
        # We need images and their true class labels (indices).
        # If forget_data yields (image, label_idx), we can compute.
        
        # Pre-compute text embeddings for classes
        device = next(model.parameters()).device
        text_embeds = self._compute_class_embeddings(model, device) # [Classes, D]
        
        if forget_data:
            results["forget_zero_shot"] = self._evaluate(model, forget_data, text_embeds, device)
        if retain_data:
            results["retain_zero_shot"] = self._evaluate(model, retain_data, text_embeds, device)
            
        return results

    def _compute_class_embeddings(self, model, device):
        # Naive implementation assuming a standard CLIP-like interface
        # Requires model.encode_text or similar.
        # This relies on the BaseVLMModel wrapper.
        prompts = [t.format(c) for c in self.classes for t in self.templates]
        # Tokenize... skipping complex tokenization here for brevity.
        # Assuming wrapper has `get_text_features(list_of_strings)`
        if hasattr(model, "get_text_features"):
             # This might be slow if many classes.
             feats = model.get_text_features(prompts)
             # Reshape and average templates per class
             # [Classes * Templates, D] -> [Classes, Templates, D] -> mean -> [Classes, D]
             feats = feats.view(len(self.classes), len(self.templates), -1).mean(dim=1)
             feats = feats / feats.norm(dim=-1, keepdim=True)
             return feats
        return torch.randn(len(self.classes), 512).to(device) # dummy

    def _evaluate(self, model, loader, text_embeds, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                imgs, labels = batch[0], batch[1]
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                # Image features
                if hasattr(model, "get_image_features"):
                    img_feats = model.get_image_features(imgs)
                else:
                    continue
                    
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                
                # Cosine sim
                logits = img_feats @ text_embeds.t()
                preds = logits.argmax(dim=-1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return correct / max(total, 1)
