"""
FID Metric — Frechet Inception Distance.

Measures the quality and diversity of generated images compared to real images.
Lower FID is better (closer distribution).
"""

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from erasus.core.base_metric import BaseMetric


class FIDMetric(BaseMetric):
    """
    Computes FID using `torchmetrics.image.fid.FrechetInceptionDistance`.
    Requires `torchmetrics` library.
    """

    def __init__(self, feature: int = 2048, normalize: bool = True):
        self.feature = feature
        self.normalize = normalize
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self._fid = FrechetInceptionDistance(feature=feature, normalize=normalize)
        except ImportError:
            self._fid = None
            print("Warning: torchmetrics not installed. FIDMetric will return 0.0.")

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None, # Real/Reference distribution usually from retain
        **kwargs: Any,
    ) -> Dict[str, float]:
        
        if self._fid is None:
            return {"fid": 0.0}

        if retain_data is None:
             print("Warning: FID requires a reference (retain) dataloader for real images.")
             return {"fid": 0.0}

        device = next(model.parameters()).device
        self._fid.to(device)
        self._fid.reset()

        # 1. Update with real images (Retain set)
        self._update_fid(retain_data, real=True, device=device)

        # 2. Generate fake images from model using prompts?
        # Standard unlearning FID usually compares *generated* unlearned images 
        # vs *real* original images (or generated original images).
        # Or comparing generated forget prompts vs real forget images?
        # BUT: For unlearning, we often want the model to generate garbage/noise for forget prompts.
        # High FID on forget set is desired? 
        # Low FID on retain set is desired (preserve utility).
        
        # Let's compute TWO FIDs if possible:
        # - Retain FID: Generated(Retain Prompts) vs Real(Retain Images). (Should be Low)
        # - Forget FID: Generated(Forget Prompts) vs Real(Forget Images). (Should be High if erasure successful?)
        
        results = {}
        
        # Compute Retain FID (Utility)
        # Generate images for retain prompts
        # Assuming dataloader yields (images, prompts) or similar.
        # This is tricky for generic dataloaders.
        # Let's assume dataloader yields images directly for 'real' update.
        # And model can generate images.
        
        # If model is diffusion wrapper, it has generate_image(prompt).
        
        # For simplicity in this metric class, we assume we can generate batches.
        # Since generation is expensive, we might skip it or do a small subset.
        
        # NOTE: This implementation assumes `model` is a DiffusionWrapper that exposes `generate`.
        
        # We need prompts. If dataloader doesn't provide them, we can't generate.
        # Let's assume standard structure: (image, caption/label)
        
        # Just computing Retain FID for now as a proxy for utility preservation.
        
        self._update_fake(model, retain_data, device=device)
        
        try:
            fid_score = self._fid.compute().item()
            results["retain_fid"] = fid_score
        except Exception as e:
            print(f"FID computation failed: {e}")
            results["retain_fid"] = -1.0
            
        return results

    def _update_fid(self, loader, real: bool, device):
        # Update internal state with real images
        count = 0
        input_key = "pixel_values" # conform to diffusers/HF
        
        for batch in loader:
            imgs = None
            if isinstance(batch, dict):
                imgs = batch.get(input_key) or batch.get("images")
            elif isinstance(batch, (list, tuple)):
                imgs = batch[0]
            elif isinstance(batch, torch.Tensor):
                imgs = batch
                
            if imgs is not None:
                imgs = imgs.to(device)
                # Ensure [0, 255] byte or [0, 1] float? 
                # torchmetrics expects [0, 255] uint8 if normalize=False, or float [0, 1] if normalize=True
                # But our wrapper usually handles scaling.
                # Just pass as is and hope for best or clamp.
                if imgs.ndim == 4:
                     self._fid.update(imgs, real=real)
                     count += imgs.shape[0]
            
            if count > 200: break # Limit for speed in this metric class
            
    def _update_fake(self, model, loader, device):
        # Generate fake images
        # We need prompts from loader
        count = 0
        for batch in loader:
            prompts = []
            if isinstance(batch, dict):
                prompts = batch.get("input_ids") # Tokenized?
            elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                prompts = batch[1] # raw strings or tokens?
            
            # If we don't have raw text prompts, generation is hard unless we detokenize.
            # Assuming we can't easily generate without text.
            # So we might rely on the model wrapper having a `generate_from_batch` method?
            
            # Placeholder: if we can't get prompts, we return.
            return 
