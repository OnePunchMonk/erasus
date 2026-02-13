"""
Contrastive Unlearning Strategy (for CLIP/VLMs).

Leverages the contrastive nature of CLIP. 
To unlearn (Image, Text) pair:
1. Maximize distance between Image embedding and Text embedding.
2. Maintain distance for Retain pairs.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("contrastive_unlearning")
class ContrastiveUnlearningStrategy(BaseStrategy):
    """
    Standard CLIP loss maximizes cosine similarity for matched pairs 
    and minimizes it for unmatched pairs.
    
    Unlearning:
    - For Forget Pairs: MINIMIZE cosine similarity (push them apart).
    - For Retain Pairs: MAXIMIZE cosine similarity (standard training).
    """

    def __init__(
        self,
        lr: float = 1e-5,
        neg_weight: float = 1.0, # Weight for unlearning term
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.neg_weight = neg_weight

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        device = next(model.parameters()).device
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        
        forget_losses = []
        retain_losses = []
        
        model.train()
        for epoch in range(epochs):
            # 1. Forget Step
            f_loss_accum = 0.0
            n_f = 0
            for batch in forget_loader:
                # batch: [images, texts]
                images = batch[0].to(device)
                texts = batch[1].to(device)
                
                optimizer.zero_grad()
                outputs = model(images, texts) 
                
                # outputs.logits_per_image: [B, B]
                # Diagonal is the pairs we want to break.
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                labels = torch.arange(len(images), device=device)
                
                # Standard Contrastive Loss
                loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
                loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
                loss_standard = (loss_i + loss_t) / 2
                
                # We want to ASCEND gradient (Maximize Loss)
                loss_unlearn = -loss_standard
                
                (loss_unlearn * self.neg_weight).backward()
                optimizer.step()
                
                f_loss_accum += -loss_unlearn.item()
                n_f += 1
            forget_losses.append(f_loss_accum / max(n_f, 1))

            # 2. Retain Step (Healing)
            r_loss_accum = 0.0
            n_r = 0
            if retain_loader:
                for batch in retain_loader:
                    images = batch[0].to(device)
                    texts = batch[1].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images, texts) 
                    
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_text
                    
                    labels = torch.arange(len(images), device=device)
                    
                    # Standard Training
                    loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
                    loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
                    loss = (loss_i + loss_t) / 2
                    
                    loss.backward()
                    optimizer.step()
                    
                    r_loss_accum += loss.item()
                    n_r += 1
                retain_losses.append(r_loss_accum / max(n_r, 1))

        return model, forget_losses, retain_losses

