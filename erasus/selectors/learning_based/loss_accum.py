"""
Loss Accumulation Selector (Learning-Based).

Selects points based on their loss magnitude.
"Hard" examples (High Loss) might be more important to target or retain depending on strategy.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry


@selector_registry.register("loss_accum")
class LossAccumulationSelector(BaseSelector):
    """
    Selects top-k samples with highest loss.
    """

    def __init__(self, selection_ratio: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.selection_ratio = selection_ratio

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        
        device = next(model.parameters()).device
        model.eval()
        
        losses = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None
                
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                if labels is not None:
                    loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
                    losses.extend(loss.cpu().tolist())
                else:
                    # Unsupervised: e.g. Perplexity / Reconstruction error
                    # Simple sum for now
                    loss = logits.sum(dim=-1) # heuristic
                    losses.extend(loss.cpu().tolist())
        
        n = len(losses)
        k = min(k, n)
        
        # Indices of top-k loss
        top_k_indices = np.argsort(losses)[-k:].tolist()
        
        return [int(i) for i in top_k_indices]
