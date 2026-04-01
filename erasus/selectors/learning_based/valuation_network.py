"""
Valuation Network Selector.

Uses a trained auxiliary model (Valuation Network) to predict the value of a datum.
Requires a pre-trained Valuation Network that accepts (input, label) pairs and outputs a scalar value.

Reference: "Data Valuation using Reinforcement Learning" (Yoon et al., ICML 2020)
"""
from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.exceptions import SelectorError
from erasus.core.registry import selector_registry

logger = logging.getLogger(__name__)

@selector_registry.register("valuation_network")
class ValuationNetworkSelector(BaseSelector):
    """
    Selects samples with the highest predicted value from an auxiliary Valuation Network.
    """
    
    def select(self, model: nn.Module, data_loader: DataLoader, k: int, val_net: nn.Module = None, **kwargs) -> List[int]:
        if val_net is None:
            raise SelectorError(
                "ValuationNetworkSelector requires a 'val_net' argument — a trained "
                "auxiliary model that scores (input, label) pairs. Pass it as: "
                "selector.select(model, loader, k, val_net=my_val_net)"
            )
        
        device = next(val_net.parameters()).device
        val_net.eval()
        scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Handle batch unpacking
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device) if len(batch) > 1 else None
                else:
                    raise SelectorError(
                        "ValuationNetworkSelector requires batches as (inputs, labels) tuples. "
                        "Your DataLoader yielded a non-tuple batch."
                    )
                
                if labels is None:
                    raise SelectorError(
                        "ValuationNetworkSelector requires labels (batch[1]) to score samples. "
                        "Your DataLoader yielded unlabeled batches."
                    )

                # We assume val_net(inputs, labels) returns [B] or [B, 1]
                batch_scores = val_net(inputs, labels)
                
                # Flatten scores
                if batch_scores.dim() > 1:
                    batch_scores = batch_scores.flatten()
                
                scores.extend(batch_scores.cpu().tolist())
                
        if not scores:
            raise SelectorError(
                "ValuationNetworkSelector did not produce any scores. "
                "Ensure your DataLoader yields at least one labeled batch."
            )
            
        n = len(scores)
        k = min(k, n)
        
        # Select top-k highest value
        top_k_indices = np.argsort(scores)[-k:].tolist()
        return [int(i) for i in top_k_indices]
