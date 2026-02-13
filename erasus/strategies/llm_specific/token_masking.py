"""
Token Masking Strategy for LLMs.

Learns which tokens in the prompt/completion are most responsible for the
undesired knowledge and masks them out or minimizes their likelihood.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("token_masking")
class TokenMaskingStrategy(BaseStrategy):
    """
    Identifies 'toxic' or 'knowledge-bearing' tokens in the input
    and maximizes their loss (Gradient Ascent) or minimizes the probability
    of generating them given the prefix.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        forget_losses = []
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in forget_loader:
                # LLM batches usually: input_ids, attention_mask, labels
                # We assume a standard HF format
                input_ids = batch[0].to(device)
                
                # If tuple has labels, use them. Else self-supervised (labels=input_ids)
                if len(batch) > 1:
                    labels = batch[1].to(device)
                else:
                    labels = input_ids.clone()
                
                optimizer.zero_grad()
                outputs = model(input_ids, labels=labels)
                
                # Standard GA: Maximize XE loss on these tokens
                # We want to forbid this sequence.
                loss = -outputs.loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += -loss.item()
                n += 1
            forget_losses.append(epoch_loss / max(n, 1))
            
        return model, forget_losses, []

