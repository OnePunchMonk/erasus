"""
Negative Gradient Strategy.

A naive baseline that performs Gradient Ascent on the forget set to increase loss.
Unlike `GradientAscentStrategy`, this implementation focuses on the raw
negative gradient update step without retain set regularization, often used
as a building block or extreme baseline.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("negative_gradient")
class NegativeGradientStrategy(BaseStrategy):
    """
    Simple Gradient Ascent.
    θ = θ + η * ∇L(forget)
    """

    def __init__(self, lr: float = 1e-4, **kwargs: Any) -> None:
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
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        device = next(model.parameters()).device
        model.train()
        
        forget_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in forget_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                else:
                    # Fallback for unlabelled: maximize confidence
                    loss = logits.max(dim=1)[0].sum()
                
                # Negative Gradient => Minimize (-Loss) => Maximize Loss
                neg_loss = -loss
                neg_loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n += 1
            
            forget_losses.append(epoch_loss / max(n, 1))
            
        return model, forget_losses, []

