"""
Amnesiac Unlearning Strategy.

Paper: "Amnesiac Machine Unlearning" (Graves et al., AAAI 2021)

Tracks parameter updates during the original training (if available) or 
approximates them. 
This implementation assumes either:
1. Access to a training history artifact (ideal but rare).
2. Or performs a "reverse update" approximation on the current model using the forget set 
   (conceptually similar to simple Gradient Ascent but framed as subtracting specific batch updates).

For a general framework without history, we implement 'Imparied Learning' style:
unlearn = train_step(forget_batch, negative_lr)
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("amnesiac")
class AmnesiacUnlearningStrategy(BaseStrategy):
    """
    Simulates 'Amnesiac' unlearning by subtracting gradients derived 
    specifically from the forget data, often with a few epochs of fine-tuning
    on the retain data to heal.
    """

    def __init__(
        self,
        unlearn_lr: float = 1e-4,
        heal_lr: float = 1e-5,
        healing_epochs: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.unlearn_lr = unlearn_lr
        self.heal_lr = heal_lr
        self.healing_epochs = healing_epochs

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        device = next(model.parameters()).device
        optimizer = torch.optim.SGD(model.parameters(), lr=self.unlearn_lr)
        
        forget_losses = []
        
        # Phase 1: Amnesiac Unlearning (Subtracting Forget Gradients)
        model.train()
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
                    loss = logits.sum() # simplified
                
                # In amnesiac, we want to negate the update that WOULD have happened.
                # Update: w = w - lr * grad.
                # To reverse: w = w + lr * grad.
                # In PyTorch optimizer.step() does w -= lr * grad.
                # So if we want w += lr * grad, we feed it NEGATIVE gradient?
                # i.e. backward(-loss).
                
                (-loss).backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n += 1
            forget_losses.append(epoch_loss / max(n, 1))
            
        # Phase 2: Healing (Fine-tuning on Retain Set)
        retain_losses = []
        if retain_loader and self.healing_epochs > 0:
            heal_opt = torch.optim.SGD(model.parameters(), lr=self.heal_lr)
            for h_epoch in range(self.healing_epochs):
                h_loss = 0.0
                n = 0
                for batch in retain_loader:
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device) if len(batch) > 1 else None
                    if labels is None: continue
                    
                    heal_opt.zero_grad()
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    loss.backward()
                    heal_opt.step()
                    
                    h_loss += loss.item()
                    n += 1
                retain_losses.append(h_loss / max(n, 1))

        return model, forget_losses, retain_losses

