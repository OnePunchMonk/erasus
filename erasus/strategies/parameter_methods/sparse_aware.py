"""
Sparse-Aware Unlearning.

Encourages sparsity in the weight updates during unlearning (`theta_unlearned - theta_orig`),
ensuring the unlearning process is "surgical" and affects as few parameters as possible.

Objective: L = L_forget + λ * ||θ - θ_orig||_1
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("sparse_aware")
class SparseAwareUnlearningStrategy(BaseStrategy):
    """
    Minimizes L1 distance between unlearned and original parameters
    while maximizing loss on forget set.
    """

    def __init__(
        self,
        sparsity_weight: float = 1e-3,
        lr: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sparsity_weight = sparsity_weight
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
        
        # Save original weights to compute sparsity of change
        original_params = {
            n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        forget_losses = []

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
                
                # Maximize forget loss
                if labels is not None:
                    loss_forget = -torch.nn.functional.cross_entropy(logits, labels)
                else:
                    loss_forget = -logits.sum()

                # Sparsity penalty: L1 norm of (theta - theta_orig)
                loss_sparse = 0.0
                for n, p in model.named_parameters():
                    if n in original_params and p.requires_grad:
                        loss_sparse += torch.norm(p - original_params[n], p=1)
                
                total_loss = loss_forget + self.sparsity_weight * loss_sparse
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += loss_forget.item()
                n += 1
            forget_losses.append(epoch_loss / max(n, 1))

        return model, forget_losses, []

