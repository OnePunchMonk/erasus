"""
Fisher Forgetting Strategy.

Uses the Fisher Information Matrix (FIM) to identify and protect parameters
that are important for the retain set, whilst allowing changes to
parameters relevant to the forget set or unimportant for retention.

Paper: Selective Forgetting in Deep Networks (Golatkar et al., CVPR 2020)
Formula: L = L_forget + λ * Σ F_ii * (θ_i - θ_orig_i)^2
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("fisher_forgetting")
class FisherForgettingStrategy(BaseStrategy):
    """
    Penalizes deviations from original weights based on their Fisher Information.
    High Fisher Info = Parameter important for Retain Set -> Don't change it.
    Low Fisher Info = Parameter unimportant -> Can be changed to minimize Forget Loss.
    """

    def __init__(
        self,
        fisher_lambda: float = 1000.0,
        lr: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.fisher_lambda = fisher_lambda
        self.lr = lr

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        if retain_loader is None:
             raise ValueError("Fisher Forgetting requires a retain_loader to compute FIM.")

        device = next(model.parameters()).device
        
        # 1. Compute diagonal Fisher Information Matrix (FIM) on Retain Set
        fisher_diag = self._compute_fisher_diag(model, retain_loader, device)
        
        # 2. Store original parameters
        original_params = {
            n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        forget_losses = []
        retain_losses = [] # Track penalty term

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            
            for batch in forget_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None # Handle unlabelled

                optimizer.zero_grad()
                
                # Forward
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                # Maximize entropy or Minimise Negative Log Likelihood (Gradient Ascent)
                # Standard approach: Maximize CrossEntropy (Gradient Ascent)
                # L_forget = - CrossEntropy(logits, labels)  (To diminish knowledge)
                
                if labels is not None:
                    loss_forget = -torch.nn.functional.cross_entropy(logits, labels)
                else:
                     # If no labels, maximize entropy of predictions
                    probs = torch.softmax(logits, dim=-1)
                    loss_forget = (probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

                # Fisher Elastic Weight Consolidation (EWC) Penalty
                # L_ewc = Σ F_i * (θ_i - θ_orig_i)²
                loss_ewc = 0.0
                for n, p in model.named_parameters():
                    if n in fisher_diag and p.requires_grad:
                        _loss = (fisher_diag[n] * (p - original_params[n]).pow(2)).sum()
                        loss_ewc += _loss
                
                # Total Loss: Minimize ( L_forget + λ * L_ewc )
                # Note: We want to INCREASE entropy/error on forget (Maximize CE), 
                # so minimizing -CE works.
                
                total_loss = loss_forget + self.fisher_lambda * loss_ewc
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += loss_forget.item()
                n += 1

            forget_losses.append(epoch_loss / max(n, 1))
            retain_losses.append(loss_ewc.item() if isinstance(loss_ewc, torch.Tensor) else loss_ewc)

        return model, forget_losses, retain_losses

    def _compute_fisher_diag(
        self, model: nn.Module, loader: DataLoader, device
    ) -> Dict[str, torch.Tensor]:
        """Compute diagonal of Fisher Information Matrix using empirical gradients."""
        fisher = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p.data)

        model.eval()
        for batch in loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else None
            
            model.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            else:
                loss = logits.sum() # Dummy
            
            loss.backward()
            
            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2
        
        # Normalize
        n_batches = len(loader)
        for n in fisher:
            fisher[n] /= n_batches
            
        return fisher

