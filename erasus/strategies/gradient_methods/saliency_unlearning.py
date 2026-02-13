"""
erasus.strategies.gradient_methods.saliency_unlearning — Saliency-guided unlearning.

Uses gradient saliency maps to identify and selectively modify the most
relevant parameters for forgetting, minimising collateral damage to
retain-set performance.

Reference: Saliency-based approaches adapted from interpretability
           literature (Simonyan et al., 2013) applied to unlearning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("saliency_unlearning")
class SaliencyUnlearningStrategy(BaseStrategy):
    """
    Saliency-guided unlearning.

    Computes per-parameter saliency scores w.r.t. the forget set,
    then applies gradient ascent **only** to the most salient parameters.
    Less salient parameters are frozen, preserving utility.

    Parameters
    ----------
    lr : float
        Learning rate.
    saliency_threshold : float
        Fraction of parameters to modify (top-k by saliency).
    weight_decay : float
        Optional weight decay.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        saliency_threshold: float = 0.2,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.saliency_threshold = saliency_threshold
        self.weight_decay = weight_decay

    def _compute_saliency(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-parameter saliency scores from forget data."""
        model.train()
        device = next(model.parameters()).device
        saliency = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

        n_batches = 0
        for batch in forget_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(logits, labels)

            model.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    saliency[name] += param.grad.abs()
            n_batches += 1

        # Average
        for name in saliency:
            saliency[name] /= max(n_batches, 1)

        return saliency

    def _create_saliency_mask(
        self,
        saliency: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Create binary masks: 1 for top-k salient, 0 otherwise."""
        # Flatten all saliency scores to find global threshold
        all_scores = torch.cat([s.flatten() for s in saliency.values()])
        k = int(len(all_scores) * self.saliency_threshold)
        if k == 0:
            k = 1
        threshold = torch.topk(all_scores, k).values[-1]

        masks = {}
        for name, scores in saliency.items():
            masks[name] = (scores >= threshold).float()

        return masks

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Run saliency-guided unlearning."""
        device = next(model.parameters()).device

        # Phase 1: Compute saliency and create masks
        saliency = self._compute_saliency(model, forget_loader)
        masks = self._create_saliency_mask(saliency)

        # Phase 2: Gradient ascent on masked parameters only
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                (-loss).backward()  # Gradient ascent

                # Mask gradients — only update salient parameters
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in masks:
                            param.grad.mul_(masks[name])

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            forget_losses.append(epoch_loss / max(n_batches, 1))

            # Retain pass
            if retain_loader is not None:
                epoch_retain = 0.0
                n_retain = 0
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = F.cross_entropy(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()

                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.grad is not None and name in masks:
                                param.grad.mul_(masks[name])

                    optimizer.step()
                    epoch_retain += loss.item()
                    n_retain += 1
                retain_losses.append(epoch_retain / max(n_retain, 1))

        return model, forget_losses, retain_losses
