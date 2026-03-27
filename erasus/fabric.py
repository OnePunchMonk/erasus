"""
erasus.fabric — Composable primitives for custom unlearning loops.

Inspired by PyTorch Lightning Fabric, these standalone functions let
users write their own training loops while leveraging Erasus utilities
for coreset selection, metric computation, and gradient operations.

Example
-------
>>> from erasus.fabric import select_coreset, compute_forgetting_quality, apply_gradient_ascent
>>>
>>> # Use in your own loop:
>>> indices = select_coreset("influence", model, forget_loader, k=100)
>>> for epoch in range(5):
...     apply_gradient_ascent(model, forget_loader, lr=1e-4)
>>> quality = compute_forgetting_quality(model, forget_loader)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ======================================================================
# Coreset selection
# ======================================================================


def select_coreset(
    selector: str,
    model: nn.Module,
    data_loader: DataLoader,
    k: int,
    **kwargs: Any,
) -> List[int]:
    """
    Run a named selector and return selected sample indices.

    Parameters
    ----------
    selector : str
        Selector name (e.g. ``"influence"``, ``"gradient_norm"``, ``"random"``).
    model : nn.Module
    data_loader : DataLoader
    k : int
        Number of samples to select.

    Returns
    -------
    list[int]
        Selected sample indices.
    """
    from erasus.core.registry import selector_registry
    import erasus.selectors  # noqa: F401 — ensure registration

    sel_cls = selector_registry.get(selector)
    sel = sel_cls(**kwargs)
    return sel.select(model=model, data_loader=data_loader, k=k)


# ======================================================================
# One-step unlearning operations
# ======================================================================


def apply_gradient_ascent(
    model: nn.Module,
    forget_loader: DataLoader,
    lr: float = 1e-4,
    retain_loader: Optional[DataLoader] = None,
    epochs: int = 1,
) -> nn.Module:
    """
    Apply gradient ascent on the forget set (single or multi-epoch).

    Parameters
    ----------
    model : nn.Module
    forget_loader : DataLoader
    lr : float
    retain_loader : DataLoader, optional
    epochs : int

    Returns
    -------
    nn.Module
        The model (modified in-place).
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        for batch in forget_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            (-loss).backward()  # ascent
            optimizer.step()

        if retain_loader is not None:
            for batch in retain_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model


# ======================================================================
# Metric computation
# ======================================================================


def compute_forgetting_quality(
    model: nn.Module,
    forget_loader: DataLoader,
) -> float:
    """
    Compute forgetting quality: accuracy on the forget set (lower = better).

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0

    with torch.no_grad():
        for batch in forget_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else None
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if labels is not None:
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    return correct / max(total, 1)


def compute_model_utility(
    model: nn.Module,
    retain_loader: DataLoader,
) -> float:
    """
    Compute model utility: accuracy on the retain set (higher = better).

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    return compute_forgetting_quality(model, retain_loader)


def compute_mia_score(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
) -> float:
    """
    Compute a simple membership inference attack score.

    Uses loss gap between forget and retain as a proxy.
    Returns AUC-like score; 0.5 = ideal (model treats both equally).
    """
    model.eval()
    device = next(model.parameters()).device

    def _avg_loss(loader: DataLoader) -> float:
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                if labels is not None:
                    loss = F.cross_entropy(logits, labels, reduction="sum")
                    total += loss.item()
                    count += labels.size(0)
        return total / max(count, 1)

    forget_loss = _avg_loss(forget_loader)
    retain_loss = _avg_loss(retain_loader)

    # Simple sigmoid-based score: large gap → high distinguishability
    gap = forget_loss - retain_loss
    score = 1.0 / (1.0 + torch.tensor(gap).exp().item())
    return score


# ======================================================================
# Utility helpers
# ======================================================================


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Enable gradient checkpointing on the model if supported.

    Works with HuggingFace models (``gradient_checkpointing_enable()``)
    and any model with a ``gradient_checkpointing`` attribute.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
    return model
