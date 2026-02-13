"""
Base Selector — Abstract base class for coreset selection methods.

A Selector identifies the most influential samples in the forget set,
reducing unlearning compute by up to 90%.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseSelector(ABC):
    """
    Abstract base class for coreset selectors.

    All selectors must implement ``select()`` which returns the indices
    of the most influential samples in the provided data loader.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        """
        Select *k* most influential samples.

        Parameters
        ----------
        model : nn.Module
            The current model whose forget set we are pruning.
        data_loader : DataLoader
            Forget-set data loader.
        k : int
            Number of samples to select.

        Returns
        -------
        List[int]
            Indices of selected samples in the underlying dataset.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(
        model: nn.Module,
        data_loader: DataLoader,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """Extract last-hidden-state features for all samples."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        all_features: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                outputs = model(inputs)
                # Handle multiple output formats
                if isinstance(outputs, torch.Tensor):
                    feats = outputs
                elif hasattr(outputs, "last_hidden_state"):
                    feats = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, "logits"):
                    feats = outputs.logits
                else:
                    feats = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                all_features.append(feats.cpu())

        return torch.cat(all_features, dim=0).numpy()

    @staticmethod
    def _compute_gradient(
        model: nn.Module,
        batch: Any,
        loss_fn: Optional[Any] = None,
    ) -> torch.Tensor:
        """Compute flattened gradient for a single batch."""
        model.zero_grad()
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        labels = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        if loss_fn is not None:
            loss = loss_fn(logits, labels)
        elif labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        else:
            loss = logits.sum()

        loss.backward()

        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
        return torch.cat(grads)
