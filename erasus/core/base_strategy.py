"""
Base Strategy — Abstract base class for unlearning algorithms.

A Strategy defines *how* to modify model weights to forget specific data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseStrategy(ABC):
    """
    Abstract base for all unlearning strategies.

    Subclasses implement ``unlearn()`` which takes a model, forget loader,
    and optionally a retain loader, and returns the modified model along
    with loss histories.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.device: str = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Execute the unlearning algorithm.

        Parameters
        ----------
        model : nn.Module
            Model to modify.
        forget_loader : DataLoader
            Data to forget.
        retain_loader : DataLoader, optional
            Data whose performance should be preserved.
        epochs : int
            Number of training epochs / steps.

        Returns
        -------
        (model, forget_losses, retain_losses)
        """
        ...
