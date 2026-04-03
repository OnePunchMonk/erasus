"""
Base classes for adversarial unlearning evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch.nn as nn
from torch.utils.data import DataLoader


class BaseAdversarialTest(ABC):
    """Common interface for adversarial unlearning tests."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the adversarial test and return a structured result."""
        raise NotImplementedError
