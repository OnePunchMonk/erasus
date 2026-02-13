"""
Base Metric — Abstract base class for evaluation metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch.nn as nn
from torch.utils.data import DataLoader


class BaseMetric(ABC):
    """Evaluate unlearning quality along a single axis."""

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Union[float, Dict[str, float]]:
        """Return metric value(s)."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
