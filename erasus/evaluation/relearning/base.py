"""
Base classes for relearning robustness attacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch.nn as nn
from torch.utils.data import DataLoader


class BaseRelearningAttack(ABC):
    """Abstract interface for relearning robustness attacks."""

    @abstractmethod
    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the relearning attack and return a result dictionary."""
        ...
