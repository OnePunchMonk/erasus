"""
Base helpers for inference-time unlearning strategies.
"""

from __future__ import annotations

from typing import Any

from erasus.core.base_strategy import BaseStrategy


class BaseInferenceTimeStrategy(BaseStrategy):
    """
    Base class for inference-time strategies that do not train weights.
    """

    requires_training = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
