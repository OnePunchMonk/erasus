"""
Erasus Exception Hierarchy.
"""


class ErasusError(Exception):
    """Base exception for all Erasus errors."""


class ModelNotFoundError(ErasusError):
    """Raised when a requested model is not available."""


class StrategyError(ErasusError):
    """Raised when a strategy fails during unlearning."""


class SelectorError(ErasusError):
    """Raised when coreset selection fails."""


class ConfigurationError(ErasusError):
    """Raised for invalid configuration."""


class MetricError(ErasusError):
    """Raised when a metric computation fails."""
