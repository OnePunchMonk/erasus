"""
Erasus Exception Hierarchy.
"""


class ErasusError(Exception):
    """Base exception for all Erasus errors."""


class ErasusWarning(UserWarning):
    """Base warning for non-fatal Erasus conditions."""


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


class DatasetError(ErasusError):
    """Raised when a dataset cannot be loaded or validated."""


class EvaluationError(ErasusError):
    """Raised when an evaluation routine fails."""


class BenchmarkError(EvaluationError):
    """Raised when benchmark setup or execution fails."""


class IntegrationError(ErasusError):
    """Raised when an external integration fails."""


class CheckpointError(ErasusError):
    """Raised when checkpoint save/load/push operations fail."""


class WarningPolicyError(ConfigurationError):
    """Raised when an invalid error-handling policy is requested."""


def handle_policy(
    *,
    policy: str,
    message: str,
    error_cls: type[ErasusError] = ErasusError,
    warning_cls: type[ErasusWarning] = ErasusWarning,
) -> None:
    """
    Standardize error/warning/silent behavior across the codebase.

    Parameters
    ----------
    policy : str
        One of ``\"error\"``, ``\"warn\"``, or ``\"silent\"``.
    message : str
        Human-readable message.
    error_cls : type[ErasusError]
        Exception type raised when ``policy='error'``.
    warning_cls : type[ErasusWarning]
        Warning type emitted when ``policy='warn'``.
    """
    import warnings

    if policy == "error":
        raise error_cls(message)
    if policy == "warn":
        warnings.warn(message, warning_cls, stacklevel=2)
        return
    if policy == "silent":
        return
    raise WarningPolicyError(f"Unknown policy '{policy}'. Expected 'error', 'warn', or 'silent'.")
