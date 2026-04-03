"""Third-party integrations for Erasus."""

from __future__ import annotations

from importlib import import_module
from typing import Any

STABLE_EXPORTS = ["HuggingFaceHub"]
EXPERIMENTAL_EXPORTS = ["UnlearningTrainerCallback", "attach_unlearning_callback"]
PUBLIC_API_STATUS = {
    "HuggingFaceHub": "stable",
    "UnlearningTrainerCallback": "experimental",
    "attach_unlearning_callback": "experimental",
}

__all__ = STABLE_EXPORTS + EXPERIMENTAL_EXPORTS


def __getattr__(name: str) -> Any:
    """Lazily resolve optional integrations."""
    if name == "HuggingFaceHub":
        value = getattr(import_module("erasus.integrations.huggingface"), name)
    elif name in {"UnlearningTrainerCallback", "attach_unlearning_callback"}:
        value = getattr(import_module("erasus.integrations.hf_trainer"), name)
    else:
        raise AttributeError(f"module 'erasus.integrations' has no attribute '{name}'")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
