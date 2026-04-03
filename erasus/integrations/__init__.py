"""Third-party integrations for Erasus."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["HuggingFaceHub"]


def __getattr__(name: str) -> Any:
    """Lazily resolve optional integrations."""
    if name != "HuggingFaceHub":
        raise AttributeError(f"module 'erasus.integrations' has no attribute '{name}'")

    value = getattr(import_module("erasus.integrations.huggingface"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
