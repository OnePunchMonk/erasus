"""
Registry — Decorator-based plugin system for dynamic component registration.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """
    Global registry for strategies, selectors, models, and metrics.

    Usage::

        strategy_registry = Registry("strategies")

        @strategy_registry.register("gradient_ascent")
        class GradientAscentStrategy(BaseStrategy):
            ...

        # Later:
        cls = strategy_registry.get("gradient_ascent")
        instance = cls(**kwargs)
    """

    _registries: Dict[str, "Registry"] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[str, Type] = {}
        Registry._registries[name] = self

    def register(self, name: str) -> Callable:
        """Decorator to register a class under *name*."""

        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(
                    f"'{name}' already registered in {self.name} registry."
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        """Retrieve a registered class by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: [{available}]"
            )
        return self._registry[name]

    def list(self) -> list[str]:
        """List all registered names."""
        return sorted(self._registry.keys())

    @classmethod
    def get_registry(cls, name: str) -> "Registry":
        if name not in cls._registries:
            raise KeyError(f"No registry named '{name}'.")
        return cls._registries[name]


# ---------- Global registries ----------
strategy_registry = Registry("strategies")
selector_registry = Registry("selectors")
model_registry = Registry("models")
metric_registry = Registry("metrics")
