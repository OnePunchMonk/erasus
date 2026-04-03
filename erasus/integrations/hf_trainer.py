"""
Hugging Face Trainer integration for post-training unlearning.
"""

from __future__ import annotations

from typing import Any, Optional

from erasus.core.registry import strategy_registry

import erasus.strategies  # noqa: F401

try:  # pragma: no cover - optional dependency
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:  # pragma: no cover - optional dependency
    class _TrainerCallbackBase:  # type: ignore[override]
        pass


class UnlearningTrainerCallback(_TrainerCallbackBase):
    """Run an unlearning strategy after standard trainer completion."""

    def __init__(
        self,
        strategy_name: str,
        forget_loader: Any,
        retain_loader: Any = None,
        strategy_kwargs: Optional[dict[str, Any]] = None,
        unlearn_epochs: int = 1,
    ) -> None:
        self.strategy_name = strategy_name
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.strategy_kwargs = strategy_kwargs or {}
        self.unlearn_epochs = unlearn_epochs

    def run_unlearning(self, model: Any) -> Any:
        strategy_cls = strategy_registry.get(self.strategy_name)
        strategy = strategy_cls(**self.strategy_kwargs)
        unlearned_model, _, _ = strategy.unlearn(
            model=model,
            forget_loader=self.forget_loader,
            retain_loader=self.retain_loader,
            epochs=self.unlearn_epochs,
        )
        return unlearned_model

    def on_train_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
        if model is not None:
            self.run_unlearning(model)
        return control


def attach_unlearning_callback(trainer: Any, callback: UnlearningTrainerCallback) -> Any:
    """Attach the callback to a trainer-like object."""
    if hasattr(trainer, "add_callback"):
        trainer.add_callback(callback)
    else:
        callbacks = getattr(trainer, "callbacks", [])
        callbacks.append(callback)
        trainer.callbacks = callbacks
    return trainer
