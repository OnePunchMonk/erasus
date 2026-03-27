"""
AutoStrategy — Automatic strategy selection based on model and data properties.

Analyses the model type, forget set size, and user-specified constraints to
select the best unlearning strategy automatically.

Example
-------
>>> from erasus import ErasusUnlearner
>>> unlearner = ErasusUnlearner(model=model, strategy="auto")
>>> result = unlearner.fit(forget_data=forget_loader, retain_data=retain_loader)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("auto")
class AutoStrategy(BaseStrategy):
    """
    Meta-strategy that analyses model type, forget set size, and constraints
    to pick the best unlearning strategy at runtime.

    Parameters
    ----------
    model_type : str
        Modality hint: ``"llm"``, ``"vlm"``, ``"diffusion"``, ``"audio"``,
        ``"video"``, or ``"classifier"`` (generic). Used to select
        modality-specific strategies when available.
    goal : str
        Optimisation goal:
        - ``"fast"`` — minimise wall-clock time
        - ``"quality"`` — maximise forgetting quality
        - ``"balanced"`` — trade-off (default)
    """

    def __init__(
        self,
        model_type: str = "classifier",
        goal: str = "balanced",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_type = model_type
        self.goal = goal

    def _select_strategy(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
    ) -> BaseStrategy:
        """Choose a concrete strategy based on heuristics."""
        n_forget = len(forget_loader.dataset)
        n_params = sum(p.numel() for p in model.parameters())

        # LLM-specific strategies
        if self.model_type == "llm":
            if self.goal == "fast":
                cls = strategy_registry.get("ssd")
            elif n_forget > 5000:
                cls = strategy_registry.get("npo")
            else:
                cls = strategy_registry.get("gradient_ascent")
            return cls(**{k: v for k, v in self.kwargs.items() if k != "model_type" and k != "goal"})

        # Diffusion-specific
        if self.model_type == "diffusion":
            cls = strategy_registry.get("concept_erasure")
            return cls(**{k: v for k, v in self.kwargs.items() if k != "model_type" and k != "goal"})

        # VLM-specific
        if self.model_type == "vlm":
            cls = strategy_registry.get("contrastive_unlearning")
            return cls(**{k: v for k, v in self.kwargs.items() if k != "model_type" and k != "goal"})

        # Generic classifier heuristics
        extra_kw = {k: v for k, v in self.kwargs.items() if k != "model_type" and k != "goal"}

        if self.goal == "fast":
            # Prefer lightweight methods
            if n_params < 1_000_000:
                return strategy_registry.get("gradient_ascent")(**extra_kw)
            return strategy_registry.get("neuron_pruning")(**extra_kw)

        if self.goal == "quality":
            if retain_loader is not None:
                return strategy_registry.get("fisher_forgetting")(**extra_kw)
            return strategy_registry.get("gradient_ascent")(**extra_kw)

        # Balanced (default)
        if n_forget < 500 and n_params < 10_000_000:
            return strategy_registry.get("fisher_forgetting")(**extra_kw)
        elif n_forget < 5000:
            return strategy_registry.get("gradient_ascent")(**extra_kw)
        else:
            # Large forget set: SCRUB balances speed and quality
            return strategy_registry.get("scrub")(**extra_kw)

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Select the best strategy and delegate to it."""
        inner = self._select_strategy(model, forget_loader, retain_loader)
        return inner.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            **kwargs,
        )
