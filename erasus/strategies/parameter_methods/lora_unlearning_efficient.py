"""
LoRA Unlearning — Efficient parameter unlearning via LoRA adapters.

Instead of modifying full model weights, apply unlearning only to
LoRA adapters. Enables:
- Memory-efficient unlearning of large models
- Fast iteration without retraining from scratch
- Easy rollback (just remove adapter)
- Composition with other LoRA adaptations
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("lora_unlearning")
class LoRAUnlearningStrategy(BaseStrategy):
    """
    Unlearning via Low-Rank Adaptation (LoRA) adapters.

    Applies unlearning only to LoRA layers, keeping base model frozen.
    Enables efficient unlearning of large models.

    Parameters
    ----------
    lora_r : int
        LoRA rank (default 16).
    lora_alpha : float
        LoRA scaling (default 32.0).
    lora_dropout : float
        LoRA dropout (default 0.1).
    target_modules : list of str
        Modules to apply LoRA to (default ["q_proj", "v_proj"]).
    unlearning_lr : float
        Learning rate for adapter unlearning (default 1e-3).
    base_strategy : str
        Strategy to use on adapters ("gradient_ascent", "flat", etc.)
        Default: "gradient_ascent"
    """

    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        unlearning_lr: float = 1e-3,
        base_strategy: str = "gradient_ascent",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.unlearning_lr = unlearning_lr
        self.base_strategy = base_strategy

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Unlearn via LoRA adapters.

        Parameters
        ----------
        model : nn.Module
            Model with LoRA adapters already attached.
        forget_loader : DataLoader
            Forget set.
        retain_loader : DataLoader, optional
            Retain set.
        epochs : int
            Training epochs.

        Returns
        -------
        tuple
            (model, forget_losses, retain_losses)
        """
        # Apply LoRA if not already applied
        model = self._apply_lora_if_needed(model)

        # Get base strategy
        from erasus.core.registry import strategy_registry

        strategy_cls = strategy_registry.get(self.base_strategy)
        strategy = strategy_cls(lr=self.unlearning_lr)

        # Freeze base model, unlearn only on LoRA
        self._freeze_base_model(model)

        # Run unlearning on LoRA adapters
        model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            **kwargs,
        )

        return model, forget_losses, retain_losses

    def _apply_lora_if_needed(self, model: nn.Module) -> nn.Module:
        """Apply LoRA to model if not already applied."""
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError(
                "PEFT library required. Install with: pip install peft"
            )

        # Check if already has LoRA
        if hasattr(model, "peft_config"):
            return model

        # Apply LoRA
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
        return model

    def _freeze_base_model(self, model: nn.Module) -> None:
        """Freeze base model parameters, keep LoRA trainable."""
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True


class LoRAComposition:
    """Compose multiple LoRA adapters for unlearning."""

    def __init__(self, base_model: nn.Module) -> None:
        """
        Initialize composition.

        Parameters
        ----------
        base_model : nn.Module
            Model with LoRA support.
        """
        self.base_model = base_model
        self.adapters = {}

    def add_adapter(
        self,
        adapter_name: str,
        lora_r: int = 16,
    ) -> None:
        """
        Add a new LoRA adapter.

        Parameters
        ----------
        adapter_name : str
            Name for the adapter.
        lora_r : int
            LoRA rank.
        """
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("PEFT library required")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=32.0,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
        )

        adapted_model = get_peft_model(self.base_model, config)
        self.adapters[adapter_name] = adapted_model

    def switch_adapter(self, adapter_name: str) -> None:
        """Switch to a specific adapter."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter {adapter_name} not found")

        try:
            self.base_model.set_adapter(adapter_name)
        except AttributeError:
            print(f"Switching to {adapter_name}")

    def unload_adapter(self, adapter_name: str) -> None:
        """Remove an adapter."""
        if adapter_name in self.adapters:
            del self.adapters[adapter_name]
