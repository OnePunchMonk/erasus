"""
LoRA-based Unlearning.

Instead of fine-tuning the entire model, we train a Low-Rank Adapter (LoRA)
Specifically designed to *negate* or *overwrite* the forget set knowledge
while keeping the base model frozen. This is efficient and reversible.

Ref: "LoRA: Low-Rank Adaptation of Large Language Models" suitable for unlearning.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Try importing PEFT, handle if missing
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry
from erasus.core.exceptions import StrategyError


@strategy_registry.register("lora")
class LoRAUnlearningStrategy(BaseStrategy):
    """
    Unlearning via Low-Rank Adaptation (LoRA).
    
    1. Freezes base model.
    2. Injects trainable LoRA layers.
    3. Optimizes LoRA layers to maximize loss on forget set (or minimize Negative Log Likelihood of 'refusal' targets).
    """

    def __init__(
        self,
        rank: int = 8,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        lr: float = 3e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.lr = lr

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        if not PEFT_AVAILABLE:
            raise StrategyError("peft library is required for LoRA unlearning. Install with `pip install peft`.")

        device = next(model.parameters()).device
        
        # Configure LoRA
        # Auto-detect task type roughly
        peft_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules, # Let PEFT auto-detect if None
            lora_dropout=self.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM if "llama" in str(type(model)).lower() or "gpt" in str(type(model)).lower() else None
        )
        
        # Wrap model
        # Note: If model is already a PEFT model, this might need care.
        # Assuming model is the base HuggingFace model or similar.
        # Wrapper handling might differ if input is Erasus ModelWrapper vs raw HF model.
        # Assuming raw HF model inner access via wrapper if needed, but BaseStrategy receives `nn.Module`.
        
        try:
            model = get_peft_model(model, peft_config)
        except Exception as e:
            # Fallback if task type inference fails
            peft_config.task_type = None 
            model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        
        forget_losses = []
        retain_losses = []

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            
            for batch in forget_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None
                
                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                # Strategy: Gradient Ascent on Forget Data
                if labels is not None:
                    loss = -torch.nn.functional.cross_entropy(logits, labels)
                else:
                    loss = -logits.sum() # Dummy
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += -loss.item() # Log the actual CE loss that we are maximizing
                n += 1
            
            forget_losses.append(epoch_loss / max(n, 1))

            # Optional Retain loop to anchor LoRA weights
            if retain_loader:
                epoch_retain_loss = 0.0
                n_retain = 0
                for batch in retain_loader:
                   # ... normal training on retain ...
                   inputs = batch[0].to(device)
                   labels = batch[1].to(device)
                   optimizer.zero_grad()
                   outputs = model(inputs)
                   logits = outputs.logits if hasattr(outputs, "logits") else outputs
                   loss = torch.nn.functional.cross_entropy(logits, labels)
                   loss.backward()
                   optimizer.step()
                   epoch_retain_loss += loss.item()
                   n_retain += 1
                retain_losses.append(epoch_retain_loss / max(n_retain, 1))

        # Return the PEFT model allowed? Yes, it's an nn.Module
        return model, forget_losses, retain_losses

