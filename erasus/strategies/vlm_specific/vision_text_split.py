"""
erasus.strategies.vlm_specific.vision_text_split — Separate encoder update strategy.

Updates the vision and text encoders independently during unlearning,
allowing asymmetric learning rates and selective freezing to minimise
cross-modal interference.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("vision_text_split")
class VisionTextSplitStrategy(BaseStrategy):
    """
    Separate encoder update strategy for VLMs.

    Rather than applying a single optimiser to the whole model, this
    strategy creates independent optimisers for vision and text
    encoders with potentially different learning rates and update
    frequencies.

    Parameters
    ----------
    vision_lr : float
        Learning rate for the vision encoder.
    text_lr : float
        Learning rate for the text encoder.
    freeze_vision : bool
        If True, freeze vision encoder entirely (text-only unlearning).
    freeze_text : bool
        If True, freeze text encoder entirely (vision-only unlearning).
    vision_epochs : int
        Number of gradient-ascent epochs for vision encoder.
    text_epochs : int
        Number of gradient-ascent epochs for text encoder.
    retain_weight : float
        Weight for retain loss (utility preservation).
    """

    def __init__(
        self,
        vision_lr: float = 1e-5,
        text_lr: float = 1e-4,
        freeze_vision: bool = False,
        freeze_text: bool = False,
        vision_epochs: int = 3,
        text_epochs: int = 5,
        retain_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vision_lr = vision_lr
        self.text_lr = text_lr
        self.freeze_vision = freeze_vision
        self.freeze_text = freeze_text
        self.vision_epochs = vision_epochs
        self.text_epochs = text_epochs
        self.retain_weight = retain_weight

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Run split encoder unlearning.

        Phase 1: Update text encoder to disrupt forget associations.
        Phase 2: Update vision encoder to disrupt forget features.
        Phase 3: Fine-tune both on retain data to recover utility.
        """
        device = next(model.parameters()).device
        model.train()

        # Identify encoder parameter groups
        vision_params, text_params, other_params = self._split_parameters(model)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        # Phase 1: Text encoder gradient ascent on forget data
        if not self.freeze_text and text_params:
            # Freeze vision
            for p in vision_params:
                p.requires_grad_(False)
            for p in other_params:
                p.requires_grad_(False)
            for p in text_params:
                p.requires_grad_(True)

            text_opt = torch.optim.Adam(text_params, lr=self.text_lr)

            for epoch in range(self.text_epochs):
                epoch_loss = 0.0
                n_batches = 0
                for batch in forget_loader:
                    text_opt.zero_grad()
                    loss = self._compute_forget_loss(model, batch, device)
                    (-loss).backward()  # Gradient ascent
                    text_opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                if n_batches > 0:
                    forget_losses.append(epoch_loss / n_batches)

        # Phase 2: Vision encoder gradient ascent on forget data
        if not self.freeze_vision and vision_params:
            for p in text_params:
                p.requires_grad_(False)
            for p in other_params:
                p.requires_grad_(False)
            for p in vision_params:
                p.requires_grad_(True)

            vision_opt = torch.optim.Adam(vision_params, lr=self.vision_lr)

            for epoch in range(self.vision_epochs):
                epoch_loss = 0.0
                n_batches = 0
                for batch in forget_loader:
                    vision_opt.zero_grad()
                    loss = self._compute_forget_loss(model, batch, device)
                    (-loss).backward()
                    vision_opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                if n_batches > 0:
                    forget_losses.append(epoch_loss / n_batches)

        # Phase 3: Retain fine-tuning (both encoders)
        if retain_loader is not None:
            for p in model.parameters():
                p.requires_grad_(True)
            retain_opt = torch.optim.Adam([
                {"params": vision_params, "lr": self.vision_lr * 0.1},
                {"params": text_params, "lr": self.text_lr * 0.1},
                {"params": other_params, "lr": self.text_lr * 0.1},
            ])

            retain_epochs = max(self.vision_epochs, self.text_epochs) // 2 + 1
            for epoch in range(retain_epochs):
                epoch_loss = 0.0
                n_batches = 0
                for batch in retain_loader:
                    retain_opt.zero_grad()
                    loss = self._compute_retain_loss(model, batch, device)
                    loss.backward()
                    retain_opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                if n_batches > 0:
                    retain_losses.append(epoch_loss / n_batches)

        # Restore all parameter gradients
        for p in model.parameters():
            p.requires_grad_(True)

        return model, forget_losses, retain_losses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_parameters(
        self, model: nn.Module
    ) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        """
        Split model parameters into vision, text, and other groups.

        Uses naming heuristics: parameters containing 'visual', 'vision',
        'image' go to vision; 'text', 'token', 'lm_head' go to text.
        """
        vision_params: List[torch.nn.Parameter] = []
        text_params: List[torch.nn.Parameter] = []
        other_params: List[torch.nn.Parameter] = []

        vision_keys = ("visual", "vision", "image_encoder", "img_encoder", "vit", "patch_embed")
        text_keys = ("text", "token", "lm_head", "word_embed", "transformer", "gpt", "bert", "llama")

        for name, param in model.named_parameters():
            name_lower = name.lower()
            if any(k in name_lower for k in vision_keys):
                vision_params.append(param)
            elif any(k in name_lower for k in text_keys):
                text_params.append(param)
            else:
                other_params.append(param)

        return vision_params, text_params, other_params

    @staticmethod
    def _compute_forget_loss(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
        """Compute cross-entropy loss on a forget batch."""
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else None
        else:
            inputs = batch.to(device)
            labels = None

        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        if labels is not None:
            return F.cross_entropy(logits, labels)
        return logits.mean()

    @staticmethod
    def _compute_retain_loss(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
        """Compute cross-entropy loss on a retain batch."""
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else None
        else:
            inputs = batch.to(device)
            labels = None

        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        if labels is not None:
            return F.cross_entropy(logits, labels)
        return logits.mean()
