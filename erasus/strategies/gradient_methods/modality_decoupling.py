"""
⭐ Modality Decoupling Strategy — THE CENTERPIECE OF ERASUS ⭐

For Vision-Language Models like CLIP, naive gradient ascent causes
'forgetting cascade' — unlearning in one modality destroys the other.

Our solution: decouple image and text encoder updates with differential
learning rates and a cross-modal alignment preservation loss.

Key Innovation Points
---------------------
- Differential learning rates for different modalities
- Vision LR typically lower than text LR  (vision is more brittle)
- Cross-modal alignment preservation loss
- Prevents 'forgetting cascade' phenomenon
- Maintains zero-shot capabilities

Section 4.2.1 of the specification.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("modality_decoupling")
class ModalityDecouplingStrategy(BaseStrategy):
    """
    Erasus core innovation: decouple vision / text updates.

    Prevents catastrophic collapse in multimodal models by using
    separate optimisers and learning rates for each modality encoder,
    and an alignment-preservation loss on the retain set.

    Parameters
    ----------
    vision_lr : float
        Learning rate for the vision encoder (kept low to avoid brittleness).
    text_lr : float
        Learning rate for the text encoder.
    alignment_weight : float
        Weight of the retain-set alignment preservation term.
    """

    def __init__(
        self,
        vision_lr: float = 1e-5,
        text_lr: float = 1e-4,
        alignment_weight: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vision_lr = vision_lr
        self.text_lr = text_lr
        self.alignment_weight = alignment_weight

    # ------------------------------------------------------------------
    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Run modality-decoupled unlearning on a CLIP-style model.

        The model must expose:
        - ``model.vision_model``
        - ``model.text_model``
        - ``model.get_image_features(pixel_values=...)``
        - ``model.get_text_features(input_ids=..., attention_mask=...)``
        - ``model.logit_scale``
        """
        model.train()
        device = next(model.parameters()).device

        # Separate optimisers for each modality
        vision_optimizer = torch.optim.Adam(
            model.vision_model.parameters(), lr=self.vision_lr,
        )
        text_optimizer = torch.optim.Adam(
            model.text_model.parameters(), lr=self.text_lr,
        )

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_forget_loss = 0.0
            epoch_retain_loss = 0.0
            n_forget = 0
            n_retain = 0

            # ═══════════════════  FORGET PHASE  ═══════════════════
            for batch in forget_loader:
                pixel_values, input_ids, attention_mask = self._unpack_batch(
                    batch, device,
                )
                image_features = model.get_image_features(pixel_values=pixel_values)
                text_features = model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask,
                )

                # Normalise embeddings
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # Symmetric contrastive loss (to MAXIMISE)
                logit_scale = model.logit_scale.exp()
                logits = image_features @ text_features.T * logit_scale
                labels = torch.arange(logits.size(0), device=device)

                loss = (
                    F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.T, labels)
                ) / 2

                # Gradient ascent → negate loss
                vision_optimizer.zero_grad()
                text_optimizer.zero_grad()
                (-loss).backward()
                vision_optimizer.step()
                text_optimizer.step()

                epoch_forget_loss += loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget_loss / max(n_forget, 1))

            # ═══════════════════  RETAIN PHASE  ═══════════════════
            if retain_loader is not None:
                for batch in retain_loader:
                    pixel_values, input_ids, attention_mask = self._unpack_batch(
                        batch, device,
                    )
                    image_features = model.get_image_features(pixel_values=pixel_values)
                    text_features = model.get_text_features(
                        input_ids=input_ids, attention_mask=attention_mask,
                    )

                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)

                    logit_scale = model.logit_scale.exp()
                    logits = image_features @ text_features.T * logit_scale
                    labels = torch.arange(logits.size(0), device=device)

                    retain_loss = (
                        F.cross_entropy(logits, labels)
                        + F.cross_entropy(logits.T, labels)
                    ) / 2

                    # Gradient descent with alignment weight
                    vision_optimizer.zero_grad()
                    text_optimizer.zero_grad()
                    retain_loss.backward()

                    # Apply alignment-scaled manual update
                    with torch.no_grad():
                        for param in model.vision_model.parameters():
                            if param.grad is not None:
                                param.data.sub_(
                                    param.grad * self.alignment_weight * self.vision_lr
                                )
                        for param in model.text_model.parameters():
                            if param.grad is not None:
                                param.data.sub_(
                                    param.grad * self.alignment_weight * self.text_lr
                                )

                    epoch_retain_loss += retain_loss.item()
                    n_retain += 1

                retain_losses.append(epoch_retain_loss / max(n_retain, 1))

        return model, forget_losses, retain_losses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch, device):
        """
        Extract pixel_values, input_ids, attention_mask from a batch.

        Supports:
        - dict  (HF processor output)
        - tuple (pixel_values, input_ids, attention_mask)
        - tuple (pixel_values, input_ids)
        """
        if isinstance(batch, dict):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            return pixel_values, input_ids, attention_mask

        # tuple / list: (x, y) for classification or (pixel_values, input_ids[, attention_mask])
        pixel_values = batch[0].to(device)
        second = batch[1].to(device) if len(batch) > 1 else pixel_values
        # If second is 1D (class labels), use pixel_values for both modalities
        if second.dim() == 1 or (second.dim() == 2 and second.size(-1) == 1):
            input_ids = pixel_values
        else:
            input_ids = second
        attention_mask = batch[2].to(device) if len(batch) > 2 else None
        return pixel_values, input_ids, attention_mask
