"""
Concept Erasure for Diffusion Models (ESD).

Paper: Erasing Concepts from Diffusion Models (Gandikota et al., ICCV 2023)
Section 4.2.3.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("concept_erasure")
class ConceptErasureStrategy(BaseStrategy):
    """
    ESD: Erase specific concepts from diffusion models.

    Examples:
    - Remove artist style ('Van Gogh')
    - Remove objects ('Snoopy')
    - Remove NSFW content
    """

    def __init__(self, lr: float = 1e-5, retain_every: int = 5, **kwargs: Any):
        super().__init__(**kwargs)
        self.lr = lr
        self.retain_every = retain_every

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        concept_prompts: Optional[List[str]] = None,
        retain_prompts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Requires the model to have: .unet, .text_encoder, .tokenizer, .scheduler
        Falls back to gradient ascent when model lacks diffusion components.
        """
        has_diffusion = (
            hasattr(model, "unet")
            and hasattr(model, "scheduler")
            and model.scheduler is not None
            and hasattr(model, "tokenizer")
            and model.tokenizer is not None
        )
        if not has_diffusion:
            return self._fallback_gradient_ascent(model, forget_loader, retain_loader, epochs)

        optimizer = torch.optim.Adam(model.unet.parameters(), lr=self.lr)
        device = next(model.unet.parameters()).device
        forget_losses: List[float] = []
        retain_losses: List[float] = []

        concept_prompts = concept_prompts or []
        retain_prompts = retain_prompts or []

        for step in range(epochs):
            step_forget = 0.0

            for prompt in concept_prompts:
                text_embeddings = self._encode_prompt(model, prompt, device)
                t = torch.randint(0, 1000, (1,), device=device)
                latent = torch.randn(1, 4, 64, 64, device=device)
                noise = torch.randn_like(latent)
                noisy_latent = model.scheduler.add_noise(latent, noise, t)

                noise_pred = model.unet(
                    noisy_latent, t, encoder_hidden_states=text_embeddings,
                ).sample

                forget_loss = -F.mse_loss(noise_pred, noise)
                optimizer.zero_grad()
                forget_loss.backward()
                optimizer.step()
                step_forget += forget_loss.item()

            forget_losses.append(step_forget / max(len(concept_prompts), 1))

            # Retain pass
            if step % self.retain_every == 0 and retain_prompts:
                step_retain = 0.0
                for prompt in retain_prompts:
                    text_embeddings = self._encode_prompt(model, prompt, device)
                    t = torch.randint(0, 1000, (1,), device=device)
                    latent = torch.randn(1, 4, 64, 64, device=device)
                    noise = torch.randn_like(latent)
                    noisy_latent = model.scheduler.add_noise(latent, noise, t)

                    noise_pred = model.unet(
                        noisy_latent, t, encoder_hidden_states=text_embeddings,
                    ).sample

                    retain_loss = F.mse_loss(noise_pred, noise)
                    optimizer.zero_grad()
                    retain_loss.backward()
                    optimizer.step()
                    step_retain += retain_loss.item()

                retain_losses.append(step_retain / max(len(retain_prompts), 1))

        return model, forget_losses, retain_losses

    def _fallback_gradient_ascent(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """Gradient ascent fallback for non-diffusion models."""
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        forget_losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in forget_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None
                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = -F.cross_entropy(logits, labels) if labels is not None else -logits.sum()
                loss.backward()
                optimizer.step()
                epoch_loss += (-loss.item())
                n += 1
            forget_losses.append(epoch_loss / max(n, 1))
        return model, forget_losses, []

    @staticmethod
    def _encode_prompt(model, prompt: str, device) -> torch.Tensor:
        tokens = model.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True,
        ).input_ids.to(device)
        return model.text_encoder(tokens)[0]
