"""
erasus.data.preprocessing — Data preprocessing utilities.

Provides common transformations and preprocessing pipelines
for different modalities used in unlearning experiments.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PreprocessingPipeline:
    """
    Composable preprocessing pipeline for datasets.

    Usage::

        pipeline = PreprocessingPipeline()
        pipeline.add_step("normalize", lambda x: (x - x.mean()) / x.std())
        pipeline.add_step("clip", lambda x: x.clamp(-3, 3))

        processed = pipeline(raw_tensor)
    """

    def __init__(self):
        self._steps: List[Tuple[str, Callable]] = []

    def add_step(self, name: str, fn: Callable) -> "PreprocessingPipeline":
        """Add a preprocessing step (fluent API)."""
        self._steps.append((name, fn))
        return self

    def __call__(self, data: Any) -> Any:
        for name, fn in self._steps:
            data = fn(data)
        return data

    def __repr__(self) -> str:
        names = [n for n, _ in self._steps]
        return f"PreprocessingPipeline(steps={names})"


def normalize_tensor(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2-normalise a tensor along the given dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-variance standardisation."""
    return (x - x.mean()) / (x.std() + 1e-8)


def create_image_transform(
    size: int = 224,
    normalize: bool = True,
    augment: bool = False,
):
    """
    Create a standard image transform pipeline.

    Returns a torchvision Compose transform.
    """
    from torchvision import transforms as T

    steps = []

    if augment:
        steps.extend([
            T.RandomResizedCrop(size),
            T.RandomHorizontalFlip(),
        ])
    else:
        steps.extend([
            T.Resize(size + 32),
            T.CenterCrop(size),
        ])

    steps.append(T.ToTensor())

    if normalize:
        steps.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ))

    return T.Compose(steps)


class TokenPreprocessor:
    """
    Preprocessing for text token sequences.

    Handles truncation, padding, and special tokens
    for LLM unlearning tasks.
    """

    def __init__(
        self,
        max_length: int = 512,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pad/truncate and create attention mask."""
        seq_len = tokens.shape[-1]

        if seq_len > self.max_length:
            tokens = tokens[..., :self.max_length]
        elif seq_len < self.max_length:
            pad = torch.full(
                (*tokens.shape[:-1], self.max_length - seq_len),
                self.pad_token_id,
                dtype=tokens.dtype,
            )
            tokens = torch.cat([tokens, pad], dim=-1)

        attention_mask = (tokens != self.pad_token_id).long()

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
        }
