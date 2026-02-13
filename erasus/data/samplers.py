"""
erasus.data.samplers — Custom samplers for unlearning.

Provides weighted, balanced, and curriculum-aware sampling strategies
for forget/retain data during unlearning.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler


class ForgetRetainBalancedSampler(Sampler):
    """
    Balances sampling between forget and retain sets.

    Ensures each mini-batch contains a fixed ratio of forget vs
    retain samples, which is important for stable unlearning.

    Parameters
    ----------
    forget_indices : list[int]
        Indices of forget samples in the combined dataset.
    retain_indices : list[int]
        Indices of retain samples.
    forget_ratio : float
        Fraction of each batch from the forget set (default 0.5).
    batch_size : int
        Total batch size.
    """

    def __init__(
        self,
        forget_indices: List[int],
        retain_indices: List[int],
        forget_ratio: float = 0.5,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.forget_indices = list(forget_indices)
        self.retain_indices = list(retain_indices)
        self.forget_ratio = forget_ratio
        self.batch_size = batch_size
        self.seed = seed

        self.n_forget_per_batch = max(1, int(batch_size * forget_ratio))
        self.n_retain_per_batch = batch_size - self.n_forget_per_batch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)

        forget_pool = list(self.forget_indices)
        retain_pool = list(self.retain_indices)
        rng.shuffle(forget_pool)
        rng.shuffle(retain_pool)

        fi, ri = 0, 0
        for _ in range(len(self)):
            batch = []

            # Sample from forget set
            for _ in range(self.n_forget_per_batch):
                if fi >= len(forget_pool):
                    rng.shuffle(forget_pool)
                    fi = 0
                batch.append(forget_pool[fi])
                fi += 1

            # Sample from retain set
            for _ in range(self.n_retain_per_batch):
                if ri >= len(retain_pool):
                    rng.shuffle(retain_pool)
                    ri = 0
                batch.append(retain_pool[ri])
                ri += 1

            yield from batch

    def __len__(self) -> int:
        total = len(self.forget_indices) + len(self.retain_indices)
        return math.ceil(total / self.batch_size)


class DifficultyAwareSampler(Sampler):
    """
    Samples harder-to-forget examples more frequently.

    Uses loss values or confidence scores to determine difficulty.

    Parameters
    ----------
    scores : list[float]
        Per-sample difficulty scores (higher = harder to forget).
    temperature : float
        Sampling temperature (higher = more uniform, lower = more focused).
    """

    def __init__(
        self,
        scores: List[float],
        temperature: float = 1.0,
        seed: int = 42,
    ):
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.temperature = temperature
        self.seed = seed

        # Compute sampling weights via softmax
        self.weights = torch.softmax(self.scores / temperature, dim=0)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        indices = torch.multinomial(
            self.weights,
            num_samples=len(self.scores),
            replacement=True,
            generator=generator,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self.scores)
