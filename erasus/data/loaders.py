"""
Data Loaders — Convenience wrappers for creating forget / retain splits.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


def create_forget_retain_loaders(
    dataset: Dataset,
    forget_indices: list[int],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split a dataset into forget and retain data loaders.

    Parameters
    ----------
    dataset : Dataset
        Full training dataset.
    forget_indices : list[int]
        Indices of samples to forget.
    batch_size : int
    shuffle : bool
    num_workers : int

    Returns
    -------
    (forget_loader, retain_loader)
    """
    all_indices = set(range(len(dataset)))
    retain_indices = sorted(all_indices - set(forget_indices))

    forget_dataset = Subset(dataset, forget_indices)
    retain_dataset = Subset(dataset, retain_indices)

    forget_loader = DataLoader(
        forget_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
    )
    retain_loader = DataLoader(
        retain_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
    )

    return forget_loader, retain_loader


def random_forget_retain_split(
    dataset: Dataset,
    forget_ratio: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Randomly split a dataset into forget and retain sets."""
    n = len(dataset)
    n_forget = max(1, int(n * forget_ratio))
    n_retain = n - n_forget

    generator = torch.Generator().manual_seed(seed)
    forget_ds, retain_ds = random_split(dataset, [n_forget, n_retain], generator=generator)

    return (
        DataLoader(forget_ds, batch_size=batch_size, shuffle=True),
        DataLoader(retain_ds, batch_size=batch_size, shuffle=True),
    )
