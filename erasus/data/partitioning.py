"""
erasus.data.partitioning — Dataset partitioning for unlearning.

Provides utilities for splitting datasets into forget/retain sets,
with support for class-based, sample-based, and random partitions.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset, Subset


def partition_by_class(
    dataset: Dataset,
    forget_classes: List[int],
    label_fn: Optional[callable] = None,
) -> Tuple[Subset, Subset]:
    """
    Split dataset by class labels.

    Parameters
    ----------
    dataset : Dataset
        Full dataset.
    forget_classes : list[int]
        Classes to include in the forget set.
    label_fn : callable, optional
        Function to extract label from a sample. Default: ``sample[1]``.

    Returns
    -------
    (forget_subset, retain_subset)
    """
    if label_fn is None:
        label_fn = lambda sample: sample[1]

    forget_indices = []
    retain_indices = []

    for i in range(len(dataset)):
        sample = dataset[i]
        label = label_fn(sample)
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label in forget_classes:
            forget_indices.append(i)
        else:
            retain_indices.append(i)

    return Subset(dataset, forget_indices), Subset(dataset, retain_indices)


def partition_by_indices(
    dataset: Dataset,
    forget_indices: List[int],
) -> Tuple[Subset, Subset]:
    """
    Split dataset by explicit sample indices.

    Parameters
    ----------
    dataset : Dataset
        Full dataset.
    forget_indices : list[int]
        Indices of samples to forget.

    Returns
    -------
    (forget_subset, retain_subset)
    """
    forget_set = set(forget_indices)
    retain_indices = [i for i in range(len(dataset)) if i not in forget_set]
    return Subset(dataset, forget_indices), Subset(dataset, retain_indices)


def partition_random(
    dataset: Dataset,
    forget_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Randomly split dataset into forget/retain.

    Parameters
    ----------
    dataset : Dataset
        Full dataset.
    forget_fraction : float
        Fraction of samples to forget (0–1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (forget_subset, retain_subset)
    """
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_forget = int(n * forget_fraction)
    forget_indices = indices[:n_forget]
    retain_indices = indices[n_forget:]

    return Subset(dataset, forget_indices), Subset(dataset, retain_indices)


def partition_by_attribute(
    dataset: Dataset,
    attribute_fn: callable,
    forget_condition: callable,
) -> Tuple[Subset, Subset]:
    """
    Split dataset based on arbitrary sample attributes.

    Parameters
    ----------
    attribute_fn : callable
        Extracts the attribute from a sample.
    forget_condition : callable
        Returns True if the sample should be forgotten.

    Returns
    -------
    (forget_subset, retain_subset)
    """
    forget_idx = []
    retain_idx = []

    for i in range(len(dataset)):
        attr = attribute_fn(dataset[i])
        if forget_condition(attr):
            forget_idx.append(i)
        else:
            retain_idx.append(i)

    return Subset(dataset, forget_idx), Subset(dataset, retain_idx)
