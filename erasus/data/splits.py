"""
Data Splitting Utilities.

Tools to split a dataset into Forget, Retain, and Validation sets.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def random_split(
    dataset: Dataset, 
    lengths: List[int],
    seed: int = 42
) -> List[Subset]:
    """
    Wrapper around torch.utils.data.random_split with explicit seeding.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    lengths : List[int]
        Lengths of splits to be produced.
    seed : int
        Random seed for reproducibility.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, lengths, generator=generator)


def class_balanced_split(
    dataset: Dataset,
    validation_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Splits dataset into Train and Validation sets while maintaining class balance.
    
    Assumes dataset has a `.targets` or `.labels` attribute.
    If not, falls back to random_split with a warning.
    
    Returns
    -------
    (train_subset, val_subset)
    """
    labels = None
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
        
    if labels is None:
        # Fallback
        n = len(dataset)
        val_len = int(n * validation_ratio)
        train_len = n - val_len
        return tuple(random_split(dataset, [train_len, val_len], seed=seed)) # type: ignore

    # Convert to numpy for easier indexing
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    else:
        labels = np.array(labels)
        
    unique_classes = np.unique(labels)
    train_indices = []
    val_indices = []
    
    rng = np.random.default_rng(seed)
    
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        
        n_cls = len(cls_indices)
        n_val = int(n_cls * validation_ratio)
        
        val_indices.extend(cls_indices[:n_val])
        train_indices.extend(cls_indices[n_val:])
        
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices)
    )
