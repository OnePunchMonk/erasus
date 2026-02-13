"""
Dataset Wrappers.

Provides unified interfaces for Forget and Retain sets.
"""

from __future__ import annotations

from typing import Any, Tuple, Optional

import torch
from torch.utils.data import Dataset


class UnlearningDataset(Dataset):
    """
    Base wrapper for any dataset used in unlearning.
    Can add 'forget' flag or other metadata.
    """

    def __init__(self, dataset: Dataset, is_forget: bool = True) -> None:
        self.dataset = dataset
        self.is_forget = is_forget

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        try:
            sample = self.dataset[index]
            # If sample is tuple (x, y), we might want to append 'is_forget' flag?
            # For now, just return sample.
            return sample
        except Exception as e:
            # Handle potential multimodal errors
            print(f"Error loading sample {index}: {e}")
            return None


class ForgetRetainDataset(Dataset):
    """
    Combines Forget and Retain datasets into one, yielding batches from both?
    Or just a concat wrapper.
    Usually we want separate loaders, but some strategies might want mixed pairs.
    """

    def __init__(self, forget_set: Dataset, retain_set: Dataset) -> None:
        self.forget_set = forget_set
        self.retain_set = retain_set
        self.f_len = len(forget_set)
        self.r_len = len(retain_set)

    def __len__(self) -> int:
        return self.f_len + self.r_len

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        if index < self.f_len:
            return self.forget_set[index], 1 # 1 = Forget
        else:
            return self.retain_set[index - self.f_len], 0 # 0 = Retain
