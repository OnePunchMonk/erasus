"""
UnlearningDataset and ForgetRetainDataset — General-purpose dataset
wrappers for machine unlearning.

Provides the ``UnlearningDataset`` abstraction that supports sample-level and
class-level forget specifications, weighted sampling, and streaming deletion.

Example
-------
>>> from erasus.data.datasets import UnlearningDataset
>>>
>>> # Sample-level forget specification
>>> ds = UnlearningDataset(base_dataset, forget_indices=[0, 5, 12, 99])
>>> forget_loader, retain_loader = ds.to_loaders(batch_size=32)
>>>
>>> # Class-level forget specification
>>> ds = UnlearningDataset(base_dataset, forget_classes=[2, 7])
>>>
>>> # Weighted sampling (higher weight = more frequent in forget loader)
>>> ds = UnlearningDataset(base_dataset, forget_indices=[0, 5], forget_weights={0: 2.0, 5: 1.5})
>>>
>>> # Streaming deletion
>>> ds.mark_forget([42, 88])           # add samples to forget set
>>> ds.mark_retain([0])                # move sample back to retain set
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


class UnlearningDataset(Dataset):
    """
    General-purpose dataset wrapper for machine unlearning.

    Wraps any base ``Dataset`` and partitions it into forget and retain
    subsets. Supports both sample-level and class-level forget
    specifications, per-sample importance weights, and dynamic
    addition/removal of samples from the forget set.

    Parameters
    ----------
    dataset : Dataset
        The underlying base dataset.
    forget_indices : list[int], optional
        Explicit sample indices to forget.
    forget_classes : list[int], optional
        Class labels to forget (all samples with these labels).
        Requires the dataset to yield ``(sample, label)`` tuples.
    forget_weights : dict[int, float], optional
        Per-sample importance weights for the forget set.
        Keys are dataset indices, values are weights (default 1.0).
    label_fn : callable, optional
        Function that extracts the label from a dataset sample.
        Default assumes ``dataset[i]`` returns ``(x, label, ...)``.
    """

    def __init__(
        self,
        dataset: Dataset,
        forget_indices: Optional[Sequence[int]] = None,
        forget_classes: Optional[Sequence[int]] = None,
        forget_weights: Optional[Dict[int, float]] = None,
        label_fn: Optional[Any] = None,
    ) -> None:
        self._dataset = dataset
        self._forget_set: Set[int] = set()
        self._weights: Dict[int, float] = dict(forget_weights or {})
        self._label_fn = label_fn or self._default_label_fn

        # Resolve forget specification
        if forget_indices is not None:
            self._forget_set.update(forget_indices)

        if forget_classes is not None:
            self._resolve_class_forget(forget_classes)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        return self._dataset[index]

    # ------------------------------------------------------------------
    # Forget/Retain partitioning
    # ------------------------------------------------------------------

    @property
    def forget_indices(self) -> List[int]:
        """Sorted list of forget-set indices."""
        return sorted(self._forget_set)

    @property
    def retain_indices(self) -> List[int]:
        """Sorted list of retain-set indices."""
        all_indices = set(range(len(self._dataset)))
        return sorted(all_indices - self._forget_set)

    @property
    def forget_size(self) -> int:
        return len(self._forget_set)

    @property
    def retain_size(self) -> int:
        return len(self._dataset) - len(self._forget_set)

    @property
    def forget_ratio(self) -> float:
        """Fraction of the dataset marked for forgetting."""
        if len(self._dataset) == 0:
            return 0.0
        return self.forget_size / len(self._dataset)

    def is_forget(self, index: int) -> bool:
        """Check whether a sample is in the forget set."""
        return index in self._forget_set

    # ------------------------------------------------------------------
    # Streaming deletion API
    # ------------------------------------------------------------------

    def mark_forget(self, indices: Sequence[int]) -> None:
        """Add samples to the forget set."""
        self._forget_set.update(indices)

    def mark_retain(self, indices: Sequence[int]) -> None:
        """Move samples from the forget set back to the retain set."""
        self._forget_set -= set(indices)

    def mark_forget_classes(self, classes: Sequence[int]) -> None:
        """Add all samples with the given class labels to the forget set."""
        self._resolve_class_forget(classes)

    def set_weight(self, index: int, weight: float) -> None:
        """Set the importance weight for a specific sample."""
        self._weights[index] = weight

    def set_weights(self, weights: Dict[int, float]) -> None:
        """Bulk-set importance weights."""
        self._weights.update(weights)

    # ------------------------------------------------------------------
    # Subset and loader creation
    # ------------------------------------------------------------------

    @property
    def forget_subset(self) -> Subset:
        """A ``Subset`` containing only the forget samples."""
        return Subset(self._dataset, self.forget_indices)

    @property
    def retain_subset(self) -> Subset:
        """A ``Subset`` containing only the retain samples."""
        return Subset(self._dataset, self.retain_indices)

    def to_loaders(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        weighted: bool = False,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create forget and retain DataLoaders.

        Parameters
        ----------
        batch_size : int
            Batch size for both loaders.
        shuffle : bool
            Whether to shuffle the data.
        weighted : bool
            If True, use ``forget_weights`` for weighted random sampling
            in the forget loader.
        num_workers : int
            Number of data loading workers.

        Returns
        -------
        (forget_loader, retain_loader)
        """
        forget_idx = self.forget_indices
        retain_idx = self.retain_indices

        # Forget loader
        forget_ds = Subset(self._dataset, forget_idx)
        if weighted and self._weights:
            sample_weights = [self._weights.get(i, 1.0) for i in forget_idx]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(forget_idx),
                replacement=True,
            )
            forget_loader = DataLoader(
                forget_ds, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, **kwargs,
            )
        else:
            forget_loader = DataLoader(
                forget_ds, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, **kwargs,
            )

        # Retain loader
        retain_ds = Subset(self._dataset, retain_idx)
        retain_loader = DataLoader(
            retain_ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, **kwargs,
        )

        return forget_loader, retain_loader

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_class_forget(self, classes: Sequence[int]) -> None:
        """Scan the dataset and mark all samples whose label is in *classes*."""
        target_classes = set(classes)
        for i in range(len(self._dataset)):
            label = self._label_fn(self._dataset[i])
            if label in target_classes:
                self._forget_set.add(i)

    @staticmethod
    def _default_label_fn(sample: Any) -> Any:
        """Extract label assuming (x, label, ...) tuple format."""
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            label = sample[1]
            if isinstance(label, torch.Tensor):
                return label.item()
            return label
        return None

    def __repr__(self) -> str:
        return (
            f"UnlearningDataset(total={len(self)}, "
            f"forget={self.forget_size}, retain={self.retain_size}, "
            f"forget_ratio={self.forget_ratio:.2%})"
        )


class ForgetRetainDataset(Dataset):
    """
    Combines Forget and Retain datasets into one.

    Each item yields ``(sample, partition_label)`` where
    ``partition_label`` is 1 for forget and 0 for retain.
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
            return self.forget_set[index], 1  # 1 = Forget
        else:
            return self.retain_set[index - self.f_len], 0  # 0 = Retain
