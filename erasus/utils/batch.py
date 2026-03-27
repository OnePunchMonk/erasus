"""
Batch unpacking utilities.

Centralises the logic for extracting inputs, labels, and optional extras
from DataLoader batches, which appear in at least three different formats
across the codebase.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def unpack_batch(
    batch: Any,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract ``(inputs, labels)`` from a DataLoader batch.

    Supports three common formats:

    1. ``(inputs, labels, ...)`` — tuple/list with >= 2 elements
    2. ``(inputs,)`` — single-element tuple (labels=None)
    3. ``dict`` with ``"input_ids"``/``"inputs"`` and optional ``"labels"``

    Parameters
    ----------
    batch : Any
        A single batch from a DataLoader.
    device : torch.device, optional
        If given, tensors are moved to this device.

    Returns
    -------
    tuple[Tensor, Tensor | None]
        ``(inputs, labels)`` where labels may be None.
    """
    if isinstance(batch, dict):
        inputs = batch.get("input_ids", batch.get("inputs"))
        if inputs is None:
            raise ValueError(
                "Dict batch must contain 'input_ids' or 'inputs' key. "
                f"Got keys: {list(batch.keys())}"
            )
        labels = batch.get("labels")
    elif isinstance(batch, (list, tuple)):
        inputs = batch[0]
        labels = batch[1] if len(batch) > 1 else None
    else:
        inputs = batch
        labels = None

    if device is not None:
        inputs = inputs.to(device)
        if labels is not None:
            labels = labels.to(device)

    return inputs, labels
