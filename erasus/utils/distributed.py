"""
erasus.utils.distributed — Distributed training utilities.

Provides helpers for multi-GPU / multi-node unlearning.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn


def is_distributed() -> bool:
    """Check if distributed training is initialised."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    """Return the global rank (0 if not distributed)."""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Return total number of processes (1 if not distributed)."""
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    """True on the rank-0 process only."""
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> None:
    """
    Initialise from environment variables (``RANK``, ``WORLD_SIZE``,
    ``MASTER_ADDR``, ``MASTER_PORT``) that are typically set by
    ``torchrun`` / ``torch.distributed.launch``.
    """
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )
            if torch.cuda.is_available():
                torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup_distributed() -> None:
    """Shut down the distributed process group."""
    if is_distributed():
        torch.distributed.destroy_process_group()


def wrap_model_ddp(
    model: nn.Module,
    device_ids: Optional[list] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Wrap a model with DistributedDataParallel if distributed.

    Falls back to DataParallel when multiple GPUs are available
    but distributed is not initialised.
    """
    if is_distributed():
        if device_ids is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device_ids = [local_rank]
        return nn.parallel.DistributedDataParallel(
            model, device_ids=device_ids, **kwargs
        )
    elif torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all distributed workers."""
    if is_distributed():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= get_world_size()
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable object from ``src`` to all workers."""
    if not is_distributed():
        return obj
    obj_list = [obj]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]
