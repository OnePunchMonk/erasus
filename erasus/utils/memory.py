"""
Adaptive memory management utilities.

Provides tools for auto-tuning batch sizes and chunked computation
to prevent OOM errors during memory-intensive operations like influence
function computation and Fisher information estimation.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, TypeVar

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_free_gpu_memory() -> int:
    """Return free GPU memory in bytes.  Returns 0 on CPU."""
    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info()
    return free


def auto_batch_size(
    model: nn.Module,
    sample_input: torch.Tensor,
    max_batch_size: int = 256,
    memory_fraction: float = 0.7,
) -> int:
    """
    Estimate the largest safe batch size for the given model and input.

    Uses a binary-search approach: tries progressively larger batches
    until an OOM occurs, then backs off.

    Parameters
    ----------
    model : nn.Module
    sample_input : torch.Tensor
        A single sample (no batch dim) used to estimate memory per sample.
    max_batch_size : int
        Upper bound.
    memory_fraction : float
        Fraction of free GPU memory to target (default 0.7).

    Returns
    -------
    int
        Recommended batch size (minimum 1).
    """
    free = get_free_gpu_memory()
    if free == 0:
        return max_batch_size  # CPU — memory is less constrained

    device = next(model.parameters()).device
    model.eval()

    # Estimate per-sample memory from a small probe
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)

    try:
        probe_bs = min(4, max_batch_size)
        probe_input = sample_input.unsqueeze(0).expand(probe_bs, *sample_input.shape).to(device)
        with torch.no_grad():
            _ = model(probe_input)
        peak = torch.cuda.max_memory_allocated(device)
        per_sample = (peak - base_mem) / probe_bs
    except RuntimeError:
        return 1

    if per_sample <= 0:
        return max_batch_size

    target_mem = free * memory_fraction
    estimated_bs = int(target_mem / per_sample)
    return max(1, min(estimated_bs, max_batch_size))


def chunked_computation(
    fn: Callable[[torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    chunk_size: Optional[int] = None,
    memory_fraction: float = 0.5,
) -> torch.Tensor:
    """
    Apply ``fn`` to ``data`` in chunks sized to fit in available GPU memory.

    Useful for influence computation, Fisher diagonal estimation, or any
    operation that would OOM if run on the full tensor at once.

    Parameters
    ----------
    fn : callable
        Function mapping (N, ...) -> (N, ...) tensor.
    data : torch.Tensor
        Input tensor with batch dimension first.
    chunk_size : int, optional
        Explicit chunk size.  If None, auto-determined from free VRAM.
    memory_fraction : float
        Fraction of free GPU memory to target per chunk.

    Returns
    -------
    torch.Tensor
        Concatenated results.
    """
    n = data.size(0)

    if chunk_size is None:
        free = get_free_gpu_memory()
        if free > 0:
            # Estimate bytes per sample from data tensor
            bytes_per_sample = data[0:1].element_size() * data[0:1].nelement() * 4  # ~4x for grads
            target = int(free * memory_fraction)
            chunk_size = max(1, target // max(bytes_per_sample, 1))
        else:
            chunk_size = n  # CPU — process at once

    chunk_size = min(chunk_size, n)
    results = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        results.append(fn(data[start:end]))

    return torch.cat(results, dim=0)
