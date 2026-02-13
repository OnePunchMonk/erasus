"""
erasus.metrics.efficiency.memory_usage — Memory tracking metrics.

Monitors peak GPU/CPU memory usage during model evaluation.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class MemoryUsageMetric(BaseMetric):
    """
    Profiles memory usage during a forward pass.

    Reports:
    - Peak GPU memory allocated (if CUDA available)
    - Current GPU memory allocated
    - Model size in MB
    - CPU RSS (resident set size) if psutil is available
    """

    name = "memory_usage"

    def __init__(self, max_batches: int = 5):
        self.max_batches = max_batches

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        results: Dict[str, float] = {}

        # Model size
        param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
        results["model_size_mb"] = (param_bytes + buffer_bytes) / (1024 ** 2)

        # GPU memory
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            mem_before = torch.cuda.memory_allocated(device)

            # Run a few forward passes
            with torch.no_grad():
                batch_iter = iter(forget_data)
                for _ in range(self.max_batches):
                    try:
                        batch = next(batch_iter)
                        inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                        model(inputs)
                    except StopIteration:
                        break

            torch.cuda.synchronize(device)
            peak_mem = torch.cuda.max_memory_allocated(device)
            current_mem = torch.cuda.memory_allocated(device)

            results["gpu_peak_memory_mb"] = peak_mem / (1024 ** 2)
            results["gpu_current_memory_mb"] = current_mem / (1024 ** 2)
            results["gpu_memory_delta_mb"] = (peak_mem - mem_before) / (1024 ** 2)

            # GPU utilization (if available)
            try:
                gpu_props = torch.cuda.get_device_properties(device)
                results["gpu_total_memory_mb"] = gpu_props.total_mem / (1024 ** 2)
                results["gpu_utilization_pct"] = (peak_mem / gpu_props.total_mem) * 100
            except Exception:
                pass

        # CPU memory
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            results["cpu_rss_mb"] = mem_info.rss / (1024 ** 2)
            results["cpu_vms_mb"] = mem_info.vms / (1024 ** 2)
        except ImportError:
            pass

        return results
