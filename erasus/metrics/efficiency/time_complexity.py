"""
erasus.metrics.efficiency.time_complexity — Wall-clock timing metrics.

Tracks the computational cost of unlearning, including wall-clock
time, per-epoch timings, and estimated FLOPs (if available).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class TimeComplexityMetric(BaseMetric):
    """
    Measures inference throughput and timing characteristics.

    Reports wall-clock time for a single forward pass over the
    forget and retain data, plus estimated samples-per-second.
    """

    name = "time_complexity"

    def __init__(self, warmup_batches: int = 2, eval_batches: int = 10):
        self.warmup_batches = warmup_batches
        self.eval_batches = eval_batches

    def compute(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        model.eval()

        forget_time, forget_samples = self._time_forward(model, forget_data, device)
        retain_time, retain_samples = self._time_forward(model, retain_data, device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results: Dict[str, float] = {
            "time_forget_seconds": forget_time,
            "time_retain_seconds": retain_time,
            "throughput_forget_sps": forget_samples / max(forget_time, 1e-6),
            "throughput_retain_sps": retain_samples / max(retain_time, 1e-6),
            "total_parameters": float(total_params),
            "trainable_parameters": float(trainable_params),
        }

        # Estimate FLOPs for a single forward pass (rough heuristic)
        flops = self._estimate_flops(model, forget_data, device)
        if flops > 0:
            results["estimated_flops_per_sample"] = flops

        return results

    def _time_forward(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> tuple:
        """Time forward passes, return (elapsed_seconds, num_samples)."""
        # Warmup
        batch_iter = iter(loader)
        for _ in range(self.warmup_batches):
            try:
                batch = next(batch_iter)
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                with torch.no_grad():
                    model(inputs)
            except StopIteration:
                break

        # Timed run
        total_samples = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        batch_iter = iter(loader)
        for _ in range(self.eval_batches):
            try:
                batch = next(batch_iter)
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                with torch.no_grad():
                    model(inputs)
                total_samples += inputs.shape[0]
            except StopIteration:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return elapsed, total_samples

    @staticmethod
    def _estimate_flops(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> float:
        """Rough FLOPs estimate using parameter count × input elements."""
        try:
            batch = next(iter(loader))
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            input_elements = inputs[0:1].numel()
            param_count = sum(p.numel() for p in model.parameters())
            # Very rough: 2 * params * input_elements (multiply-adds)
            return float(2 * param_count)
        except Exception:
            return 0.0
