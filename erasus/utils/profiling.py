"""
erasus.utils.profiling — Performance profiling for unlearning pipelines.

Provides timing, memory tracking, and computational cost estimation
for unlearning operations.
"""

from __future__ import annotations

import time
import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class UnlearningProfiler:
    """
    Profiler for machine unlearning operations.

    Tracks wall-clock time, GPU memory usage, parameter counts, and
    estimated FLOPs across unlearning steps.

    Usage
    -----
    ::

        profiler = UnlearningProfiler()

        with profiler.profile("gradient_ascent_step"):
            loss = model(x)
            loss.backward()

        profiler.log_memory("after_gradient_ascent")
        report = profiler.get_report()
    """

    def __init__(self, enable_cuda: bool = True):
        self.enable_cuda = enable_cuda and torch.cuda.is_available()
        self._timings: Dict[str, List[float]] = {}
        self._memory_snapshots: Dict[str, Dict[str, float]] = {}
        self._counters: Dict[str, int] = {}
        self._flops: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @contextmanager
    def profile(self, name: str):
        """
        Context manager that records wall-clock time for a named section.

        Example
        -------
        ::

            with profiler.profile("forward_pass"):
                output = model(x)
        """
        if self.enable_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            yield
        finally:
            if self.enable_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if name not in self._timings:
                self._timings[name] = []
            self._timings[name].append(elapsed)

    def time_function(self, name: str):
        """
        Decorator to profile a function's execution time.

        Example
        -------
        ::

            @profiler.time_function("train_step")
            def train_step(model, data):
                ...
        """
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.profile(name):
                    return fn(*args, **kwargs)
            return wrapper
        return decorator

    # ------------------------------------------------------------------
    # Memory tracking
    # ------------------------------------------------------------------

    def log_memory(self, label: str) -> Dict[str, float]:
        """
        Log current GPU memory usage.

        Parameters
        ----------
        label : str
            Label for this memory snapshot.

        Returns
        -------
        dict with allocated_mb, reserved_mb, max_allocated_mb
        """
        snapshot: Dict[str, float] = {}

        if self.enable_cuda:
            snapshot["allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            snapshot["reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
            snapshot["max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                snapshot["rss_mb"] = mem_info.rss / (1024 ** 2)
                snapshot["vms_mb"] = mem_info.vms / (1024 ** 2)
            except ImportError:
                snapshot["rss_mb"] = 0.0

        self._memory_snapshots[label] = snapshot
        return snapshot

    def reset_peak_memory(self) -> None:
        """Reset CUDA peak memory tracking."""
        if self.enable_cuda:
            torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # Parameter / FLOP estimation
    # ------------------------------------------------------------------

    def count_parameters(
        self,
        model: nn.Module,
        label: str = "model",
    ) -> Dict[str, int]:
        """
        Count model parameters.

        Returns
        -------
        dict with total, trainable, frozen counts.
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable

        result = {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }
        self._counters[label] = total
        return result

    def estimate_flops(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        label: str = "forward_pass",
    ) -> float:
        """
        Estimate FLOPs for a single forward pass using simple heuristics.

        Parameters
        ----------
        model : nn.Module
            Model to estimate.
        input_shape : tuple
            Input tensor shape (without batch dim).
        label : str
            Label for recording.

        Returns
        -------
        float
            Estimated FLOPs.
        """
        total_flops = 0.0

        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_flops += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                # FLOPs = 2 * K_h * K_w * C_in * C_out * H_out * W_out
                k = module.kernel_size
                c_in = module.in_channels // module.groups
                out_features = module.out_channels
                # Estimate output size (approximate)
                total_flops += 2 * k[0] * k[1] * c_in * out_features * 56 * 56  # approximation
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                if hasattr(module, "normalized_shape"):
                    total_flops += 5 * (
                        module.normalized_shape[0]
                        if isinstance(module.normalized_shape, (list, tuple))
                        else module.normalized_shape
                    )

        self._flops[label] = total_flops
        return total_flops

    # ------------------------------------------------------------------
    # Increment counters
    # ------------------------------------------------------------------

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        self._counters[name] = self._counters.get(name, 0) + value

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive profiling report.

        Returns
        -------
        dict with sections: timings, memory, parameters, flops, counters
        """
        report: Dict[str, Any] = {}

        # Timing summary
        timing_summary = {}
        for name, times in self._timings.items():
            timing_summary[name] = {
                "total_s": sum(times),
                "mean_s": sum(times) / len(times),
                "min_s": min(times),
                "max_s": max(times),
                "count": len(times),
            }
        report["timings"] = timing_summary

        # Memory snapshots
        report["memory"] = dict(self._memory_snapshots)

        # Counters
        report["counters"] = dict(self._counters)

        # FLOPs
        report["flops"] = dict(self._flops)

        return report

    def summary(self) -> str:
        """Return a human-readable profiling summary."""
        lines = ["=" * 60, "  Unlearning Profiler Report", "=" * 60]

        if self._timings:
            lines.append("\n⏱  Timings:")
            for name, times in self._timings.items():
                total = sum(times)
                mean = total / len(times)
                lines.append(f"  {name}: {total:.4f}s total, {mean:.4f}s mean ({len(times)} calls)")

        if self._memory_snapshots:
            lines.append("\n💾 Memory Snapshots:")
            for label, snap in self._memory_snapshots.items():
                parts = [f"{k}={v:.1f}" for k, v in snap.items()]
                lines.append(f"  {label}: {', '.join(parts)}")

        if self._counters:
            lines.append("\n📊 Counters:")
            for name, val in self._counters.items():
                lines.append(f"  {name}: {val:,}")

        if self._flops:
            lines.append("\n🔢 Estimated FLOPs:")
            for label, flops in self._flops.items():
                if flops > 1e9:
                    lines.append(f"  {label}: {flops / 1e9:.2f} GFLOPs")
                elif flops > 1e6:
                    lines.append(f"  {label}: {flops / 1e6:.2f} MFLOPs")
                else:
                    lines.append(f"  {label}: {flops:.0f} FLOPs")

        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all profiling data."""
        self._timings.clear()
        self._memory_snapshots.clear()
        self._counters.clear()
        self._flops.clear()


# ======================================================================
# Convenience
# ======================================================================


@contextmanager
def profile_section(name: str, verbose: bool = True):
    """
    Standalone context manager for quick timing.

    Example
    -------
    ::

        with profile_section("my_operation"):
            do_something()
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if verbose:
            print(f"[Profile] {name}: {elapsed:.4f}s")


def profile_model_memory(model: nn.Module) -> Dict[str, float]:
    """Quick snapshot of model memory footprint."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    grad_bytes = sum(
        p.grad.nelement() * p.grad.element_size()
        for p in model.parameters()
        if p.grad is not None
    )

    return {
        "param_mb": param_bytes / (1024 ** 2),
        "buffer_mb": buffer_bytes / (1024 ** 2),
        "grad_mb": grad_bytes / (1024 ** 2),
        "total_mb": (param_bytes + buffer_bytes + grad_bytes) / (1024 ** 2),
    }
