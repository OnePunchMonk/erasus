"""
erasus.utils.helpers — General helper functions.

Small utilities used across multiple modules in the Erasus framework.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


# --- Model helpers ---

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Return the model size in megabytes."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def freeze_model(model: nn.Module) -> None:
    """Freeze all parameters."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True


def freeze_except(model: nn.Module, layer_names: List[str]) -> None:
    """Freeze all parameters except those matching any of the given names."""
    freeze_model(model)
    for name, p in model.named_parameters():
        if any(ln in name for ln in layer_names):
            p.requires_grad = True


# --- Tensor helpers ---

def to_device(data: Any, device: Union[str, torch.device]) -> Any:
    """Recursively move data to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    return data


def gradient_norm(model: nn.Module) -> float:
    """Compute the total gradient L2 norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# --- Timing helpers ---

class Timer:
    """
    Simple context-manager timer.

    Usage::

        with Timer("my_op") as t:
            ...
        print(t.elapsed)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.3f}s")


# --- IO helpers ---

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hash_config(config: Dict[str, Any]) -> str:
    """Create a deterministic hash of a config dict."""
    js = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(js.encode()).hexdigest()[:12]


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save data as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
