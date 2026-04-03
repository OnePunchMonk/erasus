"""
PyTorch helpers shared across metrics and selectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def infer_module_device(module: nn.Module, fallback: str = "cpu") -> torch.device:
    """
    Return ``next(module.parameters()).device`` when parameters exist,
    otherwise ``torch.device(fallback)`` (e.g. stateless or buffer-only modules).
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device(fallback)
