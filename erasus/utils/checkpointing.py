"""
Checkpointing utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model weights and optional metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(p))
    if metadata:
        meta_path = p.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = True,
) -> nn.Module:
    """Load model weights from checkpoint."""
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=strict)
    return model
