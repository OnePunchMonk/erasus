"""
Checkpoint save/load for resumable unlearning runs.

Serialises model weights plus loss histories and epoch counters so
``BaseUnlearner.fit(resume_from=...)`` can continue training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def _paths(base: Path) -> tuple[Path, Path]:
    return base / "unlearning_checkpoint.pt", base / "unlearning_checkpoint.json"


def save_unlearning_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    forget_losses: List[float],
    retain_losses: List[float],
    epochs_completed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model state and metadata for resume."""
    base = Path(path)
    if base.suffix in (".pt", ".pth"):
        base = base.parent
    base.mkdir(parents=True, exist_ok=True)
    pt_path, json_path = _paths(base)
    torch.save(model.state_dict(), pt_path)
    payload = {
        "epochs_completed": epochs_completed,
        "forget_losses": forget_losses,
        "retain_losses": retain_losses,
        **(extra or {}),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_unlearning_checkpoint(
    path: str | Path,
    model: nn.Module,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load weights into ``model`` and return metadata (loss histories, epochs_completed).

    Parameters
    ----------
    path
        Directory containing ``unlearning_checkpoint.pt`` and ``.json``, or a ``.pt`` file
        (metadata JSON must sit beside it with the same stem).
    """
    p = Path(path)
    if p.is_file() and p.suffix in (".pt", ".pth"):
        pt_path = p
        json_path = p.with_suffix(".json")
        base = p.parent
    else:
        base = p
        pt_path, json_path = _paths(base)

    if not pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    state = torch.load(pt_path, map_location=map_location or "cpu")
    model.load_state_dict(state)

    meta: Dict[str, Any] = {
        "epochs_completed": 0,
        "forget_losses": [],
        "retain_losses": [],
    }
    if json_path.exists():
        meta.update(json.loads(json_path.read_text(encoding="utf-8")))
    return meta
