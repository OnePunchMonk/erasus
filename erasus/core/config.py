"""
Erasus Configuration Module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class ErasusConfig:
    """Unified configuration for an Erasus unlearning run."""

    # Model
    model_name: str = "openai/clip-vit-base-patch32"
    model_type: str = "vlm"  # vlm | llm | diffusion | audio | video
    device: str = "cuda"

    # Selector
    selector: str = "random"
    prune_ratio: float = 0.1
    selector_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Strategy
    strategy: str = "gradient_ascent"
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Training
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-4

    # Logging
    log_dir: Optional[str] = None
    wandb_project: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "ErasusConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
