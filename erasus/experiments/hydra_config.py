"""
Hydra-style configuration composition for Erasus experiments.

This module provides a lightweight composition layer that works with
``OmegaConf`` when available and still supports dotted CLI overrides in
minimal environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import yaml

from erasus.core.config import ErasusConfig

try:
    from omegaconf import OmegaConf

    OMEGACONF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    OmegaConf = None
    OMEGACONF_AVAILABLE = False


@dataclass
class ModelConfig:
    name: str = "openai/clip-vit-base-patch32"
    model_type: str = "vlm"
    device: str = "cuda"


@dataclass
class DataConfig:
    forget_data_dir: Optional[str] = None
    retain_data_dir: Optional[str] = None
    batch_size: int = 32


@dataclass
class StrategyConfig:
    name: str = "gradient_ascent"
    lr: float = 1e-4
    epochs: int = 5
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectorConfig:
    name: str = "random"
    prune_ratio: float = 0.1
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingConfig:
    log_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    backend: str = "local"
    project: str = "erasus"


@dataclass
class ExperimentConfig:
    experiment_name: str = "unlearning_exp"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    def to_erasus_config(self) -> ErasusConfig:
        """Project a composed experiment config into the runtime config."""
        proj = self.tracking.project or self.tracking.wandb_project or "erasus"
        return ErasusConfig(
            experiment_name=self.experiment_name,
            model_name=self.model.name,
            model_type=self.model.model_type,
            device=self.model.device,
            selector=self.selector.name,
            prune_ratio=self.selector.prune_ratio,
            selector_kwargs=self.selector.extra_params,
            strategy=self.strategy.name,
            strategy_kwargs={
                "lr": self.strategy.lr,
                **self.strategy.extra_params,
            },
            epochs=self.strategy.epochs,
            batch_size=self.data.batch_size,
            lr=self.strategy.lr,
            log_dir=self.tracking.log_dir,
            wandb_project=self.tracking.wandb_project,
            forget_data_dir=self.data.forget_data_dir,
            retain_data_dir=self.data.retain_data_dir,
            metrics=list(self.metrics),
            tracking_backend=self.tracking.backend,
            tracking_project=proj,
        )


class HydraConfigManager:
    """Compose experiment configs from YAML + dotted overrides."""

    @staticmethod
    def _normalize_flat_legacy(data: Dict[str, Any]) -> None:
        """Merge flat keys (``model_name``, ``strategy``, …) into nested sections."""
        # configs/default.yaml uses ``strategy: "name"`` which would overwrite the nested dict.
        if isinstance(data.get("strategy"), str):
            data["strategy"] = {
                "name": data["strategy"],
                "lr": data.get("lr", 1e-4),
                "epochs": data.get("epochs", 5),
                "extra_params": data.get("strategy_kwargs", {}) or {},
            }
        if isinstance(data.get("selector"), str):
            data["selector"] = {
                "name": data["selector"],
                "prune_ratio": data.get("prune_ratio", 0.1),
                "extra_params": data.get("selector_kwargs", {}) or {},
            }

        model = cast(Dict[str, Any], data.setdefault("model", {}))
        if "model_name" in data:
            model.setdefault("name", data["model_name"])
        if "model_type" in data:
            model.setdefault("model_type", data["model_type"])
        if "device" in data:
            model.setdefault("device", data["device"])

        strat = cast(Dict[str, Any], data.setdefault("strategy", {}))
        if "epochs" in data:
            strat.setdefault("epochs", data["epochs"])
        if "lr" in data:
            strat.setdefault("lr", data["lr"])
        sk = data.get("strategy_kwargs")
        if isinstance(sk, dict):
            ep = cast(Dict[str, Any], strat.setdefault("extra_params", {}))
            ep.update(sk)

        sel = cast(Dict[str, Any], data.setdefault("selector", {}))
        if "prune_ratio" in data:
            sel.setdefault("prune_ratio", data["prune_ratio"])
        skw = data.get("selector_kwargs")
        if isinstance(skw, dict):
            ep2 = cast(Dict[str, Any], sel.setdefault("extra_params", {}))
            ep2.update(skw)

        ddata = cast(Dict[str, Any], data.setdefault("data", {}))
        if "forget_data_dir" in data:
            ddata.setdefault("forget_data_dir", data["forget_data_dir"])
        if "retain_data_dir" in data:
            ddata.setdefault("retain_data_dir", data["retain_data_dir"])
        if "batch_size" in data:
            ddata.setdefault("batch_size", data["batch_size"])

        tr = cast(Dict[str, Any], data.setdefault("tracking", {}))
        if "log_dir" in data:
            tr.setdefault("log_dir", data["log_dir"])
        if "wandb_project" in data:
            tr.setdefault("wandb_project", data["wandb_project"])

    @staticmethod
    def _apply_override(data: Dict[str, Any], override: str) -> None:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")

        key, raw_value = override.split("=", 1)
        path = key.split(".")
        value = yaml.safe_load(raw_value)

        cursor = data
        for part in path[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[path[-1]] = value

    @classmethod
    def compose(
        cls,
        config_path: Optional[str] = None,
        overrides: Optional[List[str]] = None,
    ) -> ExperimentConfig:
        """Compose an ``ExperimentConfig`` from an optional YAML file."""
        data = asdict(ExperimentConfig())

        if config_path is not None:
            loaded = yaml.safe_load(Path(config_path).read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Experiment config must deserialize to a mapping.")
            data.update(loaded)
            cls._normalize_flat_legacy(data)

        for override in overrides or []:
            cls._apply_override(data, override)

        if OMEGACONF_AVAILABLE:
            structured = OmegaConf.structured(ExperimentConfig())
            merged = OmegaConf.merge(structured, data)
            resolved = OmegaConf.to_container(merged, resolve=True)
            data = resolved if isinstance(resolved, dict) else data

        return cls._from_mapping(data)

    @staticmethod
    def _from_mapping(data: Dict[str, Any]) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_name=data.get("experiment_name", "unlearning_exp"),
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            strategy=StrategyConfig(**data.get("strategy", {})),
            selector=SelectorConfig(**data.get("selector", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
            metrics=list(data.get("metrics", ["accuracy"])),
        )


def compose_experiment_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> ExperimentConfig:
    """Convenience wrapper mirroring Hydra-style composition."""
    return HydraConfigManager.compose(config_path=config_path, overrides=overrides)
