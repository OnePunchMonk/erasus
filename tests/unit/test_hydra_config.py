"""
Tests for Hydra-style config composition.
"""

from __future__ import annotations

from pathlib import Path

from erasus.experiments.hydra_config import compose_experiment_config


def test_compose_default_config():
    config = compose_experiment_config()
    assert config.strategy.name == "gradient_ascent"
    assert config.selector.name == "random"


def test_compose_with_overrides():
    config = compose_experiment_config(
        overrides=[
            "strategy.name=npo",
            "strategy.epochs=2",
            "selector.prune_ratio=0.25",
        ]
    )
    assert config.strategy.name == "npo"
    assert config.strategy.epochs == 2
    assert config.selector.prune_ratio == 0.25


def test_compose_from_yaml(tmp_path: Path):
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        "model:\n"
        "  name: gpt2\n"
        "strategy:\n"
        "  name: altpo\n"
        "  epochs: 3\n"
    )

    config = compose_experiment_config(str(config_path))
    assert config.model.name == "gpt2"
    assert config.strategy.name == "altpo"
    assert config.strategy.epochs == 3
