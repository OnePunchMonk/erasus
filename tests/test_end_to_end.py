"""
End-to-end integration test for the Erasus unlearning pipeline.

Tests the full workflow:  Model → Selector → Strategy → Unlearner → Evaluate
using a tiny model and random data to ensure the pipeline works.
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tests.conftest import TinyClassifier, _make_loader


# ---- Helpers ----

def _make_tiny_pipeline_data():
    """Create small forget/retain loaders for pipeline testing."""
    forget = _make_loader(n_samples=32, input_dim=16, num_classes=4, batch_size=8)
    retain = _make_loader(n_samples=64, input_dim=16, num_classes=4, batch_size=8)
    return forget, retain


# ---- Tests ----


class TestErasusUnlearnerEndToEnd:
    """Test the generic ErasusUnlearner with various strategies."""

    def test_fit_gradient_ascent_no_selector(self):
        """Simplest pipeline: gradient_ascent, no coreset selection."""
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner

        model = TinyClassifier()
        unlearner = ErasusUnlearner(
            model=model,
            strategy="gradient_ascent",
            selector=None,
            device="cpu",
        )

        forget, retain = _make_tiny_pipeline_data()
        result = unlearner.fit(forget, retain, prune_ratio=0.5, epochs=2)

        assert result.model is not None
        assert result.elapsed_time > 0
        assert len(result.forget_loss_history) > 0
        assert result.coreset_size == 32  # No selection → full set

    def test_fit_with_selector(self):
        """Pipeline with gradient_norm selector + gradient_ascent."""
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner

        model = TinyClassifier()
        unlearner = ErasusUnlearner(
            model=model,
            strategy="gradient_ascent",
            selector="gradient_norm",
            device="cpu",
        )

        forget, retain = _make_tiny_pipeline_data()
        result = unlearner.fit(forget, retain, prune_ratio=0.5, epochs=2)

        assert result.model is not None
        assert result.coreset_size <= 32
        assert result.compression_ratio <= 1.0

    def test_evaluate_returns_metrics(self):
        """Ensure evaluate() runs and returns a dict."""
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner
        from erasus.metrics.accuracy import AccuracyMetric

        model = TinyClassifier()
        unlearner = ErasusUnlearner(
            model=model,
            strategy="gradient_ascent",
            device="cpu",
        )

        forget, retain = _make_tiny_pipeline_data()
        unlearner.fit(forget, retain, epochs=1)

        metrics = unlearner.evaluate(
            forget_data=forget,
            retain_data=retain,
            metrics=[AccuracyMetric()],
        )

        assert isinstance(metrics, dict)
        assert len(metrics) > 0


class TestModalityUnlearners:
    """Test modality-specific unlearners resolve correctly."""

    def test_multimodal_factory_generic(self):
        """MultimodalUnlearner should dispatch to ErasusUnlearner for unknown models."""
        from erasus.unlearners.multimodal_unlearner import MultimodalUnlearner

        model = TinyClassifier()
        unlearner = MultimodalUnlearner.from_model(
            model, model_type="generic", strategy="gradient_ascent", device="cpu"
        )

        from erasus.unlearners.erasus_unlearner import ErasusUnlearner
        assert isinstance(unlearner, ErasusUnlearner)

    def test_llm_unlearner_creation(self):
        """LLMUnlearner should construct without errors."""
        from erasus.unlearners.llm_unlearner import LLMUnlearner

        model = TinyClassifier()
        unlearner = LLMUnlearner(
            model=model,
            strategy="gradient_ascent",
            selector=None,
            device="cpu",
        )
        assert unlearner.strategy_name == "gradient_ascent"

    def test_vlm_unlearner_creation(self):
        """VLMUnlearner should construct without errors."""
        from erasus.unlearners.vlm_unlearner import VLMUnlearner

        model = TinyClassifier()
        unlearner = VLMUnlearner(
            model=model,
            strategy="gradient_ascent",
            selector=None,
            device="cpu",
        )
        assert unlearner.strategy_name == "gradient_ascent"


class TestMetricSuite:
    """Test the unified metric runner."""

    def test_suite_runs_multiple_metrics(self):
        from erasus.metrics.metric_suite import MetricSuite

        suite = MetricSuite(["accuracy", "mia"])
        assert len(suite.metric_names) == 2

    def test_suite_default_factory(self):
        from erasus.metrics.metric_suite import MetricSuite

        suite = MetricSuite.default_for_modality("llm")
        names = suite.metric_names
        assert len(names) >= 1

    def test_suite_run_produces_results(self):
        from erasus.metrics.metric_suite import MetricSuite

        model = TinyClassifier()
        forget, retain = _make_tiny_pipeline_data()

        suite = MetricSuite(["accuracy"])
        results = suite.run(model, forget, retain)

        assert isinstance(results, dict)
        assert "_meta" in results


class TestConfigAndCLI:
    """Test configuration loading and CLI argument parsing."""

    def test_config_from_yaml(self, tmp_path):
        """ErasusConfig should load from a YAML file."""
        from erasus.core.config import ErasusConfig

        yaml_content = (
            "model_name: test_model\n"
            "model_type: vlm\n"
            "strategy: gradient_ascent\n"
            "epochs: 3\n"
        )
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml_content)

        config = ErasusConfig.from_yaml(str(cfg_path))
        assert config.model_name == "test_model"
        assert config.epochs == 3
        assert config.strategy == "gradient_ascent"

    def test_config_to_dict(self):
        from erasus.core.config import ErasusConfig

        config = ErasusConfig()
        d = config.to_dict()
        assert "model_name" in d
        assert "strategy" in d


class TestRegistryIntegrity:
    """Ensure all expected components are registered."""

    def test_strategy_registry_has_gradient_ascent(self):
        from erasus.core.registry import strategy_registry
        import erasus.strategies  # noqa: F401
        assert "gradient_ascent" in strategy_registry.list()

    def test_selector_registry_has_gradient_norm(self):
        from erasus.core.registry import selector_registry
        import erasus.selectors  # noqa: F401
        assert "gradient_norm" in selector_registry.list()

    def test_metric_registry_has_accuracy(self):
        from erasus.core.registry import metric_registry
        import erasus.metrics  # noqa: F401
        assert "accuracy" in metric_registry.list()

    def test_metric_registry_has_new_metrics(self):
        from erasus.core.registry import metric_registry
        import erasus.metrics  # noqa: F401
        for name in ["mia_full", "lira", "confidence", "feature_distance",
                      "time_complexity", "memory_usage", "dp_evaluation"]:
            assert name in metric_registry.list(), f"Missing: {name}"
