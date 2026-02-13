"""
End-to-End Integration Tests for the Erasus Framework.

Tests the full pipeline:  Model → Selector → Strategy → Metrics
using a tiny synthetic model + dataset (no downloads needed).
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------------
# Fake tiny model that behaves like a classifier
# ------------------------------------------------------------------
class TinyClassifier(nn.Module):
    """Two-layer classifier for testing."""

    def __init__(self, in_features: int = 8, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def tiny_model():
    return TinyClassifier(in_features=8, num_classes=3)


@pytest.fixture
def forget_loader():
    X = torch.randn(40, 8)
    y = torch.randint(0, 3, (40,))
    return DataLoader(TensorDataset(X, y), batch_size=10, shuffle=False)


@pytest.fixture
def retain_loader():
    X = torch.randn(80, 8)
    y = torch.randint(0, 3, (80,))
    return DataLoader(TensorDataset(X, y), batch_size=10, shuffle=False)


# ==================================================================
# Tests
# ==================================================================

class TestErasusUnlearner:
    """Test the main ErasusUnlearner API."""

    def test_fit_no_selector(self, tiny_model, forget_loader, retain_loader):
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner

        unlearner = ErasusUnlearner(
            model=tiny_model,
            strategy="gradient_ascent",
            selector=None,
            device="cpu",
        )
        result = unlearner.fit(
            forget_data=forget_loader,
            retain_data=retain_loader,
            epochs=2,
        )
        assert result.model is not None
        assert result.elapsed_time > 0
        assert len(result.forget_loss_history) == 2
        assert result.coreset_size == 40  # Full forget set used

    def test_fit_with_selector(self, tiny_model, forget_loader, retain_loader):
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner

        unlearner = ErasusUnlearner(
            model=tiny_model,
            strategy="gradient_ascent",
            selector="random",
            device="cpu",
        )
        result = unlearner.fit(
            forget_data=forget_loader,
            retain_data=retain_loader,
            prune_ratio=0.5,
            epochs=1,
        )
        assert result.model is not None
        assert result.coreset_size == 20  # 50% of 40
        assert result.compression_ratio == pytest.approx(0.5)

    def test_evaluate_with_metrics(self, tiny_model, forget_loader, retain_loader):
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner
        from erasus.metrics.accuracy import AccuracyMetric
        from erasus.metrics.membership_inference import MembershipInferenceMetric

        unlearner = ErasusUnlearner(
            model=tiny_model,
            strategy="gradient_ascent",
            selector=None,
            device="cpu",
        )
        # Run unlearning first
        unlearner.fit(
            forget_data=forget_loader,
            retain_data=retain_loader,
            epochs=1,
        )
        # Evaluate
        results = unlearner.evaluate(
            forget_data=forget_loader,
            retain_data=retain_loader,
            metrics=[AccuracyMetric(), MembershipInferenceMetric()],
        )
        assert "forget_accuracy" in results
        assert "retain_accuracy" in results
        assert "mia_accuracy" in results
        assert 0.0 <= results["mia_accuracy"] <= 1.0


class TestLLMUnlearner:
    """Test the LLM-specific unlearner."""

    def test_llm_unlearner_fit(self, tiny_model, forget_loader, retain_loader):
        from erasus.unlearners.llm_unlearner import LLMUnlearner

        unlearner = LLMUnlearner(
            model=tiny_model,
            strategy="gradient_ascent",
            selector=None,
            device="cpu",
        )
        result = unlearner.fit(
            forget_data=forget_loader,
            retain_data=retain_loader,
            epochs=1,
        )
        assert result.model is not None


class TestMultimodalUnlearner:
    """Test the auto-dispatch multimodal unlearner."""

    def test_detect_generic(self, tiny_model):
        from erasus.unlearners.multimodal_unlearner import MultimodalUnlearner
        detected = MultimodalUnlearner._detect_type(tiny_model)
        # TinyClassifier has no special attributes → generic
        assert detected == "generic"

    def test_detect_vlm(self):
        from erasus.unlearners.multimodal_unlearner import MultimodalUnlearner

        class FakeVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = nn.Linear(1, 1)
                self.text_model = nn.Linear(1, 1)

        detected = MultimodalUnlearner._detect_type(FakeVLM())
        assert detected == "vlm"

    def test_detect_llm(self):
        from erasus.unlearners.multimodal_unlearner import MultimodalUnlearner

        class FakeCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(1, 1)

        detected = MultimodalUnlearner._detect_type(FakeCausalLM())
        assert detected == "llm"

    def test_from_model_explicit_type(self, tiny_model, forget_loader, retain_loader):
        from erasus.unlearners.multimodal_unlearner import MultimodalUnlearner

        unlearner = MultimodalUnlearner.from_model(
            model=tiny_model,
            model_type="llm",
            strategy="gradient_ascent",
            selector=None,
            device="cpu",
        )
        result = unlearner.fit(
            forget_data=forget_loader,
            retain_data=retain_loader,
            epochs=1,
        )
        assert result.model is not None


class TestCLIPipeline:
    """Test the CLI entry point imports and help."""

    def test_cli_imports(self):
        from erasus.cli.main import main
        from erasus.cli.unlearn import add_parser
        from erasus.cli.evaluate import add_parser as add_eval
        assert callable(main)
        assert callable(add_parser)
        assert callable(add_eval)


class TestYAMLConfig:
    """Test YAML config loading."""

    def test_load_default_config(self, tmp_path):
        import yaml
        from erasus.core.config import ErasusConfig

        # Write a minimal YAML
        config_data = {
            "model_name": "test-model",
            "model_type": "llm",
            "strategy": "gradient_ascent",
            "selector": "random",
            "epochs": 3,
        }
        cfg_file = tmp_path / "test.yaml"
        with open(cfg_file, "w") as f:
            yaml.dump(config_data, f)

        config = ErasusConfig.from_yaml(str(cfg_file))
        assert config.model_name == "test-model"
        assert config.model_type == "llm"
        assert config.strategy == "gradient_ascent"
        assert config.selector == "random"
        assert config.epochs == 3
