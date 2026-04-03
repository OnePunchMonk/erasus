"""
Tests for remaining Tier 1 unlearning strategies.

Tests UNDIALStrategy, WGAStrategy, and MetaUnlearningStrategy.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.registry import strategy_registry
from erasus.strategies.llm_specific.undial import UNDIALStrategy
from erasus.strategies.gradient_methods.weighted_gradient_ascent import FPGAStrategy, WGAStrategy
from erasus.strategies.diffusion_specific.meta_unlearning import MetaUnlearningStrategy


@pytest.fixture
def tiny_classifier():
    """Simple 2-layer classifier for testing."""
    class TinyClassifier(nn.Module):
        def __init__(self, input_dim=16, num_classes=4):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            logits = self.fc2(x)
            return logits

    return TinyClassifier().eval()


@pytest.fixture
def forget_loader():
    """Small forget dataset."""
    X = torch.randn(32, 16)
    y = torch.randint(0, 4, (32,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


@pytest.fixture
def retain_loader():
    """Small retain dataset."""
    X = torch.randn(32, 16)
    y = torch.randint(0, 4, (32,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


class TestUNDIALStrategy:
    """Test UNDIAL (Self-Distillation with Logit Adjustment)."""

    def test_init(self):
        """Test initialization with default parameters."""
        strategy = UNDIALStrategy()
        assert strategy.temperature == 3.0
        assert strategy.alpha == 0.5
        assert strategy.retain_weight == 1.0
        assert strategy.lr == 1e-5

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        strategy = UNDIALStrategy(temperature=5.0, alpha=0.3, lr=5e-5)
        assert strategy.temperature == 5.0
        assert strategy.alpha == 0.3
        assert strategy.lr == 5e-5

    def test_registry(self):
        """Test that strategy is registered."""
        assert "undial" in strategy_registry._registry
        strategy_cls = strategy_registry.get("undial")
        assert issubclass(strategy_cls, UNDIALStrategy)

    def test_unlearn_without_retain(self, tiny_classifier, forget_loader):
        """Test unlearning without retain data."""
        strategy = UNDIALStrategy(temperature=3.0, lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=None,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert len(retain_losses) == 0
        assert all(isinstance(l, float) for l in forget_losses)

    def test_unlearn_with_retain(self, tiny_classifier, forget_loader, retain_loader):
        """Test unlearning with retain data."""
        strategy = UNDIALStrategy(temperature=3.0, lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert len(retain_losses) == 2
        assert all(isinstance(l, float) for l in retain_losses)

    def test_temperature_variations(self, tiny_classifier, forget_loader):
        """Test different temperature values."""
        for temp in [1.0, 3.0, 5.0]:
            strategy = UNDIALStrategy(temperature=temp, lr=1e-3)
            model = tiny_classifier.train()

            unlearned_model, forget_losses, _ = strategy.unlearn(
                model=model,
                forget_loader=forget_loader,
                epochs=1,
            )

            assert isinstance(unlearned_model, nn.Module)
            assert len(forget_losses) == 1

    def test_unlearn_modifies_weights(self, tiny_classifier, forget_loader):
        """Test that weights are modified during unlearning."""
        strategy = UNDIALStrategy(temperature=3.0, lr=1e-3)
        model = tiny_classifier.train()
        original_params = [p.clone() for p in model.parameters()]

        unlearned_model, _, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=2,
        )

        # At least one parameter should change
        changed = any(
            not torch.allclose(orig, param)
            for orig, param in zip(original_params, unlearned_model.parameters())
        )
        assert changed


class TestWGAStrategy:
    """Test WGA (Weighted Gradient Ascent)."""

    def test_init(self):
        """Test initialization with default parameters."""
        strategy = WGAStrategy()
        assert strategy.weighting == "entropy"
        assert strategy.lr == 1e-3
        assert strategy.weight_scale == 1.0
        assert strategy.retain_weight == 1.0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        strategy = WGAStrategy(weighting="confidence", lr=5e-4, weight_scale=2.0)
        assert strategy.weighting == "confidence"
        assert strategy.lr == 5e-4
        assert strategy.weight_scale == 2.0

    def test_registry(self):
        """Test that strategy is registered."""
        assert "wga" in strategy_registry._registry
        strategy_cls = strategy_registry.get("wga")
        assert issubclass(strategy_cls, WGAStrategy)
        assert "fpga" in strategy_registry._registry
        fpga_cls = strategy_registry.get("fpga")
        assert issubclass(fpga_cls, FPGAStrategy)

    def test_unlearn_uniform(self, tiny_classifier, forget_loader):
        """Test unlearning with uniform weighting (like standard GA)."""
        strategy = WGAStrategy(weighting="uniform", lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert all(isinstance(l, float) for l in forget_losses)

    def test_unlearn_entropy(self, tiny_classifier, forget_loader):
        """Test unlearning with entropy-based weighting."""
        strategy = WGAStrategy(weighting="entropy", lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=1,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 1

    def test_unlearn_confidence(self, tiny_classifier, forget_loader):
        """Test unlearning with confidence-based weighting."""
        strategy = WGAStrategy(weighting="confidence", lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=1,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 1

    def test_fpga_init(self):
        strategy = FPGAStrategy()
        assert strategy.token_weighted is True

    def test_invalid_weighting(self):
        """Test that invalid weighting strategy raises error."""
        strategy = WGAStrategy(weighting="invalid")
        model = nn.Linear(16, 4).train()
        forget_loader = DataLoader(TensorDataset(torch.randn(16, 16), torch.randint(0, 4, (16,))), batch_size=4)

        with pytest.raises(ValueError, match="Unknown weighting"):
            strategy.unlearn(model, forget_loader, epochs=1)

    def test_compute_weights_uniform(self, tiny_classifier):
        """Test uniform weight computation."""
        strategy = WGAStrategy(weighting="uniform")
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))

        weights = strategy._compute_weights(logits, labels)

        assert weights.shape == (8,)
        assert torch.allclose(weights, torch.ones(8))

    def test_compute_weights_entropy(self, tiny_classifier):
        """Test entropy-based weight computation."""
        strategy = WGAStrategy(weighting="entropy", weight_scale=1.0)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))

        weights = strategy._compute_weights(logits, labels)

        assert weights.shape == (8,)
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_compute_weights_confidence(self, tiny_classifier):
        """Test confidence-based weight computation."""
        strategy = WGAStrategy(weighting="confidence", weight_scale=1.0)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))

        weights = strategy._compute_weights(logits, labels)

        assert weights.shape == (8,)
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_unlearn_with_retain(self, tiny_classifier, forget_loader, retain_loader):
        """Test unlearning with retain data."""
        strategy = WGAStrategy(weighting="entropy", lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert len(retain_losses) == 2


class TestMetaUnlearningStrategy:
    """Test Meta-Unlearning for diffusion models."""

    def test_init(self):
        """Test initialization with default parameters."""
        strategy = MetaUnlearningStrategy()
        assert strategy.inner_lr == 1e-4
        assert strategy.outer_lr == 1e-5
        assert strategy.num_inner_steps == 3
        assert strategy.forget_weight == 1.0
        assert strategy.retain_weight == 1.0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        strategy = MetaUnlearningStrategy(
            inner_lr=5e-4,
            outer_lr=1e-4,
            num_inner_steps=5,
            forget_weight=2.0,
        )
        assert strategy.inner_lr == 5e-4
        assert strategy.outer_lr == 1e-4
        assert strategy.num_inner_steps == 5
        assert strategy.forget_weight == 2.0

    def test_registry(self):
        """Test that strategy is registered."""
        assert "meta_unlearning" in strategy_registry._registry
        strategy_cls = strategy_registry.get("meta_unlearning")
        assert issubclass(strategy_cls, MetaUnlearningStrategy)

    def test_unlearn_without_retain(self, tiny_classifier, forget_loader):
        """Test unlearning without retain data."""
        strategy = MetaUnlearningStrategy(
            inner_lr=1e-3,
            outer_lr=1e-3,
            num_inner_steps=1,
        )
        model = tiny_classifier.train()

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=None,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert len(retain_losses) == 0
        assert all(isinstance(l, float) for l in forget_losses)

    def test_unlearn_with_retain(self, tiny_classifier, forget_loader, retain_loader):
        """Test unlearning with retain data."""
        strategy = MetaUnlearningStrategy(
            inner_lr=1e-3,
            outer_lr=1e-3,
            num_inner_steps=1,
        )
        model = tiny_classifier.train()

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert len(retain_losses) == 2
        assert all(isinstance(l, float) for l in retain_losses)

    def test_inner_steps_variations(self, tiny_classifier, forget_loader):
        """Test different numbers of inner meta-steps."""
        for num_steps in [1, 2, 3]:
            strategy = MetaUnlearningStrategy(
                inner_lr=1e-3,
                outer_lr=1e-3,
                num_inner_steps=num_steps,
            )
            model = tiny_classifier.train()

            unlearned_model, forget_losses, _ = strategy.unlearn(
                model=model,
                forget_loader=forget_loader,
                epochs=1,
            )

            assert isinstance(unlearned_model, nn.Module)
            assert len(forget_losses) == 1

    def test_unlearn_modifies_weights(self, tiny_classifier, forget_loader):
        """Test that weights are modified during meta-unlearning."""
        strategy = MetaUnlearningStrategy(
            inner_lr=1e-3,
            outer_lr=1e-3,
            num_inner_steps=1,
        )
        model = tiny_classifier.train()
        original_params = [p.clone() for p in model.parameters()]

        unlearned_model, _, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=2,
        )

        # At least one parameter should change
        changed = any(
            not torch.allclose(orig, param)
            for orig, param in zip(original_params, unlearned_model.parameters())
        )
        assert changed
