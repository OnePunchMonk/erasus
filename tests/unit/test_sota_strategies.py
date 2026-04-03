"""
Tests for SOTA preference-based unlearning strategies.

Tests NPOStrategy, SimNPOStrategy, AltPOStrategy, FLATStrategy, and RMUStrategy.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry
from erasus.strategies.inference_time.base import BaseInferenceTimeStrategy
from erasus.strategies.llm_specific.npo import NPOStrategy
from erasus.strategies.llm_specific.simnpo import SimNPOStrategy
from erasus.strategies.llm_specific.altpo import AltPOStrategy
from erasus.strategies.llm_specific.flat import FLATStrategy
from erasus.strategies.llm_specific.delta_unlearning import (
    DeltaUnlearningStrategy,
    DeltaUnlearningWrapper,
)
from erasus.strategies.llm_specific.rmu import RMUStrategy
from erasus.strategies.inference_time.dexperts import DExpertsStrategy, DExpertsWrapper
from erasus.strategies.inference_time.activation_steering import (
    ActivationSteeringStrategy,
    SteeringModel,
)


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


class TestNPOStrategy:
    """Test Negative Preference Optimization."""

    def test_init(self):
        """Test initialization with default parameters."""
        strategy = NPOStrategy()
        assert strategy.beta == 0.1
        assert strategy.retain_weight == 1.0
        assert strategy.lr == 1e-5

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        strategy = NPOStrategy(beta=0.5, retain_weight=2.0, lr=5e-5)
        assert strategy.beta == 0.5
        assert strategy.retain_weight == 2.0
        assert strategy.lr == 5e-5

    def test_registry(self):
        """Test that strategy is registered."""
        assert "npo" in strategy_registry._registry
        strategy_cls = strategy_registry.get("npo")
        assert issubclass(strategy_cls, NPOStrategy)

    def test_unlearn_without_retain(self, tiny_classifier, forget_loader):
        """Test unlearning without retain data."""
        strategy = NPOStrategy(beta=0.1, lr=1e-3)
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
        strategy = NPOStrategy(beta=0.1, lr=1e-3)
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

    def test_unlearn_modifies_weights(self, tiny_classifier, forget_loader):
        """Test that weights are modified during unlearning."""
        strategy = NPOStrategy(beta=0.1, lr=1e-3)
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


class TestSimNPOStrategy:
    """Test Simplified Negative Preference Optimization."""

    def test_init(self):
        """Test initialization."""
        strategy = SimNPOStrategy()
        assert strategy.beta == 0.1
        assert strategy.gamma == 1.0
        assert strategy.lr == 1e-5

    def test_registry(self):
        """Test that strategy is registered."""
        assert "simnpo" in strategy_registry._registry
        strategy_cls = strategy_registry.get("simnpo")
        assert issubclass(strategy_cls, SimNPOStrategy)
        assert issubclass(strategy_cls, BaseStrategy)

    def test_unlearn(self, tiny_classifier, forget_loader, retain_loader):
        """Test forget-only unlearning even when retain data is passed."""
        strategy = SimNPOStrategy(beta=0.1, lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert all(isinstance(l, float) for l in forget_losses)
        assert len(retain_losses) == 0

    def test_no_reference_needed(self, tiny_classifier, forget_loader):
        """Test that SimNPO works without retain data."""
        strategy = SimNPOStrategy()
        model = tiny_classifier.train()

        unlearned_model, forget_losses, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=None,
            epochs=1,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 1


class TestAltPOStrategy:
    """Test Alternate Preference Optimization."""

    def test_init_default(self):
        """Test initialization with default strategy."""
        strategy = AltPOStrategy()
        assert strategy.beta == 0.1
        assert strategy.alt_strategy == "uniform"

    def test_init_strategies(self):
        """Test all alternative strategies."""
        for alt_strat in ["uniform", "random", "lowest"]:
            strategy = AltPOStrategy(alt_strategy=alt_strat)
            assert strategy.alt_strategy == alt_strat

    def test_registry(self):
        """Test that strategy is registered."""
        assert "altpo" in strategy_registry._registry
        strategy_cls = strategy_registry.get("altpo")
        assert issubclass(strategy_cls, AltPOStrategy)

    def test_unlearn_uniform(self, tiny_classifier, forget_loader, retain_loader):
        """Test unlearning with uniform alternative strategy."""
        strategy = AltPOStrategy(alt_strategy="uniform", lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2

    def test_unlearn_random(self, tiny_classifier, forget_loader):
        """Test unlearning with random alternative strategy."""
        strategy = AltPOStrategy(alt_strategy="random", lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, _, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=1,
        )

        assert isinstance(unlearned_model, nn.Module)

    def test_unlearn_lowest(self, tiny_classifier, forget_loader):
        """Test unlearning with lowest alternative strategy."""
        strategy = AltPOStrategy(alt_strategy="lowest", lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, _, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=1,
        )

        assert isinstance(unlearned_model, nn.Module)

    def test_alt_log_prob_uniform(self, tiny_classifier):
        """Test uniform alternative log probability."""
        strategy = AltPOStrategy(alt_strategy="uniform")
        log_probs = torch.randn(4, 10)  # batch_size=4, n_classes=10
        true_labels = torch.tensor([0, 1, 2, 3])

        alt_lp = strategy._alt_log_prob(log_probs, true_labels, 10)

        assert alt_lp.shape == (4,)
        expected = -torch.log(torch.tensor(10.0)).item()
        assert torch.allclose(alt_lp, torch.full((4,), expected))

    def test_alt_log_prob_lowest(self, tiny_classifier):
        """Test lowest alternative log probability."""
        strategy = AltPOStrategy(alt_strategy="lowest")
        log_probs = torch.tensor(
            [
                [-1.0, -2.0, -3.0, -4.0],
                [-5.0, -6.0, -7.0, -8.0],
            ]
        )
        true_labels = torch.tensor([0, 1])

        alt_lp = strategy._alt_log_prob(log_probs, true_labels, 4)

        assert alt_lp.shape == (2,)
        assert alt_lp[0] == -4.0  # lowest of [-1, -2, -3, -4]
        assert alt_lp[1] == -8.0  # lowest of [-5, -6, -7, -8]


class TestFLATStrategy:
    """Test FLAT (Loss Adjustment) Strategy."""

    def test_init(self):
        """Test initialization."""
        strategy = FLATStrategy()
        assert strategy.alpha == 0.5
        assert strategy.idk_weight == 1.0
        assert strategy.maintain_weight == 1.0

    def test_registry(self):
        """Test that strategy is registered."""
        assert "flat" in strategy_registry._registry
        strategy_cls = strategy_registry.get("flat")
        assert issubclass(strategy_cls, FLATStrategy)
        assert issubclass(strategy_cls, BaseStrategy)

    def test_unlearn_no_retain(self, tiny_classifier, forget_loader):
        """Test unlearning without retain data (uses self-distillation)."""
        strategy = FLATStrategy(alpha=0.5, lr=1e-3)
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

    def test_unlearn_with_retain(self, tiny_classifier, forget_loader, retain_loader):
        """Test retain data is optional and ignored by forget-only FLAT."""
        strategy = FLATStrategy(alpha=0.5, lr=1e-3)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert len(retain_losses) == 0

    def test_alpha_variations(self, tiny_classifier, forget_loader):
        """Test different alpha values."""
        for alpha in [0.0, 0.5, 1.0]:
            strategy = FLATStrategy(alpha=alpha, lr=1e-3)
            model = tiny_classifier.train()

            unlearned_model, forget_losses, _ = strategy.unlearn(
                model=model,
                forget_loader=forget_loader,
                epochs=1,
            )

            assert isinstance(unlearned_model, nn.Module)
            assert len(forget_losses) == 1


class TestRMUStrategy:
    """Test Representation Misdirection for Unlearning."""

    def test_init(self):
        """Test initialization."""
        strategy = RMUStrategy()
        assert strategy.layer_ids is None
        assert strategy.alpha == 1.0
        assert strategy.retain_weight == 1.0

    def test_init_with_layer_ids(self):
        """Test initialization with specific layer IDs."""
        strategy = RMUStrategy(layer_ids=[0, 2, 4])
        assert strategy.layer_ids == [0, 2, 4]

    def test_registry(self):
        """Test that strategy is registered."""
        assert "rmu" in strategy_registry._registry
        strategy_cls = strategy_registry.get("rmu")
        assert issubclass(strategy_cls, RMUStrategy)

    def test_unlearn_without_retain(self, tiny_classifier, forget_loader):
        """Test unlearning without retain data."""
        strategy = RMUStrategy(alpha=1.0, lr=1e-4)
        model = tiny_classifier.train()

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=None,
            epochs=1,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 1
        assert len(retain_losses) == 0

    def test_capture_hidden(self, tiny_classifier):
        """Test hidden state capture via hooks."""
        model = tiny_classifier.eval()
        inputs = torch.randn(8, 16)

        # Get all layers
        all_layers = [
            (name, mod)
            for name, mod in model.named_modules()
            if isinstance(mod, (nn.Linear, nn.Conv2d))
        ]
        assert len(all_layers) > 0

        captured = RMUStrategy._capture_hidden(model, inputs, all_layers)

        assert isinstance(captured, dict)
        assert len(captured) == len(all_layers)


class TestDExpertsStrategy:
    """Test inference-time DExperts Strategy."""

    def test_wrapper_init(self, tiny_classifier):
        """Test DExpertsWrapper initialization."""
        base_model = tiny_classifier
        anti_expert = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

        wrapper = DExpertsWrapper(base_model, anti_expert, alpha=1.5)

        assert wrapper.alpha == 1.5
        assert wrapper.base_model is base_model
        assert wrapper.anti_expert is anti_expert

    def test_wrapper_forward(self, tiny_classifier):
        """Test DExpertsWrapper forward pass."""
        base_model = tiny_classifier.eval()
        anti_expert = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        ).eval()

        wrapper = DExpertsWrapper(base_model, anti_expert, alpha=1.0)

        inputs = torch.randn(8, 16)
        outputs = wrapper(inputs)

        assert outputs.shape == (8, 4)
        assert isinstance(outputs, torch.Tensor)

    def test_registry(self):
        """Test that strategy is registered."""
        assert "dexperts" in strategy_registry._registry
        strategy_cls = strategy_registry.get("dexperts")
        assert issubclass(strategy_cls, DExpertsStrategy)
        assert issubclass(strategy_cls, BaseInferenceTimeStrategy)

    def test_strategy_unlearn(self, tiny_classifier, forget_loader):
        """Test DExpertsStrategy unlearning."""
        strategy = DExpertsStrategy(alpha=1.0)
        model = tiny_classifier.eval()

        unlearned_wrapper, anti_losses, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=2,
        )

        # DExpertsStrategy returns a wrapper, not a modified model
        assert isinstance(unlearned_wrapper, DExpertsWrapper)
        assert len(anti_losses) == 0

    def test_requires_training_false(self):
        strategy = DExpertsStrategy()
        assert strategy.requires_training is False

    def test_no_weight_modification(self, tiny_classifier, forget_loader):
        """Test that base model weights are not modified."""
        strategy = DExpertsStrategy(alpha=1.0, anti_expert_lr=1e-3)
        model = tiny_classifier.eval()
        original_params = [p.clone() for p in model.parameters()]

        wrapper, _, _ = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=1,
        )

        # Base model weights should be unchanged
        for orig, param in zip(original_params, wrapper.base_model.parameters()):
            assert torch.allclose(orig, param)

    def test_wrapper_getattr_proxy(self, tiny_classifier):
        """Test that wrapper proxies attribute access to base model."""
        base_model = tiny_classifier
        anti_expert = nn.Linear(16, 4)

        wrapper = DExpertsWrapper(base_model, anti_expert)

        # Should proxy to base_model
        if hasattr(base_model, "fc1"):
            assert wrapper.fc1 is base_model.fc1


class TestDeltaUnlearningStrategy:
    """Test delta-unlearning for black-box offset wrapping."""

    def test_registry(self):
        assert "delta_unlearning" in strategy_registry._registry
        strategy_cls = strategy_registry.get("delta_unlearning")
        assert issubclass(strategy_cls, DeltaUnlearningStrategy)

    def test_unlearn_returns_wrapper(self, tiny_classifier, forget_loader):
        strategy = DeltaUnlearningStrategy(proxy_hidden_dim=32, lr=1e-3)
        model = tiny_classifier.eval()

        wrapped_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=1,
        )

        assert isinstance(wrapped_model, DeltaUnlearningWrapper)
        assert len(forget_losses) == 1
        assert len(retain_losses) == 0

    def test_wrapper_forward(self, tiny_classifier):
        model = tiny_classifier.eval()
        proxy = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        ).eval()
        wrapper = DeltaUnlearningWrapper(model, proxy)
        inputs = torch.randn(2, 16)
        outputs = wrapper(inputs)
        assert outputs.shape == (2, 4)
        assert wrapper.base_model is model


class TestActivationSteeringStrategy:
    """Test inference-time activation steering."""

    def test_registry(self):
        assert "activation_steering" in strategy_registry._registry
        strategy_cls = strategy_registry.get("activation_steering")
        assert issubclass(strategy_cls, ActivationSteeringStrategy)
        assert issubclass(strategy_cls, BaseInferenceTimeStrategy)

    def test_unlearn_returns_steering_model(self, tiny_classifier, forget_loader):
        strategy = ActivationSteeringStrategy(
            target_layer="middle",
            steering_strength=0.5,
            lr=1e-2,
            num_vectors=1,
        )
        model = tiny_classifier.train()

        wrapped_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            epochs=1,
        )

        assert isinstance(wrapped_model, SteeringModel)
        assert len(forget_losses) == 1
        assert len(retain_losses) == 0
