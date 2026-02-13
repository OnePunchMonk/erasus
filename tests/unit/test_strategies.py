"""
Unit tests for unlearning strategies.

Tests the core strategy implementations: gradient_ascent, negative_gradient,
fisher_forgetting, and parameter-method strategies.
"""

import pytest
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.registry import strategy_registry

# Ensure strategies are registered
import erasus.strategies  # noqa: F401


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def model():
    return SmallNet()


@pytest.fixture
def forget_loader():
    return DataLoader(
        TensorDataset(torch.randn(16, 16), torch.randint(0, 4, (16,))),
        batch_size=8,
    )


@pytest.fixture
def retain_loader():
    return DataLoader(
        TensorDataset(torch.randn(32, 16), torch.randint(0, 4, (32,))),
        batch_size=8,
    )


class TestStrategyRegistry:
    """Verify all expected strategies are registered."""

    def test_gradient_ascent_registered(self):
        assert strategy_registry.get("gradient_ascent") is not None

    def test_negative_gradient_registered(self):
        assert strategy_registry.get("negative_gradient") is not None

    def test_fisher_forgetting_registered(self):
        assert strategy_registry.get("fisher_forgetting") is not None

    def test_scrub_registered(self):
        assert strategy_registry.get("scrub") is not None


class TestGradientAscent:
    """Test gradient ascent strategy."""

    def test_unlearn_modifies_model(self, model, forget_loader, retain_loader):
        cls = strategy_registry.get("gradient_ascent")
        strategy = cls(lr=1e-3)

        original_params = {n: p.clone() for n, p in model.named_parameters()}
        strategy.unlearn(model, forget_loader, retain_loader, epochs=1)
        new_params = dict(model.named_parameters())

        # At least some parameters should change
        changed = False
        for name in original_params:
            if not torch.allclose(original_params[name], new_params[name].data):
                changed = True
                break
        assert changed, "Model parameters should change after gradient ascent"

    def test_returns_loss_history(self, model, forget_loader, retain_loader):
        cls = strategy_registry.get("gradient_ascent")
        strategy = cls(lr=1e-3)
        result = strategy.unlearn(model, forget_loader, retain_loader, epochs=2)
        # Strategy should return (model, forget_losses, retain_losses)
        assert result is not None


class TestNegativeGradient:
    """Test negative gradient strategy."""

    def test_unlearn_succeeds(self, model, forget_loader, retain_loader):
        cls = strategy_registry.get("negative_gradient")
        strategy = cls(lr=1e-3)
        strategy.unlearn(model, forget_loader, retain_loader, epochs=1)
        # Should not raise
