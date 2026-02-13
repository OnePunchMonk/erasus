"""
Unit tests for coreset selectors.

Tests the core selector implementations: gradient_norm, random, k-means,
influence, and the ensemble voter.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.registry import selector_registry

# Ensure selectors are registered
import erasus.selectors  # noqa: F401


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model():
    return SmallNet()


@pytest.fixture
def loader():
    x = torch.randn(32, 16)
    y = torch.randint(0, 4, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


class TestSelectorRegistry:
    """Verify all expected selectors are registered."""

    def test_gradient_norm_registered(self):
        assert selector_registry.get("gradient_norm") is not None

    def test_random_registered(self):
        assert selector_registry.get("random") is not None

    def test_influence_registered(self):
        assert selector_registry.get("influence") is not None

    def test_el2n_registered(self):
        assert selector_registry.get("el2n") is not None

    def test_kmeans_registered(self):
        assert selector_registry.get("kmeans") is not None


class TestRandomSelector:
    """Test the random coreset selector."""

    def test_select_returns_subset(self, model, loader):
        cls = selector_registry.get("random")
        selector = cls()
        indices = selector.select(model, loader, 16)
        assert len(indices) == 16

    def test_select_respects_budget(self, model, loader):
        cls = selector_registry.get("random")
        selector = cls()
        for budget in [4, 8, 16, 32]:
            indices = selector.select(model, loader, budget)
            assert len(indices) <= budget


class TestGradientNormSelector:
    """Test the gradient-norm-based selector."""

    def test_select_returns_indices(self, model, loader):
        cls = selector_registry.get("gradient_norm")
        selector = cls()
        indices = selector.select(model, loader, 10)
        assert len(indices) == 10
        assert all(isinstance(i, int) for i in indices)

    def test_select_indices_in_range(self, model, loader):
        cls = selector_registry.get("gradient_norm")
        selector = cls()
        indices = selector.select(model, loader, 10)
        assert all(0 <= i < 32 for i in indices)
