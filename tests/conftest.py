"""
Shared test fixtures for the Erasus test suite.

Provides reusable PyTorch models, datasets, and dataloaders that are
small enough for fast CI testing.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---- Tiny Models ----

class TinyClassifier(nn.Module):
    """A minimal fully-connected classifier for testing."""

    def __init__(self, input_dim: int = 16, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class TinyCNN(nn.Module):
    """A minimal CNN for image-like testing."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.classifier(feat)


# ---- Fixtures ----

@pytest.fixture
def tiny_model() -> nn.Module:
    """Return a tiny FC classifier."""
    return TinyClassifier(input_dim=16, num_classes=4)


@pytest.fixture
def tiny_cnn() -> nn.Module:
    """Return a tiny CNN."""
    return TinyCNN(num_classes=4)


def _make_dataset(
    n_samples: int = 64,
    input_dim: int = 16,
    num_classes: int = 4,
) -> TensorDataset:
    """Create a random TensorDataset."""
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(x, y)


def _make_loader(
    n_samples: int = 64,
    input_dim: int = 16,
    num_classes: int = 4,
    batch_size: int = 16,
) -> DataLoader:
    """Create a DataLoader from random data."""
    ds = _make_dataset(n_samples, input_dim, num_classes)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@pytest.fixture
def forget_loader() -> DataLoader:
    """DataLoader for the forget set (32 random samples)."""
    return _make_loader(n_samples=32, input_dim=16, num_classes=4, batch_size=8)


@pytest.fixture
def retain_loader() -> DataLoader:
    """DataLoader for the retain set (64 random samples)."""
    return _make_loader(n_samples=64, input_dim=16, num_classes=4, batch_size=8)


@pytest.fixture
def forget_retain_pair() -> Tuple[DataLoader, DataLoader]:
    """Return (forget_loader, retain_loader) as a pair."""
    return (
        _make_loader(n_samples=32, input_dim=16, num_classes=4, batch_size=8),
        _make_loader(n_samples=64, input_dim=16, num_classes=4, batch_size=8),
    )
