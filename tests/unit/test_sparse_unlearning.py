"""
Tests for sparse unlearning strategies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.registry import strategy_registry
from erasus.strategies.parameter_methods.parameter_subset import ParameterSubsetUnlearningStrategy


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.randn(24, 16), torch.randint(0, 4, (24,))),
        batch_size=6,
    )


def test_parameter_subset_registered():
    assert strategy_registry.get("parameter_subset") is ParameterSubsetUnlearningStrategy


def test_parameter_subset_unlearn_runs():
    model = TinyClassifier().train()
    strategy = ParameterSubsetUnlearningStrategy(sparsity=0.5, lr=1e-3)
    updated, forget_losses, retain_losses = strategy.unlearn(model, _loader(), _loader(), epochs=2)
    assert updated is model
    assert len(forget_losses) == 2
    assert len(retain_losses) == 2
