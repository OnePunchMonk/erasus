"""
Focused unit tests for NPO and AltPO strategies.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry
from erasus.strategies.llm_specific.altpo import AltPOStrategy
from erasus.strategies.llm_specific.npo import NPOStrategy


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def forget_loader() -> DataLoader:
    x = torch.randn(24, 16)
    y = torch.randint(0, 4, (24,))
    return DataLoader(TensorDataset(x, y), batch_size=6)


@pytest.fixture
def retain_loader() -> DataLoader:
    x = torch.randn(24, 16)
    y = torch.randint(0, 4, (24,))
    return DataLoader(TensorDataset(x, y), batch_size=6)


class TestNPOIssueAcceptance:
    def test_subclasses_base_strategy(self) -> None:
        assert issubclass(NPOStrategy, BaseStrategy)

    def test_accepts_reference_model_parameter(self) -> None:
        reference_model = TinyClassifier()
        strategy = NPOStrategy(reference_model=reference_model)
        assert strategy.reference_model is reference_model

    def test_registered(self) -> None:
        assert strategy_registry.get("npo") is NPOStrategy

    def test_dpo_style_loss(self) -> None:
        strategy = NPOStrategy(beta=0.5)
        student = torch.log_softmax(torch.randn(5, 4), dim=-1)
        reference = torch.log_softmax(torch.randn(5, 4), dim=-1)
        labels = torch.tensor([0, 1, 2, 3, 0])
        loss = strategy.dpo_style_loss(student, reference, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_unlearn_runs_with_reference_model(self, forget_loader: DataLoader, retain_loader: DataLoader) -> None:
        model = TinyClassifier().train()
        reference_model = copy.deepcopy(model)
        strategy = NPOStrategy(reference_model=reference_model, lr=1e-3)

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert len(retain_losses) == 2


class TestAltPOIssueAcceptance:
    def test_subclasses_base_strategy(self) -> None:
        assert issubclass(AltPOStrategy, BaseStrategy)

    def test_registered(self) -> None:
        assert strategy_registry.get("altpo") is AltPOStrategy

    def test_preference_loss_prefers_alternative(self) -> None:
        strategy = AltPOStrategy(beta=0.5)
        alt_lp = torch.tensor([-0.5, -0.3, -0.1])
        true_lp = torch.tensor([-2.0, -1.5, -1.2])
        loss = strategy.preference_loss(alt_lp, true_lp)
        assert loss.ndim == 0
        assert loss.item() < 1.0

    def test_alternates_forget_and_retain_steps(self, forget_loader: DataLoader, retain_loader: DataLoader) -> None:
        model = TinyClassifier().train()
        strategy = AltPOStrategy(lr=1e-3, alt_strategy="random")

        unlearned_model, forget_losses, retain_losses = strategy.unlearn(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )

        assert isinstance(unlearned_model, nn.Module)
        assert len(forget_losses) == 2
        assert len(retain_losses) == 2
