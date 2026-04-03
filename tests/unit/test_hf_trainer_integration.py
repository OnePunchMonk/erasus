"""
Tests for Hugging Face Trainer integration helpers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.integrations import UnlearningTrainerCallback, attach_unlearning_callback


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _make_loader() -> DataLoader:
    x = torch.randn(16, 16)
    y = torch.randint(0, 4, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_attach_unlearning_callback():
    trainer = type("Trainer", (), {"callbacks": []})()
    callback = UnlearningTrainerCallback("gradient_ascent", _make_loader(), unlearn_epochs=1)
    attach_unlearning_callback(trainer, callback)
    assert callback in trainer.callbacks


def test_callback_runs_unlearning():
    callback = UnlearningTrainerCallback("gradient_ascent", _make_loader(), unlearn_epochs=1)
    model = TinyClassifier().train()
    updated = callback.run_unlearning(model)
    assert updated is model
