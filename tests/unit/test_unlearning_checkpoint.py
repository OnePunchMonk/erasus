"""Tests for resumable unlearning checkpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.utils.unlearning_checkpoint import (
    load_unlearning_checkpoint,
    save_unlearning_checkpoint,
)


def test_save_and_load_roundtrip():
    model = nn.Linear(8, 2)
    with tempfile.TemporaryDirectory() as tmp:
        save_unlearning_checkpoint(
            tmp,
            model=model,
            forget_losses=[1.0, 0.5],
            retain_losses=[2.0],
            epochs_completed=2,
        )
        m2 = nn.Linear(8, 2)
        meta = load_unlearning_checkpoint(tmp, m2)
        assert meta["epochs_completed"] == 2
        assert meta["forget_losses"] == [1.0, 0.5]
        assert torch.allclose(model.weight, m2.weight)


def test_base_unlearner_resume(tmp_path: Path):
    from erasus.unlearners.erasus_unlearner import ErasusUnlearner

    n, d, c = 24, 8, 3
    forget = DataLoader(
        TensorDataset(torch.randn(n, d), torch.randint(0, c, (n,))),
        batch_size=8,
    )
    retain = DataLoader(
        TensorDataset(torch.randn(n, d), torch.randint(0, c, (n,))),
        batch_size=8,
    )
    model = nn.Sequential(nn.Linear(d, 4), nn.ReLU(), nn.Linear(4, c))
    ck_dir = tmp_path / "ck"
    u = ErasusUnlearner(model=model, strategy="gradient_ascent", device="cpu")
    u.fit(
        forget,
        retain,
        epochs=1,
        checkpoint_dir=str(ck_dir),
        checkpoint_every=1,
    )
    assert (ck_dir / "unlearning_checkpoint.pt").exists()

    u2 = ErasusUnlearner(model=model, strategy="gradient_ascent", device="cpu")
    u2.fit(forget, retain, epochs=1, resume_from=str(ck_dir))
