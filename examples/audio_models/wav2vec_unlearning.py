"""
Wav2Vec Unlearning — Remove speaker-specific features from Wav2Vec 2.0.

Usage::

    python examples/audio_models/wav2vec_unlearning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.audio_unlearner import AudioUnlearner
import erasus.strategies  # noqa: F401


class TinyWav2Vec(nn.Module):
    def __init__(self, hidden=64, vocab=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16000, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, vocab)
        self.config = type("C", (), {"model_type": "wav2vec"})()

    def forward(self, x):
        return self.head(self.encoder(x))


def main():
    print("=" * 60)
    print("  Wav2Vec Speaker Unlearning Example")
    print("=" * 60)

    model = TinyWav2Vec()

    # Raw waveform features (simplified)
    forget = DataLoader(
        TensorDataset(torch.randn(30, 16000), torch.randint(0, 50, (30,))),
        batch_size=8,
    )
    retain = DataLoader(
        TensorDataset(torch.randn(120, 16000), torch.randint(0, 50, (120,))),
        batch_size=8,
    )

    unlearner = AudioUnlearner(
        model=model,
        strategy="negative_gradient",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    print("\n  Running speaker unlearning with negative gradient...")
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ Wav2Vec unlearning complete!")


if __name__ == "__main__":
    main()
