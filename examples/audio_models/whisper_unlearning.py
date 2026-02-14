"""
Whisper Unlearning — Remove specific speech patterns from Whisper ASR.

Demonstrates unlearning of specific speakers or phrases from an
audio model using the AudioUnlearner.

Usage::

    python examples/audio_models/whisper_unlearning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.audio_unlearner import AudioUnlearner
import erasus.strategies  # noqa: F401


class TinyASR(nn.Module):
    """Minimal ASR model for demonstration."""

    def __init__(self, input_dim=80, hidden=64, vocab_size=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden, vocab_size)
        self.config = type("C", (), {"model_type": "whisper"})()

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    print("=" * 60)
    print("  Whisper Unlearning Example")
    print("=" * 60)

    model = TinyASR()
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Synthetic mel-spectrogram features
    forget = DataLoader(
        TensorDataset(torch.randn(30, 80), torch.randint(0, 100, (30,))),
        batch_size=8,
    )
    retain = DataLoader(
        TensorDataset(torch.randn(120, 80), torch.randint(0, 100, (120,))),
        batch_size=8,
    )

    unlearner = AudioUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    print("\n  Running speaker unlearning...")
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5)

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ Whisper unlearning complete!")


if __name__ == "__main__":
    main()
