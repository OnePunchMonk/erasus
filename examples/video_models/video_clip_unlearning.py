"""
VideoCLIP Unlearning — Remove video-text associations from VideoCLIP.

Uses cross-modal decoupling to selectively unlearn concept associations
in video-language embeddings.

Usage::

    python examples/video_models/video_clip_unlearning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.video_unlearner import VideoUnlearner
import erasus.strategies  # noqa: F401


class TinyVideoCLIP(nn.Module):
    def __init__(self, hidden=64, vocab=256):
        super().__init__()
        self.video_enc = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 16 * 16, hidden), nn.ReLU())
        self.text_enc = nn.Sequential(nn.Embedding(vocab, hidden))
        self.head = nn.Linear(hidden, 10)
        self.config = type("C", (), {"model_type": "video_clip"})()

    def forward(self, x, text=None):
        return self.head(self.video_enc(x))


def main():
    print("=" * 60)
    print("  VideoCLIP Concept Unlearning Example")
    print("=" * 60)

    model = TinyVideoCLIP()

    forget = DataLoader(
        TensorDataset(torch.randn(20, 3, 4, 16, 16), torch.randint(0, 10, (20,))),
        batch_size=4,
    )
    retain = DataLoader(
        TensorDataset(torch.randn(80, 3, 4, 16, 16), torch.randint(0, 10, (80,))),
        batch_size=4,
    )

    unlearner = VideoUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    print("\n  Running video-text concept unlearning...")
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ VideoCLIP unlearning complete!")


if __name__ == "__main__":
    main()
