"""
VideoMAE Unlearning — Remove action/identity features from VideoMAE.

Usage::

    python examples/video_models/videomae_unlearning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.video_unlearner import VideoUnlearner
import erasus.strategies  # noqa: F401


class TinyVideoModel(nn.Module):
    def __init__(self, hidden=64, n_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 4 * 32 * 32, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, n_classes)
        self.config = type("C", (), {"model_type": "videomae"})()

    def forward(self, x):
        return self.head(self.encoder(x))


def main():
    print("=" * 60)
    print("  VideoMAE Action Unlearning Example")
    print("=" * 60)

    model = TinyVideoModel()

    # Synthetic video data: (B, C, T, H, W)
    forget = DataLoader(
        TensorDataset(torch.randn(20, 3, 4, 32, 32), torch.randint(0, 10, (20,))),
        batch_size=4,
    )
    retain = DataLoader(
        TensorDataset(torch.randn(80, 3, 4, 32, 32), torch.randint(0, 10, (80,))),
        batch_size=4,
    )

    unlearner = VideoUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    print("\n  Running action unlearning...")
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ VideoMAE unlearning complete!")


if __name__ == "__main__":
    main()
