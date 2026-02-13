"""
Example: Unlearning with LLaVA (Large Language and Vision Assistant).

Demonstrates:
1. Creating a VLM unlearner with modality decoupling strategy.
2. Running selective unlearning on vision–language data.
3. Evaluating with VLM-appropriate metrics.

Usage::

    python examples/vision_language/llava_unlearning.py

Note: This example uses a tiny model for speed. Replace with
``liuhaotian/llava-v1.5-7b`` for real experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.vlm_unlearner import VLMUnlearner
import erasus.strategies  # noqa: F401


def make_dummy_data(n_samples=32, batch_size=8):
    """Generate random data mimicking image–text pairs."""
    # Flatten images + text tokens as a simple tensor for demo
    x = torch.randn(n_samples, 128)
    y = torch.randint(0, 4, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


class TinyVLM(nn.Module):
    """Minimal VLM stand-in for testing."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(128, 64)
        self.head = nn.Linear(64, 4)
        # Mark as VLM for auto-detection
        self.visual = True
        self.text_model = True

    def forward(self, x):
        return self.head(torch.relu(self.encoder(x)))


def main():
    print("=" * 60)
    print("  LLaVA Unlearning Example")
    print("=" * 60)

    model = TinyVLM()
    forget_loader = make_dummy_data(32)
    retain_loader = make_dummy_data(64)

    print("\n1. Creating VLMUnlearner with gradient_ascent strategy...")
    unlearner = VLMUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector="random",
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    print("2. Running unlearning...")
    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        prune_ratio=0.5,
        epochs=3,
    )

    print(f"\n   ✓ Unlearning complete in {result.elapsed_time:.2f}s")
    print(f"   Coreset: {result.coreset_size}/{result.original_forget_size}")
    if result.forget_loss_history:
        print(f"   Loss trajectory: {' → '.join(f'{l:.3f}' for l in result.forget_loss_history)}")

    print("\n3. Evaluating...")
    from erasus.metrics.accuracy import AccuracyMetric
    metrics = unlearner.evaluate(
        forget_data=forget_loader,
        retain_data=retain_loader,
        metrics=[AccuracyMetric()],
    )
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    print("\n✅ LLaVA unlearning example complete!")


if __name__ == "__main__":
    main()
