"""
Example: Unlearning with BLIP-2.

Demonstrates:
1. Using the VLMUnlearner for BLIP-style models.
2. Running gradient ascent with coreset selection.
3. Comparing pre/post unlearning metrics.

Usage::

    python examples/vision_language/blip_unlearning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.vlm_unlearner import VLMUnlearner
import erasus.strategies  # noqa: F401


class TinyBLIP(nn.Module):
    """Minimal BLIP stand-in for testing."""

    def __init__(self):
        super().__init__()
        self.vision_model = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.text_model = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.classifier = nn.Linear(64, 4)
        # Attribute for VLM detection
        self.visual = True

    def forward(self, x):
        # Simple: split input into "vision" and "text" halves
        v = self.vision_model(x[:, :64])
        t = self.text_model(x[:, 64:])
        combined = torch.cat([v, t], dim=-1)
        return self.classifier(combined)


def main():
    print("=" * 60)
    print("  BLIP-2 Unlearning Example")
    print("=" * 60)

    model = TinyBLIP()

    # Synthetic data: 128-dim input (64 vision + 64 text)
    forget_x = torch.randn(24, 128)
    forget_y = torch.randint(0, 4, (24,))
    retain_x = torch.randn(48, 128)
    retain_y = torch.randint(0, 4, (48,))

    forget_loader = DataLoader(TensorDataset(forget_x, forget_y), batch_size=8)
    retain_loader = DataLoader(TensorDataset(retain_x, retain_y), batch_size=8)

    # Pre-unlearning accuracy
    from erasus.metrics.accuracy import AccuracyMetric
    acc_metric = AccuracyMetric()
    pre_metrics = acc_metric.compute(model=model, forget_data=forget_loader, retain_data=retain_loader)
    print(f"\n  Pre-unlearning: {pre_metrics}")

    # Run unlearning
    unlearner = VLMUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector=None,
        device="cpu",
    )

    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        epochs=3,
    )
    print(f"  Unlearning: {result.elapsed_time:.2f}s, coreset={result.coreset_size}")

    # Post-unlearning accuracy
    post_metrics = acc_metric.compute(
        model=unlearner.model,
        forget_data=forget_loader,
        retain_data=retain_loader,
    )
    print(f"  Post-unlearning: {post_metrics}")

    print("\n✅ BLIP unlearning example complete!")


if __name__ == "__main__":
    main()
