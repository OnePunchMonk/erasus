"""
Reproduction: SCRUB — Selective Concept Removal Using Barriers (CVPR 2024).

Kurmanji et al. — Reproduces the core experiment of training a
classifier and unlearning via the SCRUB two-phase approach:
1. Maximise forget loss (forget step)
2. Minimise retain loss with KL barrier (retain step)

Usage::

    python papers/reproductions/scrub_cvpr2024.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401
import erasus.selectors   # noqa: F401


class ResNetSmall(nn.Module):
    """Simplified ResNet-like model."""

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def main():
    print("=" * 60)
    print("  Paper Reproduction: SCRUB (Kurmanji et al., CVPR 2024)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    model = ResNetSmall(num_classes=10).to(device)
    print(f"  Model: ResNetSmall ({sum(p.numel() for p in model.parameters()):,} params)")

    # Synthetic CIFAR-like data
    forget_data = TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))
    retain_data = TensorDataset(torch.randn(400, 3, 32, 32), torch.randint(0, 10, (400,)))
    forget_loader = DataLoader(forget_data, batch_size=32, shuffle=True)
    retain_loader = DataLoader(retain_data, batch_size=32, shuffle=True)

    # Phase 1: Pre-training
    print("\n  Phase 1: Pre-training...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(3):
        total_loss = 0
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}: loss={total_loss/len(retain_loader):.4f}")

    # Phase 2: Pre-unlearning evaluation
    print("\n  Phase 2: Pre-unlearning metrics...")
    suite = MetricSuite(["accuracy"])
    pre_metrics = suite.run(model, forget_loader, retain_loader)
    for k, v in pre_metrics.items():
        if k != "_meta" and isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # Phase 3: SCRUB Unlearning
    print("\n  Phase 3: SCRUB Unlearning...")
    unlearner = ErasusUnlearner(
        model=model,
        strategy="scrub",
        selector=None,
        device=device,
        strategy_kwargs={"lr": 0.001},
    )

    t0 = time.time()
    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        epochs=5,
    )
    elapsed = time.time() - t0
    print(f"    ✓ Done in {elapsed:.2f}s")

    # Phase 4: Post-unlearning evaluation
    print("\n  Phase 4: Post-unlearning metrics...")
    post_metrics = suite.run(unlearner.model, forget_loader, retain_loader)
    for k, v in post_metrics.items():
        if k != "_meta" and isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("  REPRODUCTION SUMMARY")
    print("=" * 60)
    print(f"  Strategy: SCRUB (two-phase: forget + retain barrier)")
    print(f"  Unlearning Time: {elapsed:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget Loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n  Expected: forget accuracy ↓, retain accuracy stable")
    print("\n✅ SCRUB reproduction complete!")


if __name__ == "__main__":
    main()
