"""
Reproduction: Amnesiac Machine Learning (Brophy & Lowd, AAAI 2021).

Amnesiac: train shadow models with/without the forget data and align
the main model to the "without" shadow via distillation or parameter matching.

Usage::

    python papers/reproductions/amnesiac_neurips2021.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401
import erasus.selectors  # noqa: F401


class SmallCNN(nn.Module):
    """Tiny CNN for Amnesiac reproduction."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    print("=" * 60)
    print("  Paper Reproduction: Amnesiac ML (Brophy & Lowd, AAAI 2021)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    model = SmallCNN(num_classes=10).to(device)
    print(f"  Model: SmallCNN ({sum(p.numel() for p in model.parameters()):,} params)")

    forget_data = TensorDataset(
        torch.randn(60, 3, 32, 32), torch.randint(0, 10, (60,))
    )
    retain_data = TensorDataset(
        torch.randn(240, 3, 32, 32), torch.randint(0, 10, (240,))
    )
    forget_loader = DataLoader(forget_data, batch_size=16, shuffle=True)
    retain_loader = DataLoader(retain_data, batch_size=16, shuffle=True)

    print("\n  Phase 1: Pre-training...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(3):
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"    Epoch {epoch+1}/3")

    print("\n  Phase 2: Pre-unlearning metrics...")
    suite = MetricSuite(["accuracy"])
    pre = suite.run(model, forget_loader, retain_loader)
    for k, v in pre.items():
        if k != "_meta" and isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    print("\n  Phase 3: Amnesiac Unlearning...")
    unlearner = ErasusUnlearner(
        model=model,
        strategy="amnesiac",
        selector=None,
        device=device,
        strategy_kwargs={"unlearn_lr": 1e-4},
    )
    t0 = time.time()
    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        epochs=3,
    )
    elapsed = time.time() - t0
    print(f"    ✓ Done in {elapsed:.2f}s")

    print("\n  Phase 4: Post-unlearning metrics...")
    post = suite.run(unlearner.model, forget_loader, retain_loader)
    for k, v in post.items():
        if k != "_meta" and isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("  REPRODUCTION SUMMARY (Amnesiac)")
    print("=" * 60)
    print(f"  Unlearning Time: {elapsed:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget Loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ Reproduction complete!")


if __name__ == "__main__":
    main()
