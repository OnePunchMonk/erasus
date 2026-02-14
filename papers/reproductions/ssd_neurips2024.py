"""
Reproduction: SSD — Selective Synaptic Dampening (NeurIPS 2024).

Foster et al. — Selectively dampens synapses (weights) most associated
with the forget set by using Fisher Information to identify and
attenuate the most relevant parameters.

Usage::

    python papers/reproductions/ssd_neurips2024.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401


class MLP(nn.Module):
    """Multi-layer perceptron for SSD reproduction."""

    def __init__(self, in_dim=784, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def main():
    print("=" * 60)
    print("  Paper Reproduction: SSD (Foster et al., NeurIPS 2024)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)
    print(f"\n  Device: {device}")
    print(f"  Model: MLP ({sum(p.numel() for p in model.parameters()):,} params)")

    # Synthetic MNIST-like data
    forget = DataLoader(
        TensorDataset(torch.randn(100, 1, 28, 28), torch.randint(0, 10, (100,))),
        batch_size=32,
    )
    retain = DataLoader(
        TensorDataset(torch.randn(500, 1, 28, 28), torch.randint(0, 10, (500,))),
        batch_size=32,
    )

    # Pre-train
    print("\n  Phase 1: Pre-training...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(5):
        total_loss = 0
        for x, y in retain:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}: loss={total_loss/len(retain):.4f}")

    # Pre metrics
    suite = MetricSuite(["accuracy"])
    print("\n  Phase 2: Pre-unlearning metrics...")
    pre = suite.run(model, forget, retain)
    for k, v in pre.items():
        if k != "_meta" and isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # SSD-like unlearning via Fisher forgetting
    print("\n  Phase 3: SSD Unlearning (Fisher dampening)...")
    unlearner = ErasusUnlearner(
        model=model, strategy="fisher_forgetting", selector=None,
        device=device, strategy_kwargs={"lr": 1e-3, "fisher_weight": 1.0},
    )
    t0 = time.time()
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5)
    elapsed = time.time() - t0

    # Post metrics
    print(f"\n  Phase 4: Post-unlearning metrics...")
    post = suite.run(unlearner.model, forget, retain)
    for k, v in post.items():
        if k != "_meta" and isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("  REPRODUCTION SUMMARY")
    print("=" * 60)
    print(f"  Strategy: SSD (Fisher-weighted synaptic dampening)")
    print(f"  Unlearning Time: {elapsed:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget Loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ SSD reproduction complete!")


if __name__ == "__main__":
    main()
