"""
Reproduction: Selective Forgetting in Deep Networks (Golatkar et al., CVPR 2020).

Uses Fisher Information Matrix to protect retain-relevant parameters while
unlearning: L = L_forget + λ * Σ F_ii * (θ_i - θ_orig_i)².

Usage::

    python papers/reproductions/fisher_forgetting_cvpr2020.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401
import erasus.selectors  # noqa: F401


class MLP(nn.Module):
    """Simple MLP for reproduction."""

    def __init__(self, input_dim=64, num_classes=10, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("  Paper Reproduction: Fisher Forgetting (Golatkar et al., CVPR 2020)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    model = MLP(input_dim=64, num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: MLP ({n_params:,} params)")

    forget_data = TensorDataset(torch.randn(80, 64), torch.randint(0, 10, (80,)))
    retain_data = TensorDataset(torch.randn(320, 64), torch.randint(0, 10, (320,)))
    forget_loader = DataLoader(forget_data, batch_size=16, shuffle=True)
    retain_loader = DataLoader(retain_data, batch_size=16, shuffle=True)

    print("\n  Phase 1: Pre-training...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(5):
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"    Epoch {epoch+1}/5")

    print("\n  Phase 2: Pre-unlearning metrics...")
    suite = MetricSuite(["accuracy"])
    pre = suite.run(model, forget_loader, retain_loader)
    for k, v in pre.items():
        if k != "_meta" and isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    print("\n  Phase 3: Fisher Forgetting Unlearning...")
    unlearner = ErasusUnlearner(
        model=model,
        strategy="fisher_forgetting",
        selector=None,
        device=device,
        strategy_kwargs={"fisher_lambda": 500.0, "lr": 1e-4},
    )
    t0 = time.time()
    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        epochs=5,
    )
    elapsed = time.time() - t0
    print(f"    ✓ Done in {elapsed:.2f}s")

    print("\n  Phase 4: Post-unlearning metrics...")
    post = suite.run(unlearner.model, forget_loader, retain_loader)
    for k, v in post.items():
        if k != "_meta" and isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("  REPRODUCTION SUMMARY (Fisher Forgetting)")
    print("=" * 60)
    print(f"  Unlearning Time: {elapsed:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget Loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ Reproduction complete!")


if __name__ == "__main__":
    main()
