"""
Adversarial Unlearning — Adversarial robustness during unlearning.

Tests whether unlearning introduces adversarial vulnerabilities
and demonstrates robust unlearning with retain-set regularisation.

Usage::

    python examples/advanced/adversarial_unlearning.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
import erasus.strategies  # noqa: F401


class RobustClassifier(nn.Module):
    def __init__(self, dim=16, hidden=64, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def fgsm_attack(model, x, y, epsilon=0.1):
    """Fast Gradient Sign Method attack."""
    x_adv = x.clone().detach().requires_grad_(True)
    out = model(x_adv)
    loss = F.cross_entropy(out, y)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()


def eval_robust_accuracy(model, loader, epsilon=0.1):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    clean_acc = correct / total if total > 0 else 0

    # Adversarial accuracy
    adv_correct = 0
    for x, y in loader:
        x_adv = fgsm_attack(model, x, y, epsilon)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
            adv_correct += (pred == y).sum().item()
    adv_acc = adv_correct / total if total > 0 else 0

    return clean_acc, adv_acc


def main():
    print("=" * 60)
    print("  Adversarial Unlearning Example")
    print("=" * 60)

    model = RobustClassifier()
    forget = DataLoader(TensorDataset(torch.randn(40, 16), torch.randint(0, 4, (40,))), batch_size=16)
    retain = DataLoader(TensorDataset(torch.randn(160, 16), torch.randint(0, 4, (160,))), batch_size=16)

    # Pre-train
    print("\n  Phase 1: Pre-training...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        for x, y in retain:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    clean_acc, adv_acc = eval_robust_accuracy(model, retain)
    print(f"    Pre-unlearning: clean={clean_acc:.3f}, adversarial={adv_acc:.3f}")

    # Unlearn
    print("\n  Phase 2: Unlearning...")
    unlearner = ErasusUnlearner(
        model=model, strategy="gradient_ascent", selector=None,
        device="cpu", strategy_kwargs={"lr": 1e-3},
    )
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5)

    clean_acc, adv_acc = eval_robust_accuracy(unlearner.model, retain)
    print(f"    Post-unlearning: clean={clean_acc:.3f}, adversarial={adv_acc:.3f}")

    print(f"\n  ✓ Done in {result.elapsed_time:.2f}s")
    print("\n✅ Adversarial unlearning analysis complete!")


if __name__ == "__main__":
    main()
