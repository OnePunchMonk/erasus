"""
Multi-Task Unlearning — Unlearn from a multi-task model.

Demonstrates targeted unlearning from one task head while preserving
performance on other tasks.

Usage::

    python examples/advanced/multi_task_unlearning.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
import erasus.strategies  # noqa: F401


class MultiTaskModel(nn.Module):
    def __init__(self, in_dim=16, hidden=64, n_tasks=3, n_classes=4):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, n_classes) for _ in range(n_tasks)])
        self.n_tasks = n_tasks

    def forward(self, x, task_id=0):
        h = self.shared(x)
        return self.heads[task_id](h)


def main():
    print("=" * 60)
    print("  Multi-Task Unlearning Example")
    print("=" * 60)

    model = MultiTaskModel()
    n_tasks = model.n_tasks
    print(f"  {n_tasks} tasks, params: {sum(p.numel() for p in model.parameters()):,}")

    # Pre-train all tasks
    print("\n  Phase 1: Pre-training all tasks...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        for task_id in range(n_tasks):
            x = torch.randn(32, 16)
            y = torch.randint(0, 4, (32,))
            loss = F.cross_entropy(model(x, task_id), y)
            opt.zero_grad(); loss.backward(); opt.step()
    print("    ✓ Done")

    # Unlearn task 1 only
    target_task = 1
    print(f"\n  Phase 2: Unlearning task {target_task}...")

    forget = DataLoader(TensorDataset(torch.randn(40, 16), torch.randint(0, 4, (40,))), batch_size=16)
    retain = DataLoader(TensorDataset(torch.randn(160, 16), torch.randint(0, 4, (160,))), batch_size=16)

    unlearner = ErasusUnlearner(
        model=model, strategy="gradient_ascent", selector=None,
        device="cpu", strategy_kwargs={"lr": 1e-3},
    )
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)

    # Evaluate all tasks post-unlearning
    print(f"\n  Phase 3: Post-unlearning evaluation...")
    model.eval()
    test_x = torch.randn(50, 16)
    for t in range(n_tasks):
        with torch.no_grad():
            out = model(test_x, t)
            entropy = -(F.softmax(out, 1) * F.log_softmax(out, 1)).sum(1).mean()
        print(f"    Task {t}: avg entropy={entropy:.4f}")

    print(f"\n  ✓ Done in {result.elapsed_time:.2f}s")
    print("\n✅ Multi-task unlearning complete!")


if __name__ == "__main__":
    main()
