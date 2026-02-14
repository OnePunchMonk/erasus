"""
Federated Unlearning — Unlearn a client from a federated model.

Demonstrates the FederatedUnlearner API for removing a client's
contribution from a federated learning model without full retraining.

Usage::

    python examples/advanced/federated_unlearning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.federated_unlearner import FederatedUnlearner
import erasus.strategies  # noqa: F401


class FedModel(nn.Module):
    def __init__(self, in_dim=16, hidden=64, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def make_client_data(n=100, dim=16, n_classes=4, bs=16):
    return DataLoader(TensorDataset(torch.randn(n, dim), torch.randint(0, n_classes, (n,))), batch_size=bs)


def main():
    print("=" * 60)
    print("  Federated Unlearning Example")
    print("=" * 60)

    n_clients = 5
    global_model = FedModel()

    # Create client data
    client_data = {i: make_client_data() for i in range(n_clients)}

    print(f"  {n_clients} clients, model params: {sum(p.numel() for p in global_model.parameters()):,}")

    # Phase 1: Simulate federated training
    print("\n  Phase 1: Federated training (simplified)...")
    optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01)
    global_model.train()
    for epoch in range(3):
        for client_id, loader in client_data.items():
            for x, y in loader:
                loss = nn.functional.cross_entropy(global_model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    print("    ✓ Training complete")

    # Phase 2: Forget client 2
    forget_client_id = 2
    print(f"\n  Phase 2: Unlearning client {forget_client_id}...")

    fed_unlearner = FederatedUnlearner(
        model=global_model,
        strategy="gradient_ascent",
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    result = fed_unlearner.forget_client(
        client_id=forget_client_id,
        client_data=client_data[forget_client_id],
        retain_data=client_data[0],  # Use another client's data as retain
        epochs=3,
    )

    print(f"  ✓ Client {forget_client_id} unlearned in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ Federated unlearning complete!")


if __name__ == "__main__":
    main()
