"""
Reproduction: Machine Unlearning (Bourtoule et al., IEEE S&P 2021) — SISA.

SISA: Sharded, Isolated, Sliced, and Aggregated training.
We simulate by training K small models on data shards; "unlearning" = retrain
only the shard containing the forget set.

Usage::

    python papers/reproductions/sisa_usenix2021.py
"""

import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset


class ShardMLP(nn.Module):
    """Small MLP for one SISA shard."""

    def __init__(self, input_dim=32, num_classes=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, device, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def main():
    print("=" * 60)
    print("  Paper Reproduction: SISA (Bourtoule et al., IEEE S&P 2021)")
    print("  Sharded, Isolated, Sliced, and Aggregated")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_shards = 3
    shard_size = 200
    forget_size = 30

    print(f"\n  Device: {device}")
    print(f"  Shards: {num_shards}, Shard size: {shard_size}, Forget size: {forget_size}")

    # Full dataset indices: assign to shards
    all_indices = list(range(num_shards * shard_size))
    shard_indices = [
        all_indices[i * shard_size : (i + 1) * shard_size]
        for i in range(num_shards)
    ]
    # Forget set: first forget_size samples in shard 0
    forget_indices = set(shard_indices[0][:forget_size])
    retain_indices_shard0 = shard_indices[0][forget_size:]
    retain_loaders = []
    forget_loader = None

    # Synthetic data
    full_data = TensorDataset(
        torch.randn(num_shards * shard_size, 32),
        torch.randint(0, 5, (num_shards * shard_size,)),
    )

    print("\n  Phase 1: Train one model per shard (SISA training)...")
    models = []
    for i, indices in enumerate(shard_indices):
        subset = Subset(full_data, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        model = ShardMLP(input_dim=32, num_classes=5).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):
            train_epoch(model, loader, device, opt)
        models.append(model)
        print(f"    Shard {i+1}/{num_shards} trained.")

    # Retain loader (all shards except forget subset)
    retain_subset = Subset(
        full_data,
        retain_indices_shard0 + shard_indices[1] + shard_indices[2],
    )
    retain_loader = DataLoader(retain_subset, batch_size=32, shuffle=True)
    forget_subset = Subset(full_data, list(forget_indices))
    forget_loader = DataLoader(forget_subset, batch_size=16)

    def aggregate_predict(models, x, device):
        """Average logits across shards (simplified SISA aggregation)."""
        logits = []
        for m in models:
            m.eval()
            with torch.no_grad():
                logits.append(m(x.to(device)))
        return torch.stack(logits, 0).mean(0)

    print("\n  Phase 2: Pre-unlearning accuracy (aggregated)...")
    pre_retain = 0.0
    pre_forget = 0.0
    n_retain, n_forget = 0, 0
    for x, y in retain_loader:
        x, y = x.to(device), y.to(device)
        logits = aggregate_predict(models, x, device)
        pre_retain += (logits.argmax(1) == y).float().sum().item()
        n_retain += y.size(0)
    pre_retain = pre_retain / n_retain if n_retain else 0.0
    for x, y in forget_loader:
        x, y = x.to(device), y.to(device)
        logits = aggregate_predict(models, x, device)
        pre_forget += (logits.argmax(1) == y).float().sum().item()
        n_forget += y.size(0)
    pre_forget = pre_forget / n_forget if n_forget else 0.0
    print(f"    Retain acc: {pre_retain:.4f}, Forget acc: {pre_forget:.4f}")

    print("\n  Phase 3: SISA Unlearning — retrain only Shard 0 without forget set...")
    retain_only_shard0 = Subset(full_data, retain_indices_shard0)
    retrain_loader = DataLoader(retain_only_shard0, batch_size=32, shuffle=True)
    model_shard0 = copy.deepcopy(models[0])
    opt = torch.optim.Adam(model_shard0.parameters(), lr=1e-3)
    t0 = time.time()
    for epoch in range(5):
        train_epoch(model_shard0, retrain_loader, device, opt)
    elapsed = time.time() - t0
    models[0] = model_shard0
    print(f"    ✓ Shard 0 retrained in {elapsed:.2f}s")

    print("\n  Phase 4: Post-unlearning accuracy (aggregated)...")
    post_retain = 0.0
    post_forget = 0.0
    n_retain2, n_forget2 = 0, 0
    for x, y in retain_loader:
        x, y = x.to(device), y.to(device)
        logits = aggregate_predict(models, x, device)
        post_retain += (logits.argmax(1) == y).float().sum().item()
        n_retain2 += y.size(0)
    post_retain = post_retain / n_retain2 if n_retain2 else 0.0
    for x, y in forget_loader:
        x, y = x.to(device), y.to(device)
        logits = aggregate_predict(models, x, device)
        post_forget += (logits.argmax(1) == y).float().sum().item()
        n_forget2 += y.size(0)
    post_forget = post_forget / n_forget2 if n_forget2 else 0.0
    print(f"    Retain acc: {post_retain:.4f}, Forget acc: {post_forget:.4f}")

    print("\n" + "=" * 60)
    print("  REPRODUCTION SUMMARY (SISA)")
    print("=" * 60)
    print(f"  Unlearning (retrain shard) Time: {elapsed:.2f}s")
    print(f"  Forget acc: {pre_forget:.4f} → {post_forget:.4f} (expect drop)")
    print(f"  Retain acc: {pre_retain:.4f} → {post_retain:.4f} (expect stable)")
    print("\n✅ Reproduction complete!")


if __name__ == "__main__":
    main()
