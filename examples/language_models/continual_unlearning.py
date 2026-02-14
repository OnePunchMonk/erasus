"""
Continual Unlearning — Sequential unlearning of multiple concepts.

Demonstrates how to chain multiple unlearning requests while
preserving model utility, simulating GDPR "right to be forgotten"
sequential deletion requests.

Usage::

    python examples/language_models/continual_unlearning.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.llm_unlearner import LLMUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401
import erasus.selectors   # noqa: F401


class SmallLM(nn.Module):
    def __init__(self, vocab=256, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, vocab)
        self.config = type("C", (), {"model_type": "llama", "vocab_size": vocab})()

    def forward(self, x):
        return self.head(self.net(self.emb(x).mean(1)))


def make_loader(n, seq_len=32, vocab=256, bs=16):
    return DataLoader(TensorDataset(torch.randint(0, vocab, (n, seq_len)), torch.randint(0, vocab, (n,))), batch_size=bs)


def main():
    print("=" * 60)
    print("  Continual Unlearning Example")
    print("=" * 60)

    model = SmallLM()
    retain = make_loader(400)

    # Simulate 5 sequential deletion requests
    n_requests = 5

    for i in range(n_requests):
        forget = make_loader(30)  # Each request asks to forget 30 samples
        print(f"\n  Request {i+1}/{n_requests}: unlearning 30 samples...")

        unlearner = LLMUnlearner(
            model=model,
            strategy="gradient_ascent",
            selector=None,
            device="cpu",
            strategy_kwargs={"lr": 1e-3},
        )

        result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)
        model = unlearner.model  # carry forward

        loss_str = f"{result.forget_loss_history[-1]:.4f}" if result.forget_loss_history else "N/A"
        print(f"    ✓ {result.elapsed_time:.2f}s  forget_loss={loss_str}")

    print(f"\n  Completed {n_requests} sequential unlearning requests.")
    print("✅ Continual unlearning complete!")


if __name__ == "__main__":
    main()
