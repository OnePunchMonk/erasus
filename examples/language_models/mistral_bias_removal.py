"""
Mistral Bias Removal — Unlearn biased associations from a Mistral model.

Demonstrates using the LLM unlearner with gradient ascent to remove
gender/demographic biases from a language model.

Usage::

    python examples/language_models/mistral_bias_removal.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.llm_unlearner import LLMUnlearner
import erasus.strategies  # noqa: F401
import erasus.selectors   # noqa: F401


class TinyLM(nn.Module):
    """Minimal LM for demonstration."""

    def __init__(self, vocab_size=256, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.layers = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, vocab_size)
        self.config = type("C", (), {"model_type": "mistral", "vocab_size": vocab_size})()

    def forward(self, x):
        return self.head(self.layers(self.emb(x).mean(dim=1)))


def main():
    print("=" * 60)
    print("  Mistral Bias Removal Example")
    print("=" * 60)

    device = "cpu"
    model = TinyLM().to(device)

    # Create synthetic "biased" forget set and clean retain set
    forget = DataLoader(
        TensorDataset(torch.randint(0, 256, (50, 32)), torch.randint(0, 256, (50,))),
        batch_size=16,
    )
    retain = DataLoader(
        TensorDataset(torch.randint(0, 256, (200, 32)), torch.randint(0, 256, (200,))),
        batch_size=16,
    )

    unlearner = LLMUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector="random",
        device=device,
        strategy_kwargs={"lr": 1e-3},
    )

    print("\n  Running bias unlearning...")
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5, prune_ratio=0.3)

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ Bias removal complete!")


if __name__ == "__main__":
    main()
