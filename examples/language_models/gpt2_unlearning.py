"""
Example: GPT-2 Unlearning with Erasus.

Demonstrates:
1. Using the LLMUnlearner with different strategies.
2. Comparing gradient_ascent vs negative_gradient approaches.
3. Measuring forget/retain performance.

Usage::

    python examples/language_models/gpt2_unlearning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.llm_unlearner import LLMUnlearner
import erasus.strategies  # noqa: F401


class TinyGPT(nn.Module):
    """Minimal GPT-style model for testing."""

    def __init__(self, vocab_size=128, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, batch_first=True,
        )
        self.head = nn.Linear(hidden, vocab_size)
        self.config = type("Cfg", (), {"model_type": "gpt2", "vocab_size": vocab_size})()

    def forward(self, x):
        return self.head(self.transformer(self.embedding(x))[:, -1, :])


def main():
    print("=" * 60)
    print("  GPT-2 Unlearning Example")
    print("=" * 60)

    model = TinyGPT()
    print(f"  Model: TinyGPT ({sum(p.numel() for p in model.parameters()):,} params)")

    # Synthetic data
    forget = DataLoader(
        TensorDataset(torch.randint(0, 128, (24, 16)), torch.randint(0, 128, (24,))),
        batch_size=8,
    )
    retain = DataLoader(
        TensorDataset(torch.randint(0, 128, (48, 16)), torch.randint(0, 128, (48,))),
        batch_size=8,
    )

    # Strategy comparison
    strategies = ["gradient_ascent", "negative_gradient"]
    results = {}

    for strat_name in strategies:
        print(f"\n--- Strategy: {strat_name} ---")
        unlearner = LLMUnlearner(
            model=model,
            strategy=strat_name,
            selector=None,
            device="cpu",
            strategy_kwargs={"lr": 5e-4},
        )

        result = unlearner.fit(
            forget_data=forget,
            retain_data=retain,
            epochs=2,
        )

        results[strat_name] = {
            "time": f"{result.elapsed_time:.2f}s",
            "final_loss": f"{result.forget_loss_history[-1]:.3f}" if result.forget_loss_history else "N/A",
        }

    # Comparison table
    print("\n" + "=" * 50)
    print(f"{'Strategy':<25} {'Time':<10} {'Final Loss'}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<25} {r['time']:<10} {r['final_loss']}")

    print("\n✅ GPT-2 unlearning example complete!")


if __name__ == "__main__":
    main()
