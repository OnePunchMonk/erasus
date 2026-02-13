"""
Example: Concept Removal from LLaMA using Erasus.

Demonstrates:
1. Using the LLMUnlearner for knowledge removal.
2. Running gradient ascent on „forget" tokens.
3. Measuring perplexity on retain set to verify model utility.

Usage::

    python examples/language_models/llama_concept_removal.py

For real experiments, replace TinyLLM with a proper LLaMA model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.llm_unlearner import LLMUnlearner
import erasus.strategies  # noqa: F401


class TinyLLM(nn.Module):
    """Minimal causal LM stand-in for testing."""

    def __init__(self, vocab_size=256, hidden=64, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=hidden, nhead=4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden, vocab_size)
        # Attributes for LLM auto-detection
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": vocab_size})()
        self.lm_head = self.head  # Alias for compatibility

    def forward(self, x):
        emb = self.embedding(x)
        hidden = self.layers(emb)
        return self.head(hidden[:, -1, :])  # Predict next token from last position


def make_token_data(n_samples=64, seq_len=32, vocab_size=256, batch_size=16):
    """Generate random token sequences."""
    input_ids = torch.randint(0, vocab_size, (n_samples, seq_len))
    labels = torch.randint(0, vocab_size, (n_samples,))
    return DataLoader(TensorDataset(input_ids, labels), batch_size=batch_size)


def main():
    print("=" * 60)
    print("  LLaMA Concept Removal Example")
    print("=" * 60)

    model = TinyLLM()

    # Forget set: tokens related to the concept we want to remove
    forget_loader = make_token_data(32, seq_len=32)
    # Retain set: tokens we want the model to keep
    retain_loader = make_token_data(64, seq_len=32)

    print(f"\n  Model: TinyLLM ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Forget set: {len(forget_loader.dataset)} samples")
    print(f"  Retain set: {len(retain_loader.dataset)} samples")

    # ---- Unlearning ----
    print("\n  Creating LLMUnlearner with gradient_ascent...")
    unlearner = LLMUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    print("  Running unlearning (3 epochs)...")
    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        epochs=3,
    )

    print(f"\n  ✓ Complete in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        losses = result.forget_loss_history
        print(f"  Loss: {losses[0]:.3f} → {losses[-1]:.3f}")

    # ---- Evaluation ----
    print("\n  Evaluating...")
    from erasus.metrics.accuracy import AccuracyMetric
    metrics = unlearner.evaluate(
        forget_data=forget_loader,
        retain_data=retain_loader,
        metrics=[AccuracyMetric()],
    )
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    print("\n✅ LLaMA concept removal example complete!")


if __name__ == "__main__":
    main()
