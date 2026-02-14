"""
BERT Feature Unlearning — Remove learned features from BERT.

Demonstrates feature-level unlearning using Fisher forgetting on a
simplified BERT-like encoder model.

Usage::

    python examples/language_models/bert_feature_unlearning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.llm_unlearner import LLMUnlearner
import erasus.strategies  # noqa: F401


class TinyBERT(nn.Module):
    """Minimal BERT-like encoder for demonstration."""

    def __init__(self, vocab_size=256, hidden=64, n_layers=2, n_classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(hidden, n_classes)
        self.config = type("C", (), {"model_type": "bert", "vocab_size": vocab_size})()

    def forward(self, x):
        h = self.encoder(self.emb(x))
        return self.classifier(h[:, 0])  # CLS token


def main():
    print("=" * 60)
    print("  BERT Feature Unlearning Example")
    print("=" * 60)

    model = TinyBERT()
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    forget = DataLoader(
        TensorDataset(torch.randint(0, 256, (40, 32)), torch.randint(0, 4, (40,))),
        batch_size=16,
    )
    retain = DataLoader(
        TensorDataset(torch.randint(0, 256, (160, 32)), torch.randint(0, 4, (160,))),
        batch_size=16,
    )

    # Use Fisher forgetting to selectively remove features
    unlearner = LLMUnlearner(
        model=model,
        strategy="fisher_forgetting",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3, "fisher_weight": 0.5},
    )

    print("\n  Running feature unlearning with Fisher Forgetting...")
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5)

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Forget loss: {result.forget_loss_history[0]:.4f} → {result.forget_loss_history[-1]:.4f}")
    print("\n✅ Feature unlearning complete!")


if __name__ == "__main__":
    main()
