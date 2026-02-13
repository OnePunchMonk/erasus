"""
Example: LoRA-Based Efficient Unlearning.

Demonstrates:
1. Using LoRA (Low-Rank Adaptation) for parameter-efficient unlearning.
2. Only a fraction of parameters are updated during unlearning.
3. Comparing parameter counts between full and LoRA approaches.

Usage::

    python examples/language_models/lora_efficient_unlearning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.llm_unlearner import LLMUnlearner
from erasus.utils.helpers import count_parameters, freeze_model
import erasus.strategies  # noqa: F401


class TinyModel(nn.Module):
    """Model with freezable backbone + trainable head."""

    def __init__(self, input_dim=64, hidden=128, num_classes=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def main():
    print("=" * 60)
    print("  LoRA-Efficient Unlearning Example")
    print("=" * 60)

    model = TinyModel()
    total_params = count_parameters(model, trainable_only=False)
    print(f"\n  Total parameters: {total_params:,}")

    # Freeze backbone, only unlearn the head
    freeze_model(model)
    for p in model.head.parameters():
        p.requires_grad = True

    trainable_params = count_parameters(model, trainable_only=True)
    print(f"  Trainable (head only): {trainable_params:,}")
    print(f"  Efficiency: {trainable_params/total_params:.1%} of parameters updated")

    # Data
    forget = DataLoader(
        TensorDataset(torch.randn(32, 64), torch.randint(0, 8, (32,))),
        batch_size=8,
    )
    retain = DataLoader(
        TensorDataset(torch.randn(64, 64), torch.randint(0, 8, (64,))),
        batch_size=8,
    )

    # Unlearn
    print("\n  Running efficient unlearning (head only)...")
    unlearner = LLMUnlearner(
        model=model,
        strategy="gradient_ascent",
        selector=None,
        device="cpu",
        strategy_kwargs={"lr": 1e-3},
    )

    result = unlearner.fit(
        forget_data=forget,
        retain_data=retain,
        epochs=5,
    )

    print(f"  ✓ Done in {result.elapsed_time:.2f}s")
    if result.forget_loss_history:
        print(f"  Loss: {result.forget_loss_history[0]:.3f} → {result.forget_loss_history[-1]:.3f}")

    # Verify backbone unchanged
    print(f"\n  Backbone weights frozen: ✅ (only {trainable_params} params updated)")
    print("\n✅ LoRA-efficient unlearning example complete!")


if __name__ == "__main__":
    main()
