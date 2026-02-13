"""
Example: Comparing Coreset Selection Strategies for CLIP Unlearning.

This script demonstrates:
1. Loading a CLIP model (small config for testing).
2. Running unlearning with different coreset selectors.
3. Comparing unlearning quality across selectors.

Usage::

    python examples/vision_language/clip_coreset_comparison.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import CLIPModel, CLIPConfig

from erasus.unlearners.vlm_unlearner import VLMUnlearner
from erasus.metrics.metric_suite import MetricSuite

# Ensure strategies/selectors are registered
import erasus.strategies  # noqa: F401
import erasus.selectors   # noqa: F401


class DummyVLMDataset(Dataset):
    """Synthetic image–text pairs for testing."""

    def __init__(self, n_samples: int = 40):
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        input_ids = torch.randint(0, 1000, (77,))
        return image, input_ids


def main():
    # ---- 1. Create a tiny CLIP model ----
    print("=" * 60)
    print("  CLIP Coreset Comparison Example")
    print("=" * 60)

    config = CLIPConfig(
        text_config_dict={
            "vocab_size": 1000, "hidden_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 4,
        },
        vision_config_dict={
            "image_size": 224, "patch_size": 32, "hidden_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 4,
        },
    )
    model = CLIPModel(config)

    # ---- 2. Create data ----
    forget_loader = DataLoader(DummyVLMDataset(40), batch_size=8)
    retain_loader = DataLoader(DummyVLMDataset(80), batch_size=8)

    # ---- 3. Run unlearning with different selectors ----
    selectors = ["random", "gradient_norm"]
    results = {}

    for selector_name in selectors:
        print(f"\n--- Selector: {selector_name} ---")

        unlearner = VLMUnlearner(
            model=model,
            strategy="contrastive_unlearning",
            selector=selector_name,
            device="cpu",
            strategy_kwargs={"lr": 1e-4},
        )

        result = unlearner.fit(
            forget_data=forget_loader,
            retain_data=retain_loader,
            prune_ratio=0.5,
            epochs=2,
        )

        results[selector_name] = {
            "coreset_size": result.coreset_size,
            "compression": f"{result.compression_ratio:.1%}",
            "elapsed": f"{result.elapsed_time:.2f}s",
            "final_loss": result.forget_loss_history[-1] if result.forget_loss_history else "N/A",
        }

    # ---- 4. Print comparison ----
    print("\n" + "=" * 60)
    print("  RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Selector':<20} {'Coreset':<10} {'Compression':<12} {'Time':<10} {'Final Loss':<12}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['coreset_size']:<10} {r['compression']:<12} {r['elapsed']:<10} {r['final_loss']:<12}")

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
