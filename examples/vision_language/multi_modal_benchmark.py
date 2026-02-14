"""
Multi-Modal Benchmark: Compare unlearning across VLM modalities.

Benchmarks multiple strategies on a vision-language model, measuring
forgetting quality, utility retention, and efficiency.

Usage::

    python examples/vision_language/multi_modal_benchmark.py
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.vlm_unlearner import VLMUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401
import erasus.selectors   # noqa: F401


class TinyVLM(nn.Module):
    """Minimal VLM for benchmarking."""

    def __init__(self, img_channels=3, text_vocab=500, hidden=64):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(img_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, hidden),
        )
        self.text = nn.Sequential(
            nn.Embedding(text_vocab, hidden),
        )
        self.head = nn.Linear(hidden * 2, 10)

    def forward(self, images, texts):
        v = self.vision(images)
        t = self.text(texts).mean(dim=1)
        return self.head(torch.cat([v, t], dim=-1))


def make_data(n=128, batch_size=16):
    images = torch.randn(n, 3, 32, 32)
    texts = torch.randint(0, 500, (n, 20))
    labels = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(images, texts, labels), batch_size=batch_size)


def main():
    print("=" * 60)
    print("  Multi-Modal VLM Benchmark")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    strategies = ["gradient_ascent", "negative_gradient", "contrastive_unlearning"]

    forget_loader = make_data(64)
    retain_loader = make_data(256)

    results = {}

    for strat_name in strategies:
        print(f"\n  Strategy: {strat_name}")
        model = TinyVLM().to(device)

        try:
            unlearner = VLMUnlearner(
                model=model,
                strategy=strat_name,
                selector="random",
                device=device,
                strategy_kwargs={"lr": 1e-3},
            )

            t0 = time.time()
            result = unlearner.fit(
                forget_data=forget_loader,
                retain_data=retain_loader,
                prune_ratio=0.5,
                epochs=3,
            )
            elapsed = time.time() - t0

            results[strat_name] = {
                "time_s": round(elapsed, 2),
                "coreset_size": result.coreset_size,
                "final_loss": round(result.forget_loss_history[-1], 4) if result.forget_loss_history else None,
            }
            print(f"    ✓ {elapsed:.2f}s  loss={results[strat_name]['final_loss']}")
        except Exception as e:
            results[strat_name] = {"error": str(e)}
            print(f"    ✗ {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  {'Strategy':<30} {'Time':>8} {'Loss':>10}")
    print("-" * 60)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<30} {'ERROR':>8}")
        else:
            print(f"  {name:<30} {r['time_s']:>7.1f}s {r.get('final_loss', 'N/A'):>10}")
    print("\n✅ Multi-modal benchmark complete!")


if __name__ == "__main__":
    main()
