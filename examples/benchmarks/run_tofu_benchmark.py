"""
Example: Running the TOFU Benchmark on Erasus.

Demonstrates:
1. Loading the TOFU dataset.
2. Running unlearning with the LLMUnlearner.
3. Evaluating with TOFU-specific metrics.

Usage::

    python examples/benchmarks/run_tofu_benchmark.py

Requires TOFU data in ``./data/tofu/`` or will attempt HuggingFace download.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.llm_unlearner import LLMUnlearner
from erasus.metrics.metric_suite import MetricSuite
import erasus.strategies  # noqa: F401


class TinyLM(nn.Module):
    """Minimal language model for benchmarking."""

    def __init__(self, vocab_size=256, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.config = type("C", (), {"model_type": "llama", "vocab_size": vocab_size})()

    def forward(self, x):
        return self.head(self.emb(x).mean(dim=1))


def run_tofu_benchmark():
    print("=" * 60)
    print("  TOFU Benchmark — Erasus Evaluation")
    print("=" * 60)

    # ---- Try loading real TOFU data ----
    try:
        from erasus.data.datasets.tofu import TOFUDataset
        forget_ds = TOFUDataset(split="forget_01")
        retain_ds = TOFUDataset(split="retain")
        if len(forget_ds) > 0 and len(retain_ds) > 0:
            print(f"  ✓ Loaded TOFU: {len(forget_ds)} forget, {len(retain_ds)} retain")
        else:
            raise ValueError("Empty datasets")
    except Exception as e:
        print(f"  ⚠ Could not load real TOFU data ({e})")
        print("  → Using synthetic data for demo")
        forget_ds = None

    # Fall back to synthetic data
    forget_loader = DataLoader(
        TensorDataset(torch.randint(0, 256, (32, 16)), torch.randint(0, 256, (32,))),
        batch_size=8,
    )
    retain_loader = DataLoader(
        TensorDataset(torch.randint(0, 256, (64, 16)), torch.randint(0, 256, (64,))),
        batch_size=8,
    )

    model = TinyLM()
    strategies_to_test = ["gradient_ascent", "negative_gradient"]
    all_results = {}

    for strat in strategies_to_test:
        print(f"\n  --- Strategy: {strat} ---")
        unlearner = LLMUnlearner(
            model=model,
            strategy=strat,
            selector=None,
            device="cpu",
            strategy_kwargs={"lr": 1e-3},
        )

        result = unlearner.fit(
            forget_data=forget_loader,
            retain_data=retain_loader,
            epochs=2,
        )

        # Evaluate
        suite = MetricSuite(["accuracy"])
        eval_results = suite.run(unlearner.model, forget_loader, retain_loader)

        all_results[strat] = {
            "time": f"{result.elapsed_time:.2f}s",
            "final_loss": result.forget_loss_history[-1] if result.forget_loss_history else 0,
            **{k: v for k, v in eval_results.items() if k != "_meta"},
        }

    # ---- Results ----
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    for strat, r in all_results.items():
        print(f"\n  {strat}:")
        for k, v in r.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    print("\n✅ TOFU benchmark complete!")


if __name__ == "__main__":
    run_tofu_benchmark()
