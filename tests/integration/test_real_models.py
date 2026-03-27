"""
Integration tests using real (small) HuggingFace models.

These tests are gated behind the ``real_models`` marker and require
network access + optional dependencies (transformers, datasets).

Run with::

    python -m pytest tests/integration/test_real_models.py -v -m real_models

Skip in CI by default — only run when explicitly requested.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Mark all tests in this module
pytestmark = pytest.mark.real_models


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


requires_transformers = pytest.mark.skipif(
    not _has_transformers(),
    reason="transformers not installed",
)


# ======================================================================
# GPT-2 (small) — LLM unlearning
# ======================================================================


@requires_transformers
class TestGPT2Unlearning:
    """Test unlearning on a real GPT-2 model."""

    @pytest.fixture(scope="class")
    def gpt2_model(self):
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        return model

    @pytest.fixture
    def gpt2_loaders(self):
        """Create synthetic token-id loaders shaped for GPT-2."""
        vocab_size = 50257
        seq_len = 32
        n_forget, n_retain = 40, 80

        forget_ids = torch.randint(0, vocab_size, (n_forget, seq_len))
        retain_ids = torch.randint(0, vocab_size, (n_retain, seq_len))

        forget_loader = DataLoader(
            TensorDataset(forget_ids, forget_ids),
            batch_size=8,
        )
        retain_loader = DataLoader(
            TensorDataset(retain_ids, retain_ids),
            batch_size=8,
        )
        return forget_loader, retain_loader

    def test_gradient_ascent_on_gpt2(self, gpt2_model, gpt2_loaders):
        """Gradient ascent runs on a real transformer without errors."""
        from erasus.strategies.gradient_methods.gradient_ascent import (
            GradientAscentStrategy,
        )

        forget_loader, retain_loader = gpt2_loaders
        model = gpt2_model

        # Use only 1 epoch for speed
        strategy = GradientAscentStrategy(lr=1e-5)

        # GPT-2 returns CausalLMOutput, not raw logits — wrap to expose
        # a simple (input, label) interface
        class _Wrapper(nn.Module):
            def __init__(self, lm):
                super().__init__()
                self.lm = lm

            def forward(self, input_ids, labels=None):
                out = self.lm(input_ids=input_ids, labels=labels)
                return out.logits

        wrapped = _Wrapper(model)
        result_model, f_losses, r_losses = strategy.unlearn(
            model=wrapped,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=1,
        )
        assert result_model is not None
        assert len(f_losses) >= 1

    def test_auto_strategy_on_gpt2(self, gpt2_model, gpt2_loaders):
        """AutoStrategy correctly picks an LLM-appropriate strategy."""
        from erasus.strategies.auto_strategy import AutoStrategy

        strategy = AutoStrategy(model_type="llm", goal="fast")
        inner = strategy._select_strategy(
            gpt2_model, gpt2_loaders[0], gpt2_loaders[1]
        )
        # Should pick SSD for fast LLM unlearning
        assert inner.__class__.__name__ == "SelectiveSynapticDampeningStrategy"


# ======================================================================
# ViT-tiny (via timm or manual) — Vision model unlearning
# ======================================================================


class TestVisionModelUnlearning:
    """Test unlearning on a small vision architecture."""

    @pytest.fixture
    def vision_model(self):
        """Create a small ResNet-like model (no downloads needed)."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        return model

    @pytest.fixture
    def vision_loaders(self):
        n_forget, n_retain = 50, 100
        forget_x = torch.randn(n_forget, 3, 32, 32)
        forget_y = torch.randint(0, 10, (n_forget,))
        retain_x = torch.randn(n_retain, 3, 32, 32)
        retain_y = torch.randint(0, 10, (n_retain,))

        forget_loader = DataLoader(
            TensorDataset(forget_x, forget_y), batch_size=16,
        )
        retain_loader = DataLoader(
            TensorDataset(retain_x, retain_y), batch_size=16,
        )
        return forget_loader, retain_loader

    def test_fisher_forgetting_on_vision(self, vision_model, vision_loaders):
        """Fisher forgetting runs on a real-architecture vision model."""
        from erasus.strategies.gradient_methods.fisher_forgetting import (
            FisherForgettingStrategy,
        )

        forget_loader, retain_loader = vision_loaders
        strategy = FisherForgettingStrategy(lr=1e-3)
        model, f_losses, r_losses = strategy.unlearn(
            model=vision_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )
        assert model is not None
        assert len(f_losses) >= 1

    def test_pipeline_on_vision(self, vision_model, vision_loaders):
        """StrategyPipeline chains two strategies on a vision model."""
        from erasus.core.strategy_pipeline import StrategyPipeline

        pipeline = StrategyPipeline([
            ("gradient_ascent", {"epochs": 1, "lr": 1e-3}),
            ("fisher_forgetting", {"epochs": 1, "lr": 1e-3}),
        ])
        forget_loader, retain_loader = vision_loaders
        model, f_losses, r_losses = pipeline.unlearn(
            model=vision_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=1,
        )
        assert model is not None
        # Pipeline should produce losses from both stages
        assert len(f_losses) >= 2

    def test_coreset_diagnostics_on_vision(self, vision_model, vision_loaders):
        """Coreset diagnostics run on a vision model."""
        from erasus.core.coreset import Coreset

        forget_loader, _ = vision_loaders
        dataset = forget_loader.dataset
        indices = list(range(min(20, len(dataset))))

        coreset = Coreset.from_indices(
            dataset=dataset,
            indices=indices,
            scores=[float(i) for i in range(len(indices))],
        )

        diag = coreset.diagnostics(model=vision_model)
        assert "coverage" in diag
        assert "redundancy" in diag
        assert 0.0 <= diag["coverage"] <= 1.0
        assert 0.0 <= diag["redundancy"] <= 1.0


# ======================================================================
# Benchmark protocol with real evaluation
# ======================================================================


class TestBenchmarkProtocol:
    """Test the UnlearningBenchmark protocol system."""

    @pytest.fixture
    def simple_setup(self):
        in_dim, n_classes = 32, 10
        model = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )
        n_forget, n_retain = 100, 200
        forget_loader = DataLoader(
            TensorDataset(
                torch.randn(n_forget, in_dim),
                torch.randint(0, n_classes, (n_forget,)),
            ),
            batch_size=32,
        )
        retain_loader = DataLoader(
            TensorDataset(
                torch.randn(n_retain, in_dim),
                torch.randint(0, n_classes, (n_retain,)),
            ),
            batch_size=32,
        )
        return model, forget_loader, retain_loader

    def test_protocol_evaluate(self, simple_setup):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        model, forget_loader, retain_loader = simple_setup
        benchmark = UnlearningBenchmark(protocol="general", n_runs=2)
        report = benchmark.evaluate(model, forget_loader, retain_loader)
        assert report.verdict in ("PASS", "PARTIAL", "FAIL")
        assert len(report.metric_results) >= 3

    def test_protocol_with_privacy(self, simple_setup):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        model, forget_loader, retain_loader = simple_setup
        benchmark = UnlearningBenchmark(
            protocol="general",
            include_privacy=True,
        )
        report = benchmark.evaluate(model, forget_loader, retain_loader)
        # Should have standard metrics + 2 privacy metrics
        assert "epsilon_budget" in report.metric_results
        assert "certified_removal" in report.metric_results

    def test_report_persistence(self, simple_setup, tmp_path):
        from erasus.evaluation.benchmark_protocol import (
            BenchmarkReport,
            UnlearningBenchmark,
        )

        model, forget_loader, retain_loader = simple_setup
        benchmark = UnlearningBenchmark(protocol="tofu")
        report = benchmark.evaluate(model, forget_loader, retain_loader)

        # Save and load
        path = tmp_path / "report.json"
        report.save(path)
        loaded = BenchmarkReport.load(path)

        assert loaded.protocol == report.protocol
        assert loaded.verdict == report.verdict
        assert set(loaded.metric_results.keys()) == set(report.metric_results.keys())

    def test_report_comparison(self, simple_setup):
        from erasus.evaluation.benchmark_protocol import (
            BenchmarkReport,
            UnlearningBenchmark,
        )

        model, forget_loader, retain_loader = simple_setup

        report_a = UnlearningBenchmark(protocol="general").evaluate(
            model, forget_loader, retain_loader,
        )
        report_a.metadata["strategy"] = "gradient_ascent"

        report_b = UnlearningBenchmark(protocol="general").evaluate(
            model, forget_loader, retain_loader,
        )
        report_b.metadata["strategy"] = "fisher_forgetting"

        comparison = BenchmarkReport.compare(report_a, report_b)
        assert "gradient_ascent" in comparison
        assert "fisher_forgetting" in comparison

    def test_leaderboard(self, simple_setup):
        from erasus.evaluation.benchmark_protocol import (
            BenchmarkReport,
            UnlearningBenchmark,
        )

        model, forget_loader, retain_loader = simple_setup
        reports = []
        for name in ["strategy_a", "strategy_b"]:
            r = UnlearningBenchmark(protocol="general").evaluate(
                model, forget_loader, retain_loader,
            )
            r.metadata["strategy"] = name
            reports.append(r)

        lb = BenchmarkReport.leaderboard(reports)
        assert "Leaderboard" in lb
        assert "strategy_a" in lb
