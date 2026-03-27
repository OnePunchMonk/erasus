"""
Tests for all 8 improvements from issue #8.

1. Strategy composition and chaining (StrategyPipeline)
2. Coreset quality metrics and diagnostics
3. Streaming/incremental unlearning API
4. Automatic strategy selection (AutoStrategy)
5. Benchmark result persistence and comparison
6. Privacy-aware evaluation integration
7. Real model integration tests (separate file)
8. CLI integration (tested via argparse parsing)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---- Helpers ----

def _tiny_model(in_dim: int = 16, n_classes: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, n_classes),
    )


def _make_loader(n: int = 100, in_dim: int = 16, n_classes: int = 4, bs: int = 16) -> DataLoader:
    x = torch.randn(n, in_dim)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=bs)


# ======================================================================
# 1. StrategyPipeline
# ======================================================================


class TestStrategyPipeline:
    def test_import(self):
        from erasus.core.strategy_pipeline import StrategyPipeline
        assert StrategyPipeline is not None

    def test_pipeline_from_strings(self):
        from erasus.core.strategy_pipeline import StrategyPipeline
        pipeline = StrategyPipeline([
            ("gradient_ascent", {"epochs": 2, "lr": 1e-3}),
            ("gradient_ascent", {"epochs": 1, "lr": 1e-3}),
        ])
        assert len(pipeline) == 2

    def test_pipeline_runs(self):
        from erasus.core.strategy_pipeline import StrategyPipeline
        pipeline = StrategyPipeline([
            ("gradient_ascent", {"epochs": 1, "lr": 1e-3}),
        ])
        model = _tiny_model()
        forget_loader = _make_loader(50)
        retain_loader = _make_loader(100)

        result_model, f_losses, r_losses = pipeline.unlearn(
            model, forget_loader, retain_loader, epochs=1,
        )
        assert result_model is not None
        assert len(f_losses) >= 1

    def test_pipeline_chains_losses(self):
        from erasus.core.strategy_pipeline import StrategyPipeline
        pipeline = StrategyPipeline([
            ("gradient_ascent", {"epochs": 2, "lr": 1e-3}),
            ("gradient_ascent", {"epochs": 2, "lr": 1e-3}),
        ])
        model = _tiny_model()
        _, f_losses, _ = pipeline.unlearn(
            model, _make_loader(50), _make_loader(100), epochs=2,
        )
        # Should have losses from both stages
        assert len(f_losses) >= 4

    def test_pipeline_repr(self):
        from erasus.core.strategy_pipeline import StrategyPipeline
        pipeline = StrategyPipeline(["gradient_ascent", "gradient_ascent"])
        assert "StrategyPipeline" in repr(pipeline)

    def test_pipeline_accepted_by_erasus_unlearner(self):
        from erasus import ErasusUnlearner
        from erasus.core.strategy_pipeline import StrategyPipeline

        pipeline = StrategyPipeline([
            ("gradient_ascent", {"epochs": 1, "lr": 1e-3}),
        ])
        model = _tiny_model()
        unlearner = ErasusUnlearner(model=model, strategy=pipeline)
        assert unlearner.strategy is pipeline

    def test_pipeline_bad_stage_raises(self):
        from erasus.core.strategy_pipeline import StrategyPipeline
        with pytest.raises(TypeError):
            StrategyPipeline([12345])


# ======================================================================
# 2. Coreset Quality Diagnostics
# ======================================================================


class TestCoresetDiagnostics:
    def test_redundancy_with_scores(self):
        from erasus.core.coreset import Coreset

        dataset = TensorDataset(torch.randn(100, 16), torch.randint(0, 4, (100,)))
        coreset = Coreset.from_indices(
            dataset, indices=list(range(20)),
            scores=[float(i) for i in range(20)],
        )
        r = coreset.redundancy()
        assert 0.0 <= r <= 1.0

    def test_redundancy_uniform_scores(self):
        from erasus.core.coreset import Coreset

        dataset = TensorDataset(torch.randn(100, 16), torch.randint(0, 4, (100,)))
        coreset = Coreset.from_indices(
            dataset, indices=list(range(20)),
            scores=[1.0] * 20,
        )
        # All same score = max redundancy
        assert coreset.redundancy() == 1.0

    def test_overlap_with(self):
        from erasus.core.coreset import Coreset

        dataset = TensorDataset(torch.randn(100, 16), torch.randint(0, 4, (100,)))
        a = Coreset.from_indices(dataset, [0, 1, 2, 3, 4])
        b = Coreset.from_indices(dataset, [3, 4, 5, 6, 7])
        overlap = a.overlap_with(b)
        # Jaccard: |{3,4}| / |{0..7}| = 2/8 = 0.25
        assert abs(overlap - 0.25) < 1e-6

    def test_diagnostics_without_model(self):
        from erasus.core.coreset import Coreset

        dataset = TensorDataset(torch.randn(100, 16), torch.randint(0, 4, (100,)))
        coreset = Coreset.from_indices(dataset, list(range(10)))
        diag = coreset.diagnostics()
        assert diag["size"] == 10
        assert "redundancy" in diag
        assert "coverage" not in diag  # no model passed

    def test_diagnostics_with_model(self):
        from erasus.core.coreset import Coreset

        model = _tiny_model()
        dataset = TensorDataset(torch.randn(50, 16), torch.randint(0, 4, (50,)))
        coreset = Coreset.from_indices(dataset, list(range(10)))
        diag = coreset.diagnostics(model=model)
        assert "coverage" in diag
        assert 0.0 <= diag["coverage"] <= 1.0

    def test_compare_coresets(self):
        from erasus.core.coreset import Coreset

        dataset = TensorDataset(torch.randn(100, 16), torch.randint(0, 4, (100,)))
        a = Coreset.from_indices(dataset, list(range(10)))
        b = Coreset.from_indices(dataset, list(range(5, 15)))
        results = Coreset.compare(a, b)
        assert len(results) == 2
        assert "overlaps" in results[0]


# ======================================================================
# 3. Streaming / Incremental Unlearning
# ======================================================================


class TestIncrementalUnlearning:
    def test_incremental_fit_basic(self):
        from erasus.data.datasets.unlearning import UnlearningDataset
        from erasus.unlearners.continual_unlearner import ContinualUnlearner

        base_dataset = TensorDataset(
            torch.randn(200, 16), torch.randint(0, 4, (200,)),
        )
        ds = UnlearningDataset(base_dataset, forget_indices=[0, 1, 2, 3, 4])
        model = _tiny_model()

        unlearner = ContinualUnlearner(
            model=model, strategy="gradient_ascent",
            strategy_kwargs={"lr": 1e-3}, base_epochs=1,
        )
        result = unlearner.incremental_fit(ds, batch_size=16)
        assert result.model is not None
        assert len(result.deletion_requests) == 1

    def test_incremental_fit_skips_already_processed(self):
        from erasus.data.datasets.unlearning import UnlearningDataset
        from erasus.unlearners.continual_unlearner import ContinualUnlearner

        base_dataset = TensorDataset(
            torch.randn(200, 16), torch.randint(0, 4, (200,)),
        )
        ds = UnlearningDataset(base_dataset, forget_indices=[0, 1, 2])
        model = _tiny_model()

        unlearner = ContinualUnlearner(
            model=model, strategy="gradient_ascent",
            strategy_kwargs={"lr": 1e-3}, base_epochs=1,
        )
        result1 = unlearner.incremental_fit(ds, batch_size=16)

        # No new indices — should skip
        result2 = unlearner.incremental_fit(
            ds, previous_result=result1, batch_size=16,
        )
        assert result2.metadata.get("skipped") is True

    def test_incremental_fit_processes_new_indices(self):
        from erasus.data.datasets.unlearning import UnlearningDataset
        from erasus.unlearners.continual_unlearner import ContinualUnlearner

        base_dataset = TensorDataset(
            torch.randn(200, 16), torch.randint(0, 4, (200,)),
        )
        ds = UnlearningDataset(base_dataset, forget_indices=[0, 1, 2])
        model = _tiny_model()

        unlearner = ContinualUnlearner(
            model=model, strategy="gradient_ascent",
            strategy_kwargs={"lr": 1e-3}, base_epochs=1,
        )
        result1 = unlearner.incremental_fit(ds, batch_size=16)

        # Add new forget indices
        ds.mark_forget([10, 11, 12])
        result2 = unlearner.incremental_fit(
            ds, previous_result=result1, batch_size=16,
        )
        # Should have processed new indices
        assert len(result2.deletion_requests) == 2


# ======================================================================
# 4. AutoStrategy
# ======================================================================


class TestAutoStrategy:
    def test_import_and_register(self):
        from erasus.core.registry import strategy_registry
        assert "auto" in strategy_registry.list()

    def test_auto_strategy_runs(self):
        from erasus.strategies.auto_strategy import AutoStrategy

        model = _tiny_model()
        strategy = AutoStrategy(model_type="classifier", goal="balanced")
        result_model, f_losses, r_losses = strategy.unlearn(
            model, _make_loader(50), _make_loader(100), epochs=1,
        )
        assert result_model is not None

    def test_auto_strategy_fast(self):
        from erasus.strategies.auto_strategy import AutoStrategy

        model = _tiny_model()
        strategy = AutoStrategy(model_type="classifier", goal="fast")
        inner = strategy._select_strategy(model, _make_loader(50), _make_loader(100))
        # Small model + fast goal should pick gradient_ascent or neuron_pruning
        assert inner is not None

    def test_auto_via_erasus_unlearner(self):
        from erasus import ErasusUnlearner

        model = _tiny_model()
        unlearner = ErasusUnlearner(model=model, strategy="auto")
        result = unlearner.fit(
            forget_data=_make_loader(50),
            retain_data=_make_loader(100),
            epochs=1,
        )
        assert result.model is not None


# ======================================================================
# 5. Benchmark Persistence and Comparison
# ======================================================================


class TestBenchmarkPersistence:
    def _make_report(self):
        from erasus.evaluation.benchmark_protocol import (
            BenchmarkReport, MetricResult,
        )
        return BenchmarkReport(
            protocol="general",
            gold_standard="retrain",
            n_runs=3,
            confidence_level=0.95,
            metric_results={
                "forget_quality": MetricResult(
                    name="forget_quality",
                    values=[0.05, 0.08, 0.06],
                    pass_threshold=0.1,
                    direction="lower_is_better",
                ),
                "model_utility": MetricResult(
                    name="model_utility",
                    values=[0.85, 0.82, 0.88],
                    pass_threshold=0.8,
                    direction="higher_is_better",
                ),
            },
            elapsed_time=1.5,
            metadata={"strategy": "gradient_ascent"},
        )

    def test_to_dict(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["protocol"] == "general"
        assert "forget_quality" in d["metrics"]
        assert d["verdict"] == "PASS"

    def test_save_and_load(self, tmp_path):
        from erasus.evaluation.benchmark_protocol import BenchmarkReport

        report = self._make_report()
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()

        loaded = BenchmarkReport.load(path)
        assert loaded.protocol == "general"
        assert loaded.verdict == report.verdict
        assert len(loaded.metric_results) == 2

    def test_compare(self):
        from erasus.evaluation.benchmark_protocol import BenchmarkReport

        r1 = self._make_report()
        r1.metadata["strategy"] = "strategy_a"
        r2 = self._make_report()
        r2.metadata["strategy"] = "strategy_b"

        output = BenchmarkReport.compare(r1, r2)
        assert "strategy_a" in output
        assert "strategy_b" in output
        assert "VERDICT" in output

    def test_leaderboard(self):
        from erasus.evaluation.benchmark_protocol import BenchmarkReport

        reports = []
        for name in ["alpha", "beta", "gamma"]:
            r = self._make_report()
            r.metadata["strategy"] = name
            reports.append(r)

        lb = BenchmarkReport.leaderboard(reports)
        assert "Leaderboard" in lb
        assert "alpha" in lb


# ======================================================================
# 6. Privacy-Aware Evaluation
# ======================================================================


class TestPrivacyEvaluation:
    def test_include_privacy_adds_metrics(self):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        benchmark = UnlearningBenchmark(protocol="general", include_privacy=True)
        metric_names = [m.name for m in benchmark._metrics]
        assert "epsilon_budget" in metric_names
        assert "certified_removal" in metric_names

    def test_privacy_metrics_compute(self):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        model = _tiny_model()
        benchmark = UnlearningBenchmark(
            protocol="general", include_privacy=True,
        )
        report = benchmark.evaluate(
            unlearned_model=model,
            forget_data=_make_loader(50),
            retain_data=_make_loader(100),
        )
        assert "epsilon_budget" in report.metric_results
        assert "certified_removal" in report.metric_results
        # Epsilon should be a finite number
        eps = report.metric_results["epsilon_budget"].mean
        assert math.isfinite(eps)


# ======================================================================
# 8. CLI Integration (argparse validation)
# ======================================================================


class TestCLIArgs:
    def test_unlearn_parser_has_coreset_args(self):
        import argparse
        from erasus.cli.unlearn import add_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        add_parser(sub)

        args = parser.parse_args([
            "unlearn", "--config", "test.yaml",
            "--coreset-from", "influence", "--coreset-k", "100",
        ])
        assert args.coreset_from == "influence"
        assert args.coreset_k == 100

    def test_unlearn_parser_has_validation_args(self):
        import argparse
        from erasus.cli.unlearn import add_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        add_parser(sub)

        args = parser.parse_args([
            "unlearn", "--config", "test.yaml",
            "--validate-every", "2", "--early-stop-patience", "3",
        ])
        assert args.validate_every == 2
        assert args.early_stop_patience == 3

    def test_benchmark_parser_has_protocol_args(self):
        import argparse
        from erasus.cli.benchmark import add_benchmark_args

        parser = argparse.ArgumentParser()
        add_benchmark_args(parser)

        args = parser.parse_args([
            "--protocol", "tofu", "--gold-model", "retrained.pt",
            "--n-runs", "5", "--confidence-level", "0.99",
        ])
        assert args.protocol == "tofu"
        assert args.gold_model == "retrained.pt"
        assert args.n_runs == 5
        assert args.confidence_level == 0.99

    def test_evaluate_parser_has_protocol_args(self):
        import argparse
        from erasus.cli.evaluate import add_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        add_parser(sub)

        args = parser.parse_args([
            "evaluate", "--protocol", "tofu",
            "--include-privacy", "--n-runs", "3",
        ])
        assert args.protocol == "tofu"
        assert args.include_privacy is True
        assert args.n_runs == 3
