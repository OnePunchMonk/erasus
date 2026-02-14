"""
Tests for Sprint F — Research Innovations & Ecosystem.

Covers:
- Selectors: CoresetQualityAnalyzer, ActiveLearningSelector, WeightedFusionSelector
- Strategy: VisionTextSplitStrategy
- Metrics: ErasusBenchmark, CLIPScoreMetric, ExtractionAttackMetric, BLEUMetric,
           ROUGEMetric, InceptionScoreMetric, DownstreamTaskMetric,
           EpsilonDeltaMetric, PrivacyAuditMetric
- Visualization: ActivationVisualizer, InfluenceMapVisualizer, CrossModalVisualizer
- Unlearner: FederatedUnlearner
"""

import pytest
import numpy as np

# Force non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def simple_model():
    """Simple classifier for testing."""
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    return model


@pytest.fixture
def data_loaders():
    """Forget and retain data loaders."""
    torch.manual_seed(42)
    n = 40
    X = torch.randn(n, 16)
    y = torch.randint(0, 4, (n,))
    ds = TensorDataset(X, y)
    forget = DataLoader(ds, batch_size=10)
    retain = DataLoader(ds, batch_size=10)
    return forget, retain


# ──────────────────────────────────────────────────────────────
# Selectors
# ──────────────────────────────────────────────────────────────

class TestCoresetQualityAnalyzer:
    def test_analyse_basic(self, simple_model, data_loaders):
        from erasus.selectors.quality_metrics import CoresetQualityAnalyzer

        forget_loader, _ = data_loaders
        analyzer = CoresetQualityAnalyzer(model=simple_model)
        results = analyzer.analyse(
            full_loader=forget_loader,
            coreset_indices=[0, 5, 10, 15, 20],
        )
        assert "coverage" in results
        assert "diversity" in results
        assert "representativeness" in results
        assert "redundancy" in results
        assert "n_coreset" in results
        assert results["n_coreset"] == 5.0

    def test_analyse_with_influence_scores(self, simple_model, data_loaders):
        from erasus.selectors.quality_metrics import CoresetQualityAnalyzer

        forget_loader, _ = data_loaders
        analyzer = CoresetQualityAnalyzer(model=simple_model)
        scores = np.random.randn(40)
        results = analyzer.analyse(
            full_loader=forget_loader,
            coreset_indices=[0, 1, 2, 3, 4],
            influence_scores=scores,
        )
        assert "influence_gini" in results
        assert "influence_top10_share" in results


class TestActiveLearningSelector:
    def test_entropy_selection(self, simple_model, data_loaders):
        from erasus.selectors.learning_based.active_learning import ActiveLearningSelector

        forget_loader, _ = data_loaders
        selector = ActiveLearningSelector(method="entropy")
        indices = selector.select(simple_model, forget_loader, k=5)
        assert len(indices) == 5
        assert all(isinstance(i, int) for i in indices)

    def test_margin_selection(self, simple_model, data_loaders):
        from erasus.selectors.learning_based.active_learning import ActiveLearningSelector

        forget_loader, _ = data_loaders
        selector = ActiveLearningSelector(method="margin")
        indices = selector.select(simple_model, forget_loader, k=3)
        assert len(indices) == 3

    def test_invalid_method(self):
        from erasus.selectors.learning_based.active_learning import ActiveLearningSelector
        with pytest.raises(ValueError):
            ActiveLearningSelector(method="invalid")


class TestWeightedFusionSelector:
    def test_basic_fusion(self, simple_model, data_loaders):
        from erasus.selectors.ensemble.weighted_fusion import WeightedFusionSelector
        from erasus.selectors.random_selector import RandomSelector

        forget_loader, _ = data_loaders
        s1 = RandomSelector()
        s2 = RandomSelector()
        fused = WeightedFusionSelector(selectors=[s1, s2], weights=[0.7, 0.3])
        indices = fused.select(simple_model, forget_loader, k=5)
        assert len(indices) == 5

    def test_empty_selectors(self):
        from erasus.selectors.ensemble.weighted_fusion import WeightedFusionSelector
        with pytest.raises(ValueError):
            WeightedFusionSelector(selectors=[])


# ──────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────

class TestVisionTextSplitStrategy:
    def test_registered(self):
        from erasus.core.registry import strategy_registry
        cls = strategy_registry.get("vision_text_split")
        assert cls is not None

    def test_unlearn_basic(self, simple_model, data_loaders):
        from erasus.strategies.vlm_specific.vision_text_split import VisionTextSplitStrategy

        forget_loader, retain_loader = data_loaders
        strategy = VisionTextSplitStrategy(vision_epochs=1, text_epochs=1)
        model, forget_losses, retain_losses = strategy.unlearn(
            model=simple_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=2,
        )
        assert isinstance(model, nn.Module)


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

class TestErasusBenchmark:
    def test_evaluate_single_method(self, simple_model, data_loaders, tmp_path):
        from erasus.metrics.benchmarks import ErasusBenchmark

        forget_loader, retain_loader = data_loaders
        bench = ErasusBenchmark(
            name="test_bench",
            output_dir=str(tmp_path / "bench"),
        )
        results = bench.evaluate(
            model=simple_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            method_name="gradient_ascent",
        )
        assert "forgetting" in results
        assert "utility" in results
        assert "efficiency" in results

    def test_compare_methods(self, simple_model, data_loaders, tmp_path):
        from erasus.metrics.benchmarks import ErasusBenchmark

        forget_loader, retain_loader = data_loaders
        bench = ErasusBenchmark(output_dir=str(tmp_path / "bench2"))

        bench.evaluate(simple_model, forget_loader, retain_loader, method_name="A")
        bench.evaluate(simple_model, forget_loader, retain_loader, method_name="B")

        comparison = bench.compare()
        assert len(comparison["methods"]) == 2

    def test_latex_output(self, simple_model, data_loaders, tmp_path):
        from erasus.metrics.benchmarks import ErasusBenchmark

        forget_loader, retain_loader = data_loaders
        bench = ErasusBenchmark(output_dir=str(tmp_path / "bench3"))
        bench.evaluate(simple_model, forget_loader, retain_loader, method_name="test")

        latex = bench.to_latex_table()
        assert r"\begin{table}" in latex

    def test_json_output(self, simple_model, data_loaders, tmp_path):
        from erasus.metrics.benchmarks import ErasusBenchmark

        forget_loader, retain_loader = data_loaders
        bench = ErasusBenchmark(output_dir=str(tmp_path / "bench4"))
        bench.evaluate(simple_model, forget_loader, retain_loader, method_name="test")

        json_str = bench.to_json(str(tmp_path / "bench4" / "out.json"))
        assert "test" in json_str


class TestExtractionAttackMetric:
    def test_compute(self, simple_model, data_loaders):
        from erasus.metrics.forgetting.extraction_attack import ExtractionAttackMetric

        forget_loader, retain_loader = data_loaders
        metric = ExtractionAttackMetric()
        results = metric.compute(
            model=simple_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
        )
        assert "extraction_rate" in results
        assert "privacy_score" in results
        assert 0.0 <= results["privacy_score"] <= 1.0


class TestBLEUMetric:
    def test_registered(self):
        from erasus.core.registry import metric_registry
        cls = metric_registry.get("bleu")
        assert cls is not None

    def test_corpus_bleu_computation(self):
        from erasus.metrics.utility.bleu import BLEUMetric
        metric = BLEUMetric(max_n=2)
        score = metric._corpus_bleu(
            references=[["the", "cat", "sat", "on", "the", "mat"]],
            hypotheses=[["the", "cat", "sat", "on", "the", "mat"]],
        )
        assert score > 0.9  # Perfect match should be high


class TestROUGEMetric:
    def test_registered(self):
        from erasus.core.registry import metric_registry
        cls = metric_registry.get("rouge")
        assert cls is not None

    def test_rouge_n(self):
        from erasus.metrics.utility.rouge import ROUGEMetric
        r1 = ROUGEMetric._rouge_n(
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "cat", "sat", "on", "the", "mat"],
            n=1,
        )
        assert r1 == 1.0

    def test_rouge_l(self):
        from erasus.metrics.utility.rouge import ROUGEMetric
        rl = ROUGEMetric._rouge_l(
            ["the", "quick", "brown", "fox"],
            ["the", "lazy", "brown", "dog"],
        )
        assert 0.0 < rl < 1.0


class TestInceptionScoreMetric:
    def test_compute(self, simple_model, data_loaders):
        from erasus.metrics.utility.inception_score import InceptionScoreMetric

        forget_loader, _ = data_loaders
        metric = InceptionScoreMetric(n_splits=2)
        results = metric.compute(
            model=simple_model,
            forget_loader=forget_loader,
        )
        assert "inception_score_mean" in results
        assert results["inception_score_mean"] > 0


class TestDownstreamTaskMetric:
    def test_compute(self, simple_model, data_loaders):
        from erasus.metrics.utility.downstream_tasks import DownstreamTaskMetric

        forget_loader, retain_loader = data_loaders
        metric = DownstreamTaskMetric()
        results = metric.compute(
            model=simple_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
        )
        assert "forget_accuracy" in results
        assert "retain_accuracy" in results


class TestEpsilonDeltaMetric:
    def test_compute(self, simple_model, data_loaders):
        from erasus.metrics.privacy.epsilon_delta import EpsilonDeltaMetric

        forget_loader, retain_loader = data_loaders
        metric = EpsilonDeltaMetric(
            noise_multiplier=1.0, sample_rate=0.01, n_steps=10
        )
        results = metric.compute(
            model=simple_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
        )
        assert "epsilon" in results
        assert "delta" in results
        assert results["epsilon"] > 0

    def test_rdp_accounting(self):
        from erasus.metrics.privacy.epsilon_delta import EpsilonDeltaMetric
        eps = EpsilonDeltaMetric._compute_epsilon_rdp(
            noise_multiplier=1.0, sample_rate=0.01, n_steps=100, delta=1e-5
        )
        assert eps > 0


class TestPrivacyAuditMetric:
    def test_compute(self, simple_model, data_loaders):
        from erasus.metrics.privacy.privacy_audit import PrivacyAuditMetric

        forget_loader, retain_loader = data_loaders
        metric = PrivacyAuditMetric()
        results = metric.compute(
            model=simple_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
        )
        assert "loss_mia_auc" in results
        assert "confidence_mia_auc" in results
        assert "entropy_mia_auc" in results
        assert "privacy_score" in results

    def test_without_retain(self, simple_model, data_loaders):
        from erasus.metrics.privacy.privacy_audit import PrivacyAuditMetric

        forget_loader, _ = data_loaders
        metric = PrivacyAuditMetric()
        results = metric.compute(
            model=simple_model, forget_loader=forget_loader
        )
        assert results["audit_status"] == 0.0


class TestCLIPScoreMetric:
    def test_registered(self):
        from erasus.core.registry import metric_registry
        cls = metric_registry.get("clip_score")
        assert cls is not None


# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────

class TestActivationVisualizer:
    def test_extract_activations(self, simple_model):
        from erasus.visualization.activation import ActivationVisualizer

        # nn.Sequential names layers "0", "1", "2" — specify explicitly
        viz = ActivationVisualizer(model=simple_model, target_layers=["0", "2"])
        inputs = torch.randn(2, 16)
        acts = viz.extract_activations(inputs)
        assert isinstance(acts, dict)
        assert len(acts) > 0


class TestInfluenceMapVisualizer:
    def test_plot_ranking(self, simple_model, tmp_path):
        from erasus.visualization.influence_maps import InfluenceMapVisualizer

        viz = InfluenceMapVisualizer(model=simple_model)
        scores = np.random.randn(50)
        fig = viz.plot_influence_ranking(
            scores, top_k=10, save_path=str(tmp_path / "ranking.png")
        )
        assert fig is not None

    def test_plot_distribution(self, simple_model, tmp_path):
        from erasus.visualization.influence_maps import InfluenceMapVisualizer

        viz = InfluenceMapVisualizer(model=simple_model)
        scores = np.random.randn(100)
        fig = viz.plot_influence_distribution(
            scores, save_path=str(tmp_path / "dist.png")
        )
        assert fig is not None

    def test_plot_heatmap(self, simple_model, tmp_path):
        from erasus.visualization.influence_maps import InfluenceMapVisualizer

        viz = InfluenceMapVisualizer(model=simple_model)
        matrix = np.random.randn(10, 10)
        fig = viz.plot_influence_heatmap(
            matrix, save_path=str(tmp_path / "heatmap.png")
        )
        assert fig is not None


class TestCrossModalVisualizer:
    def test_instantiation(self, simple_model):
        from erasus.visualization.cross_modal import CrossModalVisualizer

        viz = CrossModalVisualizer(model=simple_model)
        assert viz.model is simple_model


# ──────────────────────────────────────────────────────────────
# Unlearner
# ──────────────────────────────────────────────────────────────

class TestFederatedUnlearner:
    def test_basic_unlearning(self, simple_model, data_loaders):
        from erasus.unlearners.federated_unlearner import FederatedUnlearner

        forget_loader, retain_loader = data_loaders
        unlearner = FederatedUnlearner(
            model=simple_model,
            strategy="gradient_ascent",
            n_clients=2,
            communication_rounds=1,
            local_epochs=1,
        )
        result = unlearner.fit(
            forget_data=forget_loader,
            retain_data=retain_loader,
            epochs=1,
        )
        assert result is not None
        assert hasattr(result, "model")

    def test_fedavg_aggregation(self, simple_model):
        from erasus.unlearners.federated_unlearner import FederatedUnlearner
        import copy

        unlearner = FederatedUnlearner(model=simple_model, n_clients=3)
        clients = [copy.deepcopy(simple_model) for _ in range(3)]
        aggregated = unlearner._fedavg(clients)
        assert isinstance(aggregated, nn.Module)

    def test_forget_client(self, simple_model, data_loaders):
        from erasus.unlearners.federated_unlearner import FederatedUnlearner
        import copy

        forget_loader, retain_loader = data_loaders
        unlearner = FederatedUnlearner(model=simple_model, n_clients=3)
        # Initialise client models
        unlearner._client_models = [copy.deepcopy(simple_model) for _ in range(3)]
        result = unlearner.forget_client(
            client_id=1,
            client_data=forget_loader,
            retain_data=retain_loader,
        )
        assert result is not None
