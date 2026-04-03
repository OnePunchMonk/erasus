"""
Unit tests for metrics.

Tests the metric implementations: accuracy, MIA, metric suite, and
the newly added forgetting/efficiency/privacy sub-metrics.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.registry import metric_registry
from erasus.metrics.accuracy import AccuracyMetric
from erasus.metrics.forgetting.knowmem import KnowMemMetric
from erasus.metrics.forgetting.mia_variants import (
    GradNormMIAMetric,
    MinKPlusPlusMetric,
    MinKProbMetric,
    ReferenceMIAMetric,
    ZLibMIAMetric,
)
from erasus.metrics.metric_suite import MetricSuite
from erasus.metrics.privacy.privacy_leakage import PrivacyLeakageMetric
from erasus.metrics.privacy.rag_leakage import RAGLeakageMetric

# Ensure metrics are registered
import erasus.metrics  # noqa: F401


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model():
    return SmallNet()


@pytest.fixture
def forget_loader():
    return DataLoader(
        TensorDataset(torch.randn(16, 8), torch.randint(0, 4, (16,))),
        batch_size=8,
    )


@pytest.fixture
def retain_loader():
    return DataLoader(
        TensorDataset(torch.randn(32, 8), torch.randint(0, 4, (32,))),
        batch_size=8,
    )


class TestMetricRegistry:
    """Verify all expected metrics are registered."""

    def test_accuracy_registered(self):
        assert metric_registry.get("accuracy") is not None

    def test_mia_registered(self):
        assert metric_registry.get("mia") is not None

    def test_perplexity_registered(self):
        assert metric_registry.get("perplexity") is not None

    def test_mink_registered(self):
        assert metric_registry.get("mink") is not None

    def test_mink_pp_registered(self):
        assert metric_registry.get("mink_pp") is not None

    def test_reference_mia_registered(self):
        assert metric_registry.get("reference_mia") is not None

    def test_gradnorm_mia_registered(self):
        assert metric_registry.get("gradnorm_mia") is not None

    def test_zlib_mia_registered(self):
        assert metric_registry.get("zlib_mia") is not None

    def test_privacy_leakage_registered(self):
        assert metric_registry.get("privacy_leakage") is not None

    def test_rag_leakage_registered(self):
        assert metric_registry.get("rag_leakage") is not None

    def test_knowmem_registered(self):
        assert metric_registry.get("knowmem") is not None


class TestAccuracyMetric:
    """Test AccuracyMetric."""

    def test_compute_returns_dict(self, model, forget_loader, retain_loader):
        metric = AccuracyMetric()
        result = metric.compute(
            model=model,
            forget_data=forget_loader,
            retain_data=retain_loader,
        )
        assert isinstance(result, dict)

    def test_compute_contains_accuracy_keys(self, model, forget_loader, retain_loader):
        metric = AccuracyMetric()
        result = metric.compute(
            model=model,
            forget_data=forget_loader,
            retain_data=retain_loader,
        )
        # Should contain some accuracy-related key
        assert len(result) > 0


class TestMetricSuite:
    """Test MetricSuite unified runner."""

    def test_suite_with_string_names(self, model, forget_loader, retain_loader):
        suite = MetricSuite(["accuracy"])
        result = suite.run(model, forget_loader, retain_loader)
        assert isinstance(result, dict)

    def test_suite_with_instances(self, model, forget_loader, retain_loader):
        suite = MetricSuite(metrics=[AccuracyMetric()])
        result = suite.run(model, forget_loader, retain_loader)
        assert isinstance(result, dict)

    def test_suite_empty_metrics(self, model, forget_loader, retain_loader):
        suite = MetricSuite([])
        result = suite.run(model, forget_loader, retain_loader)
        assert isinstance(result, dict)


class TestMinKProbMetric:
    """Test Min-K forgetting metric."""

    def test_compute_returns_expected_keys(self, model, forget_loader, retain_loader):
        metric = MinKProbMetric(k_percent=25.0)
        result = metric.compute(
            model=model,
            forget_data=forget_loader,
            retain_data=retain_loader,
        )

        assert set(result) == {"mink_forget_mean", "mink_retain_mean", "mink_auc"}

    def test_invalid_k_percent_raises(self):
        with pytest.raises(ValueError):
            MinKProbMetric(k_percent=0.0)


class TestMinKPlusPlusMetric:
    """Test Min-K++ forgetting metric."""

    def test_compute_returns_expected_keys(self, model, forget_loader, retain_loader):
        metric = MinKPlusPlusMetric(k_percent=25.0)
        result = metric.compute(
            model=model,
            forget_data=forget_loader,
            retain_data=retain_loader,
        )

        assert set(result) == {"mink_pp_forget_mean", "mink_pp_retain_mean", "mink_pp_auc"}


class TestReferenceMIAMetric:
    """Test reference-model MIA metric."""

    def test_compute_returns_expected_keys(self, model, forget_loader, retain_loader):
        reference_model = SmallNet()
        metric = ReferenceMIAMetric(reference_model=reference_model)
        result = metric.compute(
            model=model,
            forget_data=forget_loader,
            retain_data=retain_loader,
        )

        assert set(result) == {
            "reference_mia_forget_mean",
            "reference_mia_retain_mean",
            "reference_mia_auc",
        }


class TestGradNormMIAMetric:
    """Test gradient-norm MIA metric."""

    def test_compute_returns_expected_keys(self, model, forget_loader, retain_loader):
        metric = GradNormMIAMetric()
        result = metric.compute(
            model=model,
            forget_data=forget_loader,
            retain_data=retain_loader,
        )

        assert set(result) == {
            "gradnorm_mia_forget_mean",
            "gradnorm_mia_retain_mean",
            "gradnorm_mia_auc",
        }


class TestZLibMIAMetric:
    """Test zlib-normalized MIA metric."""

    def test_compute_returns_expected_keys(self, model, forget_loader, retain_loader):
        metric = ZLibMIAMetric()
        result = metric.compute(
            model=model,
            forget_data=forget_loader,
            retain_data=retain_loader,
        )

        assert set(result) == {
            "zlib_mia_forget_mean",
            "zlib_mia_retain_mean",
            "zlib_mia_auc",
        }


class TestKnowMemMetric:
    """Test TOFU-style knowledge memorization probing."""

    def test_compute_returns_expected_keys(self):
        class DummyTokenizer:
            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}

            def decode(self, tokens, skip_special_tokens=True):
                return "alpha remembered"

        class DummyGenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))

            def generate(self, input_ids, attention_mask=None, max_new_tokens=32):
                extra = torch.tensor([[99]], device=input_ids.device)
                return torch.cat([input_ids, extra], dim=1)

        qa_samples = [
            {"question": "Who is Alpha?", "answer": "alpha remembered"},
            {"question": "Who is Beta?", "answer": "alpha remembered"},
        ]
        loader = DataLoader(qa_samples, batch_size=2, shuffle=False)

        metric = KnowMemMetric(match_threshold=0.5)
        result = metric.compute(
            model=DummyGenModel(),
            forget_data=loader,
            retain_data=loader,
            tokenizer=DummyTokenizer(),
        )

        assert set(result) == {"knowmem_forget", "knowmem_retain", "knowmem_gap"}
        assert result["knowmem_forget"] >= 0.5


class TestStandaloneMemorizationModules:
    """Ensure standalone metric modules remain importable."""

    def test_direct_exact_memorization_import(self):
        from erasus.metrics.forgetting.exact_memorization import ExactMemorizationMetric

        metric = ExactMemorizationMetric()
        assert metric.name == "exact_memorization"

    def test_direct_extraction_strength_import(self):
        from erasus.metrics.forgetting.extraction_strength import ExtractionStrengthMetric

        metric = ExtractionStrengthMetric()
        assert metric.name == "extraction_strength"


class TestPrivacyLeakageMetric:
    """Test privacy leakage metric."""

    def test_compute_returns_expected_keys(self, model, forget_loader, retain_loader):
        metric = PrivacyLeakageMetric()
        result = metric.compute(
            model=model,
            forget_data=forget_loader,
            retain_data=retain_loader,
        )

        assert set(result) == {
            "privacy_leakage",
            "privacy_forget_loss",
            "privacy_retain_loss",
        }


class TestRAGLeakageMetric:
    """Test RAG leakage metric."""

    def test_compute_returns_expected_keys(self, model, forget_loader, retain_loader):
        metric = RAGLeakageMetric()
        result = metric.compute(
            model=model,
            forget_data=forget_loader,
            retain_data=retain_loader,
        )

        assert set(result) == {
            "rag_leakage",
            "rag_context_loss",
            "rag_retain_loss",
        }
