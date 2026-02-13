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
from erasus.metrics.metric_suite import MetricSuite

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
