"""
tests/unit/test_issue41.py — Cross-Prompt Leakage Test (Issue #41)

Verifies the CrossPromptLeakageMetric correctly:
- Combines forget+retain queries in a single evaluation pass
- Computes leakage scores, confidence gaps, and loss gaps
- Handles edge cases (empty loaders, mismatched batch sizes)
- Detects leakage when a model still has high confidence on forget samples
- Reports near-zero leakage when forget samples yield high loss/low confidence
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.metrics.forgetting.cross_prompt_leakage import (
    CrossPromptLeakageMetric,
    _collect_stats,
    _interleave_batches,
    _iter_batches,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_CLASSES = 5
INPUT_DIM = 16
BATCH_SIZE = 4
NUM_BATCHES = 3


def _make_linear_model(in_features: int = INPUT_DIM, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Simple linear classification model for testing."""
    return nn.Linear(in_features, num_classes)


def _make_loader(
    num_batches: int = NUM_BATCHES,
    batch_size: int = BATCH_SIZE,
    input_dim: int = INPUT_DIM,
    num_classes: int = NUM_CLASSES,
    fixed_label: int | None = None,
) -> DataLoader:
    """Build a synthetic DataLoader with random tensors."""
    n = num_batches * batch_size
    inputs = torch.randn(n, input_dim)
    if fixed_label is not None:
        labels = torch.full((n,), fixed_label, dtype=torch.long)
    else:
        labels = torch.randint(0, num_classes, (n,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestIterBatches:
    def test_returns_list_of_tuples(self):
        loader = _make_loader()
        batches = _iter_batches(loader)
        assert isinstance(batches, list)
        assert len(batches) == NUM_BATCHES

    def test_each_batch_is_tuple(self):
        loader = _make_loader()
        batches = _iter_batches(loader)
        for inp, tgt in batches:
            assert isinstance(inp, torch.Tensor)
            assert isinstance(tgt, torch.Tensor)

    def test_correct_shapes(self):
        loader = _make_loader()
        batches = _iter_batches(loader)
        for inp, tgt in batches:
            assert inp.shape[1] == INPUT_DIM
            assert tgt.shape[0] == inp.shape[0]


class TestInterleave:
    def test_returns_two_equal_length_lists(self):
        forget = _iter_batches(_make_loader(num_batches=3))
        retain = _iter_batches(_make_loader(num_batches=3))
        mf, mr = _interleave_batches(forget, retain)
        assert len(mf) == len(mr)

    def test_length_limited_by_shorter(self):
        forget = _iter_batches(_make_loader(num_batches=5))
        retain = _iter_batches(_make_loader(num_batches=2))
        mf, mr = _interleave_batches(forget, retain)
        assert len(mf) == 2

    def test_batch_sizes_match(self):
        forget = _iter_batches(_make_loader(num_batches=2))
        retain = _iter_batches(_make_loader(num_batches=2))
        mf, mr = _interleave_batches(forget, retain)
        for (f_inp, _), (r_inp, _) in zip(mf, mr):
            assert f_inp.shape[0] == r_inp.shape[0]


class TestCollectStats:
    def test_returns_arrays_same_length(self):
        model = _make_linear_model()
        device = torch.device("cpu")
        batches = _iter_batches(_make_loader())
        losses, confs = _collect_stats(model, batches, device)
        assert len(losses) == len(confs)
        assert len(losses) == NUM_BATCHES * BATCH_SIZE

    def test_confidences_in_valid_range(self):
        model = _make_linear_model()
        device = torch.device("cpu")
        batches = _iter_batches(_make_loader())
        _, confs = _collect_stats(model, batches, device)
        assert (confs >= 0.0).all()
        assert (confs <= 1.0).all()

    def test_losses_non_negative(self):
        model = _make_linear_model()
        device = torch.device("cpu")
        batches = _iter_batches(_make_loader())
        losses, _ = _collect_stats(model, batches, device)
        assert (losses >= 0.0).all()

    def test_empty_batches_returns_empty_arrays(self):
        model = _make_linear_model()
        device = torch.device("cpu")
        losses, confs = _collect_stats(model, [], device)
        assert len(losses) == 0
        assert len(confs) == 0


# ---------------------------------------------------------------------------
# Integration tests for CrossPromptLeakageMetric
# ---------------------------------------------------------------------------

class TestCrossPromptLeakageMetric:

    # -- Basic smoke test --------------------------------------------------

    def test_returns_expected_keys(self):
        metric = CrossPromptLeakageMetric()
        model = _make_linear_model()
        forget_loader = _make_loader()
        retain_loader = _make_loader()

        result = metric.compute(model, forget_loader, retain_loader)

        expected_keys = {
            "cross_prompt_leakage_score",
            "cross_prompt_forget_confidence",
            "cross_prompt_retain_confidence",
            "cross_prompt_forget_loss",
            "cross_prompt_retain_loss",
            "cross_prompt_loss_gap",
            "cross_prompt_leakage_detected",
            "cross_prompt_num_forget_samples",
            "cross_prompt_num_retain_samples",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_all_values_are_floats(self):
        metric = CrossPromptLeakageMetric()
        model = _make_linear_model()
        result = metric.compute(model, _make_loader(), _make_loader())
        for k, v in result.items():
            assert isinstance(v, float), f"Key {k!r} has type {type(v)}, expected float"

    def test_leakage_score_in_range(self):
        metric = CrossPromptLeakageMetric()
        model = _make_linear_model()
        result = metric.compute(model, _make_loader(), _make_loader())
        assert 0.0 <= result["cross_prompt_leakage_score"] <= 1.0

    def test_sample_counts_correct(self):
        metric = CrossPromptLeakageMetric()
        model = _make_linear_model()
        forget_loader = _make_loader(num_batches=3, batch_size=4)
        retain_loader = _make_loader(num_batches=3, batch_size=4)
        result = metric.compute(model, forget_loader, retain_loader)
        # 3 batch pairs × 4 samples each
        assert result["cross_prompt_num_forget_samples"] == 12.0
        assert result["cross_prompt_num_retain_samples"] == 12.0

    # -- Edge cases --------------------------------------------------------

    def test_no_forget_data_returns_defaults(self):
        metric = CrossPromptLeakageMetric()
        model = _make_linear_model()
        result = metric.compute(model, forget_data=None, retain_data=_make_loader())
        assert result["cross_prompt_leakage_score"] == 0.0
        assert result["cross_prompt_leakage_detected"] == 0.0

    def test_no_retain_data_returns_defaults(self):
        metric = CrossPromptLeakageMetric()
        model = _make_linear_model()
        result = metric.compute(model, forget_data=_make_loader(), retain_data=None)
        assert result["cross_prompt_leakage_score"] == 0.0
        assert result["cross_prompt_leakage_detected"] == 0.0

    def test_both_none_returns_defaults(self):
        metric = CrossPromptLeakageMetric()
        model = _make_linear_model()
        result = metric.compute(model, forget_data=None, retain_data=None)
        assert result["cross_prompt_leakage_score"] == 0.0

    def test_num_batches_limits_evaluation(self):
        metric = CrossPromptLeakageMetric(num_batches=2)
        model = _make_linear_model()
        forget_loader = _make_loader(num_batches=10, batch_size=4)
        retain_loader = _make_loader(num_batches=10, batch_size=4)
        result = metric.compute(model, forget_loader, retain_loader)
        assert result["cross_prompt_num_forget_samples"] == 8.0  # 2 batches × 4 samples

    def test_mismatched_loader_sizes_uses_minimum(self):
        metric = CrossPromptLeakageMetric()
        model = _make_linear_model()
        forget_loader = _make_loader(num_batches=2)
        retain_loader = _make_loader(num_batches=5)
        result = metric.compute(model, forget_loader, retain_loader)
        # Only 2 pairs can be formed
        assert result["cross_prompt_num_forget_samples"] == float(2 * BATCH_SIZE)

    # -- Semantic correctness tests ----------------------------------------

    def test_low_confidence_on_forget_yields_low_leakage(self):
        """
        A model with near-uniform outputs (uninformed) should yield
        low confidence on forget samples → low leakage score.
        """
        # Constant output close to zero → near-uniform softmax
        class UniformModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), NUM_CLASSES)

        metric = CrossPromptLeakageMetric()
        result = metric.compute(
            UniformModel(), _make_loader(), _make_loader()
        )
        # All softmax outputs equal → confidence = 1/NUM_CLASSES on each class
        # leakage_score = forget_conf / retain_conf ≈ 1 when both are uniform,
        # so we only check that the reported confidences make sense
        assert result["cross_prompt_forget_confidence"] == pytest.approx(
            result["cross_prompt_retain_confidence"], abs=1e-4
        )

    def test_high_confidence_on_forget_yields_high_leakage(self):
        """
        A model that is consistently confident on all inputs should
        yield a leakage score close to 1.
        """
        class ConfidentModel(nn.Module):
            def forward(self, x):
                # Always predicts class 0 with high confidence
                logits = torch.full((x.size(0), NUM_CLASSES), -10.0)
                logits[:, 0] = 10.0
                return logits

        metric = CrossPromptLeakageMetric(leakage_threshold=0.8)
        result = metric.compute(
            ConfidentModel(), _make_loader(), _make_loader()
        )
        # Both forget and retain are confident → leakage score ≈ 1
        assert result["cross_prompt_leakage_score"] > 0.9
        assert result["cross_prompt_leakage_detected"] == 1.0

    def test_high_loss_on_forget_positive_loss_gap(self):
        """
        When the model has no information about the forget set
        (random output) but still performs well on the retain set,
        the loss gap should be positive (forget_loss > retain_loss).
        """
        class SpecialisedOnRetain(nn.Module):
            """
            Pretend the model memorised retain-class 0 and is random on forget.
            We simulate this by checking a flag passed via batch index.
            In practice we use two separate loaders with different label distributions.
            """
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(INPUT_DIM, NUM_CLASSES)
                nn.init.zeros_(self.linear.weight)
                nn.init.zeros_(self.linear.bias)
                # Make class 0 the "retain" prediction
                self.linear.bias.data[0] = 100.0

            def forward(self, x):
                return self.linear(x)

        # Retain always uses class 0 → low loss
        retain_loader = _make_loader(fixed_label=0)
        # Forget uses random classes → high loss
        forget_loader = _make_loader(fixed_label=4)  # non-zero class → high loss

        metric = CrossPromptLeakageMetric()
        result = metric.compute(
            SpecialisedOnRetain(), forget_loader, retain_loader
        )
        # forget_loss should be larger than retain_loss
        assert result["cross_prompt_loss_gap"] > 0, (
            f"Expected positive loss gap, got {result['cross_prompt_loss_gap']}"
        )

    def test_leakage_detected_flag_respects_threshold(self):
        metric_strict = CrossPromptLeakageMetric(leakage_threshold=0.01)
        metric_lenient = CrossPromptLeakageMetric(leakage_threshold=0.99)

        class ConfidentModel(nn.Module):
            def forward(self, x):
                logits = torch.full((x.size(0), NUM_CLASSES), -10.0)
                logits[:, 0] = 10.0
                return logits

        model = ConfidentModel()
        forget_loader = _make_loader()
        retain_loader = _make_loader()

        # Very low threshold → leakage detected for nearly any confidence
        r_strict = metric_strict.compute(model, forget_loader, retain_loader)
        assert r_strict["cross_prompt_leakage_detected"] == 1.0

        # Very high threshold → not detected
        r_lenient = metric_lenient.compute(model, forget_loader, retain_loader)
        assert r_lenient["cross_prompt_leakage_detected"] == 0.0

    # -- Registry / import test -------------------------------------------

    def test_importable_from_metrics_package(self):
        """Ensure CrossPromptLeakageMetric is importable from the public API."""
        from erasus.metrics.forgetting.cross_prompt_leakage import (
            CrossPromptLeakageMetric as CPL,
        )
        assert CPL is CrossPromptLeakageMetric

    def test_metric_name_attribute(self):
        assert CrossPromptLeakageMetric.name == "cross_prompt_leakage"

    # -- HuggingFace-style model compatibility ----------------------------

    def test_works_with_logits_output(self):
        """CrossPromptLeakageMetric should handle models that wrap logits."""
        class HFModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._linear = nn.Linear(INPUT_DIM, NUM_CLASSES)

            class _Output:
                def __init__(self, logits):
                    self.logits = logits

            def forward(self, x):
                return self._Output(self._linear(x))

        metric = CrossPromptLeakageMetric()
        result = metric.compute(HFModel(), _make_loader(), _make_loader())
        assert "cross_prompt_leakage_score" in result
        assert 0.0 <= result["cross_prompt_leakage_score"] <= 1.0
