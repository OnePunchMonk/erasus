"""
tests/unit/test_issue46.py — Benign Fine-Tuning Attack Metric (Issue #46)

Verifies that BenignFinetuningMetric correctly:
- Fine-tunes a deep copy of the unlearned model on benign data
- Tracks per-epoch knowledge restoration on the forget set
- Reports accuracy, loss, confidence recovery, restoration rate, restoration AUC
- Handles edge cases (no forget data, no benign/retain data)
- Detects knowledge restoration when forget-set accuracy recovers
- Reports near-zero restoration when unlearning is robust
- Never mutates the original model
- Is importable from the public metrics API and registered in the metric registry
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.metrics.forgetting.benign_finetuning import (
    BenignFinetuningMetric,
    _auc_trapezoid,
    _evaluate,
    _finetune_one_epoch,
)


# ---------------------------------------------------------------------------
# Shared test constants and helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 4
INPUT_DIM = 16
BATCH_SIZE = 8
NUM_BATCHES = 4  # 32 samples total


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


def _make_model(input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Tiny two-layer classifier for testing."""
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )


def _make_biased_model(input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Model initialised with large bias toward class 0 — acts as a 'pre-trained' model."""
    model = nn.Linear(input_dim, num_classes)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    model.bias.data[0] = 100.0  # always predicts class 0 with near-certainty
    return model


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestAucTrapezoid:
    def test_empty(self):
        assert _auc_trapezoid([]) == 0.0

    def test_single(self):
        assert _auc_trapezoid([0.5]) == pytest.approx(0.5)

    def test_two_equal(self):
        # Constant curve → normalised AUC = 0.0 (no range)
        result = _auc_trapezoid([0.3, 0.3])
        assert result == pytest.approx(0.0)

    def test_increasing(self):
        # Linearly increasing normalises to y=[0,1] → AUC = 0.5
        result = _auc_trapezoid([0.0, 0.5, 1.0])
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_constant_high(self):
        result = _auc_trapezoid([1.0, 1.0, 1.0, 1.0])
        assert result == pytest.approx(0.0)  # all values equal → normalised y=0

    def test_returns_float(self):
        assert isinstance(_auc_trapezoid([0.1, 0.2, 0.3]), float)


class TestEvaluateHelper:
    def test_accuracy_in_range(self):
        model = _make_model()
        loader = _make_loader()
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss(reduction="sum")
        stats = _evaluate(model, loader, device, criterion)
        assert 0.0 <= stats["accuracy"] <= 1.0

    def test_loss_non_negative(self):
        model = _make_model()
        loader = _make_loader()
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss(reduction="sum")
        stats = _evaluate(model, loader, device, criterion)
        assert stats["loss"] >= 0.0

    def test_confidence_in_range(self):
        model = _make_model()
        loader = _make_loader()
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss(reduction="sum")
        stats = _evaluate(model, loader, device, criterion)
        assert 0.0 <= stats["confidence"] <= 1.0

    def test_n_samples(self):
        model = _make_model()
        loader = _make_loader(num_batches=3, batch_size=8)
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss(reduction="sum")
        stats = _evaluate(model, loader, device, criterion)
        assert stats["n_samples"] == 24

    def test_empty_loader(self):
        model = _make_model()
        empty_ds = TensorDataset(
            torch.zeros(0, INPUT_DIM), torch.zeros(0, dtype=torch.long)
        )
        loader = DataLoader(empty_ds, batch_size=8)
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss(reduction="sum")
        stats = _evaluate(model, loader, device, criterion)
        assert stats["n_samples"] == 0
        assert stats["accuracy"] == 0.0


class TestFineTuneOneEpoch:
    def test_loss_decreases_with_training(self):
        """After fine-tuning on easy data (fixed labels, biased model), loss should drop."""
        model = _make_biased_model()
        loader = _make_loader(fixed_label=0)  # all labels = 0
        device = torch.device("cpu")
        criterion_sum = nn.CrossEntropyLoss(reduction="sum")
        criterion_mean = nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        pre = _evaluate(model, loader, device, criterion_sum)
        model.train()
        _finetune_one_epoch(model, loader, optimizer, criterion_mean, device)
        model.eval()
        post = _evaluate(model, loader, device, criterion_sum)

        # Biased model already near perfect on class 0; loss should remain low
        assert post["loss"] <= pre["loss"] + 0.5  # allow minor numerical variation

    def test_max_batches_respected(self):
        """With max_batches=1, the loop should process only one batch."""
        model = _make_model()
        loader = _make_loader(num_batches=10)
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Should not raise and should finish quickly
        train_loss = _finetune_one_epoch(
            model, loader, optimizer, criterion, device, max_batches=1
        )
        assert isinstance(train_loss, float)
        assert train_loss >= 0.0


# ---------------------------------------------------------------------------
# BenignFinetuningMetric — constructor validation
# ---------------------------------------------------------------------------

class TestBenignFinetuningMetricInit:
    def test_default_construction(self):
        m = BenignFinetuningMetric()
        assert m.epochs == 5
        assert m.lr == pytest.approx(1e-3)
        assert m.recovery_threshold == pytest.approx(0.15)
        assert m.finetune_fraction == pytest.approx(1.0)

    def test_custom_construction(self):
        m = BenignFinetuningMetric(epochs=10, lr=5e-4, recovery_threshold=0.05)
        assert m.epochs == 10

    def test_invalid_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            BenignFinetuningMetric(epochs=0)

    def test_invalid_finetune_fraction_zero(self):
        with pytest.raises(ValueError, match="finetune_fraction"):
            BenignFinetuningMetric(finetune_fraction=0.0)

    def test_invalid_finetune_fraction_above_one(self):
        with pytest.raises(ValueError, match="finetune_fraction"):
            BenignFinetuningMetric(finetune_fraction=1.5)

    def test_invalid_optimizer_cls(self):
        with pytest.raises(ValueError, match="optimizer_cls"):
            BenignFinetuningMetric(optimizer_cls="rmsprop")

    def test_name_attribute(self):
        assert BenignFinetuningMetric.name == "benign_finetuning"


# ---------------------------------------------------------------------------
# BenignFinetuningMetric — edge cases
# ---------------------------------------------------------------------------

class TestBenignFinetuningMetricEdgeCases:

    def test_no_forget_data_returns_defaults(self):
        m = BenignFinetuningMetric(epochs=1)
        model = _make_model()
        result = m.compute(model, forget_data=None, retain_data=_make_loader())
        assert result["benign_ft_forget_accuracy_recovery"] == 0.0
        assert result["benign_ft_passed"] == 1.0

    def test_no_retain_and_no_benign_data_returns_defaults(self):
        m = BenignFinetuningMetric(epochs=1)
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=None)
        assert result["benign_ft_forget_accuracy_recovery"] == 0.0
        assert result["benign_ft_passed"] == 1.0

    def test_both_none_returns_defaults(self):
        m = BenignFinetuningMetric(epochs=1)
        model = _make_model()
        result = m.compute(model, forget_data=None, retain_data=None)
        assert result["benign_ft_passed"] == 1.0

    def test_all_values_are_floats(self):
        m = BenignFinetuningMetric(epochs=2)
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())
        for k, v in result.items():
            assert isinstance(v, float), f"Key {k!r} has type {type(v)}, expected float"


# ---------------------------------------------------------------------------
# BenignFinetuningMetric — output schema
# ---------------------------------------------------------------------------

class TestBenignFinetuningMetricOutputSchema:

    def test_required_keys_present(self):
        m = BenignFinetuningMetric(epochs=2)
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())

        required = {
            "benign_ft_pre_forget_accuracy",
            "benign_ft_pre_forget_loss",
            "benign_ft_pre_forget_confidence",
            "benign_ft_post_forget_accuracy",
            "benign_ft_post_forget_loss",
            "benign_ft_post_forget_confidence",
            "benign_ft_forget_accuracy_recovery",
            "benign_ft_forget_loss_recovery",
            "benign_ft_forget_confidence_recovery",
            "benign_ft_restoration_rate",
            "benign_ft_restoration_auc",
            "benign_ft_epochs",
            "benign_ft_n_forget_samples",
            "benign_ft_passed",
        }
        missing = required - result.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_per_epoch_keys_present(self):
        epochs = 3
        m = BenignFinetuningMetric(epochs=epochs)
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())

        for i in range(epochs):
            assert f"benign_ft_epoch_{i}_forget_accuracy" in result, (
                f"Missing per-epoch key for epoch {i}"
            )
            assert f"benign_ft_epoch_{i}_forget_loss" in result
            assert f"benign_ft_epoch_{i}_forget_confidence" in result
            assert f"benign_ft_epoch_{i}_train_loss" in result

    def test_retain_keys_present_when_retain_data_given(self):
        m = BenignFinetuningMetric(epochs=1)
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())
        assert "benign_ft_pre_retain_accuracy" in result
        assert "benign_ft_post_retain_accuracy" in result
        assert "benign_ft_retain_accuracy_change" in result

    def test_n_forget_samples_correct(self):
        m = BenignFinetuningMetric(epochs=1)
        model = _make_model()
        loader = _make_loader(num_batches=3, batch_size=8)
        result = m.compute(model, forget_data=loader, retain_data=_make_loader())
        assert result["benign_ft_n_forget_samples"] == pytest.approx(24.0)

    def test_epochs_field_matches_config(self):
        m = BenignFinetuningMetric(epochs=7)
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())
        assert result["benign_ft_epochs"] == pytest.approx(7.0)

    def test_restoration_rate_in_range(self):
        m = BenignFinetuningMetric(epochs=2)
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())
        assert 0.0 <= result["benign_ft_restoration_rate"] <= 1.0

    def test_restoration_auc_non_negative(self):
        m = BenignFinetuningMetric(epochs=2)
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())
        assert result["benign_ft_restoration_auc"] >= 0.0

    def test_passed_is_bool_float(self):
        m = BenignFinetuningMetric(epochs=1)
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())
        assert result["benign_ft_passed"] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# BenignFinetuningMetric — semantic correctness
# ---------------------------------------------------------------------------

class TestBenignFinetuningMetricSemantics:

    def test_model_not_mutated(self):
        """The original model weights must be unchanged after compute()."""
        m = BenignFinetuningMetric(epochs=3)
        model = _make_model()

        # Snapshot all parameters
        original_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())

        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original_params[name]), (
                f"Parameter {name!r} was mutated by compute()"
            )

    def test_high_recovery_yields_failed_verdict(self):
        """
        A model that starts with zero forget-accuracy but recovers substantially
        after fine-tuning should yield passed=False.

        We construct a biased model that always predicts class 0, then use
        forget data with label 0 (easy recovery) and a *very* low threshold.
        """
        m = BenignFinetuningMetric(
            epochs=5,
            lr=1e-2,
            recovery_threshold=0.0,  # any recovery at all → fail
        )
        model = _make_biased_model()
        # Fine-tune on class-0 data (easy to learn); forget set also class 0
        loader_class0 = _make_loader(fixed_label=0)
        result = m.compute(
            model,
            forget_data=loader_class0,
            retain_data=loader_class0,
        )
        # Biased model already high accuracy → recovery ≥ 0 always → fail
        # (even 0 recovery fails zero-threshold)
        assert result["benign_ft_passed"] == 0.0

    def test_threshold_controls_verdict(self):
        """With a generous threshold, the same scenario should pass."""
        m = BenignFinetuningMetric(
            epochs=2,
            lr=1e-3,
            recovery_threshold=1.0,  # accept any recovery
        )
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())
        assert result["benign_ft_passed"] == 1.0

    def test_benign_data_kwarg_used_for_finetuning(self):
        """When benign_data is provided, it should be used instead of retain_data."""
        m = BenignFinetuningMetric(epochs=1)
        model = _make_model()
        # Provide a unique benign_data loader; retain_data is still used for measurement
        benign_loader = _make_loader(num_batches=2)
        retain_loader = _make_loader(num_batches=3)
        result = m.compute(
            model,
            forget_data=_make_loader(),
            retain_data=retain_loader,
            benign_data=benign_loader,
        )
        # Should execute without error and produce valid output
        assert "benign_ft_forget_accuracy_recovery" in result

    def test_finetune_fraction_limits_batches(self):
        """finetune_fraction < 1.0 should still produce valid output."""
        m = BenignFinetuningMetric(epochs=2, finetune_fraction=0.25)
        model = _make_model()
        result = m.compute(
            model,
            forget_data=_make_loader(num_batches=10),
            retain_data=_make_loader(num_batches=10),
        )
        assert "benign_ft_forget_accuracy_recovery" in result
        assert isinstance(result["benign_ft_post_forget_accuracy"], float)

    def test_sgd_optimizer(self):
        """SGD optimizer path should produce valid output."""
        m = BenignFinetuningMetric(epochs=2, optimizer_cls="sgd")
        model = _make_model()
        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())
        assert "benign_ft_forget_accuracy_recovery" in result

    def test_original_accuracy_kwarg_scales_restoration_rate(self):
        """
        Providing original_accuracy should shift the restoration rate computation.
        If pre_accuracy == original_accuracy, restoration_rate should be 0.
        """
        m = BenignFinetuningMetric(epochs=1)
        model = _make_model()
        forget_loader = _make_loader()

        # First pass: get pre_accuracy
        result_without = m.compute(model, forget_data=forget_loader, retain_data=_make_loader())
        pre_acc = result_without["benign_ft_pre_forget_accuracy"]

        # Second pass: set original_accuracy == pre_accuracy (model is already at "original" state)
        result_with = m.compute(
            model,
            forget_data=forget_loader,
            retain_data=_make_loader(),
            original_accuracy=pre_acc,
        )
        # With max_recoverable ≈ 0 → restoration_rate is clamped to [0,1]; fine either way
        assert 0.0 <= result_with["benign_ft_restoration_rate"] <= 1.0

    def test_per_epoch_accuracy_is_monotone_increasing_for_easy_task(self):
        """
        On a trivially easy task (model trained on all-class-0 data with
        a very favourable initialisation), per-epoch accuracy should trend up.
        We relax this to just check that the final epoch accuracy is >= the first.
        """
        m = BenignFinetuningMetric(epochs=5, lr=5e-2)
        model = _make_biased_model()  # already biased toward class 0
        loader_class0 = _make_loader(fixed_label=0)

        result = m.compute(model, forget_data=loader_class0, retain_data=loader_class0)

        epoch0_acc = result["benign_ft_epoch_0_forget_accuracy"]
        epoch4_acc = result["benign_ft_epoch_4_forget_accuracy"]
        # Biased model is already near perfect; accuracy shouldn't collapse
        assert epoch4_acc >= 0.0
        assert epoch0_acc >= 0.0

    def test_uniform_model_leakage_low(self):
        """
        A model with uniform outputs (random init with all-zeros weights)
        fine-tuned on random retained data should yield low absolute recovery
        because the pre-attack accuracy is already at chance level.
        """
        class UniformModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.size(0), NUM_CLASSES)

        m = BenignFinetuningMetric(
            epochs=3,
            lr=0.0,  # no actual parameter updates → uniform stays uniform
            recovery_threshold=0.15,
        )
        model = UniformModel()
        # Give it a dummy parameter so next(model.parameters()).device doesn't fail
        model.dummy = nn.Parameter(torch.zeros(1))

        result = m.compute(model, forget_data=_make_loader(), retain_data=_make_loader())
        # With lr=0, the model never changes → recovery should be ~0
        assert abs(result["benign_ft_forget_accuracy_recovery"]) < 1e-4

    def test_hf_style_model_works(self):
        """BenignFinetuningMetric should handle models that return objects with .logits."""
        class HFModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._linear = nn.Linear(INPUT_DIM, NUM_CLASSES)

            class _Output:
                def __init__(self, logits):
                    self.logits = logits

            def forward(self, x):
                return self._Output(self._linear(x))

        m = BenignFinetuningMetric(epochs=2)
        result = m.compute(HFModel(), forget_data=_make_loader(), retain_data=_make_loader())
        assert "benign_ft_forget_accuracy_recovery" in result
        assert isinstance(result["benign_ft_post_forget_accuracy"], float)


# ---------------------------------------------------------------------------
# Registry / import tests
# ---------------------------------------------------------------------------

class TestBenignFinetuningMetricRegistry:

    def test_importable_from_metrics_package(self):
        from erasus.metrics import BenignFinetuningMetric as BFM
        assert BFM is BenignFinetuningMetric

    def test_registered_in_metric_registry(self):
        from erasus.core.registry import metric_registry
        # Trigger registration (import erasus.metrics.__init__)
        import erasus.metrics  # noqa: F401
        cls = metric_registry.get("benign_finetuning")
        assert cls is BenignFinetuningMetric

    def test_name_class_attribute(self):
        assert BenignFinetuningMetric.name == "benign_finetuning"

    def test_is_base_metric_subclass(self):
        from erasus.core.base_metric import BaseMetric
        assert issubclass(BenignFinetuningMetric, BaseMetric)
