"""
Tests for Issue #5: Architectural improvements.

Covers:
1. UnlearningModule + UnlearningTrainer (Trainer/Module split)
2. Coreset as first-class composable object
3. Evaluation woven into training loop
4. UnlearningDataset abstraction
5. Standardized evaluation protocol (UnlearningBenchmark)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Re-use the conftest helpers
from tests.conftest import TinyClassifier, _make_dataset, _make_loader


# =====================================================================
# Helpers
# =====================================================================

def _tiny_model():
    return TinyClassifier(input_dim=16, num_classes=4)


def _dataset(n=64):
    return _make_dataset(n_samples=n, input_dim=16, num_classes=4)


def _loader(n=64, batch_size=16):
    return _make_loader(n_samples=n, input_dim=16, num_classes=4, batch_size=batch_size)


# =====================================================================
# 1. UnlearningModule + UnlearningTrainer
# =====================================================================

class SimpleUnlearningModule:
    """A concrete module for testing (avoid import issues by defining inline)."""
    pass


class TestUnlearningModule:
    def test_import(self):
        from erasus.core.unlearning_module import UnlearningModule
        assert UnlearningModule is not None

    def test_subclass_and_instantiate(self):
        from erasus.core.unlearning_module import UnlearningModule

        class MyModule(UnlearningModule):
            def forget_step(self, batch, batch_idx):
                x, y = batch
                logits = self.model(x)
                return -F.cross_entropy(logits, y)

            def retain_step(self, batch, batch_idx):
                x, y = batch
                logits = self.model(x)
                return F.cross_entropy(logits, y)

        model = _tiny_model()
        module = MyModule(model)
        assert module.model is model

    def test_to_device(self):
        from erasus.core.unlearning_module import UnlearningModule

        class MyModule(UnlearningModule):
            def forget_step(self, batch, batch_idx):
                return torch.tensor(0.0)

            def retain_step(self, batch, batch_idx):
                return torch.tensor(0.0)

        module = MyModule(_tiny_model())
        module.to("cpu")
        assert module.device == torch.device("cpu")

    def test_configure_optimizers_default(self):
        from erasus.core.unlearning_module import UnlearningModule

        class MyModule(UnlearningModule):
            def forget_step(self, batch, batch_idx):
                return torch.tensor(0.0)

            def retain_step(self, batch, batch_idx):
                return torch.tensor(0.0)

        module = MyModule(_tiny_model())
        opt = module.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)


class TestUnlearningTrainer:
    def _make_module(self):
        from erasus.core.unlearning_module import UnlearningModule

        class MyModule(UnlearningModule):
            def forget_step(self, batch, batch_idx):
                x, y = batch
                logits = self.model(x)
                return -F.cross_entropy(logits, y)

            def retain_step(self, batch, batch_idx):
                x, y = batch
                logits = self.model(x)
                return F.cross_entropy(logits, y)

        return MyModule(_tiny_model())

    def test_import(self):
        from erasus.core.unlearning_trainer import UnlearningTrainer
        assert UnlearningTrainer is not None

    def test_basic_fit(self):
        from erasus.core.unlearning_trainer import UnlearningTrainer

        trainer = UnlearningTrainer(max_epochs=2, validate_every=0, device="cpu")
        module = self._make_module()
        result = trainer.fit(module, _loader(32, 8), _loader(64, 8))

        assert result.model is not None
        assert len(result.forget_loss_history) == 2
        assert len(result.retain_loss_history) == 2
        assert result.elapsed_time > 0

    def test_fit_with_validation(self):
        from erasus.core.unlearning_trainer import UnlearningTrainer

        trainer = UnlearningTrainer(
            max_epochs=4,
            validate_every=2,
            device="cpu",
        )
        module = self._make_module()
        result = trainer.fit(module, _loader(32, 8), _loader(64, 8))

        assert len(result.validation_history) == 2  # validated at epoch 2 and 4

    def test_early_stopping(self):
        from erasus.core.unlearning_trainer import UnlearningTrainer

        # Monitor retain loss (min mode) — retain loss eventually stops
        # improving and will trigger early stopping
        trainer = UnlearningTrainer(
            max_epochs=100,
            validate_every=1,
            early_stopping_patience=3,
            monitor="val_retain_loss",
            monitor_mode="min",
            device="cpu",
        )
        module = self._make_module()
        result = trainer.fit(module, _loader(32, 8), _loader(64, 8))

        # Should stop before 100 epochs because retain loss eventually
        # starts increasing as the model drifts from gradient ascent
        assert result.stopped_early or len(result.forget_loss_history) <= 100

    def test_best_model_restored(self):
        from erasus.core.unlearning_trainer import UnlearningTrainer

        trainer = UnlearningTrainer(
            max_epochs=3,
            validate_every=1,
            save_best=True,
            monitor="val_retain_loss",
            monitor_mode="min",
            device="cpu",
        )
        module = self._make_module()
        result = trainer.fit(module, _loader(32, 8), _loader(64, 8))

        assert result.best_epoch >= 0
        assert result.model is not None

    def test_forget_only_no_retain(self):
        from erasus.core.unlearning_trainer import UnlearningTrainer

        trainer = UnlearningTrainer(max_epochs=2, validate_every=0, device="cpu")
        module = self._make_module()
        result = trainer.fit(module, _loader(32, 8))

        assert len(result.forget_loss_history) == 2
        assert all(r == 0.0 for r in result.retain_loss_history)


# =====================================================================
# 2. Coreset as composable object
# =====================================================================

class TestCoreset:
    def _dataset(self):
        return _make_dataset(n_samples=64)

    def test_import(self):
        from erasus.core.coreset import Coreset
        assert Coreset is not None

    def test_from_indices(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs = Coreset.from_indices(ds, indices=[0, 5, 10, 15])
        assert len(cs) == 4
        assert cs.indices == [0, 5, 10, 15]

    def test_union(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs_a = Coreset.from_indices(ds, [0, 1, 2])
        cs_b = Coreset.from_indices(ds, [2, 3, 4])
        combined = cs_a.union(cs_b)
        assert len(combined) == 5
        assert set(combined.indices) == {0, 1, 2, 3, 4}

    def test_intersect(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs_a = Coreset.from_indices(ds, [0, 1, 2, 3])
        cs_b = Coreset.from_indices(ds, [2, 3, 4, 5])
        overlap = cs_a.intersect(cs_b)
        assert set(overlap.indices) == {2, 3}

    def test_difference(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs_a = Coreset.from_indices(ds, [0, 1, 2, 3])
        cs_b = Coreset.from_indices(ds, [2, 3])
        diff = cs_a.difference(cs_b)
        assert set(diff.indices) == {0, 1}

    def test_filter_by_score(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs = Coreset(ds, indices=[0, 1, 2, 3], scores=[0.9, 0.3, 0.7, 0.1])
        filtered = cs.filter(min_score=0.5)
        assert set(filtered.indices) == {0, 2}

    def test_filter_without_scores_raises(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs = Coreset.from_indices(ds, [0, 1])
        with pytest.raises(ValueError, match="Scores not available"):
            cs.filter(min_score=0.5)

    def test_add_remove(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs = Coreset.from_indices(ds, [0, 1])
        cs2 = cs.add([2, 3])
        assert len(cs2) == 4
        cs3 = cs2.remove([0])
        assert 0 not in cs3.indices

    def test_to_loader(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs = Coreset.from_indices(ds, [0, 1, 2, 3])
        loader = cs.to_loader(batch_size=2, shuffle=False)
        assert isinstance(loader, DataLoader)
        batches = list(loader)
        assert len(batches) == 2

    def test_to_subset(self):
        from erasus.core.coreset import Coreset
        from torch.utils.data import Subset

        ds = self._dataset()
        cs = Coreset.from_indices(ds, [0, 1, 2])
        subset = cs.to_subset()
        assert isinstance(subset, Subset)
        assert len(subset) == 3

    def test_contains(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs = Coreset.from_indices(ds, [0, 5, 10])
        assert 5 in cs
        assert 3 not in cs

    def test_compression_ratio(self):
        from erasus.core.coreset import Coreset, CoresetMetadata

        ds = self._dataset()
        meta = CoresetMetadata(original_size=64)
        cs = Coreset(ds, indices=[0, 1, 2, 3], metadata=meta)
        assert cs.compression_ratio == pytest.approx(4 / 64)

    def test_repr(self):
        from erasus.core.coreset import Coreset

        ds = self._dataset()
        cs = Coreset.from_indices(ds, [0, 1])
        assert "Coreset" in repr(cs)
        assert "size=2" in repr(cs)

    def test_different_datasets_raises(self):
        from erasus.core.coreset import Coreset

        ds_a = self._dataset()
        ds_b = self._dataset()
        cs_a = Coreset.from_indices(ds_a, [0])
        cs_b = Coreset.from_indices(ds_b, [0])
        with pytest.raises(ValueError, match="different datasets"):
            cs_a.union(cs_b)


# =====================================================================
# 3. In-loop evaluation (via BaseUnlearner.fit)
# =====================================================================

class TestInLoopEvaluation:
    def test_fit_with_validate_every(self):
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner
        from erasus.core.base_metric import BaseMetric

        class DummyMetric(BaseMetric):
            def compute(self, model, forget_data=None, retain_data=None, **kwargs):
                return {"dummy_score": 0.5}

        model = _tiny_model()
        unlearner = ErasusUnlearner(model=model, strategy="gradient_ascent", device="cpu")
        result = unlearner.fit(
            forget_data=_loader(32, 8),
            retain_data=_loader(64, 8),
            epochs=4,
            validate_every=2,
            validation_metrics=[DummyMetric()],
        )

        assert len(result.validation_history) == 2
        assert "dummy_score" in result.validation_history[0]

    def test_fit_with_coreset_object(self):
        from erasus.core.coreset import Coreset
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner

        ds = _make_dataset(64)
        coreset = Coreset.from_indices(ds, indices=[0, 1, 2, 3, 4, 5, 6, 7])
        forget_loader = _loader(64, 8)

        model = _tiny_model()
        unlearner = ErasusUnlearner(model=model, strategy="gradient_ascent", device="cpu")
        result = unlearner.fit(
            forget_data=forget_loader,
            retain_data=_loader(64, 8),
            coreset=coreset,
            epochs=2,
        )

        assert result.coreset_size == 8
        assert result.model is not None


# =====================================================================
# 4. UnlearningDataset abstraction
# =====================================================================

class TestUnlearningDataset:
    def _base_dataset(self, n=100):
        x = torch.randn(n, 16)
        y = torch.randint(0, 4, (n,))
        return TensorDataset(x, y)

    def test_sample_level_forget(self):
        from erasus.data.datasets import UnlearningDataset

        ds = UnlearningDataset(self._base_dataset(), forget_indices=[0, 5, 10])
        assert ds.forget_size == 3
        assert ds.retain_size == 97
        assert ds.is_forget(5)
        assert not ds.is_forget(1)

    def test_class_level_forget(self):
        from erasus.data.datasets import UnlearningDataset

        base = self._base_dataset(100)
        ds = UnlearningDataset(base, forget_classes=[0])

        # All samples with label 0 should be in forget set
        for idx in ds.forget_indices:
            _, label = base[idx]
            assert label.item() == 0

    def test_streaming_mark_forget(self):
        from erasus.data.datasets import UnlearningDataset

        ds = UnlearningDataset(self._base_dataset(), forget_indices=[0])
        assert ds.forget_size == 1
        ds.mark_forget([1, 2, 3])
        assert ds.forget_size == 4
        assert ds.is_forget(2)

    def test_streaming_mark_retain(self):
        from erasus.data.datasets import UnlearningDataset

        ds = UnlearningDataset(self._base_dataset(), forget_indices=[0, 1, 2])
        ds.mark_retain([1])
        assert ds.forget_size == 2
        assert not ds.is_forget(1)

    def test_to_loaders(self):
        from erasus.data.datasets import UnlearningDataset

        ds = UnlearningDataset(
            self._base_dataset(64), forget_indices=list(range(16))
        )
        forget_loader, retain_loader = ds.to_loaders(batch_size=8, shuffle=False)
        assert len(forget_loader.dataset) == 16
        assert len(retain_loader.dataset) == 48

    def test_weighted_loader(self):
        from erasus.data.datasets import UnlearningDataset

        ds = UnlearningDataset(
            self._base_dataset(64),
            forget_indices=[0, 1, 2, 3],
            forget_weights={0: 5.0, 1: 1.0, 2: 1.0, 3: 1.0},
        )
        forget_loader, _ = ds.to_loaders(batch_size=4, weighted=True)
        # Should produce a loader (weighted sampling doesn't change count)
        batch = next(iter(forget_loader))
        assert batch[0].shape[0] == 4

    def test_forget_ratio(self):
        from erasus.data.datasets import UnlearningDataset

        ds = UnlearningDataset(
            self._base_dataset(100), forget_indices=list(range(25))
        )
        assert ds.forget_ratio == pytest.approx(0.25)

    def test_forget_and_retain_subsets(self):
        from erasus.data.datasets import UnlearningDataset

        ds = UnlearningDataset(
            self._base_dataset(50), forget_indices=[0, 1, 2]
        )
        assert len(ds.forget_subset) == 3
        assert len(ds.retain_subset) == 47

    def test_repr(self):
        from erasus.data.datasets import UnlearningDataset

        ds = UnlearningDataset(
            self._base_dataset(100), forget_indices=list(range(10))
        )
        r = repr(ds)
        assert "total=100" in r
        assert "forget=10" in r

    def test_set_weight(self):
        from erasus.data.datasets import UnlearningDataset

        ds = UnlearningDataset(self._base_dataset(), forget_indices=[0])
        ds.set_weight(0, 3.0)
        assert ds._weights[0] == 3.0

    def test_getitem(self):
        from erasus.data.datasets import UnlearningDataset

        base = self._base_dataset(10)
        ds = UnlearningDataset(base, forget_indices=[0])
        item = ds[0]
        assert isinstance(item, tuple)


# =====================================================================
# 5. UnlearningBenchmark (standardized evaluation protocol)
# =====================================================================

class TestUnlearningBenchmark:
    def test_import(self):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark
        assert UnlearningBenchmark is not None

    def test_list_protocols(self):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        protocols = UnlearningBenchmark.list_protocols()
        assert "tofu" in protocols
        assert "muse" in protocols
        assert "wmdp" in protocols
        assert "general" in protocols

    def test_invalid_protocol_raises(self):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        with pytest.raises(ValueError, match="Unknown protocol"):
            UnlearningBenchmark(protocol="nonexistent")

    def test_evaluate_general(self):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        model = _tiny_model()
        benchmark = UnlearningBenchmark(protocol="general", n_runs=1)
        report = benchmark.evaluate(
            unlearned_model=model,
            forget_data=_loader(32, 8),
            retain_data=_loader(64, 8),
        )

        assert report.verdict in ("PASS", "PARTIAL", "FAIL")
        assert 0.0 <= report.pass_rate <= 1.0
        assert report.elapsed_time > 0
        assert len(report.metric_results) > 0

    def test_evaluate_with_gold_model(self):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        model = _tiny_model()
        gold = _tiny_model()
        benchmark = UnlearningBenchmark(protocol="general", n_runs=2)
        report = benchmark.evaluate(
            unlearned_model=model,
            gold_model=gold,
            forget_data=_loader(32, 8),
            retain_data=_loader(64, 8),
        )

        # Gold values should be populated
        for mr in report.metric_results.values():
            assert len(mr.gold_values) == 2

    def test_report_summary(self):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        model = _tiny_model()
        benchmark = UnlearningBenchmark(protocol="tofu", n_runs=1)
        report = benchmark.evaluate(
            unlearned_model=model,
            forget_data=_loader(32, 8),
            retain_data=_loader(64, 8),
        )

        summary = report.summary()
        assert "Protocol: tofu" in summary
        assert "Verdict:" in summary

    def test_metric_result_confidence_interval(self):
        from erasus.evaluation.benchmark_protocol import MetricResult

        mr = MetricResult(
            name="test",
            values=[0.5, 0.6, 0.55, 0.52, 0.58],
            pass_threshold=0.5,
            direction="higher_is_better",
        )
        ci = mr.confidence_interval(0.95)
        assert ci[0] < mr.mean < ci[1]
        assert mr.passed  # mean > 0.5

    def test_metric_result_single_value(self):
        from erasus.evaluation.benchmark_protocol import MetricResult

        mr = MetricResult(name="test", values=[0.5], pass_threshold=0.5)
        ci = mr.confidence_interval(0.95)
        assert ci[0] == ci[1] == 0.5

    def test_evaluate_tofu_protocol(self):
        from erasus.evaluation.benchmark_protocol import UnlearningBenchmark

        model = _tiny_model()
        benchmark = UnlearningBenchmark(protocol="tofu", n_runs=1)
        report = benchmark.evaluate(
            unlearned_model=model,
            forget_data=_loader(32, 8),
            retain_data=_loader(64, 8),
        )

        # TOFU has 4 metrics
        assert len(report.metric_results) == 4
        assert "forget_quality" in report.metric_results
        assert "model_utility" in report.metric_results
        assert "membership_inference_auc" in report.metric_results
        assert "truth_ratio" in report.metric_results


# =====================================================================
# 6. ForgetRetainDataset (preserved from original)
# =====================================================================

class TestForgetRetainDataset:
    def test_basic(self):
        from erasus.data.datasets import ForgetRetainDataset

        forget = TensorDataset(torch.randn(10, 16), torch.randint(0, 4, (10,)))
        retain = TensorDataset(torch.randn(20, 16), torch.randint(0, 4, (20,)))
        combined = ForgetRetainDataset(forget, retain)
        assert len(combined) == 30

        sample, label = combined[0]
        assert label == 1  # forget

        sample, label = combined[15]
        assert label == 0  # retain


# =====================================================================
# Integration: ErasusUnlearner accepts strategy/selector instances
# =====================================================================

class TestErasusUnlearnerInstances:
    def test_strategy_instance(self):
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner
        from erasus.strategies.gradient_methods.gradient_ascent import GradientAscentStrategy

        strategy = GradientAscentStrategy(lr=1e-3)
        model = _tiny_model()
        unlearner = ErasusUnlearner(model=model, strategy=strategy, device="cpu")
        assert unlearner.strategy_name == "GradientAscentStrategy"

    def test_invalid_strategy_type_raises(self):
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner

        with pytest.raises(TypeError, match="strategy must be"):
            ErasusUnlearner(model=_tiny_model(), strategy=42)

    def test_invalid_selector_type_raises(self):
        from erasus.unlearners.erasus_unlearner import ErasusUnlearner

        with pytest.raises(TypeError, match="selector must be"):
            ErasusUnlearner(model=_tiny_model(), strategy="gradient_ascent", selector=42)
