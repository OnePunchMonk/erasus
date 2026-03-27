"""
Tests for issue #11: Performance & DX improvements (Lightning + Unsloth inspired).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _tiny_model(in_dim=16, n_classes=4):
    return nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, n_classes))


def _make_loader(n=50, in_dim=16, n_classes=4, bs=16):
    return DataLoader(
        TensorDataset(torch.randn(n, in_dim), torch.randint(0, n_classes, (n,))),
        batch_size=bs,
    )


# ======================================================================
# 1. self.log() in UnlearningModule
# ======================================================================


class TestSelfLog:
    def test_log_stores_values(self):
        from erasus.core.unlearning_module import UnlearningModule

        class MyModule(UnlearningModule):
            def forget_step(self, batch, batch_idx):
                loss = -F.cross_entropy(self.model(batch[0]), batch[1])
                self.log("forget_loss", loss)
                return loss

            def retain_step(self, batch, batch_idx):
                return F.cross_entropy(self.model(batch[0]), batch[1])

        module = MyModule(_tiny_model())
        batch = (torch.randn(4, 16), torch.randint(0, 4, (4,)))
        module.forget_step(batch, 0)

        assert "forget_loss" in module.logged_metrics
        assert isinstance(module.logged_metrics["forget_loss"], float)

    def test_log_accepts_tensor(self):
        from erasus.core.unlearning_module import UnlearningModule

        class M(UnlearningModule):
            def forget_step(self, batch, batch_idx):
                self.log("x", torch.tensor(3.14))
                return torch.tensor(0.0)
            def retain_step(self, batch, batch_idx):
                return torch.tensor(0.0)

        m = M(_tiny_model())
        m.forget_step(None, 0)
        assert abs(m.logged_metrics["x"] - 3.14) < 0.01

    def test_reset_clears_metrics(self):
        from erasus.core.unlearning_module import UnlearningModule

        class M(UnlearningModule):
            def forget_step(self, batch, batch_idx):
                self.log("a", 1.0)
                return torch.tensor(0.0)
            def retain_step(self, batch, batch_idx):
                return torch.tensor(0.0)

        m = M(_tiny_model())
        m.forget_step(None, 0)
        assert "a" in m.logged_metrics
        m._reset_logged_metrics()
        assert len(m.logged_metrics) == 0


# ======================================================================
# 2. save_hyperparameters()
# ======================================================================


class TestSaveHyperparameters:
    def test_captures_init_args(self):
        from erasus.core.unlearning_module import UnlearningModule

        class MyModule(UnlearningModule):
            def __init__(self, model, lr=1e-3, alpha=0.5):
                super().__init__(model)
                self.save_hyperparameters(ignore=["model"])

            def forget_step(self, batch, batch_idx):
                return torch.tensor(0.0)
            def retain_step(self, batch, batch_idx):
                return torch.tensor(0.0)

        m = MyModule(_tiny_model(), lr=0.01, alpha=0.9)
        assert m.hparams["lr"] == 0.01
        assert m.hparams["alpha"] == 0.9
        assert "model" not in m.hparams

    def test_empty_without_call(self):
        from erasus.core.unlearning_module import UnlearningModule

        class M(UnlearningModule):
            def __init__(self, model):
                super().__init__(model)
                # deliberately not calling save_hyperparameters

            def forget_step(self, batch, batch_idx):
                return torch.tensor(0.0)
            def retain_step(self, batch, batch_idx):
                return torch.tensor(0.0)

        m = M(_tiny_model())
        assert m.hparams == {}


# ======================================================================
# 3. Richer callback hooks
# ======================================================================


class TestCallbackHooks:
    def test_new_hooks_exist(self):
        from erasus.utils.callbacks import Callback

        cb = Callback()
        # These should not raise
        cb.on_before_unlearn_step(model=None, batch=None)
        cb.on_after_unlearn_step(model=None, batch=None, loss=0.0)
        cb.on_coreset_selected(indices=[], metadata={})
        cb.on_checkpoint_save(path="/tmp/x", metadata={})
        cb.on_exception(exception=RuntimeError("test"))

    def test_callback_list_dispatches_new_hooks(self):
        from erasus.utils.callbacks import Callback, CallbackList

        class Tracker(Callback):
            def __init__(self):
                self.calls = []

            def on_before_unlearn_step(self, model, batch, **kw):
                self.calls.append("before")

            def on_after_unlearn_step(self, model, batch, loss, **kw):
                self.calls.append("after")

            def on_coreset_selected(self, indices, metadata, **kw):
                self.calls.append("coreset")

            def on_exception(self, exception, **kw):
                self.calls.append("exception")

        tracker = Tracker()
        cbl = CallbackList([tracker])
        cbl.on_before_unlearn_step(None, None)
        cbl.on_after_unlearn_step(None, None, 0.0)
        cbl.on_coreset_selected([], {})
        cbl.on_exception(RuntimeError(""))

        assert tracker.calls == ["before", "after", "coreset", "exception"]


# ======================================================================
# 4. Fabric-style composable primitives
# ======================================================================


class TestFabric:
    def test_select_coreset(self):
        from erasus.fabric import select_coreset

        model = _tiny_model()
        loader = _make_loader(50)
        indices = select_coreset("random", model, loader, k=10)
        assert len(indices) == 10

    def test_apply_gradient_ascent(self):
        from erasus.fabric import apply_gradient_ascent

        model = _tiny_model()
        loader = _make_loader(30)
        result = apply_gradient_ascent(model, loader, lr=1e-3, epochs=1)
        assert result is model

    def test_compute_forgetting_quality(self):
        from erasus.fabric import compute_forgetting_quality

        model = _tiny_model()
        loader = _make_loader(30)
        quality = compute_forgetting_quality(model, loader)
        assert 0.0 <= quality <= 1.0

    def test_compute_mia_score(self):
        from erasus.fabric import compute_mia_score

        model = _tiny_model()
        score = compute_mia_score(model, _make_loader(30), _make_loader(30))
        assert 0.0 <= score <= 1.0

    def test_enable_gradient_checkpointing_noop(self):
        from erasus.fabric import enable_gradient_checkpointing

        model = _tiny_model()
        result = enable_gradient_checkpointing(model)
        assert result is model


# ======================================================================
# 5. Mixed precision (AMP) support
# ======================================================================


class TestMixedPrecision:
    def test_gradient_ascent_with_amp_disabled(self):
        """AMP path runs without error on CPU (disabled scaler)."""
        from erasus.strategies.gradient_methods.gradient_ascent import GradientAscentStrategy

        model = _tiny_model()
        strategy = GradientAscentStrategy(lr=1e-3)
        model, f_losses, r_losses = strategy.unlearn(
            model, _make_loader(30), _make_loader(30), epochs=1,
            _amp_enabled=False,
        )
        assert len(f_losses) == 1

    def test_precision_parameter_in_fit(self):
        from erasus import ErasusUnlearner

        model = _tiny_model()
        unlearner = ErasusUnlearner(model=model, strategy="gradient_ascent")
        # precision=None should work fine (no AMP)
        result = unlearner.fit(
            forget_data=_make_loader(30),
            retain_data=_make_loader(30),
            epochs=1,
            precision=None,
        )
        assert result.model is not None

    def test_precision_via_constructor(self):
        from erasus import ErasusUnlearner

        model = _tiny_model()
        unlearner = ErasusUnlearner(
            model=model, strategy="gradient_ascent", precision=None,
        )
        assert unlearner.precision is None


# ======================================================================
# 6. Gradient checkpointing
# ======================================================================


class TestGradientCheckpointing:
    def test_fit_with_gradient_checkpointing(self):
        from erasus import ErasusUnlearner

        model = _tiny_model()
        unlearner = ErasusUnlearner(model=model, strategy="gradient_ascent")
        result = unlearner.fit(
            forget_data=_make_loader(30),
            epochs=1,
            gradient_checkpointing=True,  # no-op on Sequential, but shouldn't error
        )
        assert result.model is not None


# ======================================================================
# 7. Adaptive memory management
# ======================================================================


class TestMemoryManagement:
    def test_chunked_computation(self):
        from erasus.utils.memory import chunked_computation

        data = torch.randn(100, 16)
        result = chunked_computation(lambda x: x * 2, data, chunk_size=25)
        assert result.shape == (100, 16)
        assert torch.allclose(result, data * 2)

    def test_auto_batch_size_cpu(self):
        from erasus.utils.memory import auto_batch_size

        model = _tiny_model()
        sample = torch.randn(16)
        bs = auto_batch_size(model, sample, max_batch_size=64)
        assert bs == 64  # CPU returns max


# ======================================================================
# 9. In-place gradient operations
# ======================================================================


class TestInPlaceOps:
    def test_fisher_uses_inplace(self):
        """Fisher computation should produce finite values with in-place ops."""
        from erasus.strategies.gradient_methods.fisher_forgetting import FisherForgettingStrategy

        model = _tiny_model()
        strategy = FisherForgettingStrategy(lr=1e-3, fisher_lambda=1.0)
        loader = _make_loader(30)

        fisher = strategy._compute_fisher_diag(model, loader, "cpu")
        for name, val in fisher.items():
            assert torch.isfinite(val).all(), f"Non-finite Fisher values for {name}"


# ======================================================================
# 10. Smart defaults
# ======================================================================


class TestSmartDefaults:
    def test_detect_model_type_classifier(self):
        from erasus.unlearners.erasus_unlearner import _detect_model_type

        model = _tiny_model()
        assert _detect_model_type(model) == "classifier"

    def test_detect_model_type_gpt(self):
        from erasus.unlearners.erasus_unlearner import _detect_model_type

        class FakeGPT2Model(nn.Module):
            pass

        assert _detect_model_type(FakeGPT2Model()) == "llm"

    def test_detect_model_type_clip(self):
        from erasus.unlearners.erasus_unlearner import _detect_model_type

        class FakeCLIPModel(nn.Module):
            pass

        assert _detect_model_type(FakeCLIPModel()) == "vlm"

    def test_auto_strategy_uses_detection(self):
        from erasus import ErasusUnlearner

        model = _tiny_model()
        unlearner = ErasusUnlearner(model=model, strategy="auto")
        result = unlearner.fit(
            forget_data=_make_loader(30),
            retain_data=_make_loader(50),
            epochs=1,
        )
        assert result.model is not None
