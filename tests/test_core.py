"""
Tests for core module and new strategy implementations.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.registry import Registry, strategy_registry, selector_registry
from erasus.core.config import ErasusConfig
from erasus.core.base_unlearner import UnlearningResult
from erasus.strategies.data_methods.certified_removal import CertifiedRemovalStrategy
from erasus.strategies.parameter_methods.mask_based import MaskBasedUnlearningStrategy
from erasus.selectors.gradient_based.tracin import TracInSelector

class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("test_reg_1")
        @reg.register("test_cls")
        class TestCls: pass
        assert reg.get("test_cls") is TestCls

    def test_duplicate_raises(self):
        reg = Registry("test_reg_2")
        @reg.register("dup")
        class A: pass
        with pytest.raises(ValueError, match="already registered"):
            @reg.register("dup")
            class B: pass

    def test_missing_raises(self):
        reg = Registry("test_reg_3")
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_list(self):
        reg = Registry("test_reg_4")
        @reg.register("beta")
        class B: pass
        @reg.register("alpha")
        class A: pass
        assert reg.list() == ["alpha", "beta"]


class TestConfig:
    def test_default_config(self):
        config = ErasusConfig()
        assert config.model_name == "openai/clip-vit-base-patch32"
        assert config.epochs == 5

    def test_to_dict(self):
        config = ErasusConfig(epochs=10)
        d = config.to_dict()
        assert d["epochs"] == 10
        assert isinstance(d, dict)


class TestUnlearningResult:
    def test_compression_ratio(self):
        model = nn.Linear(10, 2)
        result = UnlearningResult(model=model, coreset_size=10, original_forget_size=100)
        assert result.compression_ratio == 0.1

    def test_zero_original_size(self):
        model = nn.Linear(10, 2)
        result = UnlearningResult(model=model, coreset_size=0, original_forget_size=0)
        assert result.compression_ratio == 0.0


class TestGlobalRegistries:
    def test_gradient_ascent_registered(self):
        import erasus.strategies.gradient_methods.gradient_ascent  # noqa: F401
        assert "gradient_ascent" in strategy_registry.list()

    def test_modality_decoupling_registered(self):
        import erasus.strategies.gradient_methods.modality_decoupling  # noqa: F401
        assert "modality_decoupling" in strategy_registry.list()

    def test_random_selector_registered(self):
        import erasus.selectors.random_selector  # noqa: F401
        assert "random" in selector_registry.list()
        
    def test_tracin_selector_registered(self):
        assert "tracin" in selector_registry.list()


class TestAlgorithms:
    def setUp(self):
        self.model = nn.Linear(5, 2)
        self.data = DataLoader(TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,))), batch_size=2)
        
    def test_tracin_selector(self):
        model = nn.Linear(5, 2)
        loader = DataLoader(TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,))), batch_size=2)
        selector = TracInSelector()
        
        # Test 1: Self-Influence (Delegate to Grad Norm)
        indices = selector.select(model, loader, k=2)
        assert len(indices) == 2
        
        # Test 2: Target Influence
        target_loader = DataLoader(TensorDataset(torch.randn(5, 5), torch.randint(0, 2, (5,))), batch_size=2)
        indices_tgt = selector.select(model, loader, k=2, target_loader=target_loader)
        assert len(indices_tgt) == 2

    def test_certified_removal(self):
        model = nn.Linear(5, 2)
        # Certified removal needs gradients, so inputs requiring grad? No, model params require grad.
        loader = DataLoader(TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,))), batch_size=2)
        retain_loader = DataLoader(TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,))), batch_size=2)
        
        strategy = CertifiedRemovalStrategy(hessian_samples=2, recursion_depth=2)
        # Should run without error
        model_new, _, _ = strategy.unlearn(model, loader, retain_loader=retain_loader)
        assert isinstance(model_new, nn.Module)

    def test_mask_based_strategy(self):
        model = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 2))
        loader = DataLoader(TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,))), batch_size=2)
        
        strategy = MaskBasedUnlearningStrategy(epochs=1)
        model_new, losses, _ = strategy.unlearn(model, loader)
        assert len(losses) > 0
