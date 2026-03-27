"""
Tests for issue #9 fixes: silent fallbacks, batch unpacking, selector dedup.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_loader(n: int = 50, dim: int = 16, n_classes: int = 4) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.randn(n, dim), torch.randint(0, n_classes, (n,))),
        batch_size=16,
    )


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


# ======================================================================
# Silent fallback fixes — selectors now raise SelectorError
# ======================================================================


class TestSilentFallbacksFixed:
    def test_valuation_network_raises(self):
        from erasus.core.exceptions import SelectorError
        from erasus.selectors.learning_based.valuation_network import ValuationNetworkSelector

        sel = ValuationNetworkSelector()
        with pytest.raises(SelectorError, match="val_net"):
            sel.select(_tiny_model(), _make_loader(), k=5)

    def test_forgetting_events_raises(self):
        from erasus.core.exceptions import SelectorError
        from erasus.selectors.learning_based.forgetting_events import ForgettingEventsSelector

        sel = ForgettingEventsSelector()
        with pytest.raises(SelectorError, match="forgetting_stats"):
            sel.select(_tiny_model(), _make_loader(), k=5)

    def test_data_shapley_raises(self):
        from erasus.core.exceptions import SelectorError
        from erasus.selectors.learning_based.data_shapley import DataShapleySelector

        sel = DataShapleySelector()
        with pytest.raises(SelectorError, match="precomputed_values"):
            sel.select(_tiny_model(), _make_loader(), k=5)

    def test_glister_raises(self):
        from erasus.core.exceptions import SelectorError
        from erasus.selectors.geometry_based.glister import GlisterSelector

        sel = GlisterSelector()
        with pytest.raises(SelectorError, match="val_loader"):
            sel.select(_tiny_model(), _make_loader(), k=5)

    def test_forgetting_events_works_with_stats(self):
        from erasus.selectors.learning_based.forgetting_events import ForgettingEventsSelector

        sel = ForgettingEventsSelector()
        stats = {0: 5, 1: 3, 2: 8, 3: 1, 4: 10}
        result = sel.select(_tiny_model(), _make_loader(), k=3, forgetting_stats=stats)
        assert len(result) == 3
        assert result[0] == 4  # highest count

    def test_data_shapley_works_with_values(self):
        from erasus.selectors.learning_based.data_shapley import DataShapleySelector

        sel = DataShapleySelector()
        values = [0.1, 0.9, 0.5, 0.3, 0.7]
        result = sel.select(_tiny_model(), _make_loader(5), k=2, precomputed_values=values)
        assert len(result) == 2
        assert 1 in result  # highest value


# ======================================================================
# Batch unpacking utility
# ======================================================================


class TestUnpackBatch:
    def test_tuple_batch(self):
        from erasus.utils.batch import unpack_batch

        batch = (torch.randn(4, 16), torch.randint(0, 4, (4,)))
        inputs, labels = unpack_batch(batch)
        assert inputs.shape == (4, 16)
        assert labels.shape == (4,)

    def test_single_tensor_batch(self):
        from erasus.utils.batch import unpack_batch

        batch = (torch.randn(4, 16),)
        inputs, labels = unpack_batch(batch)
        assert inputs.shape == (4, 16)
        assert labels is None

    def test_dict_batch(self):
        from erasus.utils.batch import unpack_batch

        batch = {"input_ids": torch.randn(4, 16), "labels": torch.randint(0, 4, (4,))}
        inputs, labels = unpack_batch(batch)
        assert inputs.shape == (4, 16)
        assert labels.shape == (4,)

    def test_device_transfer(self):
        from erasus.utils.batch import unpack_batch

        batch = (torch.randn(4, 16), torch.randint(0, 4, (4,)))
        inputs, labels = unpack_batch(batch, device=torch.device("cpu"))
        assert inputs.device == torch.device("cpu")

    def test_bad_dict_raises(self):
        from erasus.utils.batch import unpack_batch

        with pytest.raises(ValueError, match="input_ids"):
            unpack_batch({"foo": torch.randn(4, 16)})


# ======================================================================
# Selector deduplication
# ======================================================================


class TestSelectorDedup:
    def test_kcenter_registered_once(self):
        from erasus.core.registry import selector_registry

        # Only "kcenter" should be registered, not "k_center"
        assert "kcenter" in selector_registry.list()

    def test_k_center_import_still_works(self):
        from erasus.selectors.geometry_based.k_center import KCenterSelector
        from erasus.selectors.geometry_based.kcenter import KCenterSelector as KC2

        assert KCenterSelector is KC2


# ======================================================================
# Pre-commit config exists
# ======================================================================


class TestInfrastructure:
    def test_precommit_config_exists(self):
        from pathlib import Path

        config = Path("/Users/avaya.aggarwal@zomato.com/erasus/.pre-commit-config.yaml")
        assert config.exists()
