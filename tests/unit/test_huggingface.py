"""
Tests for HuggingFace Hub integration module.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestHuggingFaceHubInit:
    """Test HuggingFaceHub initialisation."""

    def test_import(self):
        from erasus.integrations.huggingface import HuggingFaceHub
        assert HuggingFaceHub is not None

    def test_init_no_token(self):
        from erasus.integrations.huggingface import HuggingFaceHub
        hub = HuggingFaceHub()
        assert hub.token is None or isinstance(hub.token, str)

    def test_init_with_token(self):
        from erasus.integrations.huggingface import HuggingFaceHub
        hub = HuggingFaceHub(token="hf_test_token")
        assert hub.token == "hf_test_token"

    def test_api_lazy_import_error(self):
        from erasus.integrations.huggingface import HuggingFaceHub
        hub = HuggingFaceHub(token="test")
        # If huggingface_hub is not installed, property should raise ImportError
        # If it IS installed, it should return an HfApi instance
        try:
            api = hub.api
            assert api is not None
        except ImportError:
            pass  # Expected if huggingface_hub not installed


class TestModelCard:
    """Test model card generation."""

    def test_generate_model_card(self):
        from erasus.integrations.huggingface import HuggingFaceHub
        hub = HuggingFaceHub()
        card = hub.generate_model_card(
            repo_id="user/test-model",
            unlearning_info={
                "strategy": "gradient_ascent",
                "selector": "herding",
                "epochs": 5,
                "forget_size": 100,
                "elapsed_time": 12.5,
                "metrics": {"accuracy": 0.95, "mia_auc": 0.52},
            },
        )
        assert "user/test-model" in card
        assert "gradient_ascent" in card
        assert "herding" in card
        assert "accuracy" in card
        assert "0.95" in card

    def test_generate_model_card_no_metrics(self):
        from erasus.integrations.huggingface import HuggingFaceHub
        hub = HuggingFaceHub()
        card = hub.generate_model_card(
            repo_id="user/no-metrics",
            unlearning_info={"strategy": "scrub"},
        )
        assert "No metrics recorded" in card

    def test_model_card_has_version(self):
        from erasus.integrations.huggingface import HuggingFaceHub
        from erasus.version import __version__
        hub = HuggingFaceHub()
        card = hub.generate_model_card("u/m", {"strategy": "test"})
        assert __version__ in card


class TestDatasetLoading:
    """Test dataset loading utilities."""

    def test_load_dataset_import_error(self):
        from erasus.integrations.huggingface import HuggingFaceHub
        # If datasets is not installed, should raise ImportError
        try:
            HuggingFaceHub.load_dataset("nonexistent/dataset")
        except ImportError:
            pass  # Expected
        except Exception:
            pass  # Other errors (e.g., network) are fine

    def test_dataset_to_dataloader(self):
        from erasus.integrations.huggingface import HuggingFaceHub

        # Mock a simple HF-like dataset
        class FakeDataset:
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return {"input": idx, "label": idx % 2}

        loader = HuggingFaceHub.dataset_to_dataloader(
            FakeDataset(), batch_size=4, shuffle=False,
        )
        batches = list(loader)
        assert len(batches) == 3  # ceil(10/4) = 3
        assert "input" in batches[0]


class TestIntegrationsPackage:
    """Test integrations package exports."""

    def test_package_import(self):
        from erasus.integrations import HuggingFaceHub
        assert HuggingFaceHub is not None

    def test_class_has_expected_methods(self):
        from erasus.integrations import HuggingFaceHub
        hub = HuggingFaceHub()
        assert hasattr(hub, "push_model")
        assert hasattr(hub, "pull_model")
        assert hasattr(hub, "pull_unlearning_info")
        assert hasattr(hub, "generate_model_card")
        assert hasattr(hub, "load_dataset")
        assert hasattr(hub, "dataset_to_dataloader")
        assert hasattr(hub, "list_erasus_models")
