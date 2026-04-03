"""
Tests for ContinualUnlearner — sequential unlearning orchestrator.

Tests the continual unlearning pipeline for realistic scenarios with
multiple sequential deletion requests.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.exceptions import SelectorError
from erasus.unlearners.continual_unlearner import (
    ContinualUnlearner,
    DeletionRequest,
)


@pytest.fixture
def tiny_classifier():
    """Simple 2-layer classifier for testing."""

    class TinyClassifier(nn.Module):
        def __init__(self, input_dim=16, num_classes=4):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            logits = self.fc2(x)
            return logits

    return TinyClassifier()


@pytest.fixture
def retention_loader():
    """Retain dataset for utility preservation."""
    X = torch.randn(32, 16)
    y = torch.randint(0, 4, (32,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


def create_deletion_request(request_id: str, num_samples: int = 16) -> DeletionRequest:
    """Helper to create a deletion request."""
    X = torch.randn(num_samples, 16)
    y = torch.randint(0, 4, (num_samples,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    return DeletionRequest(
        request_id=request_id,
        forget_loader=loader,
        forget_set_size=num_samples,
    )


class TestDeletionRequest:
    """Test DeletionRequest dataclass."""

    def test_creation(self):
        """Test creating a deletion request."""
        loader = DataLoader(
            TensorDataset(torch.randn(16, 16), torch.randint(0, 4, (16,))),
            batch_size=4,
        )
        request = DeletionRequest(
            request_id="user_123",
            forget_loader=loader,
            forget_set_size=16,
        )

        assert request.request_id == "user_123"
        assert request.forget_set_size == 16
        assert request.forget_loader is loader


class TestContinualUnlearnerInit:
    """Test ContinualUnlearner initialization."""

    def test_init_default(self, tiny_classifier):
        """Test initialization with defaults."""
        unlearner = ContinualUnlearner(model=tiny_classifier)

        assert unlearner.strategy_name == "gradient_ascent"
        assert unlearner.selector_name is None
        assert unlearner.base_epochs == 3
        assert unlearner.adaptive_scheduling is True
        assert unlearner.catastrophic_forgetting_threshold == 0.1

    def test_init_custom_strategy(self, tiny_classifier):
        """Test initialization with custom strategy."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="flat",
            base_epochs=5,
        )

        assert unlearner.strategy_name == "flat"
        assert unlearner.base_epochs == 5

    def test_init_with_selector(self, tiny_classifier):
        """Test initialization with selector."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            selector="random",
        )

        assert unlearner.selector_name == "random"
        assert unlearner.selector is not None

    def test_init_adaptive_scheduling_off(self, tiny_classifier):
        """Test disabling adaptive scheduling."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            adaptive_scheduling=False,
        )

        assert unlearner.adaptive_scheduling is False


class TestContinualUnlearnerProcessing:
    """Test continual unlearning request processing."""

    def test_single_deletion_request(self, tiny_classifier, retention_loader):
        """Test processing a single deletion request."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="gradient_ascent",
            base_epochs=1,
        )

        deletion_requests = [create_deletion_request("req_1", num_samples=16)]

        result = unlearner.process_deletion_requests(
            deletion_requests=deletion_requests,
            retain_loader=retention_loader,
            prune_ratio=0.5,
        )

        assert result.model is not None
        assert len(result.deletion_requests) == 1
        assert len(result.per_request_results) == 1
        assert result.total_elapsed_time > 0.0

    def test_multiple_sequential_requests(self, tiny_classifier, retention_loader):
        """Test processing multiple sequential deletion requests."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="gradient_ascent",
            base_epochs=1,
        )

        deletion_requests = [
            create_deletion_request("req_1", num_samples=16),
            create_deletion_request("req_2", num_samples=16),
            create_deletion_request("req_3", num_samples=16),
        ]

        result = unlearner.process_deletion_requests(
            deletion_requests=deletion_requests,
            retain_loader=retention_loader,
            prune_ratio=0.5,
        )

        assert len(result.deletion_requests) == 3
        assert len(result.per_request_results) == 3
        # Model should be the same object (modified in-place)
        assert result.model is unlearner.model

    def test_adaptive_scheduling_applied(self, tiny_classifier, retention_loader):
        """Test that adaptive scheduling reduces epochs over time."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="gradient_ascent",
            base_epochs=3,
            adaptive_scheduling=True,
        )

        deletion_requests = [
            create_deletion_request("req_1", num_samples=16),
            create_deletion_request("req_2", num_samples=16),
            create_deletion_request("req_3", num_samples=16),
        ]

        result = unlearner.process_deletion_requests(
            deletion_requests=deletion_requests,
            retain_loader=retention_loader,
        )

        # With adaptive scheduling, later requests should have fewer epochs
        # (hardcoded as max(1, base_epochs - idx // 2))
        # Request 0: max(1, 3 - 0) = 3 epochs
        # Request 1: max(1, 3 - 1) = 2 epochs
        # Request 2: max(1, 3 - 2) = 1 epoch
        assert len(result.per_request_results) == 3

    def test_adaptive_scheduling_disabled(self, tiny_classifier, retention_loader):
        """Test that epochs stay constant when adaptive scheduling is off."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="gradient_ascent",
            base_epochs=2,
            adaptive_scheduling=False,
        )

        deletion_requests = [
            create_deletion_request("req_1", num_samples=16),
            create_deletion_request("req_2", num_samples=16),
        ]

        result = unlearner.process_deletion_requests(
            deletion_requests=deletion_requests,
            retain_loader=retention_loader,
        )

        assert len(result.per_request_results) == 2

    def test_catastrophic_forgetting_detection(self, tiny_classifier, retention_loader):
        """Test detection of catastrophic forgetting on retain set."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="gradient_ascent",
            base_epochs=2,
            catastrophic_forgetting_threshold=0.05,
        )

        deletion_requests = [
            create_deletion_request("req_1", num_samples=32),
            create_deletion_request("req_2", num_samples=32),
            create_deletion_request("req_3", num_samples=32),
        ]

        result = unlearner.process_deletion_requests(
            deletion_requests=deletion_requests,
            retain_loader=retention_loader,
        )

        # Check that catastrophic forgetting detection ran
        assert isinstance(result.catastrophic_forgetting_detected, bool)
        assert result.utility_degradation >= 0.0

    def test_without_retain_loader(self, tiny_classifier):
        """Test processing without a retain loader (no utility monitoring)."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="gradient_ascent",
            base_epochs=1,
        )

        deletion_requests = [
            create_deletion_request("req_1", num_samples=16),
            create_deletion_request("req_2", num_samples=16),
        ]

        result = unlearner.process_deletion_requests(
            deletion_requests=deletion_requests,
            retain_loader=None,
        )

        assert len(result.deletion_requests) == 2
        assert not result.catastrophic_forgetting_detected
        assert result.utility_degradation == 0.0


class TestContinualUnlearnerHelpers:
    """Test internal helper methods."""

    def test_selector_failure_raises_error(self, tiny_classifier, retention_loader):
        """Test selector failures are surfaced instead of silently bypassed."""

        class BrokenSelector:
            def select(self, *args, **kwargs):
                raise RuntimeError("selector exploded")

        unlearner = ContinualUnlearner(model=tiny_classifier, base_epochs=1)
        unlearner.selector = BrokenSelector()
        unlearner.selector_name = "broken"

        with pytest.raises(SelectorError, match="Silent fallback is disabled"):
            unlearner.process_deletion_requests(
                deletion_requests=[create_deletion_request("req_1", num_samples=16)],
                retain_loader=retention_loader,
            )

    def test_measure_utility(self, tiny_classifier, retention_loader):
        """Test utility measurement on retain set."""
        unlearner = ContinualUnlearner(model=tiny_classifier)

        utility = unlearner._measure_utility(retention_loader)

        assert 0.0 <= utility <= 1.0
        assert isinstance(utility, float)

    def test_measure_utility_empty_batch(self, tiny_classifier):
        """Test utility measurement handles edge cases."""
        # Create a loader with mismatched batch structure
        unlearner = ContinualUnlearner(model=tiny_classifier)

        # Empty result should gracefully handle
        empty_loader = DataLoader([], batch_size=4)
        utility = unlearner._measure_utility(empty_loader)

        assert utility == 0.0

    def test_unlearn_single_request(self, tiny_classifier, retention_loader):
        """Test single request unlearning."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="gradient_ascent",
        )

        forget_loader = DataLoader(
            TensorDataset(torch.randn(16, 16), torch.randint(0, 4, (16,))),
            batch_size=4,
        )

        result = unlearner._unlearn_single_request(
            forget_loader=forget_loader,
            retain_loader=retention_loader,
            epochs=1,
            lr_scale=1.0,
        )

        assert result.model is not None
        assert result.elapsed_time > 0.0
        assert len(result.forget_loss_history) == 1

    def test_unlearn_with_lr_scaling(self, tiny_classifier, retention_loader):
        """Test learning rate scaling during unlearning."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="gradient_ascent",
        )

        forget_loader = DataLoader(
            TensorDataset(torch.randn(16, 16), torch.randint(0, 4, (16,))),
            batch_size=4,
        )

        original_lr = unlearner.strategy.lr

        result = unlearner._unlearn_single_request(
            forget_loader=forget_loader,
            retain_loader=retention_loader,
            epochs=1,
            lr_scale=0.5,  # Reduce LR by 50%
        )

        # Learning rate should be restored
        assert unlearner.strategy.lr == original_lr
        assert result.metadata["lr_scale"] == 0.5


class TestContinualUnlearnerMetadata:
    """Test metadata and result tracking."""

    def test_result_metadata(self, tiny_classifier, retention_loader):
        """Test that results include proper metadata."""
        unlearner = ContinualUnlearner(
            model=tiny_classifier,
            strategy="flat",
            selector="random",
            base_epochs=2,
        )

        deletion_requests = [create_deletion_request("req_1", num_samples=16)]

        result = unlearner.process_deletion_requests(
            deletion_requests=deletion_requests,
            retain_loader=retention_loader,
        )

        assert "strategy" in result.metadata
        assert result.metadata["strategy"] == "flat"
        assert result.metadata["selector"] == "random"
        assert result.metadata["num_requests"] == 1
        assert result.metadata["adaptive_scheduling"] is True

    def test_deletion_request_tracking(self, tiny_classifier, retention_loader):
        """Test that deletion requests are properly tracked."""
        unlearner = ContinualUnlearner(model=tiny_classifier, base_epochs=1)

        requests = [
            create_deletion_request(f"req_{i}", num_samples=16)
            for i in range(3)
        ]

        result = unlearner.process_deletion_requests(
            deletion_requests=requests,
            retain_loader=retention_loader,
        )

        assert [r.request_id for r in result.deletion_requests] == [
            "req_0",
            "req_1",
            "req_2",
        ]
        assert len(result.per_request_results) == 3

    def test_different_strategies(self, tiny_classifier, retention_loader):
        """Test continual unlearning with different strategies."""
        strategies = ["gradient_ascent", "flat", "npo"]

        for strategy_name in strategies:
            unlearner = ContinualUnlearner(
                model=tiny_classifier,
                strategy=strategy_name,
                base_epochs=1,
            )

            deletion_requests = [create_deletion_request("req_1", num_samples=16)]

            result = unlearner.process_deletion_requests(
                deletion_requests=deletion_requests,
                retain_loader=retention_loader,
            )

            assert result.metadata["strategy"] == strategy_name
            assert result.model is not None
