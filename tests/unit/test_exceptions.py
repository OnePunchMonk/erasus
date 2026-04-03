"""
Tests for the Erasus exception and warning policy helpers.
"""

from __future__ import annotations

import pytest


class TestExceptionHierarchy:
    def test_imports(self):
        from erasus.core.exceptions import (
            BenchmarkError,
            CheckpointError,
            DatasetError,
            ErasusError,
            ErasusWarning,
            EvaluationError,
            IntegrationError,
            WarningPolicyError,
            handle_policy,
        )

        assert BenchmarkError is not None
        assert CheckpointError is not None
        assert DatasetError is not None
        assert ErasusError is not None
        assert ErasusWarning is not None
        assert EvaluationError is not None
        assert IntegrationError is not None
        assert WarningPolicyError is not None
        assert handle_policy is not None

    def test_handle_policy_warn(self):
        from erasus.core.exceptions import ErasusWarning, handle_policy

        with pytest.warns(ErasusWarning):
            handle_policy(policy="warn", message="warning")

    def test_handle_policy_error(self):
        from erasus.core.exceptions import ErasusError, handle_policy

        with pytest.raises(ErasusError):
            handle_policy(policy="error", message="boom")

    def test_handle_policy_invalid(self):
        from erasus.core.exceptions import WarningPolicyError, handle_policy

        with pytest.raises(WarningPolicyError):
            handle_policy(policy="nope", message="bad")
