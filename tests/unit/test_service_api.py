"""
Tests for the FastAPI service helpers.
"""

from __future__ import annotations

from erasus.service.api import list_registered_strategies, plan_unlearning_request


def test_list_registered_strategies():
    strategies = list_registered_strategies()
    assert "gradient_ascent" in strategies


def test_plan_unlearning_request():
    plan = plan_unlearning_request({"strategy": "npo", "selector": "random"})
    assert plan["status"] == "accepted"
    assert plan["strategy"] == "npo"
