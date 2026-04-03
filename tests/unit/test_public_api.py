"""
Tests for public API stability metadata.
"""

from __future__ import annotations


def test_root_public_api_status():
    import erasus

    assert erasus.PUBLIC_API_STATUS["ErasusUnlearner"] == "stable"
    assert erasus.PUBLIC_API_STATUS["UnlearningTrainer"] == "experimental"


def test_strategy_public_api_status():
    import erasus.strategies as strategies

    assert strategies.PUBLIC_API_STATUS["NPOStrategy"] == "stable"
    assert strategies.PUBLIC_API_STATUS["ActivationSteeringStrategy"] == "experimental"


def test_benchmark_public_api_status():
    import erasus.benchmarks as benchmarks

    assert benchmarks.PUBLIC_API_STATUS["BenchmarkRunner"] == "stable"
    assert benchmarks.PUBLIC_API_STATUS["StandardBenchmarkSuite"] == "experimental"
