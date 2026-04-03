"""
Tests for the standard benchmark suite and temporal benchmark helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from erasus.benchmarks.standard_suite import StandardBenchmarkSuite
from erasus.benchmarks.temporal import TemporalBenchmarkRecord, TemporalUnlearningBenchmark


def test_standard_benchmark_suite_runs(tmp_path):
    suite = StandardBenchmarkSuite()
    report = suite.run()

    assert len(report.entries) == 4
    assert all(e.status == "ok" for e in report.entries)

    json_path = tmp_path / "suite.json"
    md_path = tmp_path / "suite.md"
    suite.save_report(report, json_path, md_path)

    assert json_path.exists()
    assert md_path.exists()


def test_temporal_benchmark_split():
    now = datetime.now(UTC)
    benchmark = TemporalUnlearningBenchmark(window_days=30, reference_time=now)
    records = [
        TemporalBenchmarkRecord("old", "retain", now - timedelta(days=45)),
        TemporalBenchmarkRecord("new", "forget", now - timedelta(days=5)),
    ]

    forget, retain = benchmark.split_records(records)
    assert len(forget) == 1
    assert len(retain) == 1
