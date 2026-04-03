"""
Temporal unlearning benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable, List, Sequence


@dataclass
class TemporalBenchmarkRecord:
    text: str
    label: str
    timestamp: datetime


class TemporalUnlearningBenchmark:
    """Split forget/retain data by a rolling time window."""

    def __init__(self, window_days: int = 30, reference_time: datetime | None = None) -> None:
        self.window_days = window_days
        self.reference_time = reference_time or datetime.now(UTC)

    @property
    def cutoff(self) -> datetime:
        return self.reference_time - timedelta(days=self.window_days)

    def split_records(
        self,
        records: Sequence[TemporalBenchmarkRecord],
    ) -> tuple[list[TemporalBenchmarkRecord], list[TemporalBenchmarkRecord]]:
        forget = [record for record in records if record.timestamp >= self.cutoff]
        retain = [record for record in records if record.timestamp < self.cutoff]
        return forget, retain

    def summary(self, records: Sequence[TemporalBenchmarkRecord]) -> dict[str, object]:
        forget, retain = self.split_records(records)
        return {
            "window_days": self.window_days,
            "reference_time": self.reference_time.isoformat(),
            "cutoff": self.cutoff.isoformat(),
            "forget_count": len(forget),
            "retain_count": len(retain),
        }
