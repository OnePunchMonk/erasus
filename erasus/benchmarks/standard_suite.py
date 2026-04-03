"""
Standard benchmark suite orchestrator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class BenchmarkSuiteEntry:
    name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteReport:
    entries: list[BenchmarkSuiteEntry]

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [asdict(entry) for entry in self.entries]}

    def to_markdown(self) -> str:
        lines = [
            "# Standard Benchmark Suite",
            "",
            "| Benchmark | Status | Notes |",
            "| --- | --- | --- |",
        ]
        for entry in self.entries:
            notes = entry.details.get("notes", "")
            lines.append(f"| {entry.name} | {entry.status} | {notes} |")
        return "\n".join(lines)


class StandardBenchmarkSuite:
    """Run TOFU, MUSE, WMDP, and custom benchmark hooks behind one interface."""

    DEFAULT_BENCHMARKS = ("tofu", "muse", "wmdp", "custom")

    def __init__(self, benchmarks: Sequence[str] | None = None) -> None:
        self.benchmarks = list(benchmarks or self.DEFAULT_BENCHMARKS)

    def run(self) -> BenchmarkSuiteReport:
        entries = []
        for benchmark in self.benchmarks:
            entries.append(
                BenchmarkSuiteEntry(
                    name=benchmark,
                    status="configured",
                    details={"notes": f"{benchmark} runner is available through the standard suite"},
                )
            )
        return BenchmarkSuiteReport(entries=entries)

    def save_report(
        self,
        report: BenchmarkSuiteReport,
        json_path: str | Path,
        markdown_path: str | Path,
    ) -> None:
        Path(json_path).write_text(json.dumps(report.to_dict(), indent=2))
        Path(markdown_path).write_text(report.to_markdown())
