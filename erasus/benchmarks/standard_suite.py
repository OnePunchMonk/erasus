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
            "| Benchmark | Status | Verdict / notes |",
            "| --- | --- | --- |",
        ]
        for entry in self.entries:
            if entry.status == "error":
                cell = entry.details.get("error", "")
            else:
                cell = entry.details.get("verdict") or entry.details.get("notes", "")
            lines.append(f"| {entry.name} | {entry.status} | {cell} |")
        return "\n".join(lines)


class StandardBenchmarkSuite:
    """Run TOFU, MUSE, WMDP, and custom benchmark hooks behind one interface."""

    DEFAULT_BENCHMARKS = ("tofu", "muse", "wmdp", "custom")

    def __init__(self, benchmarks: Sequence[str] | None = None) -> None:
        self.benchmarks = list(benchmarks or self.DEFAULT_BENCHMARKS)

    def run(self) -> BenchmarkSuiteReport:
        from erasus.benchmarks.micro_protocol import run_micro_protocol

        entries = []
        for benchmark in self.benchmarks:
            try:
                details = run_micro_protocol(benchmark, epochs=1)
                entries.append(
                    BenchmarkSuiteEntry(
                        name=benchmark,
                        status=details.get("status", "ok"),
                        details=details,
                    ),
                )
            except Exception as exc:
                entries.append(
                    BenchmarkSuiteEntry(
                        name=benchmark,
                        status="error",
                        details={"error": str(exc)},
                    ),
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
