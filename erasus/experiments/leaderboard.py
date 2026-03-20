"""
Leaderboard System — Track and compare unlearning results across experiments.

Maintains a persistent leaderboard of unlearning methods:
- Model performance on benchmarks
- Strategy effectiveness metrics
- Ranking and filtering capabilities
- Export to JSON/CSV for sharing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import json
import os
from datetime import datetime


@dataclass
class LeaderboardEntry:
    """Entry in the leaderboard."""

    experiment_name: str
    model_name: str
    strategy_name: str
    forget_effectiveness: float
    retain_utility: float
    mmlu_accuracy: Optional[float] = None
    gsm8k_accuracy: Optional[float] = None
    avg_benchmark_score: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def score(self) -> float:
        """Compute overall leaderboard score."""
        # Balance forgetting with utility preservation
        score = (self.forget_effectiveness * 0.5) + (self.retain_utility * 0.5)

        if self.avg_benchmark_score is not None:
            score = score * 0.7 + self.avg_benchmark_score * 0.3

        return score


class Leaderboard:
    """Leaderboard for tracking unlearning results."""

    def __init__(self, save_path: str = "./leaderboard.json") -> None:
        """
        Initialize leaderboard.

        Parameters
        ----------
        save_path : str
            Path to save leaderboard JSON.
        """
        self.save_path = save_path
        self.entries: List[LeaderboardEntry] = []
        self.load()

    def add_entry(self, entry: LeaderboardEntry) -> None:
        """Add entry to leaderboard."""
        self.entries.append(entry)
        self.save()

    def get_top_strategies(
        self,
        model_name: Optional[str] = None,
        k: int = 10,
    ) -> List[LeaderboardEntry]:
        """
        Get top strategies by score.

        Parameters
        ----------
        model_name : str, optional
            Filter by model.
        k : int
            Number of top entries.

        Returns
        -------
        list of LeaderboardEntry
            Top entries sorted by score.
        """
        filtered = self.entries
        if model_name is not None:
            filtered = [e for e in filtered if e.model_name == model_name]

        sorted_entries = sorted(filtered, key=lambda e: e.score(), reverse=True)
        return sorted_entries[:k]

    def get_strategy_stats(self, strategy_name: str) -> Dict[str, float]:
        """
        Get aggregate stats for a strategy.

        Parameters
        ----------
        strategy_name : str
            Strategy name.

        Returns
        -------
        dict
            Statistics (mean, std, min, max scores).
        """
        entries = [e for e in self.entries if e.strategy_name == strategy_name]

        if not entries:
            return {}

        scores = [e.score() for e in entries]

        return {
            "num_runs": len(entries),
            "mean_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "mean_forget_effectiveness": sum(e.forget_effectiveness for e in entries) / len(entries),
            "mean_retain_utility": sum(e.retain_utility for e in entries) / len(entries),
        }

    def compare_strategies(
        self,
        strategies: List[str],
        model_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple strategies.

        Parameters
        ----------
        strategies : list of str
            Strategy names.
        model_name : str, optional
            Filter by model.

        Returns
        -------
        dict
            Stats for each strategy.
        """
        results = {}
        for strategy in strategies:
            entries = [
                e
                for e in self.entries
                if e.strategy_name == strategy
                and (model_name is None or e.model_name == model_name)
            ]

            if entries:
                scores = [e.score() for e in entries]
                results[strategy] = {
                    "count": len(entries),
                    "mean_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                }

        return results

    def save(self) -> None:
        """Save leaderboard to JSON."""
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)

        data = [
            {
                "experiment_name": e.experiment_name,
                "model_name": e.model_name,
                "strategy_name": e.strategy_name,
                "forget_effectiveness": e.forget_effectiveness,
                "retain_utility": e.retain_utility,
                "mmlu_accuracy": e.mmlu_accuracy,
                "gsm8k_accuracy": e.gsm8k_accuracy,
                "score": e.score(),
                "timestamp": e.timestamp,
                "notes": e.notes,
            }
            for e in self.entries
        ]

        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load leaderboard from JSON."""
        if not os.path.exists(self.save_path):
            return

        with open(self.save_path, "r") as f:
            data = json.load(f)

        self.entries = [
            LeaderboardEntry(
                experiment_name=d["experiment_name"],
                model_name=d["model_name"],
                strategy_name=d["strategy_name"],
                forget_effectiveness=d["forget_effectiveness"],
                retain_utility=d["retain_utility"],
                mmlu_accuracy=d.get("mmlu_accuracy"),
                gsm8k_accuracy=d.get("gsm8k_accuracy"),
                timestamp=d.get("timestamp", ""),
                notes=d.get("notes"),
            )
            for d in data
        ]

    def export_csv(self, output_path: str) -> None:
        """Export leaderboard to CSV."""
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "experiment_name",
                    "model_name",
                    "strategy_name",
                    "forget_effectiveness",
                    "retain_utility",
                    "score",
                    "mmlu_accuracy",
                    "gsm8k_accuracy",
                    "timestamp",
                ],
            )
            writer.writeheader()

            for e in sorted(self.entries, key=lambda x: x.score(), reverse=True):
                writer.writerow({
                    "experiment_name": e.experiment_name,
                    "model_name": e.model_name,
                    "strategy_name": e.strategy_name,
                    "forget_effectiveness": f"{e.forget_effectiveness:.4f}",
                    "retain_utility": f"{e.retain_utility:.4f}",
                    "score": f"{e.score():.4f}",
                    "mmlu_accuracy": f"{e.mmlu_accuracy:.4f}" if e.mmlu_accuracy else "N/A",
                    "gsm8k_accuracy": f"{e.gsm8k_accuracy:.4f}" if e.gsm8k_accuracy else "N/A",
                    "timestamp": e.timestamp,
                })

        print(f"Leaderboard exported to {output_path}")

    def print_leaderboard(self, k: int = 20) -> None:
        """Print top entries."""
        print(f"\n{'='*80}")
        print(f"{'UNLEARNING LEADERBOARD':<40} Top {k}")
        print(f"{'='*80}\n")

        for i, entry in enumerate(self.get_top_strategies(k=k), 1):
            print(
                f"{i:2d}. {entry.strategy_name:20s} | "
                f"Score: {entry.score():.4f} | "
                f"Forget: {entry.forget_effectiveness:.4f} | "
                f"Utility: {entry.retain_utility:.4f}"
            )

        print(f"\n{'='*80}\n")
