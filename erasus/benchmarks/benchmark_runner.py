"""
Comprehensive Benchmark Runner for Unlearning Evaluation.

Orchestrates end-to-end benchmarking pipeline:
1. Load dataset (TOFU)
2. Run unlearning strategy
3. Evaluate on TOFU metrics
4. Evaluate on general benchmarks (MMLU, GSM8K, etc.)
5. Aggregate and report results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.benchmarks.tofu_loader import TOFULoader, TOFUEvaluator
from erasus.benchmarks.lm_eval_integration import (
    LMEvalBenchmark,
    BenchmarkComparison,
)
from erasus.core.registry import strategy_registry


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results."""

    strategy_name: str
    tofu_metrics: Dict[str, float] = field(default_factory=dict)
    general_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    utility_degradation: float = 0.0
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """
    End-to-end benchmark runner for unlearning evaluation.

    Pipeline:
    1. Load TOFU dataset (forget/retain/eval splits)
    2. Fine-tune or unlearn using specified strategy
    3. Evaluate on TOFU benchmark (forget effectiveness, utility)
    4. Evaluate on general benchmarks (MMLU, etc.)
    5. Compare results and report
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize benchmark runner.

        Parameters
        ----------
        model : nn.Module
            Model to evaluate.
        tokenizer : PreTrainedTokenizer
            Tokenizer for text processing.
        device : str
            Device to use ("cpu" or "cuda").
        output_dir : str, optional
            Directory to save results.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir

        # Loaders
        self.tofu_loader = TOFULoader(
            tokenizer=tokenizer,
            max_length=512,
            batch_size=16,
        )
        self.tofu_evaluator = TOFUEvaluator(model, device)

    def run_unlearning_benchmark(
        self,
        strategy_name: str,
        num_forget: int = 64,
        num_retain: int = 64,
        num_eval: int = 32,
        epochs: int = 3,
        prune_ratio: float = 0.1,
        general_tasks: Optional[List[str]] = None,
        num_fewshot: int = 0,
        eval_limit: Optional[int] = None,
        **strategy_kwargs: Any,
    ) -> BenchmarkResult:
        """
        Run complete unlearning benchmark.

        Parameters
        ----------
        strategy_name : str
            Strategy to use ("gradient_ascent", "flat", "npo", etc.).
        num_forget : int
            Size of forget set.
        num_retain : int
            Size of retain set.
        num_eval : int
            Size of evaluation set.
        epochs : int
            Training epochs.
        prune_ratio : float
            Coreset selection ratio.
        general_tasks : list of str, optional
            General benchmarks to run (default: ["mmlu", "gsm8k"]).
        num_fewshot : int
            Few-shot examples for general benchmarks.
        eval_limit : int, optional
            Limit examples for faster evaluation.
        **strategy_kwargs : dict
            Extra keyword arguments for strategy.

        Returns
        -------
        BenchmarkResult
            Complete evaluation results.
        """
        import time

        start_time = time.time()

        if general_tasks is None:
            general_tasks = ["mmlu", "gsm8k"]

        # Step 1: Load TOFU dataset
        print(f"Loading TOFU dataset...")
        forget_loader, retain_loader, eval_loader = self.tofu_loader.load_synthetic_tofu(
            num_forget=num_forget,
            num_retain=num_retain,
            num_eval=num_eval,
        )

        # Step 2: Measure baseline performance
        print(f"Measuring baseline TOFU performance...")
        baseline_metrics = self.tofu_evaluator.compute_unlearning_metrics(
            forget_loader, retain_loader
        )

        # Step 3: Run unlearning strategy
        print(f"Running unlearning strategy: {strategy_name}")
        self._run_unlearning(
            strategy_name=strategy_name,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            prune_ratio=prune_ratio,
            **strategy_kwargs,
        )

        # Step 4: Evaluate on TOFU
        print(f"Evaluating on TOFU benchmark...")
        tofu_metrics = self.tofu_evaluator.compute_unlearning_metrics(
            forget_loader, retain_loader
        )

        # Step 5: Evaluate on general benchmarks
        general_metrics = {}
        print(f"Evaluating on general benchmarks...")
        lm_eval = LMEvalBenchmark(self.model, self.tokenizer, self.device)
        for task in general_tasks:
            try:
                print(f"  - {task}...")
                results = lm_eval.run_benchmark(
                    task,
                    num_fewshot=num_fewshot,
                    limit=eval_limit,
                )
                general_metrics[task] = results
            except Exception as e:
                print(f"    Error: {e}")
                general_metrics[task] = {"error": str(e)}

        # Step 6: Compute degradation metrics
        utility_degradation = self._compute_degradation(
            baseline_metrics, tofu_metrics
        )

        elapsed_time = time.time() - start_time

        result = BenchmarkResult(
            strategy_name=strategy_name,
            tofu_metrics=tofu_metrics,
            general_metrics=general_metrics,
            utility_degradation=utility_degradation,
            total_time=elapsed_time,
            metadata={
                "baseline_metrics": baseline_metrics,
                "num_forget": num_forget,
                "num_retain": num_retain,
                "num_eval": num_eval,
                "epochs": epochs,
                "prune_ratio": prune_ratio,
            },
        )

        return result

    def _run_unlearning(
        self,
        strategy_name: str,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        epochs: int,
        prune_ratio: float,
        **strategy_kwargs: Any,
    ) -> None:
        """Run unlearning with specified strategy."""
        strategy_cls = strategy_registry.get(strategy_name)
        strategy = strategy_cls(**strategy_kwargs)

        self.model, _, _ = strategy.unlearn(
            model=self.model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
        )

    def _compute_degradation(
        self,
        baseline: Dict[str, float],
        unlearned: Dict[str, float],
    ) -> float:
        """Compute utility degradation metric."""
        if "retain_loss" not in baseline or "retain_loss" not in unlearned:
            return 0.0

        baseline_loss = baseline["retain_loss"]
        unlearned_loss = unlearned["retain_loss"]

        degradation = (unlearned_loss - baseline_loss) / (baseline_loss + 1e-8)
        return max(0.0, degradation)

    def compare_strategies(
        self,
        strategies: List[str],
        num_forget: int = 64,
        num_retain: int = 64,
        num_eval: int = 32,
        epochs: int = 3,
        **kwargs: Any,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple unlearning strategies.

        Parameters
        ----------
        strategies : list of str
            Strategy names to compare.
        num_forget : int
            Forget set size.
        num_retain : int
            Retain set size.
        num_eval : int
            Evaluation set size.
        epochs : int
            Training epochs per strategy.
        **kwargs : dict
            Extra arguments passed to run_unlearning_benchmark.

        Returns
        -------
        dict of BenchmarkResult
            Results for each strategy.
        """
        results = {}

        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Evaluating strategy: {strategy}")
            print(f"{'='*60}")

            try:
                result = self.run_unlearning_benchmark(
                    strategy_name=strategy,
                    num_forget=num_forget,
                    num_retain=num_retain,
                    num_eval=num_eval,
                    epochs=epochs,
                    **kwargs,
                )
                results[strategy] = result
            except Exception as e:
                print(f"Error with strategy {strategy}: {e}")
                results[strategy] = None

        return results

    def print_results(self, result: BenchmarkResult) -> None:
        """Pretty-print benchmark results."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS: {result.strategy_name}")
        print(f"{'='*60}")

        print(f"\nTOFU Metrics:")
        for metric, value in result.tofu_metrics.items():
            print(f"  {metric:25s}: {value:8.4f}")

        print(f"\nUtility Degradation: {result.utility_degradation:8.4f}")
        print(f"Total Time: {result.total_time:8.2f}s")

        if result.general_metrics:
            print(f"\nGeneral Benchmarks:")
            for task, metrics in result.general_metrics.items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    print(f"  {task}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"    {metric:20s}: {value:8.4f}")

    def save_results(self, results: Dict[str, BenchmarkResult]) -> None:
        """Save results to output directory."""
        if self.output_dir is None:
            return

        import json
        import os

        os.makedirs(self.output_dir, exist_ok=True)

        # Summary
        summary = {}
        for strategy_name, result in results.items():
            if result is not None:
                summary[strategy_name] = {
                    "tofu_metrics": result.tofu_metrics,
                    "utility_degradation": result.utility_degradation,
                    "total_time": result.total_time,
                }

        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {summary_path}")
