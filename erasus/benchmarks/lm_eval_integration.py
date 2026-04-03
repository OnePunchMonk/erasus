"""
LM Evaluation Harness Integration — Standard Benchmark Evaluations.

Provides seamless integration with lm-evaluation-harness (lm-eval) for
running standard NLP benchmarks:
- MMLU: Massive Multitask Language Understanding
- GSM8K: Grade School Math
- ARC: AI2 Reasoning Challenge
- HellaSwag: Commonsense Reasoning
- TruthfulQA: Truthfulness Evaluation

This enables evaluating unlearned models on standard benchmarks to ensure
general capability preservation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class LMEvalWrapper:
    """
    Wraps a model for compatibility with lm-evaluation-harness.

    Makes any PyTorch LM compatible with the lm-eval benchmarking framework.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
    ) -> None:
        """
        Initialize wrapper.

        Parameters
        ----------
        model : nn.Module
            Language model to wrap.
        tokenizer : PreTrainedTokenizer
            Tokenizer (e.g., from transformers).
        device : str
            Device to use ("cpu" or "cuda").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

    def _encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        return tokens.to(self.device)

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get logits from model."""
        with torch.no_grad():
            outputs = self.model(tokens)
            if hasattr(outputs, "logits"):
                return outputs.logits
            return outputs

    def greedy_until(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate text greedily until stop token.

        Parameters
        ----------
        requests : list of dict
            Each has keys: 'prompt', 'until' (stop tokens)

        Returns
        -------
        list of str
            Generated completions.
        """
        results = []

        for req in requests:
            prompt = req.get("prompt", "")
            until = req.get("until", ["\n"])

            # Encode prompt
            input_ids = self._encode(prompt)
            max_length = min(input_ids.shape[1] + 100, 512)

            # Generate greedily
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=False,
                top_p=1.0,
                temperature=0.0,
                num_beams=1,
            )

            # Decode
            completion = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            # Truncate at first stop token
            for stop_token in until:
                if stop_token in completion:
                    completion = completion.split(stop_token)[0]

            results.append(completion)

        return results

    def loglikelihood(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Compute log-likelihood of completions.

        Parameters
        ----------
        requests : list of dict
            Each has keys: 'prompt', 'completion'

        Returns
        -------
        list of float
            Log-likelihoods for each completion.
        """
        results = []

        for req in requests:
            prompt = req.get("prompt", "")
            completion = req.get("completion", "")

            # Encode full text
            full_text = prompt + completion
            tokens = self._encode(full_text)

            # Get logits
            with torch.no_grad():
                logits = self._logits(tokens)

            # Compute log-likelihood of completion tokens
            completion_ids = self._encode(completion)[0]
            prompt_length = len(self._encode(prompt)[0])

            log_likelihood = 0.0
            for i, token_id in enumerate(completion_ids):
                token_idx = prompt_length + i
                if token_idx < logits.shape[1]:
                    token_logits = logits[0, token_idx, :]
                    token_log_probs = torch.log_softmax(token_logits, dim=-1)
                    log_likelihood += token_log_probs[token_id].item()

            results.append(log_likelihood)

        return results


class LMEvalBenchmark:
    """
    Run standard benchmarks via lm-evaluation-harness.

    Supports MMLU, GSM8K, ARC, HellaSwag, TruthfulQA, etc.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
    ) -> None:
        """
        Initialize benchmark runner.

        Parameters
        ----------
        model : nn.Module
            Model to evaluate.
        tokenizer : PreTrainedTokenizer
            Tokenizer.
        device : str
            Device to use.
        """
        self.wrapper = LMEvalWrapper(model, tokenizer, device)

    def run_benchmark(
        self,
        task_name: str,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run a single benchmark task.

        Parameters
        ----------
        task_name : str
            Task to run:
            - "mmlu": Massive Multitask Language Understanding
            - "gsm8k": Grade School Math
            - "arc_challenge": AI2 Reasoning Challenge
            - "hellaswag": Commonsense Reasoning
            - "truthfulqa_mc1": Truthfulness (multiple choice)
        num_fewshot : int
            Number of few-shot examples (default 0, zero-shot).
        limit : int, optional
            Max examples to evaluate.

        Returns
        -------
        dict
            Task metrics (accuracy, etc.)

        Note
        ----
        Requires lm-evaluation-harness installed:
        pip install lm-eval
        """
        try:
            from lm_eval import simple_evaluate
        except ImportError:
            raise ImportError(
                "lm-evaluation-harness not found. "
                "Install with: pip install lm-eval"
            )

        # Map task names
        task_map = {
            "mmlu": "mmlu",
            "gsm8k": "gsm8k",
            "arc": "arc_challenge",
            "hellaswag": "hellaswag",
            "truthfulqa": "truthfulqa_mc1",
        }

        task = task_map.get(task_name, task_name)

        # Run evaluation
        results = simple_evaluate(
            model=self.wrapper,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=1,
        )

        return results.get("results", {}).get(task, {})

    def run_benchmark_suite(
        self,
        tasks: List[str] = None,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run multiple benchmarks.

        Parameters
        ----------
        tasks : list of str
            Tasks to run (default: ["mmlu", "gsm8k", "arc", "hellaswag"])
        num_fewshot : int
            Few-shot examples.
        limit : int, optional
            Max examples per task.

        Returns
        -------
        dict of dict
            Results for each task.
        """
        if tasks is None:
            tasks = ["mmlu", "gsm8k", "arc", "hellaswag"]

        all_results = {}

        for task in tasks:
            print(f"Running {task}...")
            try:
                results = self.run_benchmark(task, num_fewshot, limit)
                all_results[task] = results
            except Exception as e:
                print(f"Error running {task}: {e}")
                all_results[task] = {"error": str(e)}

        return all_results


class PostUnlearningBenchmarkSuite:
    """
    Standard post-unlearning benchmark wrapper for common LLM tasks.

    Provides a single entry point for evaluating utility preservation on
    MMLU, GSM8K, TruthfulQA, HellaSwag, and ARC after unlearning.
    """

    DEFAULT_TASKS = ["mmlu", "gsm8k", "truthfulqa", "hellaswag", "arc"]

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
    ) -> None:
        self.benchmark = LMEvalBenchmark(model, tokenizer, device=device)

    def run(
        self,
        tasks: Optional[List[str]] = None,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run the default post-unlearning benchmark suite."""
        selected_tasks = tasks or list(self.DEFAULT_TASKS)
        return self.benchmark.run_benchmark_suite(
            tasks=selected_tasks,
            num_fewshot=num_fewshot,
            limit=limit,
        )


class BenchmarkComparison:
    """
    Compare model performance before/after unlearning.

    Tracks capability preservation and unlearning effectiveness.
    """

    def __init__(self, baseline_model: nn.Module, tokenizer: Any) -> None:
        """
        Initialize comparison with baseline.

        Parameters
        ----------
        baseline_model : nn.Module
            Original model (before unlearning).
        tokenizer : PreTrainedTokenizer
            Tokenizer.
        """
        self.baseline_benchmark = LMEvalBenchmark(baseline_model, tokenizer)
        self.baseline_results = None

    def measure_baseline(
        self,
        tasks: List[str] = None,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
    ) -> None:
        """
        Measure baseline performance.

        Parameters
        ----------
        tasks : list of str
            Tasks to measure.
        num_fewshot : int
            Few-shot examples.
        limit : int, optional
            Max examples.
        """
        print("Measuring baseline performance...")
        self.baseline_results = self.baseline_benchmark.run_benchmark_suite(
            tasks, num_fewshot, limit
        )

    def compare_unlearned(
        self,
        unlearned_model: nn.Module,
        tokenizer: Any,
        tasks: List[str] = None,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare unlearned model to baseline.

        Parameters
        ----------
        unlearned_model : nn.Module
            Model after unlearning.
        tokenizer : PreTrainedTokenizer
            Tokenizer.
        tasks : list of str
            Tasks to evaluate.
        num_fewshot : int
            Few-shot examples.
        limit : int, optional
            Max examples.

        Returns
        -------
        dict
            Performance degradation metrics.
        """
        if self.baseline_results is None:
            print("Warning: baseline not measured. Measuring now...")
            self.measure_baseline(tasks, num_fewshot, limit)

        unlearned_benchmark = LMEvalBenchmark(unlearned_model, tokenizer)
        unlearned_results = unlearned_benchmark.run_benchmark_suite(
            tasks, num_fewshot, limit
        )

        # Compute degradation
        degradation = {}
        for task_name in (tasks or []):
            baseline = self.baseline_results.get(task_name, {})
            unlearned = unlearned_results.get(task_name, {})

            task_degradation = {}
            for metric_name in baseline:
                if metric_name in unlearned:
                    baseline_val = baseline[metric_name]
                    unlearned_val = unlearned[metric_name]

                    if isinstance(baseline_val, (int, float)):
                        degradation_pct = (
                            (baseline_val - unlearned_val) / (baseline_val + 1e-8) * 100
                        )
                        task_degradation[metric_name] = degradation_pct

            degradation[task_name] = task_degradation

        return degradation
