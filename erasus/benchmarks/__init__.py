"""
erasus.benchmarks — Machine unlearning benchmark suite.

Provides comprehensive evaluation infrastructure:
- TOFU benchmark (fictional author QA unlearning)
- lm-evaluation-harness integration (MMLU, GSM8K, etc.)
- Real model evaluation (HuggingFace models)
- Benchmark runner (end-to-end evaluation pipeline)
"""

from erasus.benchmarks.tofu_loader import (
    TOFUDataset,
    TOFULoader,
    TOFUEvaluator,
)
from erasus.benchmarks.lm_eval_integration import (
    LMEvalWrapper,
    LMEvalBenchmark,
    BenchmarkComparison,
    PostUnlearningBenchmarkSuite,
)
from erasus.benchmarks.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkResult,
)
from erasus.benchmarks.real_model_eval import (
    HuggingFaceModelLoader,
    RealModelBenchmark,
    RealModelComparison,
)

__all__ = [
    "TOFUDataset",
    "TOFULoader",
    "TOFUEvaluator",
    "LMEvalWrapper",
    "LMEvalBenchmark",
    "BenchmarkComparison",
    "PostUnlearningBenchmarkSuite",
    "BenchmarkRunner",
    "BenchmarkResult",
    "HuggingFaceModelLoader",
    "RealModelBenchmark",
    "RealModelComparison",
]
