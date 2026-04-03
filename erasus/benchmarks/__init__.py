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
from erasus.benchmarks.standard_suite import (
    BenchmarkSuiteEntry,
    BenchmarkSuiteReport,
    StandardBenchmarkSuite,
)
from erasus.benchmarks.temporal import (
    TemporalBenchmarkRecord,
    TemporalUnlearningBenchmark,
)

STABLE_EXPORTS = [
    "TOFUDataset",
    "TOFULoader",
    "TOFUEvaluator",
    "LMEvalWrapper",
    "LMEvalBenchmark",
    "BenchmarkRunner",
    "BenchmarkResult",
]

EXPERIMENTAL_EXPORTS = [
    "BenchmarkComparison",
    "PostUnlearningBenchmarkSuite",
    "HuggingFaceModelLoader",
    "RealModelBenchmark",
    "RealModelComparison",
    "BenchmarkSuiteEntry",
    "BenchmarkSuiteReport",
    "StandardBenchmarkSuite",
    "TemporalBenchmarkRecord",
    "TemporalUnlearningBenchmark",
]

PUBLIC_API_STATUS = {
    **{name: "stable" for name in STABLE_EXPORTS},
    **{name: "experimental" for name in EXPERIMENTAL_EXPORTS},
}

__all__ = STABLE_EXPORTS + EXPERIMENTAL_EXPORTS
