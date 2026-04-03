"""
erasus.evaluation — Adversarial evaluation and robustness testing.

This package provides tools that go beyond standard metrics to verify
whether unlearning *actually* worked under adversarial conditions.

Modules
-------
adversarial
    Cross-prompt leakage, keyword injection, paraphrase robustness.
relearning
    Benign fine-tuning attacks, quantization attacks, LoRA relearning.
verification_suite
    Unified runner that combines standard metrics, adversarial tests,
    and robustness checks into a single comprehensive report.
"""

from erasus.evaluation.adversarial import (
    AdversarialEvaluator,
    BaseAdversarialTest,
    CrossPromptLeakageTest,
    KeywordInjectionTest,
    MultilingualLeakageTest,
    ParaphraseRobustnessTest,
    PromptEngineeringAttack,
)
from erasus.evaluation.relearning import (
    RelearningRobustnessEvaluator,
    BenignFinetuningAttack,
    QuantizationAttack,
    LoRARelearningAttack,
    PromptExtractionAttack,
)
from erasus.evaluation.relearning.base import BaseRelearningAttack
from erasus.evaluation.verification_suite import UnlearningVerificationSuite
from erasus.evaluation.benchmark_protocol import (
    UnlearningBenchmark,
    BenchmarkReport,
    MetricResult,
    MetricSpec,
)

__all__ = [
    "AdversarialEvaluator",
    "BaseAdversarialTest",
    "CrossPromptLeakageTest",
    "KeywordInjectionTest",
    "MultilingualLeakageTest",
    "ParaphraseRobustnessTest",
    "PromptEngineeringAttack",
    "RelearningRobustnessEvaluator",
    "BaseRelearningAttack",
    "BenignFinetuningAttack",
    "QuantizationAttack",
    "LoRARelearningAttack",
    "PromptExtractionAttack",
    "UnlearningVerificationSuite",
    "UnlearningBenchmark",
    "BenchmarkReport",
    "MetricResult",
    "MetricSpec",
]
