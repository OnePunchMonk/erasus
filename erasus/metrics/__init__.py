"""
erasus.metrics — Evaluation and Privacy Metrics.
"""

from erasus.core.registry import metric_registry

# Import and register — core metrics
from erasus.metrics.accuracy import AccuracyMetric
from erasus.metrics.membership_inference import MembershipInferenceMetric
from erasus.metrics.perplexity import PerplexityMetric
from erasus.metrics.fid import FIDMetric
from erasus.metrics.retrieval_metrics import ZeroShotAccuracyMetric

# New metric modules
from erasus.metrics.metric_suite import MetricSuite
from erasus.metrics.forgetting.mia import MIAMetric
from erasus.metrics.forgetting.mia_variants import LiRAMetric, LabelOnlyMIAMetric
from erasus.metrics.forgetting.confidence import ConfidenceMetric
from erasus.metrics.forgetting.feature_distance import FeatureDistanceMetric
from erasus.metrics.efficiency.time_complexity import TimeComplexityMetric
from erasus.metrics.efficiency.memory_usage import MemoryUsageMetric
from erasus.metrics.privacy.differential_privacy import DPEvaluationMetric

# Sprint A metrics
from erasus.metrics.forgetting.activation_analysis import ActivationAnalysis
from erasus.metrics.forgetting.backdoor_activation import BackdoorActivation
from erasus.metrics.efficiency.speedup import SpeedupMetric
from erasus.metrics.efficiency.flops import FLOPsMetric

# Sprint F metrics
from erasus.metrics.forgetting.extraction_attack import ExtractionAttackMetric
from erasus.metrics.utility.clip_score import CLIPScoreMetric
from erasus.metrics.utility.bleu import BLEUMetric
from erasus.metrics.utility.rouge import ROUGEMetric
from erasus.metrics.utility.inception_score import InceptionScoreMetric
from erasus.metrics.utility.downstream_tasks import DownstreamTaskMetric
from erasus.metrics.privacy.epsilon_delta import EpsilonDeltaMetric
from erasus.metrics.privacy.privacy_audit import PrivacyAuditMetric
from erasus.metrics.benchmarks import ErasusBenchmark

# Register all metrics for CLI / registry-based resolution
for name, cls in [
    ("accuracy", AccuracyMetric),
    ("mia", MembershipInferenceMetric),
    ("perplexity", PerplexityMetric),
    ("fid", FIDMetric),
    ("zero_shot", ZeroShotAccuracyMetric),
    ("mia_full", MIAMetric),
    ("lira", LiRAMetric),
    ("label_only_mia", LabelOnlyMIAMetric),
    ("confidence", ConfidenceMetric),
    ("feature_distance", FeatureDistanceMetric),
    ("time_complexity", TimeComplexityMetric),
    ("memory_usage", MemoryUsageMetric),
    ("dp_evaluation", DPEvaluationMetric),
]:
    try:
        metric_registry.register(name)(cls)
    except ValueError:
        pass  # Already registered

__all__ = [
    "AccuracyMetric",
    "MembershipInferenceMetric",
    "PerplexityMetric",
    "FIDMetric",
    "ZeroShotAccuracyMetric",
    "MetricSuite",
    "MIAMetric",
    "LiRAMetric",
    "LabelOnlyMIAMetric",
    "ConfidenceMetric",
    "FeatureDistanceMetric",
    "TimeComplexityMetric",
    "MemoryUsageMetric",
    "DPEvaluationMetric",
    # Sprint A
    "ActivationAnalysis",
    "BackdoorActivation",
    "SpeedupMetric",
    "FLOPsMetric",
    # Sprint F
    "ExtractionAttackMetric",
    "CLIPScoreMetric",
    "BLEUMetric",
    "ROUGEMetric",
    "InceptionScoreMetric",
    "DownstreamTaskMetric",
    "EpsilonDeltaMetric",
    "PrivacyAuditMetric",
    "ErasusBenchmark",
]
