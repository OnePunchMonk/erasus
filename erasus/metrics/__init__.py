"""
erasus.metrics — Evaluation and Privacy Metrics.
"""

from erasus.core.registry import metric_registry

# Import and register
from erasus.metrics.accuracy import AccuracyMetric
from erasus.metrics.membership_inference import MembershipInferenceMetric
from erasus.metrics.perplexity import PerplexityMetric
from erasus.metrics.fid import FIDMetric
from erasus.metrics.retrieval_metrics import ZeroShotAccuracyMetric

# Registering explicitly if they don't have decorators
# BaseMetric uses inheritance, but registry needs decorators?
# Actually, the metric_registry is defined in core.registry but rarely used with decorators in current codebase?
# Let's check `accuracy.py`. It inherits BaseMetric.
# BaseUnlearner loops over `metrics` list passed as instances, NOT resolving by name usually.
# However, if we want CLI support `erasus evaluate --metrics accuracy,mia`, we need a registry.
# `accuracy.py` does NOT use @metric_registry.register.

# Let's fix that pattern. For CLI usage, we need a registry.
# I'll manually register them here or add decorators to the files.
# Adding decorators is cleaner but `accuracy.py` was already viewed and unmodified.
# I'll update it here.

for name, cls in [
    ("accuracy", AccuracyMetric),
    ("mia", MembershipInferenceMetric),
    ("perplexity", PerplexityMetric),
    ("fid", FIDMetric),
    ("zero_shot", ZeroShotAccuracyMetric),
]:
    try:
        metric_registry.register(name)(cls)
    except ValueError:
        pass # Already registered

__all__ = [
    "AccuracyMetric",
    "MembershipInferenceMetric",
    "PerplexityMetric",
    "FIDMetric",
    "ZeroShotAccuracyMetric",
]
