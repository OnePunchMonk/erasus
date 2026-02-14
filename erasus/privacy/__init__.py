"""
erasus.privacy — Differential privacy and certification.
"""

from erasus.privacy.accountant import PrivacyAccountant
from erasus.privacy.gradient_clipping import GradientClipper, clip_grad_norm_, calibrate_noise
from erasus.privacy.secure_aggregation import SecureAggregator, create_secure_aggregator

__all__ = [
    "PrivacyAccountant",
    "GradientClipper",
    "clip_grad_norm_",
    "calibrate_noise",
    "SecureAggregator",
    "create_secure_aggregator",
]
