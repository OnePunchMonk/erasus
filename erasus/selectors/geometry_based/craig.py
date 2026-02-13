"""
CRAIG Selector.

This file is a duplicate/alias for `grad_match.py`.
CRAIG (Coresets for Data-efficient Training) is implemented in `erasus.selectors.gradient_based.grad_match`.
"""

from erasus.selectors.gradient_based.grad_match import GradMatchSelector
from erasus.core.registry import selector_registry

# Alias registration if needed, or just point users to grad_match
try:
    selector_registry.register("craig")(GradMatchSelector)
except ValueError:
    pass # Already registered
