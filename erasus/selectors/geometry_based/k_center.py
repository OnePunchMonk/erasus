"""
K-Center Selector Alias.

This module is an alias for `erasus.selectors.geometry_based.kcenter`.
"""

from erasus.selectors.geometry_based.kcenter import KCenterSelector
from erasus.core.registry import selector_registry

# Re-register under legacy name just in case, though kcenter registers itself as "kcenter".
# If "k_center" string is needed, register it.
try:
    selector_registry.register("k_center")(KCenterSelector)
except ValueError:
    pass
