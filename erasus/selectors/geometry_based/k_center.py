"""
K-Center Selector — backwards-compatible alias.

The canonical implementation lives in ``kcenter.py`` and is registered
as ``"kcenter"``.  This module re-exports ``KCenterSelector`` so that
``from erasus.selectors.geometry_based.k_center import KCenterSelector``
continues to work.
"""

from erasus.selectors.geometry_based.kcenter import KCenterSelector  # noqa: F401
