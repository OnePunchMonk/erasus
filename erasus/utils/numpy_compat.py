"""
NumPy version compatibility (e.g. NumPy 2.x removed ``numpy.trapz`` from the main namespace).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def trapezoid(y: Any, x: Any = None) -> Any:
    """
    Trapezoidal rule integration.

    Uses ``numpy.trapezoid`` on NumPy 2+, and ``numpy.trapz`` on older releases.
    """
    if hasattr(np, "trapezoid"):
        if x is None:
            return np.trapezoid(y)
        return np.trapezoid(y, x)
    # NumPy 1.x
    if x is None:
        return np.trapz(y)
    return np.trapz(y, x)
