"""Tests for NumPy 1/2 trapezoid compatibility."""

from __future__ import annotations

import numpy as np
import pytest

from erasus.utils.numpy_compat import trapezoid


def test_trapezoid_matches_numpy():
    y = np.array([0.0, 1.0, 1.0])
    x = np.linspace(0.0, 1.0, 3)
    got = trapezoid(y, x)
    if hasattr(np, "trapezoid"):
        want = float(np.trapezoid(y, x))
    else:
        want = float(np.trapz(y, x))
    assert pytest.approx(got) == want
