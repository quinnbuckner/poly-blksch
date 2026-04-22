"""Unit tests for `core/surface/smooth.py` (paper §5.3).

Tensor B-spline (or thin-plate) smoothing of point estimates over (tau, m) with
shape constraints: non-negativity, edge stability, term smoothness.
"""

from __future__ import annotations

import numpy as np
import pytest

surface_smooth = pytest.importorskip(
    "blksch.core.surface.smooth",
    reason="core/surface/smooth.py not yet implemented",
)

pytestmark = pytest.mark.unit


class TestSurfaceSmoothing:
    def test_fit_reproduces_smooth_ground_truth(self) -> None:
        """On a grid with sigma_b(tau, m) = 0.05 + 0.01*sin(tau/300), fit
        must reconstruct within 2% RMSE."""
        pytest.skip("Stub")

    def test_nonnegativity_constraint(self) -> None:
        """Output surface must be >= 0 everywhere, even when noisy inputs
        would otherwise produce negative spline tails."""
        pytest.skip("Stub")

    def test_edge_stability_damps_explosive_curvature(self) -> None:
        """At extreme m (|m| > 6), curvature penalty must prevent blowup."""
        pytest.skip("Stub")

    def test_term_smoothness_relaxes_at_scheduled_news(self) -> None:
        """Near a flagged scheduled news timestamp, the tau-axis penalty is
        locally relaxed so the surface can bump for the jump layer."""
        pytest.skip("Stub")

    def test_uncertainty_bands_via_bootstrap(self) -> None:
        """Returned SurfacePoint.uncertainty should be positive and coverage
        at 1-sigma should be ~68% on a held-out synthetic test."""
        pytest.skip("Stub")
