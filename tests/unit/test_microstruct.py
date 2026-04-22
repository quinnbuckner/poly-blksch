"""Unit tests for `core/filter/microstruct.py` (paper §5.1 eq 10).

The heteroskedastic measurement-noise model:
    sigma_eta^2(t) = a_0 + a_1 s_t^2 + a_2 d_t^{-1} + a_3 r_t + a_4 iota_t^2
fit by robust regression on short-horizon squared innovations.
"""

from __future__ import annotations

import numpy as np
import pytest

microstruct = pytest.importorskip(
    "blksch.core.filter.microstruct",
    reason="core/filter/microstruct.py not yet implemented",
)

pytestmark = pytest.mark.unit


class TestHeteroskedasticNoiseModel:
    def test_recovers_known_coefficients_on_synthetic(self) -> None:
        """Simulate eq (10) with known (a_0..a_4); fit must recover within 10%."""
        pytest.skip("Stub: fill in when microstruct lands")

    def test_clips_sigma_eta2_to_floor_and_ceiling(self) -> None:
        """Paper §5.1: sigma_eta^2 clipped to [sigma_lo, sigma_hi] to avoid
        pathological filter gain."""
        pytest.skip("Stub")

    def test_robust_regression_resists_outliers(self) -> None:
        """Injected 1% outliers must not shift coefficients > 2x std error."""
        pytest.skip("Stub")

    def test_handles_missing_covariates(self) -> None:
        """When depth or trade rate are missing, fall back gracefully to base rate."""
        pytest.skip("Stub")

    def test_output_is_strictly_positive(self) -> None:
        """sigma_eta^2 must be > 0 for every observation regardless of inputs."""
        pytest.skip("Stub")
