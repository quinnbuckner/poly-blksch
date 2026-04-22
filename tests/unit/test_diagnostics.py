"""Unit tests for `core/diagnostics.py` (paper §5.1 Diagnostics).

Mandatory sanity checks for Track A's filter + EM pipeline:
    (i) Ljung-Box on residuals
    (ii) Q-Q plot near-Gaussian away from jumps
    (iii) realized p-variance matches Int S'(x)^2 sigma_b^2 dt + jumps
"""

from __future__ import annotations

import numpy as np
import pytest

diagnostics = pytest.importorskip(
    "blksch.core.diagnostics",
    reason="core/diagnostics.py not yet implemented",
)

pytestmark = pytest.mark.unit


class TestLjungBox:
    def test_accepts_iid_series(self) -> None:
        """Ljung-Box on N(0,1) iid samples must not reject at 5%."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2000)
        stat, pvalue = diagnostics.ljung_box(x, lags=20)  # type: ignore[attr-defined]
        assert pvalue > 0.05

    def test_rejects_serially_correlated_series(self) -> None:
        """AR(1) with phi=0.5 must reject at 5%."""
        rng = np.random.default_rng(2)
        n = 2000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        stat, pvalue = diagnostics.ljung_box(x, lags=20)  # type: ignore[attr-defined]
        assert pvalue < 0.05


class TestQQ:
    def test_qq_returns_nearly_45_slope_on_gaussian(self) -> None:
        pytest.skip("Stub")


class TestRealizedVsImplied:
    def test_realized_p_variance_matches_implied_within_tolerance(self) -> None:
        """Paper §5.1 check: sum S'(x)^2 sigma_b^2 dt + Sum (Delta p)^2 on jumps
        should match realized (Delta p)^2 within 20%."""
        pytest.skip("Stub")
