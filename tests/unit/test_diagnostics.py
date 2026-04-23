"""Unit tests for core/diagnostics.py — paper §6.1 filter-health checks.

Covers:
  * ljung_box — accepts iid, rejects AR(1), handles edge cases
  * qq_normal — high quantile correlation on Gaussian, low on uniform
  * variance_consistency — tolerance math, boundary cases
  * run_diagnostics orchestrator — bundles into DiagnosticsReport
  * DiagnosticsReport.all_pass — gate logic
"""

from __future__ import annotations

import numpy as np
import pytest

from blksch.core.diagnostics import (
    DiagnosticsReport,
    ljung_box,
    qq_normal,
    run_diagnostics,
    variance_consistency,
)


# ---------------------------------------------------------------------------
# Ljung-Box
# ---------------------------------------------------------------------------


class TestLjungBox:
    def test_accepts_iid_gaussian(self) -> None:
        """N(0,1) iid at length 2000 should fail-to-reject at 5%."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2000)
        stat, pvalue, lags = ljung_box(x, lags=20)
        assert pvalue > 0.05
        assert lags == 20
        assert stat >= 0.0

    def test_rejects_ar1_series(self) -> None:
        rng = np.random.default_rng(2)
        n = 2000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        stat, pvalue, _ = ljung_box(x, lags=20)
        assert pvalue < 0.05

    def test_rejects_ma1_series(self) -> None:
        rng = np.random.default_rng(3)
        n = 2000
        eps = rng.standard_normal(n + 1)
        x = eps[1:] + 0.6 * eps[:-1]   # MA(1)
        _, pvalue, _ = ljung_box(x, lags=10)
        assert pvalue < 0.05

    def test_default_lags(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.standard_normal(50)
        stat, pvalue, lags = ljung_box(x)  # no explicit lags
        assert lags == min(10, 50 // 5)

    def test_requires_enough_samples(self) -> None:
        with pytest.raises(ValueError):
            ljung_box(np.array([1.0, 2.0]))

    def test_lags_must_be_less_than_n(self) -> None:
        with pytest.raises(ValueError):
            ljung_box(np.arange(10, dtype=float), lags=10)

    def test_constant_series_returns_p_one(self) -> None:
        """All-identical residuals → vacuously no serial correlation."""
        _, p, _ = ljung_box(np.ones(50))
        assert p == 1.0


# ---------------------------------------------------------------------------
# Q-Q / Shapiro-Wilk
# ---------------------------------------------------------------------------


class TestQQNormal:
    def test_gaussian_has_high_quantile_correlation(self) -> None:
        rng = np.random.default_rng(11)
        z = rng.standard_normal(1000)
        qc, sw_p, passes = qq_normal(z)
        assert qc > 0.99
        assert passes

    def test_heavy_tailed_rejected(self) -> None:
        rng = np.random.default_rng(12)
        # Laplace is heavier-tailed than Gaussian.
        x = rng.laplace(0, 1, 1500)
        qc, sw_p, passes = qq_normal(x)
        # Shapiro-Wilk should reject at the default alpha=0.05.
        assert not passes

    def test_skewed_rejected(self) -> None:
        rng = np.random.default_rng(13)
        x = rng.exponential(1.0, 1000)  # right-skewed
        _, _, passes = qq_normal(x)
        assert not passes

    def test_requires_min_samples(self) -> None:
        with pytest.raises(ValueError):
            qq_normal(np.array([0.0, 1.0]))

    def test_constant_series_is_vacuous_pass(self) -> None:
        qc, sw_p, passes = qq_normal(np.ones(10))
        assert qc == 1.0


# ---------------------------------------------------------------------------
# Realized-vs-implied
# ---------------------------------------------------------------------------


class TestVarianceConsistency:
    def test_exact_match_passes(self) -> None:
        assert variance_consistency(1.0, 1.0, tolerance_pct=0.01)

    def test_within_tolerance_passes(self) -> None:
        assert variance_consistency(1.15, 1.0, tolerance_pct=0.2)

    def test_outside_tolerance_fails(self) -> None:
        assert not variance_consistency(1.5, 1.0, tolerance_pct=0.2)

    def test_asymmetric_deviation_direction_does_not_matter(self) -> None:
        """|r - i|/i < tol ⇒ pass, regardless of which is larger."""
        assert variance_consistency(0.85, 1.0, tolerance_pct=0.2)
        assert variance_consistency(1.15, 1.0, tolerance_pct=0.2)

    def test_both_zero_passes(self) -> None:
        assert variance_consistency(0.0, 0.0, tolerance_pct=0.1)

    def test_zero_implied_with_nonzero_realized_fails(self) -> None:
        assert not variance_consistency(1.0, 0.0, tolerance_pct=0.5)

    def test_negative_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            variance_consistency(-0.1, 1.0)
        with pytest.raises(ValueError):
            variance_consistency(1.0, -0.1)

    def test_negative_tolerance_raises(self) -> None:
        with pytest.raises(ValueError):
            variance_consistency(1.0, 1.0, tolerance_pct=-0.1)


# ---------------------------------------------------------------------------
# Orchestrator + DiagnosticsReport
# ---------------------------------------------------------------------------


class TestRunDiagnostics:
    def test_clean_gaussian_all_pass(self) -> None:
        rng = np.random.default_rng(21)
        r = rng.standard_normal(500)
        report = run_diagnostics(
            r, realized_variance=1.0, implied_variance=1.0,
        )
        assert isinstance(report, DiagnosticsReport)
        assert report.ljung_box_pass
        assert report.qq_shapiro_wilk_pass
        assert report.variance_consistency_pass
        assert report.all_pass

    def test_ar_residuals_flagged(self) -> None:
        rng = np.random.default_rng(22)
        n = 1000
        r = np.zeros(n)
        for t in range(1, n):
            r[t] = 0.6 * r[t - 1] + rng.standard_normal()
        report = run_diagnostics(r, realized_variance=1.0, implied_variance=1.0)
        assert not report.ljung_box_pass
        assert not report.all_pass

    def test_variance_mismatch_flagged(self) -> None:
        rng = np.random.default_rng(23)
        r = rng.standard_normal(500)
        report = run_diagnostics(
            r, realized_variance=3.0, implied_variance=1.0,
            variance_tolerance_pct=0.2,
        )
        assert report.ljung_box_pass
        assert report.qq_shapiro_wilk_pass
        assert not report.variance_consistency_pass
        assert not report.all_pass

    def test_non_gaussian_residuals_flagged(self) -> None:
        rng = np.random.default_rng(24)
        r = rng.exponential(1.0, 500)  # skewed
        report = run_diagnostics(r, realized_variance=1.0, implied_variance=1.0)
        assert not report.qq_shapiro_wilk_pass

    def test_report_fields_populated(self) -> None:
        rng = np.random.default_rng(25)
        r = rng.standard_normal(500)
        report = run_diagnostics(
            r, realized_variance=1.0, implied_variance=1.0, lags=15,
        )
        assert report.ljung_box_lags == 15
        assert report.ljung_box_statistic >= 0.0
        assert 0.0 <= report.ljung_box_pvalue <= 1.0
        assert -1.0 <= report.qq_quantile_correlation <= 1.0
        assert 0.0 <= report.qq_shapiro_wilk_pvalue <= 1.0
        assert report.variance_rel_error >= 0.0
        assert report.variance_tolerance_pct == 0.2  # default
