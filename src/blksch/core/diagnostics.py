"""Standard filter-health diagnostics (paper §6.1).

Three checks the paper says must pass before a calibration is trusted:

  1. Ljung-Box on whitened Kalman innovations — residuals should be serially
     uncorrelated under the null of a correctly-specified filter.
  2. Q-Q against N(0,1) — standardized residuals should be approximately
     Gaussian (away from flagged jump intervals).
  3. Realized-vs-implied variance consistency — the σ̂_b² surface, integrated
     against S'(x)², should match the microstructure-cleaned realized
     p-variance within a tolerance.

Each function is pure and independent; the `DiagnosticsReport` dataclass
bundles the three outputs for downstream reporting.

Note on dependencies: Ljung-Box is implemented directly rather than imported.
SciPy does not ship `acorr_ljungbox` and we do not want a statsmodels
dependency for one formula. The Box-Pierce/Ljung-Box closed form is
short; we validate against independent references in the unit tests.

Pure: no I/O, no asyncio. NumPy + scipy.stats only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "DiagnosticsReport",
    "ljung_box",
    "qq_normal",
    "variance_consistency",
    "run_diagnostics",
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticsReport:
    """Internal diagnostics container — intentionally not in schemas.py.

    Track A's calibration pipeline constructs one per calibration run and
    logs it; Track B never consumes this directly, so it does not need
    cross-track schema status.
    """

    # Ljung-Box
    ljung_box_lags: int
    ljung_box_statistic: float
    ljung_box_pvalue: float
    ljung_box_pass: bool

    # Q-Q
    qq_quantile_correlation: float
    qq_shapiro_wilk_pvalue: float
    qq_shapiro_wilk_pass: bool

    # Realized vs implied variance
    realized_variance: float
    implied_variance: float
    variance_rel_error: float
    variance_tolerance_pct: float
    variance_consistency_pass: bool

    @property
    def all_pass(self) -> bool:
        return (
            self.ljung_box_pass
            and self.qq_shapiro_wilk_pass
            and self.variance_consistency_pass
        )


# ---------------------------------------------------------------------------
# Ljung-Box
# ---------------------------------------------------------------------------


def ljung_box(
    residuals: np.ndarray,
    lags: int | None = None,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """Ljung-Box test for serial correlation (paper §6.1).

    Q = n(n+2) · Σ_{k=1}^{h} ρ̂_k² / (n - k),       Q ~ χ²(h) under H_0

    Args:
      residuals: 1-D array of filter innovations (ideally whitened).
      lags:      number of autocorrelation lags to include. Default:
                 min(10, n // 5) — the conventional choice for short samples.
      alpha:     significance level for pass/fail; returned pvalue is the
                 actual p-value, not the (pass, fail) boolean.

    Returns:
      (Q_statistic, p_value, lags_used). High p-value ⇒ fail-to-reject ⇒
      residuals look white.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    n = r.size
    if n < 4:
        raise ValueError(f"Ljung-Box needs ≥ 4 residuals, got {n}")
    h = int(lags) if lags is not None else max(1, min(10, n // 5))
    if h < 1:
        raise ValueError(f"lags must be ≥ 1, got {h}")
    if h >= n:
        raise ValueError(f"lags ({h}) must be < n ({n})")

    # Sample autocorrelation at lag k (biased estimator consistent with the
    # Ljung-Box formula). Subtract the sample mean so the test is robust to
    # non-zero-mean innovations.
    r_centered = r - r.mean()
    c0 = float(np.dot(r_centered, r_centered))
    if c0 == 0.0:
        # All residuals identical — vacuously white; return p=1.
        return 0.0, 1.0, h

    rho = np.empty(h, dtype=float)
    for k in range(1, h + 1):
        c_k = float(np.dot(r_centered[:-k], r_centered[k:]))
        rho[k - 1] = c_k / c0

    denom = np.array([n - k for k in range(1, h + 1)], dtype=float)
    q = n * (n + 2) * float(np.sum((rho ** 2) / denom))
    p = float(stats.chi2.sf(q, df=h))
    return q, p, h


# ---------------------------------------------------------------------------
# Q-Q normality
# ---------------------------------------------------------------------------


def qq_normal(residuals: np.ndarray, *, alpha: float = 0.05) -> tuple[float, float, bool]:
    """Q-Q check of standardized residuals against N(0, 1) (paper §6.1).

    Returns (quantile_correlation, shapiro_wilk_pvalue, passes_shapiro).

    * quantile_correlation: Pearson r between sorted standardized residuals
      and the theoretical normal quantiles at matching plotting positions
      (Blom's p_k = (k - 0.5) / n). High (≥ 0.98) is expected for well-
      behaved Gaussian innovations.
    * shapiro_wilk_pvalue: Shapiro-Wilk normality test p-value. Pass ⇔
      pvalue > alpha (fail-to-reject).
    """
    r = np.asarray(residuals, dtype=float).ravel()
    n = r.size
    if n < 3:
        raise ValueError(f"Q-Q needs ≥ 3 residuals, got {n}")

    mean = r.mean()
    std = r.std(ddof=1) if n > 1 else 1.0
    if std == 0.0:
        return 1.0, 1.0, True  # vacuous

    z = (r - mean) / std
    z_sorted = np.sort(z)
    # Blom plotting positions: p_k = (k - 0.5) / n, k=1..n
    pk = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(pk)
    qc = float(np.corrcoef(z_sorted, theoretical)[0, 1])

    sw_stat, sw_p = stats.shapiro(r)
    sw_p = float(sw_p)
    return qc, sw_p, sw_p > alpha


# ---------------------------------------------------------------------------
# Realized-vs-implied variance consistency
# ---------------------------------------------------------------------------


def variance_consistency(
    realized_variance: float,
    implied_variance: float,
    tolerance_pct: float = 0.2,
) -> bool:
    """Realized vs implied variance check (paper §6.1).

    Passes iff |realized - implied| / max(implied, ε) ≤ tolerance_pct.
    Tolerance defaults to 20% per the paper's rule-of-thumb for the
    σ̂_b² surface against microstructure-cleaned realized variance.

    Both inputs must be non-negative. A zero implied variance with
    non-zero realized variance always fails.
    """
    if realized_variance < 0.0:
        raise ValueError(f"realized_variance must be ≥ 0, got {realized_variance}")
    if implied_variance < 0.0:
        raise ValueError(f"implied_variance must be ≥ 0, got {implied_variance}")
    if tolerance_pct < 0.0:
        raise ValueError(f"tolerance_pct must be ≥ 0, got {tolerance_pct}")
    eps = 1.0e-12
    if implied_variance < eps and realized_variance < eps:
        return True
    denom = max(implied_variance, eps)
    rel = abs(realized_variance - implied_variance) / denom
    return rel <= tolerance_pct


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_diagnostics(
    residuals: np.ndarray,
    *,
    realized_variance: float,
    implied_variance: float,
    lags: int | None = None,
    variance_tolerance_pct: float = 0.2,
    alpha: float = 0.05,
) -> DiagnosticsReport:
    """Bundle all three checks into a single report.

    `residuals` is the whitened Kalman innovation stream from core/filter/.
    The paper requires all three to pass before the calibration is shipped
    to the quoting engine.
    """
    lb_stat, lb_p, lb_lags = ljung_box(residuals, lags=lags, alpha=alpha)
    qc, sw_p, sw_pass = qq_normal(residuals, alpha=alpha)
    var_pass = variance_consistency(
        realized_variance, implied_variance, tolerance_pct=variance_tolerance_pct,
    )
    rel = abs(realized_variance - implied_variance) / max(implied_variance, 1e-12)

    return DiagnosticsReport(
        ljung_box_lags=lb_lags,
        ljung_box_statistic=lb_stat,
        ljung_box_pvalue=lb_p,
        ljung_box_pass=lb_p > alpha,
        qq_quantile_correlation=qc,
        qq_shapiro_wilk_pvalue=sw_p,
        qq_shapiro_wilk_pass=sw_pass,
        realized_variance=realized_variance,
        implied_variance=implied_variance,
        variance_rel_error=float(rel),
        variance_tolerance_pct=variance_tolerance_pct,
        variance_consistency_pass=var_pass,
    )
