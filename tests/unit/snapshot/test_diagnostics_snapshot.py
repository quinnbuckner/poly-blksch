"""Regression snapshot for ``core/diagnostics.py``.

Fixed-seed whitened-residual stream, + fixed realized/implied variance
pair, → ``run_diagnostics`` → DiagnosticsReport pinned in the fixture.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from blksch.core.diagnostics import (
    ljung_box, qq_normal, run_diagnostics, variance_consistency,
)

from tests.unit.snapshot._helpers import assert_matches_snapshot

pytestmark = pytest.mark.unit


def _residuals(n: int = 400, seed: int = 17) -> np.ndarray:
    """Mild AR(1) + Gaussian noise so the Ljung-Box has something to
    measure but we're not at the white-noise limit."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, size=n)
    r = np.zeros(n)
    alpha = 0.08
    for i in range(1, n):
        r[i] = alpha * r[i - 1] + e[i]
    return r


def test_run_diagnostics_canonical_snapshot():
    r = _residuals(n=400, seed=17)
    report = run_diagnostics(
        r,
        realized_variance=0.1234,
        implied_variance=0.1310,
        variance_tolerance_pct=0.2,
        alpha=0.05,
    )
    assert_matches_snapshot(asdict(report), "diagnostics/run_report.json")


def test_ljung_box_primitive_snapshot():
    r = _residuals(n=400, seed=21)
    q, p, lags = ljung_box(r, lags=12, alpha=0.05)
    assert_matches_snapshot(
        {"q": q, "p": p, "lags": lags},
        "diagnostics/ljung_box_primitive.json",
    )


def test_qq_normal_primitive_snapshot():
    rng = np.random.default_rng(99)
    # Slightly heavy-tailed (Student-t df=6) — Shapiro should notice.
    r = rng.standard_t(df=6.0, size=300)
    qc, p, ok = qq_normal(r, alpha=0.05)
    assert_matches_snapshot(
        {"q_correlation": float(qc), "shapiro_p": float(p), "shapiro_pass": bool(ok)},
        "diagnostics/qq_normal_primitive.json",
    )


def test_variance_consistency_primitive_snapshot():
    cases = [
        (0.10, 0.10, 0.05),   # exact
        (0.10, 0.11, 0.05),   # within 5% tolerance (10% rel would pass)
        (0.10, 0.20, 0.05),   # 100% over, fails at 5% tol
        (0.00, 0.00, 0.05),   # both zero → pass
    ]
    out = [
        {
            "realized": r, "implied": im, "tol": t,
            "pass": variance_consistency(r, im, tolerance_pct=t),
        }
        for r, im, t in cases
    ]
    assert_matches_snapshot(out, "diagnostics/variance_consistency_primitive.json")
