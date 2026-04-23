"""Regression snapshots for ``core/surface/`` modules.

Covers: ``surface/smooth`` (tensor B-spline fit + evaluate) and
``surface/corr`` (de-jumped Pearson + co-jump moments).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from blksch.core.surface.corr import estimate_correlation
from blksch.core.surface.smooth import (
    SurfaceSmoother, SurfaceSmootherParams,
)
from blksch.schemas import LogitState, SurfacePoint

from tests.unit.snapshot._helpers import assert_matches_snapshot

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# surface/smooth
# ---------------------------------------------------------------------------


def test_surface_smoother_fit_and_evaluate_snapshot():
    """Deterministic (τ, m) grid → fit → evaluate on a 3×3 test grid."""
    rng = np.random.default_rng(5)
    taus = [60.0, 180.0, 600.0, 1800.0, 3600.0]
    ms = [-1.0, -0.5, 0.0, 0.5, 1.0]
    surface: list[SurfacePoint] = []
    for tau in taus:
        for m in ms:
            # Smooth ground truth: σ_b = 0.05 + 0.02·|m| - 0.005·log(tau/3600)
            sigma = float(0.05 + 0.02 * abs(m) - 0.005 * np.log(tau / 3600.0))
            lam = float(0.005 + 0.002 * abs(m))
            s2j = float(0.004 + 0.001 * abs(m))
            # Small deterministic jitter so the fit has something to smooth.
            sigma += float(rng.normal(0.0, 0.002))
            lam = max(0.0, lam + float(rng.normal(0.0, 0.0005)))
            s2j = max(0.0, s2j + float(rng.normal(0.0, 0.0003)))
            surface.append(SurfacePoint(
                token_id="surf", tau=tau, m=m,
                sigma_b=abs(sigma), **{"lambda": lam}, s2_j=s2j,
                ts=T0,
            ))

    smoother = SurfaceSmoother(
        token_id="surf",
        params=SurfaceSmootherParams(n_tau=8, n_m=8, smoothing=0.0, degree=3),
    )
    smoother.fit(surface)

    probe_grid = [
        (60.0, 0.0), (60.0, 0.5),
        (180.0, -0.5), (180.0, 0.0), (180.0, 0.5),
        (1800.0, 0.0),
        (3600.0, -1.0), (3600.0, 1.0),
    ]
    evaluations = []
    for tau, m in probe_grid:
        sp = smoother.evaluate(tau=tau, m=m)
        evaluations.append({
            "tau": tau, "m": m,
            "sigma_b": sp.sigma_b,
            "lambda": sp.lambda_,
            "s2_j": sp.s2_j,
        })
    assert_matches_snapshot(evaluations, "surface/smooth_fit_evaluate.json")


# ---------------------------------------------------------------------------
# surface/corr
# ---------------------------------------------------------------------------


def _make_correlated_states(
    rho: float, n: int, *, seed: int = 0, sigma: float = 0.02,
) -> tuple[list[LogitState], list[LogitState]]:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]]) * (sigma ** 2)
    incs = rng.multivariate_normal(np.zeros(2), cov, size=n - 1)
    x = np.concatenate([[0.0], np.cumsum(incs[:, 0])])
    y = np.concatenate([[0.0], np.cumsum(incs[:, 1])])
    states_i = [
        LogitState(token_id="i", x_hat=float(x[k]), sigma_eta2=sigma ** 2,
                   ts=T0 + timedelta(seconds=k))
        for k in range(n)
    ]
    states_j = [
        LogitState(token_id="j", x_hat=float(y[k]), sigma_eta2=sigma ** 2,
                   ts=T0 + timedelta(seconds=k))
        for k in range(n)
    ]
    return states_i, states_j


def test_surface_corr_canonical_pair_snapshot():
    """Three paths: clean ρ=0.5, clean ρ=-0.3, and one with injected
    jumps for the co-jump count."""
    results = {}

    si, sj = _make_correlated_states(rho=0.5, n=3000, seed=1)
    e = estimate_correlation(si, sj, [], [])
    results["rho_0_5_clean"] = {
        "rho": e.rho, "lambda": e.co_jump_lambda, "m2": e.co_jump_m2,
    }

    si, sj = _make_correlated_states(rho=-0.3, n=3000, seed=2)
    e = estimate_correlation(si, sj, [], [])
    results["rho_neg_0_3_clean"] = {
        "rho": e.rho, "lambda": e.co_jump_lambda, "m2": e.co_jump_m2,
    }

    # Injected co-jumps at indices 500, 1000, 1500 in both streams.
    si, sj = _make_correlated_states(rho=0.3, n=3000, seed=3)
    cojump_idx = [500, 1000, 1500]
    jumps_i = [T0 + timedelta(seconds=k) for k in cojump_idx]
    jumps_j = [T0 + timedelta(seconds=k) for k in cojump_idx]
    e = estimate_correlation(si, sj, jumps_i, jumps_j)
    results["rho_0_3_with_co_jumps"] = {
        "rho": e.rho, "lambda": e.co_jump_lambda, "m2": e.co_jump_m2,
    }

    assert_matches_snapshot(results, "surface/corr_paths.json")
