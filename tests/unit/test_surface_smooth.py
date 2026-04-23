"""Unit tests for core/surface/smooth.py — §5.3 tensor B-spline smoother.

All tests generate synthetic SurfacePoint inputs (we do not wait on Track A's
EM outputs). We verify:
  * fit/evaluate lifecycle
  * boundedness (no negative σ_b, λ, s²_J ever returned)
  * monotonicity in τ for σ_b on a ground-truth monotone surface
  * noise rejection (smoother recovers a known surface through Gaussian noise)
  * degenerate input fallbacks (single-τ observations, single-m observations)
  * clamping to observed support (no wild extrapolation)
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone

import numpy as np
import pytest

from blksch.core.surface.smooth import SurfaceSmoother, SurfaceSmootherParams
from blksch.schemas import SurfacePoint


TS = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
TOK = "0xA"


def _sp(tau: float, m: float, sigma_b: float = 0.3, lam: float = 0.1, s2j: float = 0.0) -> SurfacePoint:
    return SurfacePoint(
        token_id=TOK, tau=tau, m=m, sigma_b=sigma_b,
        **{"lambda": lam}, s2_j=s2j, ts=TS,
    )


def synthetic_surface_points(
    n_tau: int = 8, n_m: int = 6,
    tau_range: tuple[float, float] = (600.0, 7200.0),
    m_range: tuple[float, float] = (-1.5, 1.5),
    sigma_b_fn=lambda tau, m: 0.1 + 0.0002 * tau + 0.05 * abs(m),
    lambda_fn=lambda tau, m: 0.05 + 0.0001 * tau,
    s2j_fn=lambda tau, m: 0.005 + 0.0005 * abs(m),
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> list[SurfacePoint]:
    """Grid of ground-truth SurfacePoints with optional Gaussian noise.

    `sigma_b_fn` is monotone-non-decreasing in τ by construction so the
    monotonicity projection has nothing to fix on the ground truth.
    """
    rng = random.Random(seed)
    taus = np.linspace(*tau_range, n_tau)
    ms = np.linspace(*m_range, n_m)
    out: list[SurfacePoint] = []
    for tau in taus:
        for m in ms:
            sb = max(0.0, sigma_b_fn(float(tau), float(m)) + rng.gauss(0.0, noise_sigma))
            lam = max(0.0, lambda_fn(float(tau), float(m)) + rng.gauss(0.0, noise_sigma * 0.5))
            s2 = max(0.0, s2j_fn(float(tau), float(m)) + rng.gauss(0.0, noise_sigma * 0.1))
            out.append(_sp(float(tau), float(m), sb, lam, s2))
    return out


# ---------------------------------------------------------------------------
# Params validation
# ---------------------------------------------------------------------------


class TestParams:
    def test_default_params(self) -> None:
        p = SurfaceSmootherParams()
        assert p.n_tau >= 2 and p.n_m >= 2
        assert p.degree == 3
        assert p.enforce_monotone_tau

    def test_n_axis_positive(self) -> None:
        with pytest.raises(ValueError):
            SurfaceSmootherParams(n_tau=1)
        with pytest.raises(ValueError):
            SurfaceSmootherParams(n_m=1)

    def test_degree_restricted(self) -> None:
        with pytest.raises(ValueError):
            SurfaceSmootherParams(degree=2)  # RectBivariateSpline only supports 1,3,5

    def test_smoothing_nonneg(self) -> None:
        with pytest.raises(ValueError):
            SurfaceSmootherParams(smoothing=-0.1)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_evaluate_before_fit_raises(self) -> None:
        s = SurfaceSmoother(TOK)
        with pytest.raises(RuntimeError):
            s.evaluate(tau=3600.0, m=0.0)

    def test_empty_fit_raises(self) -> None:
        s = SurfaceSmoother(TOK)
        with pytest.raises(ValueError):
            s.fit([])

    def test_wrong_token_filtered(self) -> None:
        s = SurfaceSmoother(TOK)
        other = [
            SurfacePoint(
                token_id="0xOTHER", tau=3600.0, m=0.0, sigma_b=0.3,
                **{"lambda": 0.1}, s2_j=0.0, ts=TS,
            )
        ]
        with pytest.raises(ValueError):
            s.fit(other)

    def test_fit_then_evaluate_returns_surfacepoint(self) -> None:
        s = SurfaceSmoother(TOK)
        s.fit(synthetic_surface_points())
        sp = s.evaluate(tau=3600.0, m=0.0)
        assert isinstance(sp, SurfacePoint)
        assert sp.token_id == TOK
        assert sp.tau == 3600.0
        assert sp.m == 0.0


# ---------------------------------------------------------------------------
# Shape constraints
# ---------------------------------------------------------------------------


class TestBoundedness:
    def test_sigma_b_nonneg_even_with_negative_noise(self) -> None:
        """Synthetic points with heavy noise that would push fit values
        below zero — the smoother must still return σ_b ≥ 0 everywhere."""
        s = SurfaceSmoother(TOK)
        pts = synthetic_surface_points(
            sigma_b_fn=lambda tau, m: 0.02,
            noise_sigma=0.05,
        )
        s.fit(pts)
        for tau in np.linspace(600.0, 7200.0, 10):
            for m in np.linspace(-1.5, 1.5, 10):
                sp = s.evaluate(float(tau), float(m))
                assert sp.sigma_b >= 0.0
                assert sp.lambda_ >= 0.0
                assert sp.s2_j >= 0.0

    def test_all_fields_nonneg_on_query_grid(self) -> None:
        s = SurfaceSmoother(TOK)
        s.fit(synthetic_surface_points())
        for tau, m in [(600, -1.5), (7200, 1.5), (3000, 0.0)]:
            sp = s.evaluate(float(tau), float(m))
            assert sp.sigma_b >= 0
            assert sp.lambda_ >= 0
            assert sp.s2_j >= 0


class TestMonotoneTau:
    def test_monotone_in_tau_on_clean_surface(self) -> None:
        """σ_b is monotone non-decreasing in τ on the ground truth ⇒
        the smoother's output should also be monotone for any m."""
        s = SurfaceSmoother(TOK)
        s.fit(synthetic_surface_points(noise_sigma=0.0))
        taus = np.linspace(600.0, 7200.0, 12)
        for m in (-1.0, 0.0, 1.0):
            values = [s.evaluate(float(t), float(m)).sigma_b for t in taus]
            for a, b in zip(values, values[1:]):
                assert b >= a - 1e-9, f"monotonicity violated at m={m}: {a} → {b}"

    def test_monotone_survives_noise(self) -> None:
        """Even with noise that would break monotonicity locally, the
        post-projection guarantees it."""
        s = SurfaceSmoother(TOK)
        s.fit(synthetic_surface_points(noise_sigma=0.05))
        taus = np.linspace(600.0, 7200.0, 12)
        for m in (-0.5, 0.5):
            values = [s.evaluate(float(t), float(m)).sigma_b for t in taus]
            for a, b in zip(values, values[1:]):
                assert b >= a - 1e-9

    def test_monotone_off_allows_decrease(self) -> None:
        """When enforce_monotone_tau=False the smoother may return a
        decreasing sequence — confirms the projection is actually wired on."""
        s = SurfaceSmoother(
            TOK, params=SurfaceSmootherParams(enforce_monotone_tau=False),
        )
        # Ground truth decreasing in τ.
        s.fit(synthetic_surface_points(
            sigma_b_fn=lambda tau, m: 0.8 - 0.0001 * tau,
            noise_sigma=0.0,
        ))
        taus = np.linspace(600.0, 7200.0, 6)
        values = [s.evaluate(float(t), 0.0).sigma_b for t in taus]
        assert values[-1] < values[0]  # decreased somewhere


# ---------------------------------------------------------------------------
# Accuracy / noise rejection
# ---------------------------------------------------------------------------


class TestNoiseRejection:
    def test_recovers_ground_truth_without_noise(self) -> None:
        """Exact tensor-spline fit should reproduce the ground truth at
        observed grid points to spline precision."""
        s = SurfaceSmoother(TOK)
        pts = synthetic_surface_points(noise_sigma=0.0)
        s.fit(pts)
        for gp in pts[:10]:
            sp = s.evaluate(gp.tau, gp.m)
            assert sp.sigma_b == pytest.approx(gp.sigma_b, abs=1e-3)

    def test_smooths_through_noise(self) -> None:
        """With noisy observations, the smoother should land within
        typical noise of the ground truth."""
        sigma_b_fn = lambda tau, m: 0.1 + 0.0002 * tau + 0.05 * abs(m)
        pts = synthetic_surface_points(sigma_b_fn=sigma_b_fn, noise_sigma=0.05, seed=7)
        s = SurfaceSmoother(
            TOK, params=SurfaceSmootherParams(smoothing=0.5),
        )
        s.fit(pts)
        for tau, m in [(3600.0, 0.0), (2000.0, -0.5), (5000.0, 0.75)]:
            truth = sigma_b_fn(tau, m)
            sp = s.evaluate(tau, m)
            assert abs(sp.sigma_b - truth) < 0.10  # generous bound, not over-fit

    def test_uncertainty_scales_with_noise(self) -> None:
        clean = SurfaceSmoother(TOK)
        clean.fit(synthetic_surface_points(noise_sigma=0.0))

        noisy = SurfaceSmoother(TOK, params=SurfaceSmootherParams(smoothing=0.5))
        noisy.fit(synthetic_surface_points(noise_sigma=0.1, seed=99))

        sp_clean = clean.evaluate(3600.0, 0.0)
        sp_noisy = noisy.evaluate(3600.0, 0.0)
        assert (sp_noisy.uncertainty or 0.0) > (sp_clean.uncertainty or 0.0)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_single_tau_observations(self) -> None:
        """All obs at one τ — smoother falls back to 1-D along m."""
        s = SurfaceSmoother(TOK)
        pts = [_sp(3600.0, m=float(m)) for m in np.linspace(-1.0, 1.0, 6)]
        s.fit(pts)
        sp = s.evaluate(3600.0, 0.0)
        assert sp.sigma_b >= 0

    def test_single_m_observations(self) -> None:
        s = SurfaceSmoother(TOK)
        pts = [_sp(tau=float(tau), m=0.0, sigma_b=0.1 + 0.0001 * tau)
               for tau in np.linspace(600.0, 7200.0, 8)]
        s.fit(pts)
        sp_lo = s.evaluate(1000.0, 0.0)
        sp_hi = s.evaluate(6000.0, 0.0)
        assert sp_hi.sigma_b >= sp_lo.sigma_b - 1e-9  # monotone projection

    def test_single_point_fit(self) -> None:
        s = SurfaceSmoother(TOK)
        s.fit([_sp(3600.0, 0.0, sigma_b=0.3)])
        sp = s.evaluate(3600.0, 0.0)
        # Degenerate input ⇒ constant output at the observed value (after clip).
        assert sp.sigma_b == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Extrapolation clamping
# ---------------------------------------------------------------------------


class TestSupportClamping:
    def test_query_beyond_support_clamped(self) -> None:
        """Queries outside the observed (τ, m) hull should not extrapolate;
        they return the value at the nearest in-support boundary."""
        s = SurfaceSmoother(TOK)
        pts = synthetic_surface_points(
            tau_range=(1000.0, 5000.0), m_range=(-1.0, 1.0),
        )
        s.fit(pts)
        sp_edge = s.evaluate(5000.0, 1.0)
        sp_far = s.evaluate(1e9, 1e9)
        # Both should evaluate to the same clamped corner.
        assert sp_far.sigma_b == pytest.approx(sp_edge.sigma_b, abs=1e-9)
        assert sp_far.tau == sp_edge.tau
        assert sp_far.m == sp_edge.m
