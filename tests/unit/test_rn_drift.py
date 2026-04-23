"""Unit tests for ``core/em/rn_drift`` (paper §3.2 eq 3 + outer EM loop)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from blksch.core.em.increments import MixtureParams
from blksch.core.em.rn_drift import (
    DEFAULT_MC_SAMPLES,
    DEFAULT_MU_CAP_PER_SEC,
    CalibrationResult,
    RNDriftConfig,
    compile_mu_fn,
    em_calibrate,
)
from blksch.schemas import LogitState

pytestmark = pytest.mark.unit


# ---------- Helpers ----------


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _t0() -> datetime:
    return datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)


def _states_from_path(x: np.ndarray, *, dt: float = 1.0) -> list[LogitState]:
    t0 = _t0()
    return [
        LogitState(token_id="t", x_hat=float(x[i]), sigma_eta2=0.01,
                   ts=t0 + timedelta(seconds=i * dt))
        for i in range(len(x))
    ]


def _simulate_jd_with_drift(
    rng: np.random.Generator,
    *,
    n: int,
    sigma_b: float,
    lambda_jump: float,
    s_J: float,
    dt: float = 1.0,
    x0: float = 0.0,
    mu_scalar: float = 0.0,
) -> np.ndarray:
    """Forward Euler JD path with constant drift ``mu_scalar``."""
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        step = mu_scalar * dt + sigma_b * np.sqrt(dt) * rng.normal()
        if rng.random() < lambda_jump * dt:
            step += rng.normal(0.0, s_J)
        x[i] = x[i - 1] + step
    return x


# ---------- Config validation ----------


def test_config_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        RNDriftConfig(x_grid_min=1.0, x_grid_max=0.0)
    with pytest.raises(ValueError):
        RNDriftConfig(x_grid_points=2)
    with pytest.raises(ValueError):
        RNDriftConfig(mc_samples=0)
    with pytest.raises(ValueError):
        RNDriftConfig(mu_cap_per_sec=0.0)
    with pytest.raises(ValueError):
        RNDriftConfig(sprime_clip=0.0)


def test_compile_rejects_negatives() -> None:
    with pytest.raises(ValueError):
        compile_mu_fn(sigma_b=-0.1, lambda_jump=0.0, s_J=0.0)
    with pytest.raises(ValueError):
        compile_mu_fn(sigma_b=0.1, lambda_jump=-0.1, s_J=0.3)
    with pytest.raises(ValueError):
        compile_mu_fn(sigma_b=0.1, lambda_jump=0.01, s_J=-0.3)


# ---------- Pure-diffusion closed-form limit ----------


def test_pure_diffusion_matches_closed_form_at_100_points() -> None:
    """λ=0 ⇒ μ(x) = -½·(1-2p)·σ_b² exactly on the grid midpoints."""
    sigma_b = 0.1
    mu_fn = compile_mu_fn(sigma_b=sigma_b, lambda_jump=0.0, s_J=0.5)
    # Sample 100 probabilities in (ε, 1-ε) to avoid the grid-endpoint clamps
    # (boundaries at ±5 may be truncated by the mu_cap_per_sec guard).
    ps = np.linspace(0.02, 0.98, 100)
    xs = np.log(ps / (1.0 - ps))
    for p, x in zip(ps, xs):
        mu = mu_fn(_t0(), float(x))
        expected = -0.5 * (1.0 - 2.0 * p) * sigma_b * sigma_b
        # Clamped to mu_cap — truth only exceeds cap near p → 0, 1.
        if abs(expected) <= DEFAULT_MU_CAP_PER_SEC:
            assert mu == pytest.approx(expected, abs=1e-3)


def test_jump_free_boundary_at_half_is_zero() -> None:
    mu_fn = compile_mu_fn(sigma_b=0.1, lambda_jump=0.0, s_J=0.3)
    assert mu_fn(_t0(), 0.0) == pytest.approx(0.0, abs=1e-12)


def test_lambda_zero_ignores_s_J() -> None:
    """When λ=0, s_J has no effect on μ."""
    a = compile_mu_fn(sigma_b=0.1, lambda_jump=0.0, s_J=0.1)
    b = compile_mu_fn(sigma_b=0.1, lambda_jump=0.0, s_J=2.0)
    xs = np.linspace(-3.0, 3.0, 50)
    for x in xs:
        assert a(_t0(), float(x)) == b(_t0(), float(x))


# ---------- Martingale check (the big one) ----------


def test_martingale_property_with_jumps() -> None:
    """Simulate a JD path with computed μ(t, x); assert E[p_{t+Δ} | F_t] ≈ p_t.

    Paper's fundamental RN correctness test. MC 10000 paths at Δ=60s with
    dt=1s step. Tolerance 0.005 on |E[p] - p₀| at p ∈ {0.2, 0.5, 0.8}.
    """
    sigma_b = 0.1
    lambda_jump = 0.05
    s_J = 0.3
    horizon_seconds = 60
    dt = 1.0
    n_paths = 10_000

    mu_fn = compile_mu_fn(
        sigma_b=sigma_b,
        lambda_jump=lambda_jump,
        s_J=s_J,
        config=RNDriftConfig(mc_samples=3000),
    )
    rng = np.random.default_rng(2026)

    for p0 in (0.2, 0.5, 0.8):
        x0 = _logit(p0)
        x = np.full(n_paths, x0)
        for _ in range(horizon_seconds):
            # Vectorized step: constant drift at x (μ varies with x, but at
            # this timescale each path sees its own x).
            mus = np.array([mu_fn(_t0(), float(xi)) for xi in x])
            dW = rng.normal(0.0, math.sqrt(dt), size=n_paths)
            jump_hit = rng.random(n_paths) < (lambda_jump * dt)
            jumps = rng.normal(0.0, s_J, size=n_paths) * jump_hit
            x = x + mus * dt + sigma_b * dW + jumps

        p_final = 1.0 / (1.0 + np.exp(-x))
        mean_p = float(p_final.mean())
        bias = abs(mean_p - p0)
        assert bias < 0.005, (
            f"Martingale bias at p₀={p0}: E[p]={mean_p:.5f}, target={p0}, "
            f"bias={bias:.5f}"
        )


# ---------- Numerical guards ----------


def test_mu_is_capped_at_config_limit() -> None:
    """With a large σ_b the raw closed-form |μ| can exceed the cap at
    extreme x — the returned value must be clipped."""
    sigma_b = 5.0  # unphysically large → forces clipping
    mu_fn = compile_mu_fn(sigma_b=sigma_b, lambda_jump=0.0, s_J=0.0)
    xs = np.linspace(-5.0, 5.0, 50)
    mus = np.array([mu_fn(_t0(), float(x)) for x in xs])
    assert np.all(np.abs(mus) <= DEFAULT_MU_CAP_PER_SEC + 1e-12)
    assert float(np.max(np.abs(mus))) == pytest.approx(DEFAULT_MU_CAP_PER_SEC, abs=1e-9)


def test_sprime_clip_prevents_blowup_near_boundary() -> None:
    """At x = ±5 (p ≈ 0.0067 / 0.9933), S'(x) is small; μ must stay finite."""
    mu_fn = compile_mu_fn(sigma_b=0.1, lambda_jump=0.05, s_J=0.3)
    for x in (-5.0, -4.5, 4.5, 5.0):
        mu = mu_fn(_t0(), x)
        assert math.isfinite(mu)
        assert abs(mu) <= DEFAULT_MU_CAP_PER_SEC + 1e-12


def test_determinism_same_seed_same_output() -> None:
    """Byte-identical μ for repeated compilations with the default seed."""
    cfg = RNDriftConfig(mc_samples=1500)
    a = compile_mu_fn(sigma_b=0.08, lambda_jump=0.03, s_J=0.4, config=cfg)
    b = compile_mu_fn(sigma_b=0.08, lambda_jump=0.03, s_J=0.4, config=cfg)
    xs = np.linspace(-4.0, 4.0, 50)
    for x in xs:
        assert a(_t0(), float(x)) == b(_t0(), float(x))


def test_different_seed_differs_slightly() -> None:
    cfg_a = RNDriftConfig(mc_samples=1500, seed=1)
    cfg_b = RNDriftConfig(mc_samples=1500, seed=2)
    a = compile_mu_fn(sigma_b=0.08, lambda_jump=0.03, s_J=0.4, config=cfg_a)
    b = compile_mu_fn(sigma_b=0.08, lambda_jump=0.03, s_J=0.4, config=cfg_b)
    # Seeds differ → MC realizations differ → μ values differ at some points.
    differ = [a(_t0(), float(x)) != b(_t0(), float(x)) for x in np.linspace(-3, 3, 30)]
    assert any(differ)


def test_pathological_inputs_remain_finite() -> None:
    """Feed extreme (s_J=1e-9, λ=1e3, x=±5) — μ stays finite and capped."""
    cfg = RNDriftConfig(mc_samples=500)
    mu_fn = compile_mu_fn(sigma_b=1.0, lambda_jump=1e3, s_J=1e-9, config=cfg)
    for x in (-5.0, -4.0, 0.0, 4.0, 5.0):
        mu = mu_fn(_t0(), x)
        assert math.isfinite(mu)
        assert abs(mu) <= DEFAULT_MU_CAP_PER_SEC + 1e-12


def test_mc_compensator_std_under_5pct_of_magnitude() -> None:
    """At the production MC setting (K=2000 per paper §6.4), a
    compensator-dominant parameter choice has MC std < 5 % of its
    magnitude. (K=600 is tight — the default DEFAULT_MC_SAMPLES=2000
    is what's actually used; verify at that setting.)
    """
    xs = np.linspace(-2.0, 2.0, 15)
    seeds = [11 * i + 7 for i in range(24)]
    values_per_x = np.empty((len(xs), len(seeds)))
    for j, seed in enumerate(seeds):
        mu_fn = compile_mu_fn(
            sigma_b=0.01,
            lambda_jump=0.2,
            s_J=0.5,
            config=RNDriftConfig(mc_samples=DEFAULT_MC_SAMPLES, seed=seed),
        )
        for i, x in enumerate(xs):
            values_per_x[i, j] = mu_fn(_t0(), float(x))

    means = values_per_x.mean(axis=1)
    stds = values_per_x.std(axis=1)
    usable = np.abs(means) > 1e-3
    if not np.any(usable):
        pytest.skip("no grid point with meaningful μ magnitude")
    rel_stds = stds[usable] / np.abs(means[usable])
    assert float(np.median(rel_stds)) < 0.05, (
        f"MC std/median ratio = {float(np.median(rel_stds)):.4f} at K={DEFAULT_MC_SAMPLES}"
    )


# ---------- EM convergence ----------


def test_em_calibrate_converges_on_synthetic_jd() -> None:
    """Run em_calibrate on N=6000 JD path with mis-specified initial params.

    Assert: converged=True within 50 iters, final params within 30 % of
    truth for (σ_b, λ, s_J).
    """
    rng = np.random.default_rng(2026)
    truth = MixtureParams(sigma_b=0.06, s_J=0.5, lambda_jump=0.03, mu=0.0)
    x = _simulate_jd_with_drift(
        rng,
        n=6000,
        sigma_b=truth.sigma_b,
        lambda_jump=truth.lambda_jump,
        s_J=truth.s_J,
    )
    states = _states_from_path(x)

    initial = MixtureParams(sigma_b=0.15, s_J=0.2, lambda_jump=0.1, mu=0.0)  # mis-specified
    result = em_calibrate(
        states,
        initial,
        max_iters=50,
        tol=1e-4,
        jump_mc_samples=1500,
    )

    assert isinstance(result, CalibrationResult)
    assert result.converged, f"did not converge in 50 iters; LL={result.log_likelihood_history}"
    final = result.final_params
    assert abs(final.sigma_b - truth.sigma_b) / truth.sigma_b < 0.30, (
        f"sigma_b={final.sigma_b:.4f} vs truth {truth.sigma_b:.4f}"
    )
    assert abs(final.lambda_jump - truth.lambda_jump) / truth.lambda_jump < 0.30, (
        f"lambda={final.lambda_jump:.4f} vs truth {truth.lambda_jump:.4f}"
    )
    assert abs(final.s_J - truth.s_J) / truth.s_J < 0.30, (
        f"s_J={final.s_J:.4f} vs truth {truth.s_J:.4f}"
    )


def test_log_likelihood_history_non_decreasing() -> None:
    rng = np.random.default_rng(11)
    truth = MixtureParams(sigma_b=0.05, s_J=0.6, lambda_jump=0.02)
    x = _simulate_jd_with_drift(
        rng, n=3000, sigma_b=truth.sigma_b, lambda_jump=truth.lambda_jump, s_J=truth.s_J
    )
    states = _states_from_path(x)
    initial = MixtureParams(sigma_b=0.12, s_J=0.25, lambda_jump=0.08)
    result = em_calibrate(states, initial, max_iters=20, jump_mc_samples=1000)
    ll = result.log_likelihood_history
    # Allow tiny numerical drift.
    for i in range(1, len(ll)):
        assert ll[i] >= ll[i - 1] - 1e-6, (
            f"LL decreased at iter {i}: {ll[i - 1]:.4f} → {ll[i]:.4f}"
        )


def test_em_rejects_bad_hyperparams() -> None:
    with pytest.raises(ValueError):
        em_calibrate([], MixtureParams(sigma_b=0.1, s_J=0.3, lambda_jump=0.01), max_iters=0)
    with pytest.raises(ValueError):
        em_calibrate([], MixtureParams(sigma_b=0.1, s_J=0.3, lambda_jump=0.01), tol=0.0)


def test_em_handles_short_input_gracefully() -> None:
    """With <2 states there is no increment — return a coherent result and
    do not crash."""
    initial = MixtureParams(sigma_b=0.1, s_J=0.3, lambda_jump=0.01)
    result = em_calibrate([], initial, max_iters=5)
    assert result.final_params == initial
    assert result.log_likelihood_history == []


# ---------- mu_fn metadata ----------


def test_mu_fn_exposes_grid_for_diagnostics() -> None:
    mu_fn = compile_mu_fn(sigma_b=0.05, lambda_jump=0.0, s_J=0.3)
    # We document the grid / mu arrays as attributes for diagnostics.
    assert hasattr(mu_fn, "x_grid")
    assert hasattr(mu_fn, "mu_grid")
    assert len(mu_fn.x_grid) == len(mu_fn.mu_grid)  # type: ignore[attr-defined]
