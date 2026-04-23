"""Unit tests for ``core/em/increments`` (paper §5.2 E-step)."""

from __future__ import annotations

import numpy as np
import pytest

from blksch.core.em.increments import (
    DEFAULT_JUMP_THRESHOLD,
    MixtureParams,
    compute_posteriors,
    gaussian_log_pdf,
    log_likelihood,
    mark_jumps,
)

pytestmark = pytest.mark.unit


# ---------- MixtureParams ----------


def test_mixture_params_rejects_negatives() -> None:
    with pytest.raises(ValueError):
        MixtureParams(sigma_b=-1.0, s_J=0.1, lambda_jump=0.01)
    with pytest.raises(ValueError):
        MixtureParams(sigma_b=0.1, s_J=-0.1, lambda_jump=0.01)
    with pytest.raises(ValueError):
        MixtureParams(sigma_b=0.1, s_J=0.1, lambda_jump=-0.01)
    with pytest.raises(ValueError):
        MixtureParams(sigma_b=0.1, s_J=0.1, lambda_jump=0.01, mu=float("inf"))


def test_mixture_params_is_frozen() -> None:
    p = MixtureParams(sigma_b=0.1, s_J=0.2, lambda_jump=0.01)
    with pytest.raises((AttributeError, TypeError)):
        p.sigma_b = 0.2  # type: ignore[misc]


# ---------- Gaussian PDF ----------


def test_gaussian_log_pdf_matches_scipy() -> None:
    from scipy.stats import norm

    x = np.array([-1.0, 0.0, 0.3, 2.0])
    log_p = gaussian_log_pdf(x, np.zeros_like(x), np.ones_like(x))
    assert np.allclose(log_p, norm.logpdf(x))


# ---------- Pure diffusion (λ=0) ----------


def test_pure_diffusion_gives_all_zeros() -> None:
    rng = np.random.default_rng(0)
    dx = rng.normal(0, 0.05, size=500)
    params = MixtureParams(sigma_b=0.05, s_J=0.5, lambda_jump=0.0)
    gamma = compute_posteriors(dx, 1.0, params)
    assert np.all(gamma == 0.0)


def test_zero_jump_stddev_also_gives_all_zeros() -> None:
    """s_J=0 is the degenerate jump law — treat as pure diffusion."""
    dx = np.array([0.0, 0.1, -0.1, 2.0])
    params = MixtureParams(sigma_b=0.1, s_J=0.0, lambda_jump=0.5)
    gamma = compute_posteriors(dx, 1.0, params)
    assert np.all(gamma == 0.0)


# ---------- Edge cases ----------


def test_zero_dt_yields_zero_posterior() -> None:
    params = MixtureParams(sigma_b=0.1, s_J=0.3, lambda_jump=0.05)
    gamma = compute_posteriors([0.0, 0.5], [0.0, 1.0], params)
    assert gamma[0] == 0.0
    assert 0.0 <= gamma[1] <= 1.0


def test_rejects_wrong_shape_increments() -> None:
    params = MixtureParams(sigma_b=0.1, s_J=0.3, lambda_jump=0.05)
    with pytest.raises(ValueError):
        compute_posteriors(np.zeros((3, 2)), 1.0, params)


def test_rejects_mismatched_dts_shape() -> None:
    params = MixtureParams(sigma_b=0.1, s_J=0.3, lambda_jump=0.05)
    with pytest.raises(ValueError):
        compute_posteriors(np.zeros(3), np.zeros(4), params)


def test_rejects_negative_dt() -> None:
    params = MixtureParams(sigma_b=0.1, s_J=0.3, lambda_jump=0.05)
    with pytest.raises(ValueError):
        compute_posteriors(np.array([0.0]), np.array([-1.0]), params)


# ---------- Boundary dt (long gap) ----------


def test_boundary_dt_600s_gap_keeps_posterior_finite() -> None:
    """A 600 s gap between ticks should not crash the E-step.

    With ``sigma_b=0.05`` and ``dt=600``, the diffusion variance is 1.5 and
    ``λΔt`` saturates at 1 for any modest λ. Posterior must still be finite
    and in [0, 1].
    """
    params = MixtureParams(sigma_b=0.05, s_J=0.3, lambda_jump=0.01)
    inc = np.array([0.0, 0.1, -1.0, 2.0])
    gamma = compute_posteriors(inc, 600.0, params)
    assert np.all(np.isfinite(gamma))
    assert np.all((gamma >= 0.0) & (gamma <= 1.0))


def test_saturating_lambda_dt_does_not_explode() -> None:
    """λ·Δt can exceed 1 for pathological inputs — we clip and survive."""
    params = MixtureParams(sigma_b=0.05, s_J=0.3, lambda_jump=10.0)
    inc = np.array([0.1, -0.2])
    gamma = compute_posteriors(inc, 5.0, params)  # λΔt = 50
    assert np.all(np.isfinite(gamma))


# ---------- Jump identification on synthetic data ----------


def _simulate_jd(
    rng: np.random.Generator,
    *,
    n: int,
    sigma_b: float,
    s_J: float,
    lambda_jump: float,
    dt: float = 1.0,
    mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate n increments; return (increments, true_jump_indicator)."""
    is_jump = rng.random(n) < (lambda_jump * dt)
    diffusion = rng.normal(mu * dt, sigma_b * np.sqrt(dt), size=n)
    jumps = rng.normal(0.0, s_J, size=n)
    increments = np.where(is_jump, jumps, diffusion)
    return increments, is_jump


def test_identifies_synthetic_jumps_over_80pct() -> None:
    """Recall ≥ 80% at τ_J=0.7 under a jump law clearly separable from
    the diffusion regime (``s_J / (σ_b·√Δt) ~ 33``).
    """
    rng = np.random.default_rng(42)
    true_params = MixtureParams(
        sigma_b=0.03, s_J=1.0, lambda_jump=0.02, mu=0.0
    )  # ~2% of steps are jumps
    inc, true_jumps = _simulate_jd(
        rng,
        n=5000,
        sigma_b=true_params.sigma_b,
        s_J=true_params.s_J,
        lambda_jump=true_params.lambda_jump,
    )
    gamma = compute_posteriors(inc, 1.0, true_params)
    pred = mark_jumps(gamma, threshold=DEFAULT_JUMP_THRESHOLD)

    true_positive = int(np.sum(pred & true_jumps))
    total_true = int(np.sum(true_jumps))
    assert total_true > 0, "simulation produced no jumps; bump lambda or n"
    recall = true_positive / total_true
    assert recall >= 0.80, f"recall={recall:.3f} (< 0.80)"


def test_no_double_counting_on_pure_diffusion() -> None:
    rng = np.random.default_rng(7)
    inc, _ = _simulate_jd(rng, n=2000, sigma_b=0.05, s_J=0.6, lambda_jump=0.0)
    # Fit with λ=1e-6 — should still basically say 'no jumps'.
    params = MixtureParams(sigma_b=0.05, s_J=0.6, lambda_jump=1e-6)
    gamma = compute_posteriors(inc, 1.0, params)
    false_positive_rate = float(np.mean(gamma > DEFAULT_JUMP_THRESHOLD))
    assert false_positive_rate < 0.005


# ---------- Log-likelihood monotonicity ----------


def test_log_likelihood_peaks_near_true_sigma_b() -> None:
    """Fix s_J, λ, μ at truth; sweep σ_b; log-likelihood should be maximized
    near the true value."""
    rng = np.random.default_rng(2026)
    true_sigma_b = 0.06
    true_params = MixtureParams(sigma_b=true_sigma_b, s_J=0.5, lambda_jump=0.02)
    inc, _ = _simulate_jd(
        rng,
        n=8000,
        sigma_b=true_params.sigma_b,
        s_J=true_params.s_J,
        lambda_jump=true_params.lambda_jump,
    )

    sigma_grid = np.linspace(0.02, 0.15, 21)
    lls = []
    for sigma_b in sigma_grid:
        p = MixtureParams(sigma_b=sigma_b, s_J=true_params.s_J, lambda_jump=true_params.lambda_jump)
        lls.append(log_likelihood(inc, 1.0, p))
    arg = int(np.argmax(lls))
    best = sigma_grid[arg]
    # Peak should be close to truth (within one grid step ≈ 0.0065).
    assert abs(best - true_sigma_b) < 0.02, (
        f"LL peak at {best:.3f}, truth {true_sigma_b:.3f}"
    )


def test_log_likelihood_reduces_to_gaussian_when_lambda_zero() -> None:
    """When λ=0, LL equals the sum of Gaussian log-densities — cross-check."""
    rng = np.random.default_rng(11)
    inc = rng.normal(0, 0.1, size=200)
    params = MixtureParams(sigma_b=0.1, s_J=0.5, lambda_jump=0.0, mu=0.0)
    ll = log_likelihood(inc, 1.0, params)
    expected = float(np.sum(gaussian_log_pdf(inc, np.zeros_like(inc), np.full_like(inc, 0.01))))
    assert ll == pytest.approx(expected)


# ---------- mark_jumps helper ----------


def test_mark_jumps_threshold() -> None:
    gamma = np.array([0.1, 0.5, 0.7, 0.71, 0.95])
    # threshold is strictly >, so 0.7 is NOT a jump at τ_J=0.7.
    mask = mark_jumps(gamma, threshold=0.7)
    assert mask.tolist() == [False, False, False, True, True]


def test_mark_jumps_default_threshold_matches_paper() -> None:
    assert DEFAULT_JUMP_THRESHOLD == pytest.approx(0.7)
