"""Unit tests for ``core/em/jumps`` (paper §5.2 eq 11-12)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from blksch.core.em.increments import (
    DEFAULT_JUMP_THRESHOLD,
    MixtureParams,
    e_step,
    log_likelihood,
)
from blksch.core.em.jumps import (
    BIPOWER_SCALE,
    DEFAULT_S_J_SQ_FLOOR,
    JumpEstimate,
    bipower_variance,
    m_step_jumps,
)
from blksch.schemas import LogitState

pytestmark = pytest.mark.unit


# ---------- Helpers ----------


def _logit_states(
    x: np.ndarray,
    *,
    dt: float = 1.0,
    token_id: str = "t",
    start: datetime | None = None,
) -> list[LogitState]:
    start = start or datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
    return [
        LogitState(token_id=token_id, x_hat=float(x[i]), sigma_eta2=0.01,
                   ts=start + timedelta(seconds=i * dt))
        for i in range(len(x))
    ]


def _simulate_path(
    rng: np.random.Generator,
    *,
    n: int,
    sigma_b: float,
    s_J: float,
    lambda_jump: float,
    mu: float = 0.0,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_path, is_jump) where x[0] = 0 and x[t+1] = x[t] + Δx."""
    is_jump = rng.random(n - 1) < (lambda_jump * dt)
    diffusion = rng.normal(mu * dt, sigma_b * np.sqrt(dt), size=n - 1)
    jumps = rng.normal(0.0, s_J, size=n - 1)
    increments = np.where(is_jump, jumps, diffusion)
    x = np.concatenate([[0.0], np.cumsum(increments)])
    return x, is_jump


# ---------- JumpEstimate dataclass ----------


def test_jump_estimate_rejects_negatives() -> None:
    with pytest.raises(ValueError):
        JumpEstimate(lambda_hat=-1.0, s_J_sq_hat=0.1, jump_timestamps=[], log_likelihood=0.0)
    with pytest.raises(ValueError):
        JumpEstimate(lambda_hat=0.1, s_J_sq_hat=-0.1, jump_timestamps=[], log_likelihood=0.0)


# ---------- M-step on known synthetic path ----------


def test_recovers_lambda_and_s_J_squared() -> None:
    rng = np.random.default_rng(2026)
    true_sigma_b = 0.04
    true_s_J = 0.6
    true_lambda = 0.03
    x, _ = _simulate_path(
        rng, n=6000, sigma_b=true_sigma_b, s_J=true_s_J, lambda_jump=true_lambda
    )
    states = _logit_states(x)

    params = MixtureParams(sigma_b=true_sigma_b, s_J=true_s_J, lambda_jump=true_lambda)
    posteriors = e_step(states, params)
    est = m_step_jumps(states, posteriors, params)

    # Paper §6 target: 6000-sample path recovery.
    assert abs(est.lambda_hat - true_lambda) / true_lambda < 0.20
    rel = abs(est.s_J_sq_hat - true_s_J * true_s_J) / (true_s_J * true_s_J)
    assert rel < 0.30, f"s_J² recovery off: {rel:.3f}"


def test_recovery_is_stable_at_higher_jump_rate() -> None:
    rng = np.random.default_rng(7)
    x, _ = _simulate_path(rng, n=6000, sigma_b=0.03, s_J=0.8, lambda_jump=0.05)
    states = _logit_states(x)
    params = MixtureParams(sigma_b=0.03, s_J=0.8, lambda_jump=0.05)
    est = m_step_jumps(states, e_step(states, params), params)
    assert abs(est.lambda_hat - 0.05) / 0.05 < 0.20
    assert abs(est.s_J_sq_hat - 0.64) / 0.64 < 0.30


# ---------- Jump timestamp extraction ----------


def test_jump_timestamp_extraction_matches_injection() -> None:
    """Inject 10 large jumps at known positions and confirm the M-step
    records the corresponding timestamps."""
    rng = np.random.default_rng(11)
    n = 500
    # Quiet diffusive path.
    increments = rng.normal(0.0, 0.02, size=n - 1)
    # Inject 10 big jumps at fixed positions.
    jump_positions = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]
    for pos in jump_positions:
        increments[pos] = 1.5 * (1 if pos % 2 == 0 else -1)  # far beyond diffusion scale
    x = np.concatenate([[0.0], np.cumsum(increments)])
    states = _logit_states(x, dt=1.0)

    params = MixtureParams(sigma_b=0.02, s_J=1.0, lambda_jump=0.02)
    posteriors = e_step(states, params)
    est = m_step_jumps(states, posteriors, params)

    assert len(est.jump_timestamps) == 10
    expected = [states[p + 1].ts for p in jump_positions]
    assert sorted(est.jump_timestamps) == sorted(expected)


def test_jump_threshold_passthrough() -> None:
    """Lowering the threshold yields more jump timestamps."""
    rng = np.random.default_rng(1)
    x, _ = _simulate_path(rng, n=300, sigma_b=0.05, s_J=0.6, lambda_jump=0.05)
    states = _logit_states(x)
    params = MixtureParams(sigma_b=0.05, s_J=0.6, lambda_jump=0.05)
    post = e_step(states, params)
    strict = m_step_jumps(states, post, params, threshold=0.9)
    lax = m_step_jumps(states, post, params, threshold=0.4)
    assert len(lax.jump_timestamps) >= len(strict.jump_timestamps)


# ---------- Empty / degenerate branches ----------


def test_all_gamma_zero_returns_zero_lambda_and_no_jumps() -> None:
    # Pure diffusion — E-step will produce γ_t all zero because λ=0.
    rng = np.random.default_rng(3)
    x, _ = _simulate_path(rng, n=300, sigma_b=0.05, s_J=0.6, lambda_jump=0.0)
    states = _logit_states(x)
    # Fit with λ=0 so E-step shortcut hits.
    params = MixtureParams(sigma_b=0.05, s_J=0.6, lambda_jump=0.0)
    est = m_step_jumps(states, e_step(states, params), params)
    assert est.lambda_hat == 0.0
    assert est.jump_timestamps == []
    # s_J² stays at prior (floored).
    assert est.s_J_sq_hat == pytest.approx(0.6 * 0.6)


def test_shorter_than_two_states_returns_empty_estimate() -> None:
    params = MixtureParams(sigma_b=0.05, s_J=0.3, lambda_jump=0.02)
    post = e_step([], params)
    est = m_step_jumps([], post, params)
    assert est.lambda_hat == 0.0
    assert est.jump_timestamps == []
    assert est.s_J_sq_hat == pytest.approx(max(0.3 * 0.3, DEFAULT_S_J_SQ_FLOOR))
    assert est.log_likelihood == float("-inf")


def test_s_J_sq_is_floored() -> None:
    """Even with all-zero increments, s_J² never drops below the floor."""
    n = 100
    states = _logit_states(np.zeros(n))
    params = MixtureParams(sigma_b=0.05, s_J=0.0, lambda_jump=0.02)
    est = m_step_jumps(states, e_step(states, params), params)
    assert est.s_J_sq_hat >= DEFAULT_S_J_SQ_FLOOR


def test_lambda_hat_clipped_to_dt_min_inverse() -> None:
    """λ̂ cannot claim more than one jump per timestep — caps at 1/Δt_min."""
    # Build degenerate posterior with γ_t = 1 everywhere.
    rng = np.random.default_rng(9)
    x, _ = _simulate_path(rng, n=200, sigma_b=0.001, s_J=2.0, lambda_jump=10.0)
    states = _logit_states(x, dt=1.0)
    params = MixtureParams(sigma_b=0.001, s_J=2.0, lambda_jump=10.0)
    est = m_step_jumps(states, e_step(states, params), params)
    # dt_min = 1.0 → λ_max = 1.0.
    assert est.lambda_hat <= 1.0 + 1e-9


# ---------- Bi-power variation cross-check ----------


def test_bipower_on_pure_diffusion_matches_sigma_b_sq_T() -> None:
    rng = np.random.default_rng(17)
    x, _ = _simulate_path(rng, n=10000, sigma_b=0.05, s_J=0.6, lambda_jump=0.0)
    increments = np.diff(x)
    bv = bipower_variance(increments)
    target = 0.05 * 0.05 * (len(increments))  # σ_b²·T with dt=1
    assert abs(bv - target) / target < 0.05, f"BV={bv:.4f}, target={target:.4f}"


def test_bipower_much_closer_to_truth_than_realized_variance() -> None:
    """Robustness claim: in a jump-dominated regime, BV is far closer to
    σ_b²·T than naive realized variance RV = Σ Δx².

    Theoretical contamination of BV scales with λ (finite-sample, fixed
    dt); RV absorbs every jump at full weight. The practical test is the
    ratio, not an absolute tolerance.
    """
    rng = np.random.default_rng(23)
    x, _ = _simulate_path(rng, n=10000, sigma_b=0.05, s_J=0.7, lambda_jump=0.01)
    increments = np.diff(x)
    target = 0.05 * 0.05 * (len(increments))
    bv = bipower_variance(increments)
    rv = float(np.sum(increments * increments))
    # RV is massively inflated by jumps; BV is only mildly inflated.
    assert abs(bv - target) < 0.5 * abs(rv - target), (
        f"BV={bv:.4f}, RV={rv:.4f}, target={target:.4f}"
    )


def test_bipower_on_low_jump_rate_is_within_20pct() -> None:
    """At a rare-jump rate (λΔ = 0.003), BV lands within 20 % of truth —
    the asymptotic regime where the O(λ) contamination is small."""
    rng = np.random.default_rng(23)
    x, _ = _simulate_path(rng, n=10000, sigma_b=0.05, s_J=0.7, lambda_jump=0.003)
    increments = np.diff(x)
    bv = bipower_variance(increments)
    target = 0.05 * 0.05 * (len(increments))
    assert abs(bv - target) / target < 0.20, f"BV={bv:.4f}, target={target:.4f}"


def test_bipower_empty_or_single_increment() -> None:
    assert bipower_variance([]) == 0.0
    assert bipower_variance([1.0]) == 0.0


def test_bipower_rejects_nd_arrays() -> None:
    with pytest.raises(ValueError):
        bipower_variance(np.zeros((5, 2)))


def test_bipower_scale_equals_pi_half() -> None:
    assert BIPOWER_SCALE == pytest.approx(np.pi / 2.0)


# ---------- EM iteration monotonicity ----------


def test_em_iteration_monotonicity() -> None:
    """E-step → M-step → E-step with updated params should not decrease the
    mixture log-likelihood. Tolerance 1e-6 for numerical noise."""
    rng = np.random.default_rng(99)
    x, _ = _simulate_path(rng, n=4000, sigma_b=0.05, s_J=0.6, lambda_jump=0.03)
    states = _logit_states(x)
    # Deliberately mis-specified prior so the first update improves LL.
    prior = MixtureParams(sigma_b=0.05, s_J=0.2, lambda_jump=0.1)
    post0 = e_step(states, prior)
    ll0 = log_likelihood(post0.increments, post0.dts, prior)
    est = m_step_jumps(states, post0, prior)
    refined = MixtureParams(
        sigma_b=prior.sigma_b,
        s_J=float(np.sqrt(est.s_J_sq_hat)),
        lambda_jump=est.lambda_hat,
        mu=prior.mu,
    )
    post1 = e_step(states, refined)
    ll1 = log_likelihood(post1.increments, post1.dts, refined)
    # Allow tiny negative drift from float round-off.
    assert ll1 >= ll0 - 1e-6, f"LL decreased: {ll0:.4f} -> {ll1:.4f}"
    # Paper §5.2: the new LL as computed inside JumpEstimate should equal
    # log_likelihood on the updated params.
    assert est.log_likelihood == pytest.approx(
        log_likelihood(post0.increments, post0.dts, refined), rel=1e-9, abs=1e-9
    )


# ---------- Input-shape validation ----------


def test_m_step_rejects_mismatched_state_and_posterior_lengths() -> None:
    x = np.zeros(5)
    states = _logit_states(x)
    params = MixtureParams(sigma_b=0.05, s_J=0.3, lambda_jump=0.02)
    post = e_step(states, params)
    # Drop one state so len(states) != post.n + 1.
    with pytest.raises(ValueError):
        m_step_jumps(states[:-1], post, params)
