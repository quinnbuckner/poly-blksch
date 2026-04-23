"""Unit tests for ``core/filter/ewma_var.EwmaVar``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from blksch.core.filter.ewma_var import (
    DEFAULT_COLD_START_FACTOR,
    DEFAULT_HALF_LIFE_SEC,
    LN2,
    EwmaVar,
)

pytestmark = pytest.mark.unit


# ---------- Constructor validation ----------


def test_rejects_non_positive_half_life() -> None:
    with pytest.raises(ValueError):
        EwmaVar(half_life_sec=0.0)
    with pytest.raises(ValueError):
        EwmaVar(half_life_sec=-1.0)


def test_rejects_negative_cold_start_factor() -> None:
    with pytest.raises(ValueError):
        EwmaVar(half_life_sec=60.0, cold_start_factor=-0.1)


def test_default_constants_exposed() -> None:
    assert DEFAULT_HALF_LIFE_SEC > 0
    assert DEFAULT_COLD_START_FACTOR >= 0
    assert LN2 == pytest.approx(math.log(2.0))


# ---------- Cold-start behavior ----------


def test_initial_variance_is_zero() -> None:
    est = EwmaVar(half_life_sec=60.0)
    assert est.variance() == 0.0
    assert est.in_cold_start is True
    assert est.samples_seen == 0


def test_cold_start_uses_unweighted_mean() -> None:
    """Inside the cold-start span, variance is the plain weighted mean of
    the first samples — this is what keeps short runs unbiased."""
    est = EwmaVar(half_life_sec=60.0, cold_start_factor=1.0)
    samples = [0.05, 0.04, 0.06, 0.055, 0.045]  # dx, dt=1 each
    for dx in samples:
        est.update(dx, dt=1.0, jump_posterior=0.0)
    expected = np.mean([s * s for s in samples])
    assert est.variance() == pytest.approx(expected, rel=1e-9)
    assert est.in_cold_start is True


def test_cold_start_transitions_to_ewma_after_span() -> None:
    est = EwmaVar(half_life_sec=30.0, cold_start_factor=1.0)
    for _ in range(35):  # span of 30 s ≤ 35 s elapsed
        est.update(0.05, dt=1.0, jump_posterior=0.0)
    assert est.in_cold_start is False
    # After cold-start transition, variance equals the cold-start
    # accumulator value at that instant.
    assert est.variance() == pytest.approx(0.0025, rel=1e-9)


def test_cold_start_factor_zero_disables_warmup() -> None:
    """With ``cold_start_factor=0`` the very first call ends cold-start."""
    est = EwmaVar(half_life_sec=60.0, cold_start_factor=0.0)
    est.update(0.05, dt=1.0, jump_posterior=0.0)
    assert est.in_cold_start is False


# ---------- Convergence to truth on a constant-σ_b synthetic ----------


def test_constant_sigma_b_path_converges_to_truth() -> None:
    """Average over 20 seeds of a long pure-diffusion run — the EWMA
    should converge to σ_b² = 0.0025 within 5 % in expectation (the
    single-run sampling noise at 2·H can be ~20 %; average kills it)."""
    sigma_b = 0.05
    dt = 1.0
    half_life = 60.0
    samples_per_run = int(6 * half_life / dt)  # ~6 half-lives
    estimates = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        est = EwmaVar(half_life_sec=half_life, cold_start_factor=1.0)
        for _ in range(samples_per_run):
            est.update(rng.normal(0.0, sigma_b * math.sqrt(dt)), dt=dt)
        estimates.append(est.variance())
    target = sigma_b * sigma_b
    assert np.mean(estimates) == pytest.approx(target, rel=0.05)


def test_post_warmup_decay_toward_new_regime() -> None:
    """When σ_b doubles, EWMA should move at least halfway toward the new
    truth within one half-life's worth of samples."""
    rng = np.random.default_rng(7)
    dt = 1.0
    half_life = 30.0
    est = EwmaVar(half_life_sec=half_life, cold_start_factor=1.0)
    for _ in range(200):
        est.update(rng.normal(0.0, 0.05 * math.sqrt(dt)), dt=dt)
    baseline = est.variance()
    # Regime shift: σ_b doubles → new var = 4 × baseline.
    for _ in range(int(half_life / dt)):
        est.update(rng.normal(0.0, 0.10 * math.sqrt(dt)), dt=dt)
    halfway_target = 0.5 * (baseline + 4.0 * baseline)
    assert est.variance() > halfway_target * 0.8  # within ~20% of halfway point


# ---------- Jump exclusion ----------


def test_single_jump_does_not_inflate_estimate() -> None:
    """One extreme Δx with γ_t=1 must not move the estimate; one with
    γ_t=0 would, so the gap measures jump-exclusion fidelity."""
    est_a = EwmaVar(half_life_sec=60.0, cold_start_factor=1.0)
    est_b = EwmaVar(half_life_sec=60.0, cold_start_factor=1.0)
    # Warm up both identically on pure diffusion.
    rng = np.random.default_rng(11)
    warm = [rng.normal(0.0, 0.05) for _ in range(200)]
    for dx in warm:
        est_a.update(dx, dt=1.0, jump_posterior=0.0)
        est_b.update(dx, dt=1.0, jump_posterior=0.0)
    # Now inject a huge increment. A says it's all jump, B says it isn't.
    big = 2.0  # 40× a normal σ_b step
    est_a.update(big, dt=1.0, jump_posterior=1.0)
    est_b.update(big, dt=1.0, jump_posterior=0.0)
    assert est_a.variance() < est_b.variance()
    # A only decays slightly (one step's worth).
    retention = math.exp(-LN2 * 1.0 / 60.0)
    expected_a = retention * 0.0025  # base converged to ≈ 0.0025
    assert est_a.variance() == pytest.approx(expected_a, rel=0.2)


def test_jump_heavy_window_does_not_inflate() -> None:
    """With all samples declared jumps (γ_t=1), the estimate decays
    toward 0 — it does not absorb jump variance."""
    est = EwmaVar(half_life_sec=30.0, cold_start_factor=1.0)
    rng = np.random.default_rng(13)
    for _ in range(200):
        est.update(rng.normal(0.0, 0.05), dt=1.0, jump_posterior=0.0)
    pre = est.variance()
    for _ in range(60):
        est.update(rng.normal(0.0, 0.5), dt=1.0, jump_posterior=1.0)
    post = est.variance()
    # After 2 half-lives of γ=1 samples, retention^60 ≈ 0.25; variance
    # should have shrunk toward zero, NOT expanded to accommodate jumps.
    assert post < pre
    assert post < 0.5 * pre


def test_partial_jump_posterior_dampens_update() -> None:
    """γ_t ∈ (0, 1) should give a proportionally smaller update than γ_t=0."""
    rng = np.random.default_rng(5)
    warm = [rng.normal(0.0, 0.05) for _ in range(200)]

    est0 = EwmaVar(half_life_sec=60.0)
    est_half = EwmaVar(half_life_sec=60.0)
    for dx in warm:
        est0.update(dx, dt=1.0, jump_posterior=0.0)
        est_half.update(dx, dt=1.0, jump_posterior=0.0)
    big = 0.5
    est0.update(big, dt=1.0, jump_posterior=0.0)
    est_half.update(big, dt=1.0, jump_posterior=0.5)
    # Half-excluded update must lift the estimate less than full-included.
    assert est_half.variance() < est0.variance()


# ---------- Variable dt ----------


def test_half_life_normalization_across_variable_dt() -> None:
    """Two updates of dt=0.5 should compound to approximately one update
    of dt=1.0 at the same total signal, because λ_per_step depends on dt."""
    rng = np.random.default_rng(21)
    warm = rng.normal(0.0, 0.05, size=200).tolist()

    e_fine = EwmaVar(half_life_sec=60.0)
    e_coarse = EwmaVar(half_life_sec=60.0)
    for dx in warm:
        # Fine: two half-steps of (dx/√2). Coarse: one full step of dx.
        e_fine.update(dx / math.sqrt(2.0), dt=0.5)
        e_fine.update(dx / math.sqrt(2.0), dt=0.5)
        e_coarse.update(dx, dt=1.0)
    # Both should converge to ≈ 0.0025. Tighten with large sample.
    assert abs(e_fine.variance() - e_coarse.variance()) < 0.1 * e_coarse.variance()


def test_zero_dt_is_noop() -> None:
    est = EwmaVar(half_life_sec=60.0)
    # Warm up.
    for _ in range(100):
        est.update(0.05, dt=1.0)
    before = est.variance()
    samples_before = est.samples_seen
    est.update(10.0, dt=0.0)  # pathological tick
    assert est.variance() == before
    assert est.samples_seen == samples_before


# ---------- Determinism ----------


def test_reproducible_under_seed() -> None:
    def run(seed: int) -> float:
        rng = np.random.default_rng(seed)
        est = EwmaVar(half_life_sec=45.0)
        for _ in range(500):
            dx = rng.normal(0.0, 0.05)
            gamma = 1.0 if rng.random() < 0.03 else 0.0
            est.update(dx, dt=1.0, jump_posterior=gamma)
        return est.variance()

    assert run(42) == run(42)
    assert run(42) != run(43)


def test_reset_clears_all_state() -> None:
    est = EwmaVar(half_life_sec=30.0)
    for _ in range(200):
        est.update(0.1, dt=1.0)
    assert est.variance() > 0
    est.reset()
    assert est.variance() == 0.0
    assert est.in_cold_start is True
    assert est.samples_seen == 0
    assert est.elapsed_sec == 0.0
