"""Tests for ``core/surface/corr.py`` — paper §5.4 de-jumped correlation.

Synthetic data only: we generate correlated bivariate logit-state paths
inside the test file so we don't have to wait on Window A's EM to land.

Four guarantees pinned here:

1. **Recovery.** On a clean 10k-sample path with known ρ and zero jumps,
   the estimator recovers ρ within 0.05.
2. **Jump-contamination immunity.** Inject 20 jumps into one series at
   random times. The masked estimator should track ρ; the naive (unmasked)
   benchmark should be noticeably biased. We assert the masked version has
   < 50% of the naive bias.
3. **Co-jump detection.** Inject exactly 5 co-jumps. ``λ_ij`` should match
   5/T within Poisson noise (deterministic injection → exact match).
4. **Near-empty handling.** Feed a series where jumps cover >95% of the
   increments. The documented ``NotEnoughSamplesError`` must raise.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from blksch.core.surface.corr import (
    DEFAULT_MASK_WINDOW,
    NotEnoughSamplesError,
    estimate_correlation,
)
from blksch.schemas import LogitState

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Synthetic-path helper (lives in the test file per the plan contract)
# ---------------------------------------------------------------------------


def _make_correlated_logit_states(
    rho: float,
    n: int,
    *,
    jumps_at: list[int] | None = None,
    jump_size_x: float = 1.5,
    jumps_at_j: list[int] | None = None,
    jump_size_y: float = 1.5,
    seed: int = 0,
    dt_sec: float = 1.0,
    sigma: float = 0.02,
    token_i: str = "tok-i",
    token_j: str = "tok-j",
) -> tuple[list[LogitState], list[LogitState], list[datetime], list[datetime]]:
    """Generate two logit-state series with a shared Gaussian correlation.

    ``jumps_at`` — indices (on the common grid) where series i takes an
    additive jump of magnitude ``jump_size_x``. The indices converted to
    absolute timestamps go into the returned ``jumps_i`` list.

    ``jumps_at_j`` — same, for series j.

    Timestamps are aligned between the two streams (Track A's upstream
    alignment contract).
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]]) * (sigma ** 2)
    incs = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n - 1)
    x = np.zeros(n)
    y = np.zeros(n)
    x[1:] = np.cumsum(incs[:, 0])
    y[1:] = np.cumsum(incs[:, 1])

    jumps_i: list[datetime] = []
    jumps_j: list[datetime] = []
    for idx in (jumps_at or []):
        sign = 1.0 if rng.random() > 0.5 else -1.0
        x[idx:] += sign * jump_size_x
        jumps_i.append(T0 + timedelta(seconds=idx * dt_sec))
    for idx in (jumps_at_j or []):
        sign = 1.0 if rng.random() > 0.5 else -1.0
        y[idx:] += sign * jump_size_y
        jumps_j.append(T0 + timedelta(seconds=idx * dt_sec))

    states_i = [
        LogitState(
            token_id=token_i, x_hat=float(x[k]), sigma_eta2=sigma ** 2,
            ts=T0 + timedelta(seconds=k * dt_sec),
        )
        for k in range(n)
    ]
    states_j = [
        LogitState(
            token_id=token_j, x_hat=float(y[k]), sigma_eta2=sigma ** 2,
            ts=T0 + timedelta(seconds=k * dt_sec),
        )
        for k in range(n)
    ]
    return states_i, states_j, jumps_i, jumps_j


def _naive_pearson(states_i, states_j) -> float:
    """Benchmark: Pearson on the raw (unmasked) increments. The masked
    version must be measurably closer to the true ρ than this."""
    x = np.array([s.x_hat for s in states_i])
    y = np.array([s.x_hat for s in states_j])
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.corrcoef(dx, dy)[0, 1])


# ---------------------------------------------------------------------------
# (1) Recovery on a clean path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho_true", [-0.6, -0.3, 0.0, 0.3, 0.7])
def test_recovers_known_rho_within_0_05(rho_true: float):
    states_i, states_j, _, _ = _make_correlated_logit_states(
        rho=rho_true, n=10_000, seed=42,
    )
    entry = estimate_correlation(states_i, states_j, jumps_i=[], jumps_j=[])
    assert abs(entry.rho - rho_true) < 0.05, (
        f"expected ρ≈{rho_true:.2f}, got {entry.rho:.4f}"
    )
    assert entry.token_id_i == "tok-i"
    assert entry.token_id_j == "tok-j"
    # No jumps -> lambda and m2 should be zero.
    assert entry.co_jump_lambda == 0.0
    assert entry.co_jump_m2 == 0.0


def test_rho_clamped_to_unit_interval():
    """Near-perfect correlation should clamp to [-1, 1], not float-overshoot."""
    states_i, states_j, _, _ = _make_correlated_logit_states(
        rho=0.999, n=2000, seed=7,
    )
    entry = estimate_correlation(states_i, states_j, jumps_i=[], jumps_j=[])
    assert -1.0 <= entry.rho <= 1.0


# ---------------------------------------------------------------------------
# (2) Jump-contamination immunity
# ---------------------------------------------------------------------------


def test_masking_reduces_bias_vs_naive():
    """20 jumps contaminate one series. The *unmasked* Pearson is biased
    relative to the true ρ; the masked estimator tracks ρ tightly. Assert
    masked bias < 50% of naive bias.
    """
    rho_true = 0.5
    n = 10_000
    rng = random.Random(31337)
    jump_idxs = sorted(rng.sample(range(100, n - 100), 20))

    states_i, states_j, jumps_i, jumps_j = _make_correlated_logit_states(
        rho=rho_true, n=n, jumps_at=jump_idxs, jump_size_x=3.0, seed=1,
    )

    entry = estimate_correlation(states_i, states_j, jumps_i, jumps_j)
    naive_rho = _naive_pearson(states_i, states_j)

    masked_bias = abs(entry.rho - rho_true)
    naive_bias = abs(naive_rho - rho_true)

    assert masked_bias < 0.05, f"masked estimator off by {masked_bias:.3f}"
    assert naive_bias > masked_bias, (
        f"naive Pearson should be more biased than masked "
        f"(naive={naive_rho:.3f}, masked={entry.rho:.3f}, true={rho_true})"
    )
    assert masked_bias < 0.5 * naive_bias, (
        f"masked bias {masked_bias:.3f} should be <50% of naive bias "
        f"{naive_bias:.3f}"
    )


def test_masking_geometry_single_jump_midpoint():
    """A single fake-jump at the middle of a clean series should drop ~60
    increments (±30s at 1Hz) without disturbing the correlation estimate."""
    n = 1000
    states_i, states_j, _, _ = _make_correlated_logit_states(
        rho=0.2, n=n, seed=0, dt_sec=1.0,
    )
    jumps_i = [T0 + timedelta(seconds=500)]
    entry = estimate_correlation(
        states_i, states_j, jumps_i, jumps_j=[],
        mask_window=timedelta(seconds=30),
    )
    assert abs(entry.rho - 0.2) < 0.1


# ---------------------------------------------------------------------------
# (3) Co-jump detection
# ---------------------------------------------------------------------------


def test_co_jump_lambda_matches_injected_rate():
    """5 co-jumps over T seconds should yield λ ≈ 5/T. Deterministic
    injection → exact equality."""
    n = 2_000
    dt = 1.0
    co_jump_idxs = [300, 700, 1100, 1500, 1800]

    states_i, states_j, jumps_i, jumps_j = _make_correlated_logit_states(
        rho=0.3, n=n, dt_sec=dt,
        jumps_at=co_jump_idxs, jump_size_x=2.0,
        jumps_at_j=co_jump_idxs, jump_size_y=2.0,
        seed=11,
    )
    entry = estimate_correlation(states_i, states_j, jumps_i, jumps_j)
    T = (n - 1) * dt
    expected_lambda = 5.0 / T
    assert entry.co_jump_lambda == pytest.approx(expected_lambda, rel=1e-6)


def test_co_jump_m2_is_non_negative():
    """m2 is the absolute mean of dx_i * dx_j at co-jump timestamps; by
    construction ≥ 0 — the CorrelationEntry schema enforces it."""
    n = 2_000
    co_idxs = [500, 1000, 1500]

    states_i, states_j, jumps_i, jumps_j = _make_correlated_logit_states(
        rho=0.5, n=n, dt_sec=1.0,
        jumps_at=co_idxs, jump_size_x=4.0,
        jumps_at_j=co_idxs, jump_size_y=4.0,
        seed=999,
    )
    entry = estimate_correlation(states_i, states_j, jumps_i, jumps_j)
    assert entry.co_jump_m2 >= 0.0


def test_no_co_jumps_when_offset_far_exceeds_window():
    n = 2_000
    states_i, states_j, jumps_i, jumps_j = _make_correlated_logit_states(
        rho=0.0, n=n, dt_sec=1.0,
        jumps_at=[500], jumps_at_j=[1500], seed=2,
    )
    entry = estimate_correlation(states_i, states_j, jumps_i, jumps_j)
    assert entry.co_jump_lambda == 0.0
    assert entry.co_jump_m2 == 0.0


def test_each_j_jump_matched_at_most_once():
    """Two jumps in i both near the same jump in j should yield a count
    of 1, not 2 — one-to-one greedy matching."""
    n = 2_000
    states_i, states_j, _, _ = _make_correlated_logit_states(
        rho=0.0, n=n, dt_sec=1.0, seed=3,
    )
    jumps_i = [T0 + timedelta(seconds=1000), T0 + timedelta(seconds=1005)]
    jumps_j = [T0 + timedelta(seconds=1002)]
    entry = estimate_correlation(states_i, states_j, jumps_i, jumps_j)
    T = (n - 1) * 1.0
    assert entry.co_jump_lambda == pytest.approx(1.0 / T, rel=1e-6)


# ---------------------------------------------------------------------------
# (4) Near-empty handling
# ---------------------------------------------------------------------------


def test_near_empty_masked_series_raises():
    """Jumps cover nearly the whole series → masking leaves < min_samples
    increments → raise the documented exception."""
    n = 400  # short enough that 60s-wide masks wipe most of it
    dt = 1.0
    # Jumps every 50 seconds → each masks 60s around it → total coverage.
    jump_idxs = list(range(25, n, 50))
    states_i, states_j, jumps_i, _ = _make_correlated_logit_states(
        rho=0.5, n=n, dt_sec=dt, jumps_at=jump_idxs, seed=4,
    )
    with pytest.raises(NotEnoughSamplesError, match="survived"):
        estimate_correlation(
            states_i, states_j, jumps_i, jumps_j=[],
            min_samples=100,
        )


def test_misaligned_timestamps_rejected():
    states_i, states_j, _, _ = _make_correlated_logit_states(
        rho=0.0, n=500, seed=5,
    )
    states_j_shifted = [
        s.model_copy(update={"ts": s.ts + timedelta(milliseconds=500)})
        for s in states_j
    ]
    with pytest.raises(ValueError, match="aligned"):
        estimate_correlation(states_i, states_j_shifted, jumps_i=[], jumps_j=[])


def test_length_mismatch_rejected():
    states_i, states_j, _, _ = _make_correlated_logit_states(
        rho=0.0, n=500, seed=5,
    )
    with pytest.raises(ValueError, match="aligned"):
        estimate_correlation(states_i, states_j[:-1], jumps_i=[], jumps_j=[])


def test_too_few_states_raises():
    states = [
        LogitState(token_id="t", x_hat=0.0, sigma_eta2=0.01, ts=T0),
    ]
    with pytest.raises(NotEnoughSamplesError):
        estimate_correlation(states, states, [], [])


# ---------------------------------------------------------------------------
# Degenerate-variance guard
# ---------------------------------------------------------------------------


def test_constant_series_returns_zero_rho():
    """If one series has no variance, Pearson is undefined — return 0
    rather than NaN so downstream surfaces don't inherit a poisoned value."""
    n = 500
    states_i = [
        LogitState(token_id="flat", x_hat=0.5, sigma_eta2=0.01,
                   ts=T0 + timedelta(seconds=k))
        for k in range(n)
    ]
    _, states_j_src, _, _ = _make_correlated_logit_states(rho=0.0, n=n, seed=8)
    # Share timestamps with states_i exactly.
    states_j = [
        s.model_copy(update={"ts": states_i[k].ts})
        for k, s in enumerate(states_j_src)
    ]
    entry = estimate_correlation(states_i, states_j, [], [])
    assert entry.rho == 0.0
