"""Property-based fuzz for ``core/surface/corr.py`` — **self-audit**.

This is Track C's own module; the test exercises the same code the unit
tests in ``test_corr.py`` cover but with Hypothesis randomization to catch
edge cases the example-based tests miss.

Invariants under test:

* Output ρ ∈ [-1, 1] always (clamp holds).
* ``NotEnoughSamplesError`` raises iff survivor count < min_samples (exact
  boundary).
* Adding uncorrelated white noise to both paths doesn't flip the sign of
  the ρ estimate (i.e. noise doesn't bias the sign).
* Co-jump count λ_ij ≥ 0 always.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from blksch.core.surface.corr import (
    NotEnoughSamplesError,
    estimate_correlation,
)
from blksch.schemas import LogitState

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)

FUZZ = settings(max_examples=200, deadline=None, derandomize=True)


# ---------------------------------------------------------------------------
# Synthetic path helper (co-located to keep the suite self-contained).
# ---------------------------------------------------------------------------


def _make_path(
    rho: float, n: int, *, seed: int = 0, sigma: float = 0.02,
    extra_noise_sigma: float = 0.0,
) -> tuple[list[LogitState], list[LogitState]]:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]]) * (sigma ** 2)
    incs = rng.multivariate_normal(np.zeros(2), cov, size=n - 1)
    if extra_noise_sigma > 0:
        noise = rng.normal(scale=extra_noise_sigma, size=(n - 1, 2))
        incs = incs + noise
    x = np.zeros(n)
    y = np.zeros(n)
    x[1:] = np.cumsum(incs[:, 0])
    y[1:] = np.cumsum(incs[:, 1])
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


# ---------------------------------------------------------------------------
# (a) rho clamp
# ---------------------------------------------------------------------------


@FUZZ
@given(
    rho=st.floats(min_value=-0.99, max_value=0.99, allow_nan=False),
    n=st.integers(min_value=500, max_value=3000),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_rho_always_in_unit_interval(rho: float, n: int, seed: int):
    states_i, states_j = _make_path(rho, n, seed=seed)
    entry = estimate_correlation(states_i, states_j, [], [])
    assert -1.0 <= entry.rho <= 1.0, f"ρ={entry.rho} escaped [-1, 1]"


# ---------------------------------------------------------------------------
# (b) NotEnoughSamplesError boundary
# ---------------------------------------------------------------------------


@FUZZ
@given(
    n=st.integers(min_value=50, max_value=400),
    min_samples=st.integers(min_value=10, max_value=300),
)
def test_not_enough_samples_boundary(n: int, min_samples: int):
    """If the (unmasked) increment count is below min_samples, raise;
    otherwise succeed. With no jumps, all n-1 increments survive."""
    states_i, states_j = _make_path(0.2, n, seed=1)
    survivors = n - 1  # no jumps
    if survivors < min_samples:
        with pytest.raises(NotEnoughSamplesError):
            estimate_correlation(
                states_i, states_j, [], [], min_samples=min_samples,
            )
    else:
        entry = estimate_correlation(
            states_i, states_j, [], [], min_samples=min_samples,
        )
        assert -1.0 <= entry.rho <= 1.0


# ---------------------------------------------------------------------------
# (c) Uncorrelated noise doesn't flip sign of rho
# ---------------------------------------------------------------------------


@FUZZ
@given(
    rho=st.floats(min_value=0.2, max_value=0.8, allow_nan=False),
    noise_sigma=st.floats(min_value=0.001, max_value=0.01, allow_nan=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_small_noise_preserves_rho_sign_positive(rho, noise_sigma, seed):
    """For moderate-ρ positive paths, adding small i.i.d. noise to both
    series should keep the estimate positive (magnitude may shrink)."""
    n = 5000
    si, sj = _make_path(rho, n, seed=seed, extra_noise_sigma=noise_sigma)
    entry = estimate_correlation(si, sj, [], [])
    assert entry.rho > 0.0, (
        f"uncorrelated noise flipped ρ sign: true={rho} est={entry.rho} "
        f"noise_σ={noise_sigma}"
    )


@FUZZ
@given(
    rho=st.floats(min_value=-0.8, max_value=-0.2, allow_nan=False),
    noise_sigma=st.floats(min_value=0.001, max_value=0.01, allow_nan=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_small_noise_preserves_rho_sign_negative(rho, noise_sigma, seed):
    n = 5000
    si, sj = _make_path(rho, n, seed=seed, extra_noise_sigma=noise_sigma)
    entry = estimate_correlation(si, sj, [], [])
    assert entry.rho < 0.0, (
        f"uncorrelated noise flipped ρ sign: true={rho} est={entry.rho}"
    )


# ---------------------------------------------------------------------------
# (d) Co-jump count non-negative
# ---------------------------------------------------------------------------


@FUZZ
@given(
    rho=st.floats(min_value=-0.9, max_value=0.9, allow_nan=False),
    n=st.integers(min_value=500, max_value=2000),
    n_jumps_i=st.integers(min_value=0, max_value=10),
    n_jumps_j=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_co_jump_lambda_non_negative(
    rho, n, n_jumps_i, n_jumps_j, seed,
):
    rng = np.random.default_rng(seed)
    jumps_i_idx = rng.choice(range(1, n - 1), size=min(n_jumps_i, n - 2), replace=False)
    jumps_j_idx = rng.choice(range(1, n - 1), size=min(n_jumps_j, n - 2), replace=False)
    jumps_i = [T0 + timedelta(seconds=int(i)) for i in jumps_i_idx]
    jumps_j = [T0 + timedelta(seconds=int(i)) for i in jumps_j_idx]

    states_i, states_j = _make_path(rho, n, seed=seed)
    try:
        entry = estimate_correlation(states_i, states_j, jumps_i, jumps_j)
    except NotEnoughSamplesError:
        return  # legitimate — too many jumps masked out the series
    assert entry.co_jump_lambda >= 0.0, f"λ_ij={entry.co_jump_lambda} < 0"
    assert entry.co_jump_m2 >= 0.0, f"m2_ij={entry.co_jump_m2} < 0"
