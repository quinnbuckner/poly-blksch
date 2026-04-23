"""Property-based fuzz for ``core/filter/kalman.py`` (heteroskedastic KF).

Invariants under test (all on finite inputs):

* Kalman gain K_t ∈ [0, 1] (after the internal clip).
* No NaN / Inf in x̂, P, innovation, or K for any finite input.
* Determinism: running the filter forward, then replaying identical inputs
  on a fresh filter, yields identical outputs bit-for-bit.
* Inject a σ_η² = 1e-12 step — filter stays bounded (|x̂| < 1e6).

Posterior-variance monotone non-increase is property-tested across a
single update step (between innovations, the predict step adds Q so it's
*only* non-increasing at the update boundary, not across the full step).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Sequence

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from blksch.core.filter.kalman import KalmanFilter
from blksch.schemas import BookSnap, PriceLevel, TradeTick

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)

FUZZ = settings(max_examples=200, deadline=None, derandomize=True)


class _ConstantOracle:
    """VarianceOracle that returns a fixed σ_η² regardless of input."""

    def __init__(self, value: float) -> None:
        self.value = value

    def variance(
        self,
        book: BookSnap,
        trades: Sequence[TradeTick] = (),
        *,
        forward_filled: bool = False,
    ) -> float:
        return self.value


class _ScriptedOracle:
    """Oracle that returns a queued value per call, cycling."""

    def __init__(self, values: list[float]) -> None:
        assert values
        self.values = list(values)
        self.i = 0

    def variance(
        self,
        book: BookSnap,
        trades: Sequence[TradeTick] = (),
        *,
        forward_filled: bool = False,
    ) -> float:
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return v


def _cm(ts, y, p_tilde, forward_filled=False):
    from blksch.core.filter.canonical_mid import CanonicalMid
    return CanonicalMid(
        token_id="tok", ts=ts, p_tilde=p_tilde, y=y,
        forward_filled=forward_filled, rejected_outlier=False,
        trades_in_window=0, source="book_mid",
    )


def _snap(bid: float = 0.48, ask: float = 0.52, ts: datetime | None = None) -> BookSnap:
    return BookSnap(
        token_id="tok",
        bids=[PriceLevel(price=bid, size=100.0)],
        asks=[PriceLevel(price=ask, size=100.0)],
        ts=ts or T0,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _kalman_path(draw) -> tuple[float, list[float], list[float], float, float]:
    """(sigma_b, ys, R_values, dt_sec, initial_x)."""
    n = draw(st.integers(min_value=3, max_value=40))
    sigma_b = draw(st.floats(min_value=0.01, max_value=0.3, allow_nan=False))
    ys = draw(st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
        min_size=n, max_size=n,
    ))
    Rs = draw(st.lists(
        st.floats(min_value=1e-6, max_value=1.0, allow_nan=False),
        min_size=n, max_size=n,
    ))
    dt_sec = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False))
    init_x = draw(st.floats(min_value=-3.0, max_value=3.0, allow_nan=False))
    return sigma_b, ys, Rs, dt_sec, init_x


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@FUZZ
@given(path=_kalman_path())
def test_kalman_gain_in_unit_interval(path):
    sigma_b, ys, Rs, dt, init_x = path
    kf = KalmanFilter(
        token_id="tok", microstruct=_ScriptedOracle(Rs),
        sigma_b=sigma_b, initial_x=init_x,
    )
    ts = T0
    for y in ys:
        p = 1.0 / (1.0 + math.exp(-y))
        kf.step(_cm(ts, y, p), _snap(ts=ts))
        K = kf.last_K
        if K is not None:
            assert 0.0 <= K <= 1.0 + 1e-12, f"K={K} out of [0,1]"
        ts = ts + timedelta(seconds=dt)


@FUZZ
@given(path=_kalman_path())
def test_no_nan_or_inf(path):
    sigma_b, ys, Rs, dt, init_x = path
    kf = KalmanFilter(
        token_id="tok", microstruct=_ScriptedOracle(Rs),
        sigma_b=sigma_b, initial_x=init_x,
    )
    ts = T0
    for y in ys:
        p = 1.0 / (1.0 + math.exp(-y))
        state = kf.step(_cm(ts, y, p), _snap(ts=ts))
        assert math.isfinite(state.x_hat), f"x_hat={state.x_hat}"
        assert math.isfinite(kf.posterior_variance), f"P={kf.posterior_variance}"
        assert kf.posterior_variance > 0, "posterior variance must be positive"
        if kf.last_innovation is not None:
            assert math.isfinite(kf.last_innovation)
        ts = ts + timedelta(seconds=dt)


@FUZZ
@given(path=_kalman_path())
def test_determinism_forward_vs_replay(path):
    """Running twice on identical inputs produces bit-identical state."""
    sigma_b, ys, Rs, dt, init_x = path
    ts0 = T0

    def run():
        kf = KalmanFilter(
            token_id="tok", microstruct=_ScriptedOracle(Rs),
            sigma_b=sigma_b, initial_x=init_x,
        )
        outputs = []
        ts = ts0
        for y in ys:
            p = 1.0 / (1.0 + math.exp(-y))
            state = kf.step(_cm(ts, y, p), _snap(ts=ts))
            outputs.append((state.x_hat, kf.posterior_variance, kf.last_K))
            ts = ts + timedelta(seconds=dt)
        return outputs

    a = run()
    b = run()
    assert a == b, f"non-deterministic run: first={a[:3]} second={b[:3]}"


@FUZZ
@given(
    sigma_b=st.floats(min_value=0.01, max_value=0.3, allow_nan=False),
    y0=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
    n_steps=st.integers(min_value=3, max_value=20),
)
def test_tiny_R_spike_stays_bounded(sigma_b: float, y0: float, n_steps: int):
    """Inject one tick with σ_η² = 1e-12 — filter must not blow up."""
    values = [1e-3] * n_steps
    if len(values) > 1:
        values[len(values) // 2] = 1e-12
    kf = KalmanFilter(
        token_id="tok", microstruct=_ScriptedOracle(values),
        sigma_b=sigma_b, initial_x=y0,
    )
    ts = T0
    for i, R in enumerate(values):
        y = y0 + 0.1 * math.sin(i)
        p = 1.0 / (1.0 + math.exp(-y))
        state = kf.step(_cm(ts, y, p), _snap(ts=ts))
        assert abs(state.x_hat) < 1e6, f"|x̂|={state.x_hat} blew up at step {i}"
        assert math.isfinite(kf.posterior_variance)
        ts = ts + timedelta(seconds=1.0)


@FUZZ
@given(
    sigma_b=st.floats(min_value=0.01, max_value=0.3, allow_nan=False),
    y_stream=st.lists(
        st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
        min_size=5, max_size=30,
    ),
    R=st.floats(min_value=1e-4, max_value=0.5, allow_nan=False),
)
def test_posterior_variance_drops_at_update_boundary(sigma_b, y_stream, R):
    """Across a single predict+update cycle, the posterior variance after
    the update must be ≤ predict-time variance (P_pred). We approximate
    P_pred = P_prev + σ_b² · dt (the filter's own Q term).
    """
    kf = KalmanFilter(
        token_id="tok", microstruct=_ConstantOracle(R),
        sigma_b=sigma_b, initial_x=y_stream[0],
    )
    ts = T0
    prev_P = kf.posterior_variance
    dt = 1.0
    for i, y in enumerate(y_stream):
        p = 1.0 / (1.0 + math.exp(-y))
        kf.step(_cm(ts, y, p), _snap(ts=ts))
        new_P = kf.posterior_variance
        if i > 0:
            P_pred = prev_P + sigma_b * sigma_b * dt
            assert new_P <= P_pred + 1e-12, (
                f"posterior P={new_P} exceeded predict-variance P_pred={P_pred}"
            )
        prev_P = new_P
        ts = ts + timedelta(seconds=dt)
