"""Property-based fuzz for ``core/filter/canonical_mid.py`` (paper §5.1).

Targets Window A's code — this suite is error-detection only. If a property
fails, the fix lives on Window A's next branch; we record the failing seed
and report, not patch.

Invariants under test:

* Output ``p_tilde`` stays in ``[eps, 1 - eps]`` for every valid book.
* ``forward_filled=True`` iff no new BookSnap arrived in the grid interval.
* Outlier rejection is stateful-idempotent: feeding the same rejected tick
  twice produces identical internal state.
* Grid cadence: consecutive outputs are spaced exactly ``1/grid_hz`` seconds
  (no gaps, no doubles).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from blksch.core.filter.canonical_mid import (
    DEFAULT_EPS,
    CanonicalMid,
    CanonicalMidFilter,
)
from blksch.schemas import BookSnap, PriceLevel, TradeSide, TradeTick

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)

FUZZ = settings(
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _valid_book(draw) -> tuple[float, float, float, float]:
    """Draw (bid, ask, bid_sz, ask_sz) such that 0 < bid < ask < 1."""
    bid = draw(st.floats(min_value=0.01, max_value=0.98, allow_nan=False))
    spread = draw(st.floats(min_value=0.001, max_value=0.1, allow_nan=False))
    ask = min(0.99, bid + spread)
    if ask <= bid:
        ask = min(0.99, bid + 0.01)
    bid_sz = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False))
    ask_sz = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False))
    return bid, ask, bid_sz, ask_sz


def _snap(bid: float, ask: float, bid_sz: float, ask_sz: float, ts: datetime) -> BookSnap:
    return BookSnap(
        token_id="tok",
        bids=[PriceLevel(price=bid, size=bid_sz)],
        asks=[PriceLevel(price=ask, size=ask_sz)],
        ts=ts,
    )


# ---------------------------------------------------------------------------
# (a) Clipping property
# ---------------------------------------------------------------------------


@FUZZ
@given(
    books=st.lists(_valid_book(), min_size=5, max_size=40),
    gap_ms=st.integers(min_value=100, max_value=2000),
)
def test_p_tilde_always_in_clipped_interval(books, gap_ms):
    f = CanonicalMidFilter(token_id="tok", grid_hz=1.0, eps=DEFAULT_EPS)
    ts = T0
    for bid, ask, bsz, asz in books:
        snap = _snap(bid, ask, bsz, asz, ts)
        outputs = f.update(snap)
        for out in outputs:
            assert DEFAULT_EPS <= out.p_tilde <= 1.0 - DEFAULT_EPS, (
                f"p_tilde={out.p_tilde} escaped [eps, 1-eps]"
            )
            # y = logit(p_tilde) should be finite
            assert math.isfinite(out.y), f"y={out.y} is non-finite"
        ts = ts + timedelta(milliseconds=gap_ms)


# ---------------------------------------------------------------------------
# (b) forward_filled iff no new BookSnap in the grid interval
# ---------------------------------------------------------------------------


@FUZZ
@given(
    book=_valid_book(),
    n_empty_ticks=st.integers(min_value=2, max_value=5),
)
def test_forward_filled_true_iff_no_snap_in_interval(book, n_empty_ticks):
    """After an initial snap, advance the clock through multiple empty
    grid intervals. Every emit whose bin contained no snap must be
    forward_filled=True; every emit whose bin saw a fresh snap must be
    forward_filled=False."""
    bid, ask, bsz, asz = book
    f = CanonicalMidFilter(token_id="tok", grid_hz=1.0)

    # Seed mid-bin so the first emit's bin contains the seed snap.
    f.update(_snap(bid, ask, bsz, asz, T0 + timedelta(milliseconds=500)))

    # Tick forward through N *fully empty* bins. The emit at T0+1s will
    # carry the seed data (its bin was [T0, T0+1s) containing T0+500ms),
    # so we collect emits from strictly after that.
    emits: list[CanonicalMid] = []
    for k in range(1, n_empty_ticks + 2):
        outs = f.update(None, now_ts=T0 + timedelta(seconds=k))
        emits.extend(outs)

    # First emit (at T0+1s) saw the seed — not forward-filled.
    assert emits, "expected at least one emit"
    assert emits[0].forward_filled is False, (
        f"first emit after seed must be fresh; got ff={emits[0].forward_filled}"
    )
    # Every subsequent emit had an empty bin → must be forward-filled.
    for t in emits[1:]:
        assert t.forward_filled is True, (
            f"empty bin emit at {t.ts} not marked forward_filled"
        )

    # Deliver a real snap two bins later → that emit is fresh again.
    fresh_ts = T0 + timedelta(seconds=n_empty_ticks + 2)
    fresh_out = f.update(_snap(bid, ask, bsz, asz, fresh_ts))
    fresh_matches = [o for o in fresh_out if o.ts == fresh_ts]
    if fresh_matches:
        assert fresh_matches[0].forward_filled is False, (
            f"fresh snap at {fresh_ts} still marked forward_filled"
        )


# ---------------------------------------------------------------------------
# (c) Outlier rejection is stateful-idempotent
# ---------------------------------------------------------------------------


@FUZZ
@given(
    history=st.lists(_valid_book(), min_size=12, max_size=30),
)
def test_rejected_tick_is_stateful_idempotent(history):
    """Feed the same outlier tick twice — the filter's observable state
    (last emitted p̃) must be unchanged after the second feed."""
    f = CanonicalMidFilter(
        token_id="tok", grid_hz=1.0, outlier_k_iqr=2.0, outlier_min_history=4,
    )

    # Build up history.
    ts = T0
    last_out: CanonicalMid | None = None
    for bid, ask, bsz, asz in history:
        snap = _snap(bid, ask, bsz, asz, ts)
        outs = f.update(snap)
        if outs:
            last_out = outs[-1]
        ts = ts + timedelta(milliseconds=1100)

    # Craft an extreme-outlier tick far from any seen prices.
    extreme_ts = ts + timedelta(milliseconds=1100)
    outlier = _snap(0.001, 0.005, 5.0, 5.0, extreme_ts)
    outs1 = f.update(outlier)
    state_after_first = (f._last_p_tilde, f._last_source, list(f._logit_history))

    # Feed the same outlier again at a later grid boundary.
    outlier2 = _snap(0.001, 0.005, 5.0, 5.0, extreme_ts + timedelta(milliseconds=1100))
    outs2 = f.update(outlier2)
    state_after_second = (f._last_p_tilde, f._last_source, list(f._logit_history))

    # If the filter rejected the outlier, the last_p_tilde carries forward
    # (unchanged); if accepted, two identical accepts must produce identical
    # state (p_tilde equal).
    assert state_after_first[0] == pytest.approx(state_after_second[0], abs=1e-12), (
        f"repeated outlier feed changed p_tilde from "
        f"{state_after_first[0]} to {state_after_second[0]}"
    )


# ---------------------------------------------------------------------------
# (d) Grid cadence is exact — no gaps, no doubles
# ---------------------------------------------------------------------------


@FUZZ
@given(
    books=st.lists(_valid_book(), min_size=3, max_size=25),
    grid_hz=st.sampled_from([0.5, 1.0, 2.0]),
    gap_ms=st.integers(min_value=300, max_value=4000),
)
def test_grid_cadence_is_exact(books, grid_hz, gap_ms):
    f = CanonicalMidFilter(token_id="tok", grid_hz=grid_hz)
    period = timedelta(seconds=1.0 / grid_hz)
    all_outs: list[CanonicalMid] = []
    ts = T0
    for bid, ask, bsz, asz in books:
        all_outs.extend(f.update(_snap(bid, ask, bsz, asz, ts)))
        ts = ts + timedelta(milliseconds=gap_ms)

    # Timestamps of emitted ticks must be strictly increasing and separated
    # by exactly `period` — no gap, no double.
    for prev, nxt in zip(all_outs, all_outs[1:]):
        delta = nxt.ts - prev.ts
        assert delta == period, (
            f"grid cadence broken: {prev.ts} → {nxt.ts} "
            f"delta={delta} != {period}"
        )
