"""Unit tests for ``core/filter/canonical_mid.CanonicalMidFilter`` (paper §5.1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import log

import pytest

from blksch.core.filter.canonical_mid import (
    CanonicalMid,
    CanonicalMidFilter,
    clip_p,
    iqr,
    trade_vwap,
)
from blksch.schemas import BookSnap, PriceLevel, TradeSide, TradeTick

pytestmark = pytest.mark.unit


def _snap(tok: str, ts: datetime, mid: float, *, half_spread: float = 0.01, size: float = 1000) -> BookSnap:
    return BookSnap(
        token_id=tok,
        bids=[PriceLevel(price=round(mid - half_spread, 6), size=size)],
        asks=[PriceLevel(price=round(mid + half_spread, 6), size=size)],
        ts=ts,
    )


def _trade(tok: str, ts: datetime, price: float, size: float = 10.0) -> TradeTick:
    return TradeTick(
        token_id=tok,
        price=price,
        size=size,
        aggressor_side=TradeSide.BUY,
        ts=ts,
    )


# ---------- Pure helpers ----------


def test_clip_p_enforces_bounds() -> None:
    assert clip_p(0.5, 1e-5) == pytest.approx(0.5)
    assert clip_p(0.0, 1e-5) == pytest.approx(1e-5)
    assert clip_p(1.0, 1e-5) == pytest.approx(1 - 1e-5)
    assert clip_p(-0.1, 1e-5) == pytest.approx(1e-5)
    assert clip_p(1.1, 1e-5) == pytest.approx(1 - 1e-5)


def test_trade_vwap_basic() -> None:
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    trades = [
        _trade("tok", t0, price=0.40, size=100),
        _trade("tok", t0, price=0.60, size=100),
    ]
    # Notional-weighted VWAP:
    #  w0 = 0.40*100 = 40;  w1 = 0.60*100 = 60
    #  VWAP = (40*0.40 + 60*0.60) / (40+60) = (16 + 36) / 100 = 0.52
    assert trade_vwap(trades) == pytest.approx(0.52)


def test_trade_vwap_empty() -> None:
    assert trade_vwap([]) is None


def test_iqr_minimum_width() -> None:
    assert iqr([]) == 0.0
    assert iqr([5.0]) == 0.0
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Q1=2, Q3=4 → IQR=2
    assert iqr(vals) == pytest.approx(2.0)


# ---------- Constructor validation ----------


def test_filter_rejects_bad_eps() -> None:
    with pytest.raises(ValueError):
        CanonicalMidFilter(token_id="t", eps=0.0)
    with pytest.raises(ValueError):
        CanonicalMidFilter(token_id="t", eps=0.6)


def test_filter_rejects_nonpositive_grid() -> None:
    with pytest.raises(ValueError):
        CanonicalMidFilter(token_id="t", grid_hz=0)


# ---------- Required scenarios ----------


def test_trade_weighted_mid_collapses_to_book_mid_when_no_trades() -> None:
    filt = CanonicalMidFilter(token_id="t", trade_window_sec=30.0)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    filt.update(_snap("t", t0, mid=0.50))
    outs = filt.update(_snap("t", t0 + timedelta(seconds=1), mid=0.50))
    assert len(outs) == 1
    out = outs[0]
    assert out.source == "book_mid"
    assert out.p_tilde == pytest.approx(0.50)
    assert out.trades_in_window == 0
    assert not out.forward_filled


def test_trade_vwap_used_when_trades_in_window() -> None:
    filt = CanonicalMidFilter(token_id="t", trade_window_sec=30.0)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    filt.update(_snap("t", t0, mid=0.50))
    trades = [_trade("t", t0 + timedelta(seconds=0.1), price=0.60, size=100)]
    outs = filt.update(_snap("t", t0 + timedelta(seconds=1), mid=0.50), trades=trades)
    assert len(outs) == 1
    assert outs[0].source == "trade_vwap"
    assert outs[0].p_tilde == pytest.approx(0.60)
    assert outs[0].trades_in_window == 1


def test_trade_window_purges_old_trades() -> None:
    filt = CanonicalMidFilter(token_id="t", trade_window_sec=5.0)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    filt.update(_snap("t", t0, mid=0.50), trades=[_trade("t", t0, price=0.70)])
    outs = filt.update(_snap("t", t0 + timedelta(seconds=10), mid=0.50))
    fresh = [o for o in outs if not o.forward_filled]
    assert fresh
    assert fresh[-1].source == "book_mid"
    assert fresh[-1].trades_in_window == 0


# ---------- Clipping ----------


def test_clipping_activates_near_p_zero() -> None:
    eps = 1e-5
    filt = CanonicalMidFilter(token_id="t", eps=eps)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    snap_low = BookSnap(
        token_id="t",
        bids=[PriceLevel(price=0.0, size=100)],
        asks=[PriceLevel(price=eps / 100, size=100)],
        ts=t0,
    )
    filt.update(snap_low)
    outs = filt.update(
        BookSnap(
            token_id="t",
            bids=[PriceLevel(price=0.0, size=100)],
            asks=[PriceLevel(price=eps / 100, size=100)],
            ts=t0 + timedelta(seconds=1),
        )
    )
    assert outs
    assert outs[0].p_tilde == pytest.approx(eps)
    assert outs[0].y == pytest.approx(log(eps / (1 - eps)))


def test_clipping_activates_near_p_one() -> None:
    eps = 1e-5
    filt = CanonicalMidFilter(token_id="t", eps=eps)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    snap_high = BookSnap(
        token_id="t",
        bids=[PriceLevel(price=1.0 - eps / 100, size=100)],
        asks=[PriceLevel(price=1.0, size=100)],
        ts=t0,
    )
    filt.update(snap_high)
    outs = filt.update(
        BookSnap(
            token_id="t",
            bids=[PriceLevel(price=1.0 - eps / 100, size=100)],
            asks=[PriceLevel(price=1.0, size=100)],
            ts=t0 + timedelta(seconds=1),
        )
    )
    assert outs
    assert outs[0].p_tilde == pytest.approx(1 - eps)


# ---------- Outlier rejection ----------


def test_outlier_rejected_then_recovers() -> None:
    filt = CanonicalMidFilter(
        token_id="t",
        outlier_k_iqr=3.0,
        outlier_min_history=4,
    )
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    mids = [0.50, 0.501, 0.499, 0.502, 0.498, 0.500, 0.501]
    for i, m in enumerate(mids):
        filt.update(_snap("t", t0 + timedelta(seconds=i), mid=m))

    # Inject massive spike — Δlogit >> K*IQR.
    spike_outs = filt.update(_snap("t", t0 + timedelta(seconds=len(mids)), mid=0.95))
    spike_emit = next(o for o in spike_outs if not o.forward_filled)
    assert spike_emit.rejected_outlier
    assert spike_emit.source == "outlier_fallback"
    # p_tilde carried forward — near 0.5, not 0.95.
    assert spike_emit.p_tilde == pytest.approx(0.501, abs=0.01)

    # Next normal tick recovers.
    ok_outs = filt.update(_snap("t", t0 + timedelta(seconds=len(mids) + 1), mid=0.50))
    ok_emit = next(o for o in ok_outs if not o.forward_filled)
    assert not ok_emit.rejected_outlier
    assert ok_emit.source == "book_mid"
    assert ok_emit.p_tilde == pytest.approx(0.50)


def test_early_history_skips_outlier_check() -> None:
    """Before outlier_min_history samples, no rejection can fire."""
    filt = CanonicalMidFilter(token_id="t", outlier_min_history=10)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    filt.update(_snap("t", t0, mid=0.50))
    outs = filt.update(_snap("t", t0 + timedelta(seconds=1), mid=0.95))
    emit = next(o for o in outs if not o.forward_filled)
    assert not emit.rejected_outlier


# ---------- Cadence / forward-fill ----------


def test_grid_cadence_emits_at_configured_hz() -> None:
    filt = CanonicalMidFilter(token_id="t", grid_hz=1.0)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    filt.update(_snap("t", t0, mid=0.50))
    outs = filt.update(_snap("t", t0 + timedelta(seconds=3), mid=0.50))
    tss = [o.ts for o in outs]
    assert tss == [
        t0 + timedelta(seconds=1),
        t0 + timedelta(seconds=2),
        t0 + timedelta(seconds=3),
    ]


def test_forward_fill_on_gap_carries_and_flags() -> None:
    filt = CanonicalMidFilter(token_id="t", grid_hz=1.0)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    filt.update(_snap("t", t0, mid=0.42))
    outs = filt.update(_snap("t", t0 + timedelta(seconds=4), mid=0.42))
    assert [o.forward_filled for o in outs] == [False, True, True, False]
    assert all(o.p_tilde == pytest.approx(0.42) for o in outs)
    assert outs[1].source == "forward_fill"
    assert outs[2].source == "forward_fill"


def test_forward_fill_and_outlier_flags_are_independent() -> None:
    filt = CanonicalMidFilter(
        token_id="t",
        outlier_k_iqr=3.0,
        outlier_min_history=4,
        grid_hz=1.0,
    )
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    for i, m in enumerate([0.50, 0.501, 0.499, 0.502, 0.498, 0.500]):
        filt.update(_snap("t", t0 + timedelta(seconds=i * 0.1), mid=m))

    outs = filt.update(_snap("t", t0 + timedelta(seconds=1), mid=0.95))
    rejected = [o for o in outs if o.rejected_outlier]
    assert rejected and not rejected[0].forward_filled

    outs_gap = filt.update(_snap("t", t0 + timedelta(seconds=4), mid=0.50))
    ffs = [o for o in outs_gap if o.forward_filled]
    assert ffs
    assert all(not o.rejected_outlier for o in ffs)


# ---------- Bad input hygiene ----------


def test_crossed_book_is_skipped() -> None:
    filt = CanonicalMidFilter(token_id="t")
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    filt.update(_snap("t", t0, mid=0.50))
    bad = BookSnap(
        token_id="t",
        bids=[PriceLevel(price=0.55, size=10)],
        asks=[PriceLevel(price=0.50, size=10)],
        ts=t0 + timedelta(seconds=0.5),
    )
    filt.update(bad)
    outs = filt.update(_snap("t", t0 + timedelta(seconds=1), mid=0.50))
    emit = [o for o in outs if o.ts == t0 + timedelta(seconds=1)][0]
    assert not emit.forward_filled


def test_empty_book_is_skipped() -> None:
    filt = CanonicalMidFilter(token_id="t")
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    empty = BookSnap(token_id="t", bids=[], asks=[], ts=t0)
    assert filt.update(empty) == []
    filt.update(_snap("t", t0 + timedelta(seconds=0.5), mid=0.50))
    outs = filt.update(_snap("t", t0 + timedelta(seconds=1), mid=0.50))
    assert outs


# ---------- update() tick-only mode ----------


def test_update_with_now_ts_forward_fills_without_new_data() -> None:
    filt = CanonicalMidFilter(token_id="t", grid_hz=1.0)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    filt.update(_snap("t", t0, mid=0.33))
    outs = filt.update(None, now_ts=t0 + timedelta(seconds=3))
    assert len(outs) == 3
    assert all(o.p_tilde == pytest.approx(0.33) for o in outs)
    assert [o.forward_filled for o in outs] == [False, True, True]


def test_update_requires_snap_or_now_ts() -> None:
    filt = CanonicalMidFilter(token_id="t")
    with pytest.raises(ValueError):
        filt.update(None)
