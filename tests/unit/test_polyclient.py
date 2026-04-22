"""Unit tests for core/ingest/polyclient parsers + rate limiter."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from blksch.core.ingest.polyclient import (
    AsyncRateLimiter,
    _dispatch_ws_message,
    apply_price_change,
    book_from_rest,
    book_from_ws,
    trade_from_ws,
)
from blksch.schemas import BookSnap, TradeSide


# ---------- Rate limiter ----------


async def test_rate_limiter_enforces_interval() -> None:
    rl = AsyncRateLimiter(rate_per_sec=20.0)  # 50ms interval
    t0 = time.monotonic()
    for _ in range(5):
        await rl.acquire()
    elapsed = time.monotonic() - t0
    # 5 acquires = at least 4 intervals of 50ms = 0.2s
    assert elapsed >= 0.19, f"expected >=0.19s, got {elapsed:.3f}s"


async def test_rate_limiter_serializes_concurrent() -> None:
    rl = AsyncRateLimiter(rate_per_sec=50.0)  # 20ms
    t0 = time.monotonic()
    await asyncio.gather(*(rl.acquire() for _ in range(5)))
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.075, f"expected >=0.075s, got {elapsed:.3f}s"


def test_rate_limiter_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        AsyncRateLimiter(rate_per_sec=0)


# ---------- REST book parser ----------


def test_book_from_rest_sorts_defensively() -> None:
    payload = {
        "bids": [
            {"price": "0.40", "size": "10"},  # worse bid first; parser must sort
            {"price": "0.48", "size": "100"},
        ],
        "asks": [
            {"price": "0.55", "size": "20"},
            {"price": "0.52", "size": "50"},  # better ask second; parser must sort
        ],
        "timestamp": "1714000000000",  # ms
    }
    snap = book_from_rest(payload, token_id="tok-1")
    assert snap.token_id == "tok-1"
    assert snap.bids[0].price == pytest.approx(0.48)  # highest bid first
    assert snap.asks[0].price == pytest.approx(0.52)  # lowest ask first
    assert snap.mid == pytest.approx(0.5)
    assert snap.spread == pytest.approx(0.04)


def test_book_from_rest_handles_empty_book() -> None:
    snap = book_from_rest({"bids": [], "asks": []}, token_id="tok")
    assert snap.bids == []
    assert snap.asks == []
    assert snap.mid is None
    assert snap.spread is None


def test_book_from_rest_accepts_seconds_timestamp() -> None:
    snap = book_from_rest({"bids": [], "asks": [], "timestamp": "1714000000"}, token_id="t")
    assert snap.ts.year == 2024  # 2024-04-25 UTC


def test_book_from_rest_missing_timestamp_uses_now() -> None:
    snap = book_from_rest({"bids": [], "asks": []}, token_id="t")
    # Just assert it parses to something; don't pin exact value.
    assert snap.ts is not None


# ---------- WS parsers ----------


def test_book_from_ws_matches_rest_shape() -> None:
    msg = {
        "event_type": "book",
        "asset_id": "ws-tok",
        "bids": [{"price": "0.49", "size": "30"}],
        "asks": [{"price": "0.51", "size": "40"}],
        "timestamp": "1714000000000",
    }
    snap = book_from_ws(msg)
    assert snap.token_id == "ws-tok"
    assert snap.bids[0].price == pytest.approx(0.49)
    assert snap.asks[0].size == pytest.approx(40.0)


def test_trade_from_ws_buy_side() -> None:
    msg = {
        "event_type": "last_trade_price",
        "asset_id": "t",
        "price": "0.5",
        "size": "10",
        "side": "BUY",
        "timestamp": "1714000000000",
    }
    tt = trade_from_ws(msg)
    assert tt.aggressor_side == TradeSide.BUY
    assert tt.price == pytest.approx(0.5)
    assert tt.size == pytest.approx(10.0)


def test_trade_from_ws_sell_side_lowercase() -> None:
    msg = {"asset_id": "t", "price": "0.7", "size": "3", "side": "sell", "timestamp": "1714000000000"}
    tt = trade_from_ws(msg)
    assert tt.aggressor_side == TradeSide.SELL


# ---------- price_change diff application ----------


def _base_snap() -> BookSnap:
    return book_from_ws(
        {
            "event_type": "book",
            "asset_id": "tok",
            "bids": [
                {"price": "0.48", "size": "100"},
                {"price": "0.47", "size": "50"},
            ],
            "asks": [
                {"price": "0.52", "size": "80"},
                {"price": "0.53", "size": "60"},
            ],
            "timestamp": "1714000000000",
        }
    )


def test_apply_price_change_updates_existing_level() -> None:
    base = _base_snap()
    msg = {
        "event_type": "price_change",
        "asset_id": "tok",
        "changes": [{"price": "0.48", "side": "BUY", "size": "150"}],
        "timestamp": "1714000000100",
    }
    updated = apply_price_change(base, msg)
    assert updated.bids[0].price == pytest.approx(0.48)
    assert updated.bids[0].size == pytest.approx(150.0)


def test_apply_price_change_removes_level_on_zero_size() -> None:
    base = _base_snap()
    msg = {
        "changes": [{"price": "0.52", "side": "SELL", "size": "0"}],
    }
    updated = apply_price_change(base, msg)
    # Top ask should now be 0.53 since 0.52 was removed.
    assert updated.asks[0].price == pytest.approx(0.53)


def test_apply_price_change_adds_new_level() -> None:
    base = _base_snap()
    msg = {"changes": [{"price": "0.49", "side": "BUY", "size": "200"}]}
    updated = apply_price_change(base, msg)
    # New top bid.
    assert updated.bids[0].price == pytest.approx(0.49)
    assert updated.bids[0].size == pytest.approx(200.0)


# ---------- dispatch ----------


def test_dispatch_book_event_caches_and_emits() -> None:
    books: dict[str, BookSnap] = {}
    frame = json.dumps(
        {
            "event_type": "book",
            "asset_id": "tok",
            "bids": [{"price": "0.48", "size": "10"}],
            "asks": [{"price": "0.52", "size": "10"}],
            "timestamp": "1714000000000",
        }
    )
    out = _dispatch_ws_message(frame, books, maintain_book_state=True)
    assert isinstance(out, BookSnap)
    assert "tok" in books


def test_dispatch_price_change_requires_base_snapshot() -> None:
    books: dict[str, BookSnap] = {}
    frame = json.dumps(
        {
            "event_type": "price_change",
            "asset_id": "tok",
            "changes": [{"price": "0.48", "side": "BUY", "size": "10"}],
        }
    )
    # No base snap -> dropped.
    assert _dispatch_ws_message(frame, books, maintain_book_state=True) is None


def test_dispatch_price_change_after_book() -> None:
    books: dict[str, BookSnap] = {}
    book_frame = json.dumps(
        {
            "event_type": "book",
            "asset_id": "tok",
            "bids": [{"price": "0.48", "size": "10"}],
            "asks": [{"price": "0.52", "size": "10"}],
            "timestamp": "1714000000000",
        }
    )
    _dispatch_ws_message(book_frame, books, maintain_book_state=True)
    change_frame = json.dumps(
        {
            "event_type": "price_change",
            "asset_id": "tok",
            "changes": [{"price": "0.49", "side": "BUY", "size": "5"}],
            "timestamp": "1714000000100",
        }
    )
    snap = _dispatch_ws_message(change_frame, books, maintain_book_state=True)
    assert isinstance(snap, BookSnap)
    assert snap.bids[0].price == pytest.approx(0.49)


def test_dispatch_trade_event() -> None:
    books: dict[str, BookSnap] = {}
    frame = json.dumps(
        {
            "event_type": "last_trade_price",
            "asset_id": "tok",
            "price": "0.5",
            "size": "7",
            "side": "SELL",
            "timestamp": "1714000000000",
        }
    )
    out = _dispatch_ws_message(frame, books, maintain_book_state=True)
    from blksch.schemas import TradeTick

    assert isinstance(out, TradeTick)
    assert out.aggressor_side == TradeSide.SELL
    assert out.size == pytest.approx(7.0)


def test_dispatch_unknown_event_returns_none() -> None:
    books: dict[str, BookSnap] = {}
    frame = json.dumps({"event_type": "tick_size_change", "asset_id": "tok"})
    assert _dispatch_ws_message(frame, books, maintain_book_state=True) is None


def test_dispatch_non_json_returns_none() -> None:
    books: dict[str, BookSnap] = {}
    assert _dispatch_ws_message("not json", books, maintain_book_state=True) is None


def test_dispatch_malformed_book_returns_none() -> None:
    books: dict[str, BookSnap] = {}
    frame = json.dumps(
        {
            "event_type": "book",
            "asset_id": "tok",
            "bids": [{"price": "not-a-number", "size": "10"}],
            "asks": [],
        }
    )
    assert _dispatch_ws_message(frame, books, maintain_book_state=True) is None
