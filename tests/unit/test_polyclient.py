"""Unit tests for core/ingest/polyclient parsers + rate limiter."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime

import pytest

from blksch.core.ingest.polyclient import (
    AsyncRateLimiter,
    MockPolyClient,
    PolyClient,
    _dispatch_ws_message,
    apply_price_change,
    book_from_rest,
    book_from_ws,
    is_mock_token,
    trade_from_ws,
)
from blksch.schemas import BookSnap, TradeSide, TradeTick


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


# ---------- Mock-mode client ----------


def test_is_mock_token_recognizes_prefixes() -> None:
    assert is_mock_token("0xmock")
    assert is_mock_token("0xmockFOO")
    assert is_mock_token("0xMOCK")          # case-insensitive
    assert is_mock_token("mock:btc-70k")
    assert not is_mock_token("0xabc123")
    assert not is_mock_token("")


async def test_mock_polyclient_emits_scripted_stream() -> None:
    """MockPolyClient produces a bounded, schema-valid stream: 10 book
    snapshots show a token_id we asked for, prices in [0, 1], and at
    least one adjacent pair with a drifting mid (proves the GBM
    mechanic is live, not returning a constant book)."""
    t0 = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    mock = MockPolyClient(
        seed=42, interval_s=0.001, start_ts=t0, trade_probability=0.0,
    )
    stream = mock.stream_market(["0xmock-tok-a"])
    snaps: list[BookSnap] = []
    async for ev in stream:
        if isinstance(ev, BookSnap):
            snaps.append(ev)
        if len(snaps) >= 10:
            break
    await stream.aclose()

    assert len(snaps) == 10
    for s in snaps:
        assert s.token_id == "0xmock-tok-a"
        assert s.bids and s.asks
        assert 0.0 < s.bids[0].price < s.asks[0].price < 1.0
    # GBM drifted the mid at least once (not a constant book).
    mids = [s.mid for s in snaps]
    assert len(set(mids)) > 1, f"mid never changed: {mids}"


async def test_mock_polyclient_deterministic_under_seed() -> None:
    """Two MockPolyClients with the same (seed, start_ts) produce
    byte-identical event sequences. This is the property every
    replay-based unit test relies on."""
    t0 = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)

    async def _collect(n: int) -> list[tuple]:
        mock = MockPolyClient(
            seed=4242, interval_s=0.001, start_ts=t0, trade_probability=0.25,
        )
        stream = mock.stream_market(["0xmock-det"])
        out: list[tuple] = []
        async for ev in stream:
            if isinstance(ev, BookSnap):
                out.append((
                    "book", ev.token_id,
                    ev.bids[0].price, ev.bids[0].size,
                    ev.asks[0].price, ev.asks[0].size,
                    ev.ts.isoformat(),
                ))
            else:
                assert isinstance(ev, TradeTick)
                out.append((
                    "trade", ev.token_id, ev.price, ev.size,
                    ev.aggressor_side.value, ev.ts.isoformat(),
                ))
            if len(out) >= n:
                break
        await stream.aclose()
        return out

    run_a = await _collect(30)
    run_b = await _collect(30)
    assert run_a == run_b, "same seed+start_ts should yield identical streams"


async def test_mock_polyclient_routing_by_token() -> None:
    """A plain ``PolyClient`` (no ``start()`` called, no aiohttp session)
    still serves mock-token requests end-to-end via the mock-routing
    branch in ``stream_market`` / ``get_book``. If the routing were
    missing, ``get_book`` would blow up on ``self.session`` (no session)
    and ``stream_market`` would try to open a real WS."""
    client = PolyClient()
    # No client.start() — the session is None. If routing is wired,
    # both calls succeed without touching network.
    snap = await client.get_book("0xmock-routed")
    assert snap.token_id == "0xmock-routed"
    assert snap.bids and snap.asks

    stream = client.stream_market(["0xmock-routed"])
    ev = await stream.__anext__()
    assert isinstance(ev, (BookSnap, TradeTick))
    await stream.aclose()

    # Mixing mock and real must fail fast (would otherwise either block
    # on a real WS connect or silently mismatch quote uptime).
    mixed = client.stream_market(["0xmock-a", "0xabc123"])
    with pytest.raises(ValueError, match="mix mock and real"):
        await mixed.__anext__()
    await mixed.aclose()


async def test_mock_polyclient_aclose_on_cancel() -> None:
    """The mock stream propagates ``CancelledError`` through the async
    generator and shuts down cleanly when the consumer cancels.
    Regression guard for the aclose pattern that the real ``PolyClient``
    also needs on every exit path (94e39c0)."""
    t0 = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    mock = MockPolyClient(
        seed=7, interval_s=0.001, start_ts=t0, trade_probability=0.0,
    )
    stream = mock.stream_market(["0xmock-cancel"])
    received: list[BookSnap] = []

    async def _consume() -> None:
        try:
            async for ev in stream:
                assert isinstance(ev, BookSnap)
                received.append(ev)
        except asyncio.CancelledError:
            # Re-raise so the task records cancellation — the generator
            # cleanup runs via the async-for's implicit aclose path.
            raise

    task = asyncio.create_task(_consume())
    # Wait until we've seen at least one event, then cancel.
    for _ in range(200):
        if received:
            break
        await asyncio.sleep(0.001)
    assert received, "mock stream produced no events before cancel"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Explicit aclose on the generator is still accepted (idempotent).
    await stream.aclose()
