"""Paper-engine matching logic: book-through fills, trade-tick fills,
queue haircut, and halt-on-feed-gap."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from blksch.exec.ledger import Ledger
from blksch.exec.paper_engine import PaperEngine, PaperEngineConfig
from blksch.schemas import (
    BookSnap,
    Order,
    OrderSide,
    OrderStatus,
    PriceLevel,
    TradeSide,
    TradeTick,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _snap(bids, asks, token="tok", ts=None):
    return BookSnap(
        token_id=token,
        bids=[PriceLevel(price=p, size=s) for p, s in bids],
        asks=[PriceLevel(price=p, size=s) for p, s in asks],
        ts=ts or _now(),
    )


def _order(side, price, size, *, cid="c1", token="tok"):
    return Order(
        token_id=token, side=side, price=price, size=size,
        client_id=cid, status=OrderStatus.PENDING, created_ts=_now(),
    )


@pytest.fixture()
def engine():
    ledger = Ledger.in_memory()
    return PaperEngine(ledger, config=PaperEngineConfig(queue_haircut=0.5, feed_gap_sec=3.0))


async def test_buy_fills_on_book_through(engine: PaperEngine):
    buy = _order(OrderSide.BUY, 0.50, 100)
    placed = await engine.place_order(buy)
    assert placed.status is OrderStatus.OPEN
    assert placed.venue_id and placed.venue_id.startswith("paper-")

    # ask moves below our bid -> trade-through signal; available = 40 * (1 - 0.5) = 20
    fills = await engine.on_book(_snap(bids=[(0.45, 10)], asks=[(0.48, 40)]))
    assert len(fills) == 1
    f = fills[0]
    assert f.side is OrderSide.BUY
    assert f.price == 0.50  # passive maker price
    assert abs(f.size - 20.0) < 1e-9


async def test_sell_fills_on_trade_tick(engine: PaperEngine):
    sell = _order(OrderSide.SELL, 0.55, 100, cid="s1")
    await engine.place_order(sell)

    # aggressor BUY at 0.56 crosses our 0.55 ask; haircut -> fill 0.5*30=15
    tick = TradeTick(
        token_id="tok", price=0.56, size=30,
        aggressor_side=TradeSide.BUY, ts=_now(),
    )
    fills = await engine.on_trade(tick)
    assert len(fills) == 1
    f = fills[0]
    assert f.side is OrderSide.SELL
    assert f.price == 0.55
    assert abs(f.size - 15.0) < 1e-9


async def test_no_fill_when_book_does_not_cross(engine: PaperEngine):
    await engine.place_order(_order(OrderSide.BUY, 0.50, 100))
    # ask strictly above our bid -> no fill
    fills = await engine.on_book(_snap(bids=[(0.48, 50)], asks=[(0.52, 50)]))
    assert fills == []


async def test_partial_fill_preserves_remaining_size(engine: PaperEngine):
    await engine.place_order(_order(OrderSide.BUY, 0.50, 100))
    # small crossing volume -> partial
    fills1 = await engine.on_book(_snap(bids=[(0.45, 5)], asks=[(0.48, 10)]))
    assert abs(fills1[0].size - 5.0) < 1e-9

    opens = engine.open_orders()
    assert len(opens) == 1
    # remaining tracked internally in _RestingOrder, not in the Order model
    resting = list(engine._resting.values())[0]
    assert abs(resting.remaining - 95.0) < 1e-9
    assert resting.order.status is OrderStatus.OPEN  # not yet filled on the engine side


async def test_cancel_removes_resting_order(engine: PaperEngine):
    placed = await engine.place_order(_order(OrderSide.BUY, 0.50, 10))
    assert await engine.cancel_order(placed.client_id) is True
    assert engine.open_orders() == []
    stored = engine.ledger.get_order(placed.client_id)
    assert stored.status is OrderStatus.CANCELED


async def test_self_cross_rejected(engine: PaperEngine):
    await engine.place_order(_order(OrderSide.BUY, 0.55, 10, cid="buy"))
    # ask at 0.50 would cross our own 0.55 bid
    ask = await engine.place_order(_order(OrderSide.SELL, 0.50, 10, cid="ask"))
    assert ask.status is OrderStatus.REJECTED


async def test_feed_gap_halts_engine():
    ledger = Ledger.in_memory()
    eng = PaperEngine(ledger, config=PaperEngineConfig(feed_gap_sec=1.0))
    await eng.place_order(_order(OrderSide.BUY, 0.50, 10))

    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    await eng.on_book(_snap(bids=[(0.45, 10)], asks=[(0.55, 10)], ts=t0))
    # big gap — > feed_gap_sec
    t1 = t0 + timedelta(seconds=5)
    fills = await eng.on_book(_snap(bids=[(0.45, 10)], asks=[(0.48, 50)], ts=t1))
    assert eng.state.halted is True
    assert fills == []  # no fills after halt
    assert "feed_gap" in (eng.state.halt_reason or "")


async def test_halted_engine_rejects_new_orders():
    ledger = Ledger.in_memory()
    eng = PaperEngine(ledger, config=PaperEngineConfig())
    eng.halt("manual test")
    placed = await eng.place_order(_order(OrderSide.BUY, 0.50, 10))
    assert placed.status is OrderStatus.REJECTED


async def test_fee_bps_applies_to_buy_notional():
    ledger = Ledger.in_memory()
    eng = PaperEngine(ledger, config=PaperEngineConfig(queue_haircut=0.0, fee_bps=50))
    await eng.place_order(_order(OrderSide.BUY, 0.50, 10))
    fills = await eng.on_book(_snap(bids=[(0.4, 10)], asks=[(0.48, 10)]))
    # notional = 0.50 * 10 = 5 USDC; fee = 5 * 50/10_000 = 0.025
    assert abs(fills[0].fee_usd - 0.025) < 1e-12
