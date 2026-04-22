"""Order router: idempotent sync_quote, retry/backoff, live-ack gate."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from blksch.exec.ledger import Ledger
from blksch.exec.order_router import OrderRouter, RouterConfig
from blksch.exec.paper_engine import PaperEngine
from blksch.schemas import Order, OrderSide, OrderStatus, Quote


def _now() -> datetime:
    return datetime.now(UTC)


def _quote(*, bid, ask, size_bid=10, size_ask=10, token="tok") -> Quote:
    return Quote(
        token_id=token,
        p_bid=bid, p_ask=ask,
        x_bid=0.0, x_ask=0.0,
        size_bid=size_bid, size_ask=size_ask,
        half_spread_x=0.02, reservation_x=0.0,
        inventory_q=0.0, ts=_now(),
    )


@pytest.fixture()
def paper_router():
    ledger = Ledger.in_memory()
    engine = PaperEngine(ledger)
    return engine, OrderRouter(paper_backend=engine, config=RouterConfig(mode="paper"))


async def test_sync_quote_creates_both_sides(paper_router):
    engine, router = paper_router
    result = await router.sync_quote(_quote(bid=0.48, ask=0.52))
    assert result["bid"] is not None and result["bid"].side is OrderSide.BUY
    assert result["ask"] is not None and result["ask"].side is OrderSide.SELL
    assert len(engine.open_orders()) == 2


async def test_sync_quote_is_idempotent_when_unchanged(paper_router):
    engine, router = paper_router
    first = await router.sync_quote(_quote(bid=0.48, ask=0.52))
    second = await router.sync_quote(_quote(bid=0.48, ask=0.52))
    # no new orders created, same client_ids reused
    assert first["bid"].client_id == second["bid"].client_id
    assert first["ask"].client_id == second["ask"].client_id
    assert len(engine.open_orders()) == 2


async def test_sync_quote_replaces_on_price_change(paper_router):
    engine, router = paper_router
    first = await router.sync_quote(_quote(bid=0.48, ask=0.52))
    second = await router.sync_quote(_quote(bid=0.47, ask=0.53))
    # client_ids differ after replace; exactly 2 resting orders survive
    assert first["bid"].client_id != second["bid"].client_id
    assert first["ask"].client_id != second["ask"].client_id
    opens = engine.open_orders()
    assert len(opens) == 2
    prices = sorted(o.price for o in opens)
    assert prices == [0.47, 0.53]


async def test_sync_quote_cancels_side_when_size_zero(paper_router):
    engine, router = paper_router
    await router.sync_quote(_quote(bid=0.48, ask=0.52))
    await router.sync_quote(_quote(bid=0.48, ask=0.52, size_bid=0, size_ask=10))
    opens = engine.open_orders()
    assert len(opens) == 1
    assert opens[0].side is OrderSide.SELL


async def test_retry_gives_up_after_max_attempts():
    engine = AsyncMock()
    engine.place_order.side_effect = RuntimeError("transient")
    engine.cancel_order = AsyncMock(return_value=True)
    engine.cancel_all = AsyncMock(return_value=0)
    router = OrderRouter(
        paper_backend=engine,
        config=RouterConfig(mode="paper", max_retries=2, retry_base_ms=1.0),
    )
    with pytest.raises(RuntimeError, match="transient"):
        await router.place(token_id="t", side=OrderSide.BUY, price=0.5, size=1)
    assert engine.place_order.await_count == 2


async def test_retry_succeeds_after_transient():
    engine = AsyncMock()
    good = Order(
        token_id="t", side=OrderSide.BUY, price=0.5, size=1,
        client_id="cid", venue_id="paper-1", status=OrderStatus.OPEN,
        created_ts=_now(),
    )
    engine.place_order.side_effect = [RuntimeError("flap"), good]
    router = OrderRouter(
        paper_backend=engine,
        config=RouterConfig(mode="paper", max_retries=3, retry_base_ms=1.0),
    )
    placed = await router.place(
        token_id="t", side=OrderSide.BUY, price=0.5, size=1, client_id="cid",
    )
    assert placed.status is OrderStatus.OPEN
    assert engine.place_order.await_count == 2


def test_live_mode_requires_live_ack():
    live_backend = MagicMock()
    with pytest.raises(RuntimeError, match="live_ack"):
        OrderRouter(live_backend=live_backend, config=RouterConfig(mode="live", live_ack=False))
