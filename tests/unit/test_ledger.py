"""Unit tests for the SQLite ledger: position arithmetic, PnL reconciliation,
and order-status bookkeeping."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from blksch.exec.ledger import Ledger, _apply_fill_to_position, reconcile
from blksch.schemas import Fill, Order, OrderSide, OrderStatus


def _now() -> datetime:
    return datetime.now(UTC)


def _fill(side: OrderSide, price: float, size: float, *, fee: float = 0.0,
          token: str = "tok", ts: datetime | None = None, cid: str = "c1") -> Fill:
    return Fill(
        order_client_id=cid, order_venue_id=None, token_id=token,
        side=side, price=price, size=size, fee_usd=fee, ts=ts or _now(),
    )


# ---------------------------------------------------------------------------
# pure accounting
# ---------------------------------------------------------------------------


def test_apply_fill_opens_long():
    new_qty, new_avg, pnl = _apply_fill_to_position(0.0, 0.0, +10.0, 0.5)
    assert new_qty == 10.0
    assert new_avg == 0.5
    assert pnl == 0.0


def test_apply_fill_adds_long_wap():
    new_qty, new_avg, pnl = _apply_fill_to_position(10.0, 0.4, +10.0, 0.6)
    assert new_qty == 20.0
    assert abs(new_avg - 0.5) < 1e-12
    assert pnl == 0.0


def test_apply_fill_partial_close_realizes_pnl():
    # long 10 @ 0.4, then sell 4 @ 0.7 -> realized = 4 * (0.7 - 0.4) = 1.20
    new_qty, new_avg, pnl = _apply_fill_to_position(10.0, 0.4, -4.0, 0.7)
    assert new_qty == 6.0
    assert abs(new_avg - 0.4) < 1e-12  # remainder basis unchanged
    assert abs(pnl - 1.20) < 1e-12


def test_apply_fill_flip_long_to_short():
    # long 5 @ 0.4, then sell 8 @ 0.6 -> close 5 @ 0.6 (pnl = 5*0.2 = 1.0),
    # remainder short 3 at basis 0.6.
    new_qty, new_avg, pnl = _apply_fill_to_position(5.0, 0.4, -8.0, 0.6)
    assert new_qty == -3.0
    assert new_avg == 0.6
    assert abs(pnl - 1.0) < 1e-12


def test_apply_fill_short_accounting_symmetric():
    # short 10 @ 0.6 (i.e. sold expecting price to fall), then buy 4 @ 0.4
    # -> realized = 4 * (0.4 - 0.6) * (-1) = +0.8
    new_qty, new_avg, pnl = _apply_fill_to_position(-10.0, 0.6, +4.0, 0.4)
    assert new_qty == -6.0
    assert abs(new_avg - 0.6) < 1e-12
    assert abs(pnl - 0.8) < 1e-12


# ---------------------------------------------------------------------------
# ledger integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def ledger() -> Ledger:
    return Ledger.in_memory()


def test_ledger_records_order_and_updates_status(ledger: Ledger):
    now = _now()
    order = Order(
        token_id="t", side=OrderSide.BUY, price=0.5, size=10,
        client_id="c1", status=OrderStatus.PENDING, created_ts=now,
    )
    ledger.record_order(order)
    ledger.update_order_status("c1", OrderStatus.OPEN, venue_id="v-1")
    got = ledger.get_order("c1")
    assert got is not None
    assert got.venue_id == "v-1"
    assert got.status is OrderStatus.OPEN


def test_ledger_pnl_matches_reconcile(ledger: Ledger):
    fills = [
        _fill(OrderSide.BUY,  0.40, 10, fee=0.01),
        _fill(OrderSide.BUY,  0.50, 10, fee=0.01),  # WAP = 0.45, qty=20
        _fill(OrderSide.SELL, 0.60, 5,  fee=0.01),  # realizes 5*(0.60-0.45)=0.75; qty=15
        _fill(OrderSide.SELL, 0.55, 25, fee=0.01),  # close 15 @ (0.55-0.45) = 1.5;
                                                    # then flip short: qty=-10, basis=0.55
    ]
    for f in fills:
        ledger.apply_fill(f)
    ledger.update_mark("tok", 0.50)  # short 10 @ 0.55 -> unrealized = (-10) * (0.50 - 0.55) = +0.5
    snap = ledger.pnl()
    recon = reconcile(fills, mark=0.50)
    assert abs(snap.realized_usd - recon.realized_usd) < 1e-9
    assert abs(snap.unrealized_usd - recon.unrealized_usd) < 1e-9
    assert abs(snap.fees_usd - 0.04) < 1e-9
    # hand calc: realized = 0.75 + 1.5 - 4*0.01 = 2.21
    assert abs(snap.realized_usd - 2.21) < 1e-9
    assert abs(snap.unrealized_usd - 0.5) < 1e-9


def test_ledger_open_orders_filters_by_status(ledger: Ledger):
    base = Order(
        token_id="t", side=OrderSide.BUY, price=0.5, size=10,
        client_id="c1", status=OrderStatus.OPEN, created_ts=_now(),
    )
    ledger.record_order(base)
    ledger.record_order(base.model_copy(update={
        "client_id": "c2", "status": OrderStatus.FILLED,
    }))
    ledger.record_order(base.model_copy(update={
        "client_id": "c3", "status": OrderStatus.CANCELED,
    }))
    opens = ledger.open_orders("t")
    cids = {o.client_id for o in opens}
    assert cids == {"c1"}
