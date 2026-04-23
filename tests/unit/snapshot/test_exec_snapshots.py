"""Regression snapshots for ``exec/`` modules: paper_engine, order_router,
ledger (including ``reconcile_against_ledger``).

Strategy: fixed-seed deterministic scenarios that exercise the modules'
observable outputs — Fill sequences for paper_engine + ledger, resting
order state for order_router. Times are pinned to ``T0`` so the
``venue_id`` strings produced by paper_engine (which embed a uuid4 hex)
are post-processed to strip their per-run random tail before canonical
serialization — otherwise the snapshots would churn on every run.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pytest

from blksch.exec.ledger import (
    Ledger, reconcile, reconcile_against_ledger,
)
from blksch.exec.order_router import OrderRouter, RouterConfig
from blksch.exec.paper_engine import PaperEngine, PaperEngineConfig
from blksch.schemas import (
    BookSnap, Fill, Order, OrderSide, OrderStatus, PriceLevel, Quote,
    TradeSide, TradeTick,
)

from tests.unit.snapshot._helpers import assert_matches_snapshot

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
TOKEN = "0xSNAPSHOT_TOK"


# ---------------------------------------------------------------------------
# paper_engine
# ---------------------------------------------------------------------------


def _snap(bids, asks, ts):
    return BookSnap(
        token_id=TOKEN,
        bids=[PriceLevel(price=p, size=s) for p, s in bids],
        asks=[PriceLevel(price=p, size=s) for p, s in asks],
        ts=ts,
    )


def _redact_order_dict(d):
    """Strip per-run noise from Order dicts:
    * random uuid4 tail from client_id / venue_id;
    * ``datetime.now()``-derived ``created_ts`` / ``updated_ts`` that
      OrderRouter + PaperEngine stamp internally (these are not
      deterministic across runs — the snapshot would churn).
    Everything else flows through unchanged.
    """
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k == "client_id" and isinstance(v, str):
                out[k] = "<redacted-cid>"
            elif k in ("venue_id", "order_venue_id") and isinstance(v, str):
                out[k] = "paper-<redacted>"
            elif k in ("created_ts", "updated_ts") and isinstance(v, str):
                out[k] = "<redacted-now>"
            else:
                out[k] = _redact_order_dict(v)
        return out
    if isinstance(d, list):
        return [_redact_order_dict(x) for x in d]
    return d


async def _paper_engine_run():
    ledger = Ledger.in_memory()
    engine = PaperEngine(
        ledger,
        config=PaperEngineConfig(queue_haircut=0.5, fee_bps=0.0, feed_gap_sec=5.0),
    )

    # Place a resting BUY + SELL on the paper venue.
    buy = Order(
        token_id=TOKEN, side=OrderSide.BUY,
        price=0.48, size=50,
        client_id="cid-buy",
        status=OrderStatus.PENDING,
        created_ts=T0,
    )
    sell = Order(
        token_id=TOKEN, side=OrderSide.SELL,
        price=0.52, size=50,
        client_id="cid-sell",
        status=OrderStatus.PENDING,
        created_ts=T0,
    )
    await engine.place_order(buy)
    await engine.place_order(sell)

    # Book trade-through on the BUY side (ask at 0.47 with size 40 → fill 20).
    fills_buy = await engine.on_book(_snap(
        [(0.45, 10)], [(0.47, 40)], T0 + timedelta(seconds=1),
    ))
    # Aggressor BUY tick crossing the SELL at 0.52 with size 20 → fill 10.
    fills_sell = await engine.on_trade(TradeTick(
        token_id=TOKEN, price=0.53, size=20,
        aggressor_side=TradeSide.BUY,
        ts=T0 + timedelta(seconds=2),
    ))
    # Feed-gap halt at t=t0+30 (gap 28s > 5s threshold).
    fills_halt = await engine.on_book(_snap(
        [(0.46, 50)], [(0.47, 60)], T0 + timedelta(seconds=30),
    ))

    return {
        "fills_from_book_tradethrough": [
            _redact_order_dict(f.model_dump(mode="json")) for f in fills_buy
        ],
        "fills_from_trade_tick": [
            _redact_order_dict(f.model_dump(mode="json")) for f in fills_sell
        ],
        "fills_after_feed_gap": [
            _redact_order_dict(f.model_dump(mode="json")) for f in fills_halt
        ],
        "state": {
            "halted": engine.state.halted,
            "halt_reason": engine.state.halt_reason,
            "fills_count": engine.state.fills_count,
        },
        "position": _redact_order_dict(
            ledger.get_position(TOKEN).model_dump(mode="json")
        ),
        "pnl": asdict(ledger.pnl()),
    }


def test_paper_engine_canonical_run_snapshot():
    result = asyncio.run(_paper_engine_run())
    assert_matches_snapshot(result, "exec/paper_engine_canonical_run.json")


# ---------------------------------------------------------------------------
# order_router
# ---------------------------------------------------------------------------


def _quote(bid: float, ask: float, size: float = 10.0) -> Quote:
    return Quote(
        token_id=TOKEN,
        p_bid=bid, p_ask=ask,
        x_bid=0.0, x_ask=0.0,  # router doesn't read logit fields
        size_bid=size, size_ask=size,
        half_spread_x=0.02, reservation_x=0.0,
        inventory_q=0.0, ts=T0,
    )


async def _router_run():
    ledger = Ledger.in_memory()
    engine = PaperEngine(ledger)
    router = OrderRouter(
        paper_backend=engine,
        config=RouterConfig(mode="paper"),
    )

    r1 = await router.sync_quote(_quote(0.47, 0.53, size=10))
    # Idempotent re-sync — router should reuse the same client_ids.
    r2 = await router.sync_quote(_quote(0.47, 0.53, size=10))
    # Price change on both sides → cancel + place new.
    r3 = await router.sync_quote(_quote(0.46, 0.54, size=10))
    # Cancel everything on this token.
    cancelled = await router.cancel_all(token_id=TOKEN)

    def _dump(r):
        return {
            "bid": _redact_order_dict(r["bid"].model_dump(mode="json"))
                   if r["bid"] else None,
            "ask": _redact_order_dict(r["ask"].model_dump(mode="json"))
                   if r["ask"] else None,
        }

    r1_bid_cid = r1["bid"].client_id if r1["bid"] else None
    r2_bid_cid = r2["bid"].client_id if r2["bid"] else None
    r3_bid_cid = r3["bid"].client_id if r3["bid"] else None

    return {
        "sync_1_initial":             _dump(r1),
        "sync_2_idempotent":          _dump(r2),
        "sync_2_reused_bid_cid":      r1_bid_cid == r2_bid_cid,
        "sync_3_price_change":        _dump(r3),
        "sync_3_replaced_bid_cid":    r1_bid_cid != r3_bid_cid,
        "cancel_all_count":           cancelled,
        "final_resting_orders_count": len(engine.open_orders(token_id=TOKEN)),
    }


def test_order_router_sync_quote_snapshot():
    result = asyncio.run(_router_run())
    assert_matches_snapshot(result, "exec/order_router_sync_quote.json")


# ---------------------------------------------------------------------------
# ledger (apply_fill + pnl + reconcile_against_ledger)
# ---------------------------------------------------------------------------


def test_ledger_pnl_and_reconcile_snapshot():
    """Apply a canonical fill sequence, record the ledger state + pnl +
    reconcile() vs pure-python replay, then snapshot everything."""
    ledger = Ledger.in_memory()
    fills = [
        Fill(order_client_id="c1", order_venue_id="v1", token_id=TOKEN,
             side=OrderSide.BUY, price=0.40, size=10, fee_usd=0.01,
             ts=T0 + timedelta(seconds=1)),
        Fill(order_client_id="c2", order_venue_id="v2", token_id=TOKEN,
             side=OrderSide.BUY, price=0.50, size=10, fee_usd=0.01,
             ts=T0 + timedelta(seconds=2)),
        Fill(order_client_id="c3", order_venue_id="v3", token_id=TOKEN,
             side=OrderSide.SELL, price=0.60, size=5, fee_usd=0.01,
             ts=T0 + timedelta(seconds=3)),
        Fill(order_client_id="c4", order_venue_id="v4", token_id=TOKEN,
             side=OrderSide.SELL, price=0.55, size=25, fee_usd=0.01,
             ts=T0 + timedelta(seconds=4)),
    ]
    for f in fills:
        ledger.apply_fill(f)
    ledger.update_mark(TOKEN, 0.50, ts=T0 + timedelta(seconds=5))

    pnl = ledger.pnl()
    replayed = reconcile(fills, mark=0.50)
    rec = reconcile_against_ledger(ledger)

    payload = {
        "position": ledger.get_position(TOKEN).model_dump(mode="json"),
        "pnl_stored": asdict(pnl),
        "pnl_replayed": asdict(replayed),
        "reconcile_against_ledger": rec.to_dict(),
        "fills_in_order": [f.model_dump(mode="json") for f in ledger.fills(TOKEN)],
    }
    assert_matches_snapshot(payload, "exec/ledger_pnl_reconcile.json")
