"""Day-0 sanity: shared contracts import and roundtrip cleanly."""

from __future__ import annotations

from datetime import UTC, datetime

from blksch.schemas import (
    BookSnap,
    CorrelationEntry,
    Fill,
    HedgeInstruction,
    HedgeSide,
    LogitState,
    Order,
    OrderSide,
    Position,
    PriceLevel,
    Quote,
    SurfacePoint,
    TradeSide,
    TradeTick,
)


def _now() -> datetime:
    return datetime.now(UTC)


def test_book_snap_mid_and_spread() -> None:
    snap = BookSnap(
        token_id="tok",
        bids=[PriceLevel(price=0.48, size=100)],
        asks=[PriceLevel(price=0.52, size=100)],
        ts=_now(),
    )
    assert snap.mid == 0.5
    assert snap.spread == 0.04


def test_trade_tick_roundtrip() -> None:
    t = TradeTick(
        token_id="tok",
        price=0.5,
        size=10,
        aggressor_side=TradeSide.BUY,
        ts=_now(),
    )
    assert t.model_dump()["aggressor_side"] == "buy"


def test_calibration_models_construct() -> None:
    LogitState(token_id="tok", x_hat=0.0, sigma_eta2=0.01, ts=_now())
    SurfacePoint(
        token_id="tok",
        tau=60.0,
        m=0.0,
        sigma_b=0.5,
        **{"lambda": 0.1},
        s2_j=0.04,
        ts=_now(),
    )
    CorrelationEntry(
        token_id_i="a",
        token_id_j="b",
        rho=0.3,
        co_jump_lambda=0.01,
        co_jump_m2=0.02,
        ts=_now(),
    )


def test_quote_and_hedge_construct() -> None:
    Quote(
        token_id="tok",
        p_bid=0.49,
        p_ask=0.51,
        x_bid=-0.04,
        x_ask=0.04,
        size_bid=100,
        size_ask=100,
        half_spread_x=0.04,
        reservation_x=0.0,
        inventory_q=0.0,
        ts=_now(),
    )
    HedgeInstruction(
        source_token_id="a",
        hedge_token_id="b",
        side=HedgeSide.SHORT,
        notional_usd=25.0,
        reason="beta",
        ts=_now(),
    )


def test_position_unrealized_pnl() -> None:
    pos = Position(token_id="tok", qty=100, avg_entry=0.5, mark=0.55, realized_pnl_usd=0.0)
    assert abs(pos.unrealized_pnl_usd - 5.0) < 1e-9


def test_order_and_fill_construct() -> None:
    Order(
        token_id="tok",
        side=OrderSide.BUY,
        price=0.49,
        size=10,
        client_id="c1",
        created_ts=_now(),
    )
    Fill(
        order_client_id="c1",
        order_venue_id="v1",
        token_id="tok",
        side=OrderSide.BUY,
        price=0.49,
        size=10,
        fee_usd=0.01,
        ts=_now(),
    )
