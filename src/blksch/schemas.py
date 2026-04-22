"""Shared Pydantic contracts between the three tracks.

These models are the integration glue for the bot. Track A (core/) produces
BookSnap, TradeTick, LogitState, SurfacePoint, CorrelationEntry. Track B (mm/)
produces Quote and HedgeInstruction. Track C (exec/) produces Order, Fill,
Position.

FREEZE: do not modify a field signature without coordinating across all three
tracks. Adding a new model or a new optional field is usually safe; changing or
removing is not.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Market data (Track A -> B, C)
# ---------------------------------------------------------------------------


class PriceLevel(BaseModel):
    """One side of an L2 book level."""

    model_config = ConfigDict(frozen=True)

    price: float = Field(..., ge=0.0, le=1.0)
    size: float = Field(..., ge=0.0)


class BookSnap(BaseModel):
    """L2 book snapshot for a single Polymarket token."""

    token_id: str
    bids: list[PriceLevel]
    asks: list[PriceLevel]
    ts: datetime
    seq: int | None = None

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0

    @property
    def spread(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeTick(BaseModel):
    token_id: str
    price: float = Field(..., ge=0.0, le=1.0)
    size: float = Field(..., ge=0.0)
    aggressor_side: TradeSide
    ts: datetime


# ---------------------------------------------------------------------------
# Calibration outputs (Track A -> B)
# ---------------------------------------------------------------------------


class LogitState(BaseModel):
    """Filtered latent logit x_hat for a single token with its measurement noise."""

    token_id: str
    x_hat: float
    sigma_eta2: float = Field(..., ge=0.0)
    ts: datetime


class SurfacePoint(BaseModel):
    """Point estimate on the belief-volatility surface (paper §5.3)."""

    token_id: str
    tau: float = Field(..., ge=0.0, description="time-to-resolution in seconds")
    m: float = Field(..., description="moneyness coordinate (paper uses m=x)")
    sigma_b: float = Field(..., ge=0.0, description="belief volatility")
    lambda_: float = Field(..., ge=0.0, alias="lambda", description="jump intensity")
    s2_j: float = Field(..., ge=0.0, description="jump second moment")
    uncertainty: float | None = None
    ts: datetime


class CorrelationEntry(BaseModel):
    """De-jumped correlation + co-jump moments for an event pair (paper §5.4)."""

    token_id_i: str
    token_id_j: str
    rho: float = Field(..., ge=-1.0, le=1.0)
    co_jump_lambda: float = Field(..., ge=0.0)
    co_jump_m2: float = Field(..., ge=0.0)
    ts: datetime


# ---------------------------------------------------------------------------
# Quoting (Track B -> C)
# ---------------------------------------------------------------------------


class Quote(BaseModel):
    """Target bid/ask emitted by Track B's refresh loop.

    Expressed in both logit (x) and probability (p) units so downstream logging
    and the order router can use whichever is convenient.
    """

    token_id: str
    p_bid: float = Field(..., ge=0.0, le=1.0)
    p_ask: float = Field(..., ge=0.0, le=1.0)
    x_bid: float
    x_ask: float
    size_bid: float = Field(..., ge=0.0)
    size_ask: float = Field(..., ge=0.0)
    half_spread_x: float = Field(..., ge=0.0)
    reservation_x: float
    inventory_q: float
    ts: datetime


class HedgeSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class HedgeInstruction(BaseModel):
    """Cross-event or calendar hedge instruction (stages 2-3)."""

    source_token_id: str
    hedge_token_id: str
    side: HedgeSide
    notional_usd: float
    reason: Literal["beta", "calendar", "synth_strip"]
    ts: datetime


# ---------------------------------------------------------------------------
# Execution (Track C)
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class Order(BaseModel):
    token_id: str
    side: OrderSide
    price: float = Field(..., ge=0.0, le=1.0)
    size: float = Field(..., gt=0.0)
    client_id: str
    venue_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    created_ts: datetime
    updated_ts: datetime | None = None


class Fill(BaseModel):
    order_client_id: str
    order_venue_id: str | None
    token_id: str
    side: OrderSide
    price: float = Field(..., ge=0.0, le=1.0)
    size: float = Field(..., gt=0.0)
    fee_usd: float = 0.0
    ts: datetime


class Position(BaseModel):
    token_id: str
    qty: float  # signed; positive = long YES shares
    avg_entry: float = Field(..., ge=0.0, le=1.0)
    mark: float = Field(..., ge=0.0, le=1.0)
    realized_pnl_usd: float = 0.0

    @property
    def unrealized_pnl_usd(self) -> float:
        return self.qty * (self.mark - self.avg_entry)
