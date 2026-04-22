"""Stage-1 paper-trading matching engine.

Deliberately conservative: we only simulate a fill when the **market trades
through** our resting price. The model:

* ``BUY`` @ ``p_bid`` fills when a :class:`TradeTick` with
  ``aggressor_side == SELL`` and ``price <= p_bid`` arrives, or a BookSnap
  shows the best ask at or below ``p_bid`` (trade-through). Size filled is
  ``min(our_remaining, crossing_size * (1 - queue_haircut))``.
* ``SELL`` @ ``p_ask`` fills symmetrically on a BUY aggressor or a best bid
  at/above ``p_ask``.

Queue-position haircut is configurable; default ``0.5`` means we assume half
the resting size at our level is ahead of us. We never model marketable
(crossing) orders — Track B cannot cross the book in paper mode.

Additional responsibilities:

* Track feed cadence. If no book/trade is seen for ``feed_gap_sec`` the
  engine enters a ``halted`` state, stops simulating fills, and exposes the
  flag for Track B's kill-switches.
* Write every simulated :class:`Fill` through a ``Ledger`` so the paper and
  live data paths share the same books and dashboard.

The engine is asyncio-friendly but has no actual network I/O — it is driven
by callers feeding :class:`BookSnap` / :class:`TradeTick` messages.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable

from blksch.schemas import (
    BookSnap,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    TradeSide,
    TradeTick,
)

from .ledger import Ledger

log = logging.getLogger(__name__)


@dataclass
class PaperEngineConfig:
    queue_haircut: float = 0.5
    fee_bps: float = 0.0  # Polymarket maker is currently 0 bps; tune via config
    feed_gap_sec: float = 3.0
    allow_self_crossing: bool = False  # if bid >= ask from same maker, we reject


@dataclass
class _RestingOrder:
    order: Order
    remaining: float

    @property
    def key(self) -> str:
        return self.order.client_id


@dataclass
class PaperEngineState:
    halted: bool = False
    halt_reason: str | None = None
    last_book_ts: datetime | None = None
    last_trade_ts: datetime | None = None
    fills_count: int = 0


class PaperEngine:
    """Single-process paper-trading engine."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        config: PaperEngineConfig | None = None,
        on_fill: Callable[[Fill], None] | None = None,
    ):
        self.ledger = ledger
        self.cfg = config or PaperEngineConfig()
        self._on_fill = on_fill
        self._resting: dict[str, _RestingOrder] = {}
        self._last_book_by_token: dict[str, BookSnap] = {}
        self.state = PaperEngineState()
        self.fills_queue: asyncio.Queue[Fill] = asyncio.Queue()
        self._lock = asyncio.Lock()

    # -- order lifecycle ---------------------------------------------------

    async def place_order(self, order: Order) -> Order:
        """Record a resting order. Returns the order with an assigned
        ``venue_id`` (paper-mode mock) and ``status=OPEN``."""

        if self.state.halted:
            log.warning("place_order rejected (halted): %s", self.cfg)
            rejected = order.model_copy(update={"status": OrderStatus.REJECTED,
                                                 "updated_ts": datetime.now(UTC)})
            self.ledger.record_order(rejected)
            return rejected

        venue_id = order.venue_id or f"paper-{uuid.uuid4().hex[:12]}"
        opened = order.model_copy(update={
            "venue_id": venue_id,
            "status": OrderStatus.OPEN,
            "updated_ts": datetime.now(UTC),
        })

        async with self._lock:
            if self._would_cross_self(opened):
                rejected = opened.model_copy(update={"status": OrderStatus.REJECTED})
                self.ledger.record_order(rejected)
                return rejected
            self._resting[opened.client_id] = _RestingOrder(order=opened, remaining=opened.size)
            self.ledger.record_order(opened)
        return opened

    async def cancel_order(self, client_id: str) -> bool:
        async with self._lock:
            resting = self._resting.pop(client_id, None)
        if not resting:
            return False
        self.ledger.update_order_status(client_id, OrderStatus.CANCELED)
        return True

    async def cancel_all(self, token_id: str | None = None) -> int:
        async with self._lock:
            keys = [
                k for k, r in self._resting.items()
                if token_id is None or r.order.token_id == token_id
            ]
            for k in keys:
                self._resting.pop(k, None)
        for k in keys:
            self.ledger.update_order_status(k, OrderStatus.CANCELED)
        return len(keys)

    def open_orders(self, token_id: str | None = None) -> list[Order]:
        return [
            r.order for r in self._resting.values()
            if token_id is None or r.order.token_id == token_id
        ]

    # -- market-data ingress ----------------------------------------------

    async def on_book(self, snap: BookSnap) -> list[Fill]:
        self._gap_check(snap.ts)
        self.state.last_book_ts = snap.ts
        self._last_book_by_token[snap.token_id] = snap
        self.ledger.update_mark(snap.token_id, snap.mid or 0.5, ts=snap.ts)
        if self.state.halted:
            return []

        async with self._lock:
            resting = [r for r in self._resting.values() if r.order.token_id == snap.token_id]

        fills: list[Fill] = []
        for r in resting:
            fill = self._match_book(r, snap)
            if fill is not None:
                fills.append(await self._commit_fill(r, fill))
        return fills

    async def on_trade(self, tick: TradeTick) -> list[Fill]:
        self._gap_check(tick.ts)
        self.state.last_trade_ts = tick.ts
        if self.state.halted:
            return []

        async with self._lock:
            resting = [r for r in self._resting.values() if r.order.token_id == tick.token_id]

        fills: list[Fill] = []
        for r in resting:
            fill = self._match_trade(r, tick)
            if fill is not None:
                fills.append(await self._commit_fill(r, fill))
        return fills

    # -- matching core -----------------------------------------------------

    def _match_book(self, resting: _RestingOrder, snap: BookSnap) -> Fill | None:
        """Detect trade-through against the current snap and return a fill
        the size of ``remaining`` haircut by queue position."""
        order = resting.order
        if order.side is OrderSide.BUY:
            if not snap.asks:
                return None
            best_ask = snap.asks[0]
            if best_ask.price > order.price:
                return None
            crossing_size = best_ask.size
            fill_price = order.price  # passive maker: we got our posted price
        else:
            if not snap.bids:
                return None
            best_bid = snap.bids[0]
            if best_bid.price < order.price:
                return None
            crossing_size = best_bid.size
            fill_price = order.price

        available = max(0.0, crossing_size * (1.0 - self.cfg.queue_haircut))
        if available <= 0:
            return None
        size = min(resting.remaining, available)
        if size <= 0:
            return None

        return Fill(
            order_client_id=order.client_id,
            order_venue_id=order.venue_id,
            token_id=order.token_id,
            side=order.side,
            price=fill_price,
            size=size,
            fee_usd=self._fee(order, fill_price, size),
            ts=snap.ts,
        )

    def _match_trade(self, resting: _RestingOrder, tick: TradeTick) -> Fill | None:
        order = resting.order
        if order.side is OrderSide.BUY:
            if tick.aggressor_side is not TradeSide.SELL:
                return None
            if tick.price > order.price:
                return None
        else:
            if tick.aggressor_side is not TradeSide.BUY:
                return None
            if tick.price < order.price:
                return None

        available = max(0.0, tick.size * (1.0 - self.cfg.queue_haircut))
        size = min(resting.remaining, available)
        if size <= 0:
            return None
        return Fill(
            order_client_id=order.client_id,
            order_venue_id=order.venue_id,
            token_id=order.token_id,
            side=order.side,
            price=order.price,
            size=size,
            fee_usd=self._fee(order, order.price, size),
            ts=tick.ts,
        )

    async def _commit_fill(self, resting: _RestingOrder, fill: Fill) -> Fill:
        self.ledger.apply_fill(fill)
        resting.remaining -= fill.size
        async with self._lock:
            if resting.remaining <= 1e-9:
                self._resting.pop(resting.key, None)
                self.ledger.update_order_status(
                    resting.order.client_id, OrderStatus.FILLED, updated_ts=fill.ts,
                )
            else:
                self.ledger.update_order_status(
                    resting.order.client_id, OrderStatus.PARTIALLY_FILLED, updated_ts=fill.ts,
                )
        self.state.fills_count += 1
        if self._on_fill is not None:
            try:
                self._on_fill(fill)
            except Exception:
                log.exception("on_fill callback raised")
        await self.fills_queue.put(fill)
        return fill

    # -- kill-switch helpers ----------------------------------------------

    def _gap_check(self, now: datetime) -> None:
        ref = self.state.last_book_ts or self.state.last_trade_ts
        if ref is None:
            return
        gap = (now - ref).total_seconds()
        if gap > self.cfg.feed_gap_sec and not self.state.halted:
            self.halt(f"feed_gap {gap:.1f}s > {self.cfg.feed_gap_sec}s")

    def halt(self, reason: str) -> None:
        if self.state.halted:
            return
        self.state.halted = True
        self.state.halt_reason = reason
        log.warning("PaperEngine HALTED: %s", reason)

    def resume(self) -> None:
        self.state.halted = False
        self.state.halt_reason = None

    # -- misc --------------------------------------------------------------

    def _fee(self, order: Order, price: float, size: float) -> float:
        # Polymarket currently charges zero maker fees on the CLOB; keep the
        # config knob so Stage 2 promotion can plug in real numbers.
        if self.cfg.fee_bps <= 0:
            return 0.0
        # Fee on the USDC notional of this fill.
        notional = price * size if order.side is OrderSide.BUY else (1 - price) * size
        return notional * (self.cfg.fee_bps / 10_000.0)

    def _would_cross_self(self, new_order: Order) -> bool:
        if self.cfg.allow_self_crossing:
            return False
        for r in self._resting.values():
            if r.order.token_id != new_order.token_id:
                continue
            if r.order.side is new_order.side:
                continue
            if new_order.side is OrderSide.BUY and new_order.price >= r.order.price:
                return True
            if new_order.side is OrderSide.SELL and new_order.price <= r.order.price:
                return True
        return False
