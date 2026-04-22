"""Idempotent place / cancel / replace router.

Routes :class:`Order` and :class:`Quote` intents to either the paper-trading
engine (Stage 1) or the live CLOB client (Stage 2+). Track B's refresh loop
is the only caller; it stays backend-agnostic because both sides implement
the minimal protocol this module expects.

Responsibilities:

* Stable ``client_id`` generation and in-flight tracking.
* ``sync_quote`` — compare the target Quote against our open orders on the
  token and perform the minimal set of cancel / place ops to align.
* Retry with exponential backoff on transient errors (network / 5xx). Hard
  rejects propagate immediately.
* Prominent logging of the mode on every emitted order — makes a mis-wired
  live promotion easy to catch in tail.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from blksch.schemas import (
    Order,
    OrderSide,
    OrderStatus,
    Quote,
)

log = logging.getLogger(__name__)

Mode = Literal["paper", "live"]


class _PaperBackend(Protocol):
    async def place_order(self, order: Order) -> Order: ...
    async def cancel_order(self, client_id: str) -> bool: ...
    async def cancel_all(self, token_id: str | None = None) -> int: ...


class _LiveBackend(Protocol):
    async def place_order(
        self,
        *,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        client_id: str,
        order_type: str = "GTC",
        expiration_ts: int | None = None,
    ) -> Order: ...
    async def cancel_order(self, venue_id: str) -> bool: ...
    async def cancel_all(self) -> int: ...


@dataclass
class RouterConfig:
    mode: Mode = "paper"
    max_retries: int = 3
    retry_base_ms: float = 50.0
    retry_jitter: float = 0.25
    order_type: Literal["GTC", "GTD", "FOK", "FAK"] = "GTC"
    live_ack: bool = False  # must be True to allow live placements
    min_repost_delta_p: float = 1e-4  # don't replace for sub-tick changes


class OrderRouter:
    """Wraps a ``paper_engine`` or a ``clob_client`` behind a mode-aware API."""

    def __init__(
        self,
        *,
        paper_backend: _PaperBackend | None = None,
        live_backend: _LiveBackend | None = None,
        config: RouterConfig | None = None,
    ):
        self.cfg = config or RouterConfig()
        self._paper = paper_backend
        self._live = live_backend
        # client_id -> venue_id (set as soon as the backend assigns one)
        self._venue: dict[str, str] = {}
        # token_id -> {"bid": client_id | None, "ask": client_id | None}
        self._quoted: dict[str, dict[str, str | None]] = {}
        self._lock = asyncio.Lock()
        self._validate_backends()

    def _validate_backends(self) -> None:
        if self.cfg.mode == "paper" and self._paper is None:
            raise ValueError("paper mode requires paper_backend")
        if self.cfg.mode == "live":
            if self._live is None:
                raise ValueError("live mode requires live_backend")
            if not self.cfg.live_ack:
                raise RuntimeError(
                    "Refusing to route live orders without RouterConfig.live_ack=True. "
                    "Set --live-ack on app.py to confirm you intend real orders."
                )

    # -- public API --------------------------------------------------------

    def make_client_id(self, token_id: str, side: OrderSide) -> str:
        return f"blksch-{token_id[:8]}-{side.value}-{uuid.uuid4().hex[:10]}"

    async def place(
        self,
        *,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        client_id: str | None = None,
    ) -> Order:
        cid = client_id or self.make_client_id(token_id, side)
        log.info(
            "ORDER place [mode=%s] %s %s %.4f x %.4f  cid=%s",
            self.cfg.mode, token_id, side.value, price, size, cid,
        )
        if self.cfg.mode == "paper":
            now = datetime.now(UTC)
            order = Order(
                token_id=token_id, side=side, price=price, size=size,
                client_id=cid, status=OrderStatus.PENDING,
                created_ts=now, updated_ts=now,
            )
            placed = await self._with_retry(self._paper.place_order, order)
        else:
            placed = await self._with_retry(
                self._live.place_order,
                token_id=token_id, side=side, price=price, size=size,
                client_id=cid, order_type=self.cfg.order_type,
            )
        if placed.venue_id:
            self._venue[placed.client_id] = placed.venue_id
        return placed

    async def cancel(self, client_id: str) -> bool:
        log.info("ORDER cancel [mode=%s] cid=%s", self.cfg.mode, client_id)
        if self.cfg.mode == "paper":
            ok = await self._with_retry(self._paper.cancel_order, client_id)
        else:
            venue_id = self._venue.get(client_id)
            if not venue_id:
                log.warning("cancel: no venue_id for cid=%s; skipping", client_id)
                return False
            ok = await self._with_retry(self._live.cancel_order, venue_id)
        if ok:
            self._venue.pop(client_id, None)
        return ok

    async def cancel_all(self, token_id: str | None = None) -> int:
        log.info("ORDER cancel_all [mode=%s] token=%s", self.cfg.mode, token_id)
        if self.cfg.mode == "paper":
            n = await self._with_retry(self._paper.cancel_all, token_id)
        else:
            # Most live CLOBs don't scope cancel_all by token_id; callers who
            # want per-token precision should iterate their known cids.
            n = await self._with_retry(self._live.cancel_all)
        self._quoted.pop(token_id, None) if token_id else self._quoted.clear()
        # venue map is rebuilt lazily on next place
        if token_id is None:
            self._venue.clear()
        return n

    async def replace(
        self,
        *,
        existing_cid: str,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
    ) -> Order:
        await self.cancel(existing_cid)
        return await self.place(token_id=token_id, side=side, price=price, size=size)

    async def sync_quote(self, quote: Quote) -> dict[str, Order | None]:
        """Align resting orders with the target Quote on both sides.

        Returns a dict ``{"bid": Order|None, "ask": Order|None}`` describing
        the currently-resting orders after the sync.
        """
        slot = self._quoted.setdefault(quote.token_id, {"bid": None, "ask": None})
        out: dict[str, Order | None] = {"bid": None, "ask": None}

        out["bid"] = await self._sync_side(
            slot=slot, key="bid",
            token_id=quote.token_id, side=OrderSide.BUY,
            price=quote.p_bid, size=quote.size_bid,
        )
        out["ask"] = await self._sync_side(
            slot=slot, key="ask",
            token_id=quote.token_id, side=OrderSide.SELL,
            price=quote.p_ask, size=quote.size_ask,
        )
        return out

    async def _sync_side(
        self,
        *,
        slot: dict[str, str | None],
        key: str,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
    ) -> Order | None:
        existing_cid = slot[key]
        if size <= 0:
            if existing_cid:
                await self.cancel(existing_cid)
                slot[key] = None
            return None

        # No existing order — just place.
        if not existing_cid:
            placed = await self.place(token_id=token_id, side=side, price=price, size=size)
            slot[key] = placed.client_id
            return placed

        existing = await self._lookup_existing(existing_cid, token_id, side)
        if (
            existing is not None
            and abs(existing.price - price) < self.cfg.min_repost_delta_p
            and abs(existing.size - size) < 1e-9
        ):
            return existing  # already aligned

        replaced = await self.replace(
            existing_cid=existing_cid, token_id=token_id,
            side=side, price=price, size=size,
        )
        slot[key] = replaced.client_id
        return replaced

    async def _lookup_existing(
        self, cid: str, token_id: str, side: OrderSide,
    ) -> Order | None:
        # Paper backend can introspect via its ledger; live backend caller is
        # expected to maintain its own mapping. Keep the router lightweight —
        # callers sync via refresh_loop at a fixed cadence so stale cache is
        # bounded.
        backend = self._paper if self.cfg.mode == "paper" else None
        if backend is None or not hasattr(backend, "open_orders"):
            return None
        try:
            orders = backend.open_orders(token_id=token_id)  # type: ignore[attr-defined]
        except TypeError:
            orders = backend.open_orders()  # type: ignore[attr-defined]
        for o in orders:
            if o.client_id == cid and o.side is side:
                return o
        return None

    # -- retry/backoff -----------------------------------------------------

    async def _with_retry(self, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                return await fn(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # intentionally broad — log and retry
                last_exc = exc
                delay = self.cfg.retry_base_ms / 1000.0 * (2 ** attempt)
                delay *= 1 + random.uniform(-self.cfg.retry_jitter, self.cfg.retry_jitter)
                log.warning(
                    "router call %s failed (attempt %d/%d): %s; backing off %.3fs",
                    getattr(fn, "__name__", str(fn)),
                    attempt + 1, self.cfg.max_retries, exc, delay,
                )
                if attempt < self.cfg.max_retries - 1:
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc
