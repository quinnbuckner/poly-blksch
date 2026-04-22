"""Async Polymarket client: Gamma + CLOB REST and CLOB WebSocket subscriber.

Owns the Day-0 data plane for Track A: fetches L2 books and market metadata via
REST, and streams live book snapshots + trade ticks from the CLOB WebSocket
``market`` channel. Emits :class:`BookSnap` and :class:`TradeTick` as defined
in ``blksch.schemas``.

The rate-limiter and session-pooling pattern is ported from
``polyarb_v1.0/src/api.py`` and adapted to asyncio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from typing import Any

import aiohttp
import websockets

from blksch.schemas import BookSnap, PriceLevel, TradeSide, TradeTick

logger = logging.getLogger(__name__)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_BASE_URL = "https://clob.polymarket.com"
CLOB_BOOK_URL = f"{CLOB_BASE_URL}/book"
CLOB_MARKETS_URL = f"{CLOB_BASE_URL}/markets"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

DEFAULT_RATE_PER_SEC = 12.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
DEFAULT_POOL_SIZE = 10
DEFAULT_WS_PING_INTERVAL_S = 30.0


class AsyncRateLimiter:
    """Asyncio token-bucket rate limiter (single-interval form).

    Serializes ``acquire()`` calls so that consecutive requests are spaced by at
    least ``1 / rate_per_sec`` seconds. The original ``polyarb_v1.0/src/api.py``
    uses a threading.Lock + time.sleep; this variant is the asyncio-safe
    equivalent.
    """

    def __init__(self, rate_per_sec: float = DEFAULT_RATE_PER_SEC) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self._min_interval = 1.0 / rate_per_sec
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Parsers (pure functions — easy to unit test)
# ---------------------------------------------------------------------------


def _parse_level(raw: dict[str, Any]) -> PriceLevel:
    return PriceLevel(price=float(raw["price"]), size=float(raw["size"]))


def _parse_ts_ms(ts: Any) -> datetime:
    """Polymarket timestamps are sometimes milliseconds, sometimes seconds."""
    try:
        v = float(ts)
    except (TypeError, ValueError):
        return datetime.now(UTC)
    # Heuristic: anything >= 1e12 is ms, else seconds.
    if v >= 1e12:
        v /= 1000.0
    return datetime.fromtimestamp(v, tz=UTC)


def book_from_rest(payload: dict[str, Any], token_id: str) -> BookSnap:
    """Build a :class:`BookSnap` from the CLOB REST /book response.

    Bids are expected best-first; asks best-first. We sort defensively so that
    ``bids[0]`` is the highest bid and ``asks[0]`` is the lowest ask.
    """
    bids_raw = payload.get("bids") or []
    asks_raw = payload.get("asks") or []
    bids = sorted((_parse_level(b) for b in bids_raw), key=lambda lv: -lv.price)
    asks = sorted((_parse_level(a) for a in asks_raw), key=lambda lv: lv.price)
    ts_raw = payload.get("timestamp") or payload.get("ts")
    ts = _parse_ts_ms(ts_raw) if ts_raw is not None else datetime.now(UTC)
    return BookSnap(token_id=token_id, bids=bids, asks=asks, ts=ts)


def book_from_ws(msg: dict[str, Any]) -> BookSnap:
    """Build a :class:`BookSnap` from a WS 'book' event.

    WS book events carry asset_id and the full levels; shape matches REST.
    """
    token_id = str(msg.get("asset_id") or msg.get("token_id") or "")
    bids = sorted(
        (_parse_level(b) for b in (msg.get("bids") or [])),
        key=lambda lv: -lv.price,
    )
    asks = sorted(
        (_parse_level(a) for a in (msg.get("asks") or [])),
        key=lambda lv: lv.price,
    )
    ts = _parse_ts_ms(msg.get("timestamp"))
    return BookSnap(token_id=token_id, bids=bids, asks=asks, ts=ts)


def trade_from_ws(msg: dict[str, Any]) -> TradeTick:
    """Build a :class:`TradeTick` from a WS 'last_trade_price' event."""
    token_id = str(msg.get("asset_id") or msg.get("token_id") or "")
    side_raw = str(msg.get("side", "buy")).lower()
    side = TradeSide.BUY if side_raw.startswith("b") else TradeSide.SELL
    return TradeTick(
        token_id=token_id,
        price=float(msg["price"]),
        size=float(msg["size"]),
        aggressor_side=side,
        ts=_parse_ts_ms(msg.get("timestamp")),
    )


def apply_price_change(prev: BookSnap, msg: dict[str, Any]) -> BookSnap:
    """Apply a WS 'price_change' diff to a previous book, returning a new snap.

    Changes are a list of ``{price, side, size}`` entries; size=0 removes a
    level. Side is 'BUY'/'SELL' or 'bid'/'ask' depending on build — we accept
    either.
    """
    bids = {lv.price: lv.size for lv in prev.bids}
    asks = {lv.price: lv.size for lv in prev.asks}
    for ch in msg.get("changes", []) or []:
        price = float(ch["price"])
        size = float(ch["size"])
        side = str(ch.get("side", "")).lower()
        book = bids if side.startswith("b") else asks
        if size == 0:
            book.pop(price, None)
        else:
            book[price] = size
    new_bids = [PriceLevel(price=p, size=s) for p, s in sorted(bids.items(), reverse=True) if s > 0]
    new_asks = [PriceLevel(price=p, size=s) for p, s in sorted(asks.items()) if s > 0]
    return BookSnap(
        token_id=prev.token_id,
        bids=new_bids,
        asks=new_asks,
        ts=_parse_ts_ms(msg.get("timestamp")) if msg.get("timestamp") else prev.ts,
        seq=prev.seq,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PolyClient:
    """Polymarket async REST + WS client.

    Usage::

        async with PolyClient() as client:
            snap = await client.get_book(token_id)
            async for msg in client.stream_market([token_id_a, token_id_b]):
                ...  # BookSnap or TradeTick
    """

    def __init__(
        self,
        *,
        rate_per_sec: float = DEFAULT_RATE_PER_SEC,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        pool_size: int = DEFAULT_POOL_SIZE,
        gamma_url: str = GAMMA_MARKETS_URL,
        clob_book_url: str = CLOB_BOOK_URL,
        clob_markets_url: str = CLOB_MARKETS_URL,
        ws_url: str = CLOB_WS_URL,
    ) -> None:
        self._rate_per_sec = rate_per_sec
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        self._pool_size = pool_size
        self._gamma_url = gamma_url
        self._clob_book_url = clob_book_url
        self._clob_markets_url = clob_markets_url
        self._ws_url = ws_url
        self._session: aiohttp.ClientSession | None = None
        self._limiter = AsyncRateLimiter(rate_per_sec)

    async def __aenter__(self) -> PolyClient:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=self._pool_size)
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                connector=connector,
                headers={"Accept": "application/json"},
            )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            raise RuntimeError("PolyClient not started; use `async with` or call start().")
        return self._session

    # -- REST --------------------------------------------------------------

    async def get_book(self, token_id: str) -> BookSnap:
        await self._limiter.acquire()
        async with self.session.get(
            self._clob_book_url, params={"token_id": token_id}
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return book_from_rest(payload, token_id)

    async def get_markets(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Fetch a page of the Gamma markets listing."""
        await self._limiter.acquire()
        async with self.session.get(
            self._gamma_url, params={"limit": limit, "offset": offset}
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        if isinstance(data, list):
            return data
        return data.get("data") or data.get("markets") or []

    async def get_clob_market(self, condition_id: str) -> dict[str, Any]:
        await self._limiter.acquire()
        async with self.session.get(
            f"{self._clob_markets_url}/{condition_id}"
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # -- WebSocket ---------------------------------------------------------

    async def stream_market(
        self,
        token_ids: Iterable[str],
        *,
        maintain_book_state: bool = True,
        reconnect: bool = True,
        reconnect_max_delay_s: float = 30.0,
    ) -> AsyncIterator[BookSnap | TradeTick]:
        """Subscribe to the CLOB ``market`` channel and yield events.

        If ``maintain_book_state`` is True (default), ``price_change`` events
        are applied to the last ``book`` snapshot and re-emitted as updated
        :class:`BookSnap`. Otherwise price_change events are dropped.

        If ``reconnect`` is True (default), transient disconnects are retried
        with exponential backoff up to ``reconnect_max_delay_s``.
        """
        ids = list(token_ids)
        if not ids:
            raise ValueError("token_ids must be non-empty")
        books: dict[str, BookSnap] = {}
        backoff = 1.0
        while True:
            try:
                async for event in self._ws_iter(ids, books, maintain_book_state):
                    yield event
                    backoff = 1.0  # reset after each clean message
                return  # generator exited without exception — subscription ended
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if not reconnect:
                    raise
                logger.warning("CLOB WS error: %s; reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, reconnect_max_delay_s)

    async def _ws_iter(
        self,
        token_ids: list[str],
        books: dict[str, BookSnap],
        maintain_book_state: bool,
    ) -> AsyncIterator[BookSnap | TradeTick]:
        async with websockets.connect(
            self._ws_url, ping_interval=DEFAULT_WS_PING_INTERVAL_S
        ) as ws:
            await ws.send(json.dumps({"type": "market", "assets_ids": token_ids}))
            async for raw in ws:
                event = _dispatch_ws_message(raw, books, maintain_book_state)
                if event is not None:
                    yield event


def _dispatch_ws_message(
    raw: str | bytes,
    books: dict[str, BookSnap],
    maintain_book_state: bool,
) -> BookSnap | TradeTick | None:
    """Decode a raw WS frame into BookSnap|TradeTick|None.

    Returns None for events we don't expose (e.g., ``tick_size_change``,
    heartbeats, or malformed payloads).
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("WS: non-JSON frame: %r", raw[:200])
        return None
    # Polymarket sometimes wraps in a list.
    msgs = payload if isinstance(payload, list) else [payload]
    result: BookSnap | TradeTick | None = None
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        etype = msg.get("event_type") or msg.get("type")
        try:
            if etype == "book":
                snap = book_from_ws(msg)
                books[snap.token_id] = snap
                result = snap
            elif etype == "price_change" and maintain_book_state:
                token = str(msg.get("asset_id") or msg.get("token_id") or "")
                prev = books.get(token)
                if prev is None:
                    # Can't apply a diff without a base snapshot; drop.
                    continue
                snap = apply_price_change(prev, msg)
                books[token] = snap
                result = snap
            elif etype in ("last_trade_price", "trade", "last_trade"):
                result = trade_from_ws(msg)
            else:
                continue
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("WS: skipping malformed %s event: %s", etype, exc)
            continue
    return result
