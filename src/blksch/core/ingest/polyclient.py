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
import math
import os
import time
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime, timedelta
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

# --- Mock-mode routing ----------------------------------------------------
#
# Any token_id matching :data:`MOCK_TOKEN_PREFIXES` is served by
# :class:`MockPolyClient` instead of Polymarket's live endpoints. This is
# the convention that lets ``paper_soak.py --token-id 0xmock`` and
# similar rehearsals run the full pipeline without network I/O.
#
# The mock stream is driven by a seeded RNG (env var
# ``MOCK_POLYCLIENT_SEED``, default 42) so test runs are byte-reproducible.

MOCK_TOKEN_PREFIXES: tuple[str, ...] = ("0xmock", "mock:")
MOCK_SEED_ENV = "MOCK_POLYCLIENT_SEED"
MOCK_DEFAULT_SEED = 42
MOCK_DEFAULT_INTERVAL_S = 0.25
MOCK_DEFAULT_SIGMA_B = 0.02
MOCK_DEFAULT_TRADE_PROB = 0.1
MOCK_DEFAULT_HALF_SPREAD_P = 0.005
MOCK_DEFAULT_DEPTH = 100.0


def is_mock_token(token_id: str) -> bool:
    """True if ``token_id`` should route through :class:`MockPolyClient`.

    The convention is any id starting with ``0xmock`` or ``mock:`` (case
    insensitive). Example: ``0xmock``, ``0xmockFOO``, ``mock:btc-70k``.
    """
    t = token_id.lower()
    return any(t.startswith(p) for p in MOCK_TOKEN_PREFIXES)


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
        if is_mock_token(token_id):
            return _mock_initial_book(token_id)
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

    async def list_markets(
        self,
        *,
        page_size: int = 100,
        max_markets: int | None = None,
    ) -> list[dict[str, Any]]:
        """Page through the Gamma markets listing until exhausted.

        Stops when a page comes back short (fewer than ``page_size`` rows) or
        when ``max_markets`` is reached. Returns the accumulated list.
        """
        out: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = await self.get_markets(limit=page_size, offset=offset)
            if not page:
                break
            out.extend(page)
            if len(page) < page_size:
                break
            if max_markets is not None and len(out) >= max_markets:
                return out[:max_markets]
            offset += page_size
        return out

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

        Mock routing: if every ``token_id`` satisfies :func:`is_mock_token`,
        the call is transparently served by :func:`_mock_stream_market` and
        no network is touched. Mixing mock and real tokens is a
        ``ValueError`` (multiplexing a live WS with a synthetic stream
        would give false quote-uptime readings).
        """
        ids = list(token_ids)
        if not ids:
            raise ValueError("token_ids must be non-empty")
        mock_flags = [is_mock_token(t) for t in ids]
        if all(mock_flags):
            async for event in _mock_stream_market(ids):
                yield event
            return
        if any(mock_flags):
            real = [t for t, m in zip(ids, mock_flags) if not m]
            mock = [t for t, m in zip(ids, mock_flags) if m]
            raise ValueError(
                "Cannot mix mock and real token_ids in a single stream_market "
                f"call. Mock: {mock!r}; real: {real!r}."
            )
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


# ---------------------------------------------------------------------------
# Mock client — network-free rehearsal / test stream
# ---------------------------------------------------------------------------


def _mock_seed_from_env(override: int | None) -> int:
    """Resolve the mock seed: explicit override wins, else
    ``MOCK_POLYCLIENT_SEED`` env var, else :data:`MOCK_DEFAULT_SEED`."""
    if override is not None:
        return int(override)
    raw = os.environ.get(MOCK_SEED_ENV)
    if raw is None:
        return MOCK_DEFAULT_SEED
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; falling back to default seed %d",
            MOCK_SEED_ENV, raw, MOCK_DEFAULT_SEED,
        )
        return MOCK_DEFAULT_SEED


def _snap_from_logit_mid(
    token_id: str, x_mid: float, *, half_spread_p: float, depth: float, ts: datetime,
) -> BookSnap:
    """Build a :class:`BookSnap` from a logit-space mid.

    Clips ``p`` into ``[half_spread_p, 1 - half_spread_p]`` so that both
    bid and ask stay in the schema's ``[0, 1]`` band after the spread is
    taken around the mid.
    """
    p = 1.0 / (1.0 + math.exp(-x_mid))
    # Leave enough room on both sides of the mid for the half-spread.
    p = max(half_spread_p + 1e-6, min(1.0 - half_spread_p - 1e-6, p))
    return BookSnap(
        token_id=token_id,
        bids=[PriceLevel(price=p - half_spread_p, size=depth)],
        asks=[PriceLevel(price=p + half_spread_p, size=depth)],
        ts=ts,
    )


def _mock_initial_book(token_id: str) -> BookSnap:
    """Return a fresh at-the-money mock book (for ``get_book`` warmup)."""
    return _snap_from_logit_mid(
        token_id, 0.0,
        half_spread_p=MOCK_DEFAULT_HALF_SPREAD_P,
        depth=MOCK_DEFAULT_DEPTH,
        ts=datetime.now(UTC),
    )


async def _mock_stream_market(
    token_ids: Iterable[str],
    *,
    seed: int | None = None,
    interval_s: float = MOCK_DEFAULT_INTERVAL_S,
    sigma_b: float = MOCK_DEFAULT_SIGMA_B,
    trade_probability: float = MOCK_DEFAULT_TRADE_PROB,
    half_spread_p: float = MOCK_DEFAULT_HALF_SPREAD_P,
    depth: float = MOCK_DEFAULT_DEPTH,
    start_ts: datetime | None = None,
) -> AsyncIterator[BookSnap | TradeTick]:
    """Deterministic network-free stream mirroring the shape of
    :meth:`PolyClient.stream_market`.

    Each ``token_id`` gets its own drifting logit mid (initial 0.0, i.e.
    p=0.5) advanced by ``N(0, sigma_b² · interval_s)`` per tick. A book
    snapshot is yielded every ``interval_s`` seconds per token; a trade
    tick is yielded with probability ``trade_probability`` per tick.

    Timestamps advance deterministically from ``start_ts`` by
    ``interval_s`` per full round through ``token_ids``, so identical
    (seed, start_ts) pairs produce byte-identical event sequences. If
    ``start_ts`` is ``None`` the stream anchors at ``datetime.now(UTC)``
    (good for live rehearsals, not deterministic).

    The generator never exits on its own; the caller terminates it by
    breaking out of the ``async for`` or calling ``aclose``.
    """
    ids = list(token_ids)
    if not ids:
        raise ValueError("token_ids must be non-empty")
    import random as _random  # local — keeps mock isolated from real path

    rng = _random.Random(_mock_seed_from_env(seed))
    step_std = sigma_b * math.sqrt(interval_s)
    x: dict[str, float] = {tid: 0.0 for tid in ids}
    anchor = start_ts if start_ts is not None else datetime.now(UTC)
    tick = 0
    try:
        while True:
            ts = anchor + timedelta(seconds=interval_s * tick)
            for tid in ids:
                x[tid] += rng.gauss(0.0, step_std)
                yield _snap_from_logit_mid(
                    tid, x[tid],
                    half_spread_p=half_spread_p, depth=depth, ts=ts,
                )
                if rng.random() < trade_probability:
                    p = 1.0 / (1.0 + math.exp(-x[tid]))
                    p = max(0.01, min(0.99, p))
                    side = TradeSide.BUY if rng.random() < 0.5 else TradeSide.SELL
                    yield TradeTick(
                        token_id=tid,
                        price=p,
                        size=float(rng.uniform(1.0, 10.0)),
                        aggressor_side=side,
                        ts=ts,
                    )
            tick += 1
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        raise


class MockPolyClient(PolyClient):
    """Network-free :class:`PolyClient` for rehearsals and tests.

    Inherits the real client's interface (``start``, ``close``,
    ``get_book``, ``stream_market``, ``list_markets``) but skips every
    aiohttp / websockets call. Use this when you want to GUARANTEE no
    network is touched — e.g. in unit tests that might otherwise receive
    a real ``token_id`` by accident.

    For ``paper_soak.py --token-id 0xmock`` rehearsals the real
    :class:`PolyClient` already routes mock tokens to
    :func:`_mock_stream_market` transparently; wiring a
    :class:`MockPolyClient` explicitly is not required at the app layer.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        interval_s: float = MOCK_DEFAULT_INTERVAL_S,
        sigma_b: float = MOCK_DEFAULT_SIGMA_B,
        trade_probability: float = MOCK_DEFAULT_TRADE_PROB,
        start_ts: datetime | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mock_seed = _mock_seed_from_env(seed)
        self._mock_interval_s = interval_s
        self._mock_sigma_b = sigma_b
        self._mock_trade_probability = trade_probability
        self._mock_start_ts = start_ts

    async def start(self) -> None:  # no aiohttp session needed
        return None

    async def close(self) -> None:
        return None

    async def get_book(self, token_id: str) -> BookSnap:
        return _mock_initial_book(token_id)

    async def get_markets(
        self, *, limit: int = 100, offset: int = 0,
    ) -> list[dict[str, Any]]:
        return []

    async def list_markets(
        self, *, page_size: int = 100, max_markets: int | None = None,
    ) -> list[dict[str, Any]]:
        return []

    async def get_clob_market(self, condition_id: str) -> dict[str, Any]:
        return {"condition_id": condition_id, "tokens": [], "active": True}

    async def stream_market(
        self,
        token_ids: Iterable[str],
        *,
        maintain_book_state: bool = True,  # unused (mock has no diffs)
        reconnect: bool = True,             # unused (mock can't disconnect)
        reconnect_max_delay_s: float = 30.0,
    ) -> AsyncIterator[BookSnap | TradeTick]:
        async for ev in _mock_stream_market(
            list(token_ids),
            seed=self._mock_seed,
            interval_s=self._mock_interval_s,
            sigma_b=self._mock_sigma_b,
            trade_probability=self._mock_trade_probability,
            start_ts=self._mock_start_ts,
        ):
            yield ev
