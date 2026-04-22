"""Polymarket CLOB adapter.

Two paths are supported so we can ship Stage 1 (paper) without any credentials
and still promote to live in Stage 2 without rewriting callers:

1. Preferred — `py-clob-client` (PyPI). Signed order placement / cancellation
   goes through its battle-tested EIP-712 flow. The library is synchronous, so
   we dispatch calls via ``asyncio.to_thread`` to keep Track B's refresh loop
   non-blocking.

2. Fallback — a thin aiohttp + ``blksch.exec.signer`` implementation. Used only
   if ``py-clob-client`` is unavailable or mis-signs against an upstream change.

Read-only endpoints (book, market metadata, midpoint, tick size) are reachable
without credentials and are the entry point used by Stage 0/1.

All public methods are async and return plain dicts or ``BookSnap`` / ``Order``
/ ``Fill`` Pydantic models from ``blksch.schemas`` — never raw py-clob-client
objects — so callers stay coupled only to our own schema.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import aiohttp

from blksch.schemas import BookSnap, Order, OrderSide, OrderStatus, PriceLevel

log = logging.getLogger(__name__)

POLY_CLOB_BASE = "https://clob.polymarket.com"
POLY_CHAIN_ID_MAINNET = 137  # Polygon
POLY_CHAIN_ID_AMOY = 80002  # Polygon testnet


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CLOBConfig:
    """Connection config. ``private_key`` and api creds are required only for
    signed endpoints (order placement, cancellation). Read-only usage can pass
    None for all credentials."""

    base_url: str = POLY_CLOB_BASE
    chain_id: int = POLY_CHAIN_ID_MAINNET
    private_key: str | None = None
    funder: str | None = None  # Polymarket funder (usually the proxy wallet)
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    signature_type: int = 1  # 1 = Polymarket proxy wallet; 0 = EOA

    @classmethod
    def from_env(cls, *, testnet: bool = False) -> CLOBConfig:
        return cls(
            base_url=POLY_CLOB_BASE,
            chain_id=POLY_CHAIN_ID_AMOY if testnet else POLY_CHAIN_ID_MAINNET,
            private_key=os.environ.get("POLY_PRIVATE_KEY"),
            funder=os.environ.get("POLY_FUNDER_ADDRESS"),
            api_key=os.environ.get("POLY_API_KEY"),
            api_secret=os.environ.get("POLY_API_SECRET"),
            api_passphrase=os.environ.get("POLY_API_PASSPHRASE"),
        )

    def has_signing_creds(self) -> bool:
        return bool(self.private_key and self.funder)

    def has_l2_creds(self) -> bool:
        return bool(self.api_key and self.api_secret and self.api_passphrase)


# ---------------------------------------------------------------------------
# Protocol — order_router depends on this, not the concrete class
# ---------------------------------------------------------------------------


class CLOBClientProtocol(Protocol):
    async def get_book(self, token_id: str) -> BookSnap: ...

    async def get_midpoint(self, token_id: str) -> float | None: ...

    async def get_tick_size(self, token_id: str) -> float: ...

    async def place_order(
        self,
        *,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        client_id: str,
        order_type: Literal["GTC", "GTD", "FOK", "FAK"] = "GTC",
        expiration_ts: int | None = None,
    ) -> Order: ...

    async def cancel_order(self, venue_id: str) -> bool: ...

    async def cancel_all(self) -> int: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Preferred path — py-clob-client wrapped in asyncio.to_thread
# ---------------------------------------------------------------------------


class PyCLOBClient:
    """Adapter over ``py-clob-client``.

    Constructed in one of three credential modes, auto-detected from config:

    * **read-only** — no private key, no api creds. Exposes book/market reads.
    * **signing** — private key present, L2 creds absent. Derives L2 on first
      auth'd call via ``create_or_derive_api_creds``.
    * **full** — private key + L2 creds. Used as-is.
    """

    def __init__(self, cfg: CLOBConfig):
        from py_clob_client.client import ClobClient  # imported lazily

        self.cfg = cfg
        self._raw = ClobClient(
            host=cfg.base_url,
            key=cfg.private_key,
            chain_id=cfg.chain_id,
            funder=cfg.funder,
            signature_type=cfg.signature_type if cfg.private_key else None,
        )
        if cfg.has_signing_creds():
            if cfg.has_l2_creds():
                from py_clob_client.clob_types import ApiCreds

                self._raw.set_api_creds(
                    ApiCreds(
                        api_key=cfg.api_key,
                        api_secret=cfg.api_secret,
                        api_passphrase=cfg.api_passphrase,
                    )
                )
            else:
                log.info("No L2 API creds in config; will derive on first signed call")

    def _ensure_l2(self) -> None:
        if not self.cfg.has_signing_creds():
            raise RuntimeError("signed endpoints require POLY_PRIVATE_KEY + POLY_FUNDER_ADDRESS")
        if not self.cfg.has_l2_creds():
            creds = self._raw.create_or_derive_api_creds()
            self._raw.set_api_creds(creds)

    async def get_book(self, token_id: str) -> BookSnap:
        raw = await asyncio.to_thread(self._raw.get_order_book, token_id)
        # py-clob-client returns an OrderBookSummary with .bids / .asks lists
        # of objects with .price / .size attributes (both strings).
        def _level(lv: Any) -> PriceLevel:
            return PriceLevel(price=float(lv.price), size=float(lv.size))

        bids = sorted([_level(b) for b in (raw.bids or [])], key=lambda l: -l.price)
        asks = sorted([_level(a) for a in (raw.asks or [])], key=lambda l: l.price)
        return BookSnap(
            token_id=token_id,
            bids=bids,
            asks=asks,
            ts=datetime.now(UTC),
        )

    async def get_midpoint(self, token_id: str) -> float | None:
        raw = await asyncio.to_thread(self._raw.get_midpoint, token_id)
        if isinstance(raw, dict):
            mid = raw.get("mid")
            return float(mid) if mid is not None else None
        return float(raw) if raw is not None else None

    async def get_tick_size(self, token_id: str) -> float:
        raw = await asyncio.to_thread(self._raw.get_tick_size, token_id)
        return float(raw) if raw is not None else 0.01

    async def place_order(
        self,
        *,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        client_id: str,
        order_type: Literal["GTC", "GTD", "FOK", "FAK"] = "GTC",
        expiration_ts: int | None = None,
    ) -> Order:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        self._ensure_l2()

        args = OrderArgs(
            token_id=token_id,
            price=float(price),
            size=float(size),
            side=BUY if side is OrderSide.BUY else SELL,
            expiration=expiration_ts or 0,
        )
        ot = getattr(OrderType, order_type)
        now = datetime.now(UTC)
        log.warning(
            "LIVE order submission: %s %s %.4f x %.4f (%s) client_id=%s",
            token_id, side.value, price, size, order_type, client_id,
        )
        resp = await asyncio.to_thread(
            self._raw.create_and_post_order, args, ot
        )
        venue_id = None
        status = OrderStatus.PENDING
        if isinstance(resp, dict):
            venue_id = resp.get("orderID") or resp.get("orderId")
            if resp.get("success"):
                status = OrderStatus.OPEN
            elif resp.get("errorMsg"):
                status = OrderStatus.REJECTED
                log.error("CLOB rejected order %s: %s", client_id, resp.get("errorMsg"))
        return Order(
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            client_id=client_id,
            venue_id=venue_id,
            status=status,
            created_ts=now,
            updated_ts=now,
        )

    async def cancel_order(self, venue_id: str) -> bool:
        self._ensure_l2()
        resp = await asyncio.to_thread(self._raw.cancel, venue_id)
        if isinstance(resp, dict):
            # response shape: {"canceled": [...], "not_canceled": {...}}
            return venue_id in (resp.get("canceled") or [])
        return bool(resp)

    async def cancel_all(self) -> int:
        self._ensure_l2()
        resp = await asyncio.to_thread(self._raw.cancel_all)
        if isinstance(resp, dict):
            return len(resp.get("canceled") or [])
        return 0

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fallback path — thin aiohttp + signer.py client
# ---------------------------------------------------------------------------


class HttpCLOBClient:
    """Minimal aiohttp CLOB client.

    Implements only the endpoints Track C needs. Used when py-clob-client is
    unavailable or mis-signs. Signing delegates to :mod:`blksch.exec.signer`.
    """

    def __init__(self, cfg: CLOBConfig, *, session: aiohttp.ClientSession | None = None):
        self.cfg = cfg
        self._owns_session = session is None
        self._session = session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
        )

    async def _get(self, path: str, **params: Any) -> Any:
        url = f"{self.cfg.base_url}{path}"
        async with self._session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_book(self, token_id: str) -> BookSnap:
        raw = await self._get("/book", token_id=token_id)
        bids = sorted(
            [PriceLevel(price=float(b["price"]), size=float(b["size"])) for b in raw.get("bids", [])],
            key=lambda l: -l.price,
        )
        asks = sorted(
            [PriceLevel(price=float(a["price"]), size=float(a["size"])) for a in raw.get("asks", [])],
            key=lambda l: l.price,
        )
        return BookSnap(
            token_id=token_id,
            bids=bids,
            asks=asks,
            ts=datetime.now(UTC),
        )

    async def get_midpoint(self, token_id: str) -> float | None:
        raw = await self._get("/midpoint", token_id=token_id)
        mid = raw.get("mid") if isinstance(raw, dict) else raw
        return float(mid) if mid is not None else None

    async def get_tick_size(self, token_id: str) -> float:
        try:
            raw = await self._get("/tick-size", token_id=token_id)
        except aiohttp.ClientResponseError:
            return 0.01
        val = raw.get("minimum_tick_size") if isinstance(raw, dict) else raw
        return float(val) if val is not None else 0.01

    async def place_order(self, **kwargs: Any) -> Order:
        raise NotImplementedError(
            "HttpCLOBClient is read-only. Install py-clob-client or implement "
            "signed POST /order via blksch.exec.signer."
        )

    async def cancel_order(self, venue_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def cancel_all(self) -> int:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:
        if self._owns_session:
            await self._session.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_clob_client(cfg: CLOBConfig | None = None, *, prefer: str = "auto") -> CLOBClientProtocol:
    """Return a CLOB adapter.

    ``prefer``:
        * ``"auto"`` — use py-clob-client if importable, else fallback.
        * ``"py"`` — force py-clob-client (raises if missing).
        * ``"http"`` — force the aiohttp fallback.
    """
    cfg = cfg or CLOBConfig.from_env()
    if prefer == "http":
        return HttpCLOBClient(cfg)
    try:
        import py_clob_client  # noqa: F401
    except ImportError:
        if prefer == "py":
            raise
        log.info("py-clob-client not installed; using HttpCLOBClient (read-only)")
        return HttpCLOBClient(cfg)
    return PyCLOBClient(cfg)
