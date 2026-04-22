"""Dynamic top-liquidity market screener.

Picks the top-N Polymarket tokens to quote on, scored by a composite of
normalized 24 h volume and normalized ±5% book depth. Liquidity (depth) is
weighted more heavily than notional turnover — a market maker cares where a
size can actually rest, not where headline volume printed.

Inputs come from two places:
  * Gamma markets metadata via :meth:`PolyClient.list_markets` — volume, the
    ``clobTokenIds`` pair, active/closed flags, resolution timestamps.
  * CLOB order books via :meth:`PolyClient.get_book` — depth within ±5% of
    mid. Fan-out is rate-limited by the client.

Config-driven filters (see ``config/markets.yaml``):
  * ``min_volume_24h_usd``
  * ``min_depth_usd_5pct``
  * ``top_n``
  * optional ``min_spread_bps`` / ``max_spread_bps``
  * optional ``min_hours_to_resolution`` / ``max_days_to_resolution``

Correlation pair hints are resolved by token_id membership in the scanned
universe — pairs with one missing leg are dropped.

Results are cached for ``rescreen_every_sec`` (default 300 s) so we do not
hammer Gamma on every refresh.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from blksch.core.ingest.polyclient import PolyClient
from blksch.schemas import BookSnap

logger = logging.getLogger(__name__)

DEFAULT_TTL_SEC = 300.0
DEFAULT_DEPTH_BAND = 0.05
DEFAULT_VOLUME_WEIGHT = 0.3
DEFAULT_DEPTH_WEIGHT = 0.7


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreenerFilters:
    """Hard filters + scoring weights + TTL, typically loaded from YAML."""

    min_volume_24h_usd: float = 0.0
    min_depth_usd_5pct: float = 0.0
    top_n: int = 5
    min_spread_bps: float | None = None
    max_spread_bps: float | None = None
    min_hours_to_resolution: float | None = None
    max_days_to_resolution: float | None = None
    volume_weight: float = DEFAULT_VOLUME_WEIGHT
    depth_weight: float = DEFAULT_DEPTH_WEIGHT
    depth_band: float = DEFAULT_DEPTH_BAND
    ttl_sec: float = DEFAULT_TTL_SEC

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.depth_band <= 0 or self.depth_band >= 1:
            raise ValueError("depth_band must be in (0, 1)")
        if self.ttl_sec <= 0:
            raise ValueError("ttl_sec must be positive")
        if self.depth_weight < self.volume_weight:
            logger.warning(
                "depth_weight (%.2f) is lower than volume_weight (%.2f) — "
                "screener intends depth to dominate.",
                self.depth_weight,
                self.volume_weight,
            )


@dataclass(frozen=True)
class MarketScore:
    token_id: str
    market_id: str | None
    question: str | None
    volume_24h_usd: float
    depth_usd_5pct: float
    score: float


@dataclass
class ScreenResult:
    token_ids: list[str]
    correlation_pairs: list[tuple[str, str]]
    details: list[MarketScore]
    scored_at: datetime
    universe_size: int


# ---------------------------------------------------------------------------
# Helpers (pure)
# ---------------------------------------------------------------------------


def _extract_token_ids(market: dict[str, Any]) -> list[str]:
    """Polymarket Gamma markets expose ``clobTokenIds`` as a JSON string.

    Returns the YES/NO token_ids; an empty list if the market is missing or
    malformed.
    """
    raw = market.get("clobTokenIds") or market.get("clob_token_ids") or market.get("tokens")
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        out: list[str] = []
        for entry in raw:
            if isinstance(entry, str) and entry:
                out.append(entry)
            elif isinstance(entry, dict):
                tid = entry.get("token_id") or entry.get("id")
                if isinstance(tid, str) and tid:
                    out.append(tid)
        return out
    return []


def _market_volume(market: dict[str, Any]) -> float:
    for key in ("volume24hr", "volume_24h", "volume24h", "volumeNum"):
        v = market.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def depth_within_band(snap: BookSnap, band: float = DEFAULT_DEPTH_BAND) -> float:
    """USD depth within ±``band`` of mid, summed across bids+asks.

    Returns 0 if the book has no mid.
    """
    mid = snap.mid
    if mid is None:
        return 0.0
    lo, hi = mid * (1.0 - band), mid * (1.0 + band)
    depth = 0.0
    for lv in snap.bids:
        if lv.price >= lo:
            depth += lv.price * lv.size
    for lv in snap.asks:
        if lv.price <= hi:
            depth += lv.price * lv.size
    return depth


def _spread_bps(snap: BookSnap) -> float | None:
    mid = snap.mid
    spread = snap.spread
    if mid is None or spread is None or mid <= 0:
        return None
    return (spread / mid) * 10_000.0


def _hours_to_resolution(market: dict[str, Any]) -> float | None:
    for key in ("endDate", "end_date", "endDateIso", "resolutionDate"):
        v = market.get(key)
        if not v:
            continue
        try:
            if isinstance(v, (int, float)):
                end = datetime.fromtimestamp(float(v), tz=UTC)
            else:
                end = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            return max(0.0, (end - datetime.now(UTC)).total_seconds() / 3600.0)
        except (TypeError, ValueError):
            continue
    return None


def _passes_hard_filters(
    market: dict[str, Any],
    filters: ScreenerFilters,
) -> bool:
    if market.get("closed") is True:
        return False
    if market.get("active") is False:
        return False
    if _market_volume(market) < filters.min_volume_24h_usd:
        return False
    hrs = _hours_to_resolution(market)
    if hrs is not None:
        if filters.min_hours_to_resolution is not None and hrs < filters.min_hours_to_resolution:
            return False
        if (
            filters.max_days_to_resolution is not None
            and hrs > filters.max_days_to_resolution * 24.0
        ):
            return False
    return True


def _normalize(values: list[float]) -> list[float]:
    m = max(values) if values else 0.0
    if m <= 0:
        return [0.0 for _ in values]
    return [v / m for v in values]


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    result: ScreenResult
    at_monotonic: float


class Screener:
    """Top-N liquidity screener with a TTL cache."""

    def __init__(
        self,
        client: PolyClient,
        filters: ScreenerFilters,
        pair_hints: Iterable[tuple[str, str]] | None = None,
        *,
        max_markets_scanned: int | None = None,
        clock: object | None = None,
    ) -> None:
        self._client = client
        self._filters = filters
        self._pair_hints: list[tuple[str, str]] = list(pair_hints or [])
        self._max_markets_scanned = max_markets_scanned
        # ``clock`` is a zero-arg callable returning a monotonic float; injected
        # in tests to advance time deterministically.
        self._clock = clock or time.monotonic
        self._cache: _CacheEntry | None = None
        self._lock = asyncio.Lock()

    async def screen(self, *, force: bool = False) -> ScreenResult:
        async with self._lock:
            now_mono = float(self._clock())
            if (
                not force
                and self._cache is not None
                and (now_mono - self._cache.at_monotonic) < self._filters.ttl_sec
            ):
                return self._cache.result
            result = await self._do_screen()
            self._cache = _CacheEntry(result=result, at_monotonic=now_mono)
            return result

    async def _do_screen(self) -> ScreenResult:
        markets = await self._client.list_markets(
            max_markets=self._max_markets_scanned,
        )
        candidates: list[tuple[dict[str, Any], str]] = []
        for m in markets:
            if not _passes_hard_filters(m, self._filters):
                continue
            for tid in _extract_token_ids(m):
                candidates.append((m, tid))

        books = await asyncio.gather(
            *(self._client.get_book(tid) for _, tid in candidates),
            return_exceptions=True,
        )

        scored: list[MarketScore] = []
        raw_volumes: list[float] = []
        raw_depths: list[float] = []
        staging: list[tuple[str, str | None, str | None, float, float]] = []

        for (m, tid), book in zip(candidates, books):
            if isinstance(book, BaseException):
                logger.debug("screener: get_book failed for %s: %s", tid[:16], book)
                continue
            assert isinstance(book, BookSnap)
            depth = depth_within_band(book, self._filters.depth_band)
            if depth < self._filters.min_depth_usd_5pct:
                continue
            bps = _spread_bps(book)
            if bps is not None:
                if self._filters.min_spread_bps is not None and bps < self._filters.min_spread_bps:
                    continue
                if self._filters.max_spread_bps is not None and bps > self._filters.max_spread_bps:
                    continue
            volume = _market_volume(m)
            market_id = m.get("id") or m.get("conditionId") or m.get("condition_id")
            question = m.get("question") or m.get("description")
            staging.append((tid, market_id, question, volume, depth))
            raw_volumes.append(volume)
            raw_depths.append(depth)

        norm_v = _normalize(raw_volumes)
        norm_d = _normalize(raw_depths)
        for (tid, mid, q, vol, depth), nv, nd in zip(staging, norm_v, norm_d):
            score = self._filters.volume_weight * nv + self._filters.depth_weight * nd
            scored.append(
                MarketScore(
                    token_id=tid,
                    market_id=mid,
                    question=q,
                    volume_24h_usd=vol,
                    depth_usd_5pct=depth,
                    score=score,
                )
            )

        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[: self._filters.top_n]

        observed_tids = {s.token_id for s in scored}
        pairs = [
            (i, j)
            for i, j in self._pair_hints
            if i in observed_tids and j in observed_tids and i != j
        ]

        return ScreenResult(
            token_ids=[s.token_id for s in top],
            correlation_pairs=pairs,
            details=top,
            scored_at=datetime.now(UTC),
            universe_size=len(markets),
        )

    # -- test hooks --------------------------------------------------------

    def invalidate_cache(self) -> None:
        self._cache = None


__all__ = [
    "DEFAULT_TTL_SEC",
    "MarketScore",
    "ScreenResult",
    "ScreenerFilters",
    "Screener",
    "depth_within_band",
]
