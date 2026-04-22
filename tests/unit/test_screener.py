"""Unit tests for ``core/ingest/screener.Screener``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from blksch.core.ingest.polyclient import PolyClient
from blksch.core.ingest.screener import (
    MarketScore,
    ScreenerFilters,
    Screener,
    depth_within_band,
)
from blksch.schemas import BookSnap, PriceLevel


# ---------- Fake client ----------


class FakeClient:
    """In-memory PolyClient stand-in for the screener.

    Exposes ``list_markets`` and ``get_book``; tracks how many times each was
    called so the TTL cache tests can check hit/miss behavior.
    """

    def __init__(
        self,
        markets: list[dict[str, Any]],
        books: dict[str, BookSnap],
    ) -> None:
        self._markets = markets
        self._books = books
        self.list_calls = 0
        self.get_book_calls = 0

    async def list_markets(self, *, page_size: int = 100, max_markets: int | None = None):
        self.list_calls += 1
        if max_markets is None:
            return list(self._markets)
        return list(self._markets[:max_markets])

    async def get_book(self, token_id: str) -> BookSnap:
        self.get_book_calls += 1
        if token_id not in self._books:
            raise KeyError(token_id)
        return self._books[token_id]


def _book(token_id: str, mid: float, *, top_size: float = 1000.0) -> BookSnap:
    """Simple symmetric book with 2 levels each side around ``mid``."""
    return BookSnap(
        token_id=token_id,
        bids=[
            PriceLevel(price=round(mid - 0.01, 4), size=top_size),
            PriceLevel(price=round(mid - 0.02, 4), size=top_size * 2),
        ],
        asks=[
            PriceLevel(price=round(mid + 0.01, 4), size=top_size),
            PriceLevel(price=round(mid + 0.02, 4), size=top_size * 2),
        ],
        ts=datetime.now(UTC),
    )


def _market(market_id: str, yes_tid: str, no_tid: str, volume: float, *, active: bool = True) -> dict:
    return {
        "id": market_id,
        "question": f"Question {market_id}?",
        "clobTokenIds": json.dumps([yes_tid, no_tid]),
        "volume24hr": volume,
        "active": active,
        "closed": False,
    }


# ---------- Filter thresholds ----------


async def test_volume_filter_drops_below_threshold() -> None:
    markets = [
        _market("m1", "y1", "n1", volume=10_000),  # below threshold
        _market("m2", "y2", "n2", volume=100_000),
    ]
    books = {tid: _book(tid, 0.5) for tid in ("y1", "n1", "y2", "n2")}
    client = FakeClient(markets, books)
    screener = Screener(
        client,  # type: ignore[arg-type]
        ScreenerFilters(min_volume_24h_usd=50_000, top_n=10),
    )
    result = await screener.screen()
    tids = set(result.token_ids)
    assert "y2" in tids and "n2" in tids
    assert "y1" not in tids and "n1" not in tids


async def test_volume_filter_keeps_at_exact_threshold() -> None:
    markets = [_market("m1", "y1", "n1", volume=50_000)]
    books = {tid: _book(tid, 0.5) for tid in ("y1", "n1")}
    client = FakeClient(markets, books)
    screener = Screener(
        client,  # type: ignore[arg-type]
        ScreenerFilters(min_volume_24h_usd=50_000, top_n=10),
    )
    result = await screener.screen()
    assert set(result.token_ids) == {"y1", "n1"}


async def test_depth_filter_drops_thin_books() -> None:
    markets = [
        _market("m1", "y1", "n1", volume=100_000),
        _market("m2", "y2", "n2", volume=100_000),
    ]
    # Thin book for m1 (top_size=1), thick for m2 (top_size=1000).
    books = {
        "y1": _book("y1", 0.5, top_size=1),
        "n1": _book("n1", 0.5, top_size=1),
        "y2": _book("y2", 0.5, top_size=1000),
        "n2": _book("n2", 0.5, top_size=1000),
    }
    client = FakeClient(markets, books)
    # Require at least $100 depth on ±5% band.
    screener = Screener(
        client,  # type: ignore[arg-type]
        ScreenerFilters(min_volume_24h_usd=0, min_depth_usd_5pct=100.0, top_n=10),
    )
    result = await screener.screen()
    assert set(result.token_ids) == {"y2", "n2"}


async def test_closed_markets_are_dropped() -> None:
    markets = [_market("m1", "y1", "n1", volume=100_000)]
    markets[0]["closed"] = True
    client = FakeClient(markets, {tid: _book(tid, 0.5) for tid in ("y1", "n1")})
    screener = Screener(client, ScreenerFilters(min_volume_24h_usd=0, top_n=10))  # type: ignore[arg-type]
    result = await screener.screen()
    assert result.token_ids == []


# ---------- Scoring monotonicity ----------


async def test_deeper_book_ranks_higher_when_volume_equal() -> None:
    markets = [
        _market("m1", "y1", "n1", volume=100_000),
        _market("m2", "y2", "n2", volume=100_000),
    ]
    books = {
        "y1": _book("y1", 0.5, top_size=100),
        "n1": _book("n1", 0.5, top_size=100),
        "y2": _book("y2", 0.5, top_size=1000),  # 10x deeper
        "n2": _book("n2", 0.5, top_size=1000),
    }
    client = FakeClient(markets, books)
    screener = Screener(client, ScreenerFilters(top_n=4))  # type: ignore[arg-type]
    result = await screener.screen()
    # Top two should be the deeper market's tokens.
    assert result.token_ids[0] in {"y2", "n2"}
    assert result.token_ids[1] in {"y2", "n2"}


async def test_depth_dominates_volume_by_weight_default() -> None:
    """Modest depth advantage beats a large volume advantage at default weights."""
    markets = [
        # Huge volume, thin book.
        _market("m1", "y1", "n1", volume=1_000_000),
        # 1/10 the volume, but 3x the depth.
        _market("m2", "y2", "n2", volume=100_000),
    ]
    books = {
        "y1": _book("y1", 0.5, top_size=100),
        "n1": _book("n1", 0.5, top_size=100),
        "y2": _book("y2", 0.5, top_size=300),
        "n2": _book("n2", 0.5, top_size=300),
    }
    client = FakeClient(markets, books)
    screener = Screener(client, ScreenerFilters(top_n=4))  # type: ignore[arg-type]
    result = await screener.screen()
    # With depth_weight=0.7, volume_weight=0.3:
    #   m1 score = 0.3*1.0 + 0.7*(1/3) ≈ 0.533
    #   m2 score = 0.3*0.1 + 0.7*1.0 = 0.73
    # so m2 outranks.
    assert result.token_ids[0] in {"y2", "n2"}


# ---------- top_n truncation ----------


async def test_top_n_truncates() -> None:
    markets = [
        _market(f"m{i}", f"y{i}", f"n{i}", volume=100_000 * (i + 1))
        for i in range(5)
    ]
    books = {}
    for i in range(5):
        books[f"y{i}"] = _book(f"y{i}", 0.5, top_size=1000 * (i + 1))
        books[f"n{i}"] = _book(f"n{i}", 0.5, top_size=1000 * (i + 1))
    client = FakeClient(markets, books)
    screener = Screener(client, ScreenerFilters(top_n=3))  # type: ignore[arg-type]
    result = await screener.screen()
    assert len(result.token_ids) == 3


async def test_top_n_requires_positive() -> None:
    with pytest.raises(ValueError):
        ScreenerFilters(top_n=0)


# ---------- TTL cache ----------


async def test_cache_hit_skips_client() -> None:
    clock_t = [0.0]

    def clk() -> float:
        return clock_t[0]

    markets = [_market("m1", "y1", "n1", volume=100_000)]
    books = {tid: _book(tid, 0.5) for tid in ("y1", "n1")}
    client = FakeClient(markets, books)
    screener = Screener(
        client,  # type: ignore[arg-type]
        ScreenerFilters(top_n=2, ttl_sec=60),
        clock=clk,
    )
    r1 = await screener.screen()
    assert client.list_calls == 1
    # Well within TTL.
    clock_t[0] = 30.0
    r2 = await screener.screen()
    assert client.list_calls == 1
    assert r1 is r2


async def test_cache_miss_after_ttl_expiry() -> None:
    clock_t = [0.0]

    def clk() -> float:
        return clock_t[0]

    markets = [_market("m1", "y1", "n1", volume=100_000)]
    books = {tid: _book(tid, 0.5) for tid in ("y1", "n1")}
    client = FakeClient(markets, books)
    screener = Screener(
        client,  # type: ignore[arg-type]
        ScreenerFilters(top_n=2, ttl_sec=60),
        clock=clk,
    )
    await screener.screen()
    clock_t[0] = 60.5  # past TTL
    await screener.screen()
    assert client.list_calls == 2


async def test_force_bypasses_cache() -> None:
    markets = [_market("m1", "y1", "n1", volume=100_000)]
    books = {tid: _book(tid, 0.5) for tid in ("y1", "n1")}
    client = FakeClient(markets, books)
    screener = Screener(
        client,  # type: ignore[arg-type]
        ScreenerFilters(top_n=2, ttl_sec=3600),
    )
    await screener.screen()
    await screener.screen(force=True)
    assert client.list_calls == 2


# ---------- Correlation pair resolution ----------


async def test_pair_resolved_when_both_legs_exist() -> None:
    markets = [
        _market("m1", "y1", "n1", volume=100_000),
        _market("m2", "y2", "n2", volume=100_000),
    ]
    books = {tid: _book(tid, 0.5) for tid in ("y1", "n1", "y2", "n2")}
    client = FakeClient(markets, books)
    screener = Screener(
        client,  # type: ignore[arg-type]
        ScreenerFilters(top_n=10),
        pair_hints=[("y1", "y2")],
    )
    result = await screener.screen()
    assert result.correlation_pairs == [("y1", "y2")]


async def test_pair_dropped_when_one_leg_missing() -> None:
    markets = [_market("m1", "y1", "n1", volume=100_000)]
    books = {tid: _book(tid, 0.5) for tid in ("y1", "n1")}
    client = FakeClient(markets, books)
    screener = Screener(
        client,  # type: ignore[arg-type]
        ScreenerFilters(top_n=10),
        pair_hints=[("y1", "y_does_not_exist")],
    )
    result = await screener.screen()
    assert result.correlation_pairs == []


async def test_pair_dropped_when_leg_filtered_out_by_volume() -> None:
    markets = [
        _market("m1", "y1", "n1", volume=100_000),
        _market("m2", "y2", "n2", volume=10_000),  # dropped by volume filter
    ]
    books = {tid: _book(tid, 0.5) for tid in ("y1", "n1", "y2", "n2")}
    client = FakeClient(markets, books)
    screener = Screener(
        client,  # type: ignore[arg-type]
        ScreenerFilters(min_volume_24h_usd=50_000, top_n=10),
        pair_hints=[("y1", "y2")],
    )
    result = await screener.screen()
    assert result.correlation_pairs == []


# ---------- Depth helper ----------


def test_depth_within_band_sums_both_sides() -> None:
    snap = BookSnap(
        token_id="tok",
        bids=[
            PriceLevel(price=0.49, size=100),
            PriceLevel(price=0.40, size=1000),  # outside 5% band
        ],
        asks=[
            PriceLevel(price=0.51, size=100),
            PriceLevel(price=0.60, size=1000),  # outside 5% band
        ],
        ts=datetime.now(UTC),
    )
    # mid=0.50; 5% band = [0.475, 0.525]; only first levels count.
    expected = 0.49 * 100 + 0.51 * 100
    assert depth_within_band(snap, band=0.05) == pytest.approx(expected)


def test_depth_returns_zero_on_empty_book() -> None:
    snap = BookSnap(token_id="t", bids=[], asks=[], ts=datetime.now(UTC))
    assert depth_within_band(snap) == 0.0


# ---------- Polyclient.list_markets ----------


class _PagedFakeSession:
    """Minimal stand-in to verify list_markets pages to exhaustion."""

    def __init__(self, pages: list[list[dict]]) -> None:
        self._pages = pages
        self.requests: list[dict] = []
        self.closed = False

    def get(self, url: str, params: dict) -> "_PagedFakeSession":
        self.requests.append(params)
        idx = params["offset"] // params["limit"]
        self._current = self._pages[idx] if idx < len(self._pages) else []
        return self

    async def __aenter__(self) -> "_PagedFakeSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> list[dict]:
        return self._current


async def test_list_markets_pages_until_short_page() -> None:
    pages = [
        [{"id": f"m{i}"} for i in range(100)],  # full page
        [{"id": "m100"}, {"id": "m101"}],  # short page → stop
    ]
    client = PolyClient()
    fake = _PagedFakeSession(pages)
    client._session = fake  # type: ignore[assignment]
    try:
        out = await client.list_markets(page_size=100)
    finally:
        client._session = None  # type: ignore[assignment]
    assert len(out) == 102
    assert [r["offset"] for r in fake.requests] == [0, 100]


async def test_list_markets_respects_max_markets() -> None:
    pages = [
        [{"id": f"m{i}"} for i in range(100)],
        [{"id": f"m{100 + i}"} for i in range(100)],
        [{"id": "m200"}],
    ]
    client = PolyClient()
    fake = _PagedFakeSession(pages)
    client._session = fake  # type: ignore[assignment]
    try:
        out = await client.list_markets(page_size=100, max_markets=150)
    finally:
        client._session = None  # type: ignore[assignment]
    assert len(out) == 150
