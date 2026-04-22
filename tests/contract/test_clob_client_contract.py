"""Contract tests — replay recorded CLOB responses through Track C's parsers.

The fixtures live under ``tests/contract/fixtures/clob/`` and mirror what the
live Polymarket CLOB returns today (Apr 2026). They are hand-curated rather
than captured from live traffic to avoid committing sensitive headers /
addresses, but are kept field-for-field with the documented schema at
https://docs.polymarket.com/.

These tests are offline and deterministic — mark with ``contract`` so CI can
run them on every commit. The ``live_ro`` and ``canary`` suites are the ones
that actually hit production.

When the upstream schema changes, the fix is to (a) update the fixture, then
(b) update whatever parser it broke. Don't silence an assertion.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from blksch.exec.clob_client import (
    CLOBConfig,
    HttpCLOBClient,
    PyCLOBClient,
)
from blksch.schemas import BookSnap, OrderSide, OrderStatus, TradeSide

pytestmark = pytest.mark.contract


FIXTURES = Path(__file__).parent / "fixtures" / "clob"
TOKEN_ID = (
    "52114319501245915516055106046884209969926127482827954674443846427813813222426"
)


def _load(name: str) -> dict:
    with (FIXTURES / name).open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# REST — read path (HttpCLOBClient, JSON-in, Pydantic-out)
# ---------------------------------------------------------------------------


async def test_get_book_replays_into_book_snap(monkeypatch):
    fixture = _load("get_book.json")
    client = HttpCLOBClient(CLOBConfig())

    async def fake_get(path, **params):
        assert path == "/book"
        assert params == {"token_id": TOKEN_ID}
        return fixture

    monkeypatch.setattr(client, "_get", fake_get)
    try:
        snap = await client.get_book(TOKEN_ID)
    finally:
        await client.close()

    assert isinstance(snap, BookSnap)
    assert snap.token_id == TOKEN_ID
    # Sorted: bids descending, asks ascending.
    assert [lv.price for lv in snap.bids] == [0.48, 0.47, 0.45]
    assert [lv.size for lv in snap.bids] == [125.0, 300.0, 500.0]
    assert [lv.price for lv in snap.asks] == [0.52, 0.53, 0.55]
    assert snap.spread == pytest.approx(0.04)
    assert snap.mid == pytest.approx(0.50)


def test_get_markets_fixture_matches_documented_shape():
    # No through-parser needed — the adapter passes the dict through. Contract
    # here is the *shape* Polymarket returns, so Track C can spot a breaking
    # change before it lands in callers.
    for name in ("get_markets_page1.json", "get_markets_page2.json"):
        page = _load(name)
        assert set(page) >= {"limit", "count", "next_cursor", "data"}
        for market in page["data"]:
            assert set(market) >= {
                "condition_id", "question", "active", "closed",
                "minimum_tick_size", "minimum_order_size", "tokens",
            }
            assert len(market["tokens"]) == 2
            for tok in market["tokens"]:
                assert set(tok) >= {"token_id", "outcome"}
    # last page's cursor is the documented terminal sentinel
    assert _load("get_markets_page2.json")["next_cursor"] == "LTE="


# ---------------------------------------------------------------------------
# REST — write path (PyCLOBClient with py-clob-client mocked out)
# ---------------------------------------------------------------------------


def _build_signing_client() -> PyCLOBClient:
    """Construct a PyCLOBClient whose underlying py-clob-client is a MagicMock
    and whose L2 auth is pre-skipped."""
    cfg = CLOBConfig(
        private_key="0x" + "1" * 64,
        funder="0x0000000000000000000000000000000000000001",
        api_key="k", api_secret="s", api_passphrase="p",
    )
    with patch("py_clob_client.client.ClobClient", MagicMock()):
        client = PyCLOBClient(cfg)
    client._raw = MagicMock()  # replace with a fresh mock we can program
    client._ensure_l2 = lambda: None  # L2 derivation not under test here
    return client


async def test_post_order_success_maps_to_open_order():
    fixture = _load("post_order_success.json")
    client = _build_signing_client()
    client._raw.create_and_post_order.return_value = fixture

    order = await client.place_order(
        token_id=TOKEN_ID, side=OrderSide.BUY, price=0.48, size=10,
        client_id="cid-ok",
    )

    assert order.status is OrderStatus.OPEN
    assert order.venue_id == fixture["orderID"]
    assert order.client_id == "cid-ok"
    assert order.side is OrderSide.BUY
    assert order.price == 0.48
    assert order.size == 10


async def test_post_order_reject_maps_to_rejected_order():
    fixture = _load("post_order_reject.json")
    client = _build_signing_client()
    client._raw.create_and_post_order.return_value = fixture

    order = await client.place_order(
        token_id=TOKEN_ID, side=OrderSide.BUY, price=0.48, size=10,
        client_id="cid-bad",
    )

    assert order.status is OrderStatus.REJECTED
    # Rejected orders have no venue_id on Polymarket's current schema.
    assert order.venue_id in (None, "")


async def test_delete_order_success_confirms_cancellation():
    fixture = _load("delete_order_success.json")
    client = _build_signing_client()
    client._raw.cancel.return_value = fixture

    venue_id = fixture["canceled"][0]
    ok = await client.cancel_order(venue_id)
    assert ok is True
    client._raw.cancel.assert_called_once_with(venue_id)


async def test_delete_order_partial_failure_is_not_confirmed():
    # If Polymarket moves the orderID into not_canceled, our contract should
    # return False rather than silently succeed.
    client = _build_signing_client()
    venue_id = "0xorder1111111111111111111111111111111111111111111111111111111111"
    client._raw.cancel.return_value = {
        "canceled": [],
        "not_canceled": {venue_id: "order not found"},
    }

    ok = await client.cancel_order(venue_id)
    assert ok is False


async def test_py_get_book_replays_into_book_snap():
    """``PyCLOBClient.get_book`` consumes py-clob-client's ``OrderBookSummary``
    (objects with ``.price`` / ``.size`` string attrs). This asserts our
    parser still sorts and type-converts after upstream bumps."""
    fixture = _load("get_book.json")
    client = _build_signing_client()

    def _lvl(d):
        return SimpleNamespace(price=d["price"], size=d["size"])

    book_summary = SimpleNamespace(
        bids=[_lvl(b) for b in fixture["bids"]],
        asks=[_lvl(a) for a in fixture["asks"]],
    )
    client._raw.get_order_book.return_value = book_summary

    snap = await client.get_book(TOKEN_ID)

    assert isinstance(snap, BookSnap)
    assert [lv.price for lv in snap.bids] == [0.48, 0.47, 0.45]
    assert [lv.price for lv in snap.asks] == [0.52, 0.53, 0.55]


# ---------------------------------------------------------------------------
# WS — delegate to Track A's polyclient parsers, but pin the fixture shape
# ---------------------------------------------------------------------------


def test_ws_book_snapshot_replays_through_polyclient_parser():
    fixture = _load("ws_book_snapshot.json")
    # polyclient.book_from_ws is Track A's entry point; this test is a
    # cross-track contract gate — if Polymarket changes the WS shape we want
    # both tracks to fail at the same commit.
    from blksch.core.ingest.polyclient import book_from_ws

    snap = book_from_ws(fixture)
    assert isinstance(snap, BookSnap)
    assert snap.token_id == fixture["asset_id"]
    assert [lv.price for lv in snap.bids] == [0.49, 0.48]
    assert [lv.price for lv in snap.asks] == [0.51, 0.52]
    assert isinstance(snap.ts, datetime)


def test_ws_last_trade_price_replays_through_polyclient_parser():
    fixture = _load("ws_last_trade_price.json")
    from blksch.core.ingest.polyclient import trade_from_ws

    tick = trade_from_ws(fixture)
    assert tick.token_id == fixture["asset_id"]
    assert tick.price == 0.51
    assert tick.size == 25.0
    assert tick.aggressor_side is TradeSide.BUY
