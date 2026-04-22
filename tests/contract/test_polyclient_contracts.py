"""Contract tests for `core/ingest/polyclient.py` — pin against recorded responses.

These tests parse fixtures in tests/contract/fixtures/ and assert they round-trip
into our Pydantic schemas. When Polymarket changes its response schema, these
break first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

polyclient = pytest.importorskip(
    "blksch.core.ingest.polyclient",
    reason="core/ingest/polyclient.py not yet implemented",
)

pytestmark = pytest.mark.contract

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict | list:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"Fixture {name} not recorded yet — see tests/contract/README.md")
    with path.open() as f:
        return json.load(f)


class TestBookContract:
    def test_book_response_parses_into_book_snap(self, token_id: str) -> None:
        raw = _load("book_sample.json")
        snap = polyclient.parse_book_response(raw, token_id=token_id)  # type: ignore[attr-defined]
        assert snap.bids, "parsed book has no bids"
        assert snap.asks, "parsed book has no asks"
        assert snap.mid is not None
        assert 0.0 < snap.mid < 1.0

    def test_empty_side_yields_none_mid(self, token_id: str) -> None:
        raw = {"bids": [], "asks": []}
        snap = polyclient.parse_book_response(raw, token_id=token_id)  # type: ignore[attr-defined]
        assert snap.mid is None


class TestTradesContract:
    def test_trades_response_parses_into_trade_ticks(self) -> None:
        raw = _load("trades_sample.json")
        ticks = polyclient.parse_trades_response(raw)  # type: ignore[attr-defined]
        assert ticks, "parsed trades empty"
        assert all(0.0 <= t.price <= 1.0 for t in ticks)
        assert all(t.size > 0 for t in ticks)


class TestMarketsContract:
    def test_markets_page_parses_with_expected_fields(self) -> None:
        """Only pin the fields we actually use — new fields are allowed."""
        raw = _load("markets_sample.json")
        markets = polyclient.parse_markets_response(raw)  # type: ignore[attr-defined]
        assert markets, "parsed markets empty"
        for m in markets:
            # Required: every market has a token_id and a name
            assert m.get("token_id"), f"missing token_id in {m}"


class TestWSContract:
    def test_ws_book_message_parses_into_book_update(self) -> None:
        raw = _load("ws_book_messages.jsonl")  # jsonl -> parsed as list if needed
        # Typical wrapping: {"event": "book", "data": {...}}; adapt when we record.
        pytest.skip("Stub: activates when ws_book_messages.jsonl is recorded")
