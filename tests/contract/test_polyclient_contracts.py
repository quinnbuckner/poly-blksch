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
    """Load a JSON or JSONL fixture.

    ``.jsonl`` files are decoded line-by-line into a list — the recorder
    (``scripts/record_contract_fixtures.py``) writes one WS frame per
    line so re-running against a live subscription appends cleanly.
    Everything else goes through ``json.load``.
    """
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"Fixture {name} not recorded yet — see tests/contract/README.md")
    if path.suffix == ".jsonl":
        out: list = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with path.open() as f:
        return json.load(f)


class TestBookContract:
    def test_book_response_parses_into_book_snap(self, token_id: str) -> None:
        raw = _load("book_sample.json")
        snap = polyclient.book_from_rest(raw, token_id=token_id)  # type: ignore[attr-defined]
        assert snap.bids, "parsed book has no bids"
        assert snap.asks, "parsed book has no asks"
        assert snap.mid is not None
        assert 0.0 < snap.mid < 1.0

    def test_empty_side_yields_none_mid(self, token_id: str) -> None:
        raw = {"bids": [], "asks": []}
        snap = polyclient.book_from_rest(raw, token_id=token_id)  # type: ignore[attr-defined]
        assert snap.mid is None


class TestTradesContract:
    def test_trade_ws_message_parses_into_trade_tick(self) -> None:
        raw = _load("trade_ws_sample.json")
        tick = polyclient.trade_from_ws(raw)  # type: ignore[attr-defined]
        assert 0.0 <= tick.price <= 1.0
        assert tick.size > 0


class TestMarketsContract:
    def test_markets_page_parses_with_expected_fields(self) -> None:
        """Only pin the fields we actually use — new fields are allowed.

        Track A's PolyClient.list_markets() consumes Gamma's pagination directly;
        this test skips until a recorded Gamma response fixture lands.
        """
        raw = _load("markets_sample.json")
        assert isinstance(raw, list), "Gamma /markets returns a JSON list"
        for m in raw:
            assert m.get("clobTokenIds") or m.get("token_id"), f"missing token id in {m}"


class TestWSContract:
    def test_ws_book_message_parses_into_book_update(self) -> None:
        raw = _load("ws_book_messages.jsonl")  # jsonl -> parsed as list if needed
        # Typical wrapping: {"event": "book", "data": {...}}; adapt when we record.
        pytest.skip("Stub: activates when ws_book_messages.jsonl is recorded")
