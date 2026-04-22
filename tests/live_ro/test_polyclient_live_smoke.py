"""Live read-only smoke tests — hit Polymarket's public endpoints.

Run manually:
    pytest tests/live_ro -v -m live_ro

Skipped by default (excluded from CI).
"""

from __future__ import annotations

import asyncio

import pytest

polyclient_mod = pytest.importorskip(
    "blksch.core.ingest.polyclient",
    reason="core/ingest/polyclient.py not yet implemented",
)

pytestmark = pytest.mark.live_ro


# A known liquid market. Replace with one that has stable depth for testing.
# Track A should wire this via a config lookup rather than hardcoding forever.
KNOWN_LIQUID_TOKEN = "UPDATE_ME_WITH_A_STABLE_LIQUID_TOKEN_ID"


class TestPolymarketLiveReadOnly:
    def test_gamma_markets_endpoint_reachable(self) -> None:
        """GET /markets returns >= 1 market and has 'token_id' on each entry."""
        client = polyclient_mod.PolyClient()  # type: ignore[attr-defined]
        try:
            markets = asyncio.run(client.list_markets(limit=5))  # type: ignore[attr-defined]
        finally:
            asyncio.run(client.close())  # type: ignore[attr-defined]
        assert markets, "Gamma /markets returned empty"
        assert all(m.get("token_id") for m in markets)

    def test_clob_book_endpoint_returns_populated_book(self) -> None:
        """GET /book for a known liquid token returns bids AND asks."""
        if KNOWN_LIQUID_TOKEN.startswith("UPDATE_ME"):
            pytest.skip("Set KNOWN_LIQUID_TOKEN to a real liquid Polymarket token")
        client = polyclient_mod.PolyClient()  # type: ignore[attr-defined]
        try:
            snap = asyncio.run(client.get_book(KNOWN_LIQUID_TOKEN))  # type: ignore[attr-defined]
        finally:
            asyncio.run(client.close())  # type: ignore[attr-defined]
        assert snap.bids, "Book has no bids"
        assert snap.asks, "Book has no asks"
        assert snap.mid is not None
        assert 0.0 < snap.mid < 1.0

    def test_clob_ws_book_channel_delivers_messages(self) -> None:
        """Subscribe to a known token's book channel, receive >= 3 messages
        within 20s."""
        pytest.skip("Stub: implement once polyclient WS API is pinned")
