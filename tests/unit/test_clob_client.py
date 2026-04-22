"""Unit tests for `exec/clob_client.py`.

These are OFFLINE unit tests — HTTP requests are mocked. Live smoke tests live
in `tests/live_ro/test_polyclient_live_smoke.py` (Track A's polyclient) and
`tests/contract/test_polyclient_contracts.py` (contract-pinned).
"""

from __future__ import annotations

import pytest

clob_client_mod = pytest.importorskip(
    "blksch.exec.clob_client",
    reason="exec/clob_client.py not yet implemented",
)

pytestmark = pytest.mark.unit


class TestCLOBClient:
    def test_get_book_parses_into_book_snap(self) -> None:
        """Mock a known-good book response, assert the client returns a
        schemas.BookSnap with bids/asks populated."""
        pytest.skip("Stub: set up httpx/aiohttp mock and sample response")

    def test_get_markets_handles_pagination(self) -> None:
        pytest.skip("Stub")

    def test_post_order_includes_signed_payload(self) -> None:
        """POST /order body must contain the EIP-712 signature field."""
        pytest.skip("Stub")

    def test_rate_limiter_caps_qps(self) -> None:
        """Under burst, rate limiter holds below configured QPS (default 10/s)."""
        pytest.skip("Stub")

    def test_surfaces_rejections_with_reason(self) -> None:
        """A 400 with reason='INSUFFICIENT_BALANCE' must surface the reason
        string, not a generic error."""
        pytest.skip("Stub")
