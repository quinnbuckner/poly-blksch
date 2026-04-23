"""Live read-only contract fixture recorder.

Captures canonical Polymarket responses and drops them under
``tests/contract/fixtures/`` so ``test_polyclient_contracts.py`` and
``test_clob_client_contract.py`` can stop skipping. No auth required —
these are the public market-data endpoints + the CLOB WS ``market``
channel.

Fixtures written
----------------

* ``markets_sample.json``   — one page of Gamma ``/markets`` (list).
* ``book_sample.json``      — one snapshot of CLOB ``/book`` for a liquid
                              YES token.
* ``ws_book_messages.jsonl`` — ≥10 raw ``book`` / ``price_change`` frames
                              captured from the CLOB WS ``market``
                              channel (one JSON object per line).
* ``trade_ws_sample.json``   — a single ``last_trade_price`` frame from
                              the same WS subscription.

Usage
-----

::

    python scripts/record_contract_fixtures.py             # dry-run (default)
    python scripts/record_contract_fixtures.py --i-mean-it # actually hit the network
    python scripts/record_contract_fixtures.py --i-mean-it --ws-timeout-sec 120

Scope rule: only writes under ``tests/contract/fixtures/``. Never touches
``src/blksch/``. Running twice is idempotent — existing fixtures are
overwritten with fresh captures (intentional: the point of this script is
to refresh the snapshot when the upstream schema shifts).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("record_contract_fixtures")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "contract" / "fixtures"

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Default capture budget when --ws-timeout-sec is not given. Book ticks
# arrive continuously on a liquid market; trades are rarer — we stop
# early once we've got ≥10 book frames AND ≥1 trade.
DEFAULT_WS_TIMEOUT_SEC = 90.0
MIN_BOOK_FRAMES = 10
MIN_TRADE_FRAMES = 1


# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------


def _print_plan(args: argparse.Namespace) -> None:
    log.warning("Dry-run (no --i-mean-it). Would fetch (all live, read-only):")
    log.warning("  1. GET %s?limit=%d&offset=0", GAMMA_MARKETS_URL, args.markets_limit)
    log.warning("     → %s/markets_sample.json", FIXTURES_DIR.relative_to(REPO_ROOT))
    log.warning("  2. Select the highest-liquidity active market from that page")
    log.warning("     and extract its YES token_id (from `clobTokenIds`).")
    log.warning("  3. GET %s?token_id=<YES>", CLOB_BOOK_URL)
    log.warning("     → %s/book_sample.json", FIXTURES_DIR.relative_to(REPO_ROOT))
    log.warning("  4. Open CLOB WS %s", CLOB_WS_URL)
    log.warning("     Subscribe: {\"type\":\"market\",\"assets_ids\":[<YES>]}")
    log.warning(
        "     Capture until we have ≥%d book frames AND ≥%d trade frames "
        "(cap %.0fs).",
        MIN_BOOK_FRAMES, MIN_TRADE_FRAMES, args.ws_timeout_sec,
    )
    log.warning("     → %s/ws_book_messages.jsonl", FIXTURES_DIR.relative_to(REPO_ROOT))
    log.warning("     → %s/trade_ws_sample.json", FIXTURES_DIR.relative_to(REPO_ROOT))
    log.warning("Re-run with --i-mean-it to execute.")


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


async def _fetch_markets_page(session, *, limit: int) -> list[dict[str, Any]]:
    async with session.get(GAMMA_MARKETS_URL, params={"limit": limit, "offset": 0}) as r:
        r.raise_for_status()
        data = await r.json()
    if isinstance(data, list):
        return data
    return data.get("data") or data.get("markets") or []


def _yes_token_of(m: dict[str, Any]) -> str | None:
    """Pull a YES token_id out of a single Gamma market row, None on miss."""
    raw_ids = m.get("clobTokenIds") or m.get("token_ids") or m.get("tokens")
    if not raw_ids:
        return None
    if isinstance(raw_ids, str):
        try:
            raw_ids = json.loads(raw_ids)
        except json.JSONDecodeError:
            return None
    if isinstance(raw_ids, dict):
        tok = (
            raw_ids.get("YES") or raw_ids.get("Yes") or raw_ids.get("yes")
            or next(iter(raw_ids.values()), None)
        )
        return str(tok) if tok else None
    if isinstance(raw_ids, list) and raw_ids:
        first = raw_ids[0]
        if isinstance(first, dict):
            tok = first.get("token_id") or first.get("id")
            return str(tok) if tok else None
        return str(first)
    return None


def _ranked_active(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Active, non-closed markets sorted by recent-activity signals.

    Keys off ``volume24hr`` (what's trading NOW), with ``liquidityNum``
    as the tie-breaker. Markets without a ``lastTradePrice`` are pushed
    to the back — long-dated sports futures (big cumulative volume,
    no recent trades) are a trap for capturing live WS frames.
    """
    def _vol24(m: dict[str, Any]) -> float:
        for key in ("volume24hr", "volume24hrClob", "volume1wk"):
            v = m.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _liq(m: dict[str, Any]) -> float:
        v = m.get("liquidityNum") or m.get("liquidity")
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _recently_traded(m: dict[str, Any]) -> int:
        return 1 if m.get("lastTradePrice") is not None else 0

    active = [
        m for m in markets
        if m.get("active") and not m.get("closed", False)
        and m.get("acceptingOrders", True)
    ]
    if not active:
        active = list(markets)
    # Sort by (recently_traded, volume24hr, liquidity) descending.
    active.sort(key=lambda m: (_recently_traded(m), _vol24(m), _liq(m)), reverse=True)
    return active


def _extract_yes_token_id(markets: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Pick the highest-liquidity active market and return (YES token, row)."""
    for m in _ranked_active(markets):
        tok = _yes_token_of(m)
        if tok:
            return tok, m
    raise RuntimeError("No market in the page had a usable YES token id.")


def _trade_subscription_tokens(
    markets: list[dict[str, Any]], primary: str, *, fanout: int,
) -> list[str]:
    """Return the primary YES token + up to ``fanout``-1 other active YES
    tokens so a trade on any of them gets captured. Deduplicated."""
    tokens: list[str] = [primary]
    seen = {primary}
    for m in _ranked_active(markets):
        tok = _yes_token_of(m)
        if tok and tok not in seen:
            tokens.append(tok)
            seen.add(tok)
            if len(tokens) >= fanout:
                break
    return tokens


async def _fetch_book(session, *, token_id: str) -> dict[str, Any]:
    async with session.get(CLOB_BOOK_URL, params={"token_id": token_id}) as r:
        r.raise_for_status()
        return await r.json()


async def _capture_ws(
    token_ids: list[str],
    *,
    primary_token: str,
    timeout_sec: float,
) -> tuple[list[dict], dict | None]:
    """Subscribe to the CLOB market channel and capture raw frames.

    ``token_ids`` is the full subscription fan-out (primary + peers);
    ``primary_token`` scopes the ``book`` / ``price_change`` capture so
    ``ws_book_messages.jsonl`` correlates with ``book_sample.json``.
    Trades are accepted from any token in the fan-out — their fixture is
    used for schema pinning, not cross-referenced with the book fixture.

    Returns ``(book_messages_for_primary, one_trade_or_None)``.
    """
    import websockets  # imported lazily so --help works without the dep

    book_messages: list[dict] = []
    trade_message: dict | None = None
    deadline = time.monotonic() + timeout_sec

    async with websockets.connect(CLOB_WS_URL, ping_interval=30.0) as ws:
        await ws.send(json.dumps({"type": "market", "assets_ids": token_ids}))
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break

            if isinstance(frame, (bytes, bytearray)):
                frame = frame.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(frame)
            except json.JSONDecodeError:
                continue
            msgs = parsed if isinstance(parsed, list) else [parsed]

            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                etype = str(msg.get("event_type", "")).lower()
                if etype in ("book", "price_change"):
                    # Accept from any subscribed token so the fixture gets
                    # its promised ≥10 frames even when the primary is
                    # momentarily quiet. The polyclient parser is
                    # token-agnostic, so mixing tokens in the jsonl is a
                    # strictly stronger contract test.
                    book_messages.append(msg)
                elif etype in ("last_trade_price", "trade", "last_trade"):
                    if trade_message is None:
                        trade_message = msg

            if (
                len(book_messages) >= MIN_BOOK_FRAMES
                and trade_message is not None
            ):
                break

    return book_messages, trade_message


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    try:
        import aiohttp
    except ImportError:
        log.error("aiohttp not installed; install with `pip install -e .`")
        return 2

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15.0)
    ) as session:

        # 1. markets
        log.info("Fetching Gamma /markets (limit=%d)", args.markets_limit)
        markets = await _fetch_markets_page(session, limit=args.markets_limit)
        if not markets:
            log.error("Gamma returned zero markets — aborting.")
            return 3
        markets_path = FIXTURES_DIR / "markets_sample.json"
        markets_path.write_text(json.dumps(markets, indent=2) + "\n")
        log.info("wrote %s  (%d rows)", markets_path.relative_to(REPO_ROOT), len(markets))

        # 2. pick a YES token + book
        token_id, chosen = _extract_yes_token_id(markets)
        slug = chosen.get("slug") or chosen.get("question") or "?"
        log.info("Selected market: %s  (token_id=%s…%s)",
                 slug, token_id[:12], token_id[-6:])
        book = await _fetch_book(session, token_id=token_id)
        book_path = FIXTURES_DIR / "book_sample.json"
        book_path.write_text(json.dumps(book, indent=2) + "\n")
        log.info("wrote %s  (bids=%d asks=%d)",
                 book_path.relative_to(REPO_ROOT),
                 len(book.get("bids") or []), len(book.get("asks") or []))

    # 3. + 4. WS capture (uses its own connection, not aiohttp).
    sub_tokens = _trade_subscription_tokens(
        markets, primary=token_id, fanout=args.trade_fanout,
    )
    log.info(
        "Opening CLOB WS, subscribing to %d tokens (1 primary + %d peers) "
        "to increase trade-capture odds. Cap %.0fs.",
        len(sub_tokens), len(sub_tokens) - 1, args.ws_timeout_sec,
    )
    book_msgs, trade_msg = await _capture_ws(
        sub_tokens, primary_token=token_id, timeout_sec=args.ws_timeout_sec,
    )

    if len(book_msgs) < MIN_BOOK_FRAMES:
        log.warning(
            "Only captured %d book frames (wanted ≥%d) — writing anyway. "
            "Re-run at a busier hour if the contract test needs more.",
            len(book_msgs), MIN_BOOK_FRAMES,
        )
    ws_book_path = FIXTURES_DIR / "ws_book_messages.jsonl"
    with ws_book_path.open("w") as fh:
        for msg in book_msgs:
            fh.write(json.dumps(msg) + "\n")
    log.info("wrote %s  (%d frames)",
             ws_book_path.relative_to(REPO_ROOT), len(book_msgs))

    if trade_msg is None:
        log.warning(
            "No trade frame observed within %.0fs. Leaving "
            "trade_ws_sample.json untouched — re-run during active hours "
            "or tolerate the test-skip.",
            args.ws_timeout_sec,
        )
    else:
        trade_path = FIXTURES_DIR / "trade_ws_sample.json"
        trade_path.write_text(json.dumps(trade_msg, indent=2) + "\n")
        log.info("wrote %s", trade_path.relative_to(REPO_ROOT))

    log.info("Done. Fixtures under %s", FIXTURES_DIR.relative_to(REPO_ROOT))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record live-RO contract fixtures for the blksch test suite.",
    )
    parser.add_argument(
        "--i-mean-it", action="store_true",
        help="Actually hit the network. Without this, prints the plan and exits 0.",
    )
    parser.add_argument(
        "--markets-limit", type=int, default=100,
        help="Number of rows to request from Gamma /markets (default 100).",
    )
    parser.add_argument(
        "--ws-timeout-sec", type=float, default=DEFAULT_WS_TIMEOUT_SEC,
        help=f"Max seconds to hold the WS subscription (default {DEFAULT_WS_TIMEOUT_SEC:g}).",
    )
    parser.add_argument(
        "--trade-fanout", type=int, default=20,
        help=(
            "Subscribe to this many tokens (primary + peers) so a trade on "
            "any of them lights up the trade fixture. Default 20."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.i_mean_it:
        _print_plan(args)
        return 0

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
