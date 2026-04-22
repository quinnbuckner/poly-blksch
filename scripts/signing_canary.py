"""STAGE 1 -> STAGE 2 GATE: live signing canary.

Places a tiny $1 YES-buy order at a far-from-market price (won't fill), waits
3 seconds, cancels. If both the signed POST and signed DELETE succeed, our
EIP-712 signing + auth plumbing works against the live CLOB.

Do NOT promote to Stage 2 (--mode=live) until this passes.

Usage:
    python scripts/signing_canary.py --token-id <TOKEN> --i-mean-it

Env (via .env or process env):
    POLY_PRIVATE_KEY
    POLY_API_KEY
    POLY_API_SECRET
    POLY_API_PASSPHRASE
    POLY_FUNDER_ADDRESS
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

log = logging.getLogger("signing_canary")


def _require_env() -> dict[str, str]:
    missing = []
    required = (
        "POLY_PRIVATE_KEY",
        "POLY_API_KEY",
        "POLY_API_SECRET",
        "POLY_API_PASSPHRASE",
        "POLY_FUNDER_ADDRESS",
    )
    env = {}
    for key in required:
        v = os.environ.get(key)
        if not v:
            missing.append(key)
        env[key] = v or ""
    if missing:
        raise SystemExit(f"Missing required env vars: {missing}. See .env.example.")
    return env


async def _run(token_id: str, far_price: float, size_usd: float) -> int:
    # Imported lazily so --help works without the package installed.
    try:
        from blksch.exec.clob_client import make_clob_client  # type: ignore
    except ImportError as e:
        log.error("exec.clob_client not importable: %s", e)
        log.error("This canary requires Track C's clob_client to be landed.")
        return 2

    env = _require_env()
    client = make_clob_client(  # type: ignore[call-arg]
        private_key=env["POLY_PRIVATE_KEY"],
        api_key=env["POLY_API_KEY"],
        api_secret=env["POLY_API_SECRET"],
        api_passphrase=env["POLY_API_PASSPHRASE"],
        funder=env["POLY_FUNDER_ADDRESS"],
    )

    log.info(
        "Placing canary BUY: token=%s price=%s size_usd=%s (should NOT fill)",
        token_id, far_price, size_usd,
    )
    try:
        placed = await client.place_order(  # type: ignore[attr-defined]
            token_id=token_id,
            side="BUY",
            price=far_price,
            size=size_usd / max(far_price, 1e-6),
        )
    except Exception as e:
        log.error("CANARY FAILED on POST /order: %s", e)
        return 3

    order_id = placed.get("id") or placed.get("order_id")
    log.info("Placed OK: venue_id=%s. Waiting 3s, then canceling...", order_id)
    await asyncio.sleep(3.0)

    try:
        canceled = await client.cancel_order(order_id)  # type: ignore[attr-defined]
    except Exception as e:
        log.error("CANARY FAILED on DELETE /order: %s", e)
        log.error("!!! Manually cancel order_id=%s via Polymarket UI !!!", order_id)
        return 4

    log.info("Cancel OK: %s", canceled)
    log.info("CANARY PASSED — signing + auth plumbing works.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live signing canary for Polymarket CLOB")
    parser.add_argument("--token-id", required=True, help="Polymarket YES token_id")
    parser.add_argument(
        "--far-price",
        type=float,
        default=0.01,
        help="Place the canary at this price so it won't fill (default 0.01)",
    )
    parser.add_argument(
        "--size-usd",
        type=float,
        default=1.0,
        help="Notional size in USD (default $1)",
    )
    parser.add_argument(
        "--i-mean-it",
        action="store_true",
        help="Required to actually run. Without this flag, the script prints the plan and exits.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.i_mean_it:
        log.warning("Dry-run (no --i-mean-it). Plan:")
        log.warning("  1. Load credentials from env")
        log.warning("  2. POST signed BUY @ %s (size $%s)", args.far_price, args.size_usd)
        log.warning("  3. Sleep 3s")
        log.warning("  4. DELETE (cancel) the order")
        log.warning("Re-run with --i-mean-it to execute.")
        return 0

    return asyncio.run(_run(args.token_id, args.far_price, args.size_usd))


if __name__ == "__main__":
    sys.exit(main())
