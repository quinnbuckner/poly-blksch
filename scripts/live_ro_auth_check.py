"""Signed read-only pre-flight for Stage-2 promotion.

Hits ``GET /orders`` (the "list my own open orders" endpoint — requires L2
API signing but has no side effects) and asserts the plumbing works end-to-
end: private key → L2 API creds → signed headers → 200 response.

If this passes, `scripts/signing_canary.py` is safe to run. If this fails,
`signing_canary.py` will fail too but will have already placed / failed to
cancel a real order — so fix the creds here first.

Gate on ``--i-mean-it``. Without it, the script prints what it would do and
exits 0. Loads credentials from ``.env`` (via python-dotenv) if present.

Usage::

    python scripts/live_ro_auth_check.py                # dry-run
    python scripts/live_ro_auth_check.py --i-mean-it    # signed GET /orders
    python scripts/live_ro_auth_check.py --i-mean-it --testnet
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

log = logging.getLogger("live_ro_auth_check")


REQUIRED_ENV = (
    "POLY_PRIVATE_KEY",
    "POLY_FUNDER_ADDRESS",
)
OPTIONAL_L2_ENV = (
    "POLY_API_KEY",
    "POLY_API_SECRET",
    "POLY_API_PASSPHRASE",
)


def _load_env() -> None:
    """Try python-dotenv; silently no-op if not installed (creds may already
    be in process env)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        log.debug("python-dotenv not installed; relying on process env")
        return
    load_dotenv()


def _validate_env() -> tuple[bool, list[str]]:
    import os
    missing_required = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    missing_l2 = [k for k in OPTIONAL_L2_ENV if not os.environ.get(k)]
    if missing_required:
        return False, missing_required
    if missing_l2:
        log.info(
            "L2 API creds missing (%s) — will derive from private key on first signed call",
            ", ".join(missing_l2),
        )
    return True, []


async def _run(*, testnet: bool) -> int:
    from blksch.exec.clob_client import CLOBConfig, PyCLOBClient

    cfg = CLOBConfig.from_env(testnet=testnet)
    if not cfg.has_signing_creds():
        log.error("Missing POLY_PRIVATE_KEY / POLY_FUNDER_ADDRESS — cannot sign")
        return 2

    client = PyCLOBClient(cfg)
    log.info(
        "Constructed PyCLOBClient on chain_id=%d funder=%s...%s",
        cfg.chain_id, cfg.funder[:6] if cfg.funder else "?",
        cfg.funder[-4:] if cfg.funder else "?",
    )

    # ``get_orders`` is signed but read-only — list of this maker's open orders.
    try:
        orders: Any = await asyncio.to_thread(client._ensure_l2)
        orders = await asyncio.to_thread(client._raw.get_orders)
    except Exception as exc:
        log.error("SIGNED GET /orders FAILED: %s", exc)
        return 3

    count = len(orders) if hasattr(orders, "__len__") else "unknown"
    log.info("SIGNED GET /orders OK — maker has %s open order(s)", count)
    log.info("Auth plumbing verified. signing_canary.py is safe to run next.")
    return 0


def _print_plan(args: argparse.Namespace) -> None:
    log.warning("Dry-run (no --i-mean-it). Plan:")
    log.warning("  1. Load .env (if present) + process env")
    log.warning("  2. Validate %s are set", ", ".join(REQUIRED_ENV))
    log.warning("  3. Build PyCLOBClient on %s",
                "Polygon Amoy (testnet)" if args.testnet else "Polygon mainnet")
    log.warning("  4. Derive L2 API creds if missing")
    log.warning("  5. Signed GET /orders (list my own open orders — zero side effects)")
    log.warning("  6. Exit 0 on success; non-zero otherwise")
    log.warning("Re-run with --i-mean-it to execute.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Signed read-only pre-flight check for Polymarket CLOB auth",
    )
    parser.add_argument(
        "--i-mean-it",
        action="store_true",
        help="Required to actually hit the CLOB. Without it, prints the plan and exits 0.",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Point at Polygon Amoy instead of Polygon mainnet.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _load_env()

    if not args.i_mean_it:
        _print_plan(args)
        return 0

    ok, missing = _validate_env()
    if not ok:
        log.error("Missing required env vars: %s. See .env.example.", missing)
        return 2

    return asyncio.run(_run(testnet=args.testnet))


if __name__ == "__main__":
    sys.exit(main())
