"""STAGE 1 -> STAGE 2 GATE: live signing canary.

Places a tiny $1 YES-buy order at a far-from-market price (won't fill), waits
3 seconds, cancels. If both the signed POST and signed DELETE succeed, our
EIP-712 signing + auth plumbing works against the live CLOB.

Do NOT promote to Stage 2 (--mode=live) until this passes. Run
``live_ro_auth_check.py --i-mean-it`` first — it exercises the same auth
pipeline without placing an order.

Usage::

    python scripts/signing_canary.py --token-id <TOKEN>                     # dry-run
    python scripts/signing_canary.py --token-id <TOKEN> --i-mean-it         # mainnet
    python scripts/signing_canary.py --token-id <TOKEN> --i-mean-it --testnet

Env (via .env or process env):
    POLY_PRIVATE_KEY
    POLY_FUNDER_ADDRESS
    POLY_API_KEY             (optional — derived on first signed call)
    POLY_API_SECRET          (optional)
    POLY_API_PASSPHRASE      (optional)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid

log = logging.getLogger("signing_canary")


REQUIRED_ENV = ("POLY_PRIVATE_KEY", "POLY_FUNDER_ADDRESS")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _validate_env() -> list[str]:
    return [k for k in REQUIRED_ENV if not os.environ.get(k)]


async def _run(token_id: str, far_price: float, size_usd: float, *, testnet: bool) -> int:
    try:
        from blksch.exec.clob_client import CLOBConfig, PyCLOBClient
        from blksch.schemas import OrderSide, OrderStatus
    except ImportError as exc:
        log.error("blksch package not importable: %s", exc)
        log.error("Install with `pip install -e .` from the repo root.")
        return 2

    cfg = CLOBConfig.from_env(testnet=testnet)
    if not cfg.has_signing_creds():
        log.error("Missing POLY_PRIVATE_KEY / POLY_FUNDER_ADDRESS — cannot sign.")
        return 2

    # Testnet sanity: refuse to sign against the fail-closed zero-address.
    vc = cfg.resolved_verifying_contract()
    if cfg.network == "testnet" and int(vc, 16) == 0:
        log.error(
            "Testnet run refuses to sign against verifying_contract=%s. "
            "Export POLY_VERIFYING_CONTRACT to the live Amoy CTF Exchange address.",
            vc,
        )
        return 2
    log.info(
        "CLOB: network=%s chain_id=%d base=%s verifying_contract=%s",
        cfg.network, cfg.chain_id, cfg.base_url, vc,
    )

    client = PyCLOBClient(cfg)
    client_id = f"canary-{uuid.uuid4().hex[:10]}"
    size = size_usd / max(far_price, 1e-6)

    log.info(
        "Placing canary BUY: token=%s price=%.4f size=%.4f (notional≈$%.2f). Should NOT fill.",
        token_id, far_price, size, size_usd,
    )
    try:
        order = await client.place_order(
            token_id=token_id,
            side=OrderSide.BUY,
            price=far_price,
            size=size,
            client_id=client_id,
            order_type="GTC",
        )
    except Exception as exc:
        log.error("CANARY FAILED on POST /order: %s", exc)
        return 3

    if order.status is OrderStatus.REJECTED:
        log.error("CANARY FAILED — CLOB rejected order: client_id=%s", client_id)
        return 3

    venue_id = order.venue_id
    log.info("Placed OK: client_id=%s venue_id=%s. Waiting 3s, then canceling...",
             client_id, venue_id)
    await asyncio.sleep(3.0)

    if not venue_id:
        log.error("No venue_id on placed order — cannot cancel. Check response shape.")
        return 4

    try:
        canceled = await client.cancel_order(venue_id)
    except Exception as exc:
        log.error("CANARY FAILED on DELETE /order: %s", exc)
        log.error("!!! Manually cancel venue_id=%s via the Polymarket UI !!!", venue_id)
        return 4

    if not canceled:
        log.error("Cancel not confirmed: venue_id=%s. Check ledger/UI before Stage-2 promotion.",
                  venue_id)
        return 4

    log.info("Cancel OK. CANARY PASSED — signing + auth plumbing works on %s.",
             "Amoy testnet" if testnet else "Polygon mainnet")
    return 0


def _print_plan(args: argparse.Namespace) -> None:
    size = args.size_usd / max(args.far_price, 1e-6)
    log.warning("Dry-run (no --i-mean-it). Plan:")
    log.warning("  1. Load .env (if present) + process env")
    log.warning("  2. Validate %s are set", ", ".join(REQUIRED_ENV))
    log.warning("  3. Build PyCLOBClient on %s",
                "Polygon Amoy (testnet)" if args.testnet else "Polygon mainnet")
    log.warning("  4. Signed POST /order: BUY token=%s price=%.4f size=%.4f (notional≈$%.2f)",
                args.token_id, args.far_price, size, args.size_usd)
    log.warning("  5. Sleep 3s")
    log.warning("  6. Signed DELETE /order (cancel)")
    log.warning("  7. Exit 0 only if both succeed")
    log.warning("Re-run with --i-mean-it to execute.")


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
        "--testnet",
        action="store_true",
        help="Point at Polygon Amoy instead of Polygon mainnet.",
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

    _load_env()

    if not args.i_mean_it:
        _print_plan(args)
        return 0

    missing = _validate_env()
    if missing:
        log.error("Missing required env vars: %s. See .env.example.", missing)
        return 2

    return asyncio.run(_run(args.token_id, args.far_price, args.size_usd, testnet=args.testnet))


if __name__ == "__main__":
    sys.exit(main())
