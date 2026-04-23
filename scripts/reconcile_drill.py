"""Ledger-drift kill-switch drill (Stage-2 promotion gate).

Replays a recorded sequence of :class:`blksch.schemas.Fill` messages through
an in-memory :class:`blksch.exec.ledger.Ledger`, then **deliberately
corrupts one fill row** and verifies that
:func:`blksch.exec.ledger.reconcile_against_ledger` flags the mismatch.

This is the "ledger has drifted from venue" safety drill: if a fill row can
be silently mangled without reconcile() noticing, Stage-2 live trading is
unsafe because a fat-finger edit / race / double-insert would go
undetected.

Supported corruption modes::

    wrong_price         — overwrite fills.price on a single row
    wrong_size          — overwrite fills.size on a single row
    duplicate_client_id — INSERT a second row with the same order_client_id

The CLI defaults to a dry-run that prints the plan and exits 0. Passing
``--i-mean-it`` executes the drill against an in-memory ledger and exits
0 only if every mode triggers reconcile() as expected. Passing
``--mode=<mode>`` runs a single mode.

This script never touches the filesystem (in-memory SQLite) and never
places real orders — it is safe to run in any environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

from blksch.exec.ledger import Ledger, reconcile_against_ledger
from blksch.schemas import Fill, OrderSide

log = logging.getLogger("reconcile_drill")

TOKEN = "0xdeadbeef-drill"


# ---------------------------------------------------------------------------
# Canonical recorded fill sequence
# ---------------------------------------------------------------------------


def default_fill_sequence() -> list[Fill]:
    """A realistic 5-fill trajectory: open long, add, partial close, flip
    short, close out. Designed so any single-row corruption produces a
    detectable drift in either realized PnL or qty.
    """
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)

    def _f(cid: str, side: OrderSide, price: float, size: float, i: int, fee: float = 0.01) -> Fill:
        return Fill(
            order_client_id=cid,
            order_venue_id=f"venue-{cid}",
            token_id=TOKEN,
            side=side,
            price=price,
            size=size,
            fee_usd=fee,
            ts=t0 + timedelta(seconds=i),
        )

    return [
        _f("cid-001", OrderSide.BUY,  0.40, 10, 1),
        _f("cid-002", OrderSide.BUY,  0.50, 10, 2),   # WAP -> 0.45, qty=20
        _f("cid-003", OrderSide.SELL, 0.60,  5, 3),   # close 5 @ +0.15; qty=15
        _f("cid-004", OrderSide.SELL, 0.55, 25, 4),   # close 15 @ +0.10 then flip short 10 @ 0.55
        _f("cid-005", OrderSide.BUY,  0.50, 10, 5),   # close short 10 @ +0.05; qty=0
    ]


# ---------------------------------------------------------------------------
# Corruption primitives — operate on the raw SQLite fills table
# ---------------------------------------------------------------------------


CorruptionMode = str  # "wrong_price" | "wrong_size" | "duplicate_client_id"
ALL_MODES: tuple[CorruptionMode, ...] = (
    "wrong_price",
    "wrong_size",
    "duplicate_client_id",
)


def _corrupt(ledger: Ledger, mode: CorruptionMode, *, target_cid: str) -> str:
    """Apply a single deterministic corruption. Returns a human-readable
    description of what was changed.

    Corruptions bypass :meth:`Ledger.apply_fill` on purpose: the `positions`
    table stays consistent with the *original* fill, while the `fills` table
    is mutated. reconcile() must notice the two have diverged.
    """
    with ledger._lock:
        row = ledger._conn.execute(
            "SELECT id, price, size FROM fills WHERE order_client_id=?",
            (target_cid,),
        ).fetchone()
        if not row:
            raise RuntimeError(f"no fill with client_id={target_cid!r} to corrupt")
        fill_id, price, size = row

        if mode == "wrong_price":
            new_price = price + 0.10  # large, > 1e-6 tolerance
            ledger._conn.execute(
                "UPDATE fills SET price=? WHERE id=?", (new_price, fill_id),
            )
            return f"price {price} -> {new_price} on fill id={fill_id} cid={target_cid}"

        if mode == "wrong_size":
            new_size = size + 5.0
            ledger._conn.execute(
                "UPDATE fills SET size=? WHERE id=?", (new_size, fill_id),
            )
            return f"size {size} -> {new_size} on fill id={fill_id} cid={target_cid}"

        if mode == "duplicate_client_id":
            ledger._conn.execute(
                "INSERT INTO fills (order_client_id, order_venue_id, token_id, "
                "side, price, size, fee_usd, ts) "
                "SELECT order_client_id, order_venue_id, token_id, side, "
                "price, size, fee_usd, ts FROM fills WHERE id=?",
                (fill_id,),
            )
            return f"duplicated row for cid={target_cid} (was id={fill_id})"

    raise ValueError(f"unknown corruption mode: {mode}")


# ---------------------------------------------------------------------------
# Drill runner
# ---------------------------------------------------------------------------


@dataclass
class DrillResult:
    mode: CorruptionMode
    detected: bool
    corruption: str
    report_dict: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def run_drill(
    mode: CorruptionMode,
    *,
    fills: list[Fill] | None = None,
    target_cid: str = "cid-003",
    ledger_factory: Callable[[], Ledger] = Ledger.in_memory,
) -> DrillResult:
    """Run a single corruption mode end-to-end and return the result."""

    ledger = ledger_factory()
    try:
        fills_seq = fills or default_fill_sequence()
        for f in fills_seq:
            ledger.apply_fill(f)
        # Baseline must pass before we corrupt — otherwise the drill proves
        # nothing about reconcile().
        baseline = reconcile_against_ledger(ledger)
        if not baseline.passed:
            raise RuntimeError(
                f"baseline reconciliation failed before corruption: "
                f"{[d.detail for d in baseline.discrepancies]}"
            )
        corruption = _corrupt(ledger, mode, target_cid=target_cid)
        report = reconcile_against_ledger(ledger)
        return DrillResult(
            mode=mode,
            detected=not report.passed,
            corruption=corruption,
            report_dict=report.to_dict(),
        )
    finally:
        ledger.close()


def run_all(
    modes: list[CorruptionMode] | None = None,
    *,
    target_cid: str = "cid-003",
) -> tuple[bool, list[DrillResult]]:
    """Run every mode and return ``(all_detected, results)``."""
    modes = modes or list(ALL_MODES)
    results = [run_drill(m, target_cid=target_cid) for m in modes]
    return all(r.detected for r in results), results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_plan(args: argparse.Namespace) -> None:
    modes = [args.mode] if args.mode else list(ALL_MODES)
    log.warning("Dry-run (no --i-mean-it). Plan:")
    log.warning("  1. Build an in-memory Ledger (no filesystem, no network).")
    log.warning("  2. Apply %d canonical fill(s) to it.", len(default_fill_sequence()))
    log.warning("  3. Verify reconcile_against_ledger() passes on clean state.")
    log.warning("  4. For each corruption mode, deliberately corrupt fill %s:",
                args.target_cid)
    for m in modes:
        log.warning("       - %s", m)
    log.warning("  5. Assert reconcile_against_ledger() reports drift.")
    log.warning("  6. Exit 0 iff drift is detected in every mode.")
    log.warning("Re-run with --i-mean-it to execute.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ledger-drift reconciliation drill (Stage-2 kill-switch gate)",
    )
    parser.add_argument(
        "--mode", choices=ALL_MODES, default=None,
        help="Run a single corruption mode instead of all three.",
    )
    parser.add_argument(
        "--target-cid", default="cid-003",
        help="Client ID of the fill to corrupt (default: cid-003).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional JSON path to write the drill report to.",
    )
    parser.add_argument(
        "--i-mean-it", action="store_true",
        help="Required to actually run. Without it, prints the plan and exits 0.",
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

    modes = [args.mode] if args.mode else list(ALL_MODES)
    all_detected, results = run_all(modes, target_cid=args.target_cid)

    for r in results:
        status = "DETECTED" if r.detected else "MISSED"
        log.info("%-22s  %s  (%s)", r.mode, status, r.corruption)
        if not r.detected:
            log.error("  reconcile() did not flag this corruption — Stage-2 unsafe.")
        else:
            kinds = sorted({d["kind"] for d in r.report_dict["discrepancies"]})
            log.info("  discrepancy kinds: %s", ", ".join(kinds))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "passed": all_detected,
            "results": [r.to_dict() for r in results],
        }, indent=2))
        log.info("wrote %s", args.out)

    if all_detected:
        log.info("DRILL PASSED — every corruption mode was detected.")
        return 0

    log.error("DRILL FAILED — reconcile() did not detect every corruption.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
