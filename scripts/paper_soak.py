"""72-hour paper-trading soak supervisor.

Runs ``python -m blksch.app --mode=paper`` as a subprocess, polls the
FlaskDashboard ``/api/state`` endpoint at a fixed cadence, aggregates stats
into an ``HourlyReport`` every hour, and at the end evaluates the hourly
reports against the Stage-1 → Stage-2 acceptance criteria declared in the
implementation plan::

    Stage 1 gate: 72-hour paper run with
        * positive simulated edge-per-fill after modeled fees,
        * inventory within caps,
        * zero reconciliation drift,
        * no unexplained kill-switch trips.

If any criterion fails, ``main()`` returns non-zero so CI / the operator can
block Stage-2 promotion. The supervisor never places live orders — it forces
``--mode=paper`` on the child and refuses to start if that argument is
absent.

This file deliberately keeps the aggregation / evaluation logic as pure
functions so :mod:`tests.unit.test_exec_soak_harness` can exercise every
pass/fail branch without spawning subprocesses.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("paper_soak")


# ---------------------------------------------------------------------------
# Acceptance criteria — Stage-1 → Stage-2 gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Thresholds for the Stage-1 → Stage-2 gate.

    Defaults come from ``config/bot.yaml`` but are pinned here so a unit
    test can freeze behavior. Tune by passing a different instance to
    :func:`evaluate`.
    """

    min_hours: int = 72
    min_quote_uptime_pct: float = 95.0
    max_inventory_notional_usd: float = 50.0
    max_pnl_attribution_residual_usd: float = 0.5
    max_unexpected_kill_switch_events: int = 0
    min_realized_edge_per_fill_usd: float = 0.0


@dataclass
class CriterionResult:
    name: str
    passed: bool
    observed: float | int
    threshold: float | int
    message: str


@dataclass
class AcceptanceResult:
    passed: bool
    results: list[CriterionResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "criteria": [dataclasses.asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# Per-hour aggregate
# ---------------------------------------------------------------------------


@dataclass
class HourlyReport:
    hour_index: int
    started_at: str
    ended_at: str
    samples: int
    quote_uptime_pct: float
    fills_in_hour: int
    realized_pnl_hour_usd: float
    realized_pnl_cumulative_usd: float
    fees_cumulative_usd: float
    inventory_peak_notional_usd: float
    inventory_peak_qty_by_token: dict[str, float]
    kill_switch_events: int
    halt_events: int
    pnl_attribution_residual_usd: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class HourlyAggregate:
    """Streaming aggregator — one per hour. Call :meth:`observe` for each
    polled sample, then :meth:`finalize` once the hour ends.

    Stage-1 "edge per fill" is realized PnL minus fees, divided by fills.
    Attribution residual is the difference between ``pnl.total_usd`` and
    ``realized_usd + unrealized_usd`` as reported by the Ledger; by
    construction it should be zero, so any non-zero value indicates either
    a bug or a Track-B attribution stream we don't yet consume.
    """

    def __init__(self, hour_index: int, started_at: str, *,
                 fills_count_at_start: int,
                 realized_pnl_at_start: float,
                 fees_at_start: float):
        self.hour_index = hour_index
        self.started_at = started_at
        self._ended_at = started_at
        self._samples = 0
        self._samples_with_quotes = 0
        self._fills_at_start = fills_count_at_start
        self._last_fills_count = fills_count_at_start
        self._realized_at_start = realized_pnl_at_start
        self._last_realized = realized_pnl_at_start
        self._fees_at_start = fees_at_start
        self._last_fees = fees_at_start
        self._peak_notional = 0.0
        self._peak_qty_by_token: dict[str, float] = {}
        self._ks_prev: dict[str, bool] = {}
        self._halted_prev = False
        self._kill_events = 0
        self._halt_events = 0
        self._last_attribution_residual = 0.0

    def observe(self, snapshot: dict[str, Any]) -> None:
        self._samples += 1
        self._ended_at = snapshot.get("ts", self._ended_at)

        quotes = snapshot.get("quotes") or {}
        if quotes:
            self._samples_with_quotes += 1

        engine = snapshot.get("engine") or {}
        fills = int(engine.get("fills_count") or 0)
        if fills > self._last_fills_count:
            self._last_fills_count = fills

        pnl = snapshot.get("pnl") or {}
        realized = float(pnl.get("realized_usd") or 0.0)
        fees = float(pnl.get("fees_usd") or 0.0)
        self._last_realized = realized
        self._last_fees = fees

        total = float(pnl.get("total_usd") or 0.0)
        unreal = float(pnl.get("unrealized_usd") or 0.0)
        # Residual = total - (realized + unrealized). Ledger guarantees 0
        # by construction, so drift here is a signal the pipeline introduced
        # a bug or Track B started publishing an attribution we should check.
        self._last_attribution_residual = max(
            self._last_attribution_residual,
            abs(total - (realized + unreal)),
        )

        for pos in snapshot.get("positions") or []:
            tok = pos["token_id"]
            qty = abs(float(pos.get("qty") or 0.0))
            mark = float(pos.get("mark") or 0.5)
            notional = qty * mark
            if notional > self._peak_notional:
                self._peak_notional = notional
            prev = self._peak_qty_by_token.get(tok, 0.0)
            if qty > prev:
                self._peak_qty_by_token[tok] = qty

        ks = snapshot.get("kill_switches") or {}
        for name, tripped in ks.items():
            prev = self._ks_prev.get(name, False)
            if tripped and not prev:
                self._kill_events += 1
            self._ks_prev[name] = bool(tripped)

        halted_now = bool(engine.get("halted"))
        if halted_now and not self._halted_prev:
            self._halt_events += 1
        self._halted_prev = halted_now

    def finalize(self) -> HourlyReport:
        uptime = (
            100.0 * self._samples_with_quotes / self._samples
            if self._samples > 0 else 0.0
        )
        fills_in_hour = max(0, self._last_fills_count - self._fills_at_start)
        return HourlyReport(
            hour_index=self.hour_index,
            started_at=self.started_at,
            ended_at=self._ended_at,
            samples=self._samples,
            quote_uptime_pct=uptime,
            fills_in_hour=fills_in_hour,
            realized_pnl_hour_usd=self._last_realized - self._realized_at_start,
            realized_pnl_cumulative_usd=self._last_realized,
            fees_cumulative_usd=self._last_fees,
            inventory_peak_notional_usd=self._peak_notional,
            inventory_peak_qty_by_token=dict(self._peak_qty_by_token),
            kill_switch_events=self._kill_events,
            halt_events=self._halt_events,
            pnl_attribution_residual_usd=self._last_attribution_residual,
        )


# ---------------------------------------------------------------------------
# Acceptance evaluator — pure function
# ---------------------------------------------------------------------------


def evaluate(reports: list[HourlyReport], criteria: AcceptanceCriteria) -> AcceptanceResult:
    """Apply the Stage-1 → Stage-2 acceptance criteria to a list of hourly
    reports and return a structured result.

    All criteria are evaluated even if earlier ones fail — the operator
    wants to see *every* reason the soak missed, not just the first.
    """

    results: list[CriterionResult] = []
    n_hours = len(reports)

    # 1. Duration
    results.append(CriterionResult(
        name="min_hours",
        passed=n_hours >= criteria.min_hours,
        observed=n_hours, threshold=criteria.min_hours,
        message=f"observed {n_hours} hours; need ≥ {criteria.min_hours}",
    ))

    if n_hours == 0:
        # With no data the remaining criteria are vacuously-failing — emit
        # them so the report still lists every criterion.
        for name, thresh in (
            ("min_quote_uptime_pct", criteria.min_quote_uptime_pct),
            ("max_inventory_notional_usd", criteria.max_inventory_notional_usd),
            ("max_pnl_attribution_residual_usd", criteria.max_pnl_attribution_residual_usd),
            ("max_unexpected_kill_switch_events", criteria.max_unexpected_kill_switch_events),
            ("min_realized_edge_per_fill_usd", criteria.min_realized_edge_per_fill_usd),
        ):
            results.append(CriterionResult(
                name=name, passed=False, observed=0, threshold=thresh,
                message="no hourly reports collected",
            ))
        return AcceptanceResult(passed=False, results=results)

    # 2. Quote uptime — average across hours
    uptime = sum(r.quote_uptime_pct for r in reports) / n_hours
    results.append(CriterionResult(
        name="min_quote_uptime_pct",
        passed=uptime >= criteria.min_quote_uptime_pct,
        observed=round(uptime, 3),
        threshold=criteria.min_quote_uptime_pct,
        message=f"mean quote uptime {uptime:.2f}% vs ≥ {criteria.min_quote_uptime_pct}%",
    ))

    # 3. Inventory peak
    peak = max(r.inventory_peak_notional_usd for r in reports)
    results.append(CriterionResult(
        name="max_inventory_notional_usd",
        passed=peak <= criteria.max_inventory_notional_usd,
        observed=round(peak, 3),
        threshold=criteria.max_inventory_notional_usd,
        message=f"peak notional ${peak:.2f} vs ≤ ${criteria.max_inventory_notional_usd}",
    ))

    # 4. PnL attribution residual
    max_residual = max(r.pnl_attribution_residual_usd for r in reports)
    results.append(CriterionResult(
        name="max_pnl_attribution_residual_usd",
        passed=max_residual <= criteria.max_pnl_attribution_residual_usd,
        observed=round(max_residual, 4),
        threshold=criteria.max_pnl_attribution_residual_usd,
        message=f"peak residual ${max_residual:.4f} vs ≤ ${criteria.max_pnl_attribution_residual_usd}",
    ))

    # 5. Kill-switch events — sum over all hours
    ks_events = sum(r.kill_switch_events + r.halt_events for r in reports)
    results.append(CriterionResult(
        name="max_unexpected_kill_switch_events",
        passed=ks_events <= criteria.max_unexpected_kill_switch_events,
        observed=ks_events,
        threshold=criteria.max_unexpected_kill_switch_events,
        message=f"{ks_events} kill/halt event(s) vs ≤ {criteria.max_unexpected_kill_switch_events}",
    ))

    # 6. Edge per fill = (total realized PnL − total fees) / total fills
    total_fills = sum(r.fills_in_hour for r in reports)
    realized_final = reports[-1].realized_pnl_cumulative_usd
    fees_final = reports[-1].fees_cumulative_usd
    if total_fills == 0:
        edge_per_fill = 0.0
        edge_msg = "no fills observed"
    else:
        edge_per_fill = (realized_final - fees_final) / total_fills
        edge_msg = (
            f"realized ${realized_final:.4f} − fees ${fees_final:.4f} "
            f"over {total_fills} fill(s) = ${edge_per_fill:.6f}/fill"
        )
    results.append(CriterionResult(
        name="min_realized_edge_per_fill_usd",
        passed=edge_per_fill >= criteria.min_realized_edge_per_fill_usd and total_fills > 0,
        observed=round(edge_per_fill, 6),
        threshold=criteria.min_realized_edge_per_fill_usd,
        message=edge_msg,
    ))

    return AcceptanceResult(
        passed=all(r.passed for r in results),
        results=results,
    )


# ---------------------------------------------------------------------------
# HTTP sampler (thin aiohttp wrapper; isolated for test mocking)
# ---------------------------------------------------------------------------


async def sample_dashboard(url: str, *, timeout_sec: float = 2.0) -> dict[str, Any]:
    """GET the dashboard JSON endpoint. Returns the parsed JSON."""
    import aiohttp

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout_sec)
    ) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()


# ---------------------------------------------------------------------------
# Subprocess supervisor
# ---------------------------------------------------------------------------


@dataclass
class SoakConfig:
    hours: int = 72
    dashboard_url: str = "http://127.0.0.1:5055/api/state"
    sample_interval_sec: float = 5.0
    out_dir: Path = field(default_factory=lambda: Path("./soak_output"))
    app_cmd: list[str] = field(default_factory=lambda: [sys.executable, "-m", "blksch.app", "--mode=paper"])
    token_id: str | None = None
    extra_app_args: list[str] = field(default_factory=list)

    def resolved_cmd(self) -> list[str]:
        cmd = list(self.app_cmd)
        if self.token_id:
            cmd.extend(["--market", self.token_id])
        cmd.extend(self.extra_app_args)
        if "--mode=paper" not in cmd and "paper" not in cmd:
            raise RuntimeError(
                "paper_soak refuses to run without --mode=paper. "
                "This harness must NEVER spawn a live-mode child."
            )
        return cmd


def _spawn_app(cmd: list[str]) -> subprocess.Popen:
    log.info("Spawning child: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


async def run_soak(
    cfg: SoakConfig,
    *,
    criteria: AcceptanceCriteria | None = None,
    sampler: Callable[[str], "asyncio.Future[dict[str, Any]]"] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AcceptanceResult:
    """Run a full soak. ``sampler`` and ``clock`` are overridable for tests
    (see ``tests/unit/test_exec_soak_harness.py``).
    """
    criteria = criteria or AcceptanceCriteria()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    sample_fn = sampler or (lambda url: sample_dashboard(url))
    deadline_sec = cfg.hours * 3600.0

    cmd = cfg.resolved_cmd()
    proc = _spawn_app(cmd)

    reports: list[HourlyReport] = []
    hour_start = clock()
    wall_start = time.time()
    aggregate = HourlyAggregate(
        hour_index=0,
        started_at=_iso(wall_start),
        fills_count_at_start=0,
        realized_pnl_at_start=0.0,
        fees_at_start=0.0,
    )
    last_fills = 0
    last_realized = 0.0
    last_fees = 0.0

    try:
        while clock() - hour_start + (len(reports) * 3600.0) < deadline_sec:
            # Collect samples for one hour.
            while clock() - hour_start < 3600.0:
                if clock() - hour_start + (len(reports) * 3600.0) >= deadline_sec:
                    break
                if proc.poll() is not None:
                    log.error("child exited unexpectedly with code %d", proc.returncode)
                    break
                try:
                    snap = await sample_fn(cfg.dashboard_url)
                    aggregate.observe(snap)
                    engine = snap.get("engine") or {}
                    last_fills = int(engine.get("fills_count") or last_fills)
                    pnl = snap.get("pnl") or {}
                    last_realized = float(pnl.get("realized_usd") or last_realized)
                    last_fees = float(pnl.get("fees_usd") or last_fees)
                except Exception as exc:
                    log.warning("sampler error: %s", exc)
                await asyncio.sleep(cfg.sample_interval_sec)

            # Close the hour, write report.
            report = aggregate.finalize()
            reports.append(report)
            _write_hour(cfg.out_dir, report)

            if proc.poll() is not None:
                break
            hour_start = clock()
            wall_start = time.time()
            aggregate = HourlyAggregate(
                hour_index=len(reports),
                started_at=_iso(wall_start),
                fills_count_at_start=last_fills,
                realized_pnl_at_start=last_realized,
                fees_at_start=last_fees,
            )
    finally:
        _shutdown_child(proc)

    result = evaluate(reports, criteria)
    _write_final(cfg.out_dir, reports, result)
    return result


def _iso(ts_float: float) -> str:
    from datetime import UTC, datetime as _dt
    return _dt.fromtimestamp(ts_float, UTC).isoformat()


def _write_hour(out_dir: Path, report: HourlyReport) -> None:
    path = out_dir / f"soak_report_{report.hour_index:03d}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    log.info("wrote %s", path)


def _write_final(
    out_dir: Path,
    reports: list[HourlyReport],
    result: AcceptanceResult,
) -> None:
    path = out_dir / "final_report.json"
    path.write_text(json.dumps({
        "hours_observed": len(reports),
        "acceptance": result.to_dict(),
        "reports": [r.to_dict() for r in reports],
    }, indent=2))
    log.info("wrote %s — passed=%s", path, result.passed)


def _shutdown_child(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    log.info("shutting down child pid=%d", proc.pid)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_plan(cfg: SoakConfig, criteria: AcceptanceCriteria) -> None:
    log.warning("Dry-run (no --i-mean-it). Plan:")
    log.warning("  1. Spawn: %s", " ".join(cfg.resolved_cmd()))
    log.warning("  2. Poll %s every %.1fs", cfg.dashboard_url, cfg.sample_interval_sec)
    log.warning("  3. Aggregate into HourlyReport every hour for %d hour(s)", cfg.hours)
    log.warning("  4. Write soak_report_*.json under %s", cfg.out_dir)
    log.warning("  5. Evaluate against criteria:")
    for f in dataclasses.fields(criteria):
        log.warning("       %s = %s", f.name, getattr(criteria, f.name))
    log.warning("  6. Exit 0 only if all criteria pass.")
    log.warning("Re-run with --i-mean-it to execute.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="72-hour paper-trading soak supervisor")
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:5055/api/state")
    parser.add_argument("--sample-interval-sec", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=Path("./soak_output"))
    parser.add_argument("--token-id", default=None)
    parser.add_argument("--app-cmd", default=None,
                        help="Override child command (space-separated). Must include --mode=paper.")
    parser.add_argument("--i-mean-it", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    # Criteria overrides (rare; defaults match bot.yaml / plan).
    parser.add_argument("--min-quote-uptime-pct", type=float, default=95.0)
    parser.add_argument("--max-inventory-notional-usd", type=float, default=50.0)
    parser.add_argument("--max-pnl-attribution-residual-usd", type=float, default=0.5)
    parser.add_argument("--max-unexpected-kill-switch-events", type=int, default=0)
    parser.add_argument("--min-realized-edge-per-fill-usd", type=float, default=0.0)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = SoakConfig(
        hours=args.hours,
        dashboard_url=args.dashboard_url,
        sample_interval_sec=args.sample_interval_sec,
        out_dir=args.out,
        token_id=args.token_id,
    )
    if args.app_cmd:
        cfg.app_cmd = args.app_cmd.split()

    criteria = AcceptanceCriteria(
        min_hours=args.hours,
        min_quote_uptime_pct=args.min_quote_uptime_pct,
        max_inventory_notional_usd=args.max_inventory_notional_usd,
        max_pnl_attribution_residual_usd=args.max_pnl_attribution_residual_usd,
        max_unexpected_kill_switch_events=args.max_unexpected_kill_switch_events,
        min_realized_edge_per_fill_usd=args.min_realized_edge_per_fill_usd,
    )

    if not args.i_mean_it:
        _print_plan(cfg, criteria)
        return 0

    try:
        cfg.resolved_cmd()  # fail fast if --mode=paper missing
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2

    result = asyncio.run(run_soak(cfg, criteria=criteria))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
