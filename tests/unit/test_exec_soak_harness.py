"""Unit tests for scripts/paper_soak.py — the 72h paper-trading supervisor.

These tests pin the acceptance-criterion evaluator and the streaming
aggregator so nobody silently weakens the Stage-1 → Stage-2 gate. No
subprocesses, no network — pure data flow.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _import_soak():
    path = SCRIPTS_DIR / "paper_soak.py"
    spec = importlib.util.spec_from_file_location("blksch_scripts_paper_soak", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


soak = _import_soak()


# ---------------------------------------------------------------------------
# snapshot fixture factory
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    ts: str = "2026-04-22T12:00:00+00:00",
    quotes: dict | None = None,
    positions: list[dict] | None = None,
    fills_count: int = 0,
    halted: bool = False,
    kill_switches: dict[str, bool] | None = None,
    realized: float = 0.0,
    unrealized: float = 0.0,
    fees: float = 0.0,
    total_override: float | None = None,
) -> dict:
    total = total_override if total_override is not None else (realized + unrealized)
    return {
        "ts": ts,
        "mode": "paper",
        "pnl": {
            "realized_usd": realized,
            "unrealized_usd": unrealized,
            "fees_usd": fees,
            "total_usd": total,
        },
        "positions": positions or [],
        "quotes": quotes or {},
        "surface": {},
        "engine": {
            "halted": halted,
            "halt_reason": None if not halted else "injected",
            "last_book_ts": ts,
            "last_trade_ts": ts,
            "fills_count": fills_count,
        },
        "kill_switches": kill_switches or {},
        "recent_fills": [],
    }


# ---------------------------------------------------------------------------
# HourlyAggregate
# ---------------------------------------------------------------------------


def test_aggregate_uptime_and_fills():
    agg = soak.HourlyAggregate(
        hour_index=0, started_at="t0",
        fills_count_at_start=0, realized_pnl_at_start=0.0, fees_at_start=0.0,
    )
    # 10 samples, 7 with quotes; fills go 0 → 3; realized rises.
    for i in range(10):
        agg.observe(_snapshot(
            quotes={"t": {}} if i < 7 else {},
            fills_count=3 if i >= 5 else 0,
            realized=0.02 * (i + 1),
            fees=0.001 * (i + 1),
        ))
    r = agg.finalize()
    assert r.samples == 10
    assert r.quote_uptime_pct == pytest.approx(70.0)
    assert r.fills_in_hour == 3
    assert r.realized_pnl_cumulative_usd == pytest.approx(0.20)
    assert r.fees_cumulative_usd == pytest.approx(0.010)


def test_aggregate_tracks_peak_inventory_and_notional():
    agg = soak.HourlyAggregate(
        hour_index=0, started_at="t0",
        fills_count_at_start=0, realized_pnl_at_start=0.0, fees_at_start=0.0,
    )
    agg.observe(_snapshot(positions=[{"token_id": "a", "qty":  5.0, "mark": 0.60}]))  # 3.00
    agg.observe(_snapshot(positions=[{"token_id": "a", "qty": -8.0, "mark": 0.55}]))  # 4.40
    agg.observe(_snapshot(positions=[{"token_id": "b", "qty": 10.0, "mark": 0.50}]))  # 5.00
    r = agg.finalize()
    assert r.inventory_peak_notional_usd == pytest.approx(5.0)
    assert r.inventory_peak_qty_by_token == {"a": 8.0, "b": 10.0}


def test_aggregate_counts_kill_switch_edges_only():
    agg = soak.HourlyAggregate(
        hour_index=0, started_at="t0",
        fills_count_at_start=0, realized_pnl_at_start=0.0, fees_at_start=0.0,
    )
    # false, true, true, false, true  → two rising edges
    for val in (False, True, True, False, True):
        agg.observe(_snapshot(kill_switches={"feed_gap": val}))
    r = agg.finalize()
    assert r.kill_switch_events == 2


def test_aggregate_counts_halt_transitions():
    agg = soak.HourlyAggregate(
        hour_index=0, started_at="t0",
        fills_count_at_start=0, realized_pnl_at_start=0.0, fees_at_start=0.0,
    )
    for halted in (False, False, True, True, False, True):
        agg.observe(_snapshot(halted=halted))
    r = agg.finalize()
    assert r.halt_events == 2


def test_aggregate_detects_pnl_attribution_residual():
    agg = soak.HourlyAggregate(
        hour_index=0, started_at="t0",
        fills_count_at_start=0, realized_pnl_at_start=0.0, fees_at_start=0.0,
    )
    # total_override breaks the realized+unrealized == total invariant
    agg.observe(_snapshot(realized=1.0, unrealized=2.0, total_override=3.4))
    r = agg.finalize()
    assert r.pnl_attribution_residual_usd == pytest.approx(0.4)


def test_aggregate_zero_samples_reports_zero_uptime():
    agg = soak.HourlyAggregate(
        hour_index=0, started_at="t0",
        fills_count_at_start=0, realized_pnl_at_start=0.0, fees_at_start=0.0,
    )
    r = agg.finalize()
    assert r.samples == 0
    assert r.quote_uptime_pct == 0.0


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def _hour(
    *,
    idx: int = 0,
    uptime: float = 99.0,
    fills: int = 10,
    realized_cum: float = 1.0,
    fees_cum: float = 0.1,
    inventory: float = 25.0,
    residual: float = 0.0,
    kill: int = 0,
    halts: int = 0,
) -> "soak.HourlyReport":
    return soak.HourlyReport(
        hour_index=idx,
        started_at="t0", ended_at="t1",
        samples=720,
        quote_uptime_pct=uptime,
        fills_in_hour=fills,
        realized_pnl_hour_usd=realized_cum,
        realized_pnl_cumulative_usd=realized_cum,
        fees_cumulative_usd=fees_cum,
        inventory_peak_notional_usd=inventory,
        inventory_peak_qty_by_token={},
        kill_switch_events=kill,
        halt_events=halts,
        pnl_attribution_residual_usd=residual,
    )


def test_evaluate_all_green():
    criteria = soak.AcceptanceCriteria(min_hours=3)
    reports = [_hour(idx=i, realized_cum=i + 1.0, fees_cum=0.1 * (i + 1)) for i in range(3)]
    res = soak.evaluate(reports, criteria)
    assert res.passed is True
    names = {r.name for r in res.results if r.passed}
    assert "min_hours" in names
    assert "min_realized_edge_per_fill_usd" in names


def test_evaluate_fails_on_short_run():
    criteria = soak.AcceptanceCriteria(min_hours=72)
    reports = [_hour(idx=i) for i in range(5)]
    res = soak.evaluate(reports, criteria)
    assert res.passed is False
    fail = next(r for r in res.results if r.name == "min_hours")
    assert fail.passed is False
    assert fail.observed == 5


def test_evaluate_fails_on_inventory_breach():
    criteria = soak.AcceptanceCriteria(min_hours=1, max_inventory_notional_usd=50.0)
    reports = [_hour(inventory=40.0), _hour(idx=1, inventory=75.0)]
    res = soak.evaluate(reports, criteria)
    fail = next(r for r in res.results if r.name == "max_inventory_notional_usd")
    assert fail.passed is False
    assert fail.observed == 75.0


def test_evaluate_fails_on_pnl_residual():
    criteria = soak.AcceptanceCriteria(min_hours=1, max_pnl_attribution_residual_usd=0.5)
    reports = [_hour(residual=0.3), _hour(idx=1, residual=0.9)]
    res = soak.evaluate(reports, criteria)
    fail = next(r for r in res.results if r.name == "max_pnl_attribution_residual_usd")
    assert fail.passed is False
    assert fail.observed == 0.9


def test_evaluate_fails_on_kill_switch_trips():
    criteria = soak.AcceptanceCriteria(min_hours=1, max_unexpected_kill_switch_events=0)
    reports = [_hour(kill=1), _hour(idx=1, halts=2)]
    res = soak.evaluate(reports, criteria)
    fail = next(r for r in res.results if r.name == "max_unexpected_kill_switch_events")
    assert fail.passed is False
    assert fail.observed == 3


def test_evaluate_fails_when_no_fills_observed():
    """Zero fills over 72 hours means the bot wasn't actually trading —
    never claim edge > 0 out of vacuity."""
    criteria = soak.AcceptanceCriteria(min_hours=1)
    reports = [_hour(fills=0, realized_cum=0.0, fees_cum=0.0)]
    res = soak.evaluate(reports, criteria)
    edge = next(r for r in res.results if r.name == "min_realized_edge_per_fill_usd")
    assert edge.passed is False
    assert "no fills" in edge.message


def test_evaluate_fails_on_negative_edge():
    criteria = soak.AcceptanceCriteria(min_hours=1, min_realized_edge_per_fill_usd=0.0)
    # realized 0.5, fees 0.8 → edge per fill = (0.5 - 0.8)/10 = -0.03
    reports = [_hour(fills=10, realized_cum=0.5, fees_cum=0.8)]
    res = soak.evaluate(reports, criteria)
    edge = next(r for r in res.results if r.name == "min_realized_edge_per_fill_usd")
    assert edge.passed is False
    assert edge.observed < 0


def test_evaluate_returns_failure_when_reports_empty():
    res = soak.evaluate([], soak.AcceptanceCriteria(min_hours=1))
    assert res.passed is False
    # Every criterion still appears in the output.
    names = {r.name for r in res.results}
    assert {
        "min_hours", "min_quote_uptime_pct", "max_inventory_notional_usd",
        "max_pnl_attribution_residual_usd", "max_unexpected_kill_switch_events",
        "min_realized_edge_per_fill_usd",
    } <= names


def test_acceptance_result_to_dict_is_json_friendly():
    import json
    res = soak.evaluate([_hour()], soak.AcceptanceCriteria(min_hours=1))
    blob = json.dumps(res.to_dict())  # must not raise
    assert "passed" in blob
    assert "criteria" in blob


# ---------------------------------------------------------------------------
# SoakConfig — refuse non-paper children
# ---------------------------------------------------------------------------


def test_soak_config_refuses_live_app_cmd(tmp_path):
    cfg = soak.SoakConfig(
        hours=1,
        out_dir=tmp_path,
        app_cmd=["python", "-m", "blksch.app", "--mode=live"],
    )
    with pytest.raises(RuntimeError, match="--mode=paper"):
        cfg.resolved_cmd()


def test_soak_config_accepts_paper_via_token_form(tmp_path):
    cfg = soak.SoakConfig(
        hours=1,
        out_dir=tmp_path,
        app_cmd=["python", "-m", "blksch.app", "paper"],
    )
    # The bare word "paper" is also acceptable — some CLIs use it as a subcommand.
    assert cfg.resolved_cmd()[-1] == "paper"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_dry_run_prints_plan(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    rc = soak.main(["--hours", "2", "--out", str(tmp_path)])
    assert rc == 0
    assert "Dry-run" in caplog.text
    assert "--i-mean-it" in caplog.text
    assert "Evaluate against criteria" in caplog.text


# ---------------------------------------------------------------------------
# run_soak with fake sampler + fake clock — no subprocess, no network
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> None:
        self.t += dt


class _FakeProc:
    returncode: int | None = None
    pid = 99999

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


async def test_run_soak_happy_path(monkeypatch, tmp_path):
    samples = [
        _snapshot(
            quotes={"t": {}}, positions=[{"token_id": "t", "qty": 5, "mark": 0.5}],
            fills_count=i, realized=0.1 * i, fees=0.005 * i,
        )
        for i in range(1, 50)
    ]
    sample_iter = iter(samples)

    async def fake_sampler(url):
        return next(sample_iter)

    fake_clock = _FakeClock()

    # Replace subprocess spawn + shutdown with a no-op.
    monkeypatch.setattr(soak, "_spawn_app", lambda cmd: _FakeProc())
    monkeypatch.setattr(soak, "_shutdown_child", lambda proc: None)

    # Make asyncio.sleep advance our fake clock instead of wall time.
    real_sleep = asyncio.sleep

    async def instant_sleep(dt):
        fake_clock.tick(dt)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    cfg = soak.SoakConfig(
        hours=1,
        out_dir=tmp_path,
        sample_interval_sec=100.0,  # 36 samples per hour → terminates quickly
    )
    result = await soak.run_soak(
        cfg,
        criteria=soak.AcceptanceCriteria(
            min_hours=1,
            min_quote_uptime_pct=50.0,
            max_inventory_notional_usd=1000.0,
            max_pnl_attribution_residual_usd=10.0,
            max_unexpected_kill_switch_events=0,
            min_realized_edge_per_fill_usd=0.0,
        ),
        sampler=fake_sampler,
        clock=fake_clock,
    )

    # Final report file exists and includes acceptance.
    final = tmp_path / "final_report.json"
    assert final.exists()
    import json
    blob = json.loads(final.read_text())
    assert blob["hours_observed"] >= 1
    assert "acceptance" in blob
    # With monotonically increasing realized minus fees, edge per fill > 0.
    assert result.passed is True
