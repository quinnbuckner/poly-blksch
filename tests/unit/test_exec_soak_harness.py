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
    monkeypatch.setattr(soak, "_spawn_app", lambda cmd, out_dir: _FakeProc())
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
    # ``hours_observed`` is now wall-time-floor (was len(reports)). The
    # fake-clock test doesn't advance time.time() so hours_observed is 0;
    # the schedule-hours count moved to ``hourly_reports_collected``.
    assert blob["hourly_reports_collected"] >= 1
    assert "acceptance" in blob
    # With monotonically increasing realized minus fees, edge per fill > 0.
    assert result.passed is True


# ---------------------------------------------------------------------------
# Pre-soak audit regressions (BUG-1..3)
#
# Three classes of bugs the supervisor would fail on, each only after hours
# of runtime — exactly the worst time. Pin them with synchronous tests so
# refactors can't silently regress.
# ---------------------------------------------------------------------------


def test_spawn_app_redirects_child_stdout_to_log_file(monkeypatch, tmp_path):
    """BUG-1 regression: child stdout/stderr must NOT use ``subprocess.PIPE``.

    The previous implementation passed ``stdout=PIPE, stderr=STDOUT`` and
    never drained the pipe. macOS pipe buffer is ~64 KB; over 72 h of
    typical child logging it fills and the child blocks on its next
    ``write()`` — a silent supervisor hang. Fix routes child stdout to
    ``out_dir/child.log`` and attaches the file handle to the Popen so
    :func:`_shutdown_child` can close it.
    """
    captured: dict = {}

    class _FakePopen:
        pid = 12345

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr(soak.subprocess, "Popen", _FakePopen)

    proc = soak._spawn_app(["python", "-m", "blksch.app", "--mode=paper"], tmp_path)

    # The pipe-deadlock vector: stdout MUST NOT be subprocess.PIPE.
    assert captured["kwargs"]["stdout"] is not soak.subprocess.PIPE
    # The file we redirected to must be on disk and tracked on the proc.
    log_path = tmp_path / "child.log"
    assert log_path.exists(), "child.log was not created in out_dir"
    log_handle = getattr(proc, "_child_log_file", None)
    assert log_handle is not None, "Popen has no _child_log_file attribute"
    assert log_handle is captured["kwargs"]["stdout"], (
        "the file passed to Popen.stdout must be the same handle attached "
        "to the proc for _shutdown_child to close"
    )
    # And stderr must be merged into stdout (paper_soak's invariant).
    assert captured["kwargs"]["stderr"] is soak.subprocess.STDOUT
    # start_new_session must remain True so killpg signals the child group.
    assert captured["kwargs"]["start_new_session"] is True

    # Cleanup so this test doesn't leak the file handle into others.
    log_handle.close()


def test_shutdown_child_closes_attached_log_file(tmp_path):
    """BUG-1 follow-on: ``_shutdown_child`` must close the log file the
    spawn step attached, even on the early-exit path where the proc is
    already dead. Otherwise we leak file descriptors per soak run (and
    in tests, ``ResourceWarning`` clutters output).
    """

    class _DeadProc:
        pid = 99999

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    proc = _DeadProc()
    log_path = tmp_path / "child.log"
    log_file = open(log_path, "w", buffering=1)
    proc._child_log_file = log_file  # type: ignore[attr-defined]

    soak._shutdown_child(proc)

    assert log_file.closed, "_shutdown_child failed to close attached log file"


def test_shutdown_child_tolerates_missing_log_file_attribute():
    """BUG-1 follow-on, defensive: a test or external caller may construct
    a Popen-like object without the ``_child_log_file`` attribute (the
    happy-path tests do this). ``_shutdown_child`` must not raise
    AttributeError in that case.
    """

    class _NoAttrProc:
        pid = 1
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    # Should not raise.
    soak._shutdown_child(_NoAttrProc())


async def test_run_soak_escalates_after_consecutive_sampler_failures(
    monkeypatch, tmp_path, caplog
):
    """BUG-2 regression: silent sampler failures over 72 h give the operator
    no signal that the dashboard never came up. After
    ``consecutive_sampler_failure_error_threshold`` consecutive exceptions,
    a single ``log.error`` with ``DASHBOARD UNREACHABLE`` must fire so the
    operator can stop the soak instead of waking up to 72 vacuously-failing
    hourly reports.
    """
    fail_count = [0]

    async def always_fail(url):
        fail_count[0] += 1
        raise ConnectionRefusedError("test: dashboard down")

    fake_clock = _FakeClock()

    monkeypatch.setattr(soak, "_spawn_app", lambda cmd, out_dir: _FakeProc())
    monkeypatch.setattr(soak, "_shutdown_child", lambda proc: None)

    real_sleep = asyncio.sleep

    async def instant_sleep(dt):
        fake_clock.tick(dt)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    cfg = soak.SoakConfig(
        hours=1,
        out_dir=tmp_path,
        sample_interval_sec=100.0,  # 36 samples/hour
        consecutive_sampler_failure_error_threshold=5,
    )

    caplog.set_level(logging.WARNING, logger="paper_soak")
    await soak.run_soak(cfg, sampler=always_fail, clock=fake_clock)

    error_msgs = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    unreachable = [m for m in error_msgs if "DASHBOARD UNREACHABLE" in m]
    assert unreachable, (
        f"expected exactly one 'DASHBOARD UNREACHABLE' escalation, got "
        f"{len(unreachable)}; ERROR messages: {error_msgs}"
    )
    # Escalate ONCE — re-firing every cycle would spam the operator.
    assert len(unreachable) == 1, (
        f"expected one-shot escalation, got {len(unreachable)} unreachable "
        f"messages: {unreachable}"
    )
    # The escalation must include diagnostic context for the operator.
    msg = unreachable[0]
    assert "child.log" in msg, "escalation must point operator at child.log"
    assert "pid=" in msg, "escalation must include child pid"
    # And the threshold must have actually been reached, not under-reported.
    assert fail_count[0] >= 5


async def test_run_soak_resets_consecutive_failure_counter_on_success(
    monkeypatch, tmp_path, caplog
):
    """BUG-2 regression: a transient sampler hiccup (4 failures, then a
    success) must NOT escalate to 'DASHBOARD UNREACHABLE'. The counter
    has to reset on every successful sample.
    """
    seq = [
        ConnectionError("transient 1"),
        ConnectionError("transient 2"),
        ConnectionError("transient 3"),
        ConnectionError("transient 4"),  # 4 < threshold (5), no escalation
    ] + [
        _snapshot(quotes={"t": {}}, fills_count=i, realized=0.1 * i, fees=0.005 * i)
        for i in range(1, 60)
    ]
    seq_iter = iter(seq)

    async def hiccup_then_recover(url):
        item = next(seq_iter)
        if isinstance(item, BaseException):
            raise item
        return item

    fake_clock = _FakeClock()
    monkeypatch.setattr(soak, "_spawn_app", lambda cmd, out_dir: _FakeProc())
    monkeypatch.setattr(soak, "_shutdown_child", lambda proc: None)

    real_sleep = asyncio.sleep

    async def instant_sleep(dt):
        fake_clock.tick(dt)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    cfg = soak.SoakConfig(
        hours=1,
        out_dir=tmp_path,
        sample_interval_sec=100.0,
        consecutive_sampler_failure_error_threshold=5,
    )

    caplog.set_level(logging.INFO, logger="paper_soak")
    await soak.run_soak(cfg, sampler=hiccup_then_recover, clock=fake_clock)

    error_msgs = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    unreachable = [m for m in error_msgs if "DASHBOARD UNREACHABLE" in m]
    assert unreachable == [], (
        f"4 failures < 5-threshold then recovery should NOT escalate; got "
        f"unreachable messages: {unreachable}"
    )
    info_msgs = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert any("sampler recovered" in m for m in info_msgs), (
        "recovery must be announced so operator sees the all-clear"
    )


async def test_run_soak_does_not_carry_stale_realized_pnl_into_next_hour(
    monkeypatch, tmp_path
):
    """BUG-3 regression: ``last_realized = float(v or last_realized)``
    treated a legitimate observation of ``0.0`` as "missing" and carried
    the prior non-zero value forward. The next hour's
    ``HourlyAggregate(realized_pnl_at_start=last_realized)`` then started
    with stale data, producing a garbage ``realized_pnl_hour_usd`` for
    that hour.

    Verify by spying on ``HourlyAggregate.__init__`` and checking the
    second hour's ``realized_pnl_at_start`` matches the most recent
    sample's value (0.0), not the stale carryover.
    """
    captured_starts: list[tuple[int, float, float, int]] = []
    real_init = soak.HourlyAggregate.__init__

    def spy_init(self, hour_index, started_at, **kw):
        captured_starts.append((
            hour_index,
            kw["realized_pnl_at_start"],
            kw["fees_at_start"],
            kw["fills_count_at_start"],
        ))
        real_init(
            self, hour_index, started_at,
            fills_count_at_start=kw["fills_count_at_start"],
            realized_pnl_at_start=kw["realized_pnl_at_start"],
            fees_at_start=kw["fees_at_start"],
        )

    monkeypatch.setattr(soak.HourlyAggregate, "__init__", spy_init)

    # Hour 0 should end with the engine reporting realized=0.0 (e.g. after
    # halt-reset). Pre-fix, last_realized would carry over the brief 0.5.
    samples = (
        [_snapshot(realized=0.5, fees=0.05, fills_count=1, quotes={"t": {}})] +
        [_snapshot(realized=0.0, fees=0.0, fills_count=0, quotes={"t": {}})
         for _ in range(80)]
    )
    sample_iter = iter(samples)

    async def fake_sampler(url):
        try:
            return next(sample_iter)
        except StopIteration:
            return samples[-1]

    fake_clock = _FakeClock()
    monkeypatch.setattr(soak, "_spawn_app", lambda cmd, out_dir: _FakeProc())
    monkeypatch.setattr(soak, "_shutdown_child", lambda proc: None)
    real_sleep = asyncio.sleep

    async def instant_sleep(dt):
        fake_clock.tick(dt)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    cfg = soak.SoakConfig(
        hours=2,
        out_dir=tmp_path,
        sample_interval_sec=100.0,
    )
    await soak.run_soak(
        cfg,
        sampler=fake_sampler,
        clock=fake_clock,
        criteria=soak.AcceptanceCriteria(
            min_hours=2, min_quote_uptime_pct=0.0,
            max_inventory_notional_usd=10000.0,
            max_pnl_attribution_residual_usd=10.0,
            max_unexpected_kill_switch_events=999,
            min_realized_edge_per_fill_usd=-1e9,
        ),
    )

    assert len(captured_starts) >= 2, (
        f"expected at least 2 HourlyAggregate constructions, got "
        f"{len(captured_starts)}: {captured_starts}"
    )
    # Hour 0 starts at the defaults (always 0.0).
    assert captured_starts[0] == (0, 0.0, 0.0, 0)
    # Hour 1 starts with the last seen sample's values: realized=0.0,
    # fees=0.0, fills_count=0. Pre-fix, would have been 0.5 / 0.05 / 1.
    hour1 = captured_starts[1]
    assert hour1[0] == 1, f"second construction must be hour_index=1, got {hour1}"
    assert hour1[1] == 0.0, (
        f"BUG-3: hour 1's realized_pnl_at_start carried stale 0.5 instead "
        f"of the latest 0.0 observation; got {hour1[1]}"
    )
    assert hour1[2] == 0.0, (
        f"BUG-3: hour 1's fees_at_start carried stale 0.05 instead of "
        f"the latest 0.0 observation; got {hour1[2]}"
    )
    assert hour1[3] == 0, (
        f"BUG-3: hour 1's fills_count_at_start carried stale 1 instead of "
        f"the latest 0 observation; got {hour1[3]}"
    )


# ---------------------------------------------------------------------------
# Pre-soak audit round 2 — rehearsal bugs surfaced by the planning window
# against 97042c8 (stdout-pipe / sampler-recovery / hour-carryover). Each
# bug here fires only under real-run conditions so pin with focused tests.
# ---------------------------------------------------------------------------


def test_spawn_uses_tokens_not_market(tmp_path):
    """ROUND2 BUG-1 regression: ``resolved_cmd()`` must emit ``--tokens``,
    not ``--market``. app.py defines ``--markets`` (YAML path) and
    ``--tokens`` (comma-separated token_ids); argparse prefix-matches
    ``--market`` onto ``--markets``, so the child tried to
    ``open('<token_id>')`` and died with FileNotFoundError in ~1s, making
    every ``--token-id`` invocation unusable.
    """
    cfg = soak.SoakConfig(
        hours=1,
        out_dir=tmp_path,
        token_id="0xaabbccdd",
    )
    cmd = cfg.resolved_cmd()
    assert "--tokens" in cmd, (
        f"resolved_cmd must emit --tokens (app.py's flag); got {cmd}"
    )
    # The stale spelling must NOT be emitted — it prefix-matches onto
    # --markets and silently breaks.
    assert "--market" not in cmd, (
        f"--market prefix-matches onto --markets (YAML path flag); "
        f"resolved_cmd must NOT emit it. Got {cmd}"
    )
    # The token value must follow the flag positionally.
    i = cmd.index("--tokens")
    assert cmd[i + 1] == "0xaabbccdd"


async def test_artifacts_written_on_cancellation(monkeypatch, tmp_path):
    """ROUND2 BUG-2 regression: ``evaluate()`` and ``_write_final()`` used
    to live AFTER the try/finally. An operator SIGINT during hour 71 of
    a 72 h run raised KeyboardInterrupt past them and lost the verdict.

    Inside ``asyncio.run``, SIGINT manifests as ``CancelledError`` inside
    the running coroutine — trigger that directly via a sampler that
    raises it to simulate the interrupt without touching real signals
    (keeps the test deterministic across CI environments).
    """
    fake_clock = _FakeClock()

    cancelled_after = [0]

    async def sampler_that_cancels_mid_run(url):
        cancelled_after[0] += 1
        if cancelled_after[0] >= 3:
            raise asyncio.CancelledError("simulated operator SIGINT")
        return _snapshot(
            quotes={"t": {}}, fills_count=cancelled_after[0],
            realized=0.1 * cancelled_after[0], fees=0.005 * cancelled_after[0],
        )

    monkeypatch.setattr(soak, "_spawn_app", lambda cmd, out_dir: _FakeProc())
    monkeypatch.setattr(soak, "_shutdown_child", lambda proc: None)

    real_sleep = asyncio.sleep

    async def instant_sleep(dt):
        fake_clock.tick(dt)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    cfg = soak.SoakConfig(
        hours=1,
        out_dir=tmp_path,
        sample_interval_sec=100.0,
    )

    with pytest.raises(asyncio.CancelledError):
        await soak.run_soak(cfg, sampler=sampler_that_cancels_mid_run,
                            clock=fake_clock)

    # The finally block MUST have persisted the verdict even though the
    # CancelledError propagated out of run_soak.
    final = tmp_path / "final_report.json"
    assert final.exists(), (
        "BUG-2: final_report.json must be written in the finally block so "
        "SIGINT mid-run never loses the verdict"
    )
    import json as _json
    blob = _json.loads(final.read_text())
    assert "acceptance" in blob, "partial verdict must include acceptance dict"
    # Acceptance on a partial run: passed=False is expected (inventory/
    # uptime/duration criteria not met), but that is not what this test
    # pins — it pins that the file exists at all.
    assert "passed" in blob["acceptance"]


def test_exit_code_reflects_acceptance_passed(monkeypatch, tmp_path):
    """ROUND2 BUG-3 regression: supervisor must exit non-zero when
    ``acceptance.passed`` is False. Cron / CI that trust the process
    exit code rely on this.

    Stub run_soak to return a known-failing AcceptanceResult and assert
    main() returns 1. Also stub the happy path to return passed=True and
    assert main() returns 0. The stub pattern sidesteps network and
    keeps the test under a millisecond.

    Kept SYNC (not async) because main() drives its own ``asyncio.run``
    — an outer event loop (async test) would collide with that.
    """
    fail_result = soak.AcceptanceResult(
        passed=False,
        results=[soak.CriterionResult(
            name="min_hours", passed=False, observed=0, threshold=72,
            message="stubbed failure",
        )],
    )
    ok_result = soak.AcceptanceResult(
        passed=True,
        results=[soak.CriterionResult(
            name="min_hours", passed=True, observed=72, threshold=72,
            message="stubbed success",
        )],
    )

    async def _fail_run(cfg, *, criteria=None):
        # Simulate run_soak's finally: write the JSON, return the result.
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        soak._write_final(cfg.out_dir, [], fail_result,
                          elapsed_wall_time_sec=0.0)
        return fail_result

    async def _ok_run(cfg, *, criteria=None):
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        soak._write_final(cfg.out_dir, [], ok_result,
                          elapsed_wall_time_sec=0.0)
        return ok_result

    # FAIL case → exit 1
    monkeypatch.setattr(soak, "run_soak", _fail_run)
    rc = soak.main([
        "--i-mean-it", "--hours", "1",
        "--out", str(tmp_path / "fail"),
    ])
    assert rc == 1, (
        f"BUG-3: main() must exit 1 when acceptance.passed=False; got {rc}"
    )

    # PASS case → exit 0
    monkeypatch.setattr(soak, "run_soak", _ok_run)
    rc = soak.main([
        "--i-mean-it", "--hours", "1",
        "--out", str(tmp_path / "ok"),
    ])
    assert rc == 0, f"main() must exit 0 when passed=True; got {rc}"


async def test_hours_observed_reflects_wall_time(monkeypatch, tmp_path):
    """ROUND2 BUG-4 regression: ``hours_observed`` must be wall-time
    floor, not ``len(reports)``. A 1-second crashed soak used to report
    hours_observed=1 (because the inner break still finalized and
    appended one HourlyReport); operators interpret that field as
    "hours the soak actually ran" and the inflated count masked crash-
    fast scenarios. The scheduled-hour count moved to a new field
    ``hourly_reports_collected`` for callers that want that semantic.
    """
    # Stub time.time so the test is deterministic — first call (wall_run_start)
    # returns 1000.0; subsequent calls (including the elapsed computation
    # in the finally) also return 1000.0, so elapsed ≈ 0 and
    # hours_observed == 0 regardless of how many reports are appended.
    monkeypatch.setattr(soak.time, "time", lambda: 1000.0)

    samples = [
        _snapshot(quotes={"t": {}}, fills_count=i, realized=0.1 * i,
                  fees=0.005 * i)
        for i in range(1, 100)
    ]
    sample_iter = iter(samples)

    async def fake_sampler(url):
        try:
            return next(sample_iter)
        except StopIteration:
            return samples[-1]

    fake_clock = _FakeClock()
    monkeypatch.setattr(soak, "_spawn_app", lambda cmd, out_dir: _FakeProc())
    monkeypatch.setattr(soak, "_shutdown_child", lambda proc: None)

    real_sleep = asyncio.sleep

    async def instant_sleep(dt):
        fake_clock.tick(dt)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    cfg = soak.SoakConfig(
        hours=2,
        out_dir=tmp_path,
        sample_interval_sec=100.0,
    )
    await soak.run_soak(cfg, sampler=fake_sampler, clock=fake_clock)

    import json as _json
    blob = _json.loads((tmp_path / "final_report.json").read_text())
    # 2 reports accumulated via the fake-clock loop...
    assert blob["hourly_reports_collected"] >= 1
    # ...but hours_observed is 0 because NO wall time elapsed.
    assert blob["hours_observed"] == 0, (
        f"BUG-4: hours_observed must be floor(elapsed_wall_time/3600); "
        f"got {blob['hours_observed']} for a zero-wall-time fake run"
    )
    # And elapsed_wall_time_sec is a new field; check shape.
    assert "elapsed_wall_time_sec" in blob
    assert blob["elapsed_wall_time_sec"] == 0.0


def test_hours_accepts_float_via_cli(tmp_path):
    """ROUND2 enhancement: ``--hours`` accepts fractional values so
    rehearsals can run ``--hours 0.05`` (≈3 min). Previously typed as
    int so argparse rejected ``0.05`` before the script even started.
    """
    # Dry-run mode — no network, no subprocess, just argparse round-trip.
    rc = soak.main([
        "--hours", "0.05",
        "--out", str(tmp_path),
    ])
    assert rc == 0, f"dry-run with float --hours must succeed; got rc={rc}"

    # Structural: ensure the CLI is actually typed float now. If someone
    # flips it back to int, ``--hours 0.05`` errors out with argparse
    # exit code 2 (SystemExit) before returning.
    import argparse as _ap
    try:
        rc = soak.main(["--hours", "0.05", "--out", str(tmp_path / "x")])
    except SystemExit as exc:
        pytest.fail(
            f"--hours must accept float values; argparse rejected 0.05 "
            f"with SystemExit({exc.code}). Did the type revert to int?"
        )


def test_child_rc_zero_logs_at_info_not_error(monkeypatch, tmp_path, caplog):
    """ROUND2 BUG-5 regression: a clean child exit (rc=0) was being
    logged at ERROR level as 'child exited unexpectedly'. rc=0 means
    the child honored SIGTERM or hit ``--max-runtime-sec`` — a NORMAL
    exit path. Downgrade to INFO, keep ERROR for non-zero.
    """

    class _CleanExitProc:
        pid = 1234
        returncode = 0
        _polled = 0

        def poll(self):
            # First poll returns None (alive) so the loop body executes;
            # subsequent polls return 0 so the inner break fires once.
            self._polled += 1
            return None if self._polled <= 1 else 0

        def wait(self, timeout=None):
            return 0

    fake_clock = _FakeClock()
    monkeypatch.setattr(soak, "_spawn_app", lambda cmd, out_dir: _CleanExitProc())
    monkeypatch.setattr(soak, "_shutdown_child", lambda proc: None)

    samples = [_snapshot() for _ in range(5)]
    it = iter(samples)

    async def fake_sampler(url):
        try:
            return next(it)
        except StopIteration:
            return samples[-1]

    real_sleep = asyncio.sleep

    async def instant_sleep(dt):
        fake_clock.tick(dt)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    cfg = soak.SoakConfig(hours=1, out_dir=tmp_path, sample_interval_sec=100.0)

    caplog.set_level(logging.INFO, logger="paper_soak")

    async def _go():
        await soak.run_soak(cfg, sampler=fake_sampler, clock=fake_clock)

    asyncio.run(_go())

    error_msgs = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    info_msgs = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]

    # The "unexpectedly" phrasing is reserved for non-zero rc now.
    unexpected_errors = [m for m in error_msgs if "exited unexpectedly" in m]
    assert unexpected_errors == [], (
        f"BUG-5: rc=0 must NOT log at ERROR as 'exited unexpectedly'; "
        f"got: {unexpected_errors}"
    )
    # And the INFO message for the clean-exit path must have fired.
    clean_infos = [m for m in info_msgs if "exited cleanly" in m]
    assert clean_infos, (
        f"BUG-5: rc=0 child exit must be announced at INFO as 'exited "
        f"cleanly'; got INFO messages: {info_msgs}"
    )


# ---------------------------------------------------------------------------
# Stage-2 prep small additions (track-c-stage2-prep-small)
# ---------------------------------------------------------------------------


def test_paper_soak_min_hours_flag(monkeypatch, tmp_path):
    """``--min-hours FLOAT`` decouples the acceptance gate's duration
    floor from ``--hours``. A 36 h residential data-gather should
    accept at 36 h instead of demanding 72 h via the prior coupling
    (where ``criteria.min_hours = args.hours``).

    Stub ``run_soak`` to capture the criteria handed to it, then
    assert the threading is correct. This is a CLI-wiring regression,
    not a behavioral change to evaluate() itself.
    """
    captured: dict = {}

    async def _stub_run(cfg, *, criteria=None):
        captured["criteria"] = criteria
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        result = soak.AcceptanceResult(passed=True, results=[])
        soak._write_final(cfg.out_dir, [], result, elapsed_wall_time_sec=0.0)
        return result

    monkeypatch.setattr(soak, "run_soak", _stub_run)

    # Default: --min-hours unspecified → 72.0
    rc = soak.main([
        "--i-mean-it", "--hours", "1",
        "--out", str(tmp_path / "default"),
    ])
    assert rc == 0
    assert captured["criteria"].min_hours == 72.0, (
        f"default --min-hours must be 72.0; got {captured['criteria'].min_hours}"
    )

    # Explicit override decoupled from --hours.
    rc = soak.main([
        "--i-mean-it", "--hours", "36", "--min-hours", "36",
        "--out", str(tmp_path / "thirty-six"),
    ])
    assert rc == 0
    assert captured["criteria"].min_hours == 36.0, (
        f"--min-hours 36 must thread; got {captured['criteria'].min_hours}"
    )

    # Sub-hour fractional (rehearsal use case).
    rc = soak.main([
        "--i-mean-it", "--hours", "0.05", "--min-hours", "0.02",
        "--out", str(tmp_path / "rehearsal"),
    ])
    assert rc == 0
    assert captured["criteria"].min_hours == 0.02

    # And: --hours and --min-hours are independent — passing only --hours
    # must NOT change the min_hours default away from 72.0.
    rc = soak.main([
        "--i-mean-it", "--hours", "12",
        "--out", str(tmp_path / "hours-only"),
    ])
    assert captured["criteria"].min_hours == 72.0, (
        "--hours must NOT silently retune --min-hours; the decoupling "
        "is the whole point of the new flag"
    )


def test_paper_soak_soak_output_dir_flag(monkeypatch, tmp_path):
    """``--soak-output-dir PATH`` (alias of ``--out``) routes every
    artifact (final_report.json, soak_report_*.json, child.log, the
    runtime-resolved cfg.out_dir) into the override directory so
    parallel / sequential soaks don't clobber each other.

    Pin both spellings — the user-facing rehearsal command uses the
    long form; the short ``--out`` form must still work for callers
    who already wrote scripts against it.
    """
    captured: dict = {}

    async def _stub_run(cfg, *, criteria=None):
        captured["out_dir"] = cfg.out_dir
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        # Drop a soak_report_000.json so the artifact-landing assertion
        # matches the production layout.
        result = soak.AcceptanceResult(passed=True, results=[])
        report = soak.HourlyReport(
            hour_index=0, started_at="t0", ended_at="t1", samples=1,
            quote_uptime_pct=0.0, fills_in_hour=0,
            realized_pnl_hour_usd=0.0, realized_pnl_cumulative_usd=0.0,
            fees_cumulative_usd=0.0, inventory_peak_notional_usd=0.0,
            inventory_peak_qty_by_token={}, kill_switch_events=0,
            halt_events=0, pnl_attribution_residual_usd=0.0,
        )
        soak._write_hour(cfg.out_dir, report)
        soak._write_final(cfg.out_dir, [report], result, elapsed_wall_time_sec=0.0)
        return result

    monkeypatch.setattr(soak, "run_soak", _stub_run)

    # Long form (--soak-output-dir).
    long_form_dir = tmp_path / "long-form"
    rc = soak.main([
        "--i-mean-it", "--hours", "1",
        "--soak-output-dir", str(long_form_dir),
    ])
    assert rc == 0
    assert captured["out_dir"] == long_form_dir, (
        f"--soak-output-dir must thread to cfg.out_dir; "
        f"got {captured['out_dir']} expected {long_form_dir}"
    )
    assert (long_form_dir / "final_report.json").exists()
    assert (long_form_dir / "soak_report_000.json").exists()

    # Short form (--out) still works — backward compat for any operator
    # script written before the rename.
    short_form_dir = tmp_path / "short-form"
    rc = soak.main([
        "--i-mean-it", "--hours", "1",
        "--out", str(short_form_dir),
    ])
    assert rc == 0
    assert captured["out_dir"] == short_form_dir
    assert (short_form_dir / "final_report.json").exists()

    # Default still ./soak_output (unchanged).
    monkeypatch.chdir(tmp_path)
    rc = soak.main([
        "--i-mean-it", "--hours", "1",
    ])
    assert captured["out_dir"] == Path("./soak_output")
