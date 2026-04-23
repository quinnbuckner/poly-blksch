"""Determinism audit for ``scripts/paper_soak.py``.

Stage-1 promotion leans on the soak supervisor producing the same verdict
given the same observations. If ``evaluate()`` or the streaming
``HourlyAggregate`` ever depends on iteration order (dict ordering, set
iteration, ``os.urandom``), two runs of a 72-hour soak will disagree and
we won't know which pass/fail to trust.

Method: run each pure function 100 times per test over the same inputs
and assert **byte-identical** output on every field. We don't just check
the boolean gate — we check every float and int, because a drift of
1e-12 in a metric silently invalidates the acceptance threshold.

Scope rule from the error-detection plan: if this suite finds
nondeterminism, **report it** — do **not** patch ``paper_soak.py`` from
this branch. That fix is a separate approval.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _import_soak():
    path = SCRIPTS_DIR / "paper_soak.py"
    spec = importlib.util.spec_from_file_location(
        "blksch_scripts_paper_soak_det", path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


soak = _import_soak()


# ---------------------------------------------------------------------------
# Sample / report factories (shared with test_exec_soak_harness.py, but
# re-declared here so nondeterminism in *test* infrastructure doesn't cross
# suite boundaries).
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    ts: str,
    quotes: dict | None = None,
    positions: list[dict] | None = None,
    fills_count: int = 0,
    halted: bool = False,
    kill_switches: dict[str, bool] | None = None,
    realized: float = 0.0,
    unrealized: float = 0.0,
    fees: float = 0.0,
) -> dict:
    return {
        "ts": ts,
        "mode": "paper",
        "pnl": {
            "realized_usd": realized,
            "unrealized_usd": unrealized,
            "fees_usd": fees,
            "total_usd": realized + unrealized,
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


def _canonical_sample_stream() -> list[dict]:
    """A deterministic 120-sample stream spanning an hour at 30s cadence.

    Mixes multiple tokens, rising PnL, occasional kill-switch flips, and a
    halt/unhalt cycle — enough variety that any source of nondeterminism
    in the aggregator will light up."""
    stream: list[dict] = []
    tokens = ["0xAAA...", "0xBBB...", "0xCCC..."]
    for i in range(120):
        ts = f"2026-04-23T12:{i // 2:02d}:{(i % 2) * 30:02d}+00:00"
        qty = 5.0 + (i % 11)
        positions = [
            {"token_id": tok, "qty": qty * (1.0 if j == 0 else -0.5),
             "mark": 0.4 + 0.01 * j}
            for j, tok in enumerate(tokens)
        ]
        # The "quotes" dict deliberately has a different insertion order on
        # every sample — if aggregation depends on dict-iteration order,
        # observe() will see a different history on repeat runs.
        quotes = {}
        for tok in (tokens if i % 2 == 0 else reversed(tokens)):
            quotes[tok] = {}
        ks = {
            "feed_gap": (i % 7 == 0),
            "pickoff":  (i % 11 == 3),
            "vol_spike": False,
        }
        halt = (i in (47, 48, 49))
        stream.append(_snapshot(
            ts=ts, quotes=quotes, positions=positions,
            fills_count=i // 3, halted=halt, kill_switches=ks,
            realized=0.001 * i, unrealized=0.0005 * (i // 2),
            fees=0.0001 * i,
        ))
    return stream


def _canonical_hourly_reports() -> list["soak.HourlyReport"]:
    """A small, varied list of HourlyReports for the evaluator audit."""
    return [
        soak.HourlyReport(
            hour_index=h,
            started_at=f"2026-04-23T{h:02d}:00:00+00:00",
            ended_at=f"2026-04-23T{h + 1:02d}:00:00+00:00",
            samples=720,
            quote_uptime_pct=98.5 - 0.03 * h,
            fills_in_hour=50 + h,
            realized_pnl_hour_usd=0.12 * (h + 1),
            realized_pnl_cumulative_usd=0.12 * (h + 1),
            fees_cumulative_usd=0.01 * (h + 1),
            inventory_peak_notional_usd=20.0 + 0.5 * h,
            inventory_peak_qty_by_token={
                # Two tokens with deliberately mixed insertion order
                f"tok-{h % 3}": 10.0 + h,
                f"tok-{(h + 2) % 3}": 8.0 + 0.5 * h,
            },
            kill_switch_events=(h % 5 == 0),
            halt_events=0,
            pnl_attribution_residual_usd=0.0,
        )
        for h in range(72)
    ]


# ---------------------------------------------------------------------------
# Serializer for byte-identical comparisons — deep, key-sorted, canonical.
# ---------------------------------------------------------------------------


def _canonicalize(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canonicalize(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, float):
        return repr(obj)  # exact float representation
    return obj


def _to_bytes(obj) -> bytes:
    return json.dumps(_canonicalize(obj), sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# evaluate() determinism
# ---------------------------------------------------------------------------


N_ITERS = 100


def test_evaluate_is_deterministic_across_100_runs():
    reports = _canonical_hourly_reports()
    criteria = soak.AcceptanceCriteria(min_hours=24)
    baseline_blob = _to_bytes(soak.evaluate(reports, criteria).to_dict())
    baseline_passed = soak.evaluate(reports, criteria).passed

    for i in range(N_ITERS):
        result = soak.evaluate(reports, criteria)
        blob = _to_bytes(result.to_dict())
        assert blob == baseline_blob, (
            f"evaluate() run {i} diverged from baseline.\n"
            f"  baseline={baseline_blob[:200]!r}\n"
            f"  diverged={blob[:200]!r}"
        )
        assert result.passed == baseline_passed, (
            f"evaluate() pass/fail bool flipped on run {i}: "
            f"{result.passed} vs baseline {baseline_passed}"
        )


def test_evaluate_every_criterion_numerical_field_stable():
    """Beyond the aggregate blob, check that each CriterionResult's
    ``observed`` and ``threshold`` fields are exactly reproducible."""
    reports = _canonical_hourly_reports()
    criteria = soak.AcceptanceCriteria(min_hours=24)
    baseline = soak.evaluate(reports, criteria)
    ref_by_name = {r.name: (repr(r.observed), repr(r.threshold), r.passed)
                   for r in baseline.results}

    for i in range(N_ITERS):
        result = soak.evaluate(reports, criteria)
        for r in result.results:
            ref_obs, ref_thr, ref_pass = ref_by_name[r.name]
            assert repr(r.observed) == ref_obs, (
                f"criterion={r.name} observed diverged on run {i}: "
                f"{r.observed!r} vs {ref_obs}"
            )
            assert repr(r.threshold) == ref_thr, (
                f"criterion={r.name} threshold diverged on run {i}"
            )
            assert r.passed == ref_pass, (
                f"criterion={r.name} passed bool flipped on run {i}"
            )


def test_evaluate_empty_reports_is_deterministic():
    criteria = soak.AcceptanceCriteria(min_hours=72)
    baseline_blob = _to_bytes(soak.evaluate([], criteria).to_dict())
    for i in range(N_ITERS):
        blob = _to_bytes(soak.evaluate([], criteria).to_dict())
        assert blob == baseline_blob, f"empty-reports run {i} diverged"


# ---------------------------------------------------------------------------
# HourlyAggregate determinism
# ---------------------------------------------------------------------------


def _run_aggregate(stream: list[dict]) -> "soak.HourlyReport":
    agg = soak.HourlyAggregate(
        hour_index=0, started_at="2026-04-23T12:00:00+00:00",
        fills_count_at_start=0, realized_pnl_at_start=0.0, fees_at_start=0.0,
    )
    for sample in stream:
        agg.observe(sample)
    return agg.finalize()


def test_hourly_aggregate_is_deterministic_across_100_runs():
    stream = _canonical_sample_stream()
    baseline_blob = _to_bytes(_run_aggregate(stream).to_dict())

    for i in range(N_ITERS):
        blob = _to_bytes(_run_aggregate(stream).to_dict())
        assert blob == baseline_blob, (
            f"HourlyAggregate run {i} diverged.\n"
            f"  baseline[:400]={baseline_blob[:400]!r}\n"
            f"  diverged[:400]={blob[:400]!r}"
        )


def test_hourly_aggregate_every_numerical_field_stable():
    """Cross-check the blob comparison with per-field repr equality. Catches
    scenarios where json.dumps canonicalizes away a true drift."""
    stream = _canonical_sample_stream()
    baseline = _run_aggregate(stream)
    ref = {f.name: repr(getattr(baseline, f.name))
           for f in dataclasses.fields(baseline)}

    for i in range(N_ITERS):
        r = _run_aggregate(stream)
        for k, ref_repr in ref.items():
            observed = repr(getattr(r, k))
            assert observed == ref_repr, (
                f"HourlyReport.{k} diverged on run {i}: "
                f"{observed} vs {ref_repr}"
            )


def test_hourly_aggregate_inventory_peak_by_token_is_stable():
    """The per-token-peak dict is a hotspot for dict-ordering drift.
    Verify its key-set and values are bit-identical across runs."""
    stream = _canonical_sample_stream()
    baseline = _run_aggregate(stream)
    baseline_items = sorted(baseline.inventory_peak_qty_by_token.items())

    for i in range(N_ITERS):
        r = _run_aggregate(stream)
        items = sorted(r.inventory_peak_qty_by_token.items())
        assert items == baseline_items, (
            f"inventory_peak_qty_by_token diverged on run {i}: "
            f"{items} vs baseline {baseline_items}"
        )
