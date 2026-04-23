"""Unit tests for the ledger-drift drill (scripts/reconcile_drill.py) and
the reconciliation primitive it exercises (ledger.reconcile_against_ledger).

Every corruption mode gets its own test — the drill is a Stage-2 promotion
gate and must stay honest even as the accounting code evolves.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from blksch.exec.ledger import (
    Ledger,
    ReconciliationDiscrepancy,
    reconcile_against_ledger,
)
from blksch.schemas import Fill, OrderSide

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _import_drill():
    path = SCRIPTS_DIR / "reconcile_drill.py"
    spec = importlib.util.spec_from_file_location("blksch_scripts_reconcile_drill", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


drill = _import_drill()


# ---------------------------------------------------------------------------
# Baseline: clean fill history reconciles.
# ---------------------------------------------------------------------------


def test_clean_ledger_reconciles(ledger_with_fills):
    ledger, _ = ledger_with_fills
    report = reconcile_against_ledger(ledger)
    assert report.passed is True
    assert report.discrepancies == []
    assert drill.TOKEN in report.tokens_checked


# ---------------------------------------------------------------------------
# Per-corruption-mode drills.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(drill.ALL_MODES))
def test_each_corruption_mode_is_detected(mode: str):
    result = drill.run_drill(mode, target_cid="cid-003")
    assert result.detected, f"mode={mode} slipped past reconcile()"
    kinds = {d["kind"] for d in result.report_dict["discrepancies"]}
    assert kinds, "drift report had no discrepancies despite failure flag"


def test_wrong_price_flags_pnl_drift(ledger_with_fills):
    ledger, _ = ledger_with_fills
    drill._corrupt(ledger, "wrong_price", target_cid="cid-003")
    report = reconcile_against_ledger(ledger)
    kinds = {d.kind for d in report.discrepancies}
    assert "pnl_drift" in kinds
    # Changing price alone does not change qty.
    assert "qty_drift" not in kinds


def test_wrong_size_flags_both_pnl_and_qty_drift(ledger_with_fills):
    ledger, _ = ledger_with_fills
    drill._corrupt(ledger, "wrong_size", target_cid="cid-003")
    report = reconcile_against_ledger(ledger)
    kinds = {d.kind for d in report.discrepancies}
    assert "pnl_drift" in kinds
    assert "qty_drift" in kinds


def test_duplicate_client_id_flags_dup_and_drift(ledger_with_fills):
    ledger, _ = ledger_with_fills
    drill._corrupt(ledger, "duplicate_client_id", target_cid="cid-003")
    report = reconcile_against_ledger(ledger)
    kinds = {d.kind for d in report.discrepancies}
    assert "duplicate_client_id" in kinds
    # Applying the same fill twice in a replay produces different qty & PnL.
    assert "pnl_drift" in kinds
    assert "qty_drift" in kinds


def test_corruption_on_unknown_cid_raises(ledger_with_fills):
    ledger, _ = ledger_with_fills
    with pytest.raises(RuntimeError, match="no fill"):
        drill._corrupt(ledger, "wrong_price", target_cid="does-not-exist")


# ---------------------------------------------------------------------------
# Venue comparison (reconcile_against_ledger)
# ---------------------------------------------------------------------------


def test_venue_missing_fill_is_flagged(ledger_with_fills):
    """Ledger has a fill the venue doesn't know about — silent duplicate or
    local stuck-order-then-fake-fill bug."""
    ledger, fills = ledger_with_fills
    venue = {drill.TOKEN: [f for f in fills if f.order_client_id != "cid-005"]}
    # Ledger has cid-005; venue doesn't.
    report = reconcile_against_ledger(ledger, venue_fills=venue)
    kinds = {d.kind for d in report.discrepancies}
    assert "venue_missing" in kinds


def test_ledger_missing_fill_is_flagged(ledger_with_fills):
    """Venue sent a fill we never recorded — websocket drop or missed
    callback. Stage-2 kill switch should fire."""
    ledger, fills = ledger_with_fills
    extra = Fill(
        order_client_id="cid-venue-only",
        order_venue_id="venue-xyz",
        token_id=drill.TOKEN,
        side=OrderSide.BUY,
        price=0.42, size=7,
        fee_usd=0.01,
        ts=datetime(2026, 4, 22, 12, 10, 0, tzinfo=UTC),
    )
    report = reconcile_against_ledger(ledger, venue_fills={drill.TOKEN: [*fills, extra]})
    kinds = {d.kind for d in report.discrepancies}
    assert "ledger_missing" in kinds


def test_field_mismatch_is_flagged(ledger_with_fills):
    ledger, fills = ledger_with_fills
    # Same client_id, different price.
    tampered = [
        f.model_copy(update={"price": f.price + 0.05}) if f.order_client_id == "cid-002" else f
        for f in fills
    ]
    report = reconcile_against_ledger(ledger, venue_fills={drill.TOKEN: tampered})
    kinds = {d.kind for d in report.discrepancies}
    assert "field_mismatch" in kinds


def test_venue_fills_all_agree_no_drift(ledger_with_fills):
    ledger, fills = ledger_with_fills
    report = reconcile_against_ledger(ledger, venue_fills={drill.TOKEN: fills})
    assert report.passed is True


def test_tolerance_absorbs_tiny_float_noise():
    """Real trades have sub-cent rounding. A tolerance of 1e-6 USD should
    keep clean state passing even when reconcile vs stored differ in the
    15th decimal place."""
    ledger = Ledger.in_memory()
    for f in drill.default_fill_sequence():
        ledger.apply_fill(f)
    # Nudge realized_pnl_usd by < tolerance directly in sqlite.
    with ledger._lock:
        ledger._conn.execute(
            "UPDATE positions SET realized_pnl_usd = realized_pnl_usd + 1e-9",
        )
    report = reconcile_against_ledger(ledger, tolerance_usd=1e-6)
    assert report.passed is True


def test_tolerance_above_drift_flags_it():
    ledger = Ledger.in_memory()
    for f in drill.default_fill_sequence():
        ledger.apply_fill(f)
    with ledger._lock:
        ledger._conn.execute(
            "UPDATE positions SET realized_pnl_usd = realized_pnl_usd + 0.05",
        )
    report = reconcile_against_ledger(ledger, tolerance_usd=1e-6)
    assert report.passed is False
    assert any(d.kind == "pnl_drift" for d in report.discrepancies)


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


def test_report_to_dict_is_json_serialisable(ledger_with_fills):
    ledger, fills = ledger_with_fills
    drill._corrupt(ledger, "wrong_price", target_cid="cid-003")
    report = reconcile_against_ledger(
        ledger,
        venue_fills={drill.TOKEN: [*fills, fills[0].model_copy(update={"order_client_id": "extra"})]},
    )
    blob = json.dumps(report.to_dict())  # must not raise
    assert "discrepancies" in blob
    assert "passed" in blob


def test_discrepancy_by_kind_filter(ledger_with_fills):
    ledger, _ = ledger_with_fills
    drill._corrupt(ledger, "duplicate_client_id", target_cid="cid-003")
    report = reconcile_against_ledger(ledger)
    dups = report.by_kind("duplicate_client_id")
    assert len(dups) == 1
    assert isinstance(dups[0], ReconciliationDiscrepancy)
    assert dups[0].token_id == drill.TOKEN


# ---------------------------------------------------------------------------
# run_all / run_drill
# ---------------------------------------------------------------------------


def test_run_all_returns_every_mode():
    passed, results = drill.run_all()
    assert passed is True
    assert [r.mode for r in results] == list(drill.ALL_MODES)
    for r in results:
        assert r.detected is True


def test_run_drill_single_mode_detects():
    res = drill.run_drill("wrong_price")
    assert res.mode == "wrong_price"
    assert res.detected is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_dry_run_exits_zero(caplog):
    caplog.set_level(logging.WARNING)
    rc = drill.main([])  # no --i-mean-it
    assert rc == 0
    assert "Dry-run" in caplog.text
    for mode in drill.ALL_MODES:
        assert mode in caplog.text


def test_cli_armed_run_exits_zero_when_all_detected(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    out = tmp_path / "drill_report.json"
    rc = drill.main(["--i-mean-it", "--out", str(out)])
    assert rc == 0
    assert "DRILL PASSED" in caplog.text
    assert out.exists()
    blob = json.loads(out.read_text())
    assert blob["passed"] is True


def test_cli_single_mode(caplog):
    caplog.set_level(logging.INFO)
    rc = drill.main(["--i-mean-it", "--mode=wrong_size"])
    assert rc == 0
    assert "wrong_size" in caplog.text
    assert "wrong_price" not in caplog.text.split("DRILL")[-1]  # only ran one


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ledger_with_fills() -> "tuple[Ledger, list[Fill]]":
    ledger = Ledger.in_memory()
    fills = drill.default_fill_sequence()
    for f in fills:
        ledger.apply_fill(f)
    return ledger, fills
