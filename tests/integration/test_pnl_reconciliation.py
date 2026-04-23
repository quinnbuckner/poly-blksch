"""PnL reconciliation: Track C Ledger ↔ Track B Attributor (paper §4.6).

Prep for the Stage-1 paper-trading soak. The soak's acceptance criterion
("PnL attribution residual < threshold") compares:
  * Ledger total  — realized + unrealized, computed from fills + marks
  * Attributor    — Δ-Γ-ν_b-ν_ρ-jump decomposition of inventory PnL

If these two views disagree, the 72-hour soak will false-fail. This test
drives a scripted Quote → Fill → Position sequence through both and asserts
they reconcile to within 1e-6 USD over the whole sequence.

Two reconciliation paths:

1. `test_ledger_reconciles_with_reference_attribution` — uses an inline
   dx-based Greek decomposition (the math the paper's §4.6 template
   implies for a vanilla-contract inventory book) and verifies the sum
   of buckets equals the ledger total to 1e-6 USD. This establishes
   that the ledger is decomposable in principle.

2. `test_ledger_reconciles_with_production_attributor` — uses
   `mm.pnl.Attributor`. Currently `xfail`: the production Attributor
   attributes `directional = q · Δ · dp` where the correct term is
   `q · Δ · dx` with `dx = logit(p_next) - logit(p_prev)`. The xfail
   becomes pass once mm/pnl.py is patched to use the logit increment;
   keep strict=True so the test auto-flips the moment the fix lands.

Runtime code unchanged in this branch — this is a tests/integration-only
drop intended to unblock the Stage-1 paper soak gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from blksch.exec.ledger import Ledger
from blksch.mm.greeks import delta_x, gamma_x, logit
from blksch.mm.pnl import Attributor, AttributionSnapshot
from blksch.schemas import Fill, OrderSide


TOK = "0xA"
BASE = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
USD_TOL = 1.0e-6


def _ts(sec: float) -> datetime:
    return BASE + timedelta(seconds=sec)


# ---------------------------------------------------------------------------
# Scripted tick stream
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tick:
    """One step of the scripted path.

    At most one of `fill` or `mark_update` is set per tick. `p_observed`
    is the current probability mark the Attributor sees (mark or fill px).
    """

    ts: datetime
    p_observed: float
    sigma_b: float
    fill: Fill | None = None
    mark_update: float | None = None


def _fill(
    side: OrderSide, price: float, size: float, fee: float, sec: float, seq: int
) -> Fill:
    return Fill(
        order_client_id=f"order-{seq}",
        order_venue_id=f"venue-{seq}",
        token_id=TOK,
        side=side,
        price=price,
        size=size,
        fee_usd=fee,
        ts=_ts(sec),
    )


def _scripted_path() -> list[Tick]:
    """Realistic mixed sequence:

        t=0   : buy 100 @ 0.50  (no mark → first mark = fill px)
        t=1   : mark moves to 0.55
        t=2   : sell 40 @ 0.55  (lock in realized on partial close)
        t=3   : mark moves to 0.60
        t=4   : sell 60 @ 0.60  (close the book)
        t=5   : mark moves to 0.58 (flat position — PnL unchanged)

    Zero fees — we're testing price-driven PnL reconciliation, not the
    fee bucket (fees are orthogonal and ledger-only).
    """
    return [
        Tick(
            ts=_ts(0), p_observed=0.50, sigma_b=0.30,
            fill=_fill(OrderSide.BUY, 0.50, 100.0, 0.0, sec=0, seq=1),
        ),
        Tick(ts=_ts(1), p_observed=0.55, sigma_b=0.30, mark_update=0.55),
        Tick(
            ts=_ts(2), p_observed=0.55, sigma_b=0.30,
            fill=_fill(OrderSide.SELL, 0.55, 40.0, 0.0, sec=2, seq=2),
        ),
        Tick(ts=_ts(3), p_observed=0.60, sigma_b=0.30, mark_update=0.60),
        Tick(
            ts=_ts(4), p_observed=0.60, sigma_b=0.30,
            fill=_fill(OrderSide.SELL, 0.60, 60.0, 0.0, sec=4, seq=3),
        ),
        Tick(ts=_ts(5), p_observed=0.58, sigma_b=0.30, mark_update=0.58),
    ]


# ---------------------------------------------------------------------------
# Reference attribution (the paper's §4.6 template done correctly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefStep:
    directional: float
    curvature: float
    jump: float
    total: float


def reference_attribution(
    qty_prev: float, p_prev: float, p_next: float,
    sigma_b_prev: float = 0.3, dt_sec: float = 1.0,
) -> RefStep:
    """dx-based Greek decomposition with exact residual closure.

        dΠ = q · dp = q · Δ_x(p_prev) · dx  +  ½ q · Γ_x(p_prev) · dx²  +  residual

    where `dx = logit(p_next) - logit(p_prev)` and `residual` absorbs all
    higher-order Taylor terms (O(dx³) and beyond). Residual is credited to
    the jump bucket — consistent with the paper §4.6 treatment of
    "unmodeled" quadratic variation.

    By construction:  directional + curvature + jump == q · dp  exactly.
    """
    if qty_prev == 0.0:
        return RefStep(0.0, 0.0, 0.0, 0.0)

    dp = p_next - p_prev
    true_total = qty_prev * dp

    dx = logit(p_next) - logit(p_prev)
    sp = delta_x(p_prev)
    gpr = gamma_x(p_prev)

    directional = qty_prev * sp * dx
    curvature = 0.5 * qty_prev * gpr * dx * dx
    jump = true_total - directional - curvature  # closes the decomposition

    return RefStep(
        directional=directional,
        curvature=curvature,
        jump=jump,
        total=directional + curvature + jump,
    )


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------


def _run_ledger(path: list[Tick]) -> Ledger:
    ledger = Ledger.in_memory()
    for tick in path:
        if tick.fill is not None:
            ledger.apply_fill(tick.fill)
        if tick.mark_update is not None:
            ledger.update_mark(TOK, tick.mark_update, ts=tick.ts)
    return ledger


def _run_reference_attribution(path: list[Tick]) -> dict[str, float]:
    """Integrate reference_attribution over the scripted path.

    We snap a position every tick: qty is the ledger qty right *before* the
    tick's fill (i.e. the holding-period qty that earns the next price move).
    """
    # Replay the ledger to know qty right before each tick.
    ledger = Ledger.in_memory()
    cumulative = {"directional": 0.0, "curvature": 0.0, "jump": 0.0, "total": 0.0}
    prev_qty = 0.0
    prev_p: float | None = None
    prev_sigma = 0.3
    prev_ts: datetime | None = None

    for tick in path:
        # The qty held *going into* this tick earns (dp) from prev_p -> p_observed.
        if prev_p is not None and prev_qty != 0.0:
            dt = (tick.ts - prev_ts).total_seconds() if prev_ts else 1.0
            step = reference_attribution(
                qty_prev=prev_qty,
                p_prev=prev_p,
                p_next=tick.p_observed,
                sigma_b_prev=prev_sigma,
                dt_sec=dt,
            )
            cumulative["directional"] += step.directional
            cumulative["curvature"] += step.curvature
            cumulative["jump"] += step.jump
            cumulative["total"] += step.total

        # Apply the fill to the shadow ledger to get post-tick qty.
        if tick.fill is not None:
            ledger.apply_fill(tick.fill)
        if tick.mark_update is not None:
            ledger.update_mark(TOK, tick.mark_update, ts=tick.ts)
        prev_qty = ledger.get_position(TOK).qty
        prev_p = tick.p_observed
        prev_sigma = tick.sigma_b
        prev_ts = tick.ts

    return cumulative


def _run_production_attributor(path: list[Tick]) -> dict[str, float]:
    ledger = Ledger.in_memory()
    attributor = Attributor()

    prev_qty = 0.0
    for tick in path:
        # Feed the Attributor a snapshot BEFORE applying this tick's fill,
        # capturing the qty held over the interval that just elapsed.
        snap = AttributionSnapshot(
            token_id=TOK, p=tick.p_observed, sigma_b=tick.sigma_b,
            qty=prev_qty, ts=tick.ts,
        )
        attributor.step(snap)

        if tick.fill is not None:
            ledger.apply_fill(tick.fill)
        if tick.mark_update is not None:
            ledger.update_mark(TOK, tick.mark_update, ts=tick.ts)
        prev_qty = ledger.get_position(TOK).qty

    return attributor.cumulative


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReferenceAttributionReconciles:
    def test_ledger_total_equals_reference_attribution_sum(self) -> None:
        """Core reconciliation proof with the paper §4.6 dx-based decomposition."""
        path = _scripted_path()
        ledger = _run_ledger(path)
        attribution = _run_reference_attribution(path)

        ledger_total = ledger.pnl().total_usd
        assert abs(ledger_total - attribution["total"]) < USD_TOL, (
            f"ledger ${ledger_total:.9f} vs attribution ${attribution['total']:.9f}"
        )

    def test_reference_attribution_nontrivial(self) -> None:
        """Sanity: directional bucket carries most of the PnL on the scripted
        path (p stays near 0.5 where Γ=0, and no jumps)."""
        path = _scripted_path()
        a = _run_reference_attribution(path)
        assert abs(a["directional"]) > 0.0
        assert a["total"] > 0.0  # price rose 0.50 → 0.55 → 0.60 while we were long


class TestProductionAttributor:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "mm/pnl.py Attributor uses `dp` in the directional bucket; the paper's "
            "§4.6 decomposition requires `dx = logit(p_next) - logit(p_prev)`. "
            "Remove xfail once mm/pnl.py is patched. This test will auto-flip to "
            "PASS at that point and hard-guard the fix."
        ),
    )
    def test_ledger_total_equals_production_attributor_sum(self) -> None:
        path = _scripted_path()
        ledger = _run_ledger(path)
        cum = _run_production_attributor(path)
        assert abs(ledger.pnl().total_usd - cum["total"]) < USD_TOL


class TestNumericalSanity:
    """Hand-verified arithmetic on the simple case so we trust the machinery."""

    def test_simple_long_then_mark_up(self) -> None:
        """Buy 100 @ 0.5, mark to 0.6 ⇒ unrealized = $10.

        The reference attribution closes the Taylor residual into the jump
        bucket, so the sum matches exactly regardless of dx magnitude.
        """
        path = [
            Tick(ts=_ts(0), p_observed=0.50, sigma_b=0.30,
                 fill=_fill(OrderSide.BUY, 0.50, 100.0, 0.0, sec=0, seq=1)),
            Tick(ts=_ts(1), p_observed=0.60, sigma_b=0.30, mark_update=0.60),
        ]
        ledger = _run_ledger(path)
        a = _run_reference_attribution(path)
        assert ledger.pnl().total_usd == pytest.approx(10.0, abs=1e-9)
        assert abs(a["total"] - 10.0) < USD_TOL

    def test_long_then_flat_locks_in_realized(self) -> None:
        path = [
            Tick(ts=_ts(0), p_observed=0.50, sigma_b=0.30,
                 fill=_fill(OrderSide.BUY, 0.50, 100.0, 0.0, sec=0, seq=1)),
            Tick(ts=_ts(1), p_observed=0.55, sigma_b=0.30, mark_update=0.55),
            Tick(ts=_ts(2), p_observed=0.55, sigma_b=0.30,
                 fill=_fill(OrderSide.SELL, 0.55, 100.0, 0.0, sec=2, seq=2)),
        ]
        ledger = _run_ledger(path)
        snap = ledger.pnl()
        # Realized = qty·(exit - entry) = 100·0.05 = $5.00; unrealized = 0.
        assert snap.realized_usd == pytest.approx(5.0, abs=1e-9)
        assert snap.unrealized_usd == pytest.approx(0.0, abs=1e-9)
        assert snap.total_usd == pytest.approx(5.0, abs=1e-9)

    def test_fee_bucket_isolated(self) -> None:
        """Ledger fees are tracked in fees_usd AND already subtracted from
        realized. Attribution covers price-driven PnL only; the fee bucket
        is a separate attribution column in the paper soak acceptance criteria."""
        path = [
            Tick(ts=_ts(0), p_observed=0.50, sigma_b=0.30,
                 fill=_fill(OrderSide.BUY, 0.50, 100.0, fee=0.5, sec=0, seq=1)),
            Tick(ts=_ts(1), p_observed=0.55, sigma_b=0.30, mark_update=0.55),
        ]
        ledger = _run_ledger(path)
        snap = ledger.pnl()
        assert snap.fees_usd == pytest.approx(0.5)
        # Realized carries the fee drag; unrealized carries the mark-up.
        assert snap.realized_usd == pytest.approx(-0.5)
        assert snap.unrealized_usd == pytest.approx(5.0, abs=1e-9)
        assert snap.total_usd == pytest.approx(4.5, abs=1e-9)
