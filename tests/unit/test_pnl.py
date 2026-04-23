"""Unit tests for mm/pnl.py — attribution decomposition (paper §4.6).

Hand-calcs reflect the dx-increment form of the Taylor expansion:

    q · dp  =  q · Δ(prev.p) · dx_incr  +  ½ · q · Γ(prev.p) · dx_incr²  +  residual

where dx_incr = logit(next.p) - logit(prev.p). The residual always closes
into the jump bucket so Σ_buckets == q · dp exactly.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from blksch.mm.greeks import delta_x, gamma_x, logit
from blksch.mm.pnl import AttributionSnapshot, Attributor, realized_vs_expected_dp2


def _t(sec: float) -> datetime:
    return datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


class TestRealizedExpected:
    def test_expected_vanishes_at_boundary(self) -> None:
        _, exp_low = realized_vs_expected_dp2(0.01, 1.0, 1.0, p_prev=0.001)
        _, exp_high = realized_vs_expected_dp2(0.01, 1.0, 1.0, p_prev=0.999)
        _, exp_mid = realized_vs_expected_dp2(0.01, 1.0, 1.0, p_prev=0.5)
        assert exp_low < exp_mid
        assert exp_high < exp_mid

    def test_realized_is_dp_squared(self) -> None:
        real, _ = realized_vs_expected_dp2(0.02, 1.0, 0.3, 0.5)
        assert real == pytest.approx(0.0004)


class TestAttributor:
    def test_first_snap_returns_none(self) -> None:
        a = Attributor()
        snap = AttributionSnapshot("t", 0.5, 0.3, qty=10.0, ts=_t(0))
        assert a.step(snap) is None

    def test_directional_bucket_for_long(self) -> None:
        a = Attributor()
        a.step(AttributionSnapshot("t", 0.5, 0.3, qty=100.0, ts=_t(0)))
        step = a.step(AttributionSnapshot("t", 0.51, 0.3, qty=100.0, ts=_t(1.0)))
        # dx_incr = logit(0.51) - logit(0.5) ≈ 0.04001
        # directional = 100 · Δ(0.5) · dx_incr = 100 · 0.25 · 0.04001 ≈ 1.00025
        expected_dx = logit(0.51) - logit(0.5)
        assert step.directional_pnl == pytest.approx(100.0 * 0.25 * expected_dx)
        # Γ_x(0.5) = 0, so curvature is exactly zero regardless of dx.
        assert step.curvature_pnl == pytest.approx(0.0)

    def test_sum_matches_q_times_dp_exactly(self) -> None:
        """Core invariant: sum-of-buckets == q·dp to floating precision.

        This is the Stage-1 paper-soak reconciliation criterion.
        """
        a = Attributor()
        a.step(AttributionSnapshot("t", 0.30, 0.3, qty=100.0, ts=_t(0)))
        step = a.step(AttributionSnapshot("t", 0.45, 0.3, qty=100.0, ts=_t(1.0)))
        q_dp = 100.0 * (0.45 - 0.30)  # = 15.0
        assert step.total == pytest.approx(q_dp, abs=1e-9)
        # Jump bucket closes the residual (non-zero even without a true jump
        # because of the third-order Taylor residual).
        assert step.jump_pnl != 0.0

    def test_curvature_sign_with_long_convex(self) -> None:
        a = Attributor()
        # At p=0.3, Γ_x > 0 (convex). Curvature · dx_incr² > 0 for any dx ≠ 0.
        a.step(AttributionSnapshot("t", 0.3, 0.3, qty=100.0, ts=_t(0)))
        step_up = a.step(AttributionSnapshot("t", 0.32, 0.3, qty=100.0, ts=_t(1.0)))
        a.reset()
        a.step(AttributionSnapshot("t", 0.3, 0.3, qty=100.0, ts=_t(0)))
        step_down = a.step(AttributionSnapshot("t", 0.28, 0.3, qty=100.0, ts=_t(1.0)))
        assert step_up.curvature_pnl > 0.0
        assert step_down.curvature_pnl > 0.0  # convexity benefits both directions

    def test_cumulative_accumulates(self) -> None:
        a = Attributor()
        a.step(AttributionSnapshot("t", 0.5, 0.3, qty=100.0, ts=_t(0)))
        a.step(AttributionSnapshot("t", 0.51, 0.3, qty=100.0, ts=_t(1.0)))
        a.step(AttributionSnapshot("t", 0.52, 0.3, qty=100.0, ts=_t(2.0)))
        cum = a.cumulative
        assert cum["directional"] > 0.0
        # Bucket identity holds at every step, so the cumulative sum does too.
        assert cum["total"] == pytest.approx(
            cum["directional"] + cum["curvature"] + cum["belief_vega"]
            + cum["cross_event"] + cum["jump"]
        )
        # Position held 100 contracts the whole way; mark went 0.50 → 0.52 ⇒ $2.00.
        assert cum["total"] == pytest.approx(2.0, abs=1e-9)

    def test_jump_bucket_carries_residual_on_large_move(self) -> None:
        """Large dp ⇒ jump bucket carries the O(dx³) residual. The z-score
        heuristic is retained for telemetry but does not gate the arithmetic."""
        a = Attributor(jump_zscore_threshold=3.0)
        a.step(AttributionSnapshot("t", 0.3, 0.01, qty=100.0, ts=_t(0)))
        step = a.step(AttributionSnapshot("t", 0.5, 0.01, qty=100.0, ts=_t(1.0)))
        assert step.jump_pnl != 0.0
        # Even under a jump, the invariant holds.
        assert step.total == pytest.approx(100.0 * (0.5 - 0.3), abs=1e-9)

    def test_cross_event_terms_accumulate(self) -> None:
        a = Attributor()
        a.step(AttributionSnapshot("t", 0.5, 0.3, qty=100.0, ts=_t(0)))
        step = a.step(
            AttributionSnapshot("t", 0.5, 0.3, qty=100.0, ts=_t(1.0)),
            cross_event_terms=[(0.1, 0.05), (0.2, -0.01)],
        )
        assert step.cross_event_pnl == pytest.approx(0.1 * 0.05 + 0.2 * -0.01)
