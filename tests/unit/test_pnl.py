"""Unit tests for mm/pnl.py — attribution decomposition (paper §4.6)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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
        # Δ_x(0.5) = 0.25, dp = 0.01, qty = 100  ⇒  directional = 100·0.25·0.01 = 0.25
        assert step.directional_pnl == pytest.approx(0.25)
        # Curvature is 0 at p=0.5 since Γ_x(0.5)=0
        assert step.curvature_pnl == pytest.approx(0.0)

    def test_curvature_sign_with_long_convex(self) -> None:
        a = Attributor()
        # At p=0.3, Γ_x = 0.3*0.7*0.4 = 0.084 > 0 (convex). Long position gains on any dp.
        a.step(AttributionSnapshot("t", 0.3, 0.3, qty=100.0, ts=_t(0)))
        step_up = a.step(AttributionSnapshot("t", 0.32, 0.3, qty=100.0, ts=_t(1.0)))
        a.reset()
        a.step(AttributionSnapshot("t", 0.3, 0.3, qty=100.0, ts=_t(0)))
        step_down = a.step(AttributionSnapshot("t", 0.28, 0.3, qty=100.0, ts=_t(1.0)))
        assert step_up.curvature_pnl > 0.0
        assert step_down.curvature_pnl > 0.0  # positive convexity benefits both directions

    def test_cumulative_accumulates(self) -> None:
        a = Attributor()
        a.step(AttributionSnapshot("t", 0.5, 0.3, qty=100.0, ts=_t(0)))
        a.step(AttributionSnapshot("t", 0.51, 0.3, qty=100.0, ts=_t(1.0)))
        a.step(AttributionSnapshot("t", 0.52, 0.3, qty=100.0, ts=_t(2.0)))
        cum = a.cumulative
        assert cum["directional"] > 0.0
        assert cum["total"] == pytest.approx(
            cum["directional"] + cum["curvature"] + cum["belief_vega"]
            + cum["cross_event"] + cum["jump"]
        )

    def test_jump_flagged_on_huge_dp(self) -> None:
        a = Attributor(jump_zscore_threshold=3.0)
        # Tiny vol means small expected variance — any sizable dp will trip the threshold.
        a.step(AttributionSnapshot("t", 0.3, 0.01, qty=100.0, ts=_t(0)))
        step = a.step(AttributionSnapshot("t", 0.5, 0.01, qty=100.0, ts=_t(1.0)))
        assert step.jump_pnl != 0.0

    def test_cross_event_terms_accumulate(self) -> None:
        a = Attributor()
        a.step(AttributionSnapshot("t", 0.5, 0.3, qty=100.0, ts=_t(0)))
        step = a.step(
            AttributionSnapshot("t", 0.5, 0.3, qty=100.0, ts=_t(1.0)),
            cross_event_terms=[(0.1, 0.05), (0.2, -0.01)],
        )
        assert step.cross_event_pnl == pytest.approx(0.1 * 0.05 + 0.2 * -0.01)
