"""Unit tests for mm/hedge/calendar.py — variance-strip sizing (paper §4.3)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from blksch.mm.hedge.calendar import (
    CalendarHedgeParams,
    compute_calendar_hedge,
    xvar_synth_token_id,
)
from blksch.schemas import HedgeSide, SurfacePoint


TS = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
TOK = "0xA"


def _surf(sigma_b: float = 0.3, x: float = 0.0) -> SurfacePoint:
    return SurfacePoint(
        token_id=TOK, tau=3600.0, m=x, sigma_b=sigma_b,
        **{"lambda": 0.1}, s2_j=0.01, ts=TS,
    )


class TestParams:
    def test_floor_positive(self) -> None:
        with pytest.raises(ValueError):
            CalendarHedgeParams(sigma_b_floor=0.0)

    def test_max_notional_positive(self) -> None:
        with pytest.raises(ValueError):
            CalendarHedgeParams(max_abs_notional_usd=-1.0)


class TestTokenNaming:
    def test_convention(self) -> None:
        assert xvar_synth_token_id("0xABC") == "0xABC:xvar"


class TestSignCorrectness:
    def test_long_vol_exposure_short_hedges(self) -> None:
        """ν̂_b > 0 (book long σ_b) ⇒ N<0 ⇒ SHORT the variance strip."""
        h = compute_calendar_hedge(_surf(sigma_b=0.5), inventory_nu_b=10.0, ts=TS)
        assert h.side is HedgeSide.SHORT
        # |N| = 10 / 0.5 = 20
        assert h.notional_usd == pytest.approx(20.0)

    def test_short_vol_exposure_long_hedges(self) -> None:
        """ν̂_b < 0 ⇒ N>0 ⇒ LONG the variance strip."""
        h = compute_calendar_hedge(_surf(sigma_b=0.5), inventory_nu_b=-10.0, ts=TS)
        assert h.side is HedgeSide.LONG
        assert h.notional_usd == pytest.approx(20.0)


class TestZeroInventoryNoHedge:
    def test_zero_nu_b_zero_notional(self) -> None:
        h = compute_calendar_hedge(_surf(), inventory_nu_b=0.0, ts=TS)
        assert h.notional_usd == 0.0
        assert h.reason == "calendar"
        assert h.source_token_id == TOK
        assert h.hedge_token_id == xvar_synth_token_id(TOK)


class TestSigmaBFloor:
    def test_near_zero_sigma_b_clamped(self) -> None:
        """σ_b → 0 without the floor would make notional diverge. Floor caps it."""
        params = CalendarHedgeParams(sigma_b_floor=0.1, max_abs_notional_usd=1e9)
        h = compute_calendar_hedge(
            _surf(sigma_b=1e-6), inventory_nu_b=5.0, params=params, ts=TS
        )
        # Floor kicks in: |N| = 5 / 0.1 = 50
        assert h.notional_usd == pytest.approx(50.0)

    def test_max_abs_notional_hard_clamp(self) -> None:
        params = CalendarHedgeParams(sigma_b_floor=1e-4, max_abs_notional_usd=100.0)
        h = compute_calendar_hedge(
            _surf(sigma_b=1e-3), inventory_nu_b=10.0, params=params, ts=TS
        )
        # Naive |N| = 10 / 0.001 = 10000; clamped to 100.
        assert h.notional_usd == pytest.approx(100.0)


class TestScalingWithInventory:
    def test_linear_in_nu_b(self) -> None:
        a = compute_calendar_hedge(_surf(sigma_b=0.3), inventory_nu_b=5.0, ts=TS)
        b = compute_calendar_hedge(_surf(sigma_b=0.3), inventory_nu_b=10.0, ts=TS)
        assert b.notional_usd == pytest.approx(2.0 * a.notional_usd)

    def test_inverse_in_sigma_b(self) -> None:
        a = compute_calendar_hedge(_surf(sigma_b=0.2), inventory_nu_b=5.0, ts=TS)
        b = compute_calendar_hedge(_surf(sigma_b=0.4), inventory_nu_b=5.0, ts=TS)
        assert b.notional_usd == pytest.approx(0.5 * a.notional_usd)


class TestMetadata:
    def test_ts_propagates(self) -> None:
        custom = datetime(2030, 1, 1, tzinfo=timezone.utc)
        h = compute_calendar_hedge(_surf(), inventory_nu_b=1.0, ts=custom)
        assert h.ts == custom

    def test_reason_calendar(self) -> None:
        h = compute_calendar_hedge(_surf(), inventory_nu_b=1.0, ts=TS)
        assert h.reason == "calendar"
