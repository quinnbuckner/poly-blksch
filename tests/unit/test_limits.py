"""Unit tests for mm/limits.py — kill-switches (paper §4.6)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blksch.mm.limits import (
    LimitsConfig,
    LimitsState,
    gamma_exposure,
    inventory_cap_contracts,
)


def _t(sec: float = 0.0) -> datetime:
    return datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


class TestInventoryCap:
    def test_cap_grows_near_boundary(self) -> None:
        c = LimitsConfig(inventory_cap_base=50.0)
        at_half = inventory_cap_contracts(0.0, c)
        at_edge = inventory_cap_contracts(5.0, c)  # p ≈ 0.993
        assert at_edge > at_half

    def test_cap_bounded_by_sprime_floor(self) -> None:
        c = LimitsConfig(inventory_cap_base=1.0, sprime_floor=1e-4)
        cap = inventory_cap_contracts(100.0, c)
        assert cap == pytest.approx(1.0 / 1.0e-4)


class TestGammaExposure:
    def test_zero_at_p_half(self) -> None:
        assert gamma_exposure(100.0, 0.5) == pytest.approx(0.0)

    def test_sign_follows_curvature(self) -> None:
        # p<0.5 convex ⇒ long position has positive Γ exposure
        assert gamma_exposure(100.0, 0.3) > 0.0
        # p>0.5 concave ⇒ long position has negative Γ exposure
        assert gamma_exposure(100.0, 0.7) < 0.0


class TestFeedGap:
    def test_fires_after_threshold(self) -> None:
        s = LimitsState(cfg=LimitsConfig(feed_gap_sec=2.0))
        s.note_tick(_t(0))
        r = s.evaluate(now=_t(5.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "feed_gap" in r.reasons
        assert s.paused

    def test_no_fire_when_fresh(self) -> None:
        s = LimitsState(cfg=LimitsConfig(feed_gap_sec=5.0))
        s.note_tick(_t(0))
        r = s.evaluate(now=_t(1.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "feed_gap" not in r.reasons
        assert not s.paused

    def test_no_fire_when_no_ticks_yet(self) -> None:
        s = LimitsState(cfg=LimitsConfig(feed_gap_sec=1.0))
        r = s.evaluate(now=_t(100.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "feed_gap" not in r.reasons


class TestVolatilitySpike:
    def test_spike_fires(self) -> None:
        s = LimitsState(cfg=LimitsConfig(volatility_spike_z=3.0, volatility_window=30))
        s.note_tick(_t(0))
        # Inject mild jitter so stdev is non-zero.
        for i in range(30):
            s.note_sigma(0.2 + 0.01 * ((i % 5) - 2))
        r = s.evaluate(now=_t(0), current_sigma=5.0, cumulative_pnl_usd=0.0)
        assert "volatility_spike" in r.reasons

    def test_needs_history(self) -> None:
        s = LimitsState()
        s.note_tick(_t(0))
        for _ in range(3):
            s.note_sigma(0.2)
        r = s.evaluate(now=_t(0), current_sigma=5.0, cumulative_pnl_usd=0.0)
        assert "volatility_spike" not in r.reasons


class TestRepeatedPickoffs:
    def test_fires_at_count(self) -> None:
        s = LimitsState(cfg=LimitsConfig(repeated_pickoff_window_sec=60.0, repeated_pickoff_count=3))
        s.note_tick(_t(0))
        for i in range(3):
            s.note_pickoff(_t(i))
        r = s.evaluate(now=_t(3.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "repeated_pickoffs" in r.reasons

    def test_old_pickoffs_expire(self) -> None:
        s = LimitsState(cfg=LimitsConfig(repeated_pickoff_window_sec=10.0, repeated_pickoff_count=3))
        s.note_tick(_t(0))
        s.note_pickoff(_t(0))
        s.note_pickoff(_t(1))
        s.note_pickoff(_t(2))
        r = s.evaluate(now=_t(100.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "repeated_pickoffs" not in r.reasons


class TestDrawdown:
    def test_fires_at_threshold(self) -> None:
        s = LimitsState(cfg=LimitsConfig(max_drawdown_usd=100.0))
        s.note_tick(_t(0))
        r = s.evaluate(now=_t(0), current_sigma=None, cumulative_pnl_usd=-150.0)
        assert "max_drawdown" in r.reasons


class TestSwingZoneGamma:
    def test_swing_zone_fires(self) -> None:
        s = LimitsState(cfg=LimitsConfig(max_gamma_exposure=1.0, swing_zone_half_width=0.15))
        s.note_tick(_t(0))
        # qty=100, p=0.45  ⇒  Γ_x(0.45)=0.45*0.55*0.1≈0.02475, exposure≈2.475 > 1.0
        r = s.evaluate(
            now=_t(0), current_sigma=None, cumulative_pnl_usd=0.0,
            current_qty=100.0, current_p=0.45,
        )
        assert "max_gamma_swing" in r.reasons

    def test_outside_swing_no_fire(self) -> None:
        s = LimitsState(cfg=LimitsConfig(max_gamma_exposure=0.01, swing_zone_half_width=0.15))
        s.note_tick(_t(0))
        # p=0.8 is outside swing zone; no trigger even with small threshold
        r = s.evaluate(
            now=_t(0), current_sigma=None, cumulative_pnl_usd=0.0,
            current_qty=100.0, current_p=0.8,
        )
        assert "max_gamma_swing" not in r.reasons


class TestResume:
    def test_pauses_then_resumes(self) -> None:
        s = LimitsState(cfg=LimitsConfig(feed_gap_sec=1.0))
        s.note_tick(_t(0))
        s.evaluate(now=_t(5.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert s.paused
        s.resume()
        assert not s.paused


class TestStickyPauseSemantics:
    """Regressions for the kill-switch sticky-pause contract (paper §4.6,
    "halt until operator resume"). Before this contract, ``evaluate()`` set
    ``_paused`` only on cycles where something tripped — on a feed-gap
    recovery cycle (fresh ticks, no current trip), the refresh loop saw
    ``report.tripped == False`` and resumed quoting, but the paper_engine's
    own sticky halt still rejected every order, flooding the 72 h soak log
    with ``place_order rejected (halted)`` WARNs every 250 ms. The fix is
    that LimitsState itself is sticky; this class pins the guarantees
    upstream code relies on."""

    def test_pause_is_sticky_across_many_clean_cycles(self) -> None:
        """Trip feed_gap once, then run 5 more evaluate() calls with a
        fresh tick stream. ``paused`` stays True, ``pause_reasons``
        retains ``"feed_gap"`` — the feed-gap recovery does NOT auto-heal
        the halt; only ``resume()`` does."""
        s = LimitsState(cfg=LimitsConfig(feed_gap_sec=1.0))
        s.note_tick(_t(0))
        trip = s.evaluate(now=_t(5.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "feed_gap" in trip.reasons
        assert s.paused
        assert s.pause_reasons == ("feed_gap",)

        # 5 clean cycles — each advances the clock and emits a fresh tick.
        for i in range(5):
            s.note_tick(_t(5.5 + i * 0.5))
            r = s.evaluate(
                now=_t(5.6 + i * 0.5),
                current_sigma=None,
                cumulative_pnl_usd=0.0,
            )
            # Current-cycle report itself is clean …
            assert r.reasons == (), f"cycle {i}: current-cycle should be clean"
            # … but the sticky state survives.
            assert s.paused, f"cycle {i}: paused must stay True"
            assert s.pause_reasons == ("feed_gap",), f"cycle {i}: reasons retained"

    def test_pause_reasons_accumulate_on_second_distinct_trip(self) -> None:
        """When a second, different kill-switch trips while already paused,
        ``pause_reasons`` accumulates the new reason instead of replacing
        the old one. The operator audit trail records every condition
        seen during the halt."""
        s = LimitsState(cfg=LimitsConfig(
            feed_gap_sec=1.0, max_drawdown_usd=100.0,
        ))
        s.note_tick(_t(0))

        # Trip 1: feed_gap
        s.evaluate(now=_t(5.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert s.pause_reasons == ("feed_gap",)

        # Trip 2 while still paused: drawdown (fresh tick so feed_gap alone
        # would no longer fire, proving accumulation is not a side-effect
        # of feed_gap re-firing).
        s.note_tick(_t(5.5))
        r = s.evaluate(now=_t(5.6), current_sigma=None, cumulative_pnl_usd=-150.0)
        assert "max_drawdown" in r.reasons
        # Persistent set includes both — feed_gap from the earlier trip,
        # max_drawdown from this one. This is the audit-trail contract.
        assert set(s.pause_reasons) == {"feed_gap", "max_drawdown"}

    def test_caller_can_diff_to_identify_new_reasons(self) -> None:
        """Callers that want "newly tripped this cycle" vs "sticky
        continuation" diff ``report.reasons`` against a pre-evaluate
        snapshot of ``pause_reasons``. The refresh loop does exactly this
        to pick WARN vs DEBUG log severity; this pins the mechanism."""
        s = LimitsState(cfg=LimitsConfig(feed_gap_sec=1.0))
        s.note_tick(_t(0))

        # First cycle: nothing was in the set before → all reasons are new.
        pre = frozenset(s.pause_reasons)
        first = s.evaluate(now=_t(5.0), current_sigma=None, cumulative_pnl_usd=0.0)
        new_reasons_1 = tuple(r for r in first.reasons if r not in pre)
        assert new_reasons_1 == ("feed_gap",)

        # Second cycle: feed_gap still fires, but it's already in the set
        # → diff is empty → refresh loop would log DEBUG not WARN.
        pre = frozenset(s.pause_reasons)
        second = s.evaluate(now=_t(6.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "feed_gap" in second.reasons
        new_reasons_2 = tuple(r for r in second.reasons if r not in pre)
        assert new_reasons_2 == ()

    def test_resume_accepts_optional_reason_and_clears_everything(self) -> None:
        """``resume(reason="operator note")`` logs the note, clears
        ``paused`` + ``pause_reasons`` + the pick-off window. Calling
        without a reason is still supported (backward compat)."""
        s = LimitsState(cfg=LimitsConfig(feed_gap_sec=1.0))
        s.note_tick(_t(0))
        s.evaluate(now=_t(5.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert s.paused

        s.resume(reason="feed restored, venue confirmed")
        assert not s.paused
        assert s.pause_reasons == ()

        # Trip again, resume with no reason — keeps the backward-compat
        # zero-arg signature working for callers that predate the param.
        s.note_tick(_t(10.0))
        s.evaluate(now=_t(15.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert s.paused
        s.resume()
        assert not s.paused
        assert s.pause_reasons == ()
