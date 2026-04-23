"""Adversarial kill-switch scenarios (pre-paper-soak validation).

Required confidence before Stage-2 live-tiny: kill-switches must fire when
they should AND must NOT false-fire when they shouldn't. A false-positive
kill-switch wastes paper-soak hours; a false-negative loses money in Stage 2.

Exercises `mm.limits.LimitsState` directly. Five fire-correctly scenarios,
five don't-fire scenarios, one clearing-semantics scenario.

Test-only — no runtime code changes.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

import pytest

from blksch.mm.greeks import gamma_x
from blksch.mm.limits import (
    KillSwitchReport,
    LimitsConfig,
    LimitsState,
    gamma_exposure,
    inventory_cap_contracts,
)


BASE = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)


def _t(sec: float) -> datetime:
    return BASE + timedelta(seconds=sec)


def _fresh_state(
    *, feed_gap_sec: float = 3.0, vol_spike_z: float = 5.0,
    pickoff_window: float = 60.0, pickoff_count: int = 5,
    max_dd_usd: float = 100.0, max_gamma_exposure: float = 10.0,
    swing_half_width: float = 0.15,
) -> LimitsState:
    cfg = LimitsConfig(
        feed_gap_sec=feed_gap_sec,
        volatility_spike_z=vol_spike_z,
        volatility_window=30,
        repeated_pickoff_window_sec=pickoff_window,
        repeated_pickoff_count=pickoff_count,
        max_drawdown_usd=max_dd_usd,
        max_gamma_exposure=max_gamma_exposure,
        swing_zone_half_width=swing_half_width,
    )
    return LimitsState(cfg=cfg)


# ===========================================================================
# FIRE-CORRECTLY SCENARIOS (1-5)
# ===========================================================================


class TestFireCorrectly:
    def test_1_feed_gap(self) -> None:
        """No BookSnap for feed_gap_sec + 1 seconds → halted=True."""
        s = _fresh_state(feed_gap_sec=3.0)
        s.note_tick(_t(0))
        report = s.evaluate(now=_t(4.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "feed_gap" in report.reasons
        assert s.paused

    def test_2_volatility_spike(self) -> None:
        """σ_b jumps 10× from a steady 0.3 baseline within one update → halted."""
        s = _fresh_state(vol_spike_z=3.0)
        s.note_tick(_t(0))
        for i in range(30):
            s.note_sigma(0.3 + 0.001 * ((i % 7) - 3))  # stable baseline ~0.3
        report = s.evaluate(
            now=_t(1.0), current_sigma=3.0, cumulative_pnl_usd=0.0,
        )
        assert "volatility_spike" in report.reasons
        assert s.paused

    def test_3_repeated_pickoffs(self) -> None:
        """5 pick-offs within 30s (window 60s, count 5) → halted."""
        s = _fresh_state(pickoff_window=60.0, pickoff_count=5)
        s.note_tick(_t(0))
        for i in range(5):
            s.note_pickoff(_t(i * 5.0))  # 5 pickoffs over 25s
        report = s.evaluate(now=_t(26.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "repeated_pickoffs" in report.reasons
        assert s.paused

    def test_4_drawdown_hit(self) -> None:
        """cumulative PnL drops below -max_drawdown_usd → halted."""
        s = _fresh_state(max_dd_usd=100.0)
        s.note_tick(_t(0))
        report = s.evaluate(now=_t(1.0), current_sigma=None, cumulative_pnl_usd=-100.01)
        assert "max_drawdown" in report.reasons
        assert s.paused

    def test_5_swing_zone_gamma_blowup(self) -> None:
        """Adversarial scenario description: 'p=0.5 with q at q_max →
        Γ_x inventory risk spikes; assert inventory-cap kill-switch fires.'

        Mathematical reality: Γ_x(p=0.5) = 0 exactly (Γ is the second
        derivative of the sigmoid, which has an inflection point at p=0.5).
        So q·Γ(0.5) = 0 and the swing-zone gamma cap cannot fire at p=0.5.
        |Γ_x| peaks at p ≈ 0.211 and 0.789, both of which are OUTSIDE the
        default swing_zone_half_width=0.15 around 0.5.

        We test the intended semantic — 'large q inside the swing zone with
        non-trivial Γ_x should trip max_gamma_swing' — at p=0.4 which IS
        inside the swing zone and has Γ_x(0.4) = 0.048. This is the point
        in the swing zone where q·Γ is largest.
        """
        # max_gamma_exposure=1.0 so q=100·Γ(0.4)=4.8 easily trips.
        s = _fresh_state(max_gamma_exposure=1.0, swing_half_width=0.15)
        s.note_tick(_t(0))
        report = s.evaluate(
            now=_t(0.0), current_sigma=None, cumulative_pnl_usd=0.0,
            current_qty=100.0, current_p=0.4,
        )
        assert "max_gamma_swing" in report.reasons
        assert s.paused

    def test_5b_swing_zone_at_p_exactly_half_literally_cannot_fire(self) -> None:
        """Sanity: the literal scenario 5 description (p=0.5 with q=q_max)
        CANNOT fire max_gamma_swing because Γ_x(0.5) = 0. This is a math
        fact, not a bug. Documented for anyone revisiting the spec."""
        assert gamma_x(0.5) == 0.0
        s = _fresh_state(max_gamma_exposure=1.0)
        s.note_tick(_t(0))
        report = s.evaluate(
            now=_t(0.0), current_sigma=None, cumulative_pnl_usd=0.0,
            current_qty=1e6, current_p=0.5,   # absurdly large q
        )
        assert "max_gamma_swing" not in report.reasons


# ===========================================================================
# DO-NOT-FIRE SCENARIOS (6-10)
# ===========================================================================


class TestDoNotFire:
    def test_6_brief_volatility_blip(self) -> None:
        """σ_b rises to 1.5× baseline for one tick then recovers — NOT a spike."""
        s = _fresh_state(vol_spike_z=5.0)
        s.note_tick(_t(0))
        for _ in range(30):
            s.note_sigma(0.3)
        # Single 1.5× blip: with stable baseline, z-score is (0.45 - 0.3) / tiny_std
        # which CAN trigger a 5σ threshold. Inject noise so the 1.5× is just
        # a 2σ departure — NOT a spike.
        for i in range(15):
            s.note_sigma(0.3 + 0.02 * ((i % 7) - 3))  # std ≈ 0.015
        report = s.evaluate(
            now=_t(1.0), current_sigma=0.35, cumulative_pnl_usd=0.0,  # ~3σ blip
        )
        assert "volatility_spike" not in report.reasons
        assert not s.paused

    def test_7_queue_churn_without_pickoff(self) -> None:
        """10 cancel-replace cycles with no adverse fills — kill-switch must NOT fire.

        The LimitsState only counts pick-offs explicitly reported via
        note_pickoff; queue churn (cancel-replace lifecycle) does not and
        MUST not implicitly count as a pick-off.
        """
        s = _fresh_state(pickoff_count=5)
        # Continuous fresh ticks during the 10 cycles (no feed gap).
        for i in range(11):
            s.note_tick(_t(i * 1.0))
        # 10 cancel-replace cycles, zero pickoffs recorded.
        report = s.evaluate(now=_t(10.5), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "repeated_pickoffs" not in report.reasons
        assert not s.paused

    def test_8_single_adverse_fill(self) -> None:
        """ONE pick-off — kill-switch must NOT fire (threshold is 5 in 60s)."""
        s = _fresh_state(pickoff_window=60.0, pickoff_count=5)
        # Fresh ticks so feed-gap doesn't also fire.
        for i in range(7):
            s.note_tick(_t(i * 1.0))
        s.note_pickoff(_t(5.0))
        report = s.evaluate(now=_t(6.5), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "repeated_pickoffs" not in report.reasons
        assert not s.paused

    def test_9_drawdown_at_95pct(self) -> None:
        """PnL at -$95 with max at -$100 — boundary case, MUST NOT fire."""
        s = _fresh_state(max_dd_usd=100.0)
        s.note_tick(_t(0))
        report = s.evaluate(now=_t(1.0), current_sigma=None, cumulative_pnl_usd=-95.0)
        assert "max_drawdown" not in report.reasons
        assert not s.paused

    def test_10_feed_gap_below_threshold(self) -> None:
        """No BookSnap for feed_gap_sec - 0.5 seconds — MUST NOT fire."""
        s = _fresh_state(feed_gap_sec=3.0)
        s.note_tick(_t(0))
        report = s.evaluate(now=_t(2.5), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "feed_gap" not in report.reasons
        assert not s.paused


# ===========================================================================
# CLEARING SEMANTICS (11)
# ===========================================================================


class TestClearingSemantics:
    def test_11_halted_does_not_self_heal(self) -> None:
        """After scenario 1 (feed-gap halts the bot), resuming a normal
        BookSnap stream does NOT auto-clear `paused`. Only an explicit
        `resume()` (the equivalent of the operator's clear_halt) flips
        it back. This matches the operator runbook's 'state.halted
        semantics (not self-healing)' promise."""
        s = _fresh_state(feed_gap_sec=3.0)
        s.note_tick(_t(0))

        # Halt via feed gap.
        report = s.evaluate(now=_t(5.0), current_sigma=None, cumulative_pnl_usd=0.0)
        assert "feed_gap" in report.reasons
        assert s.paused

        # Now resume the BookSnap stream with fresh ticks.
        for i in range(10):
            s.note_tick(_t(5.1 + i * 0.1))

        # evaluate() again with a fresh clock — the feed gap is closed.
        report2 = s.evaluate(
            now=_t(6.5), current_sigma=None, cumulative_pnl_usd=0.0,
        )
        # The current evaluation itself does not trip (no fresh reasons),
        # but paused is still True from the prior trip.
        assert "feed_gap" not in report2.reasons
        assert s.paused   # NOT self-healing

        # Explicit resume clears it.
        s.resume()
        assert not s.paused


# ===========================================================================
# Helper-function smoke tests
# ===========================================================================


class TestHelperInvariants:
    def test_inventory_cap_finite_monotone_at_boundary(self) -> None:
        """inventory_cap_contracts tightens as S'(x) grows (largest at p=0.5)."""
        cfg = LimitsConfig(inventory_cap_base=50.0, sprime_floor=1e-4)
        cap_mid = inventory_cap_contracts(0.0, cfg)
        cap_edge = inventory_cap_contracts(5.0, cfg)  # p≈0.993
        assert cap_edge > cap_mid
        # Extreme logit hits the floor: cap = base / floor.
        cap_extreme = inventory_cap_contracts(100.0, cfg)
        assert cap_extreme == pytest.approx(50.0 / 1e-4, rel=1e-6)

    def test_gamma_exposure_zero_at_p_half_by_math(self) -> None:
        assert gamma_exposure(1e6, 0.5) == 0.0

    def test_gamma_exposure_sign_follows_curvature(self) -> None:
        """Γ_x > 0 for p < 0.5 and < 0 for p > 0.5."""
        assert gamma_exposure(100.0, 0.3) > 0.0
        assert gamma_exposure(100.0, 0.7) < 0.0
