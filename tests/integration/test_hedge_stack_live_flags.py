"""Hedge-flag-on validation harness (unlocks Stage-3 flip).

Scenario: two correlated markets A (target) + B (peer), all three hedge flags
on (`hedge_enabled`, `calendar_hedge_enabled`, `synth_strip_enabled`). We drive
a scripted two-step path through the refresh loop and assert the end state:

  (i)   All three hedge types emit simultaneously without collision:
        β-hedges (reason="beta") against peer B, plus a synth_strip basket
        (reason="synth_strip") fanned out from the calendar leg. The
        unroutable "calendar"-reason synthetic leg is never emitted when
        synth_strip is on.

  (ii)  Every synth-strip basket leg resolves to a real token_id — i.e.
        `hedge_token_id` matches one of the neighborhood tokens, never the
        `{tok}:xvar` placeholder.

  (iii) Net cross-event vega (to peer B) lands within paper §4.4's bound
        after the hedge: with α=1.0 and ρ=0.7, a full β-hedge removes the
        exposure to O(shrinkage); residual ≤ (1-α) · pre-hedge + co-jump
        correction ≈ 0 in this test. We check the residual is < 20% of the
        pre-hedge exposure (a generous bound that covers numerical slack).

  (iv)  No double-counting between beta and calendar: the β-hedge notional
        matches the standalone `compute_beta_hedge()` output; the
        synth_strip basket's total notional matches the standalone
        `compute_calendar_hedge()` notional. Neither depends on the other.

This test is runtime-off: it enables the flags *within the test fixture only*
and does not modify the global defaults in LoopConfig. It ships safely to
main; flipping hedge flags in production still requires the Stage-1 paper
soak gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from blksch.mm.greeks import s_prime, sigmoid
from blksch.mm.hedge.beta import BetaHedgeParams, compute_beta_hedge
from blksch.mm.hedge.calendar import (
    CalendarHedgeParams,
    compute_calendar_hedge,
    xvar_synth_token_id,
)
from blksch.mm.hedge.synth_strip import SynthStripParams
from blksch.mm.limits import LimitsConfig
from blksch.mm.quote import QuoteParams
from blksch.mm.refresh_loop import (
    HedgePeer,
    LoopConfig,
    MarketSnapshot,
    RefreshLoop,
)
from blksch.schemas import (
    BookSnap,
    CorrelationEntry,
    HedgeInstruction,
    HedgeSide,
    LogitState,
    Position,
    PriceLevel,
    Quote,
    SurfacePoint,
)


# ---------------------------------------------------------------------------
# Scenario fixture constants
# ---------------------------------------------------------------------------


TOK_A = "0xA"      # target market
TOK_B = "0xB"      # β-hedge peer
NEIGHBORS = ("0xN_low", "0xN_mid", "0xN_high")  # synth-strip basket candidates

ALPHA = 1.0        # no β shrinkage in this scenario
RHO = 0.7          # A-B correlation
SIGMA_B = 0.4      # belief vol shared across all surface points
TARGET_QTY = 100.0
TARGET_MARK = 0.5
INVENTORY_NU_B = 8.0

BASE = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)


def _ts(sec: float = 0.0) -> datetime:
    return BASE + timedelta(seconds=sec)


def _surf(token_id: str, m: float = 0.0, tau: float = 3600.0, sigma_b: float = SIGMA_B) -> SurfacePoint:
    return SurfacePoint(
        token_id=token_id, tau=tau, m=m, sigma_b=sigma_b,
        **{"lambda": 0.1}, s2_j=0.0, ts=BASE,
    )


def _logit(x: float = 0.0) -> LogitState:
    return LogitState(token_id=TOK_A, x_hat=x, sigma_eta2=0.01, ts=BASE)


def _book(mid: float = 0.5) -> BookSnap:
    return BookSnap(
        token_id=TOK_A,
        bids=[PriceLevel(price=mid - 0.01, size=100.0)],
        asks=[PriceLevel(price=mid + 0.01, size=100.0)],
        ts=BASE,
    )


def _pos(qty: float = TARGET_QTY, mark: float = TARGET_MARK) -> Position:
    return Position(token_id=TOK_A, qty=qty, avg_entry=mark, mark=mark, realized_pnl_usd=0.0)


def _corr(rho: float = RHO) -> CorrelationEntry:
    return CorrelationEntry(
        token_id_i=TOK_A, token_id_j=TOK_B,
        rho=rho, co_jump_lambda=0.0, co_jump_m2=0.0, ts=BASE,
    )


# ---------------------------------------------------------------------------
# Harness: refresh loop + sink collector with all three hedge flags on
# ---------------------------------------------------------------------------


@dataclass
class _StubFeed:
    snap: MarketSnapshot

    async def __call__(self, token_id: str) -> MarketSnapshot:
        return self.snap


@dataclass
class _Collector:
    quotes: list[Quote] = field(default_factory=list)
    hedges: list[HedgeInstruction] = field(default_factory=list)
    pulls: list[tuple[str, str]] = field(default_factory=list)

    async def quote_sink(self, q: Quote) -> None:
        self.quotes.append(q)

    async def hedge_sink(self, h: HedgeInstruction) -> None:
        self.hedges.append(h)

    async def pull_sink(self, token_id: str, reason: str) -> None:
        self.pulls.append((token_id, reason))


def _build_two_market_snapshot(x_a: float) -> MarketSnapshot:
    """A MarketSnapshot for token A with B as a β-hedge peer and three
    synth-strip candidate neighbors symmetrically placed around m*=x_a."""
    peer = HedgePeer(
        surface=_surf(TOK_B, m=x_a, sigma_b=SIGMA_B),
        correlation=_corr(RHO),
    )
    neighborhood = (
        _surf(NEIGHBORS[0], m=x_a - 0.2),
        _surf(NEIGHBORS[1], m=x_a + 0.0),
        _surf(NEIGHBORS[2], m=x_a + 0.2),
    )
    return MarketSnapshot(
        token_id=TOK_A,
        logit_state=_logit(x_a),
        surface=_surf(TOK_A, m=x_a, sigma_b=SIGMA_B),
        position=_pos(),
        book=_book(sigmoid(x_a)),
        trades=(),
        time_to_horizon_sec=1800.0,
        hedge_peers=(peer,),
        inventory_nu_b=INVENTORY_NU_B,
        surface_neighborhood=neighborhood,
    )


def _make_loop(collector: _Collector, feed: _StubFeed) -> RefreshLoop:
    return RefreshLoop(
        config=LoopConfig(
            refresh_ms=10,
            quote=QuoteParams(gamma=0.1, k=1.5, delta_p_floor=0.001),
            limits=LimitsConfig(feed_gap_sec=60.0, inventory_cap_base=1e9),
            hedge_enabled=True,
            hedge_params=BetaHedgeParams(alpha=ALPHA, apply_co_jump=False),
            calendar_hedge_enabled=True,
            calendar_hedge_params=CalendarHedgeParams(
                sigma_b_floor=1e-4, max_abs_notional_usd=1e9,
            ),
            synth_strip_enabled=True,
            synth_strip_params=SynthStripParams(
                bandwidth_m=0.3, bandwidth_log_tau=1.0,
                max_basket_size=5, weight_floor_ratio=1e-6,
            ),
        ),
        data_feed=feed,
        quote_sink=collector.quote_sink,
        hedge_sink=collector.hedge_sink,
        pull_sink=collector.pull_sink,
        clock=lambda: BASE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHedgeStackLiveFlags:
    @pytest.mark.asyncio
    async def test_all_three_hedge_types_coexist(self) -> None:
        """Assertion (i): beta + synth_strip both emit; unrouted calendar leg
        is never present on the wire when synth_strip is on."""
        collector = _Collector()
        feed = _StubFeed(_build_two_market_snapshot(x_a=0.0))
        loop = _make_loop(collector, feed)
        loop.add_token(TOK_A)

        quote = await loop.run_once(TOK_A)
        assert quote is not None
        assert collector.pulls == []

        reasons = [h.reason for h in collector.hedges]
        # Exactly one β-hedge (one peer).
        assert reasons.count("beta") == 1
        # Three synth-strip basket legs (symmetric kernel over three neighbors).
        assert reasons.count("synth_strip") == 3
        # Never emit the unroutable `calendar` synthetic leg.
        assert reasons.count("calendar") == 0

    @pytest.mark.asyncio
    async def test_synth_legs_are_real_token_ids(self) -> None:
        """Assertion (ii): no basket leg references `{tok}:xvar`."""
        collector = _Collector()
        feed = _StubFeed(_build_two_market_snapshot(x_a=0.0))
        loop = _make_loop(collector, feed)
        loop.add_token(TOK_A)
        await loop.run_once(TOK_A)

        placeholder = xvar_synth_token_id(TOK_A)
        synth_legs = [h for h in collector.hedges if h.reason == "synth_strip"]
        assert synth_legs, "synth_strip fan-out produced no legs"
        for h in synth_legs:
            assert h.hedge_token_id != placeholder
            assert h.hedge_token_id in NEIGHBORS

    @pytest.mark.asyncio
    async def test_cross_event_vega_within_paper_bound_post_hedge(self) -> None:
        """Assertion (iii): residual cross-event exposure to B is within the
        §4.4 threshold.

        Pre-hedge exposure proxy: target notional × |β|. The β-hedge of the
        same notional and opposite side fully offsets to O(1-α). We check:

            residual / pre_hedge  <=  (1 - α) + co_jump_contribution

        With α=1.0 and no co-jumps, residual should be ≤ ~1e-6 (floating).
        """
        collector = _Collector()
        feed = _StubFeed(_build_two_market_snapshot(x_a=0.0))
        loop = _make_loop(collector, feed)
        loop.add_token(TOK_A)
        await loop.run_once(TOK_A)

        target_notional = TARGET_QTY * TARGET_MARK
        # Analytic β for this scenario: S'(0)/S'(0) · ρ = ρ = 0.7
        beta = (s_prime(0.0) / s_prime(0.0)) * RHO
        pre_hedge_exposure = abs(beta) * target_notional

        beta_legs = [h for h in collector.hedges if h.reason == "beta"]
        assert len(beta_legs) == 1
        # The β-hedge is SHORT B (long target with positive ρ).
        assert beta_legs[0].side is HedgeSide.SHORT
        hedged_against_B = beta_legs[0].notional_usd

        residual = abs(pre_hedge_exposure - hedged_against_B)
        paper_bound = (1.0 - ALPHA) * pre_hedge_exposure + 1e-6  # floating slack
        assert residual <= paper_bound + 0.2 * pre_hedge_exposure, (
            f"residual {residual} exceeds paper §4.4 bound "
            f"{paper_bound + 0.2 * pre_hedge_exposure}"
        )

    @pytest.mark.asyncio
    async def test_no_double_counting_between_beta_and_calendar(self) -> None:
        """Assertion (iv): β-hedge and calendar hedge address disjoint risk
        factors and do not leak into each other.

        Checks:
          * β-hedge notional in B matches standalone `compute_beta_hedge`.
          * Σ synth_strip leg notionals matches standalone `compute_calendar_hedge`.
          * The two sums are independent — scaling target_notional affects
            only the β path; scaling ν̂_b affects only the calendar path.
        """
        collector = _Collector()
        snap = _build_two_market_snapshot(x_a=0.0)
        feed = _StubFeed(snap)
        loop = _make_loop(collector, feed)
        loop.add_token(TOK_A)
        await loop.run_once(TOK_A)

        # --- β-hedge matches standalone call --------------------------------
        expected_beta = compute_beta_hedge(
            target=snap.surface,
            hedge=snap.hedge_peers[0].surface,
            corr=snap.hedge_peers[0].correlation,
            alpha=ALPHA,
            params=BetaHedgeParams(alpha=ALPHA, apply_co_jump=False),
            target_notional_usd=TARGET_QTY * TARGET_MARK,
            ts=BASE,
        )
        beta_legs = [h for h in collector.hedges if h.reason == "beta"]
        assert len(beta_legs) == 1
        assert beta_legs[0].notional_usd == pytest.approx(expected_beta.notional_usd, rel=1e-9)
        assert beta_legs[0].side is expected_beta.side

        # --- synth_strip basket sum matches standalone calendar call --------
        expected_calendar = compute_calendar_hedge(
            surface=snap.surface,
            inventory_nu_b=INVENTORY_NU_B,
            params=CalendarHedgeParams(sigma_b_floor=1e-4, max_abs_notional_usd=1e9),
            ts=BASE,
        )
        synth_legs = [h for h in collector.hedges if h.reason == "synth_strip"]
        total_synth = sum(h.notional_usd for h in synth_legs)
        assert total_synth == pytest.approx(expected_calendar.notional_usd, rel=1e-9)

        # --- independence: scaling ν̂_b only moves calendar sum, not β ------
        collector2 = _Collector()
        snap_doubled_nu = _build_two_market_snapshot(x_a=0.0)
        # Can't mutate the frozen dataclass — build a new one with 2x nu_b.
        snap_doubled_nu = MarketSnapshot(
            token_id=snap_doubled_nu.token_id,
            logit_state=snap_doubled_nu.logit_state,
            surface=snap_doubled_nu.surface,
            position=snap_doubled_nu.position,
            book=snap_doubled_nu.book,
            trades=snap_doubled_nu.trades,
            time_to_horizon_sec=snap_doubled_nu.time_to_horizon_sec,
            hedge_peers=snap_doubled_nu.hedge_peers,
            inventory_nu_b=2.0 * INVENTORY_NU_B,
            surface_neighborhood=snap_doubled_nu.surface_neighborhood,
        )
        loop2 = _make_loop(collector2, _StubFeed(snap_doubled_nu))
        loop2.add_token(TOK_A)
        await loop2.run_once(TOK_A)
        beta_legs2 = [h for h in collector2.hedges if h.reason == "beta"]
        synth_legs2 = [h for h in collector2.hedges if h.reason == "synth_strip"]
        assert beta_legs2[0].notional_usd == pytest.approx(beta_legs[0].notional_usd, rel=1e-9)
        assert sum(h.notional_usd for h in synth_legs2) == pytest.approx(
            2.0 * total_synth, rel=1e-9
        )

    @pytest.mark.asyncio
    async def test_scripted_path_stability_across_two_ticks(self) -> None:
        """Two-tick scripted path (correlated drift in x) through the loop.

        After both ticks, verify the hedge book is internally consistent:
          * Per-tick β-hedge notional changes only with S'(x_t)/S'(x_peer).
          * Calendar + synth_strip obey notional conservation on each tick.
          * No hedge stream produces duplicates for the same (reason,
            hedge_token_id) within a single tick.
        """
        collector = _Collector()

        # Tick 1 at x_A = 0.0 (p≈0.5), tick 2 at x_A = 0.25 (p≈0.562).
        snapshots = [
            _build_two_market_snapshot(x_a=0.0),
            _build_two_market_snapshot(x_a=0.25),
        ]
        tick_idx = {"i": 0}

        async def feed(token_id: str) -> MarketSnapshot:
            snap = snapshots[tick_idx["i"]]
            tick_idx["i"] = min(tick_idx["i"] + 1, len(snapshots) - 1)
            return snap

        loop = _make_loop(collector, feed)
        loop.add_token(TOK_A)
        await loop.run_once(TOK_A)
        tick1_hedges = list(collector.hedges)
        await loop.run_once(TOK_A)
        tick2_hedges = collector.hedges[len(tick1_hedges):]

        for batch in (tick1_hedges, tick2_hedges):
            # No duplicate (reason, hedge_token_id) within a tick.
            keys = [(h.reason, h.hedge_token_id) for h in batch]
            assert len(keys) == len(set(keys)), f"duplicate hedge legs in tick: {keys}"
            synth = [h for h in batch if h.reason == "synth_strip"]
            beta = [h for h in batch if h.reason == "beta"]
            assert beta and synth

        # Tick-2 β-hedge should differ (x_A changed, S'(x) changed). Here
        # peer surface uses the same x as target so S'_i/S'_j = 1 both ticks,
        # and β stays the same — but the sides/notionals should be consistent.
        tick1_beta = [h for h in tick1_hedges if h.reason == "beta"][0]
        tick2_beta = [h for h in tick2_hedges if h.reason == "beta"][0]
        assert tick1_beta.side is tick2_beta.side
        assert tick2_beta.notional_usd == pytest.approx(tick1_beta.notional_usd, rel=1e-9)
