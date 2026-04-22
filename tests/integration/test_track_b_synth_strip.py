"""Track B Stage-3 integration test: synth_strip fan-out in refresh_loop.

Asserts:
  (1) synth_strip_enabled=False → the synthetic `{tok}:xvar` leg is emitted as-is
  (2) synth_strip_enabled=True  + neighborhood present → fan-out into real tokens
  (3) synth_strip_enabled=True  + no neighborhood      → hedge suppressed
  (4) notional conservation: Σ leg.notional_usd == original calendar notional
  (5) side inheritance: all legs inherit calendar side (positive weights)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from blksch.mm.hedge.beta import BetaHedgeParams
from blksch.mm.hedge.calendar import CalendarHedgeParams
from blksch.mm.hedge.synth_strip import SynthStripParams
from blksch.mm.limits import LimitsConfig
from blksch.mm.quote import QuoteParams
from blksch.mm.refresh_loop import LoopConfig, MarketSnapshot, RefreshLoop
from blksch.schemas import (
    BookSnap,
    HedgeInstruction,
    HedgeSide,
    LogitState,
    Position,
    PriceLevel,
    Quote,
    SurfacePoint,
)


TOK = "0xA"


def _ts(sec: float = 0.0) -> datetime:
    return datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


def _surf(token_id: str, m: float = 0.0, tau: float = 3600.0, sigma_b: float = 0.5) -> SurfacePoint:
    return SurfacePoint(
        token_id=token_id, tau=tau, m=m, sigma_b=sigma_b,
        **{"lambda": 0.1}, s2_j=0.0, ts=_ts(),
    )


def _logit(x: float = 0.0) -> LogitState:
    return LogitState(token_id=TOK, x_hat=x, sigma_eta2=0.01, ts=_ts())


def _book(mid: float = 0.5) -> BookSnap:
    return BookSnap(
        token_id=TOK,
        bids=[PriceLevel(price=mid - 0.01, size=100.0)],
        asks=[PriceLevel(price=mid + 0.01, size=100.0)],
        ts=_ts(),
    )


def _pos(qty: float = 100.0, mark: float = 0.5) -> Position:
    return Position(token_id=TOK, qty=qty, avg_entry=0.5, mark=mark, realized_pnl_usd=0.0)


@dataclass
class _StubFeed:
    snap: MarketSnapshot | None

    async def __call__(self, token_id: str) -> MarketSnapshot | None:
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


def _loop(
    snap: MarketSnapshot,
    collector: _Collector,
    *,
    synth_enabled: bool,
) -> RefreshLoop:
    loop = RefreshLoop(
        config=LoopConfig(
            refresh_ms=10,
            quote=QuoteParams(gamma=0.1, k=1.5, delta_p_floor=0.001),
            limits=LimitsConfig(feed_gap_sec=60.0),
            hedge_enabled=False,  # β-hedge off for these tests
            hedge_params=BetaHedgeParams(alpha=1.0),
            calendar_hedge_enabled=True,
            calendar_hedge_params=CalendarHedgeParams(
                sigma_b_floor=1e-4, max_abs_notional_usd=1e9,
            ),
            synth_strip_enabled=synth_enabled,
            synth_strip_params=SynthStripParams(bandwidth_m=0.5, bandwidth_log_tau=0.5),
        ),
        data_feed=_StubFeed(snap),
        quote_sink=collector.quote_sink,
        hedge_sink=collector.hedge_sink,
        pull_sink=collector.pull_sink,
        clock=lambda: _ts(0),
    )
    loop.add_token(TOK)
    return loop


class TestSynthStripRefreshLoop:
    @pytest.mark.asyncio
    async def test_synth_off_emits_synthetic_leg(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, m=0.0, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=10.0,
            surface_neighborhood=(
                _surf("n1", m=-0.2), _surf("n2", m=0.2),
            ),
        )
        loop = _loop(snap, collector, synth_enabled=False)
        await loop.run_once(TOK)
        assert len(collector.hedges) == 1
        h = collector.hedges[0]
        # Unrouted synthetic leg when synth_strip is off.
        assert h.hedge_token_id == f"{TOK}:xvar"
        assert h.reason == "calendar"
        assert h.notional_usd == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_synth_on_fans_out_into_real_tokens(self) -> None:
        collector = _Collector()
        neighborhood = (
            _surf("n_minus", m=-0.2, sigma_b=0.5),
            _surf("n_mid",   m= 0.0, sigma_b=0.5),
            _surf("n_plus",  m= 0.2, sigma_b=0.5),
        )
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, m=0.0, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=10.0,
            surface_neighborhood=neighborhood,
        )
        loop = _loop(snap, collector, synth_enabled=True)
        await loop.run_once(TOK)
        # All three neighbors resolved (symmetric kernel, no trimming).
        assert len(collector.hedges) == 3
        for h in collector.hedges:
            assert h.reason == "synth_strip"
            assert h.hedge_token_id != f"{TOK}:xvar"
            assert h.side is HedgeSide.SHORT  # inherited from calendar side (ν_b>0)
        # Notional conservation: sum of legs == calendar notional (20 USD).
        total = sum(h.notional_usd for h in collector.hedges)
        assert total == pytest.approx(20.0, rel=1e-9)

    @pytest.mark.asyncio
    async def test_synth_on_empty_neighborhood_suppresses(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, m=0.0, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=10.0,
            surface_neighborhood=(),  # no neighbors
        )
        loop = _loop(snap, collector, synth_enabled=True)
        await loop.run_once(TOK)
        # Neither the synthetic leg nor any basket legs; loop suppresses.
        assert collector.hedges == []

    @pytest.mark.asyncio
    async def test_self_token_in_neighborhood_is_dropped(self) -> None:
        """If the target's own token appears in the neighborhood, the basket
        must not reference it (router can't hedge a token against itself)."""
        collector = _Collector()
        neighborhood = (
            _surf(TOK, m=0.0, sigma_b=0.5),          # should be excluded
            _surf("peer", m=0.1, sigma_b=0.5),
        )
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, m=0.0, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=10.0,
            surface_neighborhood=neighborhood,
        )
        loop = _loop(snap, collector, synth_enabled=True)
        await loop.run_once(TOK)
        for h in collector.hedges:
            assert h.hedge_token_id != TOK
        assert any(h.hedge_token_id == "peer" for h in collector.hedges)

    @pytest.mark.asyncio
    async def test_negative_nu_b_long_legs(self) -> None:
        collector = _Collector()
        neighborhood = (
            _surf("n_mid", m=0.0, sigma_b=0.5),
            _surf("n_plus", m=0.2, sigma_b=0.5),
        )
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, m=0.0, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=-10.0,  # book short vol → calendar LONG
            surface_neighborhood=neighborhood,
        )
        loop = _loop(snap, collector, synth_enabled=True)
        await loop.run_once(TOK)
        assert collector.hedges
        assert all(h.side is HedgeSide.LONG for h in collector.hedges)
