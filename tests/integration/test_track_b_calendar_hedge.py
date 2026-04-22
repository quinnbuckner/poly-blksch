"""Track B Stage-3 integration test (paper §4.3).

Drives the refresh loop with a MarketSnapshot carrying an `inventory_nu_b`
value and verifies:
  (1) calendar_hedge_enabled=False → no HedgeInstruction emitted
  (2) flag on + ν̂_b > 0          → SHORT x-variance strip of correct size
  (3) flag on + ν̂_b < 0          → LONG  x-variance strip
  (4) inventory_nu_b = None        → loop skips step 6 (Stage-3 not supplied)
  (5) ν̂_b = 0                     → zero-notional suppressed
  (6) beta + calendar together     → both emitted independently
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from blksch.mm.hedge.beta import BetaHedgeParams
from blksch.mm.hedge.calendar import CalendarHedgeParams, xvar_synth_token_id
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


TOK = "0xA"
PEER = "0xB"


def _ts(sec: float = 0.0) -> datetime:
    return datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


def _surf(token_id: str, x: float = 0.0, sigma_b: float = 0.5) -> SurfacePoint:
    return SurfacePoint(
        token_id=token_id, tau=3600.0, m=x, sigma_b=sigma_b,
        **{"lambda": 0.1}, s2_j=0.01, ts=_ts(),
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
    calendar_enabled: bool,
    beta_enabled: bool = False,
) -> RefreshLoop:
    loop = RefreshLoop(
        config=LoopConfig(
            refresh_ms=10,
            quote=QuoteParams(gamma=0.1, k=1.5, delta_p_floor=0.001),
            limits=LimitsConfig(feed_gap_sec=60.0),
            hedge_enabled=beta_enabled,
            hedge_params=BetaHedgeParams(alpha=1.0),
            calendar_hedge_enabled=calendar_enabled,
            calendar_hedge_params=CalendarHedgeParams(
                sigma_b_floor=1e-4, max_abs_notional_usd=1e9,
            ),
        ),
        data_feed=_StubFeed(snap),
        quote_sink=collector.quote_sink,
        hedge_sink=collector.hedge_sink,
        pull_sink=collector.pull_sink,
        clock=lambda: _ts(0),
    )
    loop.add_token(TOK)
    return loop


class TestCalendarHedgeRefreshLoop:
    @pytest.mark.asyncio
    async def test_flag_off_no_calendar_hedge(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=10.0,
        )
        loop = _loop(snap, collector, calendar_enabled=False)
        await loop.run_once(TOK)
        assert collector.hedges == []

    @pytest.mark.asyncio
    async def test_flag_on_positive_nu_b_short_hedge(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=10.0,
        )
        loop = _loop(snap, collector, calendar_enabled=True)
        await loop.run_once(TOK)
        assert len(collector.hedges) == 1
        h = collector.hedges[0]
        # N^{x-var} = -10 / 0.5 = -20 ⇒ SHORT, |N|=20
        assert h.side is HedgeSide.SHORT
        assert h.notional_usd == pytest.approx(20.0)
        assert h.reason == "calendar"
        assert h.hedge_token_id == xvar_synth_token_id(TOK)

    @pytest.mark.asyncio
    async def test_flag_on_negative_nu_b_long_hedge(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=-10.0,
        )
        loop = _loop(snap, collector, calendar_enabled=True)
        await loop.run_once(TOK)
        assert len(collector.hedges) == 1
        assert collector.hedges[0].side is HedgeSide.LONG
        assert collector.hedges[0].notional_usd == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_missing_inventory_nu_b_skips(self) -> None:
        """Stage-3 piece not ready yet (nu_b is None) → skip cleanly, no hedge."""
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=None,
        )
        loop = _loop(snap, collector, calendar_enabled=True)
        await loop.run_once(TOK)
        assert collector.hedges == []
        assert collector.quotes  # quote still emitted

    @pytest.mark.asyncio
    async def test_zero_nu_b_suppressed(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, sigma_b=0.5),
            position=_pos(),
            book=_book(),
            inventory_nu_b=0.0,
        )
        loop = _loop(snap, collector, calendar_enabled=True)
        await loop.run_once(TOK)
        assert collector.hedges == []

    @pytest.mark.asyncio
    async def test_beta_and_calendar_emit_independently(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(),
            surface=_surf(TOK, x=0.0, sigma_b=0.5),
            position=_pos(qty=100.0, mark=0.5),
            book=_book(),
            hedge_peers=(
                HedgePeer(
                    surface=_surf(PEER, x=0.0, sigma_b=0.5),
                    correlation=CorrelationEntry(
                        token_id_i=TOK, token_id_j=PEER,
                        rho=0.6, co_jump_lambda=0.0, co_jump_m2=0.0, ts=_ts(),
                    ),
                ),
            ),
            inventory_nu_b=5.0,
        )
        loop = _loop(snap, collector, calendar_enabled=True, beta_enabled=True)
        await loop.run_once(TOK)
        reasons = {h.reason for h in collector.hedges}
        assert reasons == {"beta", "calendar"}
