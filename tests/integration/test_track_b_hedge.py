"""Track B Stage-2 integration test (paper §4.4).

Drives the refresh loop with a MarketSnapshot carrying hedge peers and
verifies:
  (1) hedge_enabled=False → no HedgeInstruction emitted
  (2) hedge_enabled=True  → HedgeInstruction emitted with correct side/size
  (3) zero position → no hedge (nothing to offset)
  (4) peer near the boundary → zero-notional hedge suppressed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from blksch.mm.hedge.beta import BetaHedgeParams
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


def _surf(token_id: str, x: float = 0.0, sigma_b: float = 0.3) -> SurfacePoint:
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


def _corr(rho: float = 0.6) -> CorrelationEntry:
    return CorrelationEntry(
        token_id_i=TOK, token_id_j=PEER,
        rho=rho, co_jump_lambda=0.0, co_jump_m2=0.0, ts=_ts(),
    )


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


def _loop(snap: MarketSnapshot, collector: _Collector, *, hedge_enabled: bool) -> RefreshLoop:
    loop = RefreshLoop(
        config=LoopConfig(
            refresh_ms=10,
            quote=QuoteParams(gamma=0.1, k=1.5, delta_p_floor=0.001),
            limits=LimitsConfig(feed_gap_sec=60.0),
            hedge_enabled=hedge_enabled,
            hedge_params=BetaHedgeParams(alpha=1.0),
        ),
        data_feed=_StubFeed(snap),
        quote_sink=collector.quote_sink,
        hedge_sink=collector.hedge_sink,
        pull_sink=collector.pull_sink,
        clock=lambda: _ts(0),
    )
    loop.add_token(TOK)
    return loop


class TestRefreshLoopHedge:
    @pytest.mark.asyncio
    async def test_flag_off_no_hedge(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0),
            surface=_surf(TOK, x=0.0),
            position=_pos(qty=100.0, mark=0.5),
            book=_book(),
            trades=(),
            hedge_peers=(HedgePeer(surface=_surf(PEER, x=0.0), correlation=_corr(0.6)),),
        )
        loop = _loop(snap, collector, hedge_enabled=False)
        await loop.run_once(TOK)
        assert collector.quotes  # quote still emitted
        assert collector.hedges == []

    @pytest.mark.asyncio
    async def test_flag_on_emits_hedge(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0),
            surface=_surf(TOK, x=0.0),           # S'=0.25
            position=_pos(qty=100.0, mark=0.5),  # target notional = $50
            book=_book(),
            trades=(),
            hedge_peers=(HedgePeer(surface=_surf(PEER, x=0.0), correlation=_corr(0.6)),),
        )
        loop = _loop(snap, collector, hedge_enabled=True)
        await loop.run_once(TOK)
        assert len(collector.hedges) == 1
        h = collector.hedges[0]
        # β = 1 · (0.25/0.25) · 0.6 = 0.6; notional = |β| · $50 = $30
        assert h.notional_usd == pytest.approx(30.0)
        assert h.source_token_id == TOK
        assert h.hedge_token_id == PEER
        assert h.side is HedgeSide.SHORT
        assert h.reason == "beta"

    @pytest.mark.asyncio
    async def test_zero_position_no_hedge(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0),
            surface=_surf(TOK, x=0.0),
            position=_pos(qty=0.0, mark=0.5),
            book=_book(),
            hedge_peers=(HedgePeer(surface=_surf(PEER, x=0.0), correlation=_corr(0.6)),),
        )
        loop = _loop(snap, collector, hedge_enabled=True)
        await loop.run_once(TOK)
        assert collector.hedges == []

    @pytest.mark.asyncio
    async def test_boundary_peer_suppresses_hedge(self) -> None:
        """Peer sitting at p≈1 has S' below floor; zero-notional instruction is dropped."""
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0),
            surface=_surf(TOK, x=0.0),
            position=_pos(qty=100.0, mark=0.5),
            book=_book(),
            hedge_peers=(HedgePeer(surface=_surf(PEER, x=15.0), correlation=_corr(0.6)),),
        )
        loop = _loop(snap, collector, hedge_enabled=True)
        await loop.run_once(TOK)
        assert collector.hedges == []

    @pytest.mark.asyncio
    async def test_short_position_flips_hedge_side(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0),
            surface=_surf(TOK, x=0.0),
            position=_pos(qty=-100.0, mark=0.5),
            book=_book(),
            hedge_peers=(HedgePeer(surface=_surf(PEER, x=0.0), correlation=_corr(0.6)),),
        )
        loop = _loop(snap, collector, hedge_enabled=True)
        await loop.run_once(TOK)
        assert len(collector.hedges) == 1
        # Positive β with short target ⇒ LONG hedge (buy B to offset short in A).
        assert collector.hedges[0].side is HedgeSide.LONG

    @pytest.mark.asyncio
    async def test_multiple_peers_all_emit(self) -> None:
        collector = _Collector()
        peer_c = "0xC"
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0),
            surface=_surf(TOK, x=0.0),
            position=_pos(qty=100.0, mark=0.5),
            book=_book(),
            hedge_peers=(
                HedgePeer(
                    surface=_surf(PEER, x=0.0),
                    correlation=CorrelationEntry(
                        token_id_i=TOK, token_id_j=PEER,
                        rho=0.5, co_jump_lambda=0.0, co_jump_m2=0.0, ts=_ts(),
                    ),
                ),
                HedgePeer(
                    surface=_surf(peer_c, x=0.0),
                    correlation=CorrelationEntry(
                        token_id_i=TOK, token_id_j=peer_c,
                        rho=-0.3, co_jump_lambda=0.0, co_jump_m2=0.0, ts=_ts(),
                    ),
                ),
            ),
        )
        loop = _loop(snap, collector, hedge_enabled=True)
        await loop.run_once(TOK)
        assert len(collector.hedges) == 2
        sides = {h.hedge_token_id: h.side for h in collector.hedges}
        assert sides[PEER] is HedgeSide.SHORT   # ρ>0 ⇒ short hedge for long target
        assert sides[peer_c] is HedgeSide.LONG  # ρ<0 ⇒ long hedge for long target
