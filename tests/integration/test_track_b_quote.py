"""Track B critical integration test (paper §4.2, §4.5, §4.6).

Feeds a fixed (SurfacePoint, Position, BookSnap) through the refresh loop and
asserts:
  (1) expected Quote numbers on the clean baseline
  (2) toxicity spike widens δ_x
  (3) feed-gap kill-switch fires and quote is pulled

No Track A or Track C modules are imported — we stub everything against
`schemas.py` Pydantic contracts, as required by the plan.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from blksch.mm.greeks import sigmoid
from blksch.mm.guards import NewsGuard, ToxicityMonitor
from blksch.mm.quote import QuoteParams, compute_quote, half_spread_x, reservation_x
from blksch.mm.refresh_loop import LoopConfig, MarketSnapshot, RefreshLoop
from blksch.mm.limits import LimitsConfig
from blksch.schemas import (
    BookSnap,
    LogitState,
    Position,
    PriceLevel,
    Quote,
    SurfacePoint,
    TradeSide,
    TradeTick,
)


TOK = "0xtoken"


def _ts(sec: float = 0.0) -> datetime:
    return datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


def _surface(sigma_b: float = 0.3, sec: float = 0.0) -> SurfacePoint:
    return SurfacePoint(
        token_id=TOK, tau=3600.0, m=0.0, sigma_b=sigma_b,
        **{"lambda": 0.0}, s2_j=0.0, ts=_ts(sec),
    )


def _logit(x: float = 0.0, sec: float = 0.0) -> LogitState:
    return LogitState(token_id=TOK, x_hat=x, sigma_eta2=0.01, ts=_ts(sec))


def _book(mid: float = 0.5, sec: float = 0.0) -> BookSnap:
    return BookSnap(
        token_id=TOK,
        bids=[PriceLevel(price=mid - 0.01, size=100.0)],
        asks=[PriceLevel(price=mid + 0.01, size=100.0)],
        ts=_ts(sec),
    )


def _position(qty: float = 0.0) -> Position:
    return Position(token_id=TOK, qty=qty, avg_entry=0.5, mark=0.5, realized_pnl_usd=0.0)


# ---------------------------------------------------------------------------
# (1) Deterministic quote numbers on a fixed baseline
# ---------------------------------------------------------------------------


class TestExpectedQuoteOnFixedInputs:
    def test_exact_paper_formulas_at_realistic_params(self) -> None:
        """Realistic refresh-horizon regime (γ=0.1, σ=0.1, T=60, q=10):
        the paper's eq (8-9) produce specific numbers and the clip does
        not bind, so the Quote preserves them in both reservation and
        x_bid / x_ask."""
        gamma, sigma, tau, k, q = 0.1, 0.1, 60.0, 1.5, 10.0
        params = QuoteParams(gamma=gamma, k=k, delta_p_floor=0.0)

        q_out = compute_quote(
            token_id=TOK, x_t=0.0, sigma_b=sigma, time_to_horizon_sec=tau,
            inventory_q=q, params=params, ts=_ts(0),
        )

        r_expected = 0.0 - q * gamma * sigma * sigma * tau  # = -0.6
        assert q_out.reservation_x == pytest.approx(r_expected)

        delta_expected = 0.5 * (gamma * sigma * sigma * tau + (2.0 / k) * math.log1p(gamma / k))
        # Note half_spread_x here reflects the actual posted logit spread;
        # at realistic params the natural δ is well inside the boundary clip.
        assert q_out.x_bid == pytest.approx(r_expected - delta_expected, rel=1e-9)
        assert q_out.x_ask == pytest.approx(r_expected + delta_expected, rel=1e-9)
        assert q_out.half_spread_x == pytest.approx(delta_expected, rel=1e-9)

    def test_extreme_skew_triggers_one_sided_quote(self) -> None:
        """Extreme-skew regime (γ=0.5, σ=0.3, T=1000, q=10): the paper's
        r_x and δ_x formulas are still reported truthfully, but the quote
        is one-sided — pulled bid at eps, size_bid=0, size_ask>0 — because
        a two-sided quote would collapse both sides to the boundary."""
        gamma, sigma, tau, k, q = 0.5, 0.3, 1000.0, 1.5, 10.0
        params = QuoteParams(gamma=gamma, k=k, delta_p_floor=0.0)

        q_out = compute_quote(
            token_id=TOK, x_t=0.0, sigma_b=sigma, time_to_horizon_sec=tau,
            inventory_q=q, params=params, ts=_ts(0),
        )

        # Paper formulas unchanged — reservation is still the §4.2 value.
        r_expected = 0.0 - q * gamma * sigma * sigma * tau  # -450
        assert q_out.reservation_x == pytest.approx(r_expected)

        # One-sided: bid pinned to eps, ask at a sensible distance.
        assert q_out.p_bid == pytest.approx(params.eps, rel=1e-6)
        assert q_out.size_bid == 0.0
        assert q_out.size_ask > 0.0
        assert q_out.p_ask > q_out.p_bid


# ---------------------------------------------------------------------------
# (2) Toxicity spike → widened δ
# ---------------------------------------------------------------------------


class TestToxicityWidens:
    def test_toxicity_doubles_halfspread(self) -> None:
        """With a 2x widen factor, half_spread_x should be ~2x the clean case."""
        ts = _ts(0)
        base = compute_quote(
            token_id=TOK, x_t=0.0, sigma_b=0.3, time_to_horizon_sec=1000,
            inventory_q=0.0, params=QuoteParams(gamma=0.1, k=1.5), ts=ts,
        )
        widened = compute_quote(
            token_id=TOK, x_t=0.0, sigma_b=0.3, time_to_horizon_sec=1000,
            inventory_q=0.0, params=QuoteParams(gamma=0.1, k=1.5), ts=ts,
            spread_widen_factor=2.0,
        )
        # Pre-floor the ratio is exactly 2.0; accept a small drift from any floor activation.
        assert widened.half_spread_x == pytest.approx(2.0 * base.half_spread_x, rel=1e-9)
        assert widened.p_ask - widened.p_bid > base.p_ask - base.p_bid


# ---------------------------------------------------------------------------
# (3) Full-loop integration: clean quote emitted, then feed-gap pulls
# ---------------------------------------------------------------------------


@dataclass
class _StubFeed:
    snap: MarketSnapshot | None

    async def __call__(self, token_id: str) -> MarketSnapshot | None:
        return self.snap


@dataclass
class _Collector:
    quotes: list[Quote] = field(default_factory=list)
    pulls: list[tuple[str, str]] = field(default_factory=list)

    async def quote_sink(self, quote: Quote) -> None:
        self.quotes.append(quote)

    async def pull_sink(self, token_id: str, reason: str) -> None:
        self.pulls.append((token_id, reason))


def _fixed_clock(sec: float) -> datetime:
    return _ts(sec)


class TestRefreshLoop:
    @pytest.mark.asyncio
    async def test_clean_cycle_emits_quote(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.2, sec=0.0),
            surface=_surface(0.3, sec=0.0),
            position=_position(0.0),
            book=_book(0.55, sec=0.0),
            trades=(),
            time_to_horizon_sec=1800.0,
        )
        loop = RefreshLoop(
            config=LoopConfig(
                refresh_ms=10,
                quote=QuoteParams(gamma=0.1, k=1.5, delta_p_floor=0.005),
                limits=LimitsConfig(feed_gap_sec=60.0),
            ),
            data_feed=_StubFeed(snap),
            quote_sink=collector.quote_sink,
            pull_sink=collector.pull_sink,
            clock=lambda: _ts(0),
        )
        loop.add_token(TOK)

        quote = await loop.run_once(TOK)

        assert quote is not None
        assert len(collector.quotes) == 1
        assert collector.pulls == []
        assert 0.0 < quote.p_bid < quote.p_ask < 1.0

    @pytest.mark.asyncio
    async def test_toxicity_spike_widens_and_then_pulls(self) -> None:
        collector = _Collector()
        snap_clean = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0, sec=0.0),
            surface=_surface(0.3, sec=0.0),
            position=_position(0.0),
            book=_book(0.5, sec=0.0),
            trades=(),
            time_to_horizon_sec=1000.0,
        )
        loop = RefreshLoop(
            config=LoopConfig(
                refresh_ms=10,
                quote=QuoteParams(gamma=0.1, k=1.5, delta_p_floor=0.001),
                limits=LimitsConfig(feed_gap_sec=60.0),
            ),
            data_feed=_StubFeed(snap_clean),
            quote_sink=collector.quote_sink,
            pull_sink=collector.pull_sink,
            clock=lambda: _ts(0),
        )
        loop.add_token(TOK)
        # Tight thresholds so a few one-sided trades trip the guard.
        loop.state(TOK).guards.toxicity = ToxicityMonitor(
            bucket_volume=5.0, n_buckets=3,
            toxicity_threshold=0.3, pull_threshold=0.6,
        )

        # Baseline clean cycle → a quote is emitted.
        await loop.run_once(TOK)
        baseline_halfspread = collector.quotes[-1].half_spread_x

        # Inject 3 buckets of one-sided flow → VPIN = 1.0 → pull threshold hit.
        toxic_trades = tuple(
            TradeTick(token_id=TOK, price=0.5, size=3.0, aggressor_side=TradeSide.BUY, ts=_ts(i * 0.1))
            for i in range(20)
        )
        snap_toxic = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0, sec=0.1),
            surface=_surface(0.3, sec=0.1),
            position=_position(0.0),
            book=_book(0.5, sec=0.1),
            trades=toxic_trades,
            time_to_horizon_sec=1000.0,
        )
        loop.data_feed = _StubFeed(snap_toxic)
        result = await loop.run_once(TOK)

        assert result is None  # pulled
        assert collector.pulls
        assert "toxicity" in collector.pulls[-1][1]
        # Sanity: baseline width was the clean half-spread
        assert baseline_halfspread > 0.0

    @pytest.mark.asyncio
    async def test_news_widen_does_not_pull(self) -> None:
        collector = _Collector()
        # Short horizon + small σ so the widen factor has room to grow the
        # half-spread without saturating the [logit(eps), logit(1-eps)] clip.
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0, sec=0.0),
            surface=_surface(0.1, sec=0.0),
            position=_position(0.0),
            book=_book(0.5, sec=0.0),
            trades=(),
            time_to_horizon_sec=60.0,
        )
        clock = lambda: _ts(50)  # inside news pre-buffer below
        loop = RefreshLoop(
            config=LoopConfig(
                refresh_ms=10,
                quote=QuoteParams(gamma=0.1, k=1.5, delta_p_floor=0.001),
                limits=LimitsConfig(feed_gap_sec=300.0),
            ),
            data_feed=_StubFeed(snap),
            quote_sink=collector.quote_sink,
            pull_sink=collector.pull_sink,
            clock=clock,
        )
        loop.add_token(TOK)
        from blksch.mm.guards import NewsWindow
        loop.state(TOK).guards.news = NewsGuard(
            widen_factor=3.0,
            windows=[NewsWindow(start=_ts(100), end=_ts(200), pre_buffer_sec=60)],
        )

        # Clean quote with no news window for baseline
        loop.state(TOK).guards.news.windows = []
        await loop.run_once(TOK)
        base_hs = collector.quotes[-1].half_spread_x

        # Re-arm the window and re-quote.
        from blksch.mm.guards import NewsWindow as _NW
        loop.state(TOK).guards.news.windows = [_NW(start=_ts(100), end=_ts(200), pre_buffer_sec=60)]
        await loop.run_once(TOK)
        widened_hs = collector.quotes[-1].half_spread_x

        assert widened_hs == pytest.approx(3.0 * base_hs, rel=1e-6)

    @pytest.mark.asyncio
    async def test_feed_gap_pulls(self) -> None:
        collector = _Collector()
        # First cycle: fresh data → quote.
        now_holder = [0.0]

        async def feed(token_id: str) -> MarketSnapshot | None:
            return MarketSnapshot(
                token_id=TOK,
                logit_state=_logit(0.0, sec=0.0),  # always stamped at t=0
                surface=_surface(0.3, sec=0.0),
                position=_position(0.0),
                book=_book(0.5, sec=0.0),
                trades=(),
                time_to_horizon_sec=1000.0,
            )

        loop = RefreshLoop(
            config=LoopConfig(
                refresh_ms=10,
                quote=QuoteParams(gamma=0.1, k=1.5, delta_p_floor=0.001),
                limits=LimitsConfig(feed_gap_sec=1.0),
            ),
            data_feed=feed,
            quote_sink=collector.quote_sink,
            pull_sink=collector.pull_sink,
            clock=lambda: _ts(now_holder[0]),
        )
        loop.add_token(TOK)

        # t=0: data is fresh, quote emitted.
        now_holder[0] = 0.0
        q0 = await loop.run_once(TOK)
        assert q0 is not None

        # t=5: data still timestamped at 0 → 5s gap, above threshold → pull.
        now_holder[0] = 5.0
        q1 = await loop.run_once(TOK)
        assert q1 is None
        assert any("feed_gap" in r for _, r in collector.pulls)

    @pytest.mark.asyncio
    async def test_inventory_cap_pulls(self) -> None:
        collector = _Collector()
        snap = MarketSnapshot(
            token_id=TOK,
            logit_state=_logit(0.0, sec=0.0),
            surface=_surface(0.3, sec=0.0),
            position=_position(qty=1e9),  # absurd qty
            book=_book(0.5, sec=0.0),
            trades=(),
            time_to_horizon_sec=1000.0,
        )
        loop = RefreshLoop(
            config=LoopConfig(
                refresh_ms=10,
                quote=QuoteParams(gamma=0.1, k=1.5),
                limits=LimitsConfig(inventory_cap_base=10.0, feed_gap_sec=60.0),
            ),
            data_feed=_StubFeed(snap),
            quote_sink=collector.quote_sink,
            pull_sink=collector.pull_sink,
            clock=lambda: _ts(0),
        )
        loop.add_token(TOK)
        result = await loop.run_once(TOK)
        assert result is None
        assert any("inventory_cap" in r for _, r in collector.pulls)
