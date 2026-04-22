"""Unit tests for mm/guards.py — execution hygiene (paper §4.2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blksch.mm.guards import (
    GuardState,
    NewsGuard,
    NewsWindow,
    QueueMonitor,
    ToxicityMonitor,
)
from blksch.schemas import TradeSide, TradeTick


def _t(sec: float = 0.0) -> datetime:
    return datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


def _trade(side: TradeSide, size: float = 10.0, sec: float = 0.0) -> TradeTick:
    return TradeTick(
        token_id="t",
        price=0.5,
        size=size,
        aggressor_side=side,
        ts=_t(sec),
    )


class TestToxicityMonitor:
    def test_balanced_flow_low_vpin(self) -> None:
        m = ToxicityMonitor(bucket_volume=20.0, n_buckets=5)
        for i in range(20):
            side = TradeSide.BUY if i % 2 == 0 else TradeSide.SELL
            m.ingest(_trade(side, 2.0, sec=i))
        assert m.vpin() < 0.3
        assert not m.is_toxic()

    def test_one_sided_flow_high_vpin(self) -> None:
        m = ToxicityMonitor(bucket_volume=20.0, n_buckets=5, toxicity_threshold=0.4, pull_threshold=0.8)
        for i in range(30):
            m.ingest(_trade(TradeSide.BUY, 2.0, sec=i))
        assert m.vpin() == pytest.approx(1.0, abs=0.05)
        assert m.is_toxic()
        assert m.should_pull()

    def test_partial_bucket_no_output(self) -> None:
        m = ToxicityMonitor(bucket_volume=100.0, n_buckets=5)
        m.ingest(_trade(TradeSide.BUY, 10.0))
        assert m.vpin() == 0.0


class TestNewsGuard:
    def test_scheduled_widen(self) -> None:
        w = NewsWindow(start=_t(600), end=_t(900), pre_buffer_sec=300)
        g = NewsGuard(widen_factor=3.0, windows=[w])
        assert g.widen_multiplier(_t(0)) == 1.0  # far before
        assert g.widen_multiplier(_t(301)) == 3.0  # inside pre-buffer
        assert g.widen_multiplier(_t(750)) == 3.0  # in window
        assert g.widen_multiplier(_t(1000)) == 1.0  # after

    def test_unscheduled_pause(self) -> None:
        g = NewsGuard()
        assert not g.is_paused(_t(0))
        g.note_unscheduled_jump(_t(10), pause_for_sec=30)
        assert g.is_paused(_t(20))
        assert not g.is_paused(_t(50))

    def test_unscheduled_pause_extends(self) -> None:
        g = NewsGuard()
        g.note_unscheduled_jump(_t(10), pause_for_sec=30)
        g.note_unscheduled_jump(_t(20), pause_for_sec=30)  # longer
        assert g.is_paused(_t(45))


class TestQueueMonitor:
    def test_no_resting_mid_no_replace(self) -> None:
        q = QueueMonitor()
        assert not q.should_replace(0.50)

    def test_drift_triggers_replace(self) -> None:
        q = QueueMonitor(mid_drift_ticks=2.0, tick_size=0.01)
        q.set_resting_mid(0.50)
        assert not q.should_replace(0.51)  # 1 tick drift
        assert q.should_replace(0.53)  # 3 ticks

    def test_dropped_from_top_replaces(self) -> None:
        q = QueueMonitor()
        q.set_resting_mid(0.50)
        q.mark_dropped_from_top()
        assert q.should_replace(0.50)


class TestGuardState:
    def test_clean(self) -> None:
        s = GuardState()
        d = s.decide(_t(0))
        assert d.spread_widen_factor == 1.0
        assert not d.pull_quotes
        assert d.reason == "clean"

    def test_toxicity_pulls(self) -> None:
        s = GuardState(toxicity=ToxicityMonitor(bucket_volume=10.0, n_buckets=3, pull_threshold=0.5))
        s.ingest_trades([_trade(TradeSide.BUY, 4.0, sec=i) for i in range(10)])
        d = s.decide(_t(0))
        assert d.pull_quotes
        assert d.reason == "toxicity"

    def test_news_widens(self) -> None:
        window = NewsWindow(start=_t(60), end=_t(120), pre_buffer_sec=30)
        s = GuardState(news=NewsGuard(widen_factor=2.5, windows=[window]))
        d = s.decide(_t(45))  # inside pre-buffer
        assert not d.pull_quotes
        assert d.spread_widen_factor == 2.5
        assert d.reason == "news_widen"

    def test_unscheduled_pause_wins_over_toxicity(self) -> None:
        s = GuardState(
            toxicity=ToxicityMonitor(bucket_volume=10.0, n_buckets=3, pull_threshold=0.5),
            news=NewsGuard(),
        )
        s.ingest_trades([_trade(TradeSide.BUY, 4.0, sec=i) for i in range(10)])
        s.news.note_unscheduled_jump(_t(0), pause_for_sec=60)
        d = s.decide(_t(10))
        assert d.pull_quotes
        assert d.reason == "news_pause"
