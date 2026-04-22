"""Execution hygiene — toxicity, news, queue discipline (paper §4.2).

Three independent guards feed one scalar output (`spread_widen_factor`) and one
boolean (`pull_quotes`) consumed by `refresh_loop.py`:

    1. Toxicity filter — VPIN-style running estimate of order-flow imbalance
       per volume bucket. When it spikes, widen δ_x or pull entirely.
    2. News-window policy — deterministic pre-announcement widening; pause on
       unscheduled jump detector.
    3. Queue discipline — cancel+replace on adverse microstructure (mid drift,
       queue position loss).

Pure-ish: state is kept in small dataclasses the refresh loop owns; all
update methods are deterministic given inputs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Literal

from blksch.schemas import TradeSide, TradeTick

__all__ = [
    "ToxicityMonitor",
    "NewsWindow",
    "NewsGuard",
    "QueueMonitor",
    "GuardState",
    "GuardDecision",
]


# ---------------------------------------------------------------------------
# VPIN toxicity (paper §4.2)
# ---------------------------------------------------------------------------


@dataclass
class ToxicityMonitor:
    """VPIN-style toxicity tracker.

    VPIN = E[|V_buy - V_sell|] / V_bucket, estimated over the trailing N
    buckets of equal volume `bucket_volume`. Values in [0, 1]; high means
    toxic flow (one-sided, likely informed).
    """

    bucket_volume: float = 50.0
    n_buckets: int = 50
    toxicity_threshold: float = 0.4  # widen above this
    pull_threshold: float = 0.7  # pull quotes above this

    _buckets: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    _current_buy: float = 0.0
    _current_sell: float = 0.0

    def __post_init__(self) -> None:
        if self.bucket_volume <= 0:
            raise ValueError("bucket_volume must be positive")
        if self.n_buckets <= 0:
            raise ValueError("n_buckets must be positive")
        self._buckets = deque(maxlen=self.n_buckets)

    def ingest(self, trade: TradeTick) -> None:
        if trade.aggressor_side is TradeSide.BUY:
            self._current_buy += trade.size
        else:
            self._current_sell += trade.size
        # Close buckets whenever cumulative volume crosses the threshold.
        while self._current_buy + self._current_sell >= self.bucket_volume:
            total = self._current_buy + self._current_sell
            imbalance = abs(self._current_buy - self._current_sell) / self.bucket_volume
            self._buckets.append(min(imbalance, 1.0))
            # Carry the overshoot proportionally into the next bucket.
            overshoot = total - self.bucket_volume
            if overshoot > 0 and total > 0:
                buy_share = self._current_buy / total
                self._current_buy = buy_share * overshoot
                self._current_sell = (1.0 - buy_share) * overshoot
            else:
                self._current_buy = 0.0
                self._current_sell = 0.0

    def vpin(self) -> float:
        if not self._buckets:
            return 0.0
        return sum(self._buckets) / len(self._buckets)

    def is_toxic(self) -> bool:
        return self.vpin() >= self.toxicity_threshold

    def should_pull(self) -> bool:
        return self.vpin() >= self.pull_threshold


# ---------------------------------------------------------------------------
# News-window policy (paper §4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsWindow:
    """A scheduled announcement window with pre- and post-buffers."""

    start: datetime
    end: datetime
    pre_buffer_sec: float = 300.0  # start widening 5 min before
    label: str = ""

    def contains(self, t: datetime) -> bool:
        return self.start - timedelta(seconds=self.pre_buffer_sec) <= t <= self.end


@dataclass
class NewsGuard:
    """Scheduled-news pre-widening + unscheduled-jump pause."""

    widen_factor: float = 2.0
    windows: list[NewsWindow] = field(default_factory=list)
    _unscheduled_pause_until: datetime | None = None

    def in_scheduled_window(self, t: datetime) -> bool:
        return any(w.contains(t) for w in self.windows)

    def note_unscheduled_jump(self, t: datetime, pause_for_sec: float = 30.0) -> None:
        until = t + timedelta(seconds=pause_for_sec)
        if self._unscheduled_pause_until is None or until > self._unscheduled_pause_until:
            self._unscheduled_pause_until = until

    def is_paused(self, t: datetime) -> bool:
        return self._unscheduled_pause_until is not None and t < self._unscheduled_pause_until

    def widen_multiplier(self, t: datetime) -> float:
        return self.widen_factor if self.in_scheduled_window(t) else 1.0


# ---------------------------------------------------------------------------
# Queue discipline (paper §4.2)
# ---------------------------------------------------------------------------


@dataclass
class QueueMonitor:
    """Track when a resting quote should be cancelled+replaced.

    Triggers: (i) mid moved more than `mid_drift_ticks`·tick away from our
    quote price since resting; (ii) we're no longer at the top of our price
    level (inferred by the router from venue events — we expose a setter).
    """

    mid_drift_ticks: float = 2.0
    tick_size: float = 0.01
    _resting_mid: float | None = None
    _ever_top_of_queue: bool = True

    def set_resting_mid(self, mid: float | None) -> None:
        self._resting_mid = mid
        self._ever_top_of_queue = True

    def mark_dropped_from_top(self) -> None:
        self._ever_top_of_queue = False

    def should_replace(self, current_mid: float | None) -> bool:
        if self._resting_mid is None or current_mid is None:
            return False
        if not self._ever_top_of_queue:
            return True
        drift = abs(current_mid - self._resting_mid)
        return drift > self.mid_drift_ticks * self.tick_size


# ---------------------------------------------------------------------------
# Aggregate state the refresh loop consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardDecision:
    """The single decision object the refresh loop acts on."""

    spread_widen_factor: float
    pull_quotes: bool
    reason: Literal["clean", "toxicity", "news_pause", "news_widen", "toxicity_widen"]


@dataclass
class GuardState:
    """Bundle of guards owned by the refresh loop."""

    toxicity: ToxicityMonitor = field(default_factory=ToxicityMonitor)
    news: NewsGuard = field(default_factory=NewsGuard)
    queue: QueueMonitor = field(default_factory=QueueMonitor)

    def ingest_trades(self, trades: Iterable[TradeTick]) -> None:
        for t in trades:
            self.toxicity.ingest(t)

    def decide(self, t: datetime) -> GuardDecision:
        """Resolve the three guards into a single widen/pull call.

        Precedence:
          1. Unscheduled jump pause    ⇒ pull
          2. Toxicity pull threshold   ⇒ pull
          3. Toxicity widen threshold  ⇒ widen by news factor if also in window, else news_widen_factor
          4. Scheduled news window     ⇒ widen by news_widen_factor
          5. Otherwise                 ⇒ no change
        """
        if self.news.is_paused(t):
            return GuardDecision(1.0, True, "news_pause")
        if self.toxicity.should_pull():
            return GuardDecision(1.0, True, "toxicity")
        news_mult = self.news.widen_multiplier(t)
        if self.toxicity.is_toxic():
            # Combine multiplicatively so news + toxicity compound.
            return GuardDecision(news_mult * self.news.widen_factor, False, "toxicity_widen")
        if news_mult > 1.0:
            return GuardDecision(news_mult, False, "news_widen")
        return GuardDecision(1.0, False, "clean")
