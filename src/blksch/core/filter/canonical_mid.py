"""Canonical mid p̃_t filter (paper §5.1 — Data Conditioning & Filtering).

Turns a raw ``BookSnap`` stream (optionally annotated with ``TradeTick``
events) into a conditioned, grid-cadenced estimate ``p̃_t`` suitable for the
heteroskedastic Kalman filter in ``core.filter.kalman``.

Pipeline (per paper):
  1. **Canonical mid** — trade-weighted VWAP when recent trades exist,
     otherwise book-mid ``(bid+ask)/2``. Trades are weighted by notional
     ``price * size``, over a sliding window (default 30 s).
  2. **Clipping** — clamp ``p̃`` to ``[ε, 1-ε]`` with ``ε`` from
     ``config/bot.yaml.boundary.eps`` (default ``1e-5``). Critical because the
     logit transform diverges at the boundary.
  3. **Outlier hygiene** — reject ticks where ``|Δlogit|`` exceeds
     ``K · IQR`` of the trailing logit-increment window. Rejections are
     logged; the filter falls back to the last accepted value and does NOT
     drop the grid output — it marks ``rejected_outlier=True``.
  4. **Cadence enforcement** — emit one :class:`CanonicalMid` per grid tick
     at the configured ``kf_grid_hz``. Grid bins ``(T-Δ, T]`` that contain at
     least one accepted tick are ``forward_filled=False`` (fresh); otherwise
     the last accepted value is carried forward and ``forward_filled=True``
     so the downstream Kalman filter can widen measurement variance.

The module deliberately does *not* extend ``schemas.py`` — the downstream
consumer (microstruct + kalman) consumes :class:`CanonicalMid` directly.
Coordinate before touching the shared contracts.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from blksch.schemas import BookSnap, TradeTick

logger = logging.getLogger(__name__)

DEFAULT_EPS = 1e-5
DEFAULT_GRID_HZ = 1.0
DEFAULT_TRADE_WINDOW_SEC = 30.0
DEFAULT_OUTLIER_K_IQR = 4.0
DEFAULT_OUTLIER_WINDOW = 60
DEFAULT_OUTLIER_MIN_HISTORY = 8
DEFAULT_TICK_SIZE = 0.01

MidSource = Literal["book_mid", "trade_vwap", "forward_fill", "outlier_fallback"]


@dataclass(frozen=True)
class CanonicalMid:
    """One grid output of the §5.1 pipeline.

    ``y = logit(p_tilde)`` is pre-computed so downstream consumers don't
    recompute it on every tick.
    """

    token_id: str
    ts: datetime
    p_tilde: float
    y: float
    forward_filled: bool
    rejected_outlier: bool
    trades_in_window: int
    source: MidSource


# ---------------------------------------------------------------------------
# Pure helpers (exported for tests / reuse)
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def clip_p(p: float, eps: float) -> float:
    """Clamp probability to ``[eps, 1-eps]``."""
    if p < eps:
        return eps
    if p > 1.0 - eps:
        return 1.0 - eps
    return p


def trade_vwap(trades: list[TradeTick]) -> float | None:
    """Notional-weighted VWAP of trade prices, or None if no trades."""
    total = 0.0
    weighted = 0.0
    for t in trades:
        notional = t.price * t.size
        if notional <= 0:
            continue
        total += notional
        weighted += notional * t.price
    if total <= 0:
        return None
    return weighted / total


def _quartile(sorted_vals: list[float], q: float) -> float:
    """Linear interpolation quartile on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def iqr(values: list[float]) -> float:
    """Interquartile range Q3-Q1; returns 0 if <2 values."""
    if len(values) < 2:
        return 0.0
    s = sorted(values)
    return _quartile(s, 0.75) - _quartile(s, 0.25)


def _next_grid_boundary(ts: datetime, period: timedelta) -> datetime:
    """Smallest grid boundary strictly greater than ``ts``.

    If ``ts`` is exactly on a grid boundary, return ``ts + period``. This
    keeps the first observation inside the first emitted bin
    ``[prev_grid, first_grid)`` rather than having the first emit collapse
    onto the observation itself.
    """
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    delta_us = int((ts - epoch).total_seconds() * 1_000_000)
    period_us = int(period.total_seconds() * 1_000_000)
    n = delta_us // period_us + 1
    return epoch + timedelta(microseconds=n * period_us)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


@dataclass
class CanonicalMidFilter:
    """Stream-or-batch canonical p̃_t filter.

    Call :meth:`update` with each ``BookSnap`` (and any ``TradeTick`` that
    have arrived since the last call). It returns zero or more
    :class:`CanonicalMid` — one per grid tick that has closed.

    The filter is stateful but not thread-safe; pin one instance per token.
    """

    token_id: str
    eps: float = DEFAULT_EPS
    grid_hz: float = DEFAULT_GRID_HZ
    trade_window_sec: float = DEFAULT_TRADE_WINDOW_SEC
    outlier_k_iqr: float = DEFAULT_OUTLIER_K_IQR
    outlier_window: int = DEFAULT_OUTLIER_WINDOW
    outlier_min_history: int = DEFAULT_OUTLIER_MIN_HISTORY
    tick_size: float = DEFAULT_TICK_SIZE

    # Internal state (initialized in __post_init__).
    _grid_period: timedelta = field(init=False, repr=False)
    _trades: deque[TradeTick] = field(init=False, repr=False)
    _logit_history: deque[float] = field(init=False, repr=False)
    _last_p_tilde: float | None = field(default=None, init=False, repr=False)
    _last_source: MidSource = field(default="book_mid", init=False, repr=False)
    _last_trade_count: int = field(default=0, init=False, repr=False)
    _next_grid_ts: datetime | None = field(default=None, init=False, repr=False)
    _had_data_in_current_bin: bool = field(default=False, init=False, repr=False)
    _current_bin_rejected: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.eps < 0.5:
            raise ValueError("eps must be in (0, 0.5)")
        if self.grid_hz <= 0:
            raise ValueError("grid_hz must be positive")
        if self.trade_window_sec <= 0:
            raise ValueError("trade_window_sec must be positive")
        if self.outlier_k_iqr <= 0:
            raise ValueError("outlier_k_iqr must be positive")
        self._grid_period = timedelta(seconds=1.0 / self.grid_hz)
        self._trades = deque()
        self._logit_history = deque(maxlen=self.outlier_window)

    # -- public --------------------------------------------------------------

    def update(
        self,
        snap: BookSnap | None,
        trades: list[TradeTick] | tuple[TradeTick, ...] = (),
        *,
        now_ts: datetime | None = None,
    ) -> list[CanonicalMid]:
        """Advance the filter with a new book snap (and any new trades).

        ``snap`` may be None when the caller wants to tick the clock forward
        for forward-fill only; pass ``now_ts`` in that case. When ``snap`` is
        provided, ``now_ts`` is ignored and ``snap.ts`` is used.
        """
        if snap is None and now_ts is None:
            raise ValueError("update() requires either snap or now_ts")

        current_ts = snap.ts if snap is not None else now_ts
        assert current_ts is not None

        outputs: list[CanonicalMid] = []

        if self._next_grid_ts is None:
            self._next_grid_ts = _next_grid_boundary(current_ts, self._grid_period)

        # Emit any grid ticks strictly before current_ts (bins that closed
        # without this new observation).
        while self._next_grid_ts < current_ts:
            forward_fill = not self._had_data_in_current_bin
            out = self._emit(self._next_grid_ts, forward_fill=forward_fill)
            if out is not None:
                outputs.append(out)
            self._next_grid_ts += self._grid_period
            self._had_data_in_current_bin = False

        # Ingest this snap + trades. A rejected tick still counts as "data in
        # the bin" — the grid output will carry it forward but flag
        # rejected_outlier=True so downstream can widen Kalman noise.
        if snap is not None:
            ingest_ok = self._ingest(snap, list(trades))
            # Book-hygiene failures (empty/crossed book) return None and do
            # NOT count as bin data. Rejected outliers (False) and accepted
            # (True) both count.
            if ingest_ok is not None:
                self._had_data_in_current_bin = True

        # If current_ts lands exactly on a grid tick, emit that tick too.
        # For snap-provided: fresh iff any data in the bin (including this
        # snap). For snap=None tick calls: always forward-fill since no
        # data arrived.
        if current_ts == self._next_grid_ts:
            forward_fill = not self._had_data_in_current_bin
            out = self._emit(self._next_grid_ts, forward_fill=forward_fill)
            if out is not None:
                outputs.append(out)
            self._next_grid_ts += self._grid_period
            self._had_data_in_current_bin = False
            self._current_bin_rejected = False

        return outputs

    # -- internals -----------------------------------------------------------

    def _ingest(self, snap: BookSnap, trades: list[TradeTick]) -> bool | None:
        """Update internal p̃ state from a fresh snap.

        Returns:
          * True  — accepted (good tick).
          * False — rejected as an outlier; last value carried forward, flagged.
          * None  — book hygiene failure (empty/crossed); caller should NOT
                   count this as bin data.
        """
        # Basic hygiene on the book.
        if not snap.bids or not snap.asks:
            logger.debug("cmid: empty book at %s; skipping ingest", snap.ts)
            return None
        if snap.bids[0].price >= snap.asks[0].price:
            logger.info(
                "cmid: crossed/locked book (bid=%.4f, ask=%.4f) at %s; skipping",
                snap.bids[0].price,
                snap.asks[0].price,
                snap.ts,
            )
            return None

        # Slide in the new trades, then purge the window.
        for t in trades:
            if t.size > 0 and 0.0 < t.price < 1.0:
                self._trades.append(t)
        cutoff = snap.ts - timedelta(seconds=self.trade_window_sec)
        while self._trades and self._trades[0].ts < cutoff:
            self._trades.popleft()

        book_mid = snap.mid
        assert book_mid is not None  # guarded above
        vwap = trade_vwap(list(self._trades))

        if vwap is not None:
            p_raw = vwap
            source: MidSource = "trade_vwap"
        else:
            p_raw = book_mid
            source = "book_mid"

        p_clipped = clip_p(p_raw, self.eps)
        y = _logit(p_clipped)

        # Outlier test via K * IQR of recent |Δlogit|.
        rejected = False
        if (
            self._last_p_tilde is not None
            and len(self._logit_history) >= self.outlier_min_history
        ):
            window_iqr = iqr(list(self._logit_history))
            if window_iqr > 0:
                delta = abs(y - _logit(self._last_p_tilde))
                if delta > self.outlier_k_iqr * window_iqr:
                    rejected = True
                    logger.info(
                        "cmid: reject outlier token=%s Δlogit=%.3f > %.1f*IQR=%.3f",
                        self.token_id,
                        delta,
                        self.outlier_k_iqr,
                        window_iqr,
                    )

        if rejected:
            # Don't update history with the rejected value; carry forward the
            # last accepted p̃ and flag the current bin.
            self._current_bin_rejected = True
            self._last_trade_count = len(self._trades)
            return False

        # Accept.
        if self._last_p_tilde is not None:
            self._logit_history.append(y - _logit(self._last_p_tilde))
        self._last_p_tilde = p_clipped
        self._last_source = source
        self._last_trade_count = len(self._trades)
        self._current_bin_rejected = False
        return True

    def _emit(self, ts: datetime, *, forward_fill: bool) -> CanonicalMid | None:
        if self._last_p_tilde is None:
            return None
        if forward_fill:
            rejected_flag = False
            source: MidSource = "forward_fill"
        else:
            rejected_flag = self._current_bin_rejected
            source = "outlier_fallback" if rejected_flag else self._last_source
        return CanonicalMid(
            token_id=self.token_id,
            ts=ts,
            p_tilde=self._last_p_tilde,
            y=_logit(self._last_p_tilde),
            forward_filled=forward_fill,
            rejected_outlier=rejected_flag,
            trades_in_window=self._last_trade_count,
            source=source,
        )


__all__ = [
    "CanonicalMid",
    "CanonicalMidFilter",
    "MidSource",
    "clip_p",
    "iqr",
    "trade_vwap",
]
