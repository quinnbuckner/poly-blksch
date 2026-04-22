"""Hard limits and kill-switches (paper §4.6).

Four auto-pause conditions, any one of which halts quoting:
  (i)   feed gap: no fresh book/trade tick for `feed_gap_sec`
  (ii)  volatility spike: σ_b jumped more than `volatility_spike_z` sd in-window
  (iii) repeated pick-offs: ≥ `repeated_pickoff_count` adverse fills in `window`
  (iv)  drawdown: cumulative PnL < -max_drawdown_usd

Additionally:
  * Inventory cap that tightens with S'(x) (also enforced in quote.py)
  * Max gamma exposure in the swing zone (|p-0.5| < 0.15)

LimitsState is a small state container. `evaluate()` returns the triggered
reasons (may be multiple); the refresh loop treats any non-empty list as a
full pause.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from .greeks import gamma_x, s_prime

__all__ = [
    "LimitsConfig",
    "LimitsState",
    "KillSwitchReport",
    "inventory_cap_contracts",
    "gamma_exposure",
]


@dataclass(frozen=True)
class LimitsConfig:
    feed_gap_sec: float = 3.0
    volatility_spike_z: float = 5.0
    volatility_window: int = 60
    repeated_pickoff_window_sec: float = 60.0
    repeated_pickoff_count: int = 3
    max_drawdown_usd: float = 100.0
    inventory_cap_base: float = 50.0
    sprime_floor: float = 1.0e-4
    max_gamma_exposure: float = 50.0
    swing_zone_half_width: float = 0.15  # |p-0.5| <= this is the swing zone


@dataclass(frozen=True)
class KillSwitchReport:
    """Aggregated result of an evaluation pass."""

    reasons: tuple[str, ...]
    detail: dict[str, float | str]

    @property
    def tripped(self) -> bool:
        return len(self.reasons) > 0


def inventory_cap_contracts(x_t: float, cfg: LimitsConfig) -> float:
    """Max |q| such that downside in logit space stays bounded (paper §4.6)."""
    denom = max(s_prime(x_t), cfg.sprime_floor)
    return cfg.inventory_cap_base / denom


def gamma_exposure(qty: float, p: float) -> float:
    """Signed gamma exposure in p-units: q · Γ_x(p). Used for swing-zone cap."""
    return qty * gamma_x(p)


@dataclass
class LimitsState:
    """Rolling state for kill-switch evaluation."""

    cfg: LimitsConfig = field(default_factory=LimitsConfig)
    _last_tick_ts: datetime | None = None
    _sigma_history: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _pickoff_times: deque[datetime] = field(default_factory=deque)
    _paused: bool = False
    _pause_reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._sigma_history = deque(maxlen=self.cfg.volatility_window)

    def note_tick(self, ts: datetime) -> None:
        """Call on every fresh market data event."""
        if self._last_tick_ts is None or ts > self._last_tick_ts:
            self._last_tick_ts = ts

    def note_sigma(self, sigma_b: float) -> None:
        self._sigma_history.append(sigma_b)

    def note_pickoff(self, ts: datetime) -> None:
        self._pickoff_times.append(ts)
        self._prune_pickoffs(ts)

    def _prune_pickoffs(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.cfg.repeated_pickoff_window_sec)
        while self._pickoff_times and self._pickoff_times[0] < cutoff:
            self._pickoff_times.popleft()

    def feed_gap_seconds(self, now: datetime) -> float:
        if self._last_tick_ts is None:
            return 0.0
        return max(0.0, (now - self._last_tick_ts).total_seconds())

    def volatility_spike_z(self, current_sigma: float) -> float | None:
        """Current σ_b's z-score against the rolling history; None if unreliable."""
        if len(self._sigma_history) < 10:
            return None
        mean = statistics.fmean(self._sigma_history)
        if len(self._sigma_history) < 2:
            return None
        stdev = statistics.pstdev(self._sigma_history)
        if stdev < 1.0e-12:
            return None
        return (current_sigma - mean) / stdev

    def evaluate(
        self,
        *,
        now: datetime,
        current_sigma: float | None,
        cumulative_pnl_usd: float,
        current_qty: float | None = None,
        current_p: float | None = None,
    ) -> KillSwitchReport:
        """Check all kill-switches; return the list of tripped reasons."""
        reasons: list[str] = []
        detail: dict[str, float | str] = {}

        self._prune_pickoffs(now)

        gap = self.feed_gap_seconds(now)
        if self._last_tick_ts is not None and gap > self.cfg.feed_gap_sec:
            reasons.append("feed_gap")
            detail["feed_gap_sec"] = gap

        if current_sigma is not None:
            z = self.volatility_spike_z(current_sigma)
            if z is not None and z > self.cfg.volatility_spike_z:
                reasons.append("volatility_spike")
                detail["volatility_z"] = z

        if len(self._pickoff_times) >= self.cfg.repeated_pickoff_count:
            reasons.append("repeated_pickoffs")
            detail["pickoff_count"] = float(len(self._pickoff_times))

        if cumulative_pnl_usd <= -abs(self.cfg.max_drawdown_usd):
            reasons.append("max_drawdown")
            detail["cumulative_pnl_usd"] = cumulative_pnl_usd

        if current_qty is not None and current_p is not None:
            exposure = abs(gamma_exposure(current_qty, current_p))
            in_swing = abs(current_p - 0.5) <= self.cfg.swing_zone_half_width
            if in_swing and exposure > self.cfg.max_gamma_exposure:
                reasons.append("max_gamma_swing")
                detail["gamma_exposure"] = exposure

        report = KillSwitchReport(tuple(reasons), detail)
        if report.tripped:
            self._paused = True
            self._pause_reasons = report.reasons
        return report

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def pause_reasons(self) -> tuple[str, ...]:
        return self._pause_reasons

    def resume(self) -> None:
        """Operator-triggered resume. Clears paused flag and pick-off window."""
        self._paused = False
        self._pause_reasons = ()
        self._pickoff_times.clear()
