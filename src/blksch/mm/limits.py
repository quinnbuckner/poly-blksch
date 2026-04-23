"""Hard limits and kill-switches (paper §4.6).

Four auto-pause conditions, any one of which halts quoting:
  (i)   feed gap: no fresh book/trade tick for `feed_gap_sec`
  (ii)  volatility spike: σ_b jumped more than `volatility_spike_z` sd in-window
  (iii) repeated pick-offs: ≥ `repeated_pickoff_count` adverse fills in `window`
  (iv)  drawdown: cumulative PnL < -max_drawdown_usd

Additionally:
  * Inventory cap that tightens with S'(x) (also enforced in quote.py)
  * Max gamma exposure in the swing zone (|p-0.5| < 0.15)

Paper §4.6 specifies the halt is sticky "until operator resume". LimitsState
enforces this directly:

  * ``paused`` stays True across cycles once any kill-switch has fired —
    ``evaluate()`` accumulates new reasons into a persistent set but does
    not auto-clear an existing pause when the current-cycle reasons fall
    back to empty. This mirrors ``exec/paper_engine``'s sticky ``halted``
    flag; before this contract, a feed-gap recovery flooded the 72 h soak
    with per-cycle ``place_order rejected (halted)`` warnings because the
    refresh loop only looked at the current-cycle ``report.tripped`` flag
    and happily re-emitted quotes that the PaperEngine then refused.
  * Callers distinguish a "newly tripped" cycle from a "sticky continuation"
    cycle via ``KillSwitchReport.new_reasons`` — the subset of this cycle's
    reasons that weren't already in the persistent pause set. The refresh
    loop logs WARN only on new trips and DEBUG on continuations.
  * ``resume(reason=...)`` is the single operator-triggered clear. The
    optional ``reason`` string lets the operator log WHY they cleared, for
    the post-soak audit trail.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
    """Aggregated result of an evaluation pass.

    ``reasons`` lists the conditions that fired *this* cycle (may be empty
    even while the switch remains paused — that's what ``LimitsState.paused``
    distinguishes from ``report.tripped``). To tell "newly tripped this
    cycle" from "sticky continuation of a prior cycle's trip", callers
    diff ``report.reasons`` against a pre-evaluate snapshot of
    ``state.limits.pause_reasons`` — the refresh loop does exactly this
    to pick WARN vs DEBUG log severity. We intentionally keep that diff
    in the caller rather than on the report so this dataclass's shape
    stays stable for snapshot-regression coverage.
    """

    reasons: tuple[str, ...]
    detail: dict[str, float | str]

    @property
    def tripped(self) -> bool:
        """True iff ``reasons`` is non-empty for *this* cycle.

        NOTE: This is NOT the same as "should pull quotes". Sticky pause
        from a prior cycle is queried via ``LimitsState.paused``. Callers
        that want pull semantics should check ``state.limits.paused``.
        """
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
    """Rolling state for kill-switch evaluation.

    Persistent across cycles:
      * ``_paused``         — sticky bool; True from the first trip until
                              ``resume()`` is called.
      * ``_pause_reasons``  — accumulated ``set[str]`` of every reason
                              seen while paused; never auto-pruned.

    ``_pause_reasons`` is a *set*, not a tuple, so that a second, different
    trip while already paused (e.g. feed_gap → then a drawdown during the
    halted window) adds to the audit trail instead of overwriting it.
    """

    cfg: LimitsConfig = field(default_factory=LimitsConfig)
    _last_tick_ts: datetime | None = None
    _sigma_history: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _pickoff_times: deque[datetime] = field(default_factory=deque)
    _paused: bool = False
    _pause_reasons: set[str] = field(default_factory=set)

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
        """Check all kill-switches; return the current-cycle report.

        Side effects on ``self``:
          * Any condition that fires this cycle is accumulated into
            ``_pause_reasons`` (the persistent set).
          * ``_paused`` is set to True on any fresh trip and is NEVER
            auto-cleared here — only ``resume()`` clears it. This matches
            the paper §4.6 "halt until operator resume" contract.

        The returned ``KillSwitchReport`` carries:
          * ``reasons``      — every condition that fired THIS cycle.
          * ``new_reasons``  — subset of ``reasons`` not already in the
                               persistent pause set (i.e. reasons the
                               operator has not yet seen). Used by the
                               refresh loop to log WARN once vs DEBUG
                               on sticky continuation.
        """
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

        # Accumulate into the persistent set — evaluate() never replaces or
        # clears _pause_reasons; only resume() does. Callers that need
        # "which reasons are *new* this cycle" diff report.reasons against
        # a pre-evaluate snapshot of self.pause_reasons (e.g. the refresh
        # loop picks WARN vs DEBUG log tier that way).
        if reasons:
            self._paused = True
            self._pause_reasons.update(reasons)

        return KillSwitchReport(reasons=tuple(reasons), detail=detail)

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def pause_reasons(self) -> tuple[str, ...]:
        """Sorted tuple snapshot of the accumulated pause reasons.

        Returned as a tuple (not the underlying set) so callers can't
        mutate the internal state, and sorted so log/dashboard ordering
        is deterministic across runs.
        """
        return tuple(sorted(self._pause_reasons))

    def resume(self, reason: str | None = None) -> None:
        """Operator-triggered resume. Clears paused flag, accumulated
        reasons, and the pick-off window.

        Args:
            reason: Optional free-text operator note recorded to the
                audit log. Use e.g. ``"feed restored, venue confirmed"``
                or ``"false-positive on stale timestamp"``. Nothing
                functional depends on the content — it exists solely so
                the soak post-mortem can explain *why* a halt was cleared.
        """
        if self._paused:
            logger.info(
                "kill-switch resumed: prior_reasons=%s note=%r",
                sorted(self._pause_reasons),
                reason,
            )
        self._paused = False
        self._pause_reasons = set()
        self._pickoff_times.clear()
