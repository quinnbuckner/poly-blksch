"""100-500 ms refresh loop (paper §4.5).

Wires Track A (market data + surface) → Track B (quote/guard/limit) → Track C
(order router). The loop is intentionally composable: callers pass in
async callables for each side so we can unit-test it against fixtures and
swap live/paper engines without touching this file.

Cycle (paper §4.5 steps 1-6):
  1. Update x̂, σ̂_b, q, toxicity flags
  2. Compute r_x, δ_x → x_{bid,ask} → p_{bid,ask} with floors/caps
  3. Widen / pull on toxicity or news
  4. Emit Quote to Track C order router
  5. Rebalance cross-event exposure           (Stage 2 — stub)
  6. Rebalance calendar exposure              (Stage 3 — stub)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Protocol

from blksch.schemas import (
    BookSnap,
    HedgeInstruction,
    LogitState,
    Position,
    Quote,
    SurfacePoint,
    TradeTick,
)

from .guards import GuardDecision, GuardState
from .limits import LimitsConfig, LimitsState, inventory_cap_contracts
from .quote import QuoteParams, compute_quote

__all__ = [
    "MarketSnapshot",
    "DataFeed",
    "LoopConfig",
    "RefreshLoop",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything the quote builder needs on one tick.

    `book` and `position` may be None briefly during warm-up.
    """

    token_id: str
    logit_state: LogitState
    surface: SurfacePoint
    position: Position | None
    book: BookSnap | None
    trades: tuple[TradeTick, ...] = ()
    time_to_horizon_sec: float = 3600.0


class DataFeed(Protocol):
    """Callable that yields the latest MarketSnapshot for a token."""

    async def __call__(self, token_id: str) -> MarketSnapshot | None: ...


QuoteSink = Callable[[Quote], Awaitable[None]]
PullSink = Callable[[str, str], Awaitable[None]]  # (token_id, reason)
HedgeSink = Callable[[HedgeInstruction], Awaitable[None]]


# ---------------------------------------------------------------------------
# Loop config + main class
# ---------------------------------------------------------------------------


@dataclass
class LoopConfig:
    refresh_ms: int = 250
    quote: QuoteParams = field(default_factory=QuoteParams)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    max_cycles: int | None = None  # None = run forever; tests set a small int


@dataclass
class _TokenState:
    """Per-token state carried between cycles."""

    guards: GuardState = field(default_factory=GuardState)
    limits: LimitsState = field(default_factory=LimitsState)
    last_quote: Quote | None = None


class RefreshLoop:
    """The §4.5 cycle.

    Pass per-token config via `add_token(token_id)`. Inject IO via DataFeed and
    the three async sinks. Call `run()` to spin (optionally bounded for tests)
    or `run_once(token_id)` to do a single pass (used in integration tests).
    """

    def __init__(
        self,
        *,
        config: LoopConfig,
        data_feed: DataFeed,
        quote_sink: QuoteSink,
        pull_sink: PullSink | None = None,
        hedge_sink: HedgeSink | None = None,
        pnl_provider: Callable[[str], float] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.data_feed = data_feed
        self.quote_sink = quote_sink
        self.pull_sink = pull_sink
        self.hedge_sink = hedge_sink
        self.pnl_provider = pnl_provider
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))
        self._tokens: dict[str, _TokenState] = {}
        self._stopped = asyncio.Event()

    # -- token management ---------------------------------------------------

    def add_token(self, token_id: str) -> None:
        self._tokens.setdefault(token_id, _TokenState(
            limits=LimitsState(cfg=self.config.limits),
        ))

    def remove_token(self, token_id: str) -> None:
        self._tokens.pop(token_id, None)

    def state(self, token_id: str) -> _TokenState:
        if token_id not in self._tokens:
            raise KeyError(token_id)
        return self._tokens[token_id]

    def stop(self) -> None:
        self._stopped.set()

    # -- main cycle ---------------------------------------------------------

    async def run(self) -> None:
        period = max(self.config.refresh_ms, 1) / 1000.0
        cycles = 0
        while not self._stopped.is_set():
            start = time.perf_counter()
            await self._one_cycle()
            cycles += 1
            if self.config.max_cycles is not None and cycles >= self.config.max_cycles:
                return
            elapsed = time.perf_counter() - start
            await asyncio.sleep(max(0.0, period - elapsed))

    async def _one_cycle(self) -> None:
        for token_id in list(self._tokens.keys()):
            try:
                await self.run_once(token_id)
            except Exception:  # one bad token shouldn't take down the loop
                logger.exception("refresh cycle failed for %s", token_id)

    async def run_once(self, token_id: str) -> Quote | None:
        """One full pass through steps 1-6 for a single token.

        Returns the Quote that was emitted, or None if the cycle pulled/paused.
        """
        state = self.state(token_id)
        now = self._clock()

        # --- Step 1: pull fresh market + inventory state --------------------
        snap = await self.data_feed(token_id)
        if snap is None:
            # No data at all — treat as feed gap for purposes of kill-switch.
            report = state.limits.evaluate(
                now=now, current_sigma=None,
                cumulative_pnl_usd=self._pnl(token_id),
            )
            if report.tripped:
                await self._emit_pull(token_id, ",".join(report.reasons))
            return None

        # Update running state
        state.limits.note_tick(snap.logit_state.ts)
        state.limits.note_sigma(snap.surface.sigma_b)
        state.guards.ingest_trades(snap.trades)

        # --- Kill-switch evaluation (step 1 continued) ----------------------
        current_qty = snap.position.qty if snap.position else 0.0
        current_p = snap.book.mid if snap.book else None
        report = state.limits.evaluate(
            now=now,
            current_sigma=snap.surface.sigma_b,
            cumulative_pnl_usd=self._pnl(token_id),
            current_qty=current_qty,
            current_p=current_p,
        )
        if report.tripped:
            logger.warning("kill-switch tripped %s: %s", token_id, report.reasons)
            await self._emit_pull(token_id, ",".join(report.reasons))
            return None

        # --- Step 3: guards (decide before sizing so we can pull cheaply) ----
        decision: GuardDecision = state.guards.decide(now)
        if decision.pull_quotes:
            logger.info("pulling quotes for %s: %s", token_id, decision.reason)
            await self._emit_pull(token_id, decision.reason)
            return None

        # --- Step 2: compute quote -----------------------------------------
        # Enforce inventory cap before we post (matches paper §4.6 point 1).
        cap = inventory_cap_contracts(snap.logit_state.x_hat, self.config.limits)
        if abs(current_qty) > cap:
            logger.info("inventory cap hit for %s: |q|=%.1f > cap=%.1f", token_id, current_qty, cap)
            await self._emit_pull(token_id, "inventory_cap")
            return None

        quote = compute_quote(
            token_id=token_id,
            x_t=snap.logit_state.x_hat,
            sigma_b=snap.surface.sigma_b,
            time_to_horizon_sec=snap.time_to_horizon_sec,
            inventory_q=current_qty,
            params=self.config.quote,
            ts=now,
            spread_widen_factor=decision.spread_widen_factor,
        )

        # --- Step 4: emit --------------------------------------------------
        state.last_quote = quote
        await self.quote_sink(quote)

        # --- Steps 5-6 are stubbed until Stage 2/3 --------------------------
        # (See mm/hedge/*.py — unused at Stage 1.)

        return quote

    # -- internals ----------------------------------------------------------

    async def _emit_pull(self, token_id: str, reason: str) -> None:
        if self.pull_sink is not None:
            try:
                await self.pull_sink(token_id, reason)
            except Exception:
                logger.exception("pull_sink failed for %s", token_id)

    def _pnl(self, token_id: str) -> float:
        return 0.0 if self.pnl_provider is None else self.pnl_provider(token_id)
