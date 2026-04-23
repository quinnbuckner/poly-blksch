"""blksch entrypoint — wires Track A ingest/filter/calibration → Track B
quoting → Track C execution into one asyncio graph.

CLI
---

    python -m blksch.app --mode=paper                      # paper-soak
    python -m blksch.app --mode=paper --tokens=0xA,0xB     # skip screener
    python -m blksch.app --mode=live --live-ack            # real orders

Architecture
------------

::

    PolyClient.stream_market(tokens)
        │ (BookSnap, TradeTick)
        ▼
    ingest_loop  ──►  per-token filter chain (canonical_mid → microstruct
        │                                   → kalman) writes LogitState
        │             into TokenState.latest_logit each tick
        ▼
    PaperEngine.on_book/on_trade  →  Fill callbacks  →  Ledger
        │
        ▼
    (periodic)  calibration_loop  ──►  em_calibrate(rolling 400 s window)
                                       → SurfacePoint into TokenState
                                       runs in an executor so its
                                       multi-second runtime never stalls
                                       the 250 ms refresh loop

    RefreshLoop(data_feed=read_token_state)
        │ Quote                              ──►  OrderRouter.sync_quote
        │ (inventory_q, surface, book, ...)       (paper: PaperEngine
        ▼                                          live:  CLOBClient)

The per-token shared state is a plain ``TokenState`` dataclass. Writes are
atomic reference swaps (Python GIL) and all tasks share the one asyncio
event loop, so no lock is needed: refresh_loop reads the newest available
values each cycle and tolerates staleness — blocking is not.

Shutdown on SIGINT / SIGTERM: stop_event is set, refresh_loop.stop() drains
its current cycle, ingest + calibration tasks cancel, any resting orders
are cancelled, the aiohttp session and ledger close.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from blksch.core.em.rn_drift import RNDriftConfig, em_calibrate
from blksch.core.filter.canonical_mid import CanonicalMidFilter
from blksch.core.filter.kalman import KalmanFilter
from blksch.core.filter.microstruct import MicrostructModel
from blksch.core.ingest.polyclient import PolyClient
from blksch.core.ingest.screener import Screener, ScreenerFilters
from blksch.exec.clob_client import CLOBConfig, make_clob_client
from blksch.exec.dashboard import DashboardContext, FlaskDashboard, RichDashboard
from blksch.exec.ledger import Ledger
from blksch.exec.order_router import OrderRouter, RouterConfig
from blksch.exec.paper_engine import PaperEngine, PaperEngineConfig
from blksch.mm.limits import LimitsConfig
from blksch.mm.quote import QuoteParams
from blksch.mm.refresh_loop import (
    LoopConfig,
    MarketSnapshot,
    RefreshLoop,
)
from blksch.schemas import (
    BookSnap,
    LogitState,
    Position,
    Quote,
    SurfacePoint,
    TradeTick,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cold-start defaults
# ---------------------------------------------------------------------------

# Until EM calibration publishes a real SurfacePoint, seed each token with
# a conservative surface so the refresh loop can emit a (wide) quote instead
# of waiting ~60 s for the first EM pass. σ_b=0.02 per √sec is a realistic
# prediction-market scale — at T=3600 s it yields ~10¢ quoted spread, plenty
# to post inside the book while EM warms up.
_BOOTSTRAP_SIGMA_B = 0.02
_BOOTSTRAP_LAMBDA = 0.01
_BOOTSTRAP_S2J = 0.0

# Microstruct cold start: flat σ_η² = 0.01. Track A's online fit can later
# replace this with a calibrated instance; until then Kalman gets a constant
# measurement-noise baseline that's at least as wide as realistic frictions.
_COLD_MICROSTRUCT = MicrostructModel(a0=0.01, a1=0.0, a2=0.0, a3=0.0, a4=0.0)

# Paper §6.4 calibration-schedule defaults.
_CALIB_WINDOW_SEC = 400.0
_CALIB_STRIDE_SEC = 60.0
_CALIB_MIN_STATES = 60  # ≈ 60 s at 1 Hz before first calibration fires

# Trade buffer per-token for guards / toxicity consumption.
_TRADE_BUFFER_SIZE = 500


# ---------------------------------------------------------------------------
# Per-token state
# ---------------------------------------------------------------------------


@dataclass
class TokenState:
    """Latest-writes-wins view of one token. Single-writer-per-field under
    the single asyncio event loop, readers tolerate staleness."""

    token_id: str
    time_to_horizon_sec: float
    latest_book: BookSnap | None = None
    latest_logit: LogitState | None = None
    latest_surface: SurfacePoint | None = None
    recent_trades: deque[TradeTick] = field(
        default_factory=lambda: deque(maxlen=_TRADE_BUFFER_SIZE),
    )
    calib_window: deque[LogitState] = field(
        default_factory=lambda: deque(maxlen=int(_CALIB_WINDOW_SEC) + 50),
    )
    last_calibration_ts: datetime | None = None

    def seed_bootstrap_surface(self, ts: datetime) -> None:
        if self.latest_surface is not None:
            return
        self.latest_surface = SurfacePoint(
            token_id=self.token_id,
            tau=self.time_to_horizon_sec,
            m=0.0,
            sigma_b=_BOOTSTRAP_SIGMA_B,
            **{"lambda": _BOOTSTRAP_LAMBDA},
            s2_j=_BOOTSTRAP_S2J,
            uncertainty=None,
            ts=ts,
        )


@dataclass
class _FilterChain:
    """Per-token stateful filter chain. Not thread-safe; one per token."""

    canonical_mid: CanonicalMidFilter
    kalman: KalmanFilter


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Loaded view of config/bot.yaml + config/markets.yaml. Additive-only
    over the existing YAML schema — no breaking renames."""

    mode: str
    network: str
    quote_params: QuoteParams
    loop_config: LoopConfig
    limits_config: LimitsConfig
    rn_drift_config: RNDriftConfig
    calib_stride_sec: float
    calib_window_sec: float
    em_max_iters: int
    grid_hz: float
    time_to_horizon_sec: float
    screener_filters: ScreenerFilters
    pair_hints: list[tuple[str, str]]
    raw: dict[str, Any]


def load_app_config(bot_yaml: Path, markets_yaml: Path) -> AppConfig:
    with open(bot_yaml) as f:
        bot = yaml.safe_load(f) or {}
    with open(markets_yaml) as f:
        markets = yaml.safe_load(f) or {}

    _require_keys(bot, ("quoting", "loop", "calibration", "limits", "boundary"))
    q = bot["quoting"]
    bnd = bot["boundary"]
    inv = bot.get("inventory", {})
    loop = bot["loop"]
    cal = bot["calibration"]
    lim = bot["limits"]
    app_extra = bot.get("app", {}) if isinstance(bot.get("app"), dict) else {}

    quote_params = QuoteParams(
        gamma=float(q["gamma"]),
        k=float(q["k"]),
        eps=float(bnd["eps"]),
        # delta_p_floor_ticks × tick_size (Polymarket tick = 0.01).
        delta_p_floor=float(bnd["delta_p_floor_ticks"]) * 0.01,
        q_max_base=float(inv.get("q_max_notional_usd", 50.0)),
        q_max_shrink=float(inv.get("q_max_shrink_factor", 1.0)),
    )

    limits_config = LimitsConfig(
        feed_gap_sec=float(lim.get("feed_gap_sec", 3.0)),
        volatility_spike_z=float(lim.get("volatility_spike_z", 5.0)),
        repeated_pickoff_window_sec=float(lim.get("repeated_pickoff_window_sec", 60.0)),
        repeated_pickoff_count=int(lim.get("repeated_pickoff_count", 3)),
        max_drawdown_usd=float(lim.get("max_drawdown_usd", 100.0)),
    )

    rn_drift = RNDriftConfig(
        mu_cap_per_sec=float(cal.get("mu_cap_per_sec", 0.25)),
        sprime_clip=float(cal.get("sprime_clip", 1.0e-4)),
        mc_samples=int(cal.get("mc_draws_per_step", 600)),
    )

    loop_config = LoopConfig(
        refresh_ms=int(loop.get("refresh_ms", 250)),
        quote=quote_params,
        limits=limits_config,
        hedge_enabled=False,
        calendar_hedge_enabled=False,
        synth_strip_enabled=False,
    )

    # Screener filters — tolerate both the old-school YAML (key:
    # rescreen_every_sec) and the dataclass name (ttl_sec).
    sc = markets.get("screener", {}) if isinstance(markets, dict) else {}
    sc_kwargs = {
        k: v for k, v in sc.items()
        if k in {
            "min_volume_24h_usd", "min_depth_usd_5pct", "top_n",
            "min_spread_bps", "max_spread_bps",
            "min_hours_to_resolution", "max_days_to_resolution",
        }
    }
    if "rescreen_every_sec" in sc:
        sc_kwargs["ttl_sec"] = float(sc["rescreen_every_sec"])
    screener_filters = ScreenerFilters(**sc_kwargs)

    pair_hints_raw = markets.get("correlation_pair_hints", []) if isinstance(markets, dict) else []
    pair_hints: list[tuple[str, str]] = []
    for entry in pair_hints_raw or []:
        pair = entry.get("pair") if isinstance(entry, dict) else None
        if pair and len(pair) == 2:
            pair_hints.append((str(pair[0]), str(pair[1])))

    return AppConfig(
        mode=str(bot.get("mode", "paper")),
        network=str(bot.get("network", "mainnet")),
        quote_params=quote_params,
        loop_config=loop_config,
        limits_config=limits_config,
        rn_drift_config=rn_drift,
        calib_stride_sec=float(app_extra.get("calibration_stride_sec", _CALIB_STRIDE_SEC)),
        calib_window_sec=float(cal.get("em_window_sec", _CALIB_WINDOW_SEC)),
        em_max_iters=int(cal.get("em_global_steps", 6)),
        grid_hz=float(cal.get("kf_grid_hz", 1.0)),
        time_to_horizon_sec=float(q.get("T_fallback_sec", 3600.0)),
        screener_filters=screener_filters,
        pair_hints=pair_hints,
        raw=bot,
    )


def _require_keys(cfg: dict[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise SystemExit(f"config missing required top-level keys: {missing}")


# ---------------------------------------------------------------------------
# Ingest + filter task
# ---------------------------------------------------------------------------


async def ingest_loop(
    *,
    client: PolyClient,
    token_ids: Sequence[str],
    filters: dict[str, _FilterChain],
    states: dict[str, TokenState],
    paper_engine: PaperEngine | None,
    dashboard_ctx: DashboardContext,
    stop_event: asyncio.Event,
) -> None:
    """One WS subscription, dispatched per-token. Exits cleanly on
    stop_event (the underlying WS async-iterator unwinds on cancellation)."""
    stream = client.stream_market(list(token_ids))
    try:
        async for event in stream:
            if stop_event.is_set():
                break
            tid = event.token_id
            state = states.get(tid)
            chain = filters.get(tid)
            if state is None or chain is None:
                continue
            try:
                if isinstance(event, BookSnap):
                    await _on_book(event, state, chain, paper_engine, dashboard_ctx)
                elif isinstance(event, TradeTick):
                    await _on_trade(event, state, paper_engine, dashboard_ctx)
            except Exception:
                logger.exception("ingest tick handler crashed for %s", tid)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("ingest loop crashed")
        raise


async def _on_book(
    snap: BookSnap,
    state: TokenState,
    chain: _FilterChain,
    paper_engine: PaperEngine | None,
    dashboard_ctx: DashboardContext,
) -> None:
    state.latest_book = snap
    state.seed_bootstrap_surface(snap.ts)
    dashboard_ctx.on_book(snap)

    # canonical mid → Kalman → LogitState (zero-or-more per grid close).
    recent_trades = list(state.recent_trades)
    cms = chain.canonical_mid.update(snap, recent_trades)
    for cm in cms:
        ls = chain.kalman.step(cm, snap, recent_trades)
        state.latest_logit = ls
        state.calib_window.append(ls)

    if paper_engine is not None:
        fills = await paper_engine.on_book(snap)
        for f in fills:
            dashboard_ctx.on_fill(f)


async def _on_trade(
    tick: TradeTick,
    state: TokenState,
    paper_engine: PaperEngine | None,
    dashboard_ctx: DashboardContext,
) -> None:
    state.recent_trades.append(tick)
    if paper_engine is not None:
        fills = await paper_engine.on_trade(tick)
        for f in fills:
            dashboard_ctx.on_fill(f)


# ---------------------------------------------------------------------------
# Calibration task (background)
# ---------------------------------------------------------------------------


async def calibration_loop(
    *,
    state: TokenState,
    cfg: AppConfig,
    dashboard_ctx: DashboardContext,
    stop_event: asyncio.Event,
    executor: Any | None,
) -> None:
    """Run em_calibrate on a rolling window every `calib_stride_sec`.

    em_calibrate is CPU-bound and can take seconds — delegated to an
    executor so the event loop (and the 250 ms refresh cycle) remain
    responsive. Intermediate results are pushed into `state.latest_surface`.
    """
    stride = max(1.0, cfg.calib_stride_sec)
    try:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=stride)
                return  # stop_event fired during the wait
            except asyncio.TimeoutError:
                pass  # stride elapsed — run a pass

            window = list(state.calib_window)
            if len(window) < _CALIB_MIN_STATES:
                continue

            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    executor,
                    _calibrate_sync,
                    window,
                    cfg.em_max_iters,
                    cfg.rn_drift_config,
                )
            except Exception:
                logger.exception("calibration failed for %s", state.token_id)
                continue

            if result is None:
                continue

            now = datetime.now(UTC)
            sp = SurfacePoint(
                token_id=state.token_id,
                tau=state.time_to_horizon_sec,
                m=window[-1].x_hat,
                sigma_b=float(result.final_params.sigma_b),
                **{"lambda": float(result.jumps.lambda_hat)},
                s2_j=float(result.jumps.s_J_sq_hat),
                uncertainty=None,
                ts=now,
            )
            state.latest_surface = sp
            state.last_calibration_ts = now
            dashboard_ctx.on_surface(sp)
    except asyncio.CancelledError:
        raise


def _calibrate_sync(
    states: list[LogitState],
    max_iters: int,
    rn_drift_config: RNDriftConfig,
) -> Any:
    """Executor entry point — pure Python, no asyncio. Returns None on
    failure so the caller can skip this cycle without crashing the loop."""
    try:
        return em_calibrate(
            states,
            initial_params=None,  # internal bi-power warm start
            max_iters=max_iters,
            drift_config=rn_drift_config,
        )
    except Exception:
        logger.exception("em_calibrate raised in executor")
        return None


# ---------------------------------------------------------------------------
# RefreshLoop wiring (data_feed + quote/pull sinks)
# ---------------------------------------------------------------------------


def _make_data_feed(states: dict[str, TokenState], ledger: Ledger):
    async def data_feed(token_id: str) -> MarketSnapshot | None:
        state = states.get(token_id)
        if state is None:
            return None
        if state.latest_logit is None or state.latest_surface is None:
            return None  # warm-up — refresh loop sees None, skips cycle
        position: Position | None
        try:
            position = ledger.get_position(token_id)
        except Exception:
            position = None
        return MarketSnapshot(
            token_id=token_id,
            logit_state=state.latest_logit,
            surface=state.latest_surface,
            position=position,
            book=state.latest_book,
            trades=tuple(state.recent_trades),
            time_to_horizon_sec=state.time_to_horizon_sec,
        )
    return data_feed


def _make_quote_sink(router: OrderRouter, dashboard_ctx: DashboardContext):
    async def quote_sink(quote: Quote) -> None:
        dashboard_ctx.on_quote(quote)
        try:
            await router.sync_quote(quote)
        except Exception:
            logger.exception("order router sync_quote failed for %s", quote.token_id)
    return quote_sink


def _make_pull_sink(router: OrderRouter):
    async def pull_sink(token_id: str, reason: str) -> None:
        try:
            await router.cancel_all(token_id=token_id)
            logger.info("quotes pulled for %s (reason=%s)", token_id, reason)
        except Exception:
            logger.exception("order router cancel_all failed for %s", token_id)
    return pull_sink


# ---------------------------------------------------------------------------
# Main runtime
# ---------------------------------------------------------------------------


@dataclass
class RunArgs:
    """Parsed CLI as a value object — lets tests construct directly."""

    mode: str = "paper"
    config_path: Path = Path("config/bot.yaml")
    markets_path: Path = Path("config/markets.yaml")
    log_level: str = "INFO"
    tokens: list[str] | None = None
    ledger_path: Path | None = None
    dashboard_port: int | None = None
    rich_dashboard: str = "auto"  # auto | on | off
    live_ack: bool = False
    max_cycles: int | None = None
    max_runtime_sec: float | None = None


def _parse_args(argv: Sequence[str] | None = None) -> RunArgs:
    p = argparse.ArgumentParser("blksch.app", description=__doc__)
    p.add_argument("--mode", choices=("paper", "live"), default="paper")
    p.add_argument("--config", default="config/bot.yaml")
    p.add_argument("--markets", default="config/markets.yaml")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--tokens", default=None, help="comma-separated token_ids (skip screener)")
    p.add_argument("--ledger-path", default=None, help="sqlite path; omit for in-memory")
    p.add_argument("--dashboard-port", type=int, default=None)
    p.add_argument("--rich-dashboard", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--live-ack", action="store_true",
                   help="Required for --mode=live — confirms real-orders intent.")
    p.add_argument("--max-cycles", type=int, default=None,
                   help="stop the refresh loop after N cycles (test harness only)")
    p.add_argument("--max-runtime-sec", type=float, default=None,
                   help="self-terminate after N seconds (test harness only)")
    ns = p.parse_args(argv)
    return RunArgs(
        mode=ns.mode,
        config_path=Path(ns.config),
        markets_path=Path(ns.markets),
        log_level=ns.log_level,
        tokens=[t.strip() for t in ns.tokens.split(",")] if ns.tokens else None,
        ledger_path=Path(ns.ledger_path) if ns.ledger_path else None,
        dashboard_port=ns.dashboard_port,
        rich_dashboard=ns.rich_dashboard,
        live_ack=ns.live_ack,
        max_cycles=ns.max_cycles,
        max_runtime_sec=ns.max_runtime_sec,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("interrupted")
    return 0


async def run(
    args: RunArgs,
    *,
    client: PolyClient | None = None,
    ledger: Ledger | None = None,
    clock: Any | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Async entrypoint. Dependency-injects the PolyClient, Ledger, and
    stop_event for tests — any object with `stream_market`, `list_markets`,
    `start`, `close` is acceptable in place of the real PolyClient."""
    cfg = load_app_config(args.config_path, args.markets_path)
    if args.mode == "live" and not args.live_ack:
        raise SystemExit(
            "Refusing to start in live mode without --live-ack. Paper first; "
            "flip the flag once the soak report is signed off."
        )

    if stop_event is None:
        stop_event = asyncio.Event()
        _install_signal_handlers(stop_event)

    owned_client = False
    if client is None:
        client = PolyClient()
        await client.start()
        owned_client = True

    owned_ledger = False
    if ledger is None:
        ledger = (
            Ledger.open(args.ledger_path) if args.ledger_path is not None
            else Ledger.in_memory()
        )
        owned_ledger = True

    # Token selection: explicit list or screener.
    if args.tokens:
        token_ids = list(args.tokens)
        logger.info("using explicit tokens: %s", token_ids)
    else:
        screener = Screener(
            client=client, filters=cfg.screener_filters, pair_hints=cfg.pair_hints,
        )
        screen = await screener.screen()
        token_ids = list(screen.token_ids)
        logger.info("screener selected %d tokens: %s", len(token_ids), token_ids)

    if not token_ids:
        logger.error("no tokens to quote — check screener filters or pass --tokens")
        await _cleanup(client if owned_client else None, ledger)
        return

    # Per-token state + filter chains.
    states: dict[str, TokenState] = {
        tid: TokenState(token_id=tid, time_to_horizon_sec=cfg.time_to_horizon_sec)
        for tid in token_ids
    }
    filters: dict[str, _FilterChain] = {
        tid: _build_filter_chain(tid, cfg) for tid in token_ids
    }

    dashboard_ctx = DashboardContext(ledger=ledger, mode=args.mode)

    # Execution plumbing.
    paper_engine: PaperEngine | None = None
    router: OrderRouter
    if args.mode == "paper":
        paper_engine = PaperEngine(
            ledger,
            config=PaperEngineConfig(feed_gap_sec=cfg.limits_config.feed_gap_sec),
            on_fill=dashboard_ctx.on_fill,
        )
        router = OrderRouter(paper_backend=paper_engine,
                             config=RouterConfig(mode="paper"))
    else:  # live
        clob_cfg = CLOBConfig.from_env(testnet=(cfg.network != "mainnet"))
        live_backend = make_clob_client(clob_cfg)
        router = OrderRouter(
            live_backend=live_backend,
            config=RouterConfig(mode="live", live_ack=True),
        )

    # RefreshLoop.
    loop_cfg = LoopConfig(
        refresh_ms=cfg.loop_config.refresh_ms,
        quote=cfg.quote_params,
        limits=cfg.limits_config,
        hedge_enabled=False,
        calendar_hedge_enabled=False,
        synth_strip_enabled=False,
        max_cycles=args.max_cycles,
    )
    refresh = RefreshLoop(
        config=loop_cfg,
        data_feed=_make_data_feed(states, ledger),
        quote_sink=_make_quote_sink(router, dashboard_ctx),
        pull_sink=_make_pull_sink(router),
        clock=clock,
    )
    for tid in token_ids:
        refresh.add_token(tid)

    # Dashboards (optional / side tasks).
    flask_thread, flask_url = _start_flask_dashboard(dashboard_ctx, args.dashboard_port)
    if flask_url:
        logger.info("dashboard listening on %s", flask_url)
    rich_task = _maybe_start_rich_dashboard(dashboard_ctx, args.rich_dashboard, stop_event)

    # Background task graph. Executor=None ⇒ asyncio default ThreadPoolExecutor.
    executor: Any | None = None
    tasks: list[asyncio.Task[Any]] = [
        asyncio.create_task(
            ingest_loop(
                client=client, token_ids=token_ids, filters=filters, states=states,
                paper_engine=paper_engine, dashboard_ctx=dashboard_ctx,
                stop_event=stop_event,
            ),
            name="ingest",
        ),
        asyncio.create_task(refresh.run(), name="refresh"),
    ]
    for tid in token_ids:
        tasks.append(asyncio.create_task(
            calibration_loop(
                state=states[tid], cfg=cfg, dashboard_ctx=dashboard_ctx,
                stop_event=stop_event, executor=executor,
            ),
            name=f"calibrate[{tid[:8]}]",
        ))
    if rich_task is not None:
        tasks.append(rich_task)
    if args.max_runtime_sec is not None:
        tasks.append(asyncio.create_task(
            _runtime_watchdog(args.max_runtime_sec, stop_event),
            name="watchdog",
        ))

    logger.info("blksch up — mode=%s tokens=%d refresh_ms=%d",
                args.mode, len(token_ids), loop_cfg.refresh_ms)

    try:
        await _wait_for_stop(tasks, stop_event)
    finally:
        # Graceful shutdown: stop_event → refresh drain → cancel tasks →
        # cancel resting orders → close client → close ledger.
        stop_event.set()
        refresh.stop()
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await router.cancel_all()
        except Exception:
            logger.exception("cancel_all on shutdown failed")
        if owned_client:
            try:
                await client.close()
            except Exception:
                logger.exception("client close failed")
        if owned_ledger:
            try:
                ledger.close()
            except Exception:
                logger.exception("ledger close failed")
        # Flask thread is a daemon — exit with the process.
        del flask_thread


def _build_filter_chain(token_id: str, cfg: AppConfig) -> _FilterChain:
    grid_hz = cfg.grid_hz or 1.0
    return _FilterChain(
        canonical_mid=CanonicalMidFilter(token_id=token_id, grid_hz=grid_hz),
        kalman=KalmanFilter(token_id=token_id, microstruct=_COLD_MICROSTRUCT),
    )


async def _wait_for_stop(
    tasks: list[asyncio.Task[Any]], stop_event: asyncio.Event,
) -> None:
    """Block until stop_event is set OR any task exits (first completion wins).

    If a task raises, we log here and let the `finally` in run() do cleanup.
    """
    stop_wait = asyncio.create_task(stop_event.wait(), name="stop_wait")
    try:
        done, _pending = await asyncio.wait(
            {*tasks, stop_wait}, return_when=asyncio.FIRST_COMPLETED,
        )
        for d in done:
            if d.get_name() == "stop_wait":
                continue
            if d.exception() is not None:
                logger.error("task %s raised: %r", d.get_name(), d.exception())
    finally:
        if not stop_wait.done():
            stop_wait.cancel()
            try:
                await stop_wait
            except (asyncio.CancelledError, Exception):
                pass


async def _runtime_watchdog(seconds: float, stop_event: asyncio.Event) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        logger.info("--max-runtime-sec=%s elapsed; shutting down", seconds)
        stop_event.set()


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """Best-effort signal wiring. Windows / non-main-thread contexts may
    not support `loop.add_signal_handler`; tests that pass max_runtime_sec
    never depend on signals."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------


def _start_flask_dashboard(
    ctx: DashboardContext, port: int | None,
) -> tuple[threading.Thread | None, str | None]:
    if port is None:
        return None, None
    dash = FlaskDashboard(ctx)
    host = os.environ.get("BLKSCH_DASHBOARD_HOST", "127.0.0.1")
    thread = threading.Thread(
        target=dash.run, kwargs={"host": host, "port": port}, daemon=True,
        name="flask-dashboard",
    )
    thread.start()
    return thread, f"http://{host}:{port}/api/state"


def _maybe_start_rich_dashboard(
    ctx: DashboardContext, mode: str, stop_event: asyncio.Event,
) -> asyncio.Task[Any] | None:
    if mode == "off":
        return None
    if mode == "auto" and not sys.stdout.isatty():
        return None
    dash = RichDashboard(ctx)
    return asyncio.create_task(dash.run(stop_event=stop_event), name="rich-dashboard")


# ---------------------------------------------------------------------------
# Early-error cleanup
# ---------------------------------------------------------------------------


async def _cleanup(client: PolyClient | None, ledger: Ledger) -> None:
    if client is not None:
        try:
            await client.close()
        except Exception:
            pass
    try:
        ledger.close()
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
