"""First-ever live Track A calibration dry-run (read-only).

Per-market GO/NO-GO gate before the 72 h paper-soak picks a token.

Flow
----
1. Parse CLI / load ``config/bot.yaml``.
2. Pick token_id: explicit ``--token-id`` or ``--auto`` via
   ``core.ingest.screener``.
3. Open ``core.ingest.polyclient`` against Polymarket's public endpoints.
   REST warm-up snapshot, then CLOB WS ``market`` channel for ``--minutes``.
4. Write every ``BookSnap`` and ``TradeTick`` into
   ``core.ingest.store.ParquetStore`` under ``--out``.
5. At each configured snapshot time, run the full Track A calibration on
   ticks collected so far: canonical_mid → microstruct → kalman →
   em_calibrate. Compute diagnostics (Ljung-Box Q(20), Q-Q Shapiro,
   realized-vs-implied variance), plus the self-forecast MSE/MAE/QLIKE
   of the EwmaVar σ̂_b²(u) forecast against realized H=60s RV on the
   collected stream.
6. Write ``report.json`` + one-line verdict (GREEN/YELLOW/RED) per
   snapshot.

Read-only: no authenticated endpoints, no orders, no ``.env`` required.

The script is decomposed into pure functions that the unit and
integration tests import directly — the async live-ingest shell is the
only part that is not unit-covered.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import math
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from blksch.core.diagnostics import DiagnosticsReport, run_diagnostics
from blksch.core.em.rn_drift import RNDriftConfig, em_calibrate
from blksch.core.filter.canonical_mid import CanonicalMidFilter
from blksch.core.filter.ewma_var import EwmaVar
from blksch.core.filter.kalman import KalmanFilter
from blksch.core.filter.microstruct import (
    MicrostructConfig,
    MicrostructFeatures,
    MicrostructModel,
    extract_features,
)
from blksch.core.ingest.store import ParquetStore
from blksch.schemas import BookSnap, LogitState, TradeTick

log = logging.getLogger("calibration_dryrun")

VERDICT_GREEN = "GREEN"
VERDICT_YELLOW = "YELLOW"
VERDICT_RED = "RED"

DEFAULT_SNAPSHOT_INTERVAL_MIN = 5
DEFAULT_HORIZON_SEC = 60
DEFAULT_EWMA_HALF_LIFE_SEC = 90.0
DEFAULT_MICROSTRUCT_FIT_WINDOW = 400

# Verdict thresholds — conservative. RED disqualifies the market for the
# 72 h soak; YELLOW means proceed with caution; GREEN means calibration
# looks clean.
THRESHOLD_LB_RED = 0.01       # Ljung-Box p-value below this → RED
THRESHOLD_LB_YELLOW = 0.05    # p-value below this (but ≥ RED) → YELLOW
THRESHOLD_VAR_REL_RED = 0.50  # realized vs implied variance error
THRESHOLD_VAR_REL_YELLOW = 0.20

# Any script touching the shared default data/ directory requires
# --i-mean-it so one-shot operators don't commit tick-level data to the
# monorepo accidentally.
DEFAULT_SHARED_DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DryrunConfig:
    """Parsed CLI + config resolution for a single dry-run."""

    minutes: int
    out_dir: Path
    snapshot_intervals_sec: tuple[int, ...]
    token_id: str | None
    auto_select: bool
    bot_config: dict[str, Any]
    horizon_sec: int = DEFAULT_HORIZON_SEC
    ewma_half_life_sec: float = DEFAULT_EWMA_HALF_LIFE_SEC
    microstruct_fit_window: int = DEFAULT_MICROSTRUCT_FIT_WINDOW
    i_mean_it: bool = False
    # ``None`` means the operator did not pass ``--log-level``; in that
    # case ``resolve_log_level`` falls back to the ``LOG_LEVEL`` env var
    # and finally to "INFO". An explicit CLI value always wins.
    log_level: str | None = None

    def __post_init__(self) -> None:
        if self.minutes <= 0:
            raise ValueError("minutes must be positive")
        if not self.snapshot_intervals_sec:
            raise ValueError("at least one snapshot interval required")
        if not (self.token_id or self.auto_select):
            raise ValueError("pass either --token-id or --auto")
        if self.token_id and self.auto_select:
            raise ValueError("--token-id and --auto are mutually exclusive")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibration_dryrun",
        description="Read-only Track A calibration dry-run against live "
        "Polymarket data. Per-market GO/NO-GO before the 72 h paper-soak.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--token-id", type=str, help="Specific token_id to ingest.")
    src.add_argument(
        "--auto", action="store_true",
        help="Let core.ingest.screener pick the top-liquidity token.",
    )
    p.add_argument(
        "--minutes", type=int, default=15,
        help="Total ingest duration in minutes (default: 15).",
    )
    p.add_argument(
        "--snapshot-every", type=int, default=DEFAULT_SNAPSHOT_INTERVAL_MIN,
        help="Run a calibration snapshot every N minutes (default: 5).",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output directory for Parquet ticks + report.json. Use a unique "
        "per-run path; the shared `data/` dir requires --i-mean-it.",
    )
    p.add_argument(
        "--config", type=Path, default=Path("config/bot.yaml"),
        help="Path to bot.yaml (default: config/bot.yaml).",
    )
    p.add_argument(
        "--horizon-sec", type=int, default=DEFAULT_HORIZON_SEC,
        help="Self-forecast horizon in seconds for the MSE/MAE/QLIKE metric.",
    )
    p.add_argument(
        "--ewma-half-life-sec", type=float, default=DEFAULT_EWMA_HALF_LIFE_SEC,
        help="EWMA σ̂_b²(u) half-life for the self-forecast (§6-tuned default).",
    )
    p.add_argument(
        "--microstruct-fit-window", type=int, default=DEFAULT_MICROSTRUCT_FIT_WINDOW,
        help="Ticks used for the initial MicrostructModel fit before rolling.",
    )
    p.add_argument(
        "--i-mean-it", action="store_true",
        help=f"Required if --out is under the shared default data dir "
             f"({DEFAULT_SHARED_DATA_DIR}/). Ad-hoc --out paths are fine "
             "without the flag.",
    )
    p.add_argument(
        "--log-level", default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity. Default: LOG_LEVEL env var, else INFO. "
             "An explicit CLI value always wins over the env var.",
    )
    return p


def resolve_log_level(
    cli_value: str | None,
    env: "Mapping[str, str] | None" = None,
) -> str:
    """Effective log level with precedence CLI > env > 'INFO'.

    Split out of ``main`` so it is unit-testable without touching the
    real logging framework or the process environment.
    """
    if cli_value:
        return cli_value
    env_map = env if env is not None else os.environ
    return env_map.get("LOG_LEVEL", "INFO")


def parse_args(argv: Sequence[str] | None = None) -> DryrunConfig:
    """Parse CLI args into a :class:`DryrunConfig`. Does not load the YAML."""
    parser = build_argparser()
    ns = parser.parse_args(argv)

    snapshot_stride_sec = int(ns.snapshot_every) * 60
    duration_sec = int(ns.minutes) * 60
    # Snapshots at stride, 2·stride, … up to and including duration.
    snapshot_intervals = tuple(
        t for t in range(snapshot_stride_sec, duration_sec + 1, snapshot_stride_sec)
    )
    if not snapshot_intervals:
        # User asked for fewer minutes than the stride — fall back to a
        # single snapshot at the end.
        snapshot_intervals = (duration_sec,)

    return DryrunConfig(
        minutes=int(ns.minutes),
        out_dir=Path(ns.out),
        snapshot_intervals_sec=snapshot_intervals,
        token_id=ns.token_id,
        auto_select=bool(ns.auto),
        bot_config={},  # filled by the loader
        horizon_sec=int(ns.horizon_sec),
        ewma_half_life_sec=float(ns.ewma_half_life_sec),
        microstruct_fit_window=int(ns.microstruct_fit_window),
        i_mean_it=bool(ns.i_mean_it),
        log_level=ns.log_level,
    )


def load_bot_config(path: Path) -> dict[str, Any]:
    """Load ``bot.yaml`` (optional — empty dict if missing, so unit tests can
    skip the config file)."""
    if not path.exists():
        return {}
    import yaml

    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def check_out_dir_safety(out_dir: Path, i_mean_it: bool) -> None:
    """Raise unless safe: ``--i-mean-it`` is required only when ``out_dir``
    is inside the shared default ``data/`` dir."""
    try:
        resolved = out_dir.resolve()
        shared = DEFAULT_SHARED_DATA_DIR.resolve()
    except (OSError, ValueError):
        return
    try:
        resolved.relative_to(shared)
    except ValueError:
        return  # out_dir is outside shared data/ → safe
    if not i_mean_it:
        raise SystemExit(
            f"--out ({out_dir}) is inside the shared {shared}; pass "
            "--i-mean-it to confirm or choose a per-run path under ./runs/"
        )


def log_header(script_name: str, argv: Sequence[str]) -> dict[str, Any]:
    """Emit the structured header required by scripts/README.md."""
    header = {
        "script": script_name,
        "args": list(argv),
        "ts": datetime.now(UTC).isoformat(),
        "user": getpass.getuser(),
    }
    log.info("header %s", json.dumps(header))
    return header


# ---------------------------------------------------------------------------
# Calibration snapshot — pure
# ---------------------------------------------------------------------------


@dataclass
class CalibrationSnapshot:
    """One snapshot of the full Track A pipeline at a given ingest duration."""

    t_elapsed_sec: int
    n_books: int
    n_trades: int
    n_states: int
    sigma_b_hat: float
    lambda_hat: float
    s_J_sq_hat: float
    em_converged: bool
    em_iters: int
    diagnostics: DiagnosticsReport | None
    self_forecast_mse: float
    self_forecast_mae: float
    self_forecast_qlike: float
    innovations_n: int
    verdict: str


def _fit_microstruct_model(
    books: list[BookSnap],
    trades: list[TradeTick],
    *,
    fit_window: int,
) -> MicrostructModel:
    """Fit the heteroskedastic noise model's coefficients on the first
    ``fit_window`` books using their observed squared innovations as the
    target.

    The target for each book t is the one-step squared ``y`` change,
    :math:`(\\Delta\\log it(\\mathrm{mid}_t))^2`. This is a first-pass
    estimate of :math:`\\sigma_\\eta^2(t)` that the Kalman update will
    refine.
    """
    n = min(len(books), fit_window)
    if n < 5:
        raise ValueError(f"need ≥ 5 books to fit microstruct; got {n}")

    # Pre-index trades by book ts for rolling-window rate feature.
    trade_ts = np.asarray([t.ts.timestamp() for t in trades], dtype=float)
    trade_window_sec = 30.0

    features: list[MicrostructFeatures] = []
    targets: list[float] = []
    prev_y: float | None = None
    for i in range(n):
        book = books[i]
        mid = book.mid
        if mid is None or not (0 < mid < 1):
            continue
        y = math.log(mid / (1.0 - mid))
        # Trades within the 30 s window ending at this book's ts.
        t_now = book.ts.timestamp()
        lo = t_now - trade_window_sec
        window_trades = [
            trades[j] for j in range(len(trades))
            if lo <= trade_ts[j] <= t_now
        ]
        feats = extract_features(book, window_trades, trade_rate_window_sec=trade_window_sec)
        if prev_y is not None:
            features.append(feats)
            targets.append((y - prev_y) ** 2)
        prev_y = y

    if len(features) < 5:
        raise ValueError(
            f"microstruct fit needs ≥ 5 feature rows; got {len(features)}"
        )
    return MicrostructModel.fit_from_features(
        features, targets,
        config=MicrostructConfig(sigma_floor=1e-8, ridge=1e-10),
    )


def _run_filter_stream(
    books: list[BookSnap],
    trades_by_ts: dict[float, list[TradeTick]],
    model: MicrostructModel,
    *,
    sigma_b_seed: float,
    token_id: str,
    grid_hz: float,
) -> tuple[list[LogitState], np.ndarray]:
    """Run canonical_mid → Kalman on a recorded stream.

    Returns (states, standardized_innovations).
    """
    cmid = CanonicalMidFilter(token_id=token_id, grid_hz=grid_hz)
    kf = KalmanFilter(
        token_id=token_id,
        microstruct=model,
        sigma_b=sigma_b_seed,
        initial_variance=1.0,
    )
    states: list[LogitState] = []
    innovations: list[float] = []
    for book in books:
        window_trades = trades_by_ts.get(book.ts.timestamp(), [])
        for cm in cmid.update(book, window_trades):
            states.append(kf.step(cm, book, window_trades))
            if (
                kf.last_innovation is not None
                and kf.last_innovation_variance is not None
                and kf.last_innovation_variance > 0
            ):
                innovations.append(
                    kf.last_innovation / math.sqrt(kf.last_innovation_variance)
                )
    return states, np.asarray(innovations, dtype=float)


def _group_trades_by_book_ts(
    books: list[BookSnap], trades: list[TradeTick],
) -> dict[float, list[TradeTick]]:
    """Bucket trades by the nearest <= book's timestamp."""
    if not books:
        return {}
    book_ts = np.asarray([b.ts.timestamp() for b in books], dtype=float)
    buckets: dict[float, list[TradeTick]] = {t: [] for t in book_ts}
    for t in trades:
        ts = t.ts.timestamp()
        # Right-edge: find the latest book with ts <= trade ts.
        idx = int(np.searchsorted(book_ts, ts, side="right")) - 1
        if 0 <= idx < len(book_ts):
            buckets[book_ts[idx]].append(t)
    return buckets


def _self_forecast_metrics(
    states: list[LogitState],
    *,
    horizon_sec: int,
    ewma_half_life_sec: float,
) -> tuple[float, float, float, int]:
    """Compute self-forecast MSE/MAE/QLIKE on the collected tick series.

    Forecast at origin t:  σ̂_b²_ewma(t) · H   (diffusion-only — the jump
    term is negligible in the paper §6 regime and would require rolling
    EM at the snapshot time, which is an expensive loop we already do
    separately).

    Realized at origin t:  Σ_{u=t+1}^{t+H} (x̂_u - x̂_{u-1})²   (§6.1).

    Returns (mse, mae, qlike, n_origins). ``n_origins == 0`` means the
    stream is too short for the horizon; metrics are NaN.
    """
    if len(states) <= horizon_sec + 1:
        return float("nan"), float("nan"), float("nan"), 0

    ewma = EwmaVar(half_life_sec=ewma_half_life_sec, cold_start_factor=1.0)
    x = np.asarray([s.x_hat for s in states], dtype=float)
    ts = [s.ts for s in states]
    sigma_b_sq = np.zeros(len(states))
    for i in range(1, len(states)):
        dt = max((ts[i] - ts[i - 1]).total_seconds(), 1e-6)
        dx = float(x[i] - x[i - 1])
        ewma.update(dx, dt=dt, jump_posterior=0.0)
        sigma_b_sq[i] = ewma.variance()

    # Realized forward-sum over horizon.
    dx_full = np.diff(x, prepend=x[0])
    dx2 = dx_full * dx_full
    n = len(states)
    origins = range(max(horizon_sec, 30), n - horizon_sec)  # warmup + horizon tail
    if len(list(origins)) == 0:
        return float("nan"), float("nan"), float("nan"), 0

    rv = np.zeros(n, dtype=float)
    for t in range(n - horizon_sec):
        rv[t] = float(np.sum(dx2[t + 1 : t + 1 + horizon_sec]))

    origins_idx = list(origins)
    r = np.asarray([rv[t] for t in origins_idx], dtype=float)
    f = np.asarray([sigma_b_sq[t] * horizon_sec for t in origins_idx], dtype=float)
    err = r - f
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    mask = (r > 0) & (f > 0)
    if not mask.any():
        qlike = float("nan")
    else:
        ratio = r[mask] / f[mask]
        qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
    return mse, mae, qlike, len(origins_idx)


def verdict_for_snapshot(
    diag: DiagnosticsReport | None,
    forecast_qlike: float,
) -> str:
    """Conservative GREEN/YELLOW/RED for one snapshot.

    - RED: Ljung-Box p < 0.01 OR variance_rel_error > 0.50.
    - YELLOW: Ljung-Box p in [0.01, 0.05) OR variance_rel_error in
      (0.20, 0.50] OR Shapiro-Wilk fails OR forecast_qlike > 2.0.
    - GREEN: all three checks pass.

    If diagnostics are missing (not enough data), the verdict is YELLOW.
    """
    if diag is None:
        return VERDICT_YELLOW
    if diag.ljung_box_pvalue < THRESHOLD_LB_RED:
        return VERDICT_RED
    if diag.variance_rel_error > THRESHOLD_VAR_REL_RED:
        return VERDICT_RED
    if (
        diag.ljung_box_pvalue < THRESHOLD_LB_YELLOW
        or diag.variance_rel_error > THRESHOLD_VAR_REL_YELLOW
        or not diag.qq_shapiro_wilk_pass
        or (not math.isnan(forecast_qlike) and forecast_qlike > 2.0)
    ):
        return VERDICT_YELLOW
    return VERDICT_GREEN


def run_calibration_snapshot(
    books: list[BookSnap],
    trades: list[TradeTick],
    *,
    token_id: str,
    t_elapsed_sec: int,
    bot_config: dict[str, Any],
    horizon_sec: int = DEFAULT_HORIZON_SEC,
    ewma_half_life_sec: float = DEFAULT_EWMA_HALF_LIFE_SEC,
    microstruct_fit_window: int = DEFAULT_MICROSTRUCT_FIT_WINDOW,
) -> CalibrationSnapshot:
    """Full offline calibration + diagnostics on a recorded stream.

    This is the pure function the integration test exercises against an
    offline replay. It does not touch I/O, network, or the ParquetStore.
    """
    cal_cfg = bot_config.get("calibration", {}) or {}
    grid_hz = float(cal_cfg.get("kf_grid_hz", 1.0))
    mc_samples = int(cal_cfg.get("mc_draws_per_step", 600))
    sprime_clip = float(cal_cfg.get("sprime_clip", 1e-4))
    mu_cap = float(cal_cfg.get("mu_cap_per_sec", 0.25))
    em_window_sec = int(cal_cfg.get("em_window_sec", 400))

    if len(books) < max(microstruct_fit_window, 50):
        return CalibrationSnapshot(
            t_elapsed_sec=t_elapsed_sec,
            n_books=len(books),
            n_trades=len(trades),
            n_states=0,
            sigma_b_hat=float("nan"),
            lambda_hat=float("nan"),
            s_J_sq_hat=float("nan"),
            em_converged=False,
            em_iters=0,
            diagnostics=None,
            self_forecast_mse=float("nan"),
            self_forecast_mae=float("nan"),
            self_forecast_qlike=float("nan"),
            innovations_n=0,
            verdict=VERDICT_YELLOW,
        )

    model = _fit_microstruct_model(books, trades, fit_window=microstruct_fit_window)
    trades_by_ts = _group_trades_by_book_ts(books, trades)

    # σ_b seed from bi-power variation on the first window (cheap, robust).
    x_seed = []
    prev_y: float | None = None
    for b in books[:microstruct_fit_window]:
        mid = b.mid
        if mid is None or not (0 < mid < 1):
            continue
        y = math.log(mid / (1.0 - mid))
        if prev_y is not None:
            x_seed.append(y - prev_y)
        prev_y = y
    dx_seed = np.asarray(x_seed, dtype=float)
    sigma_b_seed = float(np.sqrt(float(np.mean(dx_seed * dx_seed)))) if dx_seed.size else 0.05
    sigma_b_seed = max(sigma_b_seed, 1e-4)

    states, innov = _run_filter_stream(
        books, trades_by_ts, model,
        sigma_b_seed=sigma_b_seed, token_id=token_id, grid_hz=grid_hz,
    )
    if len(states) < 50:
        return CalibrationSnapshot(
            t_elapsed_sec=t_elapsed_sec,
            n_books=len(books),
            n_trades=len(trades),
            n_states=len(states),
            sigma_b_hat=float("nan"),
            lambda_hat=float("nan"),
            s_J_sq_hat=float("nan"),
            em_converged=False,
            em_iters=0,
            diagnostics=None,
            self_forecast_mse=float("nan"),
            self_forecast_mae=float("nan"),
            self_forecast_qlike=float("nan"),
            innovations_n=int(innov.size),
            verdict=VERDICT_YELLOW,
        )

    drift_cfg = RNDriftConfig(
        mc_samples=mc_samples, mu_cap_per_sec=mu_cap, sprime_clip=sprime_clip,
    )
    # Global-init EM on all states so-far (paper §6.4 "6 global EM steps").
    global_cal = em_calibrate(
        states, initial_params=None,
        max_iters=20, tol=1e-4, drift_config=drift_cfg,
    )
    p = global_cal.final_params

    # Diagnostics.
    # Realized logit variance = Σ (Δx̂)² over the recorded path; implied =
    # σ̂_b² · T + λ̂·ŝ²_J · T (paper §6.1 implied quantity).
    x = np.asarray([s.x_hat for s in states], dtype=float)
    dx = np.diff(x)
    realized_variance = float(np.sum(dx * dx))
    t_total = max((states[-1].ts - states[0].ts).total_seconds(), 1e-6)
    implied_variance = float(
        (p.sigma_b * p.sigma_b + global_cal.jumps.lambda_hat * global_cal.jumps.s_J_sq_hat)
        * t_total
    )
    # Ljung-Box needs ≥ 4 innovations; Shapiro-Wilk ≥ 3.
    diag: DiagnosticsReport | None = None
    if innov.size >= 30:
        warm = min(200, innov.size // 10)
        diag = run_diagnostics(
            innov[warm:],
            realized_variance=realized_variance,
            implied_variance=implied_variance,
            lags=min(20, (innov.size - warm) // 5),
        )

    mse, mae, qlike, _ = _self_forecast_metrics(
        states,
        horizon_sec=horizon_sec,
        ewma_half_life_sec=ewma_half_life_sec,
    )
    verdict = verdict_for_snapshot(diag, qlike)

    return CalibrationSnapshot(
        t_elapsed_sec=t_elapsed_sec,
        n_books=len(books),
        n_trades=len(trades),
        n_states=len(states),
        sigma_b_hat=float(p.sigma_b),
        lambda_hat=float(global_cal.jumps.lambda_hat),
        s_J_sq_hat=float(global_cal.jumps.s_J_sq_hat),
        em_converged=bool(global_cal.converged),
        em_iters=int(global_cal.iters),
        diagnostics=diag,
        self_forecast_mse=float(mse),
        self_forecast_mae=float(mae),
        self_forecast_qlike=float(qlike),
        innovations_n=int(innov.size),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


def _as_jsonable(obj: Any) -> Any:
    """Recursive JSON-safe coercion for dataclass / numpy / datetime output."""
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _as_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_as_jsonable(v) for v in obj.tolist()]
    return str(obj)


def render_report(
    header: dict[str, Any],
    token_id: str,
    snapshots: list[CalibrationSnapshot],
    bot_config: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON-serializable report summary."""
    overall: str
    if not snapshots:
        overall = VERDICT_YELLOW
    elif any(s.verdict == VERDICT_RED for s in snapshots):
        overall = VERDICT_RED
    elif any(s.verdict == VERDICT_YELLOW for s in snapshots):
        overall = VERDICT_YELLOW
    else:
        overall = VERDICT_GREEN

    return _as_jsonable({
        "header": header,
        "token_id": token_id,
        "bot_config_keys": sorted(bot_config.keys()),
        "n_snapshots": len(snapshots),
        "overall_verdict": overall,
        "snapshots": snapshots,
    })


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.json"
    with path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    return path


def console_summary(report: dict[str, Any]) -> str:
    lines = [
        f"[dryrun] token_id={report.get('token_id')}",
        f"[dryrun] overall verdict: {report.get('overall_verdict')}",
    ]
    for s in report.get("snapshots", []):
        lines.append(
            f"  t+{s.get('t_elapsed_sec'):>5}s  "
            f"n_states={s.get('n_states'):>5}  "
            f"σ̂_b={s.get('sigma_b_hat')}  "
            f"λ̂={s.get('lambda_hat')}  "
            f"ŝ²_J={s.get('s_J_sq_hat')}  "
            f"QLIKE={s.get('self_forecast_qlike')}  "
            f"→ {s.get('verdict')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live ingest (async shell around the pure calibration pieces)
# ---------------------------------------------------------------------------


async def ingest_and_snapshot(
    client: Any,  # blksch.core.ingest.polyclient.PolyClient — typed loose so tests can supply a fake
    token_id: str,
    *,
    config: DryrunConfig,
    store: ParquetStore | None = None,
    clock: Any = time.monotonic,
) -> list[CalibrationSnapshot]:
    """Stream CLOB market events for `config.minutes`, write to `store`,
    fire a calibration snapshot at each configured elapsed time."""
    duration_sec = config.minutes * 60
    snapshot_times = sorted(set(config.snapshot_intervals_sec))
    books: list[BookSnap] = []
    trades: list[TradeTick] = []
    snapshots: list[CalibrationSnapshot] = []
    next_idx = 0
    start = clock()

    # REST warm-up snapshot.
    try:
        warm = await client.get_book(token_id)
        books.append(warm)
        if store is not None:
            await store.append_book(warm)
    except Exception as exc:  # noqa: BLE001
        log.warning("REST warm-up failed: %s (continuing on WS only)", exc)

    # Capture the generator so we can explicitly ``aclose`` it on any exit
    # path. Relying on GC for an async-WS generator leaves the CLOB
    # connection half-open on exceptions and delays the process shutdown.
    stream = client.stream_market([token_id])
    try:
        async for event in stream:
            elapsed = float(clock() - start)
            if isinstance(event, BookSnap):
                books.append(event)
                if store is not None:
                    await store.append_book(event)
            elif isinstance(event, TradeTick):
                trades.append(event)
                if store is not None:
                    await store.append_trade(event)
            while next_idx < len(snapshot_times) and elapsed >= snapshot_times[next_idx]:
                t = snapshot_times[next_idx]
                log.info("t+%ds: running calibration snapshot "
                         "(n_books=%d, n_trades=%d)", t, len(books), len(trades))
                snap = run_calibration_snapshot(
                    books, trades,
                    token_id=token_id, t_elapsed_sec=t,
                    bot_config=config.bot_config,
                    horizon_sec=config.horizon_sec,
                    ewma_half_life_sec=config.ewma_half_life_sec,
                    microstruct_fit_window=config.microstruct_fit_window,
                )
                snapshots.append(snap)
                log.info("t+%ds: verdict=%s σ̂_b=%s λ̂=%s ŝ²_J=%s",
                         t, snap.verdict, snap.sigma_b_hat, snap.lambda_hat,
                         snap.s_J_sq_hat)
                next_idx += 1
            if elapsed >= duration_sec:
                break
    finally:
        aclose = getattr(stream, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:  # noqa: BLE001 — cleanup must never mask errors
                log.debug("stream aclose raised", exc_info=True)

    return snapshots


async def _resolve_token_id(client: Any, config: DryrunConfig) -> str:
    """Either use the explicit --token-id or let the screener pick."""
    if config.token_id:
        return config.token_id
    # --auto path.
    from blksch.core.ingest.screener import Screener, ScreenerFilters  # local import keeps unit tests light

    screener_cfg = config.bot_config.get("markets", {}) or {}
    filters = ScreenerFilters(
        min_volume_24h_usd=float(screener_cfg.get("min_volume_24h_usd", 50_000)),
        min_depth_usd_5pct=float(screener_cfg.get("min_depth_usd_5pct", 500)),
        top_n=1,
    )
    screener = Screener(client, filters)
    result = await screener.screen()
    if not result.token_ids:
        raise SystemExit("screener found no qualifying tokens; pass --token-id explicitly")
    return result.token_ids[0]


async def main_async(config: DryrunConfig, header: dict[str, Any]) -> int:
    from blksch.core.ingest.polyclient import PolyClient  # local import

    async with PolyClient() as client:
        token_id = await _resolve_token_id(client, config)
        log.info("ingesting token_id=%s for %d minutes into %s",
                 token_id, config.minutes, config.out_dir)
        # ParquetStore buffers in memory and flushes on ``close()``. On any
        # exception inside ``ingest_and_snapshot`` the buffer would be
        # lost without the explicit finally; the outer ``async with
        # PolyClient()`` would then just propagate the exception and exit
        # with the last partial snapshots un-persisted. Close it
        # regardless of exit path.
        store = ParquetStore(config.out_dir / "ticks")
        try:
            snapshots = await ingest_and_snapshot(
                client, token_id, config=config, store=store,
            )
        finally:
            try:
                await store.close()
            except Exception:  # noqa: BLE001 — cleanup must not mask the original error
                log.warning("ParquetStore.close() raised during teardown", exc_info=True)

        report = render_report(header, token_id, snapshots, config.bot_config)
        path = write_report(report, config.out_dir)
        log.info("wrote report to %s", path)
        print(console_summary(report))
        return 0 if report["overall_verdict"] != VERDICT_RED else 1


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    config = parse_args(argv)
    check_out_dir_safety(config.out_dir, config.i_mean_it)
    bot_config = load_bot_config(Path("config/bot.yaml"))
    # DryrunConfig is frozen — build a new one with the yaml loaded.
    config = DryrunConfig(
        minutes=config.minutes,
        out_dir=config.out_dir,
        snapshot_intervals_sec=config.snapshot_intervals_sec,
        token_id=config.token_id,
        auto_select=config.auto_select,
        bot_config=bot_config,
        horizon_sec=config.horizon_sec,
        ewma_half_life_sec=config.ewma_half_life_sec,
        microstruct_fit_window=config.microstruct_fit_window,
        i_mean_it=config.i_mean_it,
        log_level=config.log_level,
    )
    logging.basicConfig(
        level=resolve_log_level(config.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    header = log_header("calibration_dryrun.py", argv)
    return asyncio.run(main_async(config, header))


if __name__ == "__main__":
    sys.exit(main())
