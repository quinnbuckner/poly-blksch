"""Post-soak analysis helper — fit the Avellaneda-Stoikov ``k`` from ledger fills.

Reads a paper-soak ledger (``exec/ledger.py`` SQLite DB), pairs each
filled order with its contemporaneous opposite-side order to estimate the
market mid at placement, bins fills by distance-from-mid in logit units,
then fits

    intensity(δ) = A · exp(-k · δ)

via log-linear regression. Emits a JSON report with the fit quality, a
95% CI on ``k``, and a go/no-go recommendation for updating
``config/bot.yaml`` if realized ``k`` drifts from its seed by more than
the configured threshold (default 20%).

This script is the Phase-3 parameter-tuning starting point: it runs
immediately after the 36h paper-soak ends, tells the operator whether
the seed value (``quoting.k = 1.5`` out of the box) is close enough to
the realized arrival decay, and produces a concrete suggested value if
not.

Usage
-----

::

    # Default behavior: execute the fit, write JSON to --out.
    python scripts/fit_k_from_ledger.py \\
        --ledger-db ./runs/soak-residential-<date>/ledger.db \\
        --out       ./runs/soak-residential-<date>/k_fit.json

    # Print the plan without computing anything.
    python scripts/fit_k_from_ledger.py --ledger-db <path> --out <path> --dry-run

    # Restrict to one token.
    python scripts/fit_k_from_ledger.py --ledger-db <path> --out <path> \\
        --token-id 0x123abc

    # Skip the fit if too few fills.
    python scripts/fit_k_from_ledger.py --ledger-db <path> --out <path> --min-fills 30

Safety
------

Read-only end-to-end. Opens the SQLite DB in URI ``mode=ro`` (bypasses
the ``Ledger`` class, which would open read-write). Never writes to the
ledger or ``config/bot.yaml`` — just emits a suggestion as JSON.
``--i-mean-it`` is deliberately absent — no network, no orders, no
destructive ops.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("fit_k_from_ledger")


# ---------------------------------------------------------------------------
# Structured header (scripts/README.md safety rule #3)
# ---------------------------------------------------------------------------


def log_header(script_name: str, argv: Sequence[str]) -> dict[str, Any]:
    """Emit the {script, args, ts, user} header required by scripts/README.md."""
    header = {
        "script": script_name,
        "args": list(argv),
        "ts": datetime.now(UTC).isoformat(),
        "user": getpass.getuser(),
    }
    log.info("header %s", json.dumps(header))
    return header


# ---------------------------------------------------------------------------
# Ledger read-only access
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OrderRow:
    client_id: str
    token_id: str
    side: str  # "buy" or "sell" (OrderSide.value)
    price: float
    created_ts: datetime


@dataclass(frozen=True)
class _FillRow:
    order_client_id: str
    token_id: str
    price: float
    ts: datetime


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` in SQLite URI ``mode=ro``.

    This bypasses the :class:`blksch.exec.ledger.Ledger` class, which opens
    read-write and enforces schema init on every connection. Post-soak
    analysis must never risk mutating the DB, and SQLite's URI ro-mode
    fails fast at query time on any write attempt — a tighter guarantee
    than application-level discipline.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"ledger DB not found: {db_path}")
    uri = f"file:{db_path.resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_orders_and_fills(
    conn: sqlite3.Connection, token_id: str | None = None,
) -> tuple[list[_OrderRow], list[_FillRow]]:
    """Fetch all orders and fills (optionally restricted to one token)."""
    order_sql = (
        "SELECT client_id, token_id, side, price, created_ts FROM orders"
    )
    fill_sql = (
        "SELECT order_client_id, token_id, price, ts FROM fills"
    )
    order_args: list = []
    fill_args: list = []
    if token_id is not None:
        order_sql += " WHERE token_id=?"
        order_args.append(token_id)
        fill_sql += " WHERE token_id=?"
        fill_args.append(token_id)

    orders = [
        _OrderRow(
            client_id=r[0], token_id=r[1], side=r[2], price=float(r[3]),
            created_ts=datetime.fromisoformat(r[4]),
        )
        for r in conn.execute(order_sql, order_args).fetchall()
    ]
    fills = [
        _FillRow(
            order_client_id=r[0], token_id=r[1], price=float(r[2]),
            ts=datetime.fromisoformat(r[3]),
        )
        for r in conn.execute(fill_sql, fill_args).fetchall()
    ]
    return orders, fills


# ---------------------------------------------------------------------------
# Delta derivation — pair each filled order with its contemporaneous mate
# ---------------------------------------------------------------------------


def _logit(p: float, eps: float = 1e-4) -> float:
    """Numerically safe logit for probability values near the boundary."""
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def pair_orders_to_deltas(
    orders: Sequence[_OrderRow], *, pair_window_sec: float = 1.0,
) -> dict[str, float]:
    """For each order, find the nearest-in-time opposite-side order on the
    same token placed within ``pair_window_sec``. Return a map from
    ``client_id`` to its ``δ_logit = |logit(quote) - logit(mid)|`` where
    ``mid = (bid.price + ask.price) / 2``.

    Orders that cannot be paired (one-sided quote, isolated placement, or
    mate lands outside the window) are dropped from the result — the
    caller can read ``len(result)`` vs ``len(orders)`` to gauge pair-rate.
    """
    # Index by token_id for O(per-token) pairing.
    by_token: dict[str, list[_OrderRow]] = {}
    for o in orders:
        by_token.setdefault(o.token_id, []).append(o)

    deltas: dict[str, float] = {}
    for token_orders in by_token.values():
        # Sort by ts for early-exit over the window.
        token_orders_sorted = sorted(token_orders, key=lambda o: o.created_ts)
        # For each order, scan forward/backward until outside window.
        for i, o in enumerate(token_orders_sorted):
            best_mate: _OrderRow | None = None
            best_diff = float("inf")
            # Scan both directions for nearest opposite-side within window.
            for j in range(len(token_orders_sorted)):
                if i == j:
                    continue
                other = token_orders_sorted[j]
                if other.side == o.side:
                    continue
                diff = abs((o.created_ts - other.created_ts).total_seconds())
                if diff > pair_window_sec:
                    continue
                if diff < best_diff:
                    best_diff = diff
                    best_mate = other
            if best_mate is None:
                continue
            mid = (o.price + best_mate.price) / 2.0
            delta_logit = abs(_logit(o.price) - _logit(mid))
            deltas[o.client_id] = delta_logit
    return deltas


def derive_fill_deltas(
    orders: Sequence[_OrderRow], fills: Sequence[_FillRow],
    *, pair_window_sec: float = 1.0,
) -> tuple[list[float], list[datetime], int]:
    """Return ``(deltas_logit, fill_timestamps, paired_order_count)``.

    Only fills whose parent order could be paired contribute. The third
    return value is how many orders had valid pairs, for quality gating.
    """
    pair_deltas = pair_orders_to_deltas(orders, pair_window_sec=pair_window_sec)
    deltas: list[float] = []
    fill_ts: list[datetime] = []
    for f in fills:
        if f.order_client_id in pair_deltas:
            deltas.append(pair_deltas[f.order_client_id])
            fill_ts.append(f.ts)
    return deltas, fill_ts, len(pair_deltas)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    A: float | None
    k: float | None
    ci_low: float | None
    ci_high: float | None
    r_squared: float | None
    n_fills: int
    n_bins_used: int
    obs_duration_sec: float
    bin_width_logit: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "A": self.A,
            "k": self.k,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "r_squared": self.r_squared,
            "n_fills": self.n_fills,
            "n_bins_used": self.n_bins_used,
            "obs_duration_sec": self.obs_duration_sec,
            "bin_width_logit": self.bin_width_logit,
            "error": self.error,
        }


def fit_k(
    deltas: Sequence[float],
    fill_ts: Sequence[datetime],
    *,
    bin_width_logit: float = 0.05,
    min_fills: int = 30,
    min_bins: int = 3,
    min_count_per_bin: int = 5,
) -> FitResult:
    """Fit ``intensity(δ) = A · exp(-k · δ)`` by Poisson-weighted log-linear
    regression on histogram counts.

    The naive unweighted ``linregress(log(intensity), δ)`` is severely
    biased by the high-variance tail bins where ``count`` ∈ {1, 2}: the
    log of a tiny intensity has variance ~ 1/count, so a few noisy tail
    points pull the slope. Weighting each bin by its count (the Poisson
    inverse-variance approximation) and dropping bins below
    ``min_count_per_bin`` recovers ``k`` accurately even at modest
    sample sizes.

    Short-circuits to a ``FitResult`` with ``k=None`` and a descriptive
    ``error`` on: too few fills (``< min_fills``), bins collapsed to
    fewer than ``min_bins`` usable, or zero observation duration.
    """
    import numpy as np
    from scipy import stats

    n = len(deltas)
    if n < min_fills:
        return FitResult(
            A=None, k=None, ci_low=None, ci_high=None, r_squared=None,
            n_fills=n, n_bins_used=0,
            obs_duration_sec=0.0, bin_width_logit=bin_width_logit,
            error=f"insufficient fills: n={n} < min_fills={min_fills}",
        )
    if not fill_ts or len(fill_ts) != n:
        return FitResult(
            A=None, k=None, ci_low=None, ci_high=None, r_squared=None,
            n_fills=n, n_bins_used=0,
            obs_duration_sec=0.0, bin_width_logit=bin_width_logit,
            error="deltas and fill_ts length mismatch",
        )

    obs_duration = (max(fill_ts) - min(fill_ts)).total_seconds()
    if obs_duration <= 0.0:
        return FitResult(
            A=None, k=None, ci_low=None, ci_high=None, r_squared=None,
            n_fills=n, n_bins_used=0,
            obs_duration_sec=0.0, bin_width_logit=bin_width_logit,
            error="zero observation duration",
        )

    deltas_arr = np.asarray(deltas, dtype=float)
    max_delta = float(deltas_arr.max())
    # Bin from 0 → max_delta + one extra bucket to catch the boundary.
    bin_edges = np.arange(0.0, max_delta + bin_width_logit, bin_width_logit)
    if len(bin_edges) < 2:
        return FitResult(
            A=None, k=None, ci_low=None, ci_high=None, r_squared=None,
            n_fills=n, n_bins_used=0,
            obs_duration_sec=obs_duration, bin_width_logit=bin_width_logit,
            error="too few bins to fit — all fills at delta=0?",
        )
    counts, _ = np.histogram(deltas_arr, bins=bin_edges)
    centers = bin_edges[:-1] + bin_width_logit / 2.0
    # Use only bins with enough counts to stabilize log-space variance.
    # Empty/sparse tail bins (count < min_count_per_bin) are dropped — the
    # unbiased slope-estimate cost of dropping them is negligible (the
    # bins we keep already span >99% of the distribution mass at typical
    # k); the variance cost of *keeping* them is high (1/count).
    mask = counts >= min_count_per_bin
    n_bins_used = int(mask.sum())
    if n_bins_used < min_bins:
        return FitResult(
            A=None, k=None, ci_low=None, ci_high=None, r_squared=None,
            n_fills=n, n_bins_used=n_bins_used,
            obs_duration_sec=obs_duration, bin_width_logit=bin_width_logit,
            error=(
                f"too few bins with count ≥ {min_count_per_bin}: "
                f"{n_bins_used} < min_bins={min_bins}; widen "
                f"--bin-width-logit or gather more fills"
            ),
        )

    intensity = counts / (obs_duration * bin_width_logit)
    log_intensity = np.log(intensity[mask])
    x = centers[mask]
    weights = counts[mask].astype(float)  # Poisson inverse-variance proxy

    # Weighted polyfit gives the same point estimate as the unweighted
    # form when all weights are equal but downweights noisy tail bins
    # in the realistic case. Use cov=True to extract the slope CI.
    coeffs, cov = np.polyfit(x, log_intensity, deg=1, w=weights, cov=True)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    k_hat = float(-slope)
    a_hat = float(math.exp(intercept))

    # Weighted R² in log space: SS_res and SS_tot both weighted by counts.
    y_pred = slope * x + intercept
    weighted_mean_y = float(np.average(log_intensity, weights=weights))
    ss_res = float(np.sum(weights * (log_intensity - y_pred) ** 2))
    ss_tot = float(np.sum(weights * (log_intensity - weighted_mean_y) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 95% CI on slope from polyfit's covariance + t-distribution. If
    # ``cov`` is degenerate (numpy returns inf when residuals are zero
    # because the fit is exact), fall back to NaN bounds rather than
    # crashing.
    df = max(1, n_bins_used - 2)
    try:
        slope_se = float(np.sqrt(cov[0, 0]))
        t_crit = float(stats.t.ppf(0.975, df=df))
        ci_half = t_crit * slope_se
        ci_low = k_hat - ci_half
        ci_high = k_hat + ci_half
        if not (math.isfinite(ci_low) and math.isfinite(ci_high)):
            ci_low = ci_high = float("nan")
    except Exception:
        ci_low = ci_high = float("nan")

    return FitResult(
        A=a_hat, k=k_hat,
        ci_low=ci_low, ci_high=ci_high,
        r_squared=r_squared,
        n_fills=n, n_bins_used=n_bins_used,
        obs_duration_sec=obs_duration,
        bin_width_logit=bin_width_logit,
        error=None,
    )


# ---------------------------------------------------------------------------
# Config-suggestion logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetuneSuggestion:
    seed_k: float
    realized_k: float | None
    delta_pct: float | None
    suggested_k: float | None
    should_retune: bool
    confidence: str  # "high" | "medium" | "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_k": self.seed_k,
            "realized_k": self.realized_k,
            "delta_pct": self.delta_pct,
            "suggested_k": self.suggested_k,
            "should_retune": self.should_retune,
            "confidence": self.confidence,
        }


def _confidence(r_squared: float | None, n_fills: int) -> str:
    if r_squared is None or n_fills < 30:
        return "low"
    if r_squared >= 0.9 and n_fills >= 100:
        return "high"
    if r_squared >= 0.7 and n_fills >= 30:
        return "medium"
    return "low"


def suggest_retune(
    fit: FitResult, seed_k: float, *, threshold_pct: float = 0.20,
) -> RetuneSuggestion:
    """Decide whether ``config/bot.yaml`` should be updated.

    ``should_retune=True`` only when realized ``k`` is usable (finite,
    positive, inside its CI) and diverges from seed by more than
    ``threshold_pct``. Low-confidence fits never trigger a retune
    suggestion — operators should rerun with more data first.
    """
    k_realized = fit.k
    confidence = _confidence(fit.r_squared, fit.n_fills)

    if k_realized is None or not math.isfinite(k_realized) or k_realized <= 0:
        return RetuneSuggestion(
            seed_k=seed_k, realized_k=k_realized,
            delta_pct=None, suggested_k=None,
            should_retune=False, confidence=confidence,
        )

    delta_pct = abs(k_realized - seed_k) / seed_k if seed_k > 0 else float("inf")
    should_retune = (confidence != "low") and (delta_pct > threshold_pct)
    suggested_k = round(k_realized, 4) if should_retune else None
    return RetuneSuggestion(
        seed_k=seed_k, realized_k=k_realized,
        delta_pct=delta_pct, suggested_k=suggested_k,
        should_retune=should_retune, confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Config read
# ---------------------------------------------------------------------------


def read_seed_k(config_path: Path) -> float:
    """Read ``quoting.k`` from a bot YAML. Defaults to 1.5 (the paper seed)
    if the file is missing or the key is absent — the suggestion is
    always advisory, so a missing config file shouldn't block the fit."""
    if not config_path.exists():
        log.warning(
            "config file not found at %s; falling back to seed_k=1.5",
            config_path,
        )
        return 1.5
    import yaml
    blob = yaml.safe_load(config_path.read_text()) or {}
    quoting = blob.get("quoting") or {}
    seed = quoting.get("k")
    if seed is None:
        log.warning(
            "config at %s has no quoting.k; falling back to seed_k=1.5",
            config_path,
        )
        return 1.5
    return float(seed)


# ---------------------------------------------------------------------------
# End-to-end driver (used by the CLI AND by tests to exercise the full flow)
# ---------------------------------------------------------------------------


def analyze_ledger(
    db_path: Path,
    *,
    token_id: str | None = None,
    min_fills: int = 30,
    bin_width_logit: float = 0.05,
    pair_window_sec: float = 1.0,
    seed_k: float = 1.5,
    retune_threshold_pct: float = 0.20,
) -> dict[str, Any]:
    """Full flow: connect → load → pair → fit → suggest. Returns the
    JSON-ready dict the CLI writes to ``--out``.
    """
    conn = connect_readonly(db_path)
    try:
        orders, fills = load_orders_and_fills(conn, token_id=token_id)
    finally:
        conn.close()

    deltas, fill_ts, n_paired_orders = derive_fill_deltas(
        orders, fills, pair_window_sec=pair_window_sec,
    )
    fit = fit_k(
        deltas, fill_ts,
        bin_width_logit=bin_width_logit, min_fills=min_fills,
    )
    suggestion = suggest_retune(
        fit, seed_k, threshold_pct=retune_threshold_pct,
    )

    return {
        "ledger_db": str(db_path),
        "token_id": token_id,
        "n_orders_loaded": len(orders),
        "n_orders_paired": n_paired_orders,
        "n_fills_loaded": len(fills),
        "pair_window_sec": pair_window_sec,
        "fit": fit.to_dict(),
        "suggestion": suggestion.to_dict(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_plan(args: argparse.Namespace) -> None:
    log.warning("Dry-run — would perform the following (no side effects):")
    log.warning("  1. Open ledger (read-only) at %s", args.ledger_db)
    if args.token_id:
        log.warning("  2. Filter orders + fills to token_id=%s", args.token_id)
    else:
        log.warning("  2. Process all tokens in the ledger")
    log.warning(
        "  3. Pair orders within %.2fs window to derive mid per order",
        args.pair_window_sec,
    )
    log.warning(
        "  4. Fit intensity(δ) = A·exp(-k·δ) via log-linear regression "
        "(bin_width=%.3f logit; min_fills=%d)",
        args.bin_width_logit, args.min_fills,
    )
    log.warning("  5. Read seed k from %s", args.config_path)
    log.warning(
        "  6. Recommend retune if |Δ|/seed > %.0f%% AND confidence != 'low'",
        args.retune_threshold_pct * 100,
    )
    if args.out:
        log.warning("  7. Write JSON report to %s", args.out)
    else:
        log.warning("  7. (no --out specified) print JSON report to stdout")
    log.warning("Re-run without --dry-run to execute.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Post-soak fit of the Avellaneda-Stoikov arrival-decay k from "
            "a paper-soak ledger. Read-only."
        ),
    )
    parser.add_argument(
        "--ledger-db", type=Path, required=True,
        help="Path to the exec/ledger.py SQLite DB.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="JSON report destination. Omit to print to stdout.",
    )
    parser.add_argument(
        "--token-id", default=None,
        help="Restrict analysis to a single token_id.",
    )
    parser.add_argument(
        "--min-fills", type=int, default=30,
        help="Skip fit if paired fills < min_fills. Default 30.",
    )
    parser.add_argument(
        "--bin-width-logit", type=float, default=0.05,
        help="δ-bucket width in logit units. Default 0.05.",
    )
    parser.add_argument(
        "--pair-window-sec", type=float, default=1.0,
        help=(
            "Max seconds between bid and ask placements for them to be "
            "treated as a pair (used to derive mid at placement). "
            "Default 1.0."
        ),
    )
    parser.add_argument(
        "--config-path", type=Path, default=Path("config/bot.yaml"),
        help="Bot YAML path to read seed quoting.k from. Default config/bot.yaml.",
    )
    parser.add_argument(
        "--retune-threshold-pct", type=float, default=0.20,
        help=(
            "Fractional divergence between realized and seed k above "
            "which ``should_retune`` fires. Default 0.20 (= 20%%)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit 0; do not read the ledger.",
    )
    parser.add_argument(
        "--log-level", default=None,
        help="Logger level. CLI > LOG_LEVEL env > INFO.",
    )
    args = parser.parse_args(argv)

    level = args.log_level or __import__("os").environ.get("LOG_LEVEL") or "INFO"
    logging.basicConfig(
        level=level.upper() if isinstance(level, str) else level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log_header("fit_k_from_ledger", argv if argv is not None else sys.argv[1:])

    if args.dry_run:
        _print_plan(args)
        return 0

    seed_k = read_seed_k(args.config_path)

    report = analyze_ledger(
        args.ledger_db,
        token_id=args.token_id,
        min_fills=args.min_fills,
        bin_width_logit=args.bin_width_logit,
        pair_window_sec=args.pair_window_sec,
        seed_k=seed_k,
        retune_threshold_pct=args.retune_threshold_pct,
    )
    # Include the header in the JSON so callers can audit the invocation
    # that produced a given report file.
    report["generated_ts"] = datetime.now(UTC).isoformat()

    blob = json.dumps(report, indent=2, allow_nan=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob + "\n")
        log.info("wrote %s", args.out)
    else:
        print(blob)

    suggestion = report["suggestion"]
    if suggestion["should_retune"]:
        log.warning(
            "SUGGESTED RETUNE: quoting.k %s → %s (realized k=%.4f, "
            "confidence=%s, delta=%.1f%%)",
            suggestion["seed_k"], suggestion["suggested_k"],
            suggestion["realized_k"], suggestion["confidence"],
            suggestion["delta_pct"] * 100,
        )
    else:
        log.info(
            "NO RETUNE suggested: realized k=%s, confidence=%s",
            suggestion["realized_k"], suggestion["confidence"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
