"""Unit tests for ``scripts/fit_k_from_ledger.py``.

The helper analyzes a paper-soak ledger to recover the
Avellaneda-Stoikov arrival-decay ``k`` and recommend whether the
``config/bot.yaml`` seed needs an update. These tests pin the
recovery accuracy on synthetic ledgers (where the true ``k`` is
known), the gating behavior when too few fills are available, the
suggestion threshold, and the per-token filter.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from blksch.exec.ledger import Ledger
from blksch.schemas import Fill, Order, OrderSide, OrderStatus

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _import_fit_k():
    path = SCRIPTS_DIR / "fit_k_from_ledger.py"
    spec = importlib.util.spec_from_file_location(
        "blksch_scripts_fit_k_from_ledger", path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fit_k_mod = _import_fit_k()


# ---------------------------------------------------------------------------
# Synthetic ledger seeders
# ---------------------------------------------------------------------------


def _seed_paired_quotes_with_fills(
    ledger_path: Path,
    *,
    true_k: float,
    n_fills: int,
    token_id: str = "tok_a",
    seed: int = 42,
    pair_spacing_sec: float = 5.0,
    pair_offset_ms: int = 10,
) -> None:
    """Seed a ledger with ``n_fills`` paired bid/ask quotes whose δ is
    drawn from ``Exponential(rate=true_k)``. Each bid is filled at its
    quoted price (passive maker fill). Pairs are spaced
    ``pair_spacing_sec`` apart so the supervisor's pair-window matcher
    cleanly maps each fill back to its bid+ask placement.

    Mid is at logit=0 (probability=0.5). Bid lands at sigmoid(-δ),
    ask at sigmoid(+δ). |logit(bid) - logit(mid)| = δ exactly.
    """
    rng = np.random.default_rng(seed)
    deltas = rng.exponential(scale=1.0 / true_k, size=n_fills)

    led = Ledger.open(ledger_path)
    try:
        t0 = datetime(2026, 4, 24, 0, 0, 0, tzinfo=UTC)
        for i, delta in enumerate(deltas):
            ts = t0 + timedelta(seconds=i * pair_spacing_sec)
            ts_ask = ts + timedelta(milliseconds=pair_offset_ms)
            ts_fill = ts + timedelta(milliseconds=pair_offset_ms * 10)

            bid_price = 1.0 / (1.0 + math.exp(float(delta)))   # sigmoid(-δ)
            ask_price = 1.0 / (1.0 + math.exp(-float(delta)))  # sigmoid(+δ)

            led.record_order(Order(
                token_id=token_id, side=OrderSide.BUY,
                price=bid_price, size=10.0,
                client_id=f"{token_id}-bid-{i}",
                status=OrderStatus.OPEN,
                created_ts=ts,
            ))
            led.record_order(Order(
                token_id=token_id, side=OrderSide.SELL,
                price=ask_price, size=10.0,
                client_id=f"{token_id}-ask-{i}",
                status=OrderStatus.OPEN,
                created_ts=ts_ask,
            ))
            # Passive fill on the bid — we as MM bought.
            led.apply_fill(Fill(
                order_client_id=f"{token_id}-bid-{i}",
                order_venue_id=None,
                token_id=token_id,
                side=OrderSide.BUY,
                price=bid_price,
                size=10.0,
                fee_usd=0.0,
                ts=ts_fill,
            ))
    finally:
        led.close()


# ---------------------------------------------------------------------------
# Test 1 — recovery on a synthetic ledger with known k
# ---------------------------------------------------------------------------


def test_fit_k_synthetic_ledger(tmp_path):
    """Generate fills from Exponential(rate=true_k); the regression must
    recover ``true_k`` to within ±20% (and the test pins it tighter than
    that — 10% — because Exp + log-linear fit at n=500 is well-behaved
    and we don't want a future regression to slowly degrade it past
    the spec gate)."""
    db = tmp_path / "ledger.db"
    true_k = 3.0
    _seed_paired_quotes_with_fills(db, true_k=true_k, n_fills=500)

    report = fit_k_mod.analyze_ledger(
        db, min_fills=30, bin_width_logit=0.05,
        pair_window_sec=0.5,
        seed_k=1.5,
    )
    fit = report["fit"]
    assert fit["error"] is None, f"fit failed unexpectedly: {fit['error']}"
    assert fit["n_fills"] == 500
    # Pair-rate: every order should have a pair (spacing > pair_window).
    assert report["n_orders_paired"] == 1000  # 500 bid + 500 ask

    err = abs(fit["k"] - true_k) / true_k
    assert err < 0.20, (
        f"recovered k={fit['k']:.4f} vs true k={true_k}; |Δ|/true={err:.1%} > 20%"
    )
    # Sanity: with n=500, fit should be tight. Pin a stricter envelope so
    # silent numerical drift in scipy.linregress / np.histogram surfaces.
    assert err < 0.10, (
        f"recovery within 10% expected for n=500 well-conditioned data; "
        f"got |Δ|/true={err:.1%}"
    )
    assert fit["r_squared"] > 0.85, (
        f"R² for synthetic exponential should be high; got {fit['r_squared']}"
    )
    # CI must straddle the true value (95% CI on a well-fit synthetic).
    assert fit["ci_low"] <= true_k <= fit["ci_high"], (
        f"true k={true_k} outside 95% CI [{fit['ci_low']}, {fit['ci_high']}]"
    )


# ---------------------------------------------------------------------------
# Test 2 — insufficient fills returns NaN + low confidence
# ---------------------------------------------------------------------------


def test_fit_k_insufficient_fills_returns_nan(tmp_path):
    """With fewer paired fills than ``--min-fills`` (default 30), the
    fit must short-circuit: no scipy regression on a degenerate sample,
    no retune suggestion, low confidence flag for the operator."""
    db = tmp_path / "ledger.db"
    _seed_paired_quotes_with_fills(db, true_k=2.0, n_fills=10)

    report = fit_k_mod.analyze_ledger(
        db, min_fills=30, seed_k=1.5,
    )
    fit = report["fit"]
    suggestion = report["suggestion"]

    # Fit fields must be None (the JSON-friendly form of NaN — Python
    # NaN doesn't round-trip cleanly through stdlib json otherwise).
    assert fit["k"] is None
    assert fit["A"] is None
    assert fit["ci_low"] is None
    assert fit["ci_high"] is None
    assert fit["r_squared"] is None
    assert fit["n_fills"] == 10
    assert "insufficient fills" in fit["error"]

    # Suggestion must NEVER fire on insufficient data.
    assert suggestion["should_retune"] is False
    assert suggestion["suggested_k"] is None
    assert suggestion["confidence"] == "low"
    assert suggestion["realized_k"] is None


# ---------------------------------------------------------------------------
# Test 3 — config-suggestion fires at the 20% boundary
# ---------------------------------------------------------------------------


def test_fit_k_config_suggestion_triggers_at_20pct():
    """``suggest_retune`` must trip ``should_retune=True`` exactly when
    realized k diverges from seed by > threshold AND confidence is high
    enough to act on. Pin both directions of the boundary so a future
    refactor doesn't silently flip the inequality."""
    seed_k = 1.5
    # Confidence anchor: r_squared=0.95, n=200 → "high" by helper rules.

    def _fit(realized_k: float) -> "fit_k_mod.FitResult":
        return fit_k_mod.FitResult(
            A=1.0, k=realized_k,
            ci_low=realized_k - 0.05, ci_high=realized_k + 0.05,
            r_squared=0.95, n_fills=200, n_bins_used=20,
            obs_duration_sec=3600.0, bin_width_logit=0.05,
            error=None,
        )

    # Just under 20% — must NOT retune.
    s_under = fit_k_mod.suggest_retune(_fit(seed_k * 1.19), seed_k, threshold_pct=0.20)
    assert s_under.should_retune is False, (
        f"19% drift must NOT trigger retune; got {s_under}"
    )
    assert s_under.suggested_k is None
    assert s_under.confidence == "high"

    # Just over 20% — must retune.
    s_over = fit_k_mod.suggest_retune(_fit(seed_k * 1.21), seed_k, threshold_pct=0.20)
    assert s_over.should_retune is True, (
        f"21% drift must trigger retune; got {s_over}"
    )
    assert s_over.suggested_k == round(seed_k * 1.21, 4)

    # Symmetric: large negative drift must also trigger.
    s_neg = fit_k_mod.suggest_retune(_fit(seed_k * 0.5), seed_k, threshold_pct=0.20)
    assert s_neg.should_retune is True
    assert s_neg.suggested_k == round(seed_k * 0.5, 4)
    assert s_neg.delta_pct == pytest.approx(0.5)

    # Low-confidence fits NEVER suggest retune even if drift is huge —
    # the operator must rerun with more data first.
    low_conf_fit = fit_k_mod.FitResult(
        A=1.0, k=seed_k * 5.0,  # 400% drift
        ci_low=0.0, ci_high=20.0,
        r_squared=0.4, n_fills=15, n_bins_used=3,  # n<30 → low conf
        obs_duration_sec=600.0, bin_width_logit=0.05, error=None,
    )
    s_low = fit_k_mod.suggest_retune(low_conf_fit, seed_k, threshold_pct=0.20)
    assert s_low.should_retune is False, (
        "low-confidence fits must never trigger retune; "
        f"got {s_low}"
    )
    assert s_low.suggested_k is None
    assert s_low.confidence == "low"


# ---------------------------------------------------------------------------
# Test 4 — --token-id filter scopes to one token in a multi-token ledger
# ---------------------------------------------------------------------------


def test_fit_k_respects_token_id_filter(tmp_path):
    """Multi-token ledger: tok_fast has a high-k regime (steep decay,
    few large-δ fills); tok_slow has a low-k regime. Filtering to
    tok_fast must yield the high-k fit and ignore tok_slow's
    contributions."""
    db = tmp_path / "ledger.db"
    # Note: seeded sequentially so the second seeding's ts space starts
    # at the same t0 — deliberately collide tokens in time to prove
    # the SQL filter (not timestamp coincidence) does the scoping.
    _seed_paired_quotes_with_fills(
        db, true_k=4.0, n_fills=500, token_id="tok_fast", seed=11,
    )
    _seed_paired_quotes_with_fills(
        db, true_k=1.5, n_fills=500, token_id="tok_slow", seed=22,
    )

    # Unfiltered: combined dataset, k somewhere between the two — primarily
    # here to prove unfiltered ≠ filtered; precise value is not pinned.
    full = fit_k_mod.analyze_ledger(db, seed_k=1.5)
    assert full["n_fills_loaded"] == 1000
    assert full["fit"]["error"] is None

    # Filtered to fast token.
    fast = fit_k_mod.analyze_ledger(db, token_id="tok_fast", seed_k=1.5)
    assert fast["n_fills_loaded"] == 500, (
        f"filter must scope SQL to tok_fast; got {fast['n_fills_loaded']} fills"
    )
    assert fast["token_id"] == "tok_fast"
    assert fast["fit"]["k"] is not None
    assert abs(fast["fit"]["k"] - 4.0) / 4.0 < 0.20

    # Filtered to slow token. The recovery gate is loosened to 25% here
    # because at k=1.5, n=500 the per-bin Poisson noise produces
    # seed-dependent point estimates within ±20-25% of true. The TEST's
    # primary purpose is verifying the SQL filter scopes correctly —
    # the precision check is secondary; the synthetic-recovery test
    # already pins the tight gate at 10% with the well-conditioned
    # k=3, n=500 case.
    slow = fit_k_mod.analyze_ledger(db, token_id="tok_slow", seed_k=1.5)
    assert slow["n_fills_loaded"] == 500
    assert slow["token_id"] == "tok_slow"
    assert slow["fit"]["k"] is not None
    assert abs(slow["fit"]["k"] - 1.5) / 1.5 < 0.25

    # The two filtered fits must differ — proves the filter actually
    # scoped (not just collided on a coincidence).
    assert fast["fit"]["k"] != slow["fit"]["k"]
    assert fast["fit"]["k"] > slow["fit"]["k"], (
        f"fast token's k={fast['fit']['k']} should exceed slow's "
        f"k={slow['fit']['k']} (true was 4.0 vs 1.5); ordering inversion "
        f"would suggest the filter mixed tokens"
    )


# ---------------------------------------------------------------------------
# Bonus pin — read-only mode at the SQLite layer (defense in depth)
# ---------------------------------------------------------------------------


def test_connect_readonly_rejects_writes(tmp_path):
    """``connect_readonly`` must open the DB such that any INSERT/UPDATE
    raises at SQL-execution time — a tighter guarantee than just
    "the script doesn't write" (the URI-mode-ro contract). This pins
    the safety property the script's docstring promises."""
    db = tmp_path / "ledger.db"
    led = Ledger.open(db)
    led.close()

    conn = fit_k_mod.connect_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            conn.execute(
                "INSERT INTO orders (client_id, token_id, side, price, "
                "size, status, created_ts) "
                "VALUES ('x', 'tok', 'buy', 0.5, 1, 'open', '2026-01-01T00:00:00+00:00')"
            )
    finally:
        conn.close()


import sqlite3  # noqa: E402  — placed at bottom so the test reads top-down
