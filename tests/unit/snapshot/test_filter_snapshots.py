"""Regression snapshots for ``core/filter/`` modules.

Covers: ``canonical_mid``, ``microstruct``, ``kalman``, ``ewma_var``.

Each test pins a fixed-seed synthetic input and asserts the module's
observable output matches a committed JSON fixture within 1e-10. If a
snapshot drifts on a future commit, DO NOT patch the module — report.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Sequence

import numpy as np
import pytest

from blksch.core.filter.canonical_mid import CanonicalMidFilter
from blksch.core.filter.ewma_var import EwmaVar
from blksch.core.filter.kalman import KalmanFilter
from blksch.core.filter.microstruct import (
    MicrostructConfig,
    MicrostructFeatures,
    MicrostructModel,
    extract_features,
)
from blksch.schemas import BookSnap, PriceLevel, TradeSide, TradeTick

from tests.unit.snapshot._helpers import assert_matches_snapshot

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
TOKEN = "0xSNAP_FLT"


def _book(bid: float, ask: float, bsz: float, asz: float, ts) -> BookSnap:
    return BookSnap(
        token_id=TOKEN,
        bids=[PriceLevel(price=bid, size=bsz)],
        asks=[PriceLevel(price=ask, size=asz)],
        ts=ts,
    )


# ---------------------------------------------------------------------------
# canonical_mid
# ---------------------------------------------------------------------------


def test_canonical_mid_fixed_stream_snapshot():
    """Deterministic book + trade stream. Pin every emitted CanonicalMid."""
    f = CanonicalMidFilter(token_id=TOKEN, grid_hz=1.0)
    # 12 ticks, 1.0 s cadence with a few off-grid gaps so the forward-fill
    # logic gets exercised.
    seq = [
        (0.48, 0.52, 100.0, 100.0, 0.2),
        (0.49, 0.51, 120.0,  90.0, 0.7),
        (0.49, 0.51, 120.0,  90.0, 1.3),  # first grid boundary emit at t=1
        (0.50, 0.52, 110.0,  80.0, 2.4),
        (0.47, 0.53,  50.0,  50.0, 3.8),  # outlier-ish
        (0.50, 0.52, 100.0, 100.0, 4.5),
        (0.51, 0.53, 150.0, 120.0, 5.5),
        (0.52, 0.54, 150.0, 120.0, 7.9),  # gap between 6s and 7s → forward fill at t=6,7
        (0.53, 0.55, 160.0, 130.0, 8.4),
        (0.54, 0.56, 160.0, 130.0, 9.5),
        (0.55, 0.57, 170.0, 140.0,10.3),
        (0.56, 0.58, 170.0, 140.0,11.8),
    ]
    emits = []
    for bid, ask, bsz, asz, off in seq:
        ts = T0 + timedelta(milliseconds=int(off * 1000))
        emits.extend(f.update(_book(bid, ask, bsz, asz, ts)))
    payload = [
        {
            "ts_offset_sec": (cm.ts - T0).total_seconds(),
            "p_tilde": cm.p_tilde,
            "y": cm.y,
            "forward_filled": cm.forward_filled,
            "rejected_outlier": cm.rejected_outlier,
            "trades_in_window": cm.trades_in_window,
            "source": cm.source,
        }
        for cm in emits
    ]
    assert_matches_snapshot(payload, "filter/canonical_mid_stream.json")


# ---------------------------------------------------------------------------
# microstruct
# ---------------------------------------------------------------------------


def test_microstruct_fit_and_serve_snapshot():
    """Deterministic feature rows + squared-innovation targets → fit →
    serve on a held-out grid. Pins both the fitted coefficients and the
    per-feature variance outputs."""
    rng = np.random.default_rng(0)
    n_fit = 256
    spreads = rng.uniform(0.005, 0.05, size=n_fit)
    inv_depths = rng.uniform(1e-4, 5e-3, size=n_fit)
    rates = rng.uniform(0.0, 2.0, size=n_fit)
    imbalances = rng.uniform(0.0, 0.8, size=n_fit)

    # Synthetic target follows eq (10) with a tiny observation noise.
    noise = rng.normal(0.0, 1e-7, size=n_fit)
    targets = (
        1.0e-6
        + 0.5 * spreads * spreads
        + 2.0 * inv_depths
        + 1e-5 * rates
        + 1e-6 * imbalances
        + noise
    ).clip(min=1e-8)

    features = [
        MicrostructFeatures(
            half_spread=float(s) / 2.0,
            inv_depth=float(d),
            abs_trade_rate=float(r),
            abs_imbalance=float(im),
        )
        for s, d, r, im in zip(spreads, inv_depths, rates, imbalances)
    ]
    model = MicrostructModel.fit_from_features(
        features, targets.tolist(),
        config=MicrostructConfig(sigma_floor=1e-8, ridge=1e-10),
    )

    grid = [
        (0.005, 1e-3, 0.0, 0.0),
        (0.020, 2e-3, 1.0, 0.3),
        (0.050, 5e-4, 2.0, 0.8),
    ]
    served = []
    for s, d, r, im in grid:
        feat = MicrostructFeatures(
            half_spread=s / 2.0, inv_depth=d,
            abs_trade_rate=r, abs_imbalance=im,
        )
        served.append({
            "in": {"spread": s, "inv_depth": d, "trade_rate": r, "imbalance": im},
            "sigma_eta2": model.variance_from_features(feat),
            "sigma_eta2_ff": model.variance_from_features(feat, forward_filled=True),
        })

    payload = {
        "coefficients": {
            "a0": model.a0, "a1": model.a1, "a2": model.a2,
            "a3": model.a3, "a4": model.a4,
        },
        "served_grid": served,
    }
    assert_matches_snapshot(payload, "filter/microstruct_fit_serve.json")


# ---------------------------------------------------------------------------
# kalman
# ---------------------------------------------------------------------------


class _ConstantOracle:
    def __init__(self, value: float) -> None:
        self.value = value

    def variance(
        self,
        book: BookSnap,
        trades: Sequence[TradeTick] = (),
        *,
        forward_filled: bool = False,
    ) -> float:
        return self.value


def test_kalman_heteroskedastic_walk_snapshot():
    """Fixed 12-step walk through the Kalman filter. Pin the emitted
    LogitState + last_K + posterior_variance at every step."""
    oracle = _ConstantOracle(1e-3)
    kf = KalmanFilter(
        token_id=TOKEN, microstruct=oracle,
        sigma_b=0.08, initial_x=0.0, initial_variance=0.5,
    )
    cmid = CanonicalMidFilter(token_id=TOKEN, grid_hz=1.0)

    steps: list[dict] = []
    path_p = [0.50, 0.51, 0.52, 0.50, 0.49, 0.51, 0.55, 0.58, 0.53, 0.50, 0.47, 0.46]
    for i, p in enumerate(path_p):
        # Build a tight synthetic book so canonical_mid accepts it.
        book = _book(p - 0.005, p + 0.005, 100.0, 100.0,
                     T0 + timedelta(seconds=i))
        for cm in cmid.update(book):
            state = kf.step(cm, book, [])
            steps.append({
                "cm_ts_offset": (cm.ts - T0).total_seconds(),
                "y": cm.y,
                "x_hat": state.x_hat,
                "posterior_variance": kf.posterior_variance,
                "last_K": kf.last_K,
                "last_innovation": kf.last_innovation,
                "last_R_effective": kf.last_R_effective,
            })
    assert_matches_snapshot(steps, "filter/kalman_walk.json")


# ---------------------------------------------------------------------------
# ewma_var
# ---------------------------------------------------------------------------


def test_ewma_var_fixed_increment_sequence_snapshot():
    """Deterministic (Δx, Δt, γ) stream through EwmaVar, pin per-step
    σ̂_b²."""
    ewma = EwmaVar(half_life_sec=90.0, cold_start_factor=1.0)
    rng = np.random.default_rng(7)
    n = 40
    trace: list[dict] = []
    for i in range(n):
        dx = float(rng.normal(0.0, 0.05))
        dt = 1.0
        # Inject a jump at i=20: γ → 0.95, large |dx|.
        if i == 20:
            dx = 0.6
            gamma = 0.95
        else:
            gamma = float(min(1.0, max(0.0, 0.05 * abs(dx / 0.05) ** 2)))
        sigma_sq = ewma.update(dx, dt, jump_posterior=gamma)
        trace.append({
            "step": i, "dx": dx, "dt": dt, "gamma": gamma,
            "sigma_sq": sigma_sq,
        })
    # Final variance() == last update output
    payload = {
        "trace": trace,
        "final_sigma_sq": ewma.variance(),
    }
    assert_matches_snapshot(payload, "filter/ewma_var_trace.json")
