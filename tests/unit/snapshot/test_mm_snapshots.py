"""Regression snapshots for the merged ``mm/`` modules.

Covers: ``greeks``, ``quote``, ``pnl``, ``limits``, ``hedge/beta``,
``hedge/calendar``, ``hedge/synth_strip``.

Every test pins a fixed-seed synthetic input, calls the module's public
entry point, and asserts the output matches a committed JSON fixture
within the default 1e-10 float tolerance. Regenerate a fixture via
``SNAPSHOT_UPDATE=1 pytest tests/unit/snapshot/test_mm_snapshots.py``.

If a snapshot diverges from "reasonable expected output" on a future
commit, DO NOT patch the module — report to the planning window for
routing to the owning track.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from blksch.mm.greeks import (
    clip_p, delta_x, gamma_x, logit, s_double_prime, s_prime, sigmoid,
    vega_b_pvar, vega_b_xvar, vega_rho_pair,
)
from blksch.mm.hedge.beta import BetaHedgeParams, compute_beta_hedge
from blksch.mm.hedge.calendar import CalendarHedgeParams, compute_calendar_hedge
from blksch.mm.hedge.synth_strip import (
    SynthStripParams, replicate_xvariance_strip,
)
from blksch.mm.limits import KillSwitchReport, LimitsConfig, LimitsState
from blksch.mm.pnl import AttributionSnapshot, Attributor
from blksch.mm.quote import QuoteParams, compute_quote, half_spread_x, reservation_x
from blksch.schemas import CorrelationEntry, HedgeSide, SurfacePoint

from tests.unit.snapshot._helpers import assert_matches_snapshot

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# greeks.py
# ---------------------------------------------------------------------------


def test_greeks_identity_grid_snapshot():
    """Fixed 9-point p grid. Every identity + vega is snapshotted."""
    ps = [1e-4, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0 - 1e-4]
    payload = {
        "p_grid": ps,
        "clip_p_default": [clip_p(p) for p in ps],
        "sigmoid_inverse_roundtrip": [sigmoid(logit(p)) for p in ps],
        "s_prime": [s_prime(logit(p)) for p in ps],
        "s_double_prime": [s_double_prime(logit(p)) for p in ps],
        "delta_x": [delta_x(p) for p in ps],
        "gamma_x": [gamma_x(p) for p in ps],
        "vega_b_pvar_sigma_0_2": [vega_b_pvar(p, 0.2) for p in ps],
        "vega_b_xvar_sigma_grid": [vega_b_xvar(s) for s in (0.01, 0.05, 0.1, 0.2, 0.5)],
        "vega_rho_pair_grid": [
            vega_rho_pair(0.4, 0.6, 0.15, 0.12),
            vega_rho_pair(0.1, 0.9, 0.05, 0.3),
            vega_rho_pair(0.5, 0.5, 0.2, 0.2),
        ],
    }
    assert_matches_snapshot(payload, "mm/greeks_identity_grid.json")


# ---------------------------------------------------------------------------
# quote.py
# ---------------------------------------------------------------------------


def _quote_params() -> QuoteParams:
    return QuoteParams(
        gamma=0.1, k=1.5, eps=1.0e-5, delta_p_floor=0.01,
        q_max_base=50.0, q_max_shrink=1.0, default_size=10.0,
    )


def test_quote_compute_grid_snapshot():
    """6-point (x_t, q) grid at fixed σ_b, T — pins both spread primitive
    and the compute_quote end-to-end output."""
    params = _quote_params()
    sigma_b = 0.05
    T = 3600.0
    grid = [
        (-2.0, -15.0),
        (-0.5,   0.0),
        ( 0.0,   0.0),
        ( 0.5,   7.0),
        ( 2.0,  25.0),
        ( 3.5,  40.0),
    ]
    quotes = []
    for x_t, q in grid:
        qt = compute_quote(
            token_id="snap",
            x_t=x_t, sigma_b=sigma_b, time_to_horizon_sec=T,
            inventory_q=q, params=params,
            ts=T0,
        )
        quotes.append({
            "x_t": x_t, "q": q,
            "p_bid": qt.p_bid, "p_ask": qt.p_ask,
            "x_bid": qt.x_bid, "x_ask": qt.x_ask,
            "half_spread_x": qt.half_spread_x,
            "reservation_x": qt.reservation_x,
            "size_bid": qt.size_bid, "size_ask": qt.size_ask,
        })

    primitives = {
        "reservation_x": [
            reservation_x(0.0, q, sigma_b, T, params.gamma) for q in (-10, 0, 10)
        ],
        "half_spread_x_grid": [
            half_spread_x(s, T, params.gamma, params.k)
            for s in (0.02, 0.05, 0.1, 0.2)
        ],
    }
    assert_matches_snapshot(
        {"quotes": quotes, "primitives": primitives},
        "mm/quote_compute.json",
    )


# ---------------------------------------------------------------------------
# pnl.py
# ---------------------------------------------------------------------------


def test_pnl_attributor_walk_snapshot():
    """Deterministic 6-tick walk with one post-peak mean reversion so the
    attribution hits directional, curvature, and jump buckets."""
    attributor = Attributor(jump_zscore_threshold=4.0)
    ticks = [
        # (p, sigma_b, qty, elapsed_sec)
        (0.50, 0.05,   0.0, 0),
        (0.52, 0.05,  10.0, 1),
        (0.55, 0.06,  10.0, 2),
        (0.60, 0.06,  10.0, 3),   # directional gain on long
        (0.45, 0.08,  10.0, 4),   # jump (z > 4 on dp²)
        (0.47, 0.07,   5.0, 5),   # partial unwind
    ]
    steps = []
    from datetime import timedelta
    for p, sig, qty, dt in ticks:
        snap = AttributionSnapshot(
            token_id="T", p=p, sigma_b=sig, qty=qty,
            ts=T0 + timedelta(seconds=dt),
        )
        out = attributor.step(snap)
        steps.append({"input": {"p": p, "sigma_b": sig, "qty": qty, "t_offset": dt},
                      "step": out})
    payload = {
        "steps": steps,
        "cumulative": attributor.cumulative,
    }
    assert_matches_snapshot(payload, "mm/pnl_attributor_walk.json")


# ---------------------------------------------------------------------------
# limits.py
# ---------------------------------------------------------------------------


def test_limits_scenarios_snapshot():
    """Three canonical kill-switch scenarios: healthy, feed gap, drawdown."""
    from datetime import timedelta
    cfg = LimitsConfig(
        feed_gap_sec=3.0,
        volatility_spike_z=5.0,
        volatility_window=60,
        repeated_pickoff_window_sec=60.0,
        repeated_pickoff_count=3,
        max_drawdown_usd=100.0,
        inventory_cap_base=50.0,
        sprime_floor=1e-4,
        max_gamma_exposure=50.0,
        swing_zone_half_width=0.15,
    )

    def _healthy() -> KillSwitchReport:
        s = LimitsState(cfg=cfg)
        s.note_tick(T0)
        for sig in [0.05] * 15:
            s.note_sigma(sig)
        return s.evaluate(
            now=T0 + timedelta(seconds=1),
            current_sigma=0.05,
            cumulative_pnl_usd=0.0,
            current_qty=5.0, current_p=0.5,
        )

    def _feed_gap() -> KillSwitchReport:
        s = LimitsState(cfg=cfg)
        s.note_tick(T0)
        return s.evaluate(
            now=T0 + timedelta(seconds=10),
            current_sigma=0.05,
            cumulative_pnl_usd=0.0,
        )

    def _drawdown() -> KillSwitchReport:
        s = LimitsState(cfg=cfg)
        s.note_tick(T0)
        return s.evaluate(
            now=T0 + timedelta(seconds=1),
            current_sigma=0.05,
            cumulative_pnl_usd=-500.0,
        )

    payload = {
        "healthy": _healthy(),
        "feed_gap": _feed_gap(),
        "drawdown": _drawdown(),
    }
    assert_matches_snapshot(payload, "mm/limits_scenarios.json")


# ---------------------------------------------------------------------------
# hedge/beta.py
# ---------------------------------------------------------------------------


def _sp(token: str, m: float, sigma_b: float) -> SurfacePoint:
    return SurfacePoint(
        token_id=token, tau=3600.0, m=m, sigma_b=sigma_b,
        **{"lambda": 0.01}, s2_j=0.005,
        ts=T0,
    )


def _corr(rho: float, co_lam: float = 0.0, co_m2: float = 0.0) -> CorrelationEntry:
    return CorrelationEntry(
        token_id_i="TGT", token_id_j="HDG",
        rho=rho, co_jump_lambda=co_lam, co_jump_m2=co_m2,
        ts=T0,
    )


def test_hedge_beta_grid_snapshot():
    """6-point (rho, alpha, co-jump) grid through compute_beta_hedge."""
    target = _sp("TGT", m=0.3, sigma_b=0.15)
    hedge = _sp("HDG", m=-0.1, sigma_b=0.12)
    params = BetaHedgeParams(alpha=0.8, s_prime_floor=1e-4, max_abs_beta=5.0)

    grid = [
        ( 0.6, 0.8, False),
        (-0.6, 0.8, False),
        ( 0.0, 0.8, False),
        ( 0.95, 0.5, False),
        ( 0.6, 1.0, False),
        ( 0.6, 0.8, True),
    ]
    results = []
    for rho, alpha, co_on in grid:
        p = BetaHedgeParams(alpha=alpha, s_prime_floor=1e-4,
                            max_abs_beta=5.0, apply_co_jump=co_on)
        corr = _corr(rho, co_lam=0.02 if co_on else 0.0,
                     co_m2=0.001 if co_on else 0.0)
        out = compute_beta_hedge(
            target, hedge, corr, alpha,
            params=p, target_notional_usd=100.0, ts=T0,
        )
        results.append({
            "input": {"rho": rho, "alpha": alpha, "co_jump": co_on},
            "output": out,
        })
    assert_matches_snapshot(results, "mm/hedge_beta_grid.json")


# ---------------------------------------------------------------------------
# hedge/calendar.py
# ---------------------------------------------------------------------------


def test_hedge_calendar_grid_snapshot():
    surface = _sp("T", m=0.0, sigma_b=0.12)
    params = CalendarHedgeParams(sigma_b_floor=1e-4, max_abs_notional_usd=10_000.0)
    nu_values = [-150.0, -10.0, 0.0, 10.0, 150.0, 1e6]
    results = []
    for nu in nu_values:
        out = compute_calendar_hedge(surface, nu, params=params, ts=T0)
        results.append({"nu_b": nu, "output": out})
    assert_matches_snapshot(results, "mm/hedge_calendar_grid.json")


# ---------------------------------------------------------------------------
# hedge/synth_strip.py
# ---------------------------------------------------------------------------


def test_hedge_synth_strip_replication_snapshot():
    """Fixed surface neighbourhood + target; pins the basket weights."""
    rng = np.random.default_rng(42)
    surfaces = [
        SurfacePoint(
            token_id=f"T{i}",
            tau=float(60.0 * (1 + i)),
            m=float(rng.uniform(-1.0, 1.0)),
            sigma_b=float(0.1 + 0.05 * rng.uniform()),
            **{"lambda": 0.01}, s2_j=0.004,
            ts=T0,
        )
        for i in range(8)
    ]
    params = SynthStripParams(
        bandwidth_m=0.5, bandwidth_log_tau=0.5, max_basket_size=4,
        weight_floor_ratio=1e-3,
    )
    legs_pos = replicate_xvariance_strip(
        surfaces, target_tau=180.0, target_m=0.0,
        target_variance_notional=25.0, params=params,
    )
    legs_neg = replicate_xvariance_strip(
        surfaces, target_tau=180.0, target_m=0.0,
        target_variance_notional=-25.0, params=params,
    )
    legs_zero = replicate_xvariance_strip(
        surfaces, target_tau=180.0, target_m=0.0,
        target_variance_notional=0.0, params=params,
    )
    payload = {
        "pos_target": [dict(token_id=l.token_id, weight=l.weight, m=l.m, tau=l.tau)
                       for l in legs_pos],
        "neg_target": [dict(token_id=l.token_id, weight=l.weight, m=l.m, tau=l.tau)
                       for l in legs_neg],
        "zero_target": legs_zero,  # empty list
    }
    assert_matches_snapshot(payload, "mm/hedge_synth_strip.json")
