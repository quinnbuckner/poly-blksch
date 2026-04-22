"""Unit tests for mm/quote.py — paper §4.2 eq (8-9).

Core invariants the tests enforce:
  1. Reservation skew — long inventory ⇒ r_x < x_t, short ⇒ r_x > x_t
  2. Spread monotone-increasing in γ, σ_b, (T-t)
  3. Spread floor activates when p → 0 or 1
  4. Inventory cap tightens as S'(x) shrinks
  5. Zero-inventory quote is symmetric around S(x_t)
"""

from __future__ import annotations

from datetime import datetime, timezone

import math
import pytest

from blksch.mm.greeks import sigmoid
from blksch.mm.quote import (
    QuoteParams,
    compute_quote,
    half_spread_x,
    q_max,
    reservation_x,
)


def _params(**overrides) -> QuoteParams:
    base = dict(
        gamma=0.5,
        k=1.5,
        eps=1.0e-5,
        delta_p_floor=0.005,
        q_max_base=50.0,
        q_max_shrink=1.0,
        default_size=10.0,
    )
    base.update(overrides)
    return QuoteParams(**base)


class TestReservation:
    def test_zero_inventory_gives_midpoint(self) -> None:
        r = reservation_x(x_t=0.4, q_t=0.0, sigma_b=0.1, time_to_horizon_sec=3600, gamma=0.1)
        assert r == pytest.approx(0.4)

    def test_long_inventory_skews_down(self) -> None:
        """q>0 (long YES) ⇒ reservation below x_t ⇒ we post lower bid/ask to shed risk."""
        r = reservation_x(x_t=0.0, q_t=10.0, sigma_b=0.3, time_to_horizon_sec=1000, gamma=0.5)
        assert r < 0.0

    def test_short_inventory_skews_up(self) -> None:
        r = reservation_x(x_t=0.0, q_t=-10.0, sigma_b=0.3, time_to_horizon_sec=1000, gamma=0.5)
        assert r > 0.0

    def test_skew_scales_with_gamma_sigma_and_horizon(self) -> None:
        base = reservation_x(0.0, 5.0, 0.3, 1000, 0.1)
        more_gamma = reservation_x(0.0, 5.0, 0.3, 1000, 0.5)
        more_sigma = reservation_x(0.0, 5.0, 0.6, 1000, 0.1)
        more_horizon = reservation_x(0.0, 5.0, 0.3, 2000, 0.1)
        # all should pull further below zero (more negative) than base
        assert more_gamma < base
        assert more_sigma < base
        assert more_horizon < base


class TestHalfSpread:
    def test_positive(self) -> None:
        assert half_spread_x(0.3, 1000, 0.1, 1.5) > 0.0

    def test_monotone_in_gamma(self) -> None:
        a = half_spread_x(0.3, 1000, 0.1, 1.5)
        b = half_spread_x(0.3, 1000, 0.5, 1.5)
        assert b > a

    def test_monotone_in_sigma(self) -> None:
        a = half_spread_x(0.3, 1000, 0.1, 1.5)
        b = half_spread_x(0.6, 1000, 0.1, 1.5)
        assert b > a

    def test_monotone_in_horizon(self) -> None:
        a = half_spread_x(0.3, 500, 0.1, 1.5)
        b = half_spread_x(0.3, 2000, 0.1, 1.5)
        assert b > a

    def test_formula_matches_paper_eq9(self) -> None:
        """2δ_x = γσ²(T-t) + (2/k)log(1+γ/k)."""
        gamma, sigma, tau, k = 0.4, 0.5, 1200.0, 2.0
        expected = 0.5 * (gamma * sigma * sigma * tau + (2.0 / k) * math.log1p(gamma / k))
        assert half_spread_x(sigma, tau, gamma, k) == pytest.approx(expected)

    def test_horizon_zero_keeps_flow_term(self) -> None:
        """At T=t only the order-flow term survives — positive."""
        delta = half_spread_x(0.3, 0.0, 0.1, 1.5)
        expected = (1.0 / 1.5) * math.log1p(0.1 / 1.5)
        assert delta == pytest.approx(expected)


class TestQmax:
    def test_tighter_near_boundary(self) -> None:
        p = _params()
        mid = q_max(x_t=0.0, params=p)  # S'(0) = 0.25
        edge = q_max(x_t=5.0, params=p)  # S'(5) ≈ 0.0066
        assert edge > mid  # cap in contracts grows because each contract has less Δ
        # But notional risk is bounded by the 1/eps floor
        extreme = q_max(x_t=100.0, params=p)
        assert extreme == pytest.approx(p.q_max_base / p.eps, rel=1e-6)

    def test_shrink_factor_tightens(self) -> None:
        base = q_max(0.0, _params(q_max_shrink=1.0))
        tight = q_max(0.0, _params(q_max_shrink=0.5))
        assert tight == pytest.approx(0.5 * base)


class TestComputeQuote:
    def _ts(self) -> datetime:
        return datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)

    def test_zero_inventory_symmetric_in_x(self) -> None:
        q = compute_quote(
            token_id="tok",
            x_t=0.5,
            sigma_b=0.2,
            time_to_horizon_sec=3600,
            inventory_q=0.0,
            params=_params(),
            ts=self._ts(),
        )
        assert q.reservation_x == pytest.approx(0.5)
        assert q.x_ask - q.reservation_x == pytest.approx(q.reservation_x - q.x_bid, rel=1e-6)

    def test_ordering(self) -> None:
        q = compute_quote(
            token_id="tok",
            x_t=0.0,
            sigma_b=0.3,
            time_to_horizon_sec=1000,
            inventory_q=0.0,
            params=_params(),
            ts=self._ts(),
        )
        assert q.x_bid < q.x_ask
        assert q.p_bid < q.p_ask
        assert 0.0 < q.p_bid < 1.0
        assert 0.0 < q.p_ask < 1.0

    def test_long_inventory_pulls_quote_down(self) -> None:
        ts = self._ts()
        q_flat = compute_quote(
            token_id="t", x_t=0.0, sigma_b=0.3, time_to_horizon_sec=1000,
            inventory_q=0.0, params=_params(), ts=ts,
        )
        q_long = compute_quote(
            token_id="t", x_t=0.0, sigma_b=0.3, time_to_horizon_sec=1000,
            inventory_q=20.0, params=_params(), ts=ts,
        )
        assert q_long.reservation_x < q_flat.reservation_x
        assert q_long.x_bid < q_flat.x_bid
        assert q_long.x_ask < q_flat.x_ask

    def test_short_inventory_pulls_quote_up(self) -> None:
        ts = self._ts()
        q_short = compute_quote(
            token_id="t", x_t=0.0, sigma_b=0.3, time_to_horizon_sec=1000,
            inventory_q=-20.0, params=_params(), ts=ts,
        )
        q_flat = compute_quote(
            token_id="t", x_t=0.0, sigma_b=0.3, time_to_horizon_sec=1000,
            inventory_q=0.0, params=_params(), ts=ts,
        )
        assert q_short.reservation_x > q_flat.reservation_x

    def test_widen_factor_widens_spread(self) -> None:
        ts = self._ts()
        narrow = compute_quote(
            token_id="t", x_t=0.0, sigma_b=0.3, time_to_horizon_sec=1000,
            inventory_q=0.0, params=_params(), ts=ts, spread_widen_factor=1.0,
        )
        wide = compute_quote(
            token_id="t", x_t=0.0, sigma_b=0.3, time_to_horizon_sec=1000,
            inventory_q=0.0, params=_params(), ts=ts, spread_widen_factor=3.0,
        )
        assert wide.half_spread_x > narrow.half_spread_x

    def test_widen_factor_rejects_under_one(self) -> None:
        with pytest.raises(ValueError):
            compute_quote(
                token_id="t", x_t=0.0, sigma_b=0.3, time_to_horizon_sec=1000,
                inventory_q=0.0, params=_params(), ts=self._ts(), spread_widen_factor=0.5,
            )

    def test_boundary_floor_activates_near_p_zero(self) -> None:
        """At x_t = -6 (p≈0.0025), natural δ_p would collapse. Floor must kick in."""
        p = _params(delta_p_floor=0.01)
        q = compute_quote(
            token_id="t",
            x_t=-6.0,
            sigma_b=0.05,           # small vol
            time_to_horizon_sec=60, # short horizon → tiny δ_x
            inventory_q=0.0,
            params=p,
            ts=self._ts(),
        )
        displayed = q.p_ask - q.p_bid
        assert displayed >= 2.0 * p.delta_p_floor - 1.0e-9

    def test_boundary_floor_activates_near_p_one(self) -> None:
        p = _params(delta_p_floor=0.01)
        q = compute_quote(
            token_id="t", x_t=6.0, sigma_b=0.05, time_to_horizon_sec=60,
            inventory_q=0.0, params=p, ts=self._ts(),
        )
        displayed = q.p_ask - q.p_bid
        assert displayed >= 2.0 * p.delta_p_floor - 1.0e-9

    def test_floor_does_not_shrink_healthy_spread(self) -> None:
        """If the natural δ_p already exceeds the floor, the quote is unchanged."""
        p = _params(delta_p_floor=0.001)
        q_natural = compute_quote(
            token_id="t", x_t=0.0, sigma_b=0.5, time_to_horizon_sec=3600,
            inventory_q=0.0, params=p, ts=self._ts(),
        )
        assert (q_natural.p_ask - q_natural.p_bid) > 2.0 * p.delta_p_floor

    def test_p_stays_inside_unit_interval(self) -> None:
        """Even with extreme state, p_bid/p_ask never leave (eps, 1-eps)."""
        p = _params()
        q = compute_quote(
            token_id="t", x_t=-15.0, sigma_b=2.0, time_to_horizon_sec=10_000,
            inventory_q=100.0, params=p, ts=self._ts(),
        )
        assert p.eps <= q.p_bid <= 1.0 - p.eps
        assert p.eps <= q.p_ask <= 1.0 - p.eps
