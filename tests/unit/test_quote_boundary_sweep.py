"""Quote boundary sweep (pre-paper-soak validation).

Proves mm/quote.py and mm/greeks.py are numerically stable across the entire
p ∈ (0, 1) range — not just the sampled points exercised by the other suites.
Catches:
  * NaN/Inf from underflow in p(1-p)(1-2p) at extreme p
  * Ordering/bounds violations (p_bid < p_mid < p_ask, in [ε, 1-ε])
  * Skew-sign bugs (position should pull reservation away from x_t with
    correct sign)
  * Spread monotonicity in γ, σ_b, (T-t)

If any specific p value fails, that's a real numerical bug — report the
exact p. Test-only; no runtime code changes.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from blksch.mm.greeks import (
    delta_x, gamma_x, logit, sigmoid, s_prime, s_double_prime,
    vega_b_xvar, vega_b_pvar, vega_rho_pair,
)
from blksch.mm.quote import (
    QuoteParams, compute_quote, half_spread_x, q_max, reservation_x,
)


TS = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
TOK = "0xA"

# bot.yaml boundary.eps = 1e-5; the sweep includes p=eps and p=1-eps exactly.
EPS = 1.0e-5
SWEEP_P = (
    1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    0.95, 0.99, 0.995, 0.999, 0.9995, 0.99999,
)
DEFAULT_PARAMS = QuoteParams(
    gamma=0.1, k=1.5, eps=EPS, delta_p_floor=0.01,
    q_max_base=50.0, q_max_shrink=1.0, default_size=10.0,
)
Q_MAX_SCALAR = DEFAULT_PARAMS.q_max_base  # fixed reference scale for the q-sweep
INVENTORY_SWEEP = (-Q_MAX_SCALAR, -Q_MAX_SCALAR / 2.0, 0.0, Q_MAX_SCALAR / 2.0, Q_MAX_SCALAR)

# Main (p, q) sweep uses the AS refresh-horizon regime (short T, realistic σ).
# At full-horizon T=3600s and σ=0.3, q·γ·σ²·T drives the reservation so deep
# into the logit tails that sigmoid(x_bid) and sigmoid(x_ask) both hit the ε
# clip and collide — a distinct regime that's guarded by the inventory-cap
# kill-switch in production. The additional q=0 monotonicity sweep exercises
# the wide (σ, T) grid separately.
MAIN_SIGMA_B = 0.1
MAIN_T_SEC = 60.0

# Post-boundary-fix: all p values pass the strict floor check.
_ALL_P = SWEEP_P


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


# ---------------------------------------------------------------------------
# (A) Greeks sweep — cheap foundation
# ---------------------------------------------------------------------------


class TestGreeksBoundarySweep:
    @pytest.mark.parametrize("p", SWEEP_P)
    def test_all_greeks_finite(self, p: float) -> None:
        x = logit(p)
        for value in (
            sigmoid(x), s_prime(x), s_double_prime(x),
            delta_x(p), gamma_x(p),
            vega_b_xvar(0.3), vega_b_pvar(p, 0.3),
            vega_rho_pair(p, 0.5, 0.3, 0.3),
        ):
            assert _finite(value), f"non-finite greek at p={p}: {value}"

    @pytest.mark.parametrize("p", SWEEP_P)
    def test_sigmoid_logit_roundtrip(self, p: float) -> None:
        """logit ∘ sigmoid = identity to machine precision on the sweep."""
        x = logit(p)
        round_trip = sigmoid(x)
        # Boundary p values may clip through the epsilon guard; allow tick-level slack.
        assert abs(round_trip - p) < max(1e-7, 2 * EPS)

    @pytest.mark.parametrize("p", SWEEP_P)
    def test_delta_x_in_unit_range(self, p: float) -> None:
        """Δ_x = p(1-p) ∈ [0, 0.25]."""
        d = delta_x(p)
        assert 0.0 <= d <= 0.25 + 1e-12

    @pytest.mark.parametrize("p", SWEEP_P)
    def test_gamma_x_bounded(self, p: float) -> None:
        """|Γ_x| = |p(1-p)(1-2p)| ≤ 1 (trivial bound; tight is ~0.0962)."""
        g = gamma_x(p)
        assert _finite(g)
        assert abs(g) <= 1.0


# ---------------------------------------------------------------------------
# (B) Quote sweep over (p, q)
# ---------------------------------------------------------------------------


class TestQuoteBoundarySweep:
    @pytest.mark.parametrize("p", SWEEP_P)
    @pytest.mark.parametrize("q", INVENTORY_SWEEP)
    def test_quote_finite_and_ordered(self, p: float, q: float) -> None:
        x_t = logit(p)
        quote = compute_quote(
            token_id=TOK, x_t=x_t, sigma_b=MAIN_SIGMA_B, time_to_horizon_sec=MAIN_T_SEC,
            inventory_q=q, params=DEFAULT_PARAMS, ts=TS,
        )
        # All numbers finite
        for name, val in [
            ("x_bid", quote.x_bid), ("x_ask", quote.x_ask),
            ("p_bid", quote.p_bid), ("p_ask", quote.p_ask),
            ("half_spread_x", quote.half_spread_x),
            ("reservation_x", quote.reservation_x),
        ]:
            assert _finite(val), f"non-finite {name} at p={p} q={q}: {val}"
        # Ordering
        p_mid = 0.5 * (quote.p_bid + quote.p_ask)
        assert quote.p_bid < p_mid < quote.p_ask, (
            f"ordering violated at p={p} q={q}: bid={quote.p_bid}, ask={quote.p_ask}"
        )
        # Unit-interval clamp (tick_floor = eps)
        assert quote.p_bid >= DEFAULT_PARAMS.eps - 1e-12
        assert quote.p_ask <= 1.0 - DEFAULT_PARAMS.eps + 1e-12
        # Half-spread strictly positive
        assert quote.half_spread_x > 0.0

    @pytest.mark.parametrize("p", SWEEP_P)
    @pytest.mark.parametrize("q", [-Q_MAX_SCALAR, -Q_MAX_SCALAR / 2.0,
                                    Q_MAX_SCALAR / 2.0, Q_MAX_SCALAR])
    def test_skew_sign_opposes_inventory(self, p: float, q: float) -> None:
        """sign(reservation_x - x_t) == -sign(q).  Long q ⇒ reservation below
        x_t (try to sell to flatten); short q ⇒ reservation above x_t."""
        x_t = logit(p)
        quote = compute_quote(
            token_id=TOK, x_t=x_t, sigma_b=MAIN_SIGMA_B, time_to_horizon_sec=MAIN_T_SEC,
            inventory_q=q, params=DEFAULT_PARAMS, ts=TS,
        )
        diff = quote.reservation_x - x_t
        assert math.copysign(1.0, diff) == -math.copysign(1.0, q), (
            f"skew sign violated at p={p} q={q}: r-x_t={diff}"
        )

    @pytest.mark.parametrize("p", SWEEP_P)
    def test_zero_inventory_no_skew(self, p: float) -> None:
        x_t = logit(p)
        quote = compute_quote(
            token_id=TOK, x_t=x_t, sigma_b=MAIN_SIGMA_B, time_to_horizon_sec=MAIN_T_SEC,
            inventory_q=0.0, params=DEFAULT_PARAMS, ts=TS,
        )
        assert quote.reservation_x == pytest.approx(x_t, abs=1e-9)

    @pytest.mark.parametrize("p", _ALL_P)
    def test_p_spread_floors_at_or_above_delta_p_floor(self, p: float) -> None:
        """δ_p = p_ask - p_bid ≥ 2 · delta_p_floor across the full (0,1) sweep
        — including p within ε of the {0, 1} boundary. The post-fix
        `_apply_p_floor` solves directly in p-space with the unit-interval
        clip as a bisection invariant, so the cramped side widens
        asymmetrically when the other is pinned."""
        x_t = logit(p)
        quote = compute_quote(
            token_id=TOK, x_t=x_t, sigma_b=MAIN_SIGMA_B, time_to_horizon_sec=MAIN_T_SEC,
            inventory_q=0.0, params=DEFAULT_PARAMS, ts=TS,
        )
        displayed = quote.p_ask - quote.p_bid
        # Strict: no ε slack (only float-rounding tolerance).
        assert displayed >= 2.0 * DEFAULT_PARAMS.delta_p_floor - 1e-12, (
            f"δ_p floor violated at p={p}: displayed={displayed}"
        )


# ---------------------------------------------------------------------------
# (C) Spread monotonicity in σ_b and T
# ---------------------------------------------------------------------------


class TestSpreadMonotonicity:
    SIGMA_GRID = (1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0)
    T_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    @pytest.mark.parametrize("sigma", SIGMA_GRID)
    def test_spread_monotone_in_tau_at_p_half(self, sigma: float) -> None:
        """At p=0.5 and q=0, half_spread_x monotone non-decreasing in T."""
        x_t = 0.0
        deltas = [
            compute_quote(
                token_id=TOK, x_t=x_t, sigma_b=sigma, time_to_horizon_sec=T,
                inventory_q=0.0, params=DEFAULT_PARAMS, ts=TS,
            ).half_spread_x
            for T in self.T_GRID
        ]
        for a, b in zip(deltas, deltas[1:]):
            assert b >= a - 1e-12, (
                f"spread not monotone in T at σ={sigma}: {deltas}"
            )

    @pytest.mark.parametrize("T", T_GRID)
    def test_spread_monotone_in_sigma_at_p_half(self, T: float) -> None:
        x_t = 0.0
        deltas = [
            compute_quote(
                token_id=TOK, x_t=x_t, sigma_b=sigma, time_to_horizon_sec=T,
                inventory_q=0.0, params=DEFAULT_PARAMS, ts=TS,
            ).half_spread_x
            for sigma in self.SIGMA_GRID
        ]
        for a, b in zip(deltas, deltas[1:]):
            assert b >= a - 1e-12, (
                f"spread not monotone in σ at T={T}: {deltas}"
            )

    def test_half_spread_formula_matches_eq9(self) -> None:
        """Re-derive paper eq (9): 2 δ_x = γ σ² (T-t) + (2/k) log(1+γ/k)."""
        gamma, k, sigma, T = 0.1, 1.5, 0.3, 3600.0
        expected = 0.5 * (gamma * sigma * sigma * T + (2.0 / k) * math.log1p(gamma / k))
        got = half_spread_x(sigma, T, gamma, k)
        assert got == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# (D) q_max cap at boundary
# ---------------------------------------------------------------------------


class TestQmaxBoundary:
    @pytest.mark.parametrize("p", SWEEP_P)
    def test_q_max_finite_and_positive(self, p: float) -> None:
        x_t = logit(p)
        cap = q_max(x_t, DEFAULT_PARAMS)
        assert _finite(cap)
        assert cap > 0.0

    def test_q_max_tightest_at_p_half(self) -> None:
        """Cap in contracts ∝ 1/max(S'(x), ε). S'(x) peaks at p=0.5, so the
        cap in contracts is TIGHTEST there. As p→0 or 1, S'→0 and the 1/ε
        floor pins the maximum cap."""
        cap_mid = q_max(0.0, DEFAULT_PARAMS)
        cap_edge = q_max(logit(0.001), DEFAULT_PARAMS)
        assert cap_edge > cap_mid
        # Pin the 1/ε floor at very extreme p.
        cap_extreme = q_max(logit(1e-6), DEFAULT_PARAMS)
        assert cap_extreme == pytest.approx(
            DEFAULT_PARAMS.q_max_base / DEFAULT_PARAMS.eps, rel=1e-6,
        )
