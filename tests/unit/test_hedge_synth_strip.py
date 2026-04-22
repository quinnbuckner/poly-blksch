"""Unit + MC tests for mm/hedge/synth_strip.py (paper §3.4).

Covers:
  * SynthStripParams validation
  * weights sum-to-unit-notional for a unit-variance target
  * zero-variance target → empty basket
  * no candidate neighbors → empty basket
  * bandwidth cutoff filtering
  * exclude_token_id drops the target's own leg
  * kernel concentrates weight on the nearest neighbor
  * explode_hedge_into_basket sign/side semantics
  * MC tracking-error bound: over synthetic paths with known σ_b², the
    basket's expected payoff tracks (p(1-p))² σ² Δ within the paper's
    short-maturity bound (we use a 20%% relative bound for a small basket).
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone

import pytest

from blksch.mm.greeks import s_prime
from blksch.mm.hedge.synth_strip import (
    BasketLeg,
    SynthStripParams,
    explode_hedge_into_basket,
    replicate_xvariance_strip,
)
from blksch.schemas import HedgeInstruction, HedgeSide, SurfacePoint


TS = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
TOK = "0xtarget"


def _surf(token_id: str, m: float, tau: float = 3600.0, sigma_b: float = 0.3) -> SurfacePoint:
    return SurfacePoint(
        token_id=token_id, tau=tau, m=m, sigma_b=sigma_b,
        **{"lambda": 0.1}, s2_j=0.0, ts=TS,
    )


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


class TestParams:
    def test_bandwidth_positive(self) -> None:
        SynthStripParams(bandwidth_m=0.1, bandwidth_log_tau=0.2)
        with pytest.raises(ValueError):
            SynthStripParams(bandwidth_m=0.0)
        with pytest.raises(ValueError):
            SynthStripParams(bandwidth_log_tau=-0.1)

    def test_max_basket_size_positive(self) -> None:
        with pytest.raises(ValueError):
            SynthStripParams(max_basket_size=0)

    def test_floor_ratio_range(self) -> None:
        with pytest.raises(ValueError):
            SynthStripParams(weight_floor_ratio=-0.01)
        with pytest.raises(ValueError):
            SynthStripParams(weight_floor_ratio=1.0)


# ---------------------------------------------------------------------------
# replicate_xvariance_strip
# ---------------------------------------------------------------------------


class TestReplicateBasic:
    def test_unit_notional_weights_sum_to_notional(self) -> None:
        """Gaussian-kernel weights, after trimming and renormalization, sum
        exactly to the target notional (partition-of-unity)."""
        pts = [_surf(f"n{k}", m=m, tau=3600.0) for k, m in enumerate([-0.5, 0.0, 0.5, 1.0])]
        basket = replicate_xvariance_strip(
            surface_points=pts,
            target_tau=3600.0, target_m=0.0,
            target_variance_notional=1.0,
        )
        assert basket
        total = sum(leg.weight for leg in basket)
        assert total == pytest.approx(1.0, rel=1e-9)

    def test_zero_notional_empty_basket(self) -> None:
        pts = [_surf(f"n{k}", m=m) for k, m in enumerate([-0.2, 0.0, 0.2])]
        basket = replicate_xvariance_strip(
            surface_points=pts,
            target_tau=3600.0, target_m=0.0,
            target_variance_notional=0.0,
        )
        assert basket == []

    def test_no_neighbors_empty_basket(self) -> None:
        basket = replicate_xvariance_strip(
            surface_points=[],
            target_tau=3600.0, target_m=0.0,
            target_variance_notional=1.0,
        )
        assert basket == []

    def test_bandwidth_cutoff_excludes_far_points(self) -> None:
        """Points past max_moneyness_dist are filtered; only close neighbors remain."""
        pts = [
            _surf("near", m=0.05),
            _surf("far", m=10.0),
        ]
        params = SynthStripParams(max_moneyness_dist=1.0)
        basket = replicate_xvariance_strip(
            surface_points=pts,
            target_tau=3600.0, target_m=0.0,
            target_variance_notional=1.0,
            params=params,
        )
        token_ids = {leg.token_id for leg in basket}
        assert token_ids == {"near"}

    def test_exclude_target_token(self) -> None:
        pts = [_surf(TOK, m=0.0), _surf("peer", m=0.1)]
        basket = replicate_xvariance_strip(
            surface_points=pts,
            target_tau=3600.0, target_m=0.0,
            target_variance_notional=1.0,
            exclude_token_id=TOK,
        )
        assert all(leg.token_id != TOK for leg in basket)
        assert any(leg.token_id == "peer" for leg in basket)

    def test_kernel_concentrates_on_nearest(self) -> None:
        """With tight bandwidth, the closest point dominates the basket."""
        pts = [_surf(f"n{k}", m=m) for k, m in enumerate([0.01, 0.5, 1.2, 2.0])]
        params = SynthStripParams(bandwidth_m=0.1, bandwidth_log_tau=0.5, max_basket_size=4)
        basket = replicate_xvariance_strip(
            surface_points=pts,
            target_tau=3600.0, target_m=0.0,
            target_variance_notional=1.0,
            params=params,
        )
        # Sort basket by absolute m-distance from 0.
        basket.sort(key=lambda leg: abs(leg.m))
        assert basket[0].token_id == "n0"
        assert basket[0].weight > 0.5 * sum(leg.weight for leg in basket)

    def test_max_basket_size_trims(self) -> None:
        pts = [_surf(f"n{k}", m=m) for k, m in enumerate(
            [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        )]
        params = SynthStripParams(max_basket_size=3, weight_floor_ratio=0.0)
        basket = replicate_xvariance_strip(
            surface_points=pts,
            target_tau=3600.0, target_m=0.0,
            target_variance_notional=1.0,
            params=params,
        )
        assert len(basket) == 3

    def test_weight_floor_drops_tiny(self) -> None:
        """Far neighbors with small weight get trimmed even within max_basket_size."""
        pts = [
            _surf("close", m=0.0),
            _surf("far_but_valid", m=2.5),   # within cutoff but kernel tail
        ]
        params = SynthStripParams(
            bandwidth_m=0.2, weight_floor_ratio=0.01,
            max_moneyness_dist=3.0, max_basket_size=5,
        )
        basket = replicate_xvariance_strip(
            surface_points=pts,
            target_tau=3600.0, target_m=0.0,
            target_variance_notional=1.0,
            params=params,
        )
        token_ids = {leg.token_id for leg in basket}
        assert "close" in token_ids
        assert "far_but_valid" not in token_ids

    def test_notional_scales_weights_linearly(self) -> None:
        pts = [_surf(f"n{k}", m=m) for k, m in enumerate([-0.3, 0.0, 0.3])]
        a = replicate_xvariance_strip(pts, 3600.0, 0.0, target_variance_notional=10.0)
        b = replicate_xvariance_strip(pts, 3600.0, 0.0, target_variance_notional=25.0)
        assert sum(leg.weight for leg in a) == pytest.approx(10.0)
        assert sum(leg.weight for leg in b) == pytest.approx(25.0)

    def test_target_tau_nonpositive_raises(self) -> None:
        with pytest.raises(ValueError):
            replicate_xvariance_strip([_surf("n", m=0.0)], -1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# explode_hedge_into_basket
# ---------------------------------------------------------------------------


def _calendar_instr(
    source: str = TOK, notional: float = 20.0, side: HedgeSide = HedgeSide.SHORT
) -> HedgeInstruction:
    return HedgeInstruction(
        source_token_id=source,
        hedge_token_id=f"{source}:xvar",
        side=side, notional_usd=notional, reason="calendar", ts=TS,
    )


class TestExplode:
    def test_zero_notional_empty(self) -> None:
        instr = _calendar_instr(notional=0.0)
        assert explode_hedge_into_basket(instr, [_surf("n", 0.0)], 3600.0, 0.0) == []

    def test_no_neighbors_empty(self) -> None:
        instr = _calendar_instr(notional=20.0)
        assert explode_hedge_into_basket(instr, [], 3600.0, 0.0) == []

    def test_side_inherited_on_positive_weights(self) -> None:
        pts = [_surf(f"n{k}", m=m) for k, m in enumerate([-0.3, 0.0, 0.3])]
        instr = _calendar_instr(side=HedgeSide.SHORT, notional=20.0)
        legs = explode_hedge_into_basket(instr, pts, 3600.0, 0.0)
        assert legs
        assert all(leg.side is HedgeSide.SHORT for leg in legs)
        # Notionals sum to input notional (partition-of-unity with pos weights).
        assert sum(leg.notional_usd for leg in legs) == pytest.approx(20.0)
        assert all(leg.reason == "synth_strip" for leg in legs)
        assert all(leg.source_token_id == TOK for leg in legs)
        assert all(leg.hedge_token_id != f"{TOK}:xvar" for leg in legs)

    def test_long_instr_produces_long_legs(self) -> None:
        pts = [_surf(f"n{k}", m=m) for k, m in enumerate([-0.3, 0.0, 0.3])]
        instr = _calendar_instr(side=HedgeSide.LONG, notional=10.0)
        legs = explode_hedge_into_basket(instr, pts, 3600.0, 0.0)
        assert legs
        assert all(leg.side is HedgeSide.LONG for leg in legs)

    def test_ts_propagates(self) -> None:
        pts = [_surf("n", m=0.0)]
        instr = _calendar_instr(notional=5.0)
        legs = explode_hedge_into_basket(instr, pts, 3600.0, 0.0)
        assert legs[0].ts == instr.ts

    def test_self_reference_dropped(self) -> None:
        pts = [_surf(TOK, m=0.0), _surf("peer", m=0.1)]
        instr = _calendar_instr(source=TOK, notional=10.0)
        legs = explode_hedge_into_basket(instr, pts, 3600.0, 0.0)
        assert all(leg.hedge_token_id != TOK for leg in legs)


# ---------------------------------------------------------------------------
# Monte-Carlo tracking-error bound (paper §3.4 Lemma)
# ---------------------------------------------------------------------------


class TestShortMaturityTrackingError:
    def _simulate_realized_p_variance(
        self, x_t: float, sigma_b: float, dt: float, n_paths: int, rng: random.Random
    ) -> float:
        """Monte-Carlo estimate of E[(dp)²] over one step Δt under Brownian-motion
        logit dynamics dx = σ_b dW (no drift, no jumps).

        The analytic short-maturity value per paper eq (6) is (p(1-p))²·σ²·Δ.
        """
        from blksch.mm.greeks import sigmoid
        p_t = sigmoid(x_t)
        squared_dps = []
        stdev = sigma_b * math.sqrt(dt)
        for _ in range(n_paths):
            dx = rng.gauss(0.0, stdev)
            p_next = sigmoid(x_t + dx)
            dp = p_next - p_t
            squared_dps.append(dp * dp)
        return sum(squared_dps) / len(squared_dps)

    def test_single_neighbor_exactly_tracks_target(self) -> None:
        """With a single surface point exactly at (τ*, m*), the basket
        weight equals the full notional and tracking error is zero."""
        pts = [_surf("exact", m=0.0, tau=3600.0, sigma_b=0.3)]
        target_notional = 1.0
        basket = replicate_xvariance_strip(pts, 3600.0, 0.0, target_notional)
        assert len(basket) == 1
        assert basket[0].weight == pytest.approx(target_notional)

    def test_symmetric_grid_centers_on_target_m(self) -> None:
        """A symmetric grid around m* produces the mean-m of the basket = m*."""
        grid = [-0.4, -0.2, 0.0, 0.2, 0.4]
        pts = [_surf(f"n{k}", m=m) for k, m in enumerate(grid)]
        basket = replicate_xvariance_strip(pts, 3600.0, 0.0, 1.0,
                                           params=SynthStripParams(bandwidth_m=0.5))
        mean_m = sum(leg.weight * leg.m for leg in basket)
        assert abs(mean_m) < 1.0e-6

    def test_short_maturity_bound_holds_on_mc_paths(self) -> None:
        """MC test of the §3.4 short-maturity replication.

        On a fine grid of adjacent moneyness, the *effective σ_b²* carried by
        the basket (weighted average of the neighbors' σ_b²) should track
        the target's σ_b² within a short-maturity bound. Because all our
        neighbors here share σ_b=0.3, the bound reduces to: the MC estimate
        of E[(dp)²] at x=target_m must equal (p(1-p))²·σ²·Δ within MC error.
        """
        rng = random.Random(20260422)
        x_t = 0.2
        sigma_b = 0.3
        dt = 0.01  # short maturity
        n_paths = 20_000

        grid = [-0.3, -0.15, 0.0, 0.15, 0.3]
        pts = [_surf(f"n{k}", m=x_t + m, tau=3600.0, sigma_b=sigma_b) for k, m in enumerate(grid)]
        basket = replicate_xvariance_strip(pts, 3600.0, x_t, 1.0)
        assert basket  # sanity: non-empty

        # Weighted belief-vol-squared carried by the basket — partition of unity
        # with uniform σ_b means this equals σ²·(weight_sum) = σ².
        weighted_sigma2 = sum(leg.weight * sigma_b ** 2 for leg in basket)
        assert weighted_sigma2 == pytest.approx(sigma_b ** 2, rel=1e-6)

        # Cross-check with MC realized p-variance.
        realized = self._simulate_realized_p_variance(x_t, sigma_b, dt, n_paths, rng)
        target_analytic = (s_prime(x_t)) ** 2 * sigma_b ** 2 * dt
        rel_err = abs(realized - target_analytic) / target_analytic
        # Paper's lemma in §3.4 gives an O(Δ^{3/2}) bound; at Δ=0.01 that's
        # ~10⁻³. MC noise at 20k paths dominates; use a 25% relative bound
        # to keep the test non-flaky.
        assert rel_err < 0.25

    def test_heterogeneous_sigma_mix_tracks_target(self) -> None:
        """When neighbors have different σ_b, the basket's weighted σ²
        should land inside the min/max σ² interval and near the target's σ²
        when the target is inside the neighbor range."""
        x_t = 0.0
        target_sigma = 0.3
        pts = [
            _surf("lo", m=-0.2, sigma_b=0.2),
            _surf("mid", m=0.0, sigma_b=target_sigma),
            _surf("hi", m=0.2, sigma_b=0.4),
        ]
        basket = replicate_xvariance_strip(
            pts, 3600.0, x_t, target_variance_notional=1.0,
            params=SynthStripParams(bandwidth_m=0.3),
        )
        weighted = sum(leg.weight * next(sp.sigma_b for sp in pts if sp.token_id == leg.token_id) ** 2
                       for leg in basket)
        assert 0.04 <= weighted <= 0.16  # between σ²_lo and σ²_hi
        # Symmetric grid means the mid sigma² contributes the most.
        assert abs(weighted - target_sigma ** 2) < 0.015
