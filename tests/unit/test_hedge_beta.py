"""Unit tests for mm/hedge/beta.py — cross-event β-hedge (paper §4.4)."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from blksch.mm.greeks import s_prime
from blksch.mm.hedge.beta import (
    BetaHedgeParams,
    co_jump_correction,
    compute_beta_hedge,
    raw_beta,
)
from blksch.schemas import CorrelationEntry, HedgeSide, SurfacePoint


TS = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)


def _surf(token_id: str, x: float = 0.0, sigma_b: float = 0.3) -> SurfacePoint:
    return SurfacePoint(
        token_id=token_id,
        tau=3600.0,
        m=x,
        sigma_b=sigma_b,
        **{"lambda": 0.1},
        s2_j=0.01,
        ts=TS,
    )


def _corr(
    i: str, j: str, rho: float = 0.6, co_jump_lambda: float = 0.0, co_jump_m2: float = 0.0
) -> CorrelationEntry:
    return CorrelationEntry(
        token_id_i=i, token_id_j=j,
        rho=rho, co_jump_lambda=co_jump_lambda, co_jump_m2=co_jump_m2,
        ts=TS,
    )


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


class TestRawBeta:
    def test_simple_ratio(self) -> None:
        # sp_i=0.2, sp_j=0.25, rho=0.5  ⇒  β = 0.2/0.25 · 0.5 = 0.4
        assert raw_beta(0.2, 0.25, 0.5) == pytest.approx(0.4)

    def test_sign_follows_rho(self) -> None:
        assert raw_beta(0.25, 0.25, 0.6) > 0.0
        assert raw_beta(0.25, 0.25, -0.6) < 0.0

    def test_guard_zero_denom(self) -> None:
        with pytest.raises(ZeroDivisionError):
            raw_beta(0.25, 0.0, 0.5)


class TestCoJumpCorrection:
    def test_zero_when_no_cojumps(self) -> None:
        assert co_jump_correction(0.0, 0.0, 0.2, 0.3) == 0.0

    def test_positive_co_jump_pushes_beta_up(self) -> None:
        c = co_jump_correction(co_jump_lambda=0.5, co_jump_m2=1e-3, sp_hedge=0.2, sigma_b_hedge=0.3)
        assert c > 0.0

    def test_denom_floor(self) -> None:
        assert co_jump_correction(0.5, 1e-3, sp_hedge=0.0, sigma_b_hedge=0.3) == 0.0


# ---------------------------------------------------------------------------
# BetaHedgeParams validation
# ---------------------------------------------------------------------------


class TestParams:
    def test_alpha_range(self) -> None:
        BetaHedgeParams(alpha=0.5)
        BetaHedgeParams(alpha=1.0)
        with pytest.raises(ValueError):
            BetaHedgeParams(alpha=0.4)
        with pytest.raises(ValueError):
            BetaHedgeParams(alpha=1.2)

    def test_floor_positive(self) -> None:
        with pytest.raises(ValueError):
            BetaHedgeParams(s_prime_floor=0.0)

    def test_max_abs_beta_positive(self) -> None:
        with pytest.raises(ValueError):
            BetaHedgeParams(max_abs_beta=-1.0)


# ---------------------------------------------------------------------------
# compute_beta_hedge — main entry
# ---------------------------------------------------------------------------


class TestComputeBetaHedge:
    def test_symmetric_midpoint_matches_rho(self) -> None:
        """When both markets sit at p=0.5, S'_i/S'_j = 1 so β = ρ exactly."""
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c = _corr("A", "B", rho=0.6)
        result = compute_beta_hedge(t, h, c, alpha=1.0, ts=TS)
        # β = 1 · (0.25/0.25) · 0.6 = 0.6; target_notional=1 ⇒ notional=0.6
        assert result.notional_usd == pytest.approx(0.6)
        assert result.side is HedgeSide.SHORT  # positive β ⇒ short the hedge

    def test_alpha_scales_output(self) -> None:
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c = _corr("A", "B", rho=0.6)
        full = compute_beta_hedge(t, h, c, alpha=1.0, ts=TS)
        shrunk = compute_beta_hedge(t, h, c, alpha=0.5, ts=TS)
        assert shrunk.notional_usd == pytest.approx(0.5 * full.notional_usd)

    def test_anti_correlation_flips_side(self) -> None:
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c = _corr("A", "B", rho=-0.6)
        result = compute_beta_hedge(t, h, c, alpha=1.0, ts=TS)
        assert result.side is HedgeSide.LONG
        assert result.notional_usd == pytest.approx(0.6)

    def test_boundary_clamp_zeros_out(self) -> None:
        """If S'_hedge is near zero, the hedge is unreliable → zero notional."""
        t = _surf("A", x=0.0)        # p=0.5, S'=0.25
        h = _surf("B", x=15.0)       # p≈1, S' extremely tiny
        c = _corr("A", "B", rho=0.6)
        params = BetaHedgeParams(alpha=1.0, s_prime_floor=1e-4, max_abs_beta=1000.0)
        result = compute_beta_hedge(t, h, c, alpha=1.0, params=params, ts=TS)
        assert result.notional_usd == 0.0

    def test_max_abs_beta_clamp(self) -> None:
        """Asymmetry between S'_i and S'_j can produce huge β; max_abs_beta caps it."""
        # target at p=0.5 (S'=0.25), hedge at p≈0.9 (S'≈0.09) ⇒ ratio ~ 2.78
        t = _surf("A", x=0.0)
        h = _surf("B", x=2.2)
        c = _corr("A", "B", rho=0.9)
        params = BetaHedgeParams(alpha=1.0, max_abs_beta=0.5)
        result = compute_beta_hedge(t, h, c, alpha=1.0, params=params, ts=TS)
        assert result.notional_usd == pytest.approx(0.5)

    def test_co_jump_only_when_flag_on(self) -> None:
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c = _corr("A", "B", rho=0.6, co_jump_lambda=0.5, co_jump_m2=1e-3)

        off = compute_beta_hedge(t, h, c, alpha=1.0,
                                 params=BetaHedgeParams(alpha=1.0, apply_co_jump=False), ts=TS)
        on = compute_beta_hedge(t, h, c, alpha=1.0,
                                params=BetaHedgeParams(alpha=1.0, apply_co_jump=True), ts=TS)
        assert on.notional_usd > off.notional_usd

    def test_co_jump_correction_positive_contribution(self) -> None:
        """Positive co-jump covariance should push β in the same direction as ρ."""
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c_base = _corr("A", "B", rho=0.6, co_jump_lambda=0.0, co_jump_m2=0.0)
        c_jump = _corr("A", "B", rho=0.6, co_jump_lambda=1.0, co_jump_m2=1e-3)
        params = BetaHedgeParams(alpha=1.0, apply_co_jump=True)
        base = compute_beta_hedge(t, h, c_base, alpha=1.0, params=params, ts=TS)
        with_jump = compute_beta_hedge(t, h, c_jump, alpha=1.0, params=params, ts=TS)
        assert with_jump.notional_usd > base.notional_usd
        assert with_jump.side is base.side  # still positive β → short hedge

    def test_target_notional_scales(self) -> None:
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c = _corr("A", "B", rho=0.5)
        unit = compute_beta_hedge(t, h, c, alpha=1.0, ts=TS, target_notional_usd=1.0)
        ten = compute_beta_hedge(t, h, c, alpha=1.0, ts=TS, target_notional_usd=10.0)
        assert ten.notional_usd == pytest.approx(10.0 * unit.notional_usd)

    def test_short_target_flips_side(self) -> None:
        """A negative target_notional_usd (short position) flips the hedge direction."""
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c = _corr("A", "B", rho=0.6)  # positive β ⇒ short hedge for long target
        long_target = compute_beta_hedge(t, h, c, alpha=1.0, ts=TS, target_notional_usd=1.0)
        short_target = compute_beta_hedge(t, h, c, alpha=1.0, ts=TS, target_notional_usd=-1.0)
        assert long_target.side is HedgeSide.SHORT
        assert short_target.side is HedgeSide.LONG
        assert short_target.notional_usd == pytest.approx(long_target.notional_usd)

    def test_pair_mismatch_raises(self) -> None:
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c = _corr("X", "Y", rho=0.6)
        with pytest.raises(ValueError):
            compute_beta_hedge(t, h, c, alpha=1.0, ts=TS)

    def test_reversed_pair_ok(self) -> None:
        """CorrelationEntry's (i, j) can be given in reversed order — rho is symmetric."""
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c = _corr("B", "A", rho=0.6)  # reversed
        result = compute_beta_hedge(t, h, c, alpha=1.0, ts=TS)
        assert result.notional_usd == pytest.approx(0.6)

    def test_params_override_alpha_argument(self) -> None:
        """If params.alpha is set, it takes precedence over the alpha argument."""
        t = _surf("A", x=0.0)
        h = _surf("B", x=0.0)
        c = _corr("A", "B", rho=0.6)
        result = compute_beta_hedge(
            t, h, c, alpha=1.0, params=BetaHedgeParams(alpha=0.5), ts=TS
        )
        assert result.notional_usd == pytest.approx(0.3)  # 0.5 · 0.6
