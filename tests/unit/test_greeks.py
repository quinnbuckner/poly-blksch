"""Unit tests for mm/greeks.py — the math must be airtight (paper §4.1)."""

from __future__ import annotations

import math

import pytest

from blksch.mm.greeks import (
    clip_p,
    delta_x,
    gamma_x,
    logit,
    s_double_prime,
    s_prime,
    sigmoid,
    vega_b_pvar,
    vega_b_xvar,
    vega_rho_pair,
)


class TestSigmoidLogit:
    def test_sigmoid_zero_is_half(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    @pytest.mark.parametrize("x", [-10.0, -1.0, 0.0, 0.5, 3.0, 10.0])
    def test_sigmoid_logit_inverse(self, x: float) -> None:
        assert logit(sigmoid(x)) == pytest.approx(x, abs=1.0e-6)

    @pytest.mark.parametrize("x", [-1000.0, 1000.0])
    def test_sigmoid_stable_at_extremes(self, x: float) -> None:
        y = sigmoid(x)
        assert 0.0 <= y <= 1.0
        assert not math.isnan(y) and not math.isinf(y)

    def test_sigmoid_monotone(self) -> None:
        xs = [-5.0, -1.0, 0.0, 1.0, 5.0]
        ys = [sigmoid(x) for x in xs]
        assert ys == sorted(ys)


class TestGreekIdentities:
    @pytest.mark.parametrize("p", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    def test_delta_equals_sprime(self, p: float) -> None:
        """Δ_x = p(1-p) = S'(logit(p))."""
        assert delta_x(p) == pytest.approx(s_prime(logit(p)), abs=1.0e-9)

    @pytest.mark.parametrize("p", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    def test_gamma_equals_sdoubleprime(self, p: float) -> None:
        """Γ_x = p(1-p)(1-2p) = S''(logit(p))."""
        assert gamma_x(p) == pytest.approx(s_double_prime(logit(p)), abs=1.0e-9)

    def test_delta_max_at_p_half(self) -> None:
        """Δ_x is maximized at p=0.5 with value 0.25."""
        assert delta_x(0.5) == pytest.approx(0.25)
        assert delta_x(0.5) > delta_x(0.1)
        assert delta_x(0.5) > delta_x(0.9)

    def test_gamma_zero_at_p_half(self) -> None:
        """Γ_x = 0 at p=0.5 (inflection point of S)."""
        assert gamma_x(0.5) == pytest.approx(0.0, abs=1.0e-12)

    def test_gamma_sign_flip_through_half(self) -> None:
        """Γ_x is positive for p<0.5, negative for p>0.5."""
        assert gamma_x(0.3) > 0.0
        assert gamma_x(0.7) < 0.0

    @pytest.mark.parametrize("p", [0.001, 0.5, 0.999])
    def test_delta_nonneg(self, p: float) -> None:
        assert delta_x(p) >= 0.0


class TestBoundary:
    def test_delta_vanishes_at_boundary(self) -> None:
        """Δ_x → 0 as p → 0 or 1 — this is why the inventory cap tightens."""
        assert delta_x(0.001) < 0.01
        assert delta_x(0.999) < 0.01

    def test_clip_p_inside_range(self) -> None:
        assert clip_p(0.5, eps=0.01) == 0.5

    def test_clip_p_below(self) -> None:
        assert clip_p(-0.1, eps=0.01) == 0.01

    def test_clip_p_above(self) -> None:
        assert clip_p(1.5, eps=0.01) == 0.99

    def test_clip_bad_eps(self) -> None:
        with pytest.raises(ValueError):
            clip_p(0.5, eps=0.6)


class TestVegas:
    def test_vega_b_xvar_linear_in_sigma(self) -> None:
        assert vega_b_xvar(0.5) == pytest.approx(0.5)
        assert vega_b_xvar(1.0) == pytest.approx(1.0)

    def test_vega_b_pvar_zero_at_boundary(self) -> None:
        """p-variance vega scales with (p(1-p))^2 so it vanishes at p→0,1."""
        assert vega_b_pvar(0.001, 1.0) < 1.0e-4
        assert vega_b_pvar(0.999, 1.0) < 1.0e-4

    def test_vega_b_pvar_peaks_at_half(self) -> None:
        v_half = vega_b_pvar(0.5, 1.0)
        assert v_half > vega_b_pvar(0.2, 1.0)
        assert v_half > vega_b_pvar(0.8, 1.0)

    def test_vega_rho_pair_scales_with_both_prices(self) -> None:
        """ν_ρ for a pair swap is symmetric and vanishes when either leg is at boundary."""
        assert vega_rho_pair(0.5, 0.5, 1.0, 1.0) > vega_rho_pair(0.5, 0.9, 1.0, 1.0)
        # 0.01 * 0.99 * 0.25 = 0.002475 — much smaller than 0.5,0.5 case (0.0625).
        assert vega_rho_pair(0.01, 0.5, 1.0, 1.0) < 0.1 * vega_rho_pair(0.5, 0.5, 1.0, 1.0)
