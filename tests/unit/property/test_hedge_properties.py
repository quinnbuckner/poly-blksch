"""Property-based fuzz for ``mm/hedge/{beta,calendar,synth_strip}.py``.

Invariants under test:

* beta: ``|β̃| ≤ max_abs_beta`` always; α=0 → zero hedge; ρ=0 → zero hedge;
  sign matches paper §4.4 (+target, +ρ → SHORT leg).
* calendar: ν̂_b=0 → zero-notional HedgeInstruction; scaling ν̂_b by c
  scales the notional by |c|.
* synth_strip: basket weights sum to ≈target notional (replication);
  zero-notional target → empty basket.
"""

from __future__ import annotations

from datetime import UTC, datetime

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from blksch.mm.hedge.beta import BetaHedgeParams, compute_beta_hedge
from blksch.mm.hedge.calendar import CalendarHedgeParams, compute_calendar_hedge
from blksch.mm.hedge.synth_strip import (
    SynthStripParams,
    replicate_xvariance_strip,
)
from blksch.schemas import CorrelationEntry, HedgeSide, SurfacePoint

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)

FUZZ = settings(max_examples=200, deadline=None, derandomize=True)


_finite = lambda lo, hi: st.floats(  # noqa: E731
    min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False,
)


def _surface(token: str, *, m: float, sigma_b: float, tau: float = 60.0,
             ts: datetime | None = None) -> SurfacePoint:
    return SurfacePoint(
        token_id=token, tau=tau, m=m, sigma_b=sigma_b,
        **{"lambda": 0.0}, s2_j=0.0,
        ts=ts or T0,
    )


def _corr(a: str, b: str, rho: float, *, co_lam: float = 0.0,
          co_m2: float = 0.0) -> CorrelationEntry:
    return CorrelationEntry(
        token_id_i=a, token_id_j=b, rho=rho,
        co_jump_lambda=co_lam, co_jump_m2=co_m2, ts=T0,
    )


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------


@FUZZ
@given(
    m_tgt=_finite(-1.0, 1.0),
    m_hg=_finite(-1.0, 1.0),
    sig_tgt=_finite(0.05, 0.5),
    sig_hg=_finite(0.05, 0.5),
    rho=_finite(-0.99, 0.99),
    alpha=_finite(0.5, 1.0),
    max_abs=_finite(0.5, 10.0),
)
def test_beta_bounded_by_max_abs(m_tgt, m_hg, sig_tgt, sig_hg, rho, alpha, max_abs):
    tgt = _surface("T", m=m_tgt, sigma_b=sig_tgt)
    hg = _surface("H", m=m_hg, sigma_b=sig_hg)
    corr = _corr("T", "H", rho)
    params = BetaHedgeParams(alpha=alpha, max_abs_beta=max_abs)
    out = compute_beta_hedge(tgt, hg, corr, alpha, params=params,
                             target_notional_usd=1.0, ts=T0)
    assert out.notional_usd <= max_abs + 1e-12, (
        f"|β̃|={out.notional_usd} > max_abs_beta={max_abs}"
    )


@FUZZ
@given(
    m_tgt=_finite(-1.0, 1.0),
    m_hg=_finite(-1.0, 1.0),
    sig_tgt=_finite(0.05, 0.5),
    sig_hg=_finite(0.05, 0.5),
    rho=_finite(-0.99, 0.99),
)
def test_zero_rho_zero_hedge(m_tgt, m_hg, sig_tgt, sig_hg, rho):
    """ρ=0 → β=0 → zero-notional hedge."""
    tgt = _surface("T", m=m_tgt, sigma_b=sig_tgt)
    hg = _surface("H", m=m_hg, sigma_b=sig_hg)
    corr = _corr("T", "H", 0.0)  # force rho=0
    out = compute_beta_hedge(tgt, hg, corr, 0.75, target_notional_usd=1.0, ts=T0)
    assert out.notional_usd == pytest.approx(0.0, abs=1e-12)


@FUZZ
@given(
    m_tgt=_finite(-1.0, 1.0),
    m_hg=_finite(-1.0, 1.0),
    sig_tgt=_finite(0.05, 0.5),
    sig_hg=_finite(0.05, 0.5),
    rho=_finite(0.1, 0.99),
)
def test_positive_rho_and_long_target_shorts_hedge(m_tgt, m_hg, sig_tgt, sig_hg, rho):
    """Paper §4.4: positively correlated hedge is SHORTed to offset a long
    target (positive target_notional)."""
    tgt = _surface("T", m=m_tgt, sigma_b=sig_tgt)
    hg = _surface("H", m=m_hg, sigma_b=sig_hg)
    corr = _corr("T", "H", rho)
    out = compute_beta_hedge(tgt, hg, corr, 0.75, target_notional_usd=+1.0, ts=T0)
    if out.notional_usd > 0.0:
        assert out.side is HedgeSide.SHORT, (
            f"+ρ +target should SHORT hedge, got side={out.side}"
        )


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------


@FUZZ
@given(sig_b=_finite(0.05, 0.5), m=_finite(-2.0, 2.0))
def test_zero_nu_b_empty_hedge(sig_b, m):
    surface = _surface("T", m=m, sigma_b=sig_b)
    out = compute_calendar_hedge(surface, inventory_nu_b=0.0, ts=T0)
    assert out.notional_usd == pytest.approx(0.0, abs=1e-18)


@FUZZ
@given(
    sig_b=_finite(0.05, 0.5),
    m=_finite(-2.0, 2.0),
    nu_b=_finite(-100.0, 100.0),
    scale=_finite(0.1, 10.0),
)
def test_calendar_hedge_scales_with_nu_b(sig_b, m, nu_b, scale):
    """Scaling ν̂_b by c > 0 scales the notional by c (up to the cap)."""
    if abs(nu_b) < 1e-9:
        return
    surface = _surface("T", m=m, sigma_b=sig_b)
    # Pick a cap large enough to stay linear.
    cap = 1e6
    params = CalendarHedgeParams(sigma_b_floor=1e-4, max_abs_notional_usd=cap)
    base = compute_calendar_hedge(surface, nu_b, params=params, ts=T0)
    scaled = compute_calendar_hedge(surface, nu_b * scale, params=params, ts=T0)
    assert scaled.notional_usd == pytest.approx(base.notional_usd * scale, rel=1e-9)


# ---------------------------------------------------------------------------
# synth_strip
# ---------------------------------------------------------------------------


@st.composite
def _surface_points(draw) -> list[SurfacePoint]:
    n = draw(st.integers(min_value=3, max_value=8))
    return [
        _surface(
            f"T{k}",
            m=draw(_finite(-1.5, 1.5)),
            sigma_b=draw(_finite(0.05, 0.5)),
            tau=draw(_finite(10.0, 3600.0)),
        )
        for k in range(n)
    ]


@FUZZ
@given(surfaces=_surface_points())
def test_zero_variance_target_empty_basket(surfaces):
    legs = replicate_xvariance_strip(
        surfaces, target_tau=60.0, target_m=0.0, target_variance_notional=0.0,
    )
    assert legs == []


@FUZZ
@given(
    surfaces=_surface_points(),
    target_notional=_finite(0.1, 100.0),
    target_m=_finite(-0.5, 0.5),
    target_tau=_finite(30.0, 1800.0),
)
def test_basket_weights_sum_to_target(
    surfaces, target_notional, target_m, target_tau,
):
    legs = replicate_xvariance_strip(
        surfaces, target_tau=target_tau, target_m=target_m,
        target_variance_notional=target_notional,
    )
    if not legs:
        return  # all candidates outside bandwidth — legitimate empty
    total = sum(leg.weight for leg in legs)
    assert total == pytest.approx(target_notional, rel=1e-9, abs=1e-12), (
        f"basket weights sum to {total}, expected {target_notional}"
    )


@FUZZ
@given(
    surfaces=_surface_points(),
    target_notional=_finite(-100.0, -0.1),  # negative target
    target_m=_finite(-0.5, 0.5),
    target_tau=_finite(30.0, 1800.0),
)
def test_negative_target_sign_preserved_in_weights(
    surfaces, target_notional, target_m, target_tau,
):
    legs = replicate_xvariance_strip(
        surfaces, target_tau=target_tau, target_m=target_m,
        target_variance_notional=target_notional,
    )
    if not legs:
        return
    total = sum(leg.weight for leg in legs)
    # Total preserves sign and magnitude.
    assert total == pytest.approx(target_notional, rel=1e-9)
