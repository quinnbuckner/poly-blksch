"""Property-based fuzz for ``core/filter/microstruct.py`` (paper §5.1 eq 10).

Invariants:

* ``σ_η² ≥ sigma_floor`` for any valid input.
* Monotone: increasing spread (all else equal) weakly increases σ_η² when
  the fit has a non-negative spread coefficient.
* Monotone: increasing depth weakly decreases σ_η².
* ``forward_filled=True`` produces exactly ``widen_factor × forward_filled=False``.
* No NaN / Inf on any valid input.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from blksch.core.filter.microstruct import (
    DEFAULT_FORWARD_FILL_WIDEN_FACTOR,
    DEFAULT_SIGMA_FLOOR,
    MicrostructConfig,
    MicrostructFeatures,
    MicrostructModel,
)

pytestmark = pytest.mark.unit

FUZZ = settings(max_examples=200, deadline=None, derandomize=True)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


_finite = lambda lo, hi: st.floats(  # noqa: E731
    min_value=lo, max_value=hi,
    allow_nan=False, allow_infinity=False,
)


@st.composite
def _features(draw) -> MicrostructFeatures:
    return MicrostructFeatures(
        half_spread=draw(_finite(0.0, 0.2)),
        inv_depth=draw(_finite(0.0, 0.5)),  # 1/total_depth
        abs_trade_rate=draw(_finite(0.0, 20.0)),
        abs_imbalance=draw(_finite(0.0, 1.0)),
    )


@st.composite
def _non_negative_model(draw) -> MicrostructModel:
    """Model with non-negative coefficients so monotonicity holds.

    The fit procedure does not itself guarantee sign, but well-behaved
    fits tend to produce a positive spread/depth coefficient. This
    strategy reflects that regime; a fit with a negative coefficient is
    flagged by the diagnostics suite.
    """
    cfg = MicrostructConfig(
        sigma_floor=DEFAULT_SIGMA_FLOOR,
        forward_fill_widen_factor=DEFAULT_FORWARD_FILL_WIDEN_FACTOR,
    )
    return MicrostructModel(
        a0=draw(_finite(1e-5, 1e-3)),
        a1=draw(_finite(0.0, 10.0)),
        a2=draw(_finite(0.0, 1e-4)),
        a3=draw(_finite(0.0, 1e-4)),
        a4=draw(_finite(0.0, 1e-5)),
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@FUZZ
@given(model=_non_negative_model(), feat=_features())
def test_sigma_floor_always_holds(model: MicrostructModel, feat: MicrostructFeatures):
    sigma2 = model.variance_from_features(feat)
    assert sigma2 >= model.config.sigma_floor - 1e-18
    assert math.isfinite(sigma2)


@FUZZ
@given(model=_non_negative_model(), feat=_features())
def test_no_nan_or_inf(model: MicrostructModel, feat: MicrostructFeatures):
    for ff in (False, True):
        sigma2 = model.variance_from_features(feat, forward_filled=ff)
        assert math.isfinite(sigma2), f"sigma2={sigma2} for ff={ff}"


@FUZZ
@given(
    model=_non_negative_model(),
    feat=_features(),
    bump=_finite(0.001, 0.05),
)
def test_monotone_in_spread(
    model: MicrostructModel, feat: MicrostructFeatures, bump: float,
):
    """Holding other covariates fixed, widening the spread should weakly
    raise σ_η² when the spread coefficient is non-negative."""
    wider = MicrostructFeatures(
        half_spread=feat.half_spread + bump,
        inv_depth=feat.inv_depth,
        abs_trade_rate=feat.abs_trade_rate,
        abs_imbalance=feat.abs_imbalance,
    )
    base = model.variance_from_features(feat)
    wider_sig = model.variance_from_features(wider)
    # Strict sigma floor can equalize them when both are below floor — allow.
    assert wider_sig + 1e-15 >= base, (
        f"spread monotonicity broken: base={base:.6e} wider={wider_sig:.6e}"
    )


@FUZZ
@given(
    model=_non_negative_model(),
    feat=_features(),
    extra_depth=_finite(0.1, 100.0),
)
def test_monotone_in_depth(
    model: MicrostructModel, feat: MicrostructFeatures, extra_depth: float,
):
    """More depth means smaller 1/depth means weakly lower σ_η²."""
    # Convert inv_depth to depth, add, convert back.
    if feat.inv_depth <= 0.0:
        return  # nothing to compare against
    base_depth = 1.0 / feat.inv_depth
    new_depth = base_depth + extra_depth
    deeper = MicrostructFeatures(
        half_spread=feat.half_spread,
        inv_depth=1.0 / new_depth,
        abs_trade_rate=feat.abs_trade_rate,
        abs_imbalance=feat.abs_imbalance,
    )
    base = model.variance_from_features(feat)
    deeper_sig = model.variance_from_features(deeper)
    assert deeper_sig <= base + 1e-15, (
        f"depth monotonicity broken: base={base:.6e} deeper={deeper_sig:.6e}"
    )


@FUZZ
@given(model=_non_negative_model(), feat=_features())
def test_forward_filled_widens_by_exact_factor(
    model: MicrostructModel, feat: MicrostructFeatures,
):
    """ff=True output is exactly widen_factor × ff=False output, EXCEPT when
    the ff=False result is the sigma_floor (which itself is not widened
    before the multiply — widen happens after the floor clamp)."""
    base = model.variance_from_features(feat, forward_filled=False)
    wide = model.variance_from_features(feat, forward_filled=True)
    expected = base * model.config.forward_fill_widen_factor
    assert wide == pytest.approx(expected, rel=1e-9, abs=1e-18), (
        f"ff widen: wide={wide} vs expected {expected} (base={base}, "
        f"factor={model.config.forward_fill_widen_factor})"
    )
