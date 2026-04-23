"""Unit tests for ``core/filter/microstruct`` (paper §5.1 eq 10)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from blksch.core.filter.microstruct import (
    MicrostructConfig,
    MicrostructFeatures,
    MicrostructModel,
    extract_features,
)
from blksch.schemas import BookSnap, PriceLevel, TradeSide, TradeTick

pytestmark = pytest.mark.unit


# ---------- Helpers ----------


def _book(
    mid: float,
    *,
    half_spread: float = 0.01,
    top_size: float = 1000.0,
    levels: int = 5,
) -> BookSnap:
    bids = [
        PriceLevel(price=round(mid - half_spread - 0.01 * i, 6), size=top_size)
        for i in range(levels)
    ]
    asks = [
        PriceLevel(price=round(mid + half_spread + 0.01 * i, 6), size=top_size)
        for i in range(levels)
    ]
    return BookSnap(token_id="t", bids=bids, asks=asks, ts=datetime.now(UTC))


def _trade(price: float, size: float = 10.0, side: TradeSide = TradeSide.BUY) -> TradeTick:
    return TradeTick(
        token_id="t",
        price=price,
        size=size,
        aggressor_side=side,
        ts=datetime.now(UTC),
    )


def _gen_synthetic_design(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random (half_spread, inv_depth, abs_rate, abs_imbalance) in realistic ranges."""
    half_spread = rng.uniform(0.001, 0.05, size=n)
    inv_depth = rng.uniform(1e-5, 1e-2, size=n)
    abs_rate = rng.uniform(0.0, 2.0, size=n)
    abs_imb = rng.uniform(0.0, 0.9, size=n)
    return half_spread, inv_depth, abs_rate, abs_imb


# ---------- Config validation ----------


def test_config_rejects_nonpositive_fields() -> None:
    with pytest.raises(ValueError):
        MicrostructConfig(fit_window_sec=0)
    with pytest.raises(ValueError):
        MicrostructConfig(depth_levels=0)
    with pytest.raises(ValueError):
        MicrostructConfig(sigma_floor=0)
    with pytest.raises(ValueError):
        MicrostructConfig(forward_fill_widen_factor=0.5)
    with pytest.raises(ValueError):
        MicrostructConfig(ridge=-1.0)


# ---------- Feature extraction ----------


def test_extract_features_basic() -> None:
    # Mid=0.50, half_spread=0.01, top depth 1000 per level × 5 levels × 2 sides.
    book = _book(mid=0.50, half_spread=0.01, top_size=1000.0, levels=5)
    feats = extract_features(book, trades=[], depth_levels=5, trade_rate_window_sec=30.0)
    assert feats.half_spread == pytest.approx(0.01)
    assert feats.inv_depth == pytest.approx(1.0 / (5 * 1000 * 2))
    assert feats.abs_trade_rate == pytest.approx(0.0)
    assert feats.abs_imbalance == pytest.approx(0.0)


def test_extract_features_trade_rate() -> None:
    book = _book(mid=0.50)
    trades = [_trade(0.5) for _ in range(9)]  # 9 trades in 30s window → 0.3/s
    feats = extract_features(book, trades, trade_rate_window_sec=30.0)
    assert feats.abs_trade_rate == pytest.approx(0.3)


def test_extract_features_imbalance() -> None:
    # Heavier bid side.
    book = BookSnap(
        token_id="t",
        bids=[PriceLevel(price=0.49, size=900), PriceLevel(price=0.48, size=100)],
        asks=[PriceLevel(price=0.51, size=100), PriceLevel(price=0.52, size=100)],
        ts=datetime.now(UTC),
    )
    feats = extract_features(book, trades=[], depth_levels=5)
    # bid_depth=1000, ask_depth=200, imbalance=(1000-200)/1200 = 0.667
    assert feats.abs_imbalance == pytest.approx(800 / 1200)


def test_extract_features_empty_book_is_graceful() -> None:
    """No bids/asks → zero covariates; intercept will carry the base rate."""
    book = BookSnap(token_id="t", bids=[], asks=[], ts=datetime.now(UTC))
    feats = extract_features(book, trades=[])
    assert feats.half_spread == 0.0
    assert feats.inv_depth == 0.0
    assert feats.abs_imbalance == 0.0


# ---------- Coefficient recovery ----------


def test_recovers_known_coefficients_on_synthetic() -> None:
    rng = np.random.default_rng(0xB17C5C)
    n = 4000
    true_a = np.array([5.0e-4, 1.5, 2.0, 3.0e-4, 1.0e-4])

    half_spread, inv_depth, abs_rate, abs_imb = _gen_synthetic_design(rng, n)
    X = np.column_stack(
        [
            np.ones(n),
            half_spread * half_spread,
            inv_depth,
            abs_rate,
            abs_imb,
        ]
    )
    sigma2_true = X @ true_a
    # Guard against simulated σ² < 0 from the uniform mix.
    sigma2_true = np.maximum(sigma2_true, 1e-8)

    innov = rng.normal(0.0, np.sqrt(sigma2_true))
    y = innov * innov

    model = MicrostructModel.fit(X, y)
    got = np.array([model.a0, model.a1, model.a2, model.a3, model.a4])

    # OLS of squared innovations is noisy (Var[ε²] = 2σ⁴). Use a tolerance
    # that scales with the coefficient's true value.
    abs_tol = np.maximum(np.abs(true_a) * 0.5, 5e-4)
    assert np.all(np.abs(got - true_a) < abs_tol), (
        f"coeff recovery off: got={got}, true={true_a}, tol={abs_tol}"
    )


def test_fit_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        MicrostructModel.fit(np.zeros((10, 4)), np.zeros(10))  # wrong cols
    with pytest.raises(ValueError):
        MicrostructModel.fit(np.zeros((10, 5)), np.zeros(9))  # y mismatch
    with pytest.raises(ValueError):
        MicrostructModel.fit(np.zeros((3, 5)), np.zeros(3))  # too few rows


def test_fit_from_features_convenience() -> None:
    feats = [
        MicrostructFeatures(half_spread=0.01, inv_depth=1e-3, abs_trade_rate=0.1, abs_imbalance=0.2)
        for _ in range(10)
    ]
    y = [1e-4] * 10
    model = MicrostructModel.fit_from_features(feats, y)
    # Constant y → intercept ≈ 1e-4, other coefs small.
    assert model.a0 == pytest.approx(1e-4, abs=1e-5)


# ---------- Variance monotonicity ----------


def test_tighter_spread_lowers_variance() -> None:
    model = MicrostructModel(a0=0.0, a1=1.0, a2=0.0, a3=0.0, a4=0.0,
                             config=MicrostructConfig(sigma_floor=1e-20))
    wide = model.variance(_book(mid=0.5, half_spread=0.05))
    tight = model.variance(_book(mid=0.5, half_spread=0.005))
    assert tight < wide
    # Ratio should be (0.005/0.05)^2 = 0.01 exactly (only a1 active).
    assert tight == pytest.approx(wide * (0.005 / 0.05) ** 2, rel=1e-9)


def test_deeper_book_lowers_variance() -> None:
    model = MicrostructModel(a0=0.0, a1=0.0, a2=1.0, a3=0.0, a4=0.0,
                             config=MicrostructConfig(sigma_floor=1e-20))
    thin = model.variance(_book(mid=0.5, top_size=100.0))
    thick = model.variance(_book(mid=0.5, top_size=5000.0))
    assert thick < thin
    # inv_depth halves when top_size 10×s → variance ratio is 1/50 (5x * 10x).
    # Just assert strict monotone & rough order of magnitude.
    assert thick < thin / 40.0


def test_higher_trade_rate_raises_variance() -> None:
    model = MicrostructModel(a0=0.0, a1=0.0, a2=0.0, a3=1.0, a4=0.0,
                             config=MicrostructConfig(sigma_floor=1e-20))
    calm = model.variance(_book(mid=0.5), trades=[])
    busy = model.variance(_book(mid=0.5), trades=[_trade(0.5) for _ in range(60)])
    assert busy > calm


def test_stronger_imbalance_raises_variance() -> None:
    model = MicrostructModel(a0=0.0, a1=0.0, a2=0.0, a3=0.0, a4=1.0,
                             config=MicrostructConfig(sigma_floor=1e-20))
    balanced = BookSnap(
        token_id="t",
        bids=[PriceLevel(price=0.49, size=1000)],
        asks=[PriceLevel(price=0.51, size=1000)],
        ts=datetime.now(UTC),
    )
    skewed = BookSnap(
        token_id="t",
        bids=[PriceLevel(price=0.49, size=1900)],
        asks=[PriceLevel(price=0.51, size=100)],
        ts=datetime.now(UTC),
    )
    assert model.variance(skewed) > model.variance(balanced)


# ---------- Forward-fill widening ----------


def test_forward_fill_widens_by_config_factor() -> None:
    model = MicrostructModel(
        a0=1e-4, a1=1.0, a2=1e-2, a3=1e-3, a4=1e-4,
        config=MicrostructConfig(forward_fill_widen_factor=10.0),
    )
    book = _book(mid=0.5)
    fresh = model.variance(book, forward_filled=False)
    stale = model.variance(book, forward_filled=True)
    assert stale == pytest.approx(fresh * 10.0, rel=1e-9)


def test_forward_fill_factor_custom() -> None:
    model = MicrostructModel(
        a0=1e-4, a1=0.0, a2=0.0, a3=0.0, a4=0.0,
        config=MicrostructConfig(forward_fill_widen_factor=4.0),
    )
    fresh = model.variance(_book(mid=0.5), forward_filled=False)
    stale = model.variance(_book(mid=0.5), forward_filled=True)
    assert stale == pytest.approx(fresh * 4.0, rel=1e-9)


# ---------- Floor activation ----------


def test_floor_activates_when_book_is_perfect() -> None:
    floor = 1e-5
    model = MicrostructModel(
        a0=0.0, a1=1.0, a2=1.0, a3=1.0, a4=1.0,
        config=MicrostructConfig(sigma_floor=floor),
    )
    # Book: near-zero spread, huge depth.
    perfect = BookSnap(
        token_id="t",
        bids=[PriceLevel(price=0.5 - 1e-6, size=1e8)],
        asks=[PriceLevel(price=0.5 + 1e-6, size=1e8)],
        ts=datetime.now(UTC),
    )
    assert model.variance(perfect) == pytest.approx(floor)


def test_floor_also_applies_to_forward_fill() -> None:
    """If raw σ² < floor, σ² is clamped to floor; forward-fill widens from the
    floor."""
    floor = 1e-5
    factor = 10.0
    model = MicrostructModel(
        a0=0.0, a1=0.0, a2=0.0, a3=0.0, a4=0.0,
        config=MicrostructConfig(sigma_floor=floor, forward_fill_widen_factor=factor),
    )
    book = _book(mid=0.5)
    assert model.variance(book, forward_filled=False) == pytest.approx(floor)
    assert model.variance(book, forward_filled=True) == pytest.approx(floor * factor)


# ---------- Graceful degenerate input ----------


def test_no_trades_in_window_is_graceful() -> None:
    model = MicrostructModel(
        a0=1e-4, a1=1.0, a2=1.0, a3=1.0, a4=1.0,
        config=MicrostructConfig(),
    )
    sigma2 = model.variance(_book(mid=0.5), trades=[])
    # Finite, positive, >= floor.
    assert sigma2 > 0 and np.isfinite(sigma2)


def test_empty_book_does_not_crash() -> None:
    model = MicrostructModel(a0=5e-4, a1=1.0, a2=1.0, a3=1.0, a4=1.0)
    empty = BookSnap(token_id="t", bids=[], asks=[], ts=datetime.now(UTC))
    sigma2 = model.variance(empty)
    # Covariates all zero → σ² = a0 (above floor).
    assert sigma2 == pytest.approx(5e-4)


def test_output_is_strictly_positive_for_any_input() -> None:
    """Regardless of (possibly negative) learned coefficients, σ² > 0 via floor."""
    # Pathological: negative coefficients that would drive σ² negative.
    model = MicrostructModel(
        a0=-1.0, a1=-1.0, a2=-1.0, a3=-1.0, a4=-1.0,
        config=MicrostructConfig(sigma_floor=1e-9),
    )
    rng = np.random.default_rng(1)
    for _ in range(50):
        book = _book(mid=rng.uniform(0.05, 0.95), half_spread=rng.uniform(0.001, 0.05))
        assert model.variance(book) > 0
