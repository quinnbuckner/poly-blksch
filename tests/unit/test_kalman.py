"""Unit tests for ``core/filter/kalman.KalmanFilter`` (paper §5.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy import stats

from blksch.core.filter.canonical_mid import CanonicalMid
from blksch.core.filter.kalman import (
    DEFAULT_BOUNDARY_P_HIGH,
    DEFAULT_BOUNDARY_P_LOW,
    KalmanFilter,
    _sigmoid,
)
from blksch.schemas import BookSnap, PriceLevel

pytestmark = pytest.mark.unit


# ---------- Helpers ----------


@dataclass
class ConstantMicrostruct:
    """Fixed σ_η² for every call — decouples KF tests from eq (10) fit."""

    sigma_eta2: float

    def variance(self, book, trades=(), *, forward_filled: bool = False) -> float:
        return self.sigma_eta2


@dataclass
class ScheduledMicrostruct:
    """Returns a sequence of variances in order — for regime / divergence drills."""

    variances: list[float]
    _i: int = 0

    def variance(self, book, trades=(), *, forward_filled: bool = False) -> float:
        if self._i >= len(self.variances):
            return self.variances[-1]
        v = self.variances[self._i]
        self._i += 1
        return v


def _book(p: float) -> BookSnap:
    """Minimal dummy book — content doesn't matter for the Constant oracle."""
    p = max(1e-5, min(1 - 1e-5, p))
    return BookSnap(
        token_id="t",
        bids=[PriceLevel(price=max(0.0, p - 0.01), size=1000)],
        asks=[PriceLevel(price=min(1.0, p + 0.01), size=1000)],
        ts=datetime.now(UTC),
    )


def _cm(ts: datetime, y: float, *, forward_filled: bool = False) -> CanonicalMid:
    p = max(1e-5, min(1 - 1e-5, _sigmoid(y)))
    return CanonicalMid(
        token_id="t",
        ts=ts,
        p_tilde=p,
        y=y,
        forward_filled=forward_filled,
        rejected_outlier=False,
        trades_in_window=0,
        source="book_mid",
    )


def _ljung_box_pvalue(residuals: np.ndarray, lags: int) -> float:
    """Manual Ljung–Box (statsmodels not in the project deps).

    Q = n(n+2) Σ_{k=1}^{lags} ρ̂_k² / (n - k),  Q ~ χ²(lags) under H₀.
    Returns the right-tail p-value.
    """
    n = residuals.size
    x = residuals - residuals.mean()
    var = np.dot(x, x)
    if var <= 0:
        return 1.0
    q = 0.0
    for k in range(1, lags + 1):
        rho = np.dot(x[:-k], x[k:]) / var
        q += rho * rho / (n - k)
    q *= n * (n + 2)
    return float(stats.chi2.sf(q, df=lags))


# ---------- Constructor validation ----------


def test_rejects_negative_sigma_b() -> None:
    with pytest.raises(ValueError):
        KalmanFilter(token_id="t", microstruct=ConstantMicrostruct(0.01), sigma_b=-0.1)


def test_rejects_non_positive_initial_variance() -> None:
    with pytest.raises(ValueError):
        KalmanFilter(token_id="t", microstruct=ConstantMicrostruct(0.01), initial_variance=0)


def test_rejects_bad_boundary_thresholds() -> None:
    with pytest.raises(ValueError):
        KalmanFilter(token_id="t", microstruct=ConstantMicrostruct(0.01), boundary_p_low=0.6)
    with pytest.raises(ValueError):
        KalmanFilter(token_id="t", microstruct=ConstantMicrostruct(0.01), boundary_p_high=0.4)


# ---------- Synthetic-path recovery ----------


def _simulate_random_walk(
    rng: np.random.Generator,
    *,
    n: int,
    sigma_b: float,
    sigma_eta: float,
    dt: float = 1.0,
    x0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    x_true = np.empty(n)
    x_true[0] = x0
    for t in range(1, n):
        x_true[t] = x_true[t - 1] + rng.normal(0.0, sigma_b * np.sqrt(dt))
    y_obs = x_true + rng.normal(0.0, sigma_eta, size=n)
    return x_true, y_obs


def test_recovery_within_3_sigma_on_95pct_of_steps() -> None:
    rng = np.random.default_rng(2026)
    n = 2000
    sigma_b = 0.2
    sigma_eta = 0.15
    x_true, y_obs = _simulate_random_walk(rng, n=n, sigma_b=sigma_b, sigma_eta=sigma_eta)

    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(sigma_eta * sigma_eta),
        sigma_b=sigma_b,
    )
    t0 = datetime(2026, 4, 23, 7, 0, 0, tzinfo=UTC)
    within = 0
    # Skip the first few ticks where the posterior is still warming up.
    warmup = 5
    counted = 0
    for t in range(n):
        ts = t0 + timedelta(seconds=t)
        state = kf.step(_cm(ts, y_obs[t]), _book(_sigmoid(y_obs[t])))
        if t < warmup:
            continue
        counted += 1
        sigma_total = float(np.sqrt(kf.posterior_variance + state.sigma_eta2))
        if abs(state.x_hat - x_true[t]) < 3.0 * sigma_total:
            within += 1
    assert within / counted >= 0.95, f"only {within}/{counted} within 3σ"


def test_recovery_mse_small_vs_noise() -> None:
    rng = np.random.default_rng(7)
    n = 1500
    sigma_b = 0.1
    sigma_eta = 0.2
    x_true, y_obs = _simulate_random_walk(rng, n=n, sigma_b=sigma_b, sigma_eta=sigma_eta)

    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(sigma_eta * sigma_eta),
        sigma_b=sigma_b,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    x_hat = np.empty(n)
    for t in range(n):
        ts = t0 + timedelta(seconds=t)
        state = kf.step(_cm(ts, y_obs[t]), _book(_sigmoid(y_obs[t])))
        x_hat[t] = state.x_hat

    # KF should beat the raw observations by a healthy margin.
    mse_kf = float(np.mean((x_hat[50:] - x_true[50:]) ** 2))
    mse_raw = float(np.mean((y_obs[50:] - x_true[50:]) ** 2))
    assert mse_kf < 0.5 * mse_raw, f"KF MSE={mse_kf:.4f} did not beat raw MSE={mse_raw:.4f}"


# ---------- Innovation whitening ----------


def test_innovations_pass_ljung_box() -> None:
    """Normalized one-step innovations are white under correct σ_η² calibration.

    We keep the simulated path well away from the logit boundary (small
    sigma_b × sqrt(n) keeps |x| < 4 in practice), so the UKF augmentation
    stays inactive and the KF operates in its linear regime.
    """
    rng = np.random.default_rng(11)
    n = 3000
    sigma_b = 0.03
    sigma_eta = 0.1
    x_true, y_obs = _simulate_random_walk(rng, n=n, sigma_b=sigma_b, sigma_eta=sigma_eta)

    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(sigma_eta * sigma_eta),
        sigma_b=sigma_b,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    normed = []
    for t in range(n):
        ts = t0 + timedelta(seconds=t)
        kf.step(_cm(ts, y_obs[t]), _book(_sigmoid(y_obs[t])))
        if kf.last_innovation is None or kf.last_innovation_variance is None:
            continue
        if t == 0:
            continue
        normed.append(kf.last_innovation / np.sqrt(kf.last_innovation_variance))

    normed_arr = np.array(normed[100:])  # warmup
    p_value = _ljung_box_pvalue(normed_arr, lags=20)
    assert p_value > 0.05, f"Ljung–Box rejected whiteness (p={p_value:.3f})"


# ---------- UKF / KF handoff smoothness ----------


def test_no_discontinuity_at_boundary_sweep() -> None:
    """Drive p smoothly across the p_low=0.02 boundary and back.

    With fine steps in p, consecutive posterior means x̂ should not jump by
    more than ~1σ (the natural scale of a single innovation); the UKF blend
    weight is continuous in p so there is no step at p=0.02.
    """
    sigma_eta = 0.05
    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(sigma_eta * sigma_eta),
        sigma_b=0.05,
        boundary_p_low=DEFAULT_BOUNDARY_P_LOW,
        boundary_p_high=DEFAULT_BOUNDARY_P_HIGH,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)

    # Smooth sweep of p values: interior → through 0.02 → deeper into boundary → back out.
    p_path = np.concatenate([
        np.linspace(0.05, 0.012, 200),
        np.linspace(0.012, 0.05, 200),
    ])
    y_path = np.log(p_path / (1.0 - p_path))

    # Burn in so the posterior stabilizes before we hit the boundary.
    for i in range(30):
        kf.step(_cm(t0 + timedelta(seconds=i), float(y_path[0])), _book(float(p_path[0])))
    prev_x = kf.x_hat
    jumps = []
    for i, y in enumerate(y_path):
        ts = t0 + timedelta(seconds=30 + i)
        state = kf.step(_cm(ts, float(y)), _book(float(p_path[i])))
        sigma = float(np.sqrt(kf.posterior_variance + state.sigma_eta2))
        jumps.append(abs(state.x_hat - prev_x) / sigma)
        prev_x = state.x_hat
    assert max(jumps) < 1.0, f"max step jump was {max(jumps):.3f} σ"


def test_handoff_blend_weight_is_continuous_at_p_low() -> None:
    """Probe the blend weight at adjacent p values straddling p_low; the
    internal R_effective must not step (< ~2% relative jump for a small p move).
    """
    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(0.01),
        sigma_b=0.05,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    # Warm up at p=0.5.
    kf.step(_cm(t0, 0.0), _book(0.5))
    # Step 1: p just above p_low (0.021).
    kf.step(_cm(t0 + timedelta(seconds=1), float(np.log(0.021 / 0.979))), _book(0.021))
    r_just_outside = kf.last_R_effective
    # Step 2: p just below p_low (0.019).
    kf.step(_cm(t0 + timedelta(seconds=2), float(np.log(0.019 / 0.981))), _book(0.019))
    r_just_inside = kf.last_R_effective
    # Both should be close — the blend weight moved only a little.
    assert r_just_outside is not None and r_just_inside is not None
    rel = abs(r_just_inside - r_just_outside) / max(r_just_outside, 1e-12)
    assert rel < 0.05, f"R jumped {rel:.3f} across the p_low boundary"


def test_far_from_boundary_matches_linear_kf() -> None:
    """In the interior (p ∈ (p_low, p_high)), R_effective equals R_raw."""
    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(0.01),
        sigma_b=0.1,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    kf.step(_cm(t0, 0.0), _book(0.5))  # initialize at p=0.5
    kf.step(_cm(t0 + timedelta(seconds=1), 0.1), _book(0.52))
    assert kf.last_R_effective == pytest.approx(0.01)


# ---------- Divergence protection ----------


def test_divergence_under_pathological_variance() -> None:
    """Alternate σ_η² between 1e-12 and 1e12 each step — KF must stay bounded."""
    variances = []
    for _ in range(100):
        variances.extend([1e-12, 1e12])
    kf = KalmanFilter(
        token_id="t",
        microstruct=ScheduledMicrostruct(variances=variances),
        sigma_b=0.1,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    for i in range(200):
        ts = t0 + timedelta(seconds=i)
        state = kf.step(_cm(ts, 0.0), _book(0.5))
        assert abs(state.x_hat) < 100.0, f"x̂ blew up to {state.x_hat} at step {i}"
        assert np.isfinite(kf.posterior_variance)
        assert 0.0 <= (kf.last_K or 0.0) <= 1.0


def test_gain_clipped_to_unit_interval() -> None:
    """Even with P_pred >> R (extreme case), gain stays ≤ 1."""
    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(1e-12),
        sigma_b=10.0,  # huge process noise → P_pred grows fast
        initial_variance=1000.0,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    for i in range(10):
        ts = t0 + timedelta(seconds=i)
        kf.step(_cm(ts, 0.5), _book(0.6))
        if kf.last_K is not None:
            assert 0.0 <= kf.last_K <= 1.0


# ---------- LogitState shape ----------


def test_emits_logit_state_with_microstruct_variance() -> None:
    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(0.05),
        sigma_b=0.1,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    state = kf.step(_cm(t0, 0.25), _book(0.56))
    assert state.token_id == "t"
    assert state.ts == t0
    assert state.sigma_eta2 == pytest.approx(0.05)
    assert state.x_hat == pytest.approx(0.25)  # anchored to obs on first tick


def test_surfaces_raw_microstruct_variance_even_near_boundary() -> None:
    """Inflation inside _effective_R is an internal knob; the published
    sigma_eta2 stays faithful to MicrostructModel.variance output."""
    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(0.04),
        sigma_b=0.1,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    # Initialize first.
    kf.step(_cm(t0, 0.0), _book(0.5))
    state = kf.step(_cm(t0 + timedelta(seconds=1), 4.0), _book(0.98))  # near boundary
    assert state.sigma_eta2 == pytest.approx(0.04)
    # The internal effective R should be ≥ raw.
    assert kf.last_R_effective >= 0.04


# ---------- Forward-fill passthrough ----------


@dataclass
class FFMicrostruct:
    base: float
    factor: float
    last_ff: bool = False

    def variance(self, book, trades=(), *, forward_filled: bool = False) -> float:
        self.last_ff = forward_filled
        return self.base * (self.factor if forward_filled else 1.0)


def test_forward_filled_flag_passed_to_microstruct() -> None:
    m = FFMicrostruct(base=0.01, factor=10.0)
    kf = KalmanFilter(token_id="t", microstruct=m, sigma_b=0.1)
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    kf.step(_cm(t0, 0.0), _book(0.5))  # fresh
    assert m.last_ff is False
    kf.step(_cm(t0 + timedelta(seconds=1), 0.0, forward_filled=True), _book(0.5))
    assert m.last_ff is True


# ---------- dt handling ----------


def test_long_gap_is_capped() -> None:
    """A long gap between ticks should not drive P_pred to infinity."""
    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(0.01),
        sigma_b=0.5,
        max_dt_sec=60.0,
    )
    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    kf.step(_cm(t0, 0.0), _book(0.5))
    # 1 hour gap — would be Q = σ_b²·3600 = 900 without cap. With cap: 15.
    kf.step(_cm(t0 + timedelta(hours=1), 0.0), _book(0.5))
    # Posterior variance stays comparable to capped predict + innovation.
    assert kf.posterior_variance < 30.0


# ---------- Boundary-regime EM inflation regression (track-a-boundary-regime-kalman) ----------


def test_kalman_no_sigma_b_inflation_at_boundary() -> None:
    """Drive a synthetic path toward |x|=5 at constant true σ_b; the
    filter's Δx̂ variance (which the EM reads as σ̂_b²) must stay within
    2× of the truth in the boundary regime, not 30-70% inflated per the
    pre-fix pathology (see ``project_boundary_regime_em_inflation.md``).

    Pre-fix mechanism: inside p ∈ [p_low, p_high] the ``_effective_R``
    blend computes ``r_ukf = p_var / S'(x)²``. At |x|>4.5 the Jacobian
    S'(x) ≈ p(1-p) is O(1e-3) or smaller, so ``r_ukf`` explodes to
    ~P_pred / ε range, K collapses to near zero, the filter lags the
    observation, and each re-entry to the interior regime produces a
    catch-up burst in Δx̂ that the downstream EM reads as elevated σ_b².
    """
    rng = np.random.default_rng(100)
    n = 2000
    sigma_b = 0.026  # matches the paper §6 gate regime
    sigma_eta = 0.005

    # Mean-reverting process that parks ~5 and oscillates with the
    # diffusion. Under a pure RW the path rarely returns once past |x|=4;
    # we want both entries AND exits from the boundary region because the
    # pathology is specifically in the filter catch-up when p re-crosses
    # back into the interior (KF gain recovers, Δx̂ burst — the EM reads
    # the burst as elevated σ̂_b).
    x_true = np.empty(n)
    x_true[0] = 0.0
    target = 4.8
    half_life = 300.0  # steps; gentle pull so there's meaningful path variance
    mean_rev = math.log(2.0) / half_life
    for t in range(1, n):
        pull = -mean_rev * (x_true[t - 1] - target)
        x_true[t] = x_true[t - 1] + pull + rng.normal(0.0, sigma_b)
    y_obs = x_true + rng.normal(0.0, sigma_eta, size=n)

    # Sanity: path must enter the boundary regime and also leave it at
    # least a few times; otherwise we're not testing the catch-up path.
    in_boundary_true = np.abs(x_true) >= 4.0
    crossings = int(np.sum(np.diff(in_boundary_true.astype(int)) != 0))
    assert in_boundary_true.sum() >= 300 and crossings >= 4, (
        f"test setup: path didn't oscillate enough across |x|=4 "
        f"(in-boundary={in_boundary_true.sum()}, crossings={crossings})"
    )

    kf = KalmanFilter(
        token_id="t",
        microstruct=ConstantMicrostruct(sigma_eta * sigma_eta),
        sigma_b=sigma_b,
    )
    t0 = datetime(2026, 4, 24, tzinfo=UTC)
    x_hat = np.empty(n)
    for t in range(n):
        ts = t0 + timedelta(seconds=t)
        p_obs = max(1e-5, min(1.0 - 1e-5, _sigmoid(float(y_obs[t]))))
        state = kf.step(_cm(ts, float(y_obs[t])), _book(p_obs))
        x_hat[t] = state.x_hat

    dx = np.diff(x_hat)
    # Measure empirical σ̂_b in the boundary regime (|x̂|≥4.0). The EM
    # estimates σ_b from Δx̂; if the filter's steady-state Δx̂ std in the
    # boundary is ≥2× truth, the gate gets the 10× MSE blowup.
    in_boundary = np.abs(x_hat[:-1]) >= 4.0
    n_boundary = int(in_boundary.sum())
    assert n_boundary >= 150, (
        f"test setup: only {n_boundary} steps with |x̂|≥4.0, need ≥150 "
        "for a meaningful σ̂_b estimate"
    )
    emp_sigma_b = float(np.std(dx[in_boundary]))
    inflation = emp_sigma_b / sigma_b
    assert inflation < 2.0, (
        f"σ̂_b inflation in boundary regime: empirical "
        f"{emp_sigma_b:.4f} is {inflation:.2f}× truth {sigma_b:.4f}. "
        "The _effective_R blend is amplifying filter noise via the "
        "1/S'(x)² term — saturate or cap r_ukf (see commit message)."
    )
