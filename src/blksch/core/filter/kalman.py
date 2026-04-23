"""Heteroskedastic Kalman filter for x̂_t in logit space (paper §5.1).

State model (random walk on the logit)::

    x_{t+1} = x_t + w_t,   w_t ~ N(0, Q_t),   Q_t = σ_b²·dt

Observation model::

    y_t = logit(p̃_t) = x_t + η_t,   η_t ~ N(0, R_t)

Where:
  * ``σ_b`` is the belief volatility — accepted as a constructor argument
    for now; the EM loop in ``em/rn_drift.py`` will supply it in production.
  * ``R_t = σ_η²(t)`` is produced by a :class:`MicrostructModel`
    (paper §5.1 eq (10)), which the KF pulls per step through the
    :class:`VarianceOracle` protocol. The model is told whether the tick
    was forward-filled so it can widen R_t.
  * ``σ_η²`` surfaced in the emitted :class:`LogitState` is the raw
    microstruct output — the internal UKF-adjusted R used for the update
    is not published.

Near the p ∈ {0, 1} boundary the logit map's Jacobian S'(x) = p(1-p) → 0,
so small changes in x produce tiny changes in p and a calibrated σ_η²
in x-space can understate the effective measurement variance. In the
boundary region (``p ∈ [0, p_low] ∪ [p_high, 1]``, defaults 0.02/0.98)
we run a scalar UKF: three sigma points in x-space are mapped through
S(x), a p-space predictive variance is recovered, and the corresponding
x-space variance is used when it exceeds the microstruct R. The blend
weight is quadratic in the margin-to-edge so the estimate is continuous
at the boundary.

Divergence safeguards:
  * Kalman gain K_t is clipped to [0, 1].
  * Innovation variance has a configurable floor (default 1e-8).
  * Posterior variance is floored at the same value.

Interface (minimal)::

    kf = KalmanFilter(token_id=tid, microstruct=model, sigma_b=0.3)
    for cm in canonical_mid_stream:
        state = kf.step(cm, book, trades)  # -> LogitState
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from blksch.core.filter.canonical_mid import CanonicalMid
from blksch.schemas import BookSnap, LogitState, TradeTick

logger = logging.getLogger(__name__)

DEFAULT_BOUNDARY_P_LOW = 0.02
DEFAULT_BOUNDARY_P_HIGH = 0.98
DEFAULT_INNOVATION_FLOOR = 1.0e-8
DEFAULT_INITIAL_VARIANCE = 1.0
DEFAULT_SIGMA_B = 0.3
DEFAULT_MAX_DT_SEC = 60.0  # cap dt so a long gap doesn't blow up Q


class VarianceOracle(Protocol):
    """A model that supplies σ_η² for a (book, trades, forward_filled) tuple."""

    def variance(
        self,
        book: BookSnap,
        trades: Sequence[TradeTick] = (),
        *,
        forward_filled: bool = False,
    ) -> float: ...


def _sigmoid(x: float) -> float:
    """Numerically stable logistic."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class KalmanFilter:
    """Scalar heteroskedastic KF with a continuous UKF augmentation at the boundary.

    Attributes exposed for diagnostics after :meth:`step`:
      * :attr:`x_hat` — posterior mean (same as the LogitState).
      * :attr:`posterior_variance` — posterior P_t (not in the schema).
      * :attr:`last_innovation` — y_t − x̂_t⁻ (pre-update).
      * :attr:`last_innovation_variance` — S_t = P_t⁻ + R_eff.
      * :attr:`last_R_effective` — R used in the update (may be inflated).
      * :attr:`last_K` — Kalman gain after clipping.
    """

    token_id: str
    microstruct: VarianceOracle
    sigma_b: float = DEFAULT_SIGMA_B
    initial_x: float | None = None
    initial_variance: float = DEFAULT_INITIAL_VARIANCE
    boundary_p_low: float = DEFAULT_BOUNDARY_P_LOW
    boundary_p_high: float = DEFAULT_BOUNDARY_P_HIGH
    innovation_variance_floor: float = DEFAULT_INNOVATION_FLOOR
    max_dt_sec: float = DEFAULT_MAX_DT_SEC

    _x: float = field(init=False, default=0.0, repr=False)
    _P: float = field(init=False, default=1.0, repr=False)
    _last_ts: datetime | None = field(init=False, default=None, repr=False)
    _initialized: bool = field(init=False, default=False, repr=False)
    _last_innovation: float | None = field(init=False, default=None, repr=False)
    _last_innovation_variance: float | None = field(init=False, default=None, repr=False)
    _last_R_effective: float | None = field(init=False, default=None, repr=False)
    _last_K: float | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.sigma_b < 0:
            raise ValueError("sigma_b must be non-negative")
        if self.initial_variance <= 0:
            raise ValueError("initial_variance must be positive")
        if self.innovation_variance_floor <= 0:
            raise ValueError("innovation_variance_floor must be positive")
        if not 0.0 < self.boundary_p_low < 0.5:
            raise ValueError("boundary_p_low must be in (0, 0.5)")
        if not 0.5 < self.boundary_p_high < 1.0:
            raise ValueError("boundary_p_high must be in (0.5, 1)")
        if self.max_dt_sec <= 0:
            raise ValueError("max_dt_sec must be positive")
        self._P = self.initial_variance
        if self.initial_x is not None:
            self._x = self.initial_x
            self._initialized = True

    # -- read-only diagnostics ------------------------------------------------

    @property
    def x_hat(self) -> float:
        return self._x

    @property
    def posterior_variance(self) -> float:
        return self._P

    @property
    def last_innovation(self) -> float | None:
        return self._last_innovation

    @property
    def last_innovation_variance(self) -> float | None:
        return self._last_innovation_variance

    @property
    def last_R_effective(self) -> float | None:
        return self._last_R_effective

    @property
    def last_K(self) -> float | None:
        return self._last_K

    # -- main step ------------------------------------------------------------

    def step(
        self,
        canonical_mid: CanonicalMid,
        book: BookSnap,
        trades: Sequence[TradeTick] = (),
    ) -> LogitState:
        """Run one predict + update and return the emitted LogitState.

        ``book`` and ``trades`` feed :meth:`VarianceOracle.variance` to
        compute R_t; the filter does not retain them.
        """
        y = canonical_mid.y
        ts = canonical_mid.ts
        p = canonical_mid.p_tilde
        ff = canonical_mid.forward_filled

        R_raw = self.microstruct.variance(book, trades, forward_filled=ff)
        R_raw = max(R_raw, self.innovation_variance_floor)

        # First-tick initialization: anchor to observation.
        if not self._initialized:
            self._x = y
            self._P = R_raw
            self._last_ts = ts
            self._initialized = True
            self._last_innovation = 0.0
            self._last_innovation_variance = R_raw
            self._last_R_effective = R_raw
            self._last_K = 1.0
            return LogitState(
                token_id=self.token_id,
                x_hat=self._x,
                sigma_eta2=R_raw,
                ts=ts,
            )

        dt = (ts - self._last_ts).total_seconds() if self._last_ts is not None else 1.0
        if dt <= 0.0:
            dt = 1e-6
        if dt > self.max_dt_sec:
            dt = self.max_dt_sec
        Q = self.sigma_b * self.sigma_b * dt

        # Predict
        x_pred = self._x
        P_pred = self._P + Q

        # UKF-augmented R near the boundary.
        R_eff = self._effective_R(x_pred, P_pred, R_raw, p)

        # Update
        innov_var = P_pred + R_eff
        if innov_var < self.innovation_variance_floor:
            innov_var = self.innovation_variance_floor
        K = P_pred / innov_var
        # Gain clip — belt and suspenders for numerical corner cases.
        if K < 0.0:
            K = 0.0
        elif K > 1.0:
            K = 1.0
        innovation = y - x_pred
        self._x = x_pred + K * innovation
        self._P = (1.0 - K) * P_pred
        if self._P < self.innovation_variance_floor:
            self._P = self.innovation_variance_floor
        self._last_ts = ts
        self._last_innovation = innovation
        self._last_innovation_variance = innov_var
        self._last_R_effective = R_eff
        self._last_K = K

        return LogitState(
            token_id=self.token_id,
            x_hat=self._x,
            sigma_eta2=R_raw,
            ts=ts,
        )

    # -- UKF augmentation -----------------------------------------------------

    def _effective_R(
        self,
        x_pred: float,
        P_pred: float,
        R_raw: float,
        p: float,
    ) -> float:
        """Blend raw R with a UKF-inflated R near the boundary.

        ``alpha = 0`` far from the edge → returns R_raw.
        ``alpha → 1`` at p ∈ {0, 1} → returns max(R_raw, R_ukf).
        Blend weight is quadratic in the margin-to-edge so the posterior
        mean transitions continuously as p moves through p_low / p_high.
        """
        margin = min(p, 1.0 - p)
        threshold = min(self.boundary_p_low, 1.0 - self.boundary_p_high)
        if margin >= threshold:
            return R_raw

        rel = (threshold - margin) / threshold  # 0 at edge-of-interior → 1 at p∈{0,1}
        alpha = rel * rel

        # Scaled unscented transform, scalar state, κ=2 → λ=2.
        n = 1
        kappa = 2.0
        lam = kappa
        scale = math.sqrt(lam + n)
        sigma = math.sqrt(max(P_pred, 0.0))
        chi0 = x_pred
        chi_plus = x_pred + scale * sigma
        chi_minus = x_pred - scale * sigma
        w0 = lam / (lam + n)
        w_side = 0.5 / (lam + n)

        p_chi0 = _sigmoid(chi0)
        p_plus = _sigmoid(chi_plus)
        p_minus = _sigmoid(chi_minus)
        p_mean = w0 * p_chi0 + w_side * (p_plus + p_minus)
        p_var = (
            w0 * (p_chi0 - p_mean) ** 2
            + w_side * (p_plus - p_mean) ** 2
            + w_side * (p_minus - p_mean) ** 2
        )

        # Translate p-space variance back to x-space via the Jacobian at the
        # predicted mean. Near total saturation S'(x) collapses; widen R a lot
        # so the gain clip takes over.
        dpdx = p_chi0 * (1.0 - p_chi0)
        if dpdx < 1e-8:
            r_ukf = R_raw * 1.0e6
        else:
            r_ukf = p_var / (dpdx * dpdx)

        return (1.0 - alpha) * R_raw + alpha * max(R_raw, r_ukf)


__all__ = [
    "DEFAULT_BOUNDARY_P_HIGH",
    "DEFAULT_BOUNDARY_P_LOW",
    "DEFAULT_INITIAL_VARIANCE",
    "DEFAULT_INNOVATION_FLOOR",
    "DEFAULT_MAX_DT_SEC",
    "DEFAULT_SIGMA_B",
    "KalmanFilter",
    "VarianceOracle",
]
