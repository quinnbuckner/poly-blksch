"""Heteroskedastic measurement-noise model (paper §5.1 eq 10).

Implements::

    sigma_eta^2(t) = a_0 + a_1 * (spread/2)^2
                        + a_2 * (1/depth)
                        + a_3 * |trade_rate|
                        + a_4 * |imbalance|

fit by OLS (optionally ridge-regularized) on empirical squared
logit-innovations over a rolling window. At serve time the model produces a
per-tick variance that the Kalman filter uses as R_t. When the upstream
``CanonicalMid`` is ``forward_filled=True`` — i.e. no fresh book arrived in
this grid bin — the model widens sigma^2 by a multiplicative factor
(default 10x) so the Kalman learns that a stale tick carries less
information.

The module deliberately does not pre-compute squared innovations; the caller
(EM loop) supplies them along with the per-tick feature vector. This keeps
the regression decoupled from any particular x_t estimator, so the same
fit path works against (a) rough first-pass ``(Δy_t)^2`` proxies during
bootstrap and (b) refined KF residuals after convergence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from blksch.schemas import BookSnap

if TYPE_CHECKING:  # imported for signatures only
    from blksch.schemas import TradeTick


DEFAULT_FIT_WINDOW_SEC = 400.0
DEFAULT_DEPTH_LEVELS = 5
DEFAULT_TRADE_RATE_WINDOW_SEC = 30.0
DEFAULT_FORWARD_FILL_WIDEN_FACTOR = 10.0
DEFAULT_SIGMA_FLOOR = 1.0e-6
DEFAULT_RIDGE = 1.0e-8

N_COEFFICIENTS = 5  # [a0 (intercept), a1, a2, a3, a4]


# ---------------------------------------------------------------------------
# Config & feature row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrostructConfig:
    """Knobs for fit and serve. Defaults match ``config/bot.yaml``."""

    fit_window_sec: float = DEFAULT_FIT_WINDOW_SEC
    depth_levels: int = DEFAULT_DEPTH_LEVELS
    trade_rate_window_sec: float = DEFAULT_TRADE_RATE_WINDOW_SEC
    forward_fill_widen_factor: float = DEFAULT_FORWARD_FILL_WIDEN_FACTOR
    sigma_floor: float = DEFAULT_SIGMA_FLOOR
    ridge: float = DEFAULT_RIDGE

    def __post_init__(self) -> None:
        if self.fit_window_sec <= 0:
            raise ValueError("fit_window_sec must be positive")
        if self.depth_levels < 1:
            raise ValueError("depth_levels must be >= 1")
        if self.trade_rate_window_sec <= 0:
            raise ValueError("trade_rate_window_sec must be positive")
        if self.sigma_floor <= 0:
            raise ValueError("sigma_floor must be positive")
        if self.forward_fill_widen_factor < 1:
            raise ValueError("forward_fill_widen_factor must be >= 1")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")


@dataclass(frozen=True)
class MicrostructFeatures:
    """Paper eq (10) covariates for one tick.

    Kept as a dataclass (and not a bare tuple) so downstream diagnostics can
    log them.
    """

    half_spread: float
    inv_depth: float
    abs_trade_rate: float
    abs_imbalance: float

    def as_row(self) -> list[float]:
        """[1, (spread/2)^2, 1/depth, |rate|, |imbalance|] — intercept leads."""
        return [
            1.0,
            self.half_spread * self.half_spread,
            self.inv_depth,
            self.abs_trade_rate,
            self.abs_imbalance,
        ]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    book: BookSnap,
    trades: Sequence["TradeTick"] | Iterable["TradeTick"] = (),
    *,
    depth_levels: int = DEFAULT_DEPTH_LEVELS,
    trade_rate_window_sec: float = DEFAULT_TRADE_RATE_WINDOW_SEC,
) -> MicrostructFeatures:
    """Build the per-tick feature row from a book + recent trades window.

    ``trades`` is the pre-filtered sliding window (the caller — typically
    ``canonical_mid`` — maintains it). ``trade_rate`` is ``len(trades) /
    trade_rate_window_sec``; a missing/zero-size book yields zero-covariates
    rather than raising so the caller can still fit, with the intercept
    carrying the base rate.
    """
    if book.bids and book.asks:
        spread = book.asks[0].price - book.bids[0].price
    else:
        spread = 0.0
    half_spread = max(0.0, spread / 2.0)

    bid_depth = sum(lv.size for lv in book.bids[:depth_levels])
    ask_depth = sum(lv.size for lv in book.asks[:depth_levels])
    total_depth = bid_depth + ask_depth
    if total_depth > 0:
        inv_depth = 1.0 / total_depth
        imbalance = (bid_depth - ask_depth) / total_depth
    else:
        inv_depth = 0.0
        imbalance = 0.0

    # ``trades`` may be an iterator (consumed once). Materialize only the
    # count — we don't need the individual ticks here.
    n_trades = sum(1 for _ in trades)
    trade_rate = n_trades / trade_rate_window_sec if trade_rate_window_sec > 0 else 0.0

    return MicrostructFeatures(
        half_spread=half_spread,
        inv_depth=inv_depth,
        abs_trade_rate=abs(trade_rate),
        abs_imbalance=abs(imbalance),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class MicrostructModel:
    """Fit coefficients + serve-time variance API.

    Construct via :meth:`fit` / :meth:`fit_from_features` or directly with
    known coefficients (useful for tests and serialization round-trips).
    """

    a0: float
    a1: float
    a2: float
    a3: float
    a4: float
    config: MicrostructConfig = field(default_factory=MicrostructConfig)

    # -- serve-time -----------------------------------------------------------

    def variance_from_features(
        self,
        features: MicrostructFeatures,
        *,
        forward_filled: bool = False,
    ) -> float:
        """Compute sigma_eta^2 from a pre-built feature row."""
        sigma2 = (
            self.a0
            + self.a1 * features.half_spread * features.half_spread
            + self.a2 * features.inv_depth
            + self.a3 * features.abs_trade_rate
            + self.a4 * features.abs_imbalance
        )
        if sigma2 < self.config.sigma_floor:
            sigma2 = self.config.sigma_floor
        if forward_filled:
            sigma2 *= self.config.forward_fill_widen_factor
        return float(sigma2)

    def variance(
        self,
        book: BookSnap,
        trades: Sequence["TradeTick"] | Iterable["TradeTick"] = (),
        *,
        forward_filled: bool = False,
    ) -> float:
        """Compute sigma_eta^2 for a live (book, trades-window, ff-flag) tuple."""
        features = extract_features(
            book,
            trades,
            depth_levels=self.config.depth_levels,
            trade_rate_window_sec=self.config.trade_rate_window_sec,
        )
        return self.variance_from_features(features, forward_filled=forward_filled)

    # -- fit ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        feature_rows: Sequence[Sequence[float]] | np.ndarray,
        squared_innovations: Sequence[float] | np.ndarray,
        config: MicrostructConfig | None = None,
    ) -> MicrostructModel:
        """OLS (ridge) fit of ``squared_innovations ~ feature_rows``.

        ``feature_rows`` must be ``(N, 5)`` shaped: columns are
        ``[1, (spread/2)^2, 1/depth, |rate|, |imbalance|]``.
        """
        cfg = config or MicrostructConfig()
        X = np.asarray(feature_rows, dtype=float)
        y = np.asarray(squared_innovations, dtype=float)
        if X.ndim != 2 or X.shape[1] != N_COEFFICIENTS:
            raise ValueError(
                f"feature_rows must be (N, {N_COEFFICIENTS}); got {X.shape}"
            )
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"squared_innovations shape {y.shape} does not match "
                f"feature_rows row count {X.shape[0]}"
            )
        if X.shape[0] < N_COEFFICIENTS:
            raise ValueError(
                f"need >= {N_COEFFICIENTS} observations; got {X.shape[0]}"
            )

        # Ridge-regularized normal equations: (X'X + λI) a = X'y.
        xtx = X.T @ X + cfg.ridge * np.eye(N_COEFFICIENTS)
        xty = X.T @ y
        a = np.linalg.solve(xtx, xty)
        return cls(
            a0=float(a[0]),
            a1=float(a[1]),
            a2=float(a[2]),
            a3=float(a[3]),
            a4=float(a[4]),
            config=cfg,
        )

    @classmethod
    def fit_from_features(
        cls,
        features: Sequence[MicrostructFeatures],
        squared_innovations: Sequence[float] | np.ndarray,
        config: MicrostructConfig | None = None,
    ) -> MicrostructModel:
        """Convenience wrapper around :meth:`fit` that builds the design matrix."""
        rows = [f.as_row() for f in features]
        return cls.fit(rows, squared_innovations, config=config)


__all__ = [
    "DEFAULT_FIT_WINDOW_SEC",
    "DEFAULT_FORWARD_FILL_WIDEN_FACTOR",
    "DEFAULT_SIGMA_FLOOR",
    "MicrostructConfig",
    "MicrostructFeatures",
    "MicrostructModel",
    "N_COEFFICIENTS",
    "extract_features",
]
