"""Tensor B-spline surface smoother over (τ, m) — paper §5.3.

Input: a bag of noisy `SurfacePoint` observations for a single token id,
scattered across (τ, m) coordinates. Output: a callable `(τ, m) → SurfacePoint`
that returns a smoothed estimate subject to shape constraints:

  * boundedness: σ̂_b ≥ 0, λ̂ ≥ 0, ŝ²_J ≥ 0   (hard floor at 0)
  * monotonicity in τ: σ̂_b(τ, m) non-decreasing in τ for fixed m
    (variance should not shrink as you look further ahead for a
    well-behaved market — paper §5.3 "term smoothness")

Strategy:
    unconstrained tensor-product bicubic spline fit  (scipy.RectBivariateSpline)
         ↓
    post-projection onto the feasible set:
       - clip to 0 on the floor (element-wise)
       - sweep along τ for each m grid column and enforce a running-max
         cumulative projection (cheap L∞-closest monotone sequence)

This matches paper §5.3's suggestion ("shape-constrained QP" → if hard, do
post-projection and document the approximation). The post-projection is:
  * exact-monotone in the evaluation grid the smoother was fit on
  * O(1) per query afterwards (we cache the projected surfaces)

Uncertainty: we carry the fit residual std per field as a single scalar
attached to every returned SurfacePoint (no per-cell variance — the
smoother would need a full GP for that). Track B reads `uncertainty` only
for telemetry, never for sizing, so this is enough.

Pure: no I/O, no asyncio. SciPy is the only dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import RectBivariateSpline, SmoothBivariateSpline

from blksch.schemas import SurfacePoint

__all__ = [
    "SurfaceSmootherParams",
    "SurfaceSmoother",
]


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceSmootherParams:
    """Knobs for the tensor B-spline fit + post-projection.

    - n_tau, n_m:          size of the internal evaluation grid (spans the
                           observed (τ, m) range). Larger = smoother but
                           slower and more prone to boundary artifacts.
    - smoothing:           SciPy's `s` parameter — 0 means exact interpolation,
                           positive values trade fidelity for smoothness. We
                           default to a mild positive to denoise EM outputs.
    - degree:              spline degree in each axis (kx, ky). Cubic = 3.
    - enforce_monotone_tau: project onto non-decreasing-in-τ along each
                           m column. Set False to turn the monotonicity
                           constraint off (useful for pathological markets).
    - nonneg_fields:       fields on the surface that must be clipped to ≥0
                           (sigma_b, lambda_, s2_j).
    - min_points_per_axis: fewer than this many unique τ (or m) values ⇒ fall
                           back to 1-D smoothing on the other axis.
    """

    n_tau: int = 20
    n_m: int = 15
    smoothing: float = 0.0        # 0 = exact tensor fit; SciPy default
    degree: int = 3
    enforce_monotone_tau: bool = True
    nonneg_fields: tuple[str, ...] = ("sigma_b", "lambda_", "s2_j")
    min_points_per_axis: int = 4

    def __post_init__(self) -> None:
        if self.n_tau < 2 or self.n_m < 2:
            raise ValueError("n_tau and n_m must be ≥ 2")
        if self.degree not in (1, 3, 5):
            raise ValueError(f"degree must be 1, 3, or 5 (got {self.degree})")
        if self.smoothing < 0.0:
            raise ValueError(f"smoothing must be ≥ 0 (got {self.smoothing})")
        if self.min_points_per_axis < 2:
            raise ValueError("min_points_per_axis must be ≥ 2")


# ---------------------------------------------------------------------------
# Smoother
# ---------------------------------------------------------------------------


_FIELDS: tuple[str, ...] = ("sigma_b", "lambda_", "s2_j")


@dataclass
class _FittedAxis:
    """One fitted field over a shared (τ, m) grid."""

    values: np.ndarray        # shape (n_tau, n_m) after projection
    fallback_1d: tuple[str, np.ndarray, np.ndarray] | None  # axis, xs, ys
    residual_std: float


class SurfaceSmoother:
    """Tensor B-spline smoother with post-projection shape constraints.

    Lifecycle:

        smoother = SurfaceSmoother(token_id="0xA", params=SurfaceSmootherParams())
        smoother.fit(surface_points)
        sp = smoother.evaluate(tau=3600.0, m=0.0)
    """

    def __init__(self, token_id: str, params: SurfaceSmootherParams | None = None) -> None:
        self.token_id = token_id
        self.params = params or SurfaceSmootherParams()
        self._tau_grid: np.ndarray | None = None
        self._m_grid: np.ndarray | None = None
        self._fits: dict[str, _FittedAxis] = {}
        self._latest_ts: datetime | None = None

    # -- fitting ------------------------------------------------------------

    def fit(self, surface_points: Sequence[SurfacePoint] | Iterable[SurfacePoint]) -> None:
        pts = [sp for sp in surface_points if sp.token_id == self.token_id]
        if not pts:
            raise ValueError(f"no SurfacePoints for token_id={self.token_id}")

        taus = np.array([sp.tau for sp in pts], dtype=float)
        ms = np.array([sp.m for sp in pts], dtype=float)
        self._latest_ts = max(sp.ts for sp in pts)

        # Build the evaluation grid spanning the observed support.
        tau_lo, tau_hi = float(taus.min()), float(taus.max())
        m_lo, m_hi = float(ms.min()), float(ms.max())
        if tau_hi == tau_lo:
            tau_hi = tau_lo + 1.0
        if m_hi == m_lo:
            m_hi = m_lo + 1e-3
        self._tau_grid = np.linspace(tau_lo, tau_hi, self.params.n_tau)
        self._m_grid = np.linspace(m_lo, m_hi, self.params.n_m)

        for field_name in _FIELDS:
            values = np.array(
                [_get_field(sp, field_name) for sp in pts], dtype=float
            )
            self._fits[field_name] = self._fit_one(taus, ms, values, field_name)

    def _fit_one(
        self, taus: np.ndarray, ms: np.ndarray, values: np.ndarray, field_name: str,
    ) -> _FittedAxis:
        assert self._tau_grid is not None and self._m_grid is not None
        unique_tau = np.unique(taus).size
        unique_m = np.unique(ms).size
        need = self.params.min_points_per_axis

        fallback: tuple[str, np.ndarray, np.ndarray] | None = None

        if unique_tau < need or unique_m < need:
            # Degenerate along one axis — fall back to 1-D smoothing along
            # whichever axis has enough support, or a constant if neither.
            if unique_tau >= need:
                order = np.argsort(taus)
                fallback = ("tau", taus[order], values[order])
            elif unique_m >= need:
                order = np.argsort(ms)
                fallback = ("m", ms[order], values[order])
            grid_values = _evaluate_1d_or_mean(
                fallback, float(values.mean()), self._tau_grid, self._m_grid,
            )
            pred_at_obs = np.array([
                _evaluate_fallback(fallback, float(values.mean()), t, m)
                for t, m in zip(taus, ms)
            ])
        else:
            grid_values, pred_at_obs = self._tensor_fit_on_grid(
                taus, ms, values, unique_tau, unique_m,
            )

        # -- shape projection -------------------------------------------------
        if field_name in self.params.nonneg_fields:
            np.clip(grid_values, 0.0, None, out=grid_values)

        if self.params.enforce_monotone_tau and field_name == "sigma_b":
            # Running-max along τ axis (axis=0) for each m column.
            grid_values = np.maximum.accumulate(grid_values, axis=0)

        residual_std = float(np.std(values - pred_at_obs)) if values.size > 1 else 0.0

        return _FittedAxis(
            values=grid_values,
            fallback_1d=fallback,
            residual_std=residual_std,
        )

    def _tensor_fit_on_grid(
        self, taus: np.ndarray, ms: np.ndarray, values: np.ndarray,
        unique_tau: int, unique_m: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a 2-D tensor-product spline over (τ, m).

        Two paths:
          * Rectangular input (one observation per (τ_k, m_j) cell, no dupes):
            use RectBivariateSpline on the unique grid — this is exact
            interpolation at the input knots when smoothing==0.
          * Scattered input: use SmoothBivariateSpline which takes
            (x, y, z) arrays and applies penalized least squares directly.

        Returns (values_on_internal_grid, predictions_at_input_points).
        """
        assert self._tau_grid is not None and self._m_grid is not None
        n_unique = unique_tau * unique_m
        rectangular = (
            taus.size == n_unique
            and np.unique(list(zip(taus, ms)), axis=0).shape[0] == n_unique
        )

        if rectangular and self.params.smoothing == 0.0:
            u_tau = np.unique(taus)
            u_m = np.unique(ms)
            # Lay values out on the rectangular knot grid.
            z = np.zeros((u_tau.size, u_m.size), dtype=float)
            tau_idx = {float(t): i for i, t in enumerate(u_tau)}
            m_idx = {float(x): j for j, x in enumerate(u_m)}
            for tau_k, m_k, v_k in zip(taus, ms, values):
                z[tau_idx[float(tau_k)], m_idx[float(m_k)]] = v_k
            kx = min(self.params.degree, u_tau.size - 1)
            ky = min(self.params.degree, u_m.size - 1)
            try:
                spline = RectBivariateSpline(u_tau, u_m, z, kx=kx, ky=ky, s=0.0)
                grid_values = spline(self._tau_grid, self._m_grid)
                pred_at_obs = spline(taus, ms, grid=False)
                return grid_values, pred_at_obs
            except Exception:
                pass  # fall through to scattered path

        # Scattered / smoothing>0: SmoothBivariateSpline penalizes residuals.
        try:
            kx = min(self.params.degree, unique_tau - 1)
            ky = min(self.params.degree, unique_m - 1)
            # Scale default smoothing by the number of points when smoothing==0
            # so SmoothBivariateSpline produces an interpolating spline rather
            # than its auto-chosen fit.
            s = self.params.smoothing
            sb = SmoothBivariateSpline(taus, ms, values, kx=kx, ky=ky, s=s)
            grid_values = sb(self._tau_grid, self._m_grid)
            pred_at_obs = sb(taus, ms, grid=False)
            return grid_values, pred_at_obs
        except Exception:
            # Completely degenerate ⇒ mean fallback.
            mean_v = float(values.mean())
            return (
                np.full((self.params.n_tau, self.params.n_m), mean_v),
                np.full_like(values, mean_v),
            )

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, tau: float, m: float) -> SurfacePoint:
        if self._tau_grid is None or self._m_grid is None:
            raise RuntimeError("evaluate() called before fit()")

        # Clamp the query to the observed support — extrapolation beyond the
        # grid is unsafe for a shape-constrained surface.
        tau_c = float(np.clip(tau, self._tau_grid[0], self._tau_grid[-1]))
        m_c = float(np.clip(m, self._m_grid[0], self._m_grid[-1]))

        out: dict[str, float] = {}
        for field_name, fit in self._fits.items():
            val = self._evaluate_projected(fit, tau_c, m_c)
            if field_name in self.params.nonneg_fields:
                val = max(val, 0.0)
            out[field_name] = val

        # Aggregate uncertainty: max of per-field residual std (conservative).
        uncertainty = max(
            (fit.residual_std for fit in self._fits.values()), default=0.0
        )

        assert self._latest_ts is not None
        return SurfacePoint(
            token_id=self.token_id,
            tau=tau_c,
            m=m_c,
            sigma_b=out["sigma_b"],
            **{"lambda": out["lambda_"]},
            s2_j=out["s2_j"],
            uncertainty=uncertainty,
            ts=self._latest_ts,
        )

    def _evaluate_projected(self, fit: _FittedAxis, tau: float, m: float) -> float:
        """Bilinear-interp into the projected grid (preserves monotonicity)."""
        assert self._tau_grid is not None and self._m_grid is not None
        tg, mg = self._tau_grid, self._m_grid

        # Locate cell.
        i = int(np.searchsorted(tg, tau, side="right") - 1)
        j = int(np.searchsorted(mg, m, side="right") - 1)
        i = max(0, min(i, len(tg) - 2))
        j = max(0, min(j, len(mg) - 2))

        t0, t1 = tg[i], tg[i + 1]
        m0, m1 = mg[j], mg[j + 1]
        a = 0.0 if t1 == t0 else (tau - t0) / (t1 - t0)
        b = 0.0 if m1 == m0 else (m - m0) / (m1 - m0)

        v00 = fit.values[i, j]
        v01 = fit.values[i, j + 1]
        v10 = fit.values[i + 1, j]
        v11 = fit.values[i + 1, j + 1]
        return (
            (1 - a) * (1 - b) * v00
            + (1 - a) * b * v01
            + a * (1 - b) * v10
            + a * b * v11
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_field(sp: SurfacePoint, field_name: str) -> float:
    if field_name == "sigma_b":
        return sp.sigma_b
    if field_name == "lambda_":
        return sp.lambda_
    if field_name == "s2_j":
        return sp.s2_j
    raise KeyError(field_name)


def _evaluate_1d_or_mean(
    fallback: tuple[str, np.ndarray, np.ndarray] | None,
    mean_value: float,
    tau_grid: np.ndarray,
    m_grid: np.ndarray,
) -> np.ndarray:
    """Build a grid of values from a 1-D fallback or a flat mean."""
    n_t, n_m = len(tau_grid), len(m_grid)
    if fallback is None:
        return np.full((n_t, n_m), mean_value)
    axis, xs, ys = fallback
    if axis == "tau":
        interp = np.interp(tau_grid, xs, ys)
        return np.tile(interp[:, None], (1, n_m))
    else:
        interp = np.interp(m_grid, xs, ys)
        return np.tile(interp[None, :], (n_t, 1))


def _evaluate_fallback(
    fallback: tuple[str, np.ndarray, np.ndarray] | None,
    mean_value: float,
    tau: float,
    m: float,
) -> float:
    if fallback is None:
        return mean_value
    axis, xs, ys = fallback
    query = tau if axis == "tau" else m
    return float(np.interp(query, xs, ys))
