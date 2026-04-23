"""E-step of the EM loop: mixture posteriors on filtered increments (paper §5.2).

Given a stream of filtered logit increments ``Δx̂_t = x̂_t - x̂_{t-1}`` from
:mod:`blksch.core.filter.kalman`, compute the per-step posterior
``γ_t = P(jump at t | Δx̂_t, θ)`` under a Gaussian+jump mixture::

    Δx̂_t | γ_t = 0 ~ N(μ·Δt, σ_b²·Δt)     (diffusion regime)
    Δx̂_t | γ_t = 1 ~ N(0, s_J²)            (jump regime)
    P(γ_t = 1)     = λ·Δt                 (Bernoulli jump indicator)

Outputs
    γ_t ∈ [0, 1] per-increment jump-posterior probability.

The M-step lives in :mod:`blksch.core.em.jumps` (updates ``σ_b, s_J, λ``
from the γ posteriors) and :mod:`blksch.core.em.rn_drift` (updates the
drift ``μ`` under the risk-neutral restriction).

Frozen contract — do NOT change the shape without coordinating with
``em/jumps.py`` and ``em/rn_drift.py`` which import ``MixtureParams``::

    @dataclass(frozen=True)
    class MixtureParams:
        sigma_b: float        # diffusion vol (per √time)
        s_J: float            # jump size standard deviation
        lambda_jump: float    # jump rate (per second)
        mu: float             # RN drift — constant for now; em/rn_drift.py
                              # will upgrade to μ(t, x)

All numerical work is done in log-space with log-sum-exp so a large ``Δt``
(e.g. a 600 s data gap) does not under/overflow the Gaussian kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy.special import logsumexp

if TYPE_CHECKING:
    from blksch.schemas import LogitState

DEFAULT_JUMP_THRESHOLD = 0.7
LOG_TWO_PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Frozen contract — shared with em/jumps.py and em/rn_drift.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureParams:
    """Gaussian+jump mixture parameters (paper §5.2).

    Frozen interface — the M-step modules import this and return fresh
    instances; they do NOT add fields or rename them without cross-track
    coordination.
    """

    sigma_b: float
    s_J: float
    lambda_jump: float
    mu: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_b < 0:
            raise ValueError("sigma_b must be non-negative")
        if self.s_J < 0:
            raise ValueError("s_J must be non-negative")
        if self.lambda_jump < 0:
            raise ValueError("lambda_jump must be non-negative")
        if not np.isfinite(self.mu):
            raise ValueError("mu must be finite")


# ---------------------------------------------------------------------------
# Kernels (pure, exported for tests)
# ---------------------------------------------------------------------------


def gaussian_log_pdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log Gaussian PDF, vectorized. ``var`` must be strictly positive."""
    var = np.asarray(var, dtype=float)
    mean = np.asarray(mean, dtype=float)
    x = np.asarray(x, dtype=float)
    var = np.maximum(var, np.finfo(float).tiny)  # floor to tiniest positive
    return -0.5 * (LOG_TWO_PI + np.log(var) + (x - mean) ** 2 / var)


def _coerce_dts(dts: Union[float, np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if np.isscalar(dts) or np.ndim(dts) == 0:
        return np.full(shape, float(dts), dtype=float)
    arr = np.asarray(dts, dtype=float)
    if arr.shape != shape:
        raise ValueError(
            f"dts shape {arr.shape} does not match increments shape {shape}"
        )
    return arr


# ---------------------------------------------------------------------------
# E-step
# ---------------------------------------------------------------------------


def compute_posteriors(
    increments: np.ndarray | list[float],
    dts: Union[float, np.ndarray, list[float]],
    params: MixtureParams,
) -> np.ndarray:
    """Compute ``γ_t = P(jump | Δx̂_t, params)`` for every increment.

    Edge cases:
      * ``lambda_jump == 0`` or ``s_J == 0`` → returns all zeros (pure diffusion).
      * ``dt == 0`` at some index → ``γ_t = 0`` (no time elapsed means no
        Bernoulli trial).
      * ``lambda * dt`` may exceed 1 for very long gaps; we clip to
        ``[0, 1]`` and treat ``λΔt = 1`` as a saturated jump prior.
    """
    inc = np.asarray(increments, dtype=float)
    if inc.ndim != 1:
        raise ValueError(f"increments must be 1-D, got shape {inc.shape}")
    dt_arr = _coerce_dts(dts, inc.shape)
    if np.any(dt_arr < 0):
        raise ValueError("dts must be non-negative")

    # Pure-diffusion shortcut — no need for LSE.
    if params.lambda_jump <= 0 or params.s_J <= 0:
        return np.zeros_like(inc)

    p_jump = np.clip(params.lambda_jump * dt_arr, 0.0, 1.0)

    # Gaussian log-densities under each regime.
    var_phi = np.maximum(params.sigma_b**2 * dt_arr, np.finfo(float).tiny)
    var_psi = max(params.s_J**2, np.finfo(float).tiny)
    log_phi = gaussian_log_pdf(inc, params.mu * dt_arr, var_phi)
    log_psi = gaussian_log_pdf(inc, np.zeros_like(inc), np.full_like(inc, var_psi))

    # Mixture posterior in log space:
    #   log a = log(λΔt) + log_psi
    #   log b = log(1-λΔt) + log_phi
    #   γ    = exp(log a - log(a + b))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_a = np.where(p_jump > 0, np.log(p_jump) + log_psi, -np.inf)
        log_b = np.where(p_jump < 1, np.log1p(-p_jump) + log_phi, -np.inf)
    stacked = np.stack([log_a, log_b], axis=0)
    log_den = logsumexp(stacked, axis=0)
    # dt == 0 → both p_jump == 0 and we want γ = 0.
    gamma = np.where(dt_arr == 0, 0.0, np.exp(log_a - log_den))
    gamma = np.where(np.isnan(gamma), 0.0, gamma)
    return np.clip(gamma, 0.0, 1.0)


def log_likelihood(
    increments: np.ndarray | list[float],
    dts: Union[float, np.ndarray, list[float]],
    params: MixtureParams,
) -> float:
    """Mixture log-likelihood ``Σ_t log p(Δx̂_t | params)``.

    Used for the M-step's monotonicity check and for external EM loops that
    want to monitor convergence.
    """
    inc = np.asarray(increments, dtype=float)
    if inc.ndim != 1:
        raise ValueError("increments must be 1-D")
    dt_arr = _coerce_dts(dts, inc.shape)

    var_phi = np.maximum(params.sigma_b**2 * dt_arr, np.finfo(float).tiny)
    log_phi = gaussian_log_pdf(inc, params.mu * dt_arr, var_phi)

    if params.lambda_jump <= 0 or params.s_J <= 0:
        return float(np.sum(log_phi))

    p_jump = np.clip(params.lambda_jump * dt_arr, 0.0, 1.0)
    var_psi = max(params.s_J**2, np.finfo(float).tiny)
    log_psi = gaussian_log_pdf(inc, np.zeros_like(inc), np.full_like(inc, var_psi))

    with np.errstate(divide="ignore", invalid="ignore"):
        log_a = np.where(p_jump > 0, np.log(p_jump) + log_psi, -np.inf)
        log_b = np.where(p_jump < 1, np.log1p(-p_jump) + log_phi, -np.inf)
    stacked = np.stack([log_a, log_b], axis=0)
    log_p = logsumexp(stacked, axis=0)
    return float(np.sum(log_p))


def mark_jumps(
    gamma: np.ndarray,
    *,
    threshold: float = DEFAULT_JUMP_THRESHOLD,
) -> np.ndarray:
    """Boolean mask where ``γ_t > threshold`` (jump-dominant increments).

    Paper §5.2 uses τ_J=0.7 by default. Exposed as a helper so downstream
    ``surface/corr.py`` and diagnostics can agree on the cutoff.
    """
    g = np.asarray(gamma, dtype=float)
    return g > threshold


# ---------------------------------------------------------------------------
# PosteriorResult + e_step — convenience wrappers for the EM loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorResult:
    """Output of :func:`e_step` — everything the M-step modules consume.

    Shapes:
      * ``increments``, ``dts``, ``gamma`` are parallel 1-D arrays of length
        ``N-1`` given ``N`` input states (one per adjacent pair).
      * ``end_timestamps`` carries the timestamp of the state that **ends**
        each increment, so ``jumps.m_step_jumps`` can attribute jump events
        to their occurrence time without re-taking the state list.
      * ``params`` is the :class:`MixtureParams` instance used to compute
        ``gamma`` — M-step modules use it to know what they are refining.
    """

    increments: np.ndarray
    dts: np.ndarray
    gamma: np.ndarray
    end_timestamps: tuple[datetime, ...]
    params: MixtureParams

    def __post_init__(self) -> None:
        n = self.increments.shape[0]
        if self.dts.shape[0] != n or self.gamma.shape[0] != n:
            raise ValueError(
                "increments / dts / gamma length mismatch: "
                f"{self.increments.shape}, {self.dts.shape}, {self.gamma.shape}"
            )
        if len(self.end_timestamps) != n:
            raise ValueError(
                f"end_timestamps length {len(self.end_timestamps)} != n={n}"
            )

    @property
    def n(self) -> int:
        return int(self.increments.shape[0])


def e_step(
    states: list["LogitState"],
    params: MixtureParams,
) -> PosteriorResult:
    """Run the E-step on a sequence of :class:`LogitState`.

    Extracts ``Δx̂_t`` and ``Δt_t`` from adjacent states, computes
    ``γ_t = compute_posteriors(...)``, and packs everything into a
    :class:`PosteriorResult`. Returns an empty result for runs shorter
    than two states.
    """
    if len(states) < 2:
        return PosteriorResult(
            increments=np.zeros(0, dtype=float),
            dts=np.zeros(0, dtype=float),
            gamma=np.zeros(0, dtype=float),
            end_timestamps=tuple(),
            params=params,
        )
    x = np.array([s.x_hat for s in states], dtype=float)
    ts = [s.ts for s in states]
    increments = np.diff(x)
    dts = np.array([(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)], dtype=float)
    gamma = compute_posteriors(increments, dts, params)
    return PosteriorResult(
        increments=increments,
        dts=dts,
        gamma=gamma,
        end_timestamps=tuple(ts[1:]),
        params=params,
    )


__all__ = [
    "DEFAULT_JUMP_THRESHOLD",
    "MixtureParams",
    "PosteriorResult",
    "compute_posteriors",
    "e_step",
    "gaussian_log_pdf",
    "log_likelihood",
    "mark_jumps",
]
