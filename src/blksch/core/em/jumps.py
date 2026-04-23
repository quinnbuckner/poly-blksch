"""M-step: jump parameters from posterior-weighted increments (paper §5.2).

Given the E-step output :class:`PosteriorResult` produced by
:func:`blksch.core.em.increments.e_step`, update the jump rate ``λ̂`` and
the jump-size second moment ``ŝ²_J`` by maximum (weighted) likelihood::

    λ̂     = Σ γ_t / T              (T = Σ Δt, total observation span)
    ŝ²_J  = Σ γ_t · (Δx̂_t - μ·Δt)² / Σ γ_t

Plus a bi-power variation diagnostic for the jump/diffusion decomposition:

    BV   = (π/2) · Σ |Δx̂_t| · |Δx̂_{t-1}|       (robust to jumps, paper §5.2)

When ``Σ γ_t ≈ 0`` (all diffusion), ``λ̂`` collapses to 0 and ``ŝ²_J``
stays at the prior ``params.s_J²`` (floored), avoiding a divide-by-zero
in downstream likelihood evaluations.

Outputs a frozen :class:`JumpEstimate` with the updated scalars, the
extracted jump-event timestamps (``γ_t > τ_J``), and the mixture
log-likelihood under the new params — the full EM-iteration monotonicity
test simply compares successive ``log_likelihood`` values.

No I/O; no ``schemas.py`` changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from blksch.core.em.increments import (
    DEFAULT_JUMP_THRESHOLD,
    MixtureParams,
    PosteriorResult,
    log_likelihood,
)
from blksch.schemas import LogitState

DEFAULT_S_J_SQ_FLOOR = 1.0e-8
MU_1 = float(np.sqrt(2.0 / np.pi))  # E[|Z|] for Z ~ N(0, 1)
BIPOWER_SCALE = 1.0 / (MU_1 * MU_1)  # = π / 2


@dataclass(frozen=True)
class JumpEstimate:
    """M-step output. Internal to Track A — not added to ``schemas.py``."""

    lambda_hat: float
    s_J_sq_hat: float
    jump_timestamps: list[datetime]
    log_likelihood: float

    def __post_init__(self) -> None:
        if self.lambda_hat < 0:
            raise ValueError("lambda_hat must be non-negative")
        if self.s_J_sq_hat < 0:
            raise ValueError("s_J_sq_hat must be non-negative")


def m_step_jumps(
    states: list[LogitState],
    posteriors: PosteriorResult,
    params: MixtureParams,
    *,
    threshold: float = DEFAULT_JUMP_THRESHOLD,
    s_J_sq_floor: float = DEFAULT_S_J_SQ_FLOOR,
) -> JumpEstimate:
    """Compute the jump-parameter M-step.

    ``states`` and ``posteriors`` must be aligned — ``posteriors`` should
    be the direct output of :func:`e_step(states, params)`. We accept both
    arguments explicitly because future callers may want to re-score the
    same E-step output against alternative priors.
    """
    if s_J_sq_floor <= 0:
        raise ValueError("s_J_sq_floor must be positive")
    # Alignment check, with the edge case that an empty input has posteriors.n == 0
    # and either 0 or 1 states (not 2+).
    if posteriors.n == 0:
        if len(states) >= 2:
            raise ValueError(
                f"posteriors is empty but got {len(states)} states"
            )
    elif len(states) - 1 != posteriors.n:
        raise ValueError(
            f"state count ({len(states)}) and posteriors length ({posteriors.n}) "
            "disagree — expected len(states) - 1 == posteriors.n"
        )

    gamma = posteriors.gamma
    increments = posteriors.increments
    dts = posteriors.dts

    if posteriors.n == 0:
        return JumpEstimate(
            lambda_hat=0.0,
            s_J_sq_hat=max(params.s_J * params.s_J, s_J_sq_floor),
            jump_timestamps=[],
            log_likelihood=float("-inf"),
        )

    total_time = float(np.sum(dts))
    sum_gamma = float(np.sum(gamma))

    # Empty-jumps branch: caller said nothing-looks-like-a-jump. Hold s_J²
    # at the prior and declare λ̂ = 0; the EM loop may widen s_J elsewhere.
    if sum_gamma <= 0.0 or total_time <= 0.0:
        lambda_hat = 0.0
        s_J_sq_hat = max(params.s_J * params.s_J, s_J_sq_floor)
        jump_ts: list[datetime] = []
    else:
        lambda_raw = sum_gamma / total_time
        dt_min = float(np.min(dts[dts > 0])) if np.any(dts > 0) else 0.0
        lambda_cap = 1.0 / dt_min if dt_min > 0.0 else float("inf")
        lambda_hat = float(min(lambda_raw, lambda_cap))

        # Drift-corrected jump second moment.
        mean_free = increments - params.mu * dts
        num = float(np.sum(gamma * mean_free * mean_free))
        s_J_sq_hat = max(num / sum_gamma, s_J_sq_floor)

        mask = gamma > threshold
        jump_ts = [posteriors.end_timestamps[i] for i in np.where(mask)[0]]

    new_params = MixtureParams(
        sigma_b=params.sigma_b,
        s_J=float(np.sqrt(s_J_sq_hat)),
        lambda_jump=lambda_hat,
        mu=params.mu,
    )
    ll = log_likelihood(increments, dts, new_params)

    return JumpEstimate(
        lambda_hat=lambda_hat,
        s_J_sq_hat=s_J_sq_hat,
        jump_timestamps=jump_ts,
        log_likelihood=ll,
    )


def bipower_variance(increments: np.ndarray | list[float]) -> float:
    """Bi-power variation estimator of integrated diffusive variance.

    Formula:  BV = (π/2) · Σ_{t=2}^{N} |Δx̂_t| · |Δx̂_{t-1}|

    Under a diffusive + finite-activity-jump process this converges to
    ``∫ σ_b²(s) ds`` — jumps contribute to at most one product in each
    consecutive pair and therefore wash out in the limit. We expose this
    as a diagnostic only; the M-step uses the posterior-weighted estimator
    because it is more efficient when ``γ_t`` is already available.

    Returns 0 for fewer than two increments.
    """
    inc = np.asarray(increments, dtype=float)
    if inc.ndim != 1:
        raise ValueError(f"increments must be 1-D; got shape {inc.shape}")
    if inc.size < 2:
        return 0.0
    abs_inc = np.abs(inc)
    return float(BIPOWER_SCALE * np.sum(abs_inc[:-1] * abs_inc[1:]))


__all__ = [
    "BIPOWER_SCALE",
    "DEFAULT_S_J_SQ_FLOOR",
    "JumpEstimate",
    "MU_1",
    "bipower_variance",
    "m_step_jumps",
]
