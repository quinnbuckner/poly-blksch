"""Risk-neutral drift compensator + outer EM loop (paper §3.2 eq 3, §5.2).

The paper's Q-martingale condition for ``p_t = S(x_t)`` pins the drift
``μ(t, x)`` once ``σ_b²``, the jump intensity ``λ``, and the jump law
(here Gaussian ``N(0, s_J²)``) are known::

    μ(t, x) = -[½·S''(x)·σ_b² + λ·E_z(S(x+z) − S(x) − S'(x)·z)] / S'(x)

With ``S(x) = σ(x) = 1/(1+e^{-x})``:
  * ``S'(x)  = p(1-p)``
  * ``S''(x) = p(1-p)(1-2p)``

Under ``λ = 0`` this collapses to the analytic ``μ(x) = -½·(1-2p)·σ_b²``
(paper's pure-diffusion limit); the tests pin that closed form exactly.

Implementation choices
----------------------
* ``compile_mu_fn`` builds a once-per-call table of ``μ`` on a logit grid
  (default ``x ∈ [-5, 5]``, 201 points) and linearly interpolates at
  serve time, so the online Kalman predict step never runs 2 000 Monte
  Carlo draws itself. This matches the paper §6.4 "compiled once per
  call" recipe.
* Monte Carlo compensator uses a fixed seed by default so calibration is
  deterministic (tests lean on this).
* Numerical guards mirror the paper's stability notes in §6.4:
    - ``S'(x)`` floored at ``sprime_clip=1e-4`` before the division
      (prevents blow-up near ``p ∈ {0, 1}``).
    - ``|μ| ≤ mu_cap_per_sec=0.25`` clip (default matches ``config/bot.yaml``).
* When ``λ̂ = 0`` the jump-compensator branch is skipped entirely and
  ``μ`` reduces to ``-½·(1-2p)·σ_b²``.

The outer-EM orchestrator :func:`em_calibrate` wires:

    E-step  (increments.e_step)  →
    M-step jumps                  →
    M-step σ_b   (paper eq 11)   →
    μ via rn_drift                →
    E-step again …

stopping when the mixture log-likelihood's relative change falls below
``tol`` or ``max_iters`` is hit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from blksch.core.em.increments import (
    MixtureParams,
    PosteriorResult,
    e_step,
    log_likelihood,
)
from blksch.core.em.jumps import (
    DEFAULT_S_J_SQ_FLOOR,
    JumpEstimate,
    bipower_variance,
    m_step_jumps,
)
from blksch.schemas import LogitState

logger = logging.getLogger(__name__)

DEFAULT_X_GRID_MIN = -5.0
DEFAULT_X_GRID_MAX = 5.0
DEFAULT_X_GRID_POINTS = 401
DEFAULT_MC_SAMPLES = 2000
DEFAULT_MU_CAP_PER_SEC = 0.25
DEFAULT_SPRIME_CLIP = 1.0e-4
DEFAULT_SEED = 20260423

DEFAULT_MAX_ITERS = 50
DEFAULT_TOL = 1.0e-4

# Warm-start heuristics when ``em_calibrate(initial_params=None)``.
# Paper §5.2: bi-power variation is a jump-robust estimator of the integrated
# diffusion variance. Seeding σ_b² with BV/T breaks the identifiability
# degeneracy between (σ_b, λ, s_J²) — the mixture only has to sort out
# (λ, s_J²).
DEFAULT_WARM_START_LAMBDA = 0.01
DEFAULT_WARM_START_S_J_SQ_FLOOR_MULT = 4.0  # s_J_init = sqrt(mult · s_J_sq_floor)


MuFn = Callable[[datetime, float], float]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RNDriftConfig:
    """Knobs for :func:`compile_mu_fn`. Defaults match ``config/bot.yaml``."""

    x_grid_min: float = DEFAULT_X_GRID_MIN
    x_grid_max: float = DEFAULT_X_GRID_MAX
    x_grid_points: int = DEFAULT_X_GRID_POINTS
    mc_samples: int = DEFAULT_MC_SAMPLES
    mu_cap_per_sec: float = DEFAULT_MU_CAP_PER_SEC
    sprime_clip: float = DEFAULT_SPRIME_CLIP
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.x_grid_min >= self.x_grid_max:
            raise ValueError("x_grid_min must be < x_grid_max")
        if self.x_grid_points < 3:
            raise ValueError("x_grid_points must be >= 3")
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")
        if self.mu_cap_per_sec <= 0:
            raise ValueError("mu_cap_per_sec must be positive")
        if self.sprime_clip <= 0:
            raise ValueError("sprime_clip must be positive")


# ---------------------------------------------------------------------------
# Numerically stable sigmoid (vector)
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable logistic — avoids overflow in ``exp(-x)`` for large negative x."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ---------------------------------------------------------------------------
# μ(x) grid builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MuGrid:
    """Precomputed μ on a logit grid + the interpolant closure key."""

    x_grid: np.ndarray
    mu_grid: np.ndarray

    def at(self, x: float) -> float:
        # np.interp clamps to the grid endpoints, which matches our intent:
        # beyond x_grid_max (p → 1) μ saturates at whatever was computed at
        # the boundary, already capped.
        return float(np.interp(x, self.x_grid, self.mu_grid))


def _build_mu_grid(
    sigma_b: float,
    lambda_jump: float,
    s_J: float,
    config: RNDriftConfig,
) -> _MuGrid:
    x_grid = np.linspace(config.x_grid_min, config.x_grid_max, config.x_grid_points)
    p = _sigmoid(x_grid)
    one_minus_p = 1.0 - p
    sprime_raw = p * one_minus_p
    sprime_safe = np.maximum(sprime_raw, config.sprime_clip)
    s_double_prime = sprime_raw * (1.0 - 2.0 * p)

    diffusion_term = 0.5 * s_double_prime * sigma_b * sigma_b

    if lambda_jump <= 0.0 or s_J <= 0.0:
        jump_term = np.zeros_like(x_grid)
    else:
        rng = np.random.default_rng(config.seed)
        z = rng.normal(0.0, s_J, size=config.mc_samples)
        # (G, K) = (n_grid, mc_samples) evaluation of S(x + z).
        x_plus_z = x_grid[:, None] + z[None, :]
        sigma_plus = _sigmoid(x_plus_z)
        sigma_x = p[:, None]
        # χ(z) = z for small-jump truncation; paper's convention.
        integrand = sigma_plus - sigma_x - sprime_raw[:, None] * z[None, :]
        jump_term = lambda_jump * integrand.mean(axis=1)

    mu_grid = -(diffusion_term + jump_term) / sprime_safe
    mu_grid = np.clip(mu_grid, -config.mu_cap_per_sec, config.mu_cap_per_sec)
    return _MuGrid(x_grid=x_grid, mu_grid=mu_grid)


def compile_mu_fn(
    sigma_b: float,
    lambda_jump: float,
    s_J: float,
    *,
    config: RNDriftConfig | None = None,
) -> MuFn:
    """Return a ``mu_fn(t, x) -> float`` that satisfies the martingale drift.

    The function is deterministic for fixed (sigma_b, lambda_jump, s_J,
    config) — same inputs, byte-identical output — because the MC
    compensator is evaluated once with a fixed seed and interpolated
    thereafter. ``t`` is accepted for future time-varying extensions but
    is not currently used.
    """
    if sigma_b < 0:
        raise ValueError("sigma_b must be non-negative")
    if lambda_jump < 0:
        raise ValueError("lambda_jump must be non-negative")
    if s_J < 0:
        raise ValueError("s_J must be non-negative")
    cfg = config or RNDriftConfig()
    grid = _build_mu_grid(sigma_b, lambda_jump, s_J, cfg)

    def mu_fn(t: datetime, x: float) -> float:  # noqa: ARG001 — t reserved
        return grid.at(x)

    # Attach metadata for tests/diagnostics without leaking implementation.
    mu_fn.x_grid = grid.x_grid  # type: ignore[attr-defined]
    mu_fn.mu_grid = grid.mu_grid  # type: ignore[attr-defined]
    return mu_fn


# ---------------------------------------------------------------------------
# σ_b M-step (paper eq 11)
# ---------------------------------------------------------------------------


def _m_step_sigma_b(posteriors: PosteriorResult, params: MixtureParams) -> float:
    """Weighted diffusive-variance update.

    σ̂_b² = Σ (1-γ_t)·(Δx̂_t - μ·Δt)² / Σ (1-γ_t)·Δt

    Returns the prior σ_b when the weighted denominator is degenerate
    (all increments marked as jumps, or a single-step sequence).
    """
    gamma = posteriors.gamma
    inc = posteriors.increments
    dts = posteriors.dts
    weight = 1.0 - gamma
    denom = float(np.sum(weight * dts))
    if denom <= 0.0:
        return params.sigma_b
    residual = inc - params.mu * dts
    numer = float(np.sum(weight * residual * residual))
    sigma_b_sq = numer / denom
    if sigma_b_sq <= 0.0 or not np.isfinite(sigma_b_sq):
        return params.sigma_b
    return float(np.sqrt(sigma_b_sq))


# ---------------------------------------------------------------------------
# Outer EM loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationResult:
    """Output of :func:`em_calibrate`. Internal to Track A."""

    final_params: MixtureParams
    jumps: JumpEstimate
    mu_fn: MuFn
    log_likelihood_history: list[float] = field(default_factory=list)
    converged: bool = False
    iters: int = 0


def _mean_state_mu(mu_fn: MuFn, states: list[LogitState]) -> float:
    if not states:
        return 0.0
    return float(np.mean([mu_fn(s.ts, s.x_hat) for s in states]))


def _warm_start_from_bipower(states: list[LogitState]) -> MixtureParams:
    """Derive a self-consistent warm start from bi-power variation + RV.

    Paper §5.2: BV is a jump-robust estimator of ``∫ σ_b²(s) ds`` — use it
    to seed ``σ_b``. The residual ``RV - BV`` approximates the jump
    contribution ``λ · s_J² · T`` (total quadratic variation minus the
    diffusion piece), so given a neutral ``λ_init`` we can back out a
    ``s_J_init`` that makes the seed self-consistent. This keeps EM from
    collapsing into the "no jumps" local optimum when the data is weakly
    informative about ``(λ, s_J)``.

    Falls back to conservative scale-only seeds when the state sequence is
    too short (< 3 increments).
    """
    if len(states) < 3:
        return MixtureParams(
            sigma_b=0.1,
            s_J=0.3,
            lambda_jump=DEFAULT_WARM_START_LAMBDA,
            mu=0.0,
        )

    x = np.array([s.x_hat for s in states], dtype=float)
    ts = [s.ts for s in states]
    increments = np.diff(x)
    dts = np.array(
        [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)],
        dtype=float,
    )
    total_time = float(np.sum(dts))
    if total_time <= 0:
        return MixtureParams(sigma_b=0.1, s_J=0.3, lambda_jump=DEFAULT_WARM_START_LAMBDA)

    bv = bipower_variance(increments)
    rv = float(np.sum(increments * increments))
    sigma_b_sq_init = max(bv / total_time, 1.0e-8)
    sigma_b_init = float(np.sqrt(sigma_b_sq_init))

    # Excess quadratic variation above the diffusion component is attributed
    # to jumps. Convert to a per-jump magnitude given λ_init.
    excess = max(rv - bv, 0.0)
    lambda_init = DEFAULT_WARM_START_LAMBDA
    # Expected total jump events at λ_init; ensure at least ~1 so s_J_init
    # has a meaningful scale on short windows.
    effective_n_jumps = max(lambda_init * total_time, 1.0)
    s_J_sq_init = max(
        excess / effective_n_jumps,
        DEFAULT_WARM_START_S_J_SQ_FLOOR_MULT * DEFAULT_S_J_SQ_FLOOR,
    )
    # Cap s_J_init so we don't seed pathologically wide (would wash out
    # M-step signal); a few multiples of σ_b is a reasonable ceiling.
    s_J_sq_init = min(s_J_sq_init, 100.0 * sigma_b_sq_init)
    s_J_init = float(np.sqrt(s_J_sq_init))

    return MixtureParams(
        sigma_b=sigma_b_init,
        s_J=s_J_init,
        lambda_jump=lambda_init,
        mu=0.0,
    )


def em_calibrate(
    states: list[LogitState],
    initial_params: MixtureParams | None = None,
    *,
    max_iters: int = DEFAULT_MAX_ITERS,
    tol: float = DEFAULT_TOL,
    jump_mc_samples: int = DEFAULT_MC_SAMPLES,
    drift_config: RNDriftConfig | None = None,
) -> CalibrationResult:
    """Outer EM loop — E-step → M-steps → μ update, until LL converges.

    Parameters
    ----------
    states
        Filtered :class:`LogitState` history; adjacent pairs form the
        increments driving the E-step.
    initial_params
        If ``None``, auto-warm-start σ_b from the bi-power variation of
        ``Δx̂_t`` (paper §5.2 — jump-robust diffusion estimator) with
        neutral seeds for ``λ``/``s_J``. This is the recommended default
        because it breaks the (σ_b, λ, s_J²) identifiability degeneracy
        that shows up in short (≤ 400 s) rolling windows with rare jumps.
        Pass an explicit :class:`MixtureParams` when you want to bypass
        warm-start (e.g. for sensitivity studies or to seed from a
        previous calibration).

    Notes
    -----
    * On each pass we update **all three** parameters ``(σ_b, λ, s_J)``
      plus the constant-μ scalar (a path-averaged version of the
      state-dependent μ(t, x)); ``jumps.m_step_jumps`` handles (λ, s_J)
      and ``_m_step_sigma_b`` handles σ_b.
    * ``converged`` is True iff consecutive ``log_likelihood`` values
      satisfy ``|ΔLL|/|LL| < tol``. If the LL crashes to -inf (e.g. at
      a parameter boundary), we do **not** declare convergence.
    * The returned ``mu_fn`` is compiled from the final parameters and is
      safe to serialize / re-use at serve time.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    if tol <= 0:
        raise ValueError("tol must be positive")

    cfg = drift_config or RNDriftConfig(mc_samples=jump_mc_samples)

    if initial_params is None:
        initial_params = _warm_start_from_bipower(states)
    params = initial_params
    mu_fn: MuFn = compile_mu_fn(
        params.sigma_b, params.lambda_jump, params.s_J, config=cfg
    )
    ll_history: list[float] = []
    converged = False
    last_jumps: JumpEstimate | None = None

    for iteration in range(max_iters):
        posteriors = e_step(states, params)
        if posteriors.n == 0:
            break

        jumps_est = m_step_jumps(states, posteriors, params)
        new_sigma_b = _m_step_sigma_b(posteriors, params)

        interim_params = MixtureParams(
            sigma_b=new_sigma_b,
            s_J=float(np.sqrt(jumps_est.s_J_sq_hat)),
            lambda_jump=jumps_est.lambda_hat,
            mu=params.mu,
        )
        # Re-compile μ under the refined diffusion / jump params, then
        # fold the path-average μ into the scalar for next iteration.
        mu_fn = compile_mu_fn(
            interim_params.sigma_b,
            interim_params.lambda_jump,
            interim_params.s_J,
            config=cfg,
        )
        mu_scalar = _mean_state_mu(mu_fn, states)
        new_params = MixtureParams(
            sigma_b=interim_params.sigma_b,
            s_J=interim_params.s_J,
            lambda_jump=interim_params.lambda_jump,
            mu=mu_scalar,
        )

        # Log-likelihood under the refined parameters.
        ll = log_likelihood(posteriors.increments, posteriors.dts, new_params)
        ll_history.append(ll)
        last_jumps = jumps_est

        if len(ll_history) >= 2 and np.isfinite(ll) and np.isfinite(ll_history[-2]):
            denom = max(abs(ll_history[-2]), 1.0e-10)
            rel_change = abs(ll_history[-1] - ll_history[-2]) / denom
            if rel_change < tol:
                params = new_params
                converged = True
                return CalibrationResult(
                    final_params=params,
                    jumps=last_jumps,
                    mu_fn=mu_fn,
                    log_likelihood_history=ll_history,
                    converged=True,
                    iters=iteration + 1,
                )

        params = new_params

    # Fell out of the loop without converging — build a final jumps estimate
    # against the last params so the caller gets a coherent snapshot.
    if last_jumps is None:
        posteriors = e_step(states, params)
        last_jumps = m_step_jumps(states, posteriors, params)

    return CalibrationResult(
        final_params=params,
        jumps=last_jumps,
        mu_fn=mu_fn,
        log_likelihood_history=ll_history,
        converged=converged,
        iters=len(ll_history),
    )


__all__ = [
    "CalibrationResult",
    "DEFAULT_MAX_ITERS",
    "DEFAULT_MC_SAMPLES",
    "DEFAULT_MU_CAP_PER_SEC",
    "DEFAULT_SPRIME_CLIP",
    "DEFAULT_TOL",
    "MuFn",
    "RNDriftConfig",
    "compile_mu_fn",
    "em_calibrate",
]
