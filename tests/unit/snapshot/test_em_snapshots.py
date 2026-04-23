"""Regression snapshots for ``core/em/`` modules.

Covers: ``em/increments`` (posteriors, log-likelihood, e-step),
``em/jumps`` (M-step for λ and s_J²), ``em/rn_drift`` (``compile_mu_fn``
and the outer ``em_calibrate`` loop).

We run the outer loop on a short synthetic path with small ``max_iters``
and ``mc_samples`` so the test stays fast yet deterministic — the goal
is regression coverage, not calibration quality.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from blksch.core.em.increments import (
    MixtureParams, compute_posteriors, e_step, log_likelihood,
)
from blksch.core.em.jumps import JumpEstimate, m_step_jumps
from blksch.core.em.rn_drift import (
    RNDriftConfig, compile_mu_fn, em_calibrate,
)
from blksch.schemas import LogitState

from tests.unit.snapshot._helpers import assert_matches_snapshot

pytestmark = pytest.mark.unit

T0 = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)


def _synthetic_states(n: int = 200, seed: int = 42) -> list[LogitState]:
    """Short synthetic path with a handful of injected jumps."""
    rng = np.random.default_rng(seed)
    sigma_b = 0.05
    dx = rng.normal(0.0, sigma_b, size=n - 1)
    # Inject 3 jumps.
    for idx in (40, 90, 150):
        dx[idx] += 0.4 * (1.0 if rng.random() > 0.5 else -1.0)
    x = np.concatenate([[0.0], np.cumsum(dx)])
    return [
        LogitState(
            token_id="em-snap",
            x_hat=float(x[i]),
            sigma_eta2=1e-4,
            ts=T0 + timedelta(seconds=i),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# em/increments
# ---------------------------------------------------------------------------


def test_increments_posteriors_log_likelihood_snapshot():
    states = _synthetic_states(n=200)
    params = MixtureParams(sigma_b=0.05, s_J=0.12, lambda_jump=0.02, mu=0.0)
    posteriors = e_step(states, params)
    ll = log_likelihood(posteriors.increments, posteriors.dts, params)

    # Sample gamma at start / middle / end so the snapshot stays compact
    # while still picking up any drift in the E-step kernel.
    g = posteriors.gamma
    sample_indices = [0, 10, 20, 39, 40, 89, 90, 149, 150, len(g) - 1]
    payload = {
        "n_increments": posteriors.n,
        "gamma_sample": {
            str(i): float(g[i]) for i in sample_indices if 0 <= i < len(g)
        },
        "gamma_mean": float(g.mean()),
        "gamma_max": float(g.max()),
        "log_likelihood": ll,
        "params": asdict(params),
    }
    assert_matches_snapshot(payload, "em/increments_e_step.json")


def test_compute_posteriors_direct_snapshot():
    """Run ``compute_posteriors`` on a hand-picked increment set — no
    Kalman plumbing — to pin the kernel numerics."""
    params = MixtureParams(sigma_b=0.05, s_J=0.12, lambda_jump=0.02, mu=0.0)
    increments = np.array([0.001, 0.02, -0.01, 0.15, -0.3, 0.00001])
    dts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    gamma = compute_posteriors(increments, dts, params)
    payload = {
        "increments": increments.tolist(),
        "gamma":      gamma.tolist(),
    }
    assert_matches_snapshot(payload, "em/increments_posteriors_direct.json")


# ---------------------------------------------------------------------------
# em/jumps
# ---------------------------------------------------------------------------


def test_m_step_jumps_snapshot():
    states = _synthetic_states(n=200)
    params = MixtureParams(sigma_b=0.05, s_J=0.12, lambda_jump=0.02, mu=0.0)
    posteriors = e_step(states, params)
    estimate: JumpEstimate = m_step_jumps(states, posteriors, params)
    payload = {
        "lambda_hat": estimate.lambda_hat,
        "s_J_sq_hat": estimate.s_J_sq_hat,
        "n_jump_timestamps": len(estimate.jump_timestamps),
        "log_likelihood": estimate.log_likelihood,
    }
    assert_matches_snapshot(payload, "em/jumps_m_step.json")


# ---------------------------------------------------------------------------
# em/rn_drift: compile_mu_fn + em_calibrate
# ---------------------------------------------------------------------------


def test_compile_mu_fn_grid_snapshot():
    """Compile μ(t,x) under fixed (σ_b, λ, s_J) and probe it on a grid."""
    mu_fn = compile_mu_fn(
        sigma_b=0.05, lambda_jump=0.02, s_J=0.12,
        config=RNDriftConfig(mc_samples=500, seed=12345),
    )
    grid_x = [-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0]
    values = [float(mu_fn(T0, x)) for x in grid_x]
    assert_matches_snapshot(
        {"x_grid": grid_x, "mu_values": values},
        "em/rn_drift_mu_grid.json",
    )


def test_em_calibrate_snapshot():
    """End-to-end EM loop on a short path. Pin final params + iter count +
    convergence flag; log-likelihood history is bucketed to len (not
    listed entry-by-entry) because MC noise in the drift can shift the
    LL magnitude while preserving the converged fit.
    """
    states = _synthetic_states(n=200, seed=7)
    initial = MixtureParams(sigma_b=0.06, s_J=0.15, lambda_jump=0.03, mu=0.0)
    cal = em_calibrate(
        states, initial,
        max_iters=6, tol=1e-3,
        drift_config=RNDriftConfig(mc_samples=300, seed=987),
    )
    payload = {
        "final_params": asdict(cal.final_params),
        "jumps": {
            "lambda_hat": cal.jumps.lambda_hat,
            "s_J_sq_hat": cal.jumps.s_J_sq_hat,
            "log_likelihood": cal.jumps.log_likelihood,
        },
        "iters": cal.iters,
        "converged": cal.converged,
        "ll_history_len": len(cal.log_likelihood_history),
        "ll_final": cal.log_likelihood_history[-1] if cal.log_likelihood_history else None,
    }
    assert_matches_snapshot(payload, "em/rn_drift_em_calibrate.json")
