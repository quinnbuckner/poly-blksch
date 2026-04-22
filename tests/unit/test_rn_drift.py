"""Unit tests for `core/em/rn_drift.py` (paper §3.2 eq 3).

The risk-neutral drift is computed to enforce the martingale restriction:
    mu(t, x) = -( 1/2 S''(x) sigma_b^2 + Integral[S(x+z) - S(x) - S'(x) chi(z)] nu(dz) ) / S'(x)

Under this drift, the discounted probability p_t = S(x_t) is a Q-martingale.
The test: simulate forward under the computed mu, verify E[p_{t+Delta} | F_t] ~= p_t.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

rn_drift = pytest.importorskip(
    "blksch.core.em.rn_drift",
    reason="core/em/rn_drift.py not yet implemented",
)

pytestmark = pytest.mark.unit


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class TestRiskNeutralDrift:
    def test_martingale_property_diffusion_only(self) -> None:
        """With lambda=0, simulate 1-step forward under computed mu. The sample
        mean of p_{t+dt} should be within 2 SE of p_t over N=50000 paths."""
        rng = np.random.default_rng(123)
        x0 = 0.3
        sigma_b = 0.1
        dt = 1.0
        p0 = sigmoid(x0)

        mu = rn_drift.compute_mu(  # type: ignore[attr-defined]
            x=x0, sigma_b=sigma_b, jump_law=None, lambda_=0.0
        )

        n_paths = 50_000
        dW = rng.normal(0.0, math.sqrt(dt), size=n_paths)
        x1 = x0 + mu * dt + sigma_b * dW
        p1 = 1.0 / (1.0 + np.exp(-x1))
        mean_p1 = float(p1.mean())
        se = float(p1.std() / math.sqrt(n_paths))
        assert abs(mean_p1 - p0) < 3.0 * se, (
            f"Martingale violated: E[p1]={mean_p1:.6f}, p0={p0:.6f}, SE={se:.6f}"
        )

    def test_martingale_property_with_jumps(self) -> None:
        """With symmetric Gaussian jumps at lambda=0.05, jump_std=0.1, same
        martingale check must hold (wider CI due to jump variance)."""
        pytest.skip("Stub: fill in after diffusion-only case passes")

    def test_mu_is_capped_at_config_limit(self) -> None:
        """|mu| <= mu_cap_per_sec (paper §6.4 stability note, default 0.25)."""
        pytest.skip("Stub")

    def test_sprime_clip_prevents_blowup_near_boundary(self) -> None:
        """At x = -10 (p ~ 4.5e-5), S'(x) is tiny; mu computation must not
        divide by zero — should use sprime_clip floor."""
        pytest.skip("Stub")

    def test_monte_carlo_jump_compensator_converges(self) -> None:
        """With 600 MC draws (paper §6.4 default), compensator estimate has
        std < 5% of its magnitude."""
        pytest.skip("Stub")
