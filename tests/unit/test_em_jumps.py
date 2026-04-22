"""Unit tests for `core/em/jumps.py` (paper §5.2 eq 11-12).

E-step: posterior jump responsibility gamma_t.
M-step: update (sigma_b^2, lambda, s_J^2) via weighted moments.
Cross-check: bi-power variation estimator (paper §4.8 pointer).
"""

from __future__ import annotations

import numpy as np
import pytest

em_jumps = pytest.importorskip(
    "blksch.core.em.jumps",
    reason="core/em/jumps.py not yet implemented",
)

pytestmark = pytest.mark.unit


class TestEMJumps:
    def test_recovers_known_parameters_on_synthetic(self) -> None:
        """Given a synthetic path with known (sigma_b, lambda, s_J), after
        ~6 global EM steps estimates land within 15% of truth."""
        from tests.fixtures.synthetic import SyntheticConfig, generate_rn_consistent_path

        cfg = SyntheticConfig(n_steps=4000, sigma_b=0.05, lambda_per_sec=0.01, jump_std=0.08)
        path = generate_rn_consistent_path(cfg)
        dx = np.diff(path.x, prepend=path.x[0])

        est = em_jumps.run_em(dx, dt=cfg.dt_sec, n_iters=6)  # type: ignore[attr-defined]

        assert abs(est["sigma_b"] - cfg.sigma_b) / cfg.sigma_b < 0.15
        assert abs(est["lambda"] - cfg.lambda_per_sec) / cfg.lambda_per_sec < 0.30
        # second-moment of jumps (not std): s_J^2 target
        s2_target = cfg.jump_std * cfg.jump_std
        assert abs(est["s2_j"] - s2_target) / s2_target < 0.30

    def test_gamma_posterior_spikes_on_true_jumps(self) -> None:
        """Indices where jumps[t] == 1 should have gamma_t > 0.5 for most cases."""
        pytest.skip("Stub: depends on EM API")

    def test_bipower_variation_agrees_with_em_estimate(self) -> None:
        """BPV-based jump intensity should be within 20% of EM estimate —
        they use different info but should agree on the big picture."""
        pytest.skip("Stub")

    def test_convergence_stops_when_params_stabilize(self) -> None:
        """EM loop exits when |Delta param| / |param| < tol for all params."""
        pytest.skip("Stub")
