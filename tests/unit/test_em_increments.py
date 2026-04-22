"""Unit tests for `core/em/increments.py` (paper §5.2).

Mixture likelihood on discretized increments:
    Delta x_t ~ N(mu_t * Delta, sigma_b^2(t) * Delta)   w.p. 1 - lambda_t * Delta
               ~ Z_t ~ f_J(.; theta_t)                   w.p. lambda_t * Delta
"""

from __future__ import annotations

import numpy as np
import pytest

em_increments = pytest.importorskip(
    "blksch.core.em.increments",
    reason="core/em/increments.py not yet implemented",
)

pytestmark = pytest.mark.unit


class TestIncrementMixture:
    def test_likelihood_is_convex_combination(self) -> None:
        """phi + psi weights sum consistently with the mixture definition."""
        pytest.skip("Stub: fill in when increments lands")

    def test_gaussian_and_jump_kernels_match_pdf(self) -> None:
        """phi(dx) matches Gaussian pdf; psi(dx) matches chosen jump law pdf."""
        pytest.skip("Stub")

    def test_delta_scaling_matches_itoprocess(self) -> None:
        """Diffusive variance scales linearly with Delta; this is the basic
        invariance that everything else builds on."""
        pytest.skip("Stub")
