"""Unit tests for `core/surface/corr.py` (paper §5.4).

De-jumped instantaneous correlation on intervals without jumps, with Jacobian
rescaling S'(x_i) * S'(x_j); co-jump intensity and second-moment estimates.
"""

from __future__ import annotations

import numpy as np
import pytest

surface_corr = pytest.importorskip(
    "blksch.core.surface.corr",
    reason="core/surface/corr.py not yet implemented",
)

pytestmark = pytest.mark.unit


class TestDeJumpedCorrelation:
    def test_recovers_known_rho_diffusion_only(self) -> None:
        """Simulate correlated Brownian motions with known rho = 0.6; de-jumped
        estimate within 0.05 of truth on 1000s of data."""
        pytest.skip("Stub")

    def test_ignores_windows_with_jumps(self) -> None:
        """Intervals with gamma_t > tau_J in either series are dropped from
        the covariance sum."""
        pytest.skip("Stub")

    def test_clamps_rho_to_unit_interval(self) -> None:
        """Even with sampling noise, output rho in [-1, 1]."""
        pytest.skip("Stub")


class TestCoJumps:
    def test_detects_common_jump_events(self) -> None:
        """Common jumps injected at known timestamps must be detected with
        hit rate > 80% at per-series threshold tau_J = 0.7."""
        pytest.skip("Stub")

    def test_co_jump_second_moment_recovered(self) -> None:
        """Estimated m2 within 20% of injected value on 500+ co-jumps."""
        pytest.skip("Stub")
