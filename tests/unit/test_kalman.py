"""Unit tests for `core/filter/kalman.py` (paper §5.1).

Heteroskedastic Kalman (or UKF near boundaries) on y = logit(p~) with
time-varying measurement variance sigma_eta^2(t). The transition model is a
local level; the RN drift is injected later by the EM loop.
"""

from __future__ import annotations

import numpy as np
import pytest

kalman = pytest.importorskip(
    "blksch.core.filter.kalman",
    reason="core/filter/kalman.py not yet implemented",
)

pytestmark = pytest.mark.unit


class TestKalmanOnSyntheticPath:
    def test_recovers_true_x_on_diffusion_only(self) -> None:
        """With jumps disabled, mean squared error of (x_hat - x_true) should be
        ~ sigma_eta^2 / (effective observation count), i.e. small."""
        from tests.fixtures.synthetic import (
            SyntheticConfig,
            generate_rn_consistent_path,
            inject_microstructure_noise,
        )

        cfg = SyntheticConfig(lambda_per_sec=0.0, sched_jump_lambda_boost=0.0, n_steps=2000)
        path = generate_rn_consistent_path(cfg)
        y, sigma_eta2 = inject_microstructure_noise(path.x)

        # Expected API - adapt to actual module signature:
        filt = kalman.HeteroskedasticKalman()  # type: ignore[attr-defined]
        x_hat = filt.filter(y, sigma_eta2)  # type: ignore[attr-defined]

        mse = float(np.mean((x_hat - path.x) ** 2))
        assert mse < 0.02, f"KF MSE too high: {mse:.4f} — filter is not tracking"

    def test_handles_heteroskedastic_noise_gracefully(self) -> None:
        """When sigma_eta^2(t) swings 10x between regimes, filter should
        down-weight high-noise periods — MSE no worse than 2x baseline."""
        pytest.skip("Stub: implement after baseline test passes")

    def test_smoother_beats_filter_on_past_points(self) -> None:
        """RTS smoother should reduce MSE vs. forward-only filter on indices < N-h."""
        pytest.skip("Stub")

    def test_ukf_fallback_near_boundaries(self) -> None:
        """With p pinned near 0.01 for long stretches, UKF variant should not
        diverge even though KF may be unstable."""
        pytest.skip("Stub")

    def test_innovations_are_serially_uncorrelated(self) -> None:
        """Ljung-Box on one-step-ahead innovations must not reject at 5%."""
        pytest.skip("Stub: depends on diagnostics module")
