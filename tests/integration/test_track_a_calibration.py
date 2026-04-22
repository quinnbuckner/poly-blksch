"""Track A integration test: full calibration pipeline on a synthetic path.

Chain: canonical_mid -> microstruct -> kalman -> em -> rn_drift -> surface.
Asserts the pipeline runs end-to-end and the recovered (sigma_b, lambda, s_J)
are within tolerance of the seeded ground truth.

Distinct from the pipeline-level Sec-6 replication test: this is a unit-of-work
integration (one synthetic scenario, looser bounds); the pipeline test is the
full §6 evaluation with MSE/MAE/QLIKE numeric gates.
"""

from __future__ import annotations

import numpy as np
import pytest

# Defer import failures to per-test so single missing module doesn't skip all
try:  # pragma: no cover - availability varies during build
    from blksch.core.filter import canonical_mid, kalman, microstruct  # noqa: F401
    from blksch.core.em import increments, jumps, rn_drift  # noqa: F401
    from blksch.core.surface import corr, smooth  # noqa: F401
    _TRACK_A_READY = True
except ImportError:
    _TRACK_A_READY = False

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not _TRACK_A_READY, reason="Track A pipeline not yet fully implemented")
class TestTrackAEndToEnd:
    def test_pipeline_runs_without_crash(self) -> None:
        from tests.fixtures.synthetic import (
            SyntheticConfig,
            generate_rn_consistent_path,
            inject_microstructure_noise,
        )

        cfg = SyntheticConfig(n_steps=1500)
        path = generate_rn_consistent_path(cfg)
        y, _ = inject_microstructure_noise(path.x)
        # Chain calls go here once the module APIs are stabilized.
        pytest.skip("Stub: wire the actual pipeline chain once APIs are stable")

    def test_recovers_sigma_b_within_tolerance(self) -> None:
        pytest.skip("Stub: loose bound (20%) integration check")

    def test_recovers_lambda_within_tolerance(self) -> None:
        pytest.skip("Stub")

    def test_surface_is_nonnegative_and_smooth(self) -> None:
        pytest.skip("Stub")

    def test_diagnostics_pass_on_clean_synthetic(self) -> None:
        """Ljung-Box and realized-vs-implied checks should pass on a
        deliberately clean synthetic — if they fail here, the pipeline is
        broken, not the diagnostics."""
        pytest.skip("Stub")
