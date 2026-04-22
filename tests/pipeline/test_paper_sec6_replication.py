"""STAGE 0 -> STAGE 1 CORRECTNESS GATE.

Reproduces the paper's §6 evaluation (Dalen 2026 arXiv:2510.15205v2):
synthetic RN-consistent path, 1 Hz grid, N=6000 steps, forecast horizon
h=60s. Runs the full Track A calibration pipeline causally, computes the
paper's forward-sum realized logit variance, and asserts MSE/MAE/QLIKE
are within 10% of the RN-JD row of Table 1:

    MSE_all  ≈ 70.28
    MAE_all  ≈ 1.59
    QLIKE_all ≈ 1.46

If this test fails, Track A's calibration is not trustworthy and we must
not promote to Stage 1. Tune (in order) EM window, jump threshold, MC
draws, or Kalman process-noise proxy. Consult diagnostics.py outputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# This test depends on the whole Track A pipeline.
try:  # pragma: no cover
    from blksch.core.filter.canonical_mid import resample_to_grid  # type: ignore  # noqa: F401
    from blksch.core.filter.kalman import HeteroskedasticKalman  # type: ignore  # noqa: F401
    from blksch.core.em.rn_drift import compute_mu  # type: ignore  # noqa: F401
    _PIPELINE_READY = True
except ImportError:
    _PIPELINE_READY = False

pytestmark = [pytest.mark.pipeline, pytest.mark.slow]


# Paper Table 1 targets (RN-JD row)
TARGET_MSE = 70.281
TARGET_MAE = 1.588
TARGET_QLIKE = 1.462
TOLERANCE = 0.10  # 10%


def _run_track_a_pipeline_causally(y: np.ndarray, sigma_eta2: np.ndarray) -> dict:
    """Placeholder for the full causal Track A chain.

    Real implementation should:
      1. Heteroskedastic KF smoother -> x_hat
      2. EM on rolling 400s window -> (sigma_b_hat, lambda_hat, s2_j_hat)
      3. Compute mu(t, x) from martingale restriction
      4. Re-filter with refined mu
      5. Return per-step (x_hat, sigma_b_hat, lambda_hat, s2_j_hat)
    """
    raise NotImplementedError(
        "Track A pipeline not wired yet — replace this stub when ready"
    )


def _causal_forward_sum_forecast(
    sigma_b_hat: np.ndarray,
    lambda_hat: np.ndarray,
    s2_j_hat: np.ndarray,
    c_j: float,
    h: int,
) -> np.ndarray:
    """Paper eq (causal forward-sum, §6.3)."""
    n = len(sigma_b_hat)
    out = np.zeros(n)
    for t in range(n - h):
        diffusion = float(np.sum(sigma_b_hat[t + 1 : t + 1 + h] ** 2))
        jumps = float(s2_j_hat[t] * np.sum(lambda_hat[t + 1 : t + 1 + h]))
        out[t] = diffusion + c_j * jumps
    return out


def _qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    mask = (realized > 0) & (forecast > 0)
    if not mask.any():
        return float("nan")
    r = realized[mask]
    f = forecast[mask]
    return float(np.mean(r / f - np.log(r / f) - 1.0))


@pytest.mark.skipif(not _PIPELINE_READY, reason="Track A pipeline not yet implemented")
def test_rn_jd_replicates_paper_table_1_causal_h60s() -> None:
    """Full causal replication of paper's Table 1 RN-JD row."""
    from tests.fixtures.synthetic import (
        SyntheticConfig,
        generate_rn_consistent_path,
        inject_microstructure_noise,
    )

    cfg = SyntheticConfig(n_steps=6000, dt_sec=1.0, rng_seed=42)
    path = generate_rn_consistent_path(cfg)
    y, sigma_eta2 = inject_microstructure_noise(path.x, rng_seed=43)

    est = _run_track_a_pipeline_causally(y, sigma_eta2)
    x_hat = est["x_hat"]
    sigma_b_hat = est["sigma_b_hat"]
    lambda_hat = est["lambda_hat"]
    s2_j_hat = est["s2_j_hat"]
    c_j = est.get("c_j", 0.5)

    # realized logit variance on filtered path (paper §6.1)
    h = 60
    dx = np.diff(x_hat, prepend=x_hat[0])
    rv = np.zeros(len(x_hat))
    for t in range(len(x_hat) - h):
        rv[t] = float(np.sum(dx[t + 1 : t + 1 + h] ** 2))

    vhat = _causal_forward_sum_forecast(sigma_b_hat, lambda_hat, s2_j_hat, c_j, h)

    valid = slice(0, len(x_hat) - h)
    err = rv[valid] - vhat[valid]
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    qlike = _qlike(rv[valid], vhat[valid])

    assert mse <= TARGET_MSE * (1.0 + TOLERANCE), (
        f"MSE regression: {mse:.2f} vs target {TARGET_MSE:.2f} +/- {TOLERANCE:.0%}"
    )
    assert mae <= TARGET_MAE * (1.0 + TOLERANCE), (
        f"MAE regression: {mae:.3f} vs target {TARGET_MAE:.3f} +/- {TOLERANCE:.0%}"
    )
    assert qlike <= TARGET_QLIKE * (1.0 + TOLERANCE), (
        f"QLIKE regression: {qlike:.3f} vs target {TARGET_QLIKE:.3f} +/- {TOLERANCE:.0%}"
    )


@pytest.mark.skipif(not _PIPELINE_READY, reason="Track A pipeline not yet implemented")
def test_rn_jd_beats_rw_baseline() -> None:
    """Paper's RN-JD should beat naive RW-logit baseline on QLIKE.

    Baseline: x_{t+1} = x_t + sigma_RW * eps with sigma_RW constant set from
    training-slice mean of (Delta x)^2.
    """
    pytest.skip("Stub: implement after main replication passes")


def test_forward_sum_operator_is_causal() -> None:
    """Sanity test of the forward-sum operator itself — uses no Track A code,
    runs unconditionally to confirm the evaluation plumbing is correct."""
    n = 100
    h = 5
    a = np.arange(n, dtype=float)
    # Expected: out[t] = a[t+1] + ... + a[t+h] for t < n-h, else 0
    from tests.fixtures.synthetic import causal_forward_sum_variance

    out = causal_forward_sum_variance(a, h)
    expected_0 = float(np.sum((np.diff(a, prepend=a[0])[1 : 1 + h]) ** 2))
    assert out[0] == pytest.approx(expected_0)
    assert out[-h:].sum() == 0.0  # last h entries are zero by construction
