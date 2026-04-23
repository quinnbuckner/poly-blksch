"""STAGE 0 -> STAGE 1 CORRECTNESS GATE — paper §6 replication.

Reproduces the paper's §6 evaluation (Dalen 2026 arXiv:2510.15205v2) with
the **real** merged modules (not stubs): synthetic RN-consistent path at
1 Hz × N=6000 steps, heteroskedastic Kalman, rolling EM calibration, and
an H=60s forecast evaluated against the paper's Table 1 RN-JD row:

    MSE   ≈ 70.28
    MAE   ≈ 1.59
    QLIKE ≈ 1.46

Assertion tolerance: ±10% per axis. If this test fails, Stage-0 is not
cleared. Do NOT patch any module from this branch — report metrics +
innovation whitening p-value to the planning window for routing.

Likely failure modes (priority order):

1. Bias in ``rn_drift.py``'s MC jump-compensator (K=2000 samples).
2. Microstruct coefficient drift across the rolling window.
3. Kalman UKF/KF handoff producing bias near the boundary.
4. EM convergence to a local optimum from the synthetic's initial params.

Marked ``pipeline`` — excluded from default unit / integration CI; takes
roughly 60–120 s on a 2024 laptop.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from blksch.core.em.increments import MixtureParams
from blksch.core.em.rn_drift import RNDriftConfig, em_calibrate
from blksch.core.filter.canonical_mid import CanonicalMidFilter
from blksch.core.filter.kalman import KalmanFilter
from blksch.core.filter.microstruct import (
    MicrostructConfig,
    MicrostructFeatures,
    MicrostructModel,
    extract_features,
)
from blksch.schemas import BookSnap, PriceLevel

from tests.fixtures.synthetic import (
    SyntheticConfig,
    causal_forward_sum_variance,
    generate_rn_consistent_path,
    inject_microstructure_noise,
    qlike,
    sigmoid,
)

pytestmark = [pytest.mark.pipeline, pytest.mark.slow]

# --- Paper Table 1 RN-JD targets (±10% tolerance) -------------------------
TARGET_MSE = 70.28
TARGET_MAE = 1.59
TARGET_QLIKE = 1.46
TOL = 0.10

T0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _clip_p(p: float, eps: float = 1e-4) -> float:
    return max(eps, min(1.0 - eps, p))


def _build_book_stream(
    y: np.ndarray, sigma_eta2: np.ndarray, *, dt_sec: float = 1.0,
) -> list[BookSnap]:
    """Turn (noisy observed logit, measurement variance) into a BookSnap
    sequence. Spread ∝ sqrt(σ_η²) (loose book-width proxy); depth is held
    constant so the microstructure fit picks up spread as the dominant
    covariate.
    """
    n = len(y)
    bid_depth = ask_depth = 400.0
    books: list[BookSnap] = []
    for t in range(n):
        p_mid = _clip_p(sigmoid(y[t]))
        half = min(
            0.4 * p_mid, 0.4 * (1.0 - p_mid),
            max(5e-4, math.sqrt(max(sigma_eta2[t], 1e-18)) * 0.5),
        )
        p_bid = _clip_p(p_mid - half)
        p_ask = _clip_p(p_mid + half)
        books.append(BookSnap(
            token_id="sec6",
            bids=[PriceLevel(price=p_bid, size=bid_depth)],
            asks=[PriceLevel(price=p_ask, size=ask_depth)],
            ts=T0 + timedelta(seconds=int(t * dt_sec)),
        ))
    return books


def _fit_microstruct(
    books: list[BookSnap], sigma_eta2: np.ndarray, *, fit_window: int = 1200,
) -> MicrostructModel:
    """Warm-up fit on the first ``fit_window`` ticks, regressing the
    injected ground-truth σ_η² on the per-book feature row."""
    features: list[MicrostructFeatures] = []
    targets: list[float] = []
    for t in range(min(fit_window, len(books))):
        features.append(extract_features(books[t], []))
        targets.append(float(sigma_eta2[t]))
    return MicrostructModel.fit_from_features(
        features, targets,
        config=MicrostructConfig(sigma_floor=1e-8, ridge=1e-10),
    )


def _run_filter_stream(
    books: list[BookSnap], model: MicrostructModel, *, sigma_b_seed: float,
):
    """canonical_mid → Kalman. Returns (states, innovations_standardized)."""
    cmid = CanonicalMidFilter(token_id="sec6", grid_hz=1.0)
    kf = KalmanFilter(
        token_id="sec6", microstruct=model,
        sigma_b=sigma_b_seed, initial_variance=1.0,
    )
    states = []
    innovations: list[float] = []
    for book in books:
        for cm in cmid.update(book):
            states.append(kf.step(cm, book, []))
            if (kf.last_innovation is not None
                    and kf.last_innovation_variance is not None
                    and kf.last_innovation_variance > 0):
                innovations.append(
                    kf.last_innovation / math.sqrt(kf.last_innovation_variance)
                )
    return states, np.asarray(innovations, dtype=float)


def _rolling_forecasts(
    states, *,
    window_sec: int, stride_sec: int, horizon_sec: int,
    initial_params: MixtureParams | None, drift_cfg: RNDriftConfig,
    max_iters: int = 12, tol: float = 1e-3,
) -> tuple[list[int], list[float], list[tuple[float, float, float]]]:
    """At each origin t (stride ``stride_sec``), fit EM on the last
    ``window_sec`` states and emit ::

        forecast(t) = (σ̂_b² + λ̂ · ŝ²_J) · horizon_sec

    Also returns the per-origin trajectory of (σ̂_b, λ̂, ŝ²_J) so callers
    can diff against ground truth when the gate fails.
    """
    origins: list[int] = []
    preds: list[float] = []
    traj: list[tuple[float, float, float]] = []
    n = len(states)
    for t in range(window_sec, n - horizon_sec, stride_sec):
        window = states[t - window_sec: t]
        try:
            cal = em_calibrate(
                window, initial_params,
                max_iters=max_iters, tol=tol,
                drift_config=drift_cfg,
            )
        except Exception:
            continue
        p = cal.final_params
        forecast = (
            p.sigma_b ** 2
            + cal.jumps.lambda_hat * cal.jumps.s_J_sq_hat
        ) * horizon_sec
        origins.append(t)
        preds.append(float(forecast))
        traj.append((
            float(p.sigma_b),
            float(cal.jumps.lambda_hat),
            float(cal.jumps.s_J_sq_hat),
        ))
    return origins, preds, traj


def _box_pierce_q(arr: np.ndarray, lags: int = 20) -> tuple[float, int]:
    """Return (Q statistic, n). χ²(20, 0.95) ≈ 31.41."""
    if arr.size < lags * 5:
        return 0.0, int(arr.size)
    x0 = arr - arr.mean()
    denom = float((x0 * x0).sum())
    q = 0.0
    for k in range(1, lags + 1):
        num = float((x0[:-k] * x0[k:]).sum())
        rho_k = num / denom if denom > 0 else 0.0
        q += rho_k * rho_k
    return float(arr.size * q), int(arr.size)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_rn_jd_replicates_paper_table_1_causal_h60s(capsys) -> None:
    cfg = SyntheticConfig(n_steps=6000, dt_sec=1.0, rng_seed=42)
    path = generate_rn_consistent_path(cfg)
    y, sigma_eta2 = inject_microstructure_noise(path.x, rng_seed=43)

    books = _build_book_stream(y, sigma_eta2, dt_sec=cfg.dt_sec)
    microstruct = _fit_microstruct(books, sigma_eta2, fit_window=1200)
    states, innov = _run_filter_stream(
        books, microstruct, sigma_b_seed=cfg.sigma_b,
    )
    assert len(states) >= int(0.9 * cfg.n_steps), (
        f"expected ~{cfg.n_steps} LogitStates; got {len(states)}"
    )

    drift_cfg = RNDriftConfig(
        mc_samples=2000,
        mu_cap_per_sec=cfg.mu_cap_per_sec,
        sprime_clip=cfg.sprime_clip,
    )
    # Paper §6.4 recipe: 6 global EM steps to initialize, then rolling EM
    # with 400 s windows. Each rolling fit resumes from the global params
    # so short windows don't have to re-discover (σ_b, λ, s_J) from
    # scratch — this is what breaks the 400 s identifiability ridge.
    # em_calibrate(initial_params=None) auto-warm-starts from BV on the
    # global window.
    global_cal = em_calibrate(
        states, initial_params=None,
        max_iters=30, tol=1e-5,
        drift_config=drift_cfg,
    )
    global_params = global_cal.final_params
    origins, preds, traj = _rolling_forecasts(
        states,
        window_sec=400, stride_sec=60, horizon_sec=60,
        initial_params=global_params, drift_cfg=drift_cfg,
        max_iters=12, tol=1e-3,
    )
    assert len(origins) > 50, f"expected >50 forecast origins, got {len(origins)}"

    # Ground truth: forward-sum of realized (dx)² over 60s on the true
    # latent path.
    realized = causal_forward_sum_variance(path.x, h=60)
    r = np.asarray([realized[t] for t in origins], dtype=float)
    f = np.asarray(preds, dtype=float)

    # Paper Table 1 scaling: logit-increment variance at σ_b=0.05 is
    # O(1e-3) per sample; the paper's MSE/MAE targets live on a ×1e3
    # scale.
    scale = 1e3
    r_s = r * scale
    f_s = f * scale
    err = r_s - f_s
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    ql = qlike(r_s, f_s)

    # Innovation whitening diagnostic (report only).
    q, n_innov = _box_pierce_q(innov[200:], lags=20)

    # Sample the calibration trajectory at beginning / middle / end.
    # Truth: σ_b=cfg.sigma_b, λ=cfg.lambda_per_sec, s_J²=cfg.jump_std².
    sample_idx = (0, len(traj) // 2, len(traj) - 1) if traj else ()
    truth_sig = cfg.sigma_b
    truth_lam = cfg.lambda_per_sec
    truth_sJ2 = cfg.jump_std ** 2

    # Print a compact report the operator can paste into the handoff
    # message. We dump unconditionally via capsys.
    lines = [
        "",
        "[paper §6 replication]",
        f"  forecast origins: {len(origins)}",
        f"  MSE   = {mse:.4f}  (target {TARGET_MSE},  ±{TOL*100:.0f}%)",
        f"  MAE   = {mae:.4f}  (target {TARGET_MAE},  ±{TOL*100:.0f}%)",
        f"  QLIKE = {ql:.4f}  (target {TARGET_QLIKE}, ±{TOL*100:.0f}%)",
        f"[diagnostic] Box-Pierce Q(20)={q:.3f}  (χ²(20,0.95)=31.41, "
        f"n_innov={n_innov})",
        f"[truth] σ_b={truth_sig:.4f}  λ={truth_lam:.4f}  s_J²={truth_sJ2:.4f}",
        "[trajectory sample]  idx    σ̂_b       λ̂        ŝ²_J",
    ]
    for k in sample_idx:
        sig, lam, sJ2 = traj[k]
        lines.append(
            f"                      {k:4d}  {sig:.4f}  {lam:.4f}  {sJ2:.4f}"
        )
    if traj:
        sigs = np.asarray([t[0] for t in traj])
        lams = np.asarray([t[1] for t in traj])
        sJ2s = np.asarray([t[2] for t in traj])
        lines.append(
            f"[trajectory stats]   σ̂_b mean={sigs.mean():.4f} std={sigs.std():.4f}  "
            f"λ̂ mean={lams.mean():.4f}  ŝ²_J mean={sJ2s.mean():.4f}"
        )
    print("\n".join(lines))

    in_tol = (
        abs(mse - TARGET_MSE) <= TOL * TARGET_MSE
        and abs(mae - TARGET_MAE) <= TOL * TARGET_MAE
        and abs(ql - TARGET_QLIKE) <= TOL * TARGET_QLIKE
    )
    assert in_tol, (
        f"Stage-0 gate NOT cleared. "
        f"MSE={mse:.4f} (target {TARGET_MSE}±{TOL*100:.0f}%), "
        f"MAE={mae:.4f} (target {TARGET_MAE}±{TOL*100:.0f}%), "
        f"QLIKE={ql:.4f} (target {TARGET_QLIKE}±{TOL*100:.0f}%). "
        "Do NOT patch any module — report to planning window for diagnosis."
    )


# ---------------------------------------------------------------------------
# Causality / plumbing unit test — runs unconditionally even if the full
# pipeline is slow / fails.
# ---------------------------------------------------------------------------


def test_forward_sum_operator_is_causal() -> None:
    """Sanity check of the evaluation plumbing itself. No Track A code
    involved — if this breaks, the synthetic fixture changed shape."""
    n = 100
    h = 5
    a = np.arange(n, dtype=float)
    from tests.fixtures.synthetic import causal_forward_sum_variance as _cfsv

    out = _cfsv(a, h)
    expected_0 = float(np.sum((np.diff(a, prepend=a[0])[1: 1 + h]) ** 2))
    assert out[0] == pytest.approx(expected_0)
    assert out[-h:].sum() == 0.0
