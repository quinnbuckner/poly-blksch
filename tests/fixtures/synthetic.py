"""Synthetic path generators for Track A calibration tests.

The Stage 0 correctness gate (``tests/pipeline/test_paper_sec6_replication.py``)
needs an RN-consistent logit jump-diffusion path matching the paper's §6 setup:
N=6000 steps at 1 Hz, symmetric Gaussian jumps, scheduled jump windows, and
heteroskedastic microstructure noise.

These generators are the ground truth — tests validate that Track A's filter
and EM recover the parameters we seeded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sprime(x: np.ndarray) -> np.ndarray:
    p = sigmoid(x)
    return p * (1.0 - p)


def sdprime(x: np.ndarray) -> np.ndarray:
    p = sigmoid(x)
    return p * (1.0 - p) * (1.0 - 2.0 * p)


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters for the paper's §6 synthetic path.

    Defaults mirror paper §6.2/§6.4: 1 Hz grid, N=6000, symmetric Gaussian jumps,
    and a mild scheduled-jump window near index 4000 (simulating a news print).
    """

    n_steps: int = 6000
    dt_sec: float = 1.0
    sigma_b: float = 0.05  # belief volatility (per sqrt-sec) in logit space
    lambda_per_sec: float = 0.005  # jump intensity
    jump_std: float = 0.08  # symmetric Gaussian jump std
    sched_jump_index: int = 4000  # scheduled jump window center
    sched_jump_halfwidth: int = 30  # +/- seconds
    sched_jump_lambda_boost: float = 0.05  # extra intensity in window
    initial_x: float = 0.0  # start near p = 0.5
    sprime_clip: float = 1.0e-4
    mu_cap_per_sec: float = 0.25
    rng_seed: int = 42


class SyntheticPath(NamedTuple):
    x: np.ndarray  # true latent logit, length N
    jumps: np.ndarray  # jump indicator (1 if jumped this step)
    jump_sizes: np.ndarray  # jump magnitude at each step (0 if no jump)
    sigma_b_arr: np.ndarray  # realized sigma_b per step (constant in baseline)
    lambda_arr: np.ndarray  # realized jump intensity per step
    mu_arr: np.ndarray  # risk-neutral drift per step (implied from eq 3)
    config: SyntheticConfig


def _mu_rn(x: float, sigma_b: float, lam: float, jump_std: float, cfg: SyntheticConfig) -> float:
    """Risk-neutral drift on x_t implied by the martingale restriction (paper §3.2 eq 3).

    Closed-form for symmetric Gaussian jumps is not available, so we Monte Carlo
    the jump compensator with a modest draw count — this is ground truth for
    the test, precision is cheap.
    """
    if lam <= 0.0 or jump_std <= 0.0:
        jump_term = 0.0
    else:
        rng = np.random.default_rng(seed=0)
        z = rng.normal(0.0, jump_std, size=2000)
        sx = 1.0 / (1.0 + math.exp(-x))
        sxz = 1.0 / (1.0 + np.exp(-(x + z)))
        chi = z * (np.abs(z) < 1.0)
        jump_term = float(lam * np.mean(sxz - sx - sx * (1.0 - sx) * chi))

    sp = max(sigmoid_scalar_prime(x), cfg.sprime_clip)
    sdp = sigmoid_scalar_dprime(x)
    mu = -(0.5 * sdp * sigma_b * sigma_b + jump_term) / sp
    # cap for numerical stability (matches §6.4 note)
    return float(np.clip(mu, -cfg.mu_cap_per_sec, cfg.mu_cap_per_sec))


def sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_scalar_prime(x: float) -> float:
    p = sigmoid_scalar(x)
    return p * (1.0 - p)


def sigmoid_scalar_dprime(x: float) -> float:
    p = sigmoid_scalar(x)
    return p * (1.0 - p) * (1.0 - 2.0 * p)


def generate_rn_consistent_path(cfg: SyntheticConfig | None = None) -> SyntheticPath:
    """Generate an RN-consistent jump-diffusion path in logit space.

    The drift at each step is computed from the martingale restriction using
    the current ``sigma_b`` and ``lambda`` — this is the ground truth Track A
    is supposed to recover.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.rng_seed)
    n = cfg.n_steps
    dt = cfg.dt_sec

    x = np.zeros(n)
    x[0] = cfg.initial_x
    jumps = np.zeros(n, dtype=np.int8)
    jump_sizes = np.zeros(n)
    sigma_b_arr = np.full(n, cfg.sigma_b)
    lambda_arr = np.full(n, cfg.lambda_per_sec)
    mu_arr = np.zeros(n)

    # scheduled jump window boosts lambda
    lo = cfg.sched_jump_index - cfg.sched_jump_halfwidth
    hi = cfg.sched_jump_index + cfg.sched_jump_halfwidth
    lambda_arr[lo:hi] += cfg.sched_jump_lambda_boost

    for t in range(1, n):
        lam = lambda_arr[t]
        sig = sigma_b_arr[t]
        mu = _mu_rn(x[t - 1], sig, lam, cfg.jump_std, cfg)
        mu_arr[t] = mu

        # diffusion
        dW = rng.normal(0.0, math.sqrt(dt))
        diff = sig * dW

        # jump with prob lambda * dt
        jumped = rng.uniform() < lam * dt
        jz = rng.normal(0.0, cfg.jump_std) if jumped else 0.0
        jumps[t] = 1 if jumped else 0
        jump_sizes[t] = jz

        x[t] = x[t - 1] + mu * dt + diff + jz

    return SyntheticPath(
        x=x,
        jumps=jumps,
        jump_sizes=jump_sizes,
        sigma_b_arr=sigma_b_arr,
        lambda_arr=lambda_arr,
        mu_arr=mu_arr,
        config=cfg,
    )


def inject_microstructure_noise(
    x: np.ndarray,
    spread: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    rng_seed: int = 43,
) -> tuple[np.ndarray, np.ndarray]:
    """Add heteroskedastic microstructure noise per paper §5.1 eq (10).

    Returns (y, sigma_eta2) where y = x + eta and sigma_eta2 varies with
    observable frictions (spread, inverse depth). If ``spread`` and ``depth``
    aren't provided, uses randomly varying regimes.
    """
    n = len(x)
    rng = np.random.default_rng(rng_seed)
    if spread is None:
        # base spread with bursts
        spread = 0.01 + 0.005 * np.abs(rng.standard_normal(n))
    if depth is None:
        depth = 500.0 + 200.0 * rng.uniform(size=n)

    a0, a1, a2 = 1e-6, 0.5, 2.0
    sigma_eta2 = a0 + a1 * spread * spread + a2 / depth
    noise = rng.normal(0.0, np.sqrt(sigma_eta2))
    y = x + noise
    return y, sigma_eta2


def causal_forward_sum_variance(x_hat: np.ndarray, h: int) -> np.ndarray:
    """Compute realized forward-sum logit variance over horizon h (paper §6.1)."""
    dx = np.diff(x_hat, prepend=x_hat[0])
    dx2 = dx * dx
    n = len(x_hat)
    out = np.zeros(n)
    for t in range(n - h):
        out[t] = dx2[t + 1 : t + 1 + h].sum()
    return out


def qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    """QLIKE loss from paper eq (13)."""
    mask = (realized > 0) & (forecast > 0)
    if not mask.any():
        return float("nan")
    r = realized[mask]
    f = forecast[mask]
    return float(np.mean(r / f - np.log(r / f) - 1.0))
