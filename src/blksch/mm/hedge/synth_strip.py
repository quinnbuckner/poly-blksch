"""Synthetic variance-strip replication (paper §3.4) — Stage 3.

Polymarket does not list x-variance strips natively. This module turns the
synthetic `{token_id}:xvar` leg emitted by `calendar.py` into a basket of
real vanilla event contracts at adjacent moneyness m and maturity τ.

Math (paper §3.4, short-maturity approximation, eq 6):

    K^{p-var}_{t, t+Δ}  ≈  (p(1-p))^2 · ∫_t^{t+Δ} σ_b^2 du  +  jump term
                        =  S'(x_t)^2 · ∫ σ_b^2 du   (diffusion piece)

For static replication, the key observation is that the *marginal sensitivity*
to σ_b² of a portfolio of vanilla contracts at adjacent (m_k, τ_k) is
approximately

    Σ_k w_k · ν_b(m_k, τ_k)   ≈   target ν^{x-var}_b   =   σ_b(target)

when the `w_k` are a normalized local kernel centered at (m*, τ*). The paper's
Lemma in §3.4 bounds the tracking error of this construction at short maturity
by O(Δ^{3/2}) when the basket step in m is commensurate with the local
belief-vol scale.

Concretely we use a Gaussian kernel in (m - m*, log(τ/τ*)) space:

    w_k  ∝  exp(-½ (Δm_k / h_m)^2 - ½ (Δlogτ_k / h_τ)^2)

normalized to Σ w_k = 1, then scaled by the target notional. Weights
approaching zero (relative tolerance) are trimmed to keep the basket sparse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from blksch.schemas import HedgeInstruction, HedgeSide, SurfacePoint

__all__ = [
    "BasketLeg",
    "SynthStripParams",
    "replicate_xvariance_strip",
    "explode_hedge_into_basket",
]


@dataclass(frozen=True)
class BasketLeg:
    """One leg of the synthetic variance basket.

    `weight` is signed and has notional units (target_notional_usd); the
    refresh loop multiplies by its own sign when turning legs into
    HedgeInstructions.
    """

    token_id: str
    weight: float
    m: float
    tau: float


@dataclass(frozen=True)
class SynthStripParams:
    """Knobs for the replication kernel.

    - bandwidth_m:   Gaussian kernel scale in moneyness (logit) units.
    - bandwidth_log_tau: Gaussian kernel scale in log-maturity units.
    - max_basket_size: cap on the number of legs kept after weight-trimming.
    - weight_floor_ratio: drop legs whose |weight|/|max_weight| < this.
    - max_moneyness_dist: hard skip on points further than this in |m-m*|.
    - max_log_tau_dist:   hard skip on points further than this in |log(τ/τ*)|.
    """

    bandwidth_m: float = 0.5
    bandwidth_log_tau: float = 0.5
    max_basket_size: int = 5
    weight_floor_ratio: float = 1.0e-3
    max_moneyness_dist: float = 3.0
    max_log_tau_dist: float = 2.0

    def __post_init__(self) -> None:
        if self.bandwidth_m <= 0.0:
            raise ValueError(f"bandwidth_m must be positive, got {self.bandwidth_m}")
        if self.bandwidth_log_tau <= 0.0:
            raise ValueError(
                f"bandwidth_log_tau must be positive, got {self.bandwidth_log_tau}"
            )
        if self.max_basket_size <= 0:
            raise ValueError(f"max_basket_size must be positive, got {self.max_basket_size}")
        if not 0.0 <= self.weight_floor_ratio < 1.0:
            raise ValueError(
                f"weight_floor_ratio must be in [0, 1), got {self.weight_floor_ratio}"
            )


# ---------------------------------------------------------------------------
# Core replication
# ---------------------------------------------------------------------------


def replicate_xvariance_strip(
    surface_points: Iterable[SurfacePoint],
    target_tau: float,
    target_m: float,
    target_variance_notional: float,
    *,
    params: SynthStripParams | None = None,
    exclude_token_id: str | None = None,
) -> list[BasketLeg]:
    """Static replication of an x-variance strip by a basket of vanilla contracts.

    Args:
      surface_points:           available neighboring SurfacePoint candidates
      target_tau, target_m:     (τ*, m*) of the strip being replicated
      target_variance_notional: N^{x-var} notional to replicate (USD)
      exclude_token_id:         when set, drop the target's own token from the
                                basket (self-referential weight is not routable)

    Returns:
      A list of BasketLeg whose weights sum to `target_variance_notional` (up
      to the weight-floor trimming) when `target_variance_notional != 0`.
      Empty list when:
        * `target_variance_notional == 0`
        * no candidate surface points remain after filtering
        * all candidates are outside the bandwidth cutoff

    The caller owns scaling/sign conventions — weights are positive on the
    long-variance side; `explode_hedge_into_basket` below does the flip when
    the hedge instruction is SHORT.
    """
    p = params or SynthStripParams()
    if target_variance_notional == 0.0:
        return []
    if target_tau <= 0.0:
        raise ValueError(f"target_tau must be positive, got {target_tau}")

    log_tau_star = math.log(target_tau)

    # Build (distance² in kernel units, point) list, filtered by hard cutoff.
    candidates: list[tuple[float, SurfacePoint]] = []
    for sp in surface_points:
        if exclude_token_id is not None and sp.token_id == exclude_token_id:
            continue
        if sp.tau <= 0.0:
            continue
        dm = sp.m - target_m
        dlog_tau = math.log(sp.tau) - log_tau_star
        if abs(dm) > p.max_moneyness_dist:
            continue
        if abs(dlog_tau) > p.max_log_tau_dist:
            continue
        d2 = 0.5 * ((dm / p.bandwidth_m) ** 2 + (dlog_tau / p.bandwidth_log_tau) ** 2)
        candidates.append((d2, sp))

    if not candidates:
        return []

    # Raw Gaussian weights (subtract min d² for numerical stability — the
    # partition-of-unity normalization is invariant to this shift).
    min_d2 = min(d2 for d2, _ in candidates)
    raw = [(math.exp(-(d2 - min_d2)), sp) for d2, sp in candidates]
    total = sum(w for w, _ in raw)
    if total <= 0.0:
        return []
    normalized = [(w / total, sp) for w, sp in raw]

    # Sort by weight descending and keep the top-K, trim below weight floor.
    normalized.sort(key=lambda ws: -ws[0])
    top = normalized[: p.max_basket_size]
    max_w = top[0][0]
    threshold = max_w * p.weight_floor_ratio
    trimmed = [(w, sp) for w, sp in top if w >= threshold]

    # Renormalize after trimming so partition-of-unity still holds.
    post_total = sum(w for w, _ in trimmed)
    if post_total <= 0.0:
        return []

    scale = target_variance_notional / post_total
    return [
        BasketLeg(token_id=sp.token_id, weight=w * scale, m=sp.m, tau=sp.tau)
        for w, sp in trimmed
    ]


# ---------------------------------------------------------------------------
# Hedge-instruction bridge
# ---------------------------------------------------------------------------


def explode_hedge_into_basket(
    instruction: HedgeInstruction,
    surface_points: Iterable[SurfacePoint],
    target_tau: float,
    target_m: float,
    *,
    params: SynthStripParams | None = None,
) -> list[HedgeInstruction]:
    """Translate a synthetic `{tok}:xvar` calendar-hedge instruction into a
    basket of concrete-token HedgeInstructions that the order router can handle.

    The input instruction's `notional_usd` is treated as the target variance
    notional; the output legs' notionals are |weight_k| and each leg's side
    is the input side flipped iff weight_k < 0 (signed weights allow mixed
    long/short legs when the kernel ever produces them).

    Returns [] for a zero-notional input (nothing to route).
    """
    if instruction.notional_usd <= 0.0:
        return []
    legs = replicate_xvariance_strip(
        surface_points=surface_points,
        target_tau=target_tau,
        target_m=target_m,
        target_variance_notional=instruction.notional_usd,
        params=params,
        exclude_token_id=instruction.source_token_id,
    )
    out: list[HedgeInstruction] = []
    base_side = instruction.side
    flipped = HedgeSide.LONG if base_side is HedgeSide.SHORT else HedgeSide.SHORT
    for leg in legs:
        if leg.weight == 0.0:
            continue
        side = base_side if leg.weight > 0.0 else flipped
        out.append(
            HedgeInstruction(
                source_token_id=instruction.source_token_id,
                hedge_token_id=leg.token_id,
                side=side,
                notional_usd=abs(leg.weight),
                reason="synth_strip",
                ts=instruction.ts,
            )
        )
    return out
