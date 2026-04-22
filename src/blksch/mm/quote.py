"""Inventory-aware quoting in logit units (paper §4.2 eq 8-9).

Avellaneda-Stoikov transplanted to the event-contract kernel:

    r_x(t)     = x_t  -  q_t γ σ̄²_b (T-t)                     (8)
    2 δ_x(t)   ≈ γ σ̄²_b (T-t)  +  (2/k) log(1 + γ/k)           (9)
    x_bid      = r_x - δ_x
    x_ask      = r_x + δ_x
    p_{bid,ask} = S(x_{bid,ask})

Boundary handling:
    δ_p ≈ S'(x) · δ_x    — floor at `delta_p_floor`
    |q_t| ≤ q_max ∝ 1/max(S'(x), ε)

This module is intentionally a pure function: the refresh loop feeds it
current state, it returns a Quote. No IO, no mutation, no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from blksch.schemas import Quote

from .greeks import sigmoid, s_prime

__all__ = ["QuoteParams", "compute_quote", "reservation_x", "half_spread_x", "q_max"]


@dataclass(frozen=True)
class QuoteParams:
    """Tunable knobs from config/bot.yaml — one place, so tests can pin them."""

    gamma: float = 0.1
    k: float = 1.5
    eps: float = 1.0e-5
    delta_p_floor: float = 0.01  # in probability units (e.g. 1 tick = 0.01)
    q_max_base: float = 50.0  # baseline cap in "contracts"; scales by 1/max(S'(x), eps)
    q_max_shrink: float = 1.0  # multiplier ≤ 1 tightens the cap
    default_size: float = 10.0

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if not (0 < self.eps < 0.5):
            raise ValueError(f"eps must be in (0, 0.5), got {self.eps}")
        if self.delta_p_floor < 0:
            raise ValueError(f"delta_p_floor must be non-negative, got {self.delta_p_floor}")


def reservation_x(
    x_t: float, q_t: float, sigma_b: float, time_to_horizon_sec: float, gamma: float
) -> float:
    """r_x = x_t - q_t · γ · σ̄_b² · (T-t)  — paper eq (8).

    (T-t) is in seconds; σ_b is belief vol per sqrt(sec), so σ_b²(T-t) has
    units of variance and matches the logit-units balance sheet.
    """
    return x_t - q_t * gamma * (sigma_b * sigma_b) * max(time_to_horizon_sec, 0.0)


def half_spread_x(
    sigma_b: float, time_to_horizon_sec: float, gamma: float, k: float
) -> float:
    """δ_x = ½ [ γ σ̄_b² (T-t)  +  (2/k) log(1 + γ/k) ]  — paper eq (9)."""
    risk_term = gamma * (sigma_b * sigma_b) * max(time_to_horizon_sec, 0.0)
    flow_term = (2.0 / k) * math.log1p(gamma / k)
    return 0.5 * (risk_term + flow_term)


def q_max(x_t: float, params: QuoteParams) -> float:
    """Inventory cap ∝ 1/max(S'(x), ε) — tightens as p→0,1 (paper §4.2)."""
    denom = max(s_prime(x_t), params.eps)
    return params.q_max_base * params.q_max_shrink / denom


def _apply_p_floor(x_bid: float, x_ask: float, r_x: float, delta_p_floor: float) -> tuple[float, float]:
    """If S(x_ask) - S(x_bid) < 2·δ_p_floor, widen symmetrically around r_x.

    Near the boundary S'(x) collapses, so δ_p = S(r_x+δ)-S(r_x-δ) shrinks
    even for a healthy δ_x. The paper's remedy: floor the display spread at
    the tick grid. We root-find the δ_x that produces δ_p = floor, expanding
    outward from the current δ_x.
    """
    p_bid = sigmoid(x_bid)
    p_ask = sigmoid(x_ask)
    displayed = p_ask - p_bid
    if displayed >= 2.0 * delta_p_floor:
        return x_bid, x_ask

    # Bisect on δ such that S(r+δ) - S(r-δ) == 2·δ_p_floor.
    target = 2.0 * delta_p_floor
    lo = max(x_ask - r_x, 1.0e-9)
    hi = max(lo * 2.0, 1.0)
    # Expand hi until the spread exceeds target (caps at ~40 logit units, effectively the whole domain).
    for _ in range(60):
        if sigmoid(r_x + hi) - sigmoid(r_x - hi) >= target:
            break
        hi *= 2.0
        if hi > 40.0:
            hi = 40.0
            break

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if sigmoid(r_x + mid) - sigmoid(r_x - mid) >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0e-9:
            break
    delta = hi
    return r_x - delta, r_x + delta


def compute_quote(
    *,
    token_id: str,
    x_t: float,
    sigma_b: float,
    time_to_horizon_sec: float,
    inventory_q: float,
    params: QuoteParams,
    size_bid: float | None = None,
    size_ask: float | None = None,
    ts: datetime | None = None,
    spread_widen_factor: float = 1.0,
) -> Quote:
    """Build a Quote from surface state + inventory (paper §4.2).

    Args:
      x_t:                filtered logit (from Track A LogitState)
      sigma_b:            belief vol at (τ, m=x) from SurfacePoint
      time_to_horizon_sec: (T - t); for event contracts use time-to-resolution
                          (paper §4.2 T_horizon_mode). Guard against 0.
      inventory_q:        signed position in "contracts" (shares if 1 contract = 1 share)
      spread_widen_factor: toxicity / news guard multiplier on δ_x (≥ 1.0)

    Returns Quote with both logit and probability fields populated.
    """
    if spread_widen_factor < 1.0:
        raise ValueError(f"spread_widen_factor must be ≥ 1.0, got {spread_widen_factor}")

    r_x = reservation_x(x_t, inventory_q, sigma_b, time_to_horizon_sec, params.gamma)
    delta_x = half_spread_x(sigma_b, time_to_horizon_sec, params.gamma, params.k)
    delta_x *= spread_widen_factor

    x_bid = r_x - delta_x
    x_ask = r_x + delta_x

    x_bid, x_ask = _apply_p_floor(x_bid, x_ask, r_x, params.delta_p_floor)

    p_bid = sigmoid(x_bid)
    p_ask = sigmoid(x_ask)

    # Clip p to valid Polymarket tick domain (leaving boundary buffer).
    p_bid = max(params.eps, min(1.0 - params.eps, p_bid))
    p_ask = max(params.eps, min(1.0 - params.eps, p_ask))

    final_half_x = 0.5 * (x_ask - x_bid)
    size_b = params.default_size if size_bid is None else size_bid
    size_a = params.default_size if size_ask is None else size_ask

    if ts is None:
        ts = datetime.now(tz=timezone.utc)

    return Quote(
        token_id=token_id,
        p_bid=p_bid,
        p_ask=p_ask,
        x_bid=x_bid,
        x_ask=x_ask,
        size_bid=size_b,
        size_ask=size_a,
        half_spread_x=final_half_x,
        reservation_x=r_x,
        inventory_q=inventory_q,
        ts=ts,
    )
