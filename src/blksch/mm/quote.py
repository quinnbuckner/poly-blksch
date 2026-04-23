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

Two pathological regimes the naive formula trips on — both fixed here:

  * **Extreme-skew reservation** (paper §4.2 footnote, pathology flagged by
    fuzz): when |q·γ·σ²·T| pushes r_x past ~±12 logit units, sigmoid(x_bid)
    and sigmoid(x_ask) both saturate to ε (or 1-ε) and the final clip
    collapses them to p_bid == p_ask. The model is telling us "inventory is
    so skewed, don't post a two-sided quote". We detect the condition and
    return a one-sided quote with the pulled side at the boundary, the
    other side widened by 2·δ_x (paper §4.2 permits one-sided quoting
    under inventory-cap stress). The `mm/limits.py` inventory-cap kill-
    switch should fire *before* this condition in production; one-sided
    quoting is the graceful fallback when it doesn't.

  * **Boundary ε-shaving in _apply_p_floor**: the bisection previously
    solved for a symmetric δ around r_x, then the final (ε, 1-ε) clip
    ate ε off the cramped side. Now we solve directly in p-space with
    the unit-interval clip baked into the bisection invariants
    (x_bid ≥ logit(ε), x_ask ≤ logit(1-ε)), so the floor is achieved
    asymmetrically when one side is boundary-pinned.

This module is intentionally a pure function (apart from the structured-
event log call in the one-sided branch): the refresh loop feeds it current
state, it returns a Quote. No IO, no mutation, no network.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from blksch.schemas import Quote

from .greeks import logit, sigmoid, s_prime

__all__ = ["QuoteParams", "compute_quote", "reservation_x", "half_spread_x", "q_max"]

logger = logging.getLogger(__name__)


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
    """r_x = x_t - q_t · γ · σ̄_b² · (T-t)  — paper eq (8)."""
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


def _apply_p_floor(
    p_bid: float, p_ask: float, delta_p_floor: float, eps: float,
) -> tuple[float, float]:
    """Enforce `p_ask - p_bid ≥ 2·delta_p_floor` while keeping both sides in
    [eps, 1-eps]. Works directly in p-space so the unit-interval clip is a
    bisection invariant, not a post-clip patch — fixes the ε-shaving bug
    where the final clip ate ε off the cramped side.

    When one side is pinned at the boundary, the other side widens
    asymmetrically to achieve the full 2·delta_p_floor spread.
    """
    target = 2.0 * delta_p_floor
    spread = p_ask - p_bid
    if spread >= target and p_bid >= eps and p_ask <= 1.0 - eps:
        return p_bid, p_ask

    # Step 1: clip to [eps, 1-eps].
    p_bid = max(p_bid, eps)
    p_ask = min(p_ask, 1.0 - eps)
    spread = p_ask - p_bid
    if spread >= target:
        return p_bid, p_ask

    # Step 2: need to widen by `shortfall`. Distribute symmetrically when
    # there is room on both sides; asymmetrically when one side is pinned.
    shortfall = target - spread
    room_bid = p_bid - eps
    room_ask = (1.0 - eps) - p_ask
    half = shortfall / 2.0

    if room_bid >= half and room_ask >= half:
        p_bid -= half
        p_ask += half
    elif room_bid < half:
        # Bid pinned at ε; use all its room, compensate the rest on the ask.
        needed_on_ask = shortfall - room_bid
        p_bid = eps
        p_ask = min(p_ask + needed_on_ask, 1.0 - eps)
    else:
        # Ask pinned at 1-ε; symmetric case.
        needed_on_bid = shortfall - room_ask
        p_ask = 1.0 - eps
        p_bid = max(p_bid - needed_on_bid, eps)

    return p_bid, p_ask


def _one_sided_quote(
    *,
    r_x: float,
    delta_x: float,
    eps: float,
    delta_p_floor: float,
    pulled_side: str,
) -> tuple[float, float, float, float]:
    """Construct a one-sided quote when the naive two-sided formula would
    produce a collapsed (p_bid == p_ask) quote.

    Returns (x_bid, x_ask, p_bid, p_ask). The pulled side's price sits at
    the boundary (ε or 1-ε); the non-pulled side sits at r_x ± 2·δ_x,
    further clipped to preserve the 2·delta_p_floor ordering invariant.

    `pulled_side` ∈ {"bid", "ask"}.
    """
    # Ordering guard — at minimum, the two sides must be one ε apart even
    # when delta_p_floor is 0. Normally the full 2·delta_p_floor floor applies.
    min_separation = max(2.0 * delta_p_floor, eps)

    if pulled_side == "bid":
        p_bid = eps
        # Ask widens by the δ the bid would have contributed: x_ask = r_x + 2·δ_x.
        x_ask_widened = r_x + 2.0 * delta_x
        p_ask = sigmoid(x_ask_widened)
        p_ask = max(p_ask, p_bid + min_separation)
        p_ask = min(p_ask, 1.0 - eps)
    elif pulled_side == "ask":
        p_ask = 1.0 - eps
        x_bid_widened = r_x - 2.0 * delta_x
        p_bid = sigmoid(x_bid_widened)
        p_bid = min(p_bid, p_ask - min_separation)
        p_bid = max(p_bid, eps)
    else:
        raise ValueError(f"pulled_side must be 'bid' or 'ask', got {pulled_side!r}")

    x_bid = logit(p_bid, eps=eps)
    x_ask = logit(p_ask, eps=eps)
    return x_bid, x_ask, p_bid, p_ask


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
      x_t:                 filtered logit (from Track A LogitState)
      sigma_b:             belief vol at (τ, m=x) from SurfacePoint
      time_to_horizon_sec: (T-t); for event contracts use time-to-resolution
      inventory_q:         signed position in contracts
      spread_widen_factor: toxicity / news guard multiplier on δ_x (≥ 1.0)

    Returns a Quote with both sides populated. In the extreme-skew regime
    (|q·γ·σ²·T| so large that the naive two-sided clip would collapse),
    one side is set to the boundary and `size_{bid,ask}` on that side is
    zeroed so the downstream order router can detect & skip it. A
    structured warning is logged for the dashboard.
    """
    if spread_widen_factor < 1.0:
        raise ValueError(f"spread_widen_factor must be ≥ 1.0, got {spread_widen_factor}")

    r_x = reservation_x(x_t, inventory_q, sigma_b, time_to_horizon_sec, params.gamma)
    delta_x = half_spread_x(sigma_b, time_to_horizon_sec, params.gamma, params.k)
    delta_x *= spread_widen_factor

    x_bid_raw = r_x - delta_x
    x_ask_raw = r_x + delta_x

    # Pull-one-side detection: BOTH sides saturate the SAME boundary.
    # Since x_bid < x_ask always, this reduces to:
    #   pull bid  iff sigmoid(x_ask) < 2·eps        (both ≤ ε)
    #   pull ask  iff sigmoid(x_bid) > 1 - 2·eps    (both ≥ 1-ε)
    # This correctly distinguishes inventory-skew saturation (both on one
    # side, two-sided quote impossible) from ordinary wide-spread quotes
    # (x_bid deeply negative, x_ask deeply positive, still orderable).
    p_bid_raw = sigmoid(x_bid_raw)
    p_ask_raw = sigmoid(x_ask_raw)
    pulled_side: str | None = None
    if p_ask_raw < 2.0 * params.eps:
        pulled_side = "bid"
    elif p_bid_raw > 1.0 - 2.0 * params.eps:
        pulled_side = "ask"

    size_b = params.default_size if size_bid is None else size_bid
    size_a = params.default_size if size_ask is None else size_ask

    if pulled_side is not None:
        x_bid, x_ask, p_bid, p_ask = _one_sided_quote(
            r_x=r_x, delta_x=delta_x, eps=params.eps,
            delta_p_floor=params.delta_p_floor, pulled_side=pulled_side,
        )
        # Zero out size on the pulled side so the router skips it.
        if pulled_side == "bid":
            size_b = 0.0
        else:
            size_a = 0.0
        # Structured warning — the dashboard consumes these.
        logger.warning(
            "quote_one_sided",
            extra={
                "event": "quote_one_sided",
                "token_id": token_id,
                "reason": "extreme_skew_reservation",
                "q": inventory_q,
                "r_x": r_x,
                "pulled_side": pulled_side,
            },
        )
    else:
        # Two-sided path: apply the boundary-aware δ_p floor in p-space.
        p_bid, p_ask = _apply_p_floor(
            p_bid_raw, p_ask_raw, params.delta_p_floor, params.eps,
        )
        x_bid = logit(p_bid, eps=params.eps)
        x_ask = logit(p_ask, eps=params.eps)

    final_half_x = 0.5 * (x_ask - x_bid)

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
