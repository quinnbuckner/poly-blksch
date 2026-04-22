"""Cross-event β-hedge (paper §4.4) — Stage 2.

For hedging event i with event j over short horizons,

    β_{i←j} ≈ Cov(dp_i, dp_j) / Var(dp_j)  ≈  S'_i / S'_j · ρ_ij

Shrinkage  β̃ = α · β  with α ∈ [0.5, 1] (default 0.75) dampens the naive ratio
(see paper §4.4 — α < 1 trims noise, especially around regime changes).

Near the boundary of either event (p_k → 0 or 1) S'_k collapses and β explodes;
we clamp |β̃| at a configurable `max_abs_beta` and short-circuit to zero when
either S' falls below `s_prime_floor`.

Co-jump correction (paper §4.4):

    Δβ^{jump} = ∫ Δp_i Δp_j ν_{ij,t}(dz_i, dz_j) / ( (S'_j)^2 · σ_b^{j,2} )

We approximate the double integral using the second-moment summary shipped in
`CorrelationEntry.co_jump_m2` (which Track A defines as E[(Δp_i · Δp_j)] under
the co-jump measure, aggregated) scaled by the co-jump intensity `co_jump_lambda`.

The function is pure: no IO, no state. `refresh_loop.py` step 5 consumes the
returned `HedgeInstruction` when the `hedge_enabled` config flag is on.
"""

from __future__ import annotations

from datetime import datetime, timezone

from blksch.schemas import CorrelationEntry, HedgeInstruction, HedgeSide, SurfacePoint

from ..greeks import s_prime

__all__ = [
    "BetaHedgeParams",
    "compute_beta_hedge",
    "raw_beta",
    "co_jump_correction",
]


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


def raw_beta(sp_target: float, sp_hedge: float, rho: float) -> float:
    """β_{i←j} = S'_i / S'_j · ρ_ij — unclamped, no shrinkage.

    Callers are expected to guard against `sp_hedge → 0`. `compute_beta_hedge`
    wraps this with the appropriate clamps.
    """
    if sp_hedge == 0.0:
        raise ZeroDivisionError("hedge S'(x) is zero; caller must guard with s_prime_floor")
    return (sp_target / sp_hedge) * rho


def co_jump_correction(
    co_jump_lambda: float, co_jump_m2: float, sp_hedge: float, sigma_b_hedge: float
) -> float:
    """Δβ^{jump} ≈ (λ_ij · E[Δp_i·Δp_j | jump]) / (S'_j)^2 · σ_b^{j,2}  (paper §4.4).

    Returns 0 if the denominator vanishes (near-boundary) — the caller will
    already have short-circuited via s_prime_floor in that case.
    """
    denom = (sp_hedge * sp_hedge) * (sigma_b_hedge * sigma_b_hedge)
    if denom <= 0.0:
        return 0.0
    return (co_jump_lambda * co_jump_m2) / denom


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


class BetaHedgeParams:
    """Knobs for the hedge ratio.

    - alpha:           shrinkage in [0.5, 1.0]; paper §4.4 recommends α < 1
                       to trim noise; α closer to 1 pre-announcement.
    - s_prime_floor:   when either S'_i or S'_j < floor, β is unreliable —
                       zero out the hedge. Matches `config.boundary.eps`.
    - max_abs_beta:    hard clamp on |β̃| so a single ρ̂ outlier can't blow
                       out the book.
    - apply_co_jump:   off by default; on near known announcement windows.
    """

    __slots__ = ("alpha", "s_prime_floor", "max_abs_beta", "apply_co_jump")

    def __init__(
        self,
        alpha: float = 0.75,
        s_prime_floor: float = 1.0e-4,
        max_abs_beta: float = 5.0,
        apply_co_jump: bool = False,
    ) -> None:
        if not 0.5 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.5, 1.0], got {alpha}")
        if s_prime_floor <= 0.0:
            raise ValueError(f"s_prime_floor must be positive, got {s_prime_floor}")
        if max_abs_beta <= 0.0:
            raise ValueError(f"max_abs_beta must be positive, got {max_abs_beta}")
        self.alpha = alpha
        self.s_prime_floor = s_prime_floor
        self.max_abs_beta = max_abs_beta
        self.apply_co_jump = apply_co_jump

    def __repr__(self) -> str:
        return (
            f"BetaHedgeParams(alpha={self.alpha}, s_prime_floor={self.s_prime_floor}, "
            f"max_abs_beta={self.max_abs_beta}, apply_co_jump={self.apply_co_jump})"
        )


def compute_beta_hedge(
    target: SurfacePoint,
    hedge: SurfacePoint,
    corr: CorrelationEntry,
    alpha: float,
    *,
    params: BetaHedgeParams | None = None,
    target_notional_usd: float = 1.0,
    ts: datetime | None = None,
) -> HedgeInstruction:
    """Return the per-unit-exposure hedge ratio β̃ as a HedgeInstruction.

    Semantics of the returned object:
      - `notional_usd` = |β̃| · target_notional_usd.  With default
        target_notional_usd=1.0, the caller receives the raw hedge ratio
        and can scale by actual target exposure before routing to the venue.
      - `side`  = SHORT if β̃ > 0 (hedge moves with target — sell the hedge
                  to offset the long), LONG if β̃ < 0 (anti-correlated — buy
                  the hedge to offset).
      - The CorrelationEntry must reference the same (target, hedge) pair
        in either order; we flip ρ's sign if the roles are reversed
        (ρ_ij = ρ_ji, so no sign change; included as a consistency check).
      - If either S' < s_prime_floor OR α is forced into clamp, returns a
        zero-notional instruction with reason="beta" — the refresh_loop
        treats a zero-notional hedge as "no action".

    `alpha` is accepted as a loose argument to preserve the signature
    contracted with refresh_loop; if `params` is supplied, its alpha wins
    (lets tests pin the full param object when needed).
    """
    p = params if params is not None else BetaHedgeParams(alpha=alpha)
    # Resolve effective alpha.
    eff_alpha = p.alpha

    if ts is None:
        ts = datetime.now(tz=timezone.utc)

    # Consistency: the correlation entry's token pair must match our pair
    # (in either direction). `rho` is symmetric, so we just verify membership.
    pair = {corr.token_id_i, corr.token_id_j}
    if pair != {target.token_id, hedge.token_id}:
        raise ValueError(
            f"CorrelationEntry pair {pair} does not match "
            f"(target={target.token_id}, hedge={hedge.token_id})"
        )

    # SurfacePoint.m is the moneyness coordinate; paper uses m = x (logit).
    sp_i = s_prime(target.m)
    sp_j = s_prime(hedge.m)

    # Near-boundary short-circuit: if either leg's Δ is tiny, β̃ is unreliable.
    if sp_i < p.s_prime_floor or sp_j < p.s_prime_floor:
        return HedgeInstruction(
            source_token_id=target.token_id,
            hedge_token_id=hedge.token_id,
            side=HedgeSide.SHORT,  # side is meaningless at zero notional
            notional_usd=0.0,
            reason="beta",
            ts=ts,
        )

    beta = raw_beta(sp_i, sp_j, corr.rho)
    beta_shrunk = eff_alpha * beta

    if p.apply_co_jump and corr.co_jump_lambda > 0.0 and corr.co_jump_m2 > 0.0:
        beta_shrunk += co_jump_correction(
            corr.co_jump_lambda, corr.co_jump_m2, sp_j, hedge.sigma_b
        )

    # Hard clamp.
    if beta_shrunk > p.max_abs_beta:
        beta_shrunk = p.max_abs_beta
    elif beta_shrunk < -p.max_abs_beta:
        beta_shrunk = -p.max_abs_beta

    # Notional is always non-negative; direction goes into `side`.
    # β̃ > 0  (positively correlated)      ⇒  hedge has same sign risk as target.
    #                                         To offset a long target, SHORT hedge.
    # β̃ < 0  (negatively correlated)      ⇒  target and hedge move opposite ways.
    #                                         To offset a long target, LONG hedge
    #                                         (the hedge rises when the target falls).
    notional = abs(beta_shrunk) * abs(target_notional_usd)
    side = HedgeSide.SHORT if beta_shrunk >= 0.0 else HedgeSide.LONG

    # If target_notional_usd is signed (short target), flip the side.
    if target_notional_usd < 0.0:
        side = HedgeSide.LONG if side is HedgeSide.SHORT else HedgeSide.SHORT

    return HedgeInstruction(
        source_token_id=target.token_id,
        hedge_token_id=hedge.token_id,
        side=side,
        notional_usd=notional,
        reason="beta",
        ts=ts,
    )
