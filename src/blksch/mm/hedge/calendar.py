"""Calendar (variance-strip) hedge — paper §4.3, Stage 3.

Two-leg template. Given the book's aggregate belief-vol sensitivity
ν̂_b(t, Δ) over a near-dated window, size an x-variance strip at notional:

    N^{x-var}  ≈  - ν̂_b(t, Δ) / (∂K^{x-var}_{t,t+Δ}/∂σ_b)
               ∝  - ν̂_b / σ_b

Sign convention (paper §4.3):
  * ν̂_b > 0 → book already long belief-vol → SHORT x-variance strip
  * ν̂_b < 0 → book short belief-vol → LONG x-variance strip

Boundary guards:
  * σ_b < `sigma_b_floor` (usually `config.boundary.eps`): σ_b is unreliable
    near the boundary — clamp so hedge notional cannot explode.
  * ν̂_b = 0 → no hedge (zero notional; refresh_loop drops it).

Polymarket does not list x-variance strips natively. Stage 3's `synth_strip.py`
will turn each HedgeInstruction emitted here into a basket of vanilla contracts
at adjacent moneyness and maturity (paper §3.4). Until then we tag the hedge
token as `"{token_id}:xvar"` — a synthetic placeholder the order router must
refuse to route if synth_strip isn't wired.

The function is pure: no IO, no state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from blksch.schemas import HedgeInstruction, HedgeSide, SurfacePoint

from .synth_strip import SynthStripParams, explode_hedge_into_basket

__all__ = [
    "CalendarHedgeParams",
    "compute_calendar_hedge",
    "resolve_with_synth_strip",
    "xvar_synth_token_id",
]


def xvar_synth_token_id(token_id: str) -> str:
    """Naming convention for the synthetic x-variance hedge leg on a token."""
    return f"{token_id}:xvar"


class CalendarHedgeParams:
    """Knobs for the calendar-hedge sizing.

    - `sigma_b_floor`: lower bound on σ_b used in the denominator. Prevents
      N^{x-var} from blowing up when the surface reports a near-zero belief
      vol (e.g. pinned markets).
    - `max_abs_notional_usd`: absolute cap on |N^{x-var}|. A single bad
      ν̂_b read cannot create a runaway hedge.
    """

    __slots__ = ("sigma_b_floor", "max_abs_notional_usd")

    def __init__(
        self, sigma_b_floor: float = 1.0e-4, max_abs_notional_usd: float = 1_000.0
    ) -> None:
        if sigma_b_floor <= 0.0:
            raise ValueError(f"sigma_b_floor must be positive, got {sigma_b_floor}")
        if max_abs_notional_usd <= 0.0:
            raise ValueError(
                f"max_abs_notional_usd must be positive, got {max_abs_notional_usd}"
            )
        self.sigma_b_floor = sigma_b_floor
        self.max_abs_notional_usd = max_abs_notional_usd


def compute_calendar_hedge(
    surface: SurfacePoint,
    inventory_nu_b: float,
    *,
    params: CalendarHedgeParams | None = None,
    ts: datetime | None = None,
) -> HedgeInstruction:
    """Return the variance-strip HedgeInstruction for the given book ν̂_b.

    Args:
      surface:          SurfacePoint giving the current σ_b we divide by
      inventory_nu_b:   aggregate belief-vol sensitivity of the book (signed);
                        ν̂_b > 0 means the book profits from σ_b increases

    Returns a HedgeInstruction with reason="calendar":
      * notional_usd = |N^{x-var}|  (always ≥ 0)
      * side         = SHORT when ν̂_b > 0, LONG when ν̂_b < 0
      * zero-notional instruction when ν̂_b == 0; refresh_loop drops it
    """
    p = params if params is not None else CalendarHedgeParams()
    if ts is None:
        ts = datetime.now(tz=timezone.utc)

    base_instr = HedgeInstruction(
        source_token_id=surface.token_id,
        hedge_token_id=xvar_synth_token_id(surface.token_id),
        side=HedgeSide.SHORT,  # overwritten below; meaningless at zero notional
        notional_usd=0.0,
        reason="calendar",
        ts=ts,
    )

    if inventory_nu_b == 0.0:
        return base_instr

    sigma_b = max(surface.sigma_b, p.sigma_b_floor)
    # N^{x-var} ≈ -ν̂_b / σ_b
    raw_notional = -inventory_nu_b / sigma_b
    abs_notional = min(abs(raw_notional), p.max_abs_notional_usd)
    side = HedgeSide.LONG if raw_notional > 0.0 else HedgeSide.SHORT

    return base_instr.model_copy(update={"notional_usd": abs_notional, "side": side})


def resolve_with_synth_strip(
    instruction: HedgeInstruction,
    surface_points: Iterable[SurfacePoint],
    target_tau: float,
    target_m: float,
    *,
    params: SynthStripParams | None = None,
) -> list[HedgeInstruction]:
    """Turn a synthetic `{tok}:xvar` calendar HedgeInstruction into a basket
    of concrete-token HedgeInstructions the order router can handle (paper §3.4).

    Usage from refresh_loop step 6:

        legs = resolve_with_synth_strip(
            instruction=compute_calendar_hedge(...),
            surface_points=snap.surface_neighborhood,
            target_tau=snap.surface.tau,
            target_m=snap.surface.m,
            params=config.synth_strip_params,
        )
        for leg in legs:
            await hedge_sink(leg)

    Returns [] when the input is zero-notional or when no surface neighbors
    are available — the caller should fall back to emitting the synthetic
    instruction only if the router declares it handles `:xvar` tokens
    (Stage 3 router work).
    """
    return explode_hedge_into_basket(
        instruction=instruction,
        surface_points=surface_points,
        target_tau=target_tau,
        target_m=target_m,
        params=params,
    )
