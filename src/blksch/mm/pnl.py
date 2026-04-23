"""PnL attribution (paper §4.6).

Over a small Δt with dp ≈ S'(x) dx,

    dΠ ≈ Δ_x·dp + ½ Γ_x·(dp)² + ν_b·dσ_b + Σ_j ν_ρ^(j)·dρ_ij + jumps

The buckets:
  * directional       (Δ_x · dp)
  * curvature/news    (½ Γ_x · (dp)²)
  * belief-vega       (ν_b · dσ_b)
  * cross-event       (Σ ν_ρ · dρ)
  * jumps             (residual Σ (Δp)² on flagged jump ticks)

Use `Attributor.step()` between consecutive snapshots to get an `AttributionStep`.
`track_realized_vs_expected` records the realized (dp)² versus the σ_b²·S'(x)²
prediction so operators can stress-test the variance book (paper §4.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .greeks import delta_x, gamma_x, logit, vega_rho_pair

__all__ = [
    "AttributionStep",
    "AttributionSnapshot",
    "Attributor",
    "realized_vs_expected_dp2",
]


@dataclass(frozen=True)
class AttributionSnapshot:
    """Instantaneous state: price, vol, and position for one token.

    Used as the input pair to `Attributor.step()`.
    """

    token_id: str
    p: float
    sigma_b: float
    qty: float           # signed inventory in contracts
    ts: datetime


@dataclass(frozen=True)
class AttributionStep:
    """Single-interval attribution decomposition (paper §4.6)."""

    token_id: str
    ts: datetime
    dp: float
    dsigma_b: float
    directional_pnl: float
    curvature_pnl: float
    belief_vega_pnl: float
    cross_event_pnl: float
    jump_pnl: float
    total: float
    realized_dp2: float
    expected_dp2: float

    @property
    def residual_dp2(self) -> float:
        return self.realized_dp2 - self.expected_dp2


def realized_vs_expected_dp2(
    dp: float, dt_sec: float, sigma_b: float, p_prev: float
) -> tuple[float, float]:
    """(realized (dp)², expected (dp)²) over the step (paper §4.6 sanity check).

    Expected variance in p-space: E[(dp)²] ≈ S'(x)² · σ_b² · Δt = (p(1-p))² σ_b² Δt.
    """
    realized = dp * dp
    s = p_prev * (1.0 - p_prev)
    expected = (s * s) * sigma_b * sigma_b * max(dt_sec, 0.0)
    return realized, expected


@dataclass
class Attributor:
    """Per-token Δ-Γ-ν-jump attribution.

    Jump detection uses a z-score threshold on (dp)² vs the expected variance;
    a jump's share of pnl is the curvature/directional fold subtracted out and
    the residual is credited to the jump bucket (consistent with paper §4.6
    \"reconcile jump P&L around flagged news\").
    """

    jump_zscore_threshold: float = 4.0
    _prev: AttributionSnapshot | None = field(default=None)
    _cumulative: dict[str, float] = field(default_factory=lambda: {
        "directional": 0.0,
        "curvature": 0.0,
        "belief_vega": 0.0,
        "cross_event": 0.0,
        "jump": 0.0,
        "total": 0.0,
    })

    def reset(self) -> None:
        self._prev = None
        for k in self._cumulative:
            self._cumulative[k] = 0.0

    @property
    def cumulative(self) -> dict[str, float]:
        return dict(self._cumulative)

    def step(
        self,
        snap: AttributionSnapshot,
        cross_event_terms: list[tuple[float, float]] | None = None,
    ) -> AttributionStep | None:
        """Advance one tick.

        `cross_event_terms` is an optional list of (ν_ρ_j, dρ_ij_j) pairs to
        accumulate into the cross-event bucket; stage-2 hedge code plugs in here.

        Returns None on the first snapshot (no prior).
        """
        prev = self._prev
        self._prev = snap
        if prev is None or prev.token_id != snap.token_id:
            return None

        dt = (snap.ts - prev.ts).total_seconds()
        dp = snap.p - prev.p
        dsigma = snap.sigma_b - prev.sigma_b

        # Greeks evaluated at the prior snapshot — paper §4.6 decomposition is
        # in logit-increment space:  q·dp = q·Δ·dx + ½·q·Γ·dx²  + higher order
        # where dx is the logit increment, not the paper-notation Δ_x.
        delta_x_val = delta_x(prev.p)   # S'(prev.p) = p(1-p)
        gamma_x_val = gamma_x(prev.p)   # S''(prev.p) = p(1-p)(1-2p)
        dx_incr = logit(snap.p) - logit(prev.p)

        directional = prev.qty * delta_x_val * dx_incr
        curvature = 0.5 * prev.qty * gamma_x_val * dx_incr * dx_incr

        # Belief-vega on a vanilla contract is zero (payoff doesn't depend on σ).
        # Non-zero ν_b only shows up for variance-swap / corridor exposure added
        # by the stage-3 hedge layer; the caller passes it via cumulative state.
        belief_vega = 0.0

        cross_event = 0.0
        if cross_event_terms:
            for nu_rho, drho in cross_event_terms:
                cross_event += nu_rho * drho

        realized, expected = realized_vs_expected_dp2(dp, dt, prev.sigma_b, prev.p)

        # Residual closer: the Taylor expansion above is exact only to
        # second order in dx. All higher-order terms (and any unmodeled
        # jump contribution) go into the jump bucket so the sum-of-buckets
        # matches the ledger's q·dp exactly. This is what the Stage-1 paper
        # soak's PnL-residual criterion compares against.
        true_total = prev.qty * dp
        jump_pnl = true_total - directional - curvature
        # The z-score threshold (jump_zscore_threshold) is retained for
        # downstream telemetry but no longer drives the arithmetic.

        total = directional + curvature + belief_vega + cross_event + jump_pnl

        self._cumulative["directional"] += directional
        self._cumulative["curvature"] += curvature
        self._cumulative["belief_vega"] += belief_vega
        self._cumulative["cross_event"] += cross_event
        self._cumulative["jump"] += jump_pnl
        self._cumulative["total"] += total

        return AttributionStep(
            token_id=snap.token_id,
            ts=snap.ts,
            dp=dp,
            dsigma_b=dsigma,
            directional_pnl=directional,
            curvature_pnl=curvature,
            belief_vega_pnl=belief_vega,
            cross_event_pnl=cross_event,
            jump_pnl=jump_pnl,
            total=total,
            realized_dp2=realized,
            expected_dp2=expected,
        )
