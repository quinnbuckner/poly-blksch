"""Streaming EWMA estimator of the diffusive variance ``σ_b²(t)``.

Purpose
-------
The paper §6.3 forward-sum forecast evaluates ``Σ σ̂_b²(u)`` at a *per-step*
granularity — the rolling-EM's window-constant ``σ̂_b`` is too coarse to
track the per-origin realized variance that the Stage-0 gate measures
against. This module provides a cheap, causal, jump-aware per-step
variance estimate that the pipeline forecast can query at any time index.

Model
-----
The estimator runs an exponentially-weighted moving average of
``(Δx̂_t)²/Δt`` (the per-time-unit squared increment) with jump exclusion
via the E-step posterior ``γ_t``::

    s²(t) ← λ · s²(t-1) + (1 − λ) · (1 − γ_t) · (Δx̂_t)² / Δt

where ``λ = exp(−ln(2) · Δt / H)`` is the half-life retention factor for
a configurable half-life ``H`` in seconds. When ``γ_t ≈ 1`` (jump
dominant) the new sample contributes zero and ``s²`` decays toward
zero at the EWMA rate, so a truly jump-contaminated bin neither
inflates nor anchors the estimate.

Cold start
----------
For the first ``cold_start_factor · H`` seconds of data the estimator
falls back to the unweighted jump-excluded mean identity used by
paper eq (11)::

    s² = Σ (1 − γ_t)(Δx̂_t)² / Σ (1 − γ_t)·Δt

This avoids the "EWMA initialized at 0" bias that would otherwise take
~2·H to unwind. Once ``cold_start_factor · H`` of observation time has
elapsed the estimator snapshots this mean into its running ``s²`` and
switches to the EWMA recursion.

The module does not depend on any Track-B / Track-C code; it lives
under ``core/filter/`` because it is a filter primitive that both
``canonical_mid`` downstream and the ``rn_drift`` calibration loop can
consume in principle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_HALF_LIFE_SEC = 120.0
DEFAULT_COLD_START_FACTOR = 1.0  # cold-start span = factor × half_life_sec
LN2 = math.log(2.0)


@dataclass
class EwmaVar:
    """Jump-aware EWMA of ``(Δx̂)² / Δt``.

    Parameters
    ----------
    half_life_sec
        EWMA half-life in seconds. ``λ_per_step = exp(-ln(2) Δt / H)``
        so one half-life of elapsed time halves the weight of past
        observations. Typical choices: 30–300 s on the 1 Hz grid.
    cold_start_factor
        Length of the unweighted mean window expressed in half-lives.
        ``1.0`` covers the first half-life, which removes the
        EWMA-initial bias without over-committing to the warm-up. Set
        to 0 to disable cold-start and start at the first sample.
    """

    half_life_sec: float = DEFAULT_HALF_LIFE_SEC
    cold_start_factor: float = DEFAULT_COLD_START_FACTOR

    # Cold-start accumulators: paper eq (11) form.
    _cold_num: float = field(init=False, default=0.0, repr=False)
    _cold_denom: float = field(init=False, default=0.0, repr=False)
    _elapsed: float = field(init=False, default=0.0, repr=False)

    # Running EWMA state (populated once cold-start ends).
    _sigma_sq: float | None = field(init=False, default=None, repr=False)
    _samples_seen: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.half_life_sec <= 0:
            raise ValueError("half_life_sec must be positive")
        if self.cold_start_factor < 0:
            raise ValueError("cold_start_factor must be non-negative")

    # -- core ------------------------------------------------------------

    def update(
        self,
        delta_x: float,
        dt: float,
        jump_posterior: float = 0.0,
    ) -> float:
        """Advance the estimator by one (``Δx̂``, ``Δt``, ``γ``) tick.

        ``jump_posterior`` should be the per-step γ_t from the E-step
        (``em/increments.compute_posteriors``); the estimator clamps it
        to ``[0, 1]`` and treats 0 as "no jump" (full-weight update) and
        1 as "all jump" (no new information).

        Returns the current ``σ̂_b²(t)`` estimate after the update.
        """
        if dt <= 0:
            return self.variance()
        self._samples_seen += 1
        w = 1.0 - jump_posterior
        if w < 0.0:
            w = 0.0
        elif w > 1.0:
            w = 1.0

        dx2 = delta_x * delta_x
        self._elapsed += dt

        cold_span = self.cold_start_factor * self.half_life_sec

        if self._sigma_sq is None:
            # Cold-start: accumulate the jump-excluded weighted-mean numerator/denominator.
            self._cold_num += w * dx2
            self._cold_denom += w * dt
            if self._elapsed >= cold_span and self._cold_denom > 0:
                # Snapshot into EWMA state; after this point we're on the recursion.
                self._sigma_sq = self._cold_num / self._cold_denom
            return self.variance()

        # EWMA recursion on (Δx̂)²/Δt scale.
        retention = math.exp(-LN2 * dt / self.half_life_sec)
        per_time_sample = dx2 / dt
        self._sigma_sq = (
            retention * self._sigma_sq
            + (1.0 - retention) * w * per_time_sample
        )
        return self._sigma_sq

    def variance(self) -> float:
        """Current ``σ̂_b²(t)``. Zero until the first weighted sample arrives."""
        if self._sigma_sq is not None:
            return self._sigma_sq
        if self._cold_denom > 0:
            return self._cold_num / self._cold_denom
        return 0.0

    # -- diagnostics -----------------------------------------------------

    @property
    def in_cold_start(self) -> bool:
        return self._sigma_sq is None

    @property
    def samples_seen(self) -> int:
        return self._samples_seen

    @property
    def elapsed_sec(self) -> float:
        return self._elapsed

    def reset(self) -> None:
        self._cold_num = 0.0
        self._cold_denom = 0.0
        self._elapsed = 0.0
        self._sigma_sq = None
        self._samples_seen = 0


__all__ = [
    "DEFAULT_COLD_START_FACTOR",
    "DEFAULT_HALF_LIFE_SEC",
    "EwmaVar",
    "LN2",
]
