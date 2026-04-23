"""De-jumped correlation + co-jump moments (paper §5.4).

Given two filtered logit-state streams (``LogitState`` from Track A's Kalman
filter) and their jump-timestamp lists (from Track A's EM jump detector),
estimate:

* **ρ_ij** — the continuous-part (i.e. *de-jumped*) correlation of the logit
  increments, via Pearson on the masked series.
* **λ_ij** — the co-jump rate per second. A co-jump is a pair of jumps
  (one in each series) whose timestamps fall within a small window ``w``.
* **m2_ij** — the mean of the increment product at co-jump timestamps. This
  is what §5.4 calls the second moment of the co-jump distribution.

Contract with Window A (frozen — see the plan's "STRATEGY CHANGE" prompt):

* Inputs are ``list[LogitState]`` for each token and ``list[datetime]`` of
  jump timestamps for each token. Both states streams are assumed to share a
  common time grid (same length, same timestamps) — Track A's upstream is
  responsible for alignment.
* Output is a :class:`blksch.schemas.CorrelationEntry`. The schema is frozen.

Numerical notes:

* ρ is clipped to ``[-1, 1]``; Pearson can exceed the interval in the 15th
  decimal place due to float error and we don't want downstream shape
  constraints in ``smooth.py`` to fail on that.
* If fewer than ``min_samples`` masked increments survive, raise
  :class:`NotEnoughSamplesError`. Callers are expected to treat this as a
  "surface undefined for this window" signal rather than substitute zero.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

import numpy as np

from blksch.schemas import CorrelationEntry, LogitState


DEFAULT_MASK_WINDOW = timedelta(seconds=30)
DEFAULT_MIN_SAMPLES = 100


class NotEnoughSamplesError(RuntimeError):
    """Raised when masking leaves fewer than ``min_samples`` usable
    increments. The caller should skip this (i,j) token pair for this
    surface update, not substitute a zero or NaN correlation."""


def estimate_correlation(
    states_i: Sequence[LogitState],
    states_j: Sequence[LogitState],
    jumps_i: Sequence[datetime],
    jumps_j: Sequence[datetime],
    *,
    mask_window: timedelta = DEFAULT_MASK_WINDOW,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    co_jump_window: timedelta | None = None,
    ts: datetime | None = None,
) -> CorrelationEntry:
    """De-jumped correlation estimator (paper §5.4).

    Parameters
    ----------
    states_i, states_j
        Filtered logit-state streams, one per token. Must be the same length
        with matching timestamps — alignment is the caller's responsibility.
    jumps_i, jumps_j
        Jump-event timestamps for tokens i and j. Order doesn't matter; the
        estimator sorts internally.
    mask_window
        Half-width ``w`` of the exclusion window around each jump. Any
        increment whose right endpoint lies within ``w`` of any jump (in
        either series) is dropped before computing the correlation.
    min_samples
        Minimum number of surviving increments required. If fewer survive,
        raise :class:`NotEnoughSamplesError`.
    co_jump_window
        Window for matching a jump in series i against a jump in series j.
        Defaults to ``mask_window``. Each jump in j is matched at most once.
    ts
        Timestamp to stamp on the returned ``CorrelationEntry``. Defaults to
        the last observation timestamp.
    """

    if len(states_i) != len(states_j):
        raise ValueError(
            f"states_i and states_j must be aligned "
            f"(got {len(states_i)} vs {len(states_j)})"
        )
    if len(states_i) < 2:
        raise NotEnoughSamplesError(
            f"need at least 2 aligned states, got {len(states_i)}"
        )

    ts_i = np.array([s.ts.timestamp() for s in states_i], dtype=float)
    ts_j = np.array([s.ts.timestamp() for s in states_j], dtype=float)
    # Absolute tolerance, not np.allclose's relative default: Unix epoch
    # seconds are ~1.7e9, and the default rtol=1e-5 would admit 17 km of
    # timestamp drift.
    if np.abs(ts_i - ts_j).max() > 1e-6:
        raise ValueError(
            "states_i and states_j timestamps are not aligned; "
            "resample to a common grid before calling estimate_correlation"
        )

    x_i = np.array([s.x_hat for s in states_i], dtype=float)
    x_j = np.array([s.x_hat for s in states_j], dtype=float)

    dx_i = np.diff(x_i)
    dx_j = np.diff(x_j)
    # Each increment is anchored at its right-endpoint timestamp.
    ts_inc = ts_i[1:]

    all_jumps = sorted({*jumps_i, *jumps_j})
    mask = _mask_around_jumps(ts_inc, all_jumps, mask_window.total_seconds())

    dx_i_m = dx_i[mask]
    dx_j_m = dx_j[mask]

    if dx_i_m.size < min_samples:
        raise NotEnoughSamplesError(
            f"only {dx_i_m.size} increments survived jump masking; "
            f"need >= {min_samples}. Widen the observation window or "
            f"reduce mask_window."
        )

    rho = _pearson(dx_i_m, dx_j_m)
    rho = float(np.clip(rho, -1.0, 1.0))

    cjw = (co_jump_window or mask_window).total_seconds()
    co_count, co_products = _match_co_jumps(
        jumps_i=sorted(jumps_i),
        jumps_j=sorted(jumps_j),
        window_sec=cjw,
        dx_i=dx_i, dx_j=dx_j, ts_inc=ts_inc,
    )

    T = ts_i[-1] - ts_i[0]
    if T <= 0:
        raise ValueError("states span zero or negative duration")
    lam = co_count / T
    m2 = float(np.mean(co_products)) if co_products else 0.0
    # m2 is defined as the second moment of the jump-product; §5.4 treats it
    # as a magnitude, so surface the absolute value (keeps downstream
    # non-negative constraints happy).
    m2 = abs(m2)

    return CorrelationEntry(
        token_id_i=states_i[0].token_id,
        token_id_j=states_j[0].token_id,
        rho=rho,
        co_jump_lambda=lam,
        co_jump_m2=m2,
        ts=ts or states_i[-1].ts,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Plain-Pearson with guards for degenerate variance. Returns 0 if
    either series is effectively constant — a flat series has no
    information about its partner, so contributing 0 to downstream
    smoothing is the right default."""
    if a.size == 0:
        return 0.0
    a_mean = a.mean()
    b_mean = b.mean()
    a_c = a - a_mean
    b_c = b - b_mean
    denom = np.sqrt((a_c * a_c).sum() * (b_c * b_c).sum())
    if denom <= 0.0:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def _mask_around_jumps(
    ts_inc: np.ndarray,
    jumps: Sequence[datetime],
    window_sec: float,
) -> np.ndarray:
    """Boolean mask of len(ts_inc): True for increments NOT within
    ``window_sec`` of any jump timestamp. Uses binary search to stay
    O((n + m) log m) rather than O(n*m)."""
    mask = np.ones(ts_inc.size, dtype=bool)
    if not jumps:
        return mask
    jumps_sec = np.array([j.timestamp() for j in jumps], dtype=float)
    jumps_sec.sort()
    # For each increment ts, check if any jump lies in [ts - w, ts + w].
    left = np.searchsorted(jumps_sec, ts_inc - window_sec, side="left")
    right = np.searchsorted(jumps_sec, ts_inc + window_sec, side="right")
    mask[right > left] = False
    return mask


def _match_co_jumps(
    *,
    jumps_i: Sequence[datetime],
    jumps_j: Sequence[datetime],
    window_sec: float,
    dx_i: np.ndarray,
    dx_j: np.ndarray,
    ts_inc: np.ndarray,
) -> tuple[int, list[float]]:
    """Greedy one-to-one matching of jumps_i → jumps_j within ``window_sec``.

    Returns ``(count, products)`` where ``products[k]`` is the product of
    the two series' increment values at the midpoint timestamp of the k-th
    matched co-jump. Each jump in j is matched at most once.
    """
    if not jumps_i or not jumps_j or ts_inc.size == 0:
        return 0, []
    j_float = np.array([j.timestamp() for j in jumps_j], dtype=float)
    # Sorted already (caller sorts). Keep an index array so we can mark
    # used j-jumps.
    used = np.zeros(j_float.size, dtype=bool)
    count = 0
    products: list[float] = []
    for ti in jumps_i:
        ti_f = ti.timestamp()
        lo = np.searchsorted(j_float, ti_f - window_sec, side="left")
        hi = np.searchsorted(j_float, ti_f + window_sec, side="right")
        if hi <= lo:
            continue
        # Pick the closest unused j-jump in [lo, hi).
        best = -1
        best_d = window_sec + 1.0
        for idx in range(lo, hi):
            if used[idx]:
                continue
            d = abs(j_float[idx] - ti_f)
            if d < best_d:
                best_d = d
                best = idx
        if best < 0:
            continue
        used[best] = True
        count += 1
        mid = 0.5 * (ti_f + j_float[best])
        k = int(np.argmin(np.abs(ts_inc - mid)))
        products.append(float(dx_i[k] * dx_j[k]))
    return count, products
