# TRACK A — `core/` — Data & Calibration

**Owner:** Window 1 (see plan's *Window Bootstrap Prompts*)

**Responsibility:** ingest Polymarket market data, filter out microstructure noise, separate diffusion from jumps, produce a calibrated belief-volatility surface and cross-event dependence estimates.

**Status:** ⬜ pending. Day-0 scaffold only.

## Inputs and outputs

**Consumes:** raw Polymarket CLOB REST + WS (order books, trades, market metadata)

**Emits** (via `schemas.py`):
- `BookSnap`, `TradeTick` — raw market data for Tracks B and C
- `LogitState` — filtered latent `x̂_t` and measurement-noise variance
- `SurfacePoint` — `(τ, m) → (σ̂_b, λ̂, ŝ²_J)` with uncertainty bands
- `CorrelationEntry` — de-jumped `ρ̂_ij` and co-jump moments for event pairs

## Subpackages

| Folder | Paper § | Purpose |
|---|---|---|
| `ingest/` | — | REST + WS clients, tick store, market screener |
| `filter/` | 5.1 | Canonical mid, heteroskedastic noise model, Kalman / UKF |
| `em/` | 5.2, 3.2 | EM for diffusion vs. jumps; risk-neutral drift enforcement |
| `surface/` | 5.3, 5.4 | Tensor B-spline surface, de-jumped correlation |
| `diagnostics.py` | 5.1 | Ljung–Box, Q–Q, realized-vs-implied variance sanity checks |

## Build order

Follow the order in the plan's Window-1 prompt:
1. `ingest/polyclient.py` — REST + WS
2. `ingest/store.py` — Parquet tick store
3. `ingest/screener.py` — top-liquidity screener
4. `filter/canonical_mid.py` → `filter/microstruct.py` → `filter/kalman.py`
5. `em/increments.py` → `em/jumps.py` → `em/rn_drift.py`
6. `surface/smooth.py` → `surface/corr.py`
7. `diagnostics.py`

## Correctness gate (Stage 0 → 1)

`tests/pipeline/test_paper_sec6_replication.py` must produce MSE/MAE/QLIKE within 10% of the RN-JD row of Table 1 in the paper (MSE≈70.28, MAE≈1.59, QLIKE≈1.46) on the synthetic RN-consistent path. Until this passes, Track B and C can proceed against **fixtures**, but live integration is blocked.

## Rules of engagement

- `schemas.py` is frozen — propose changes, don't commit them unilaterally
- Keep `ingest/` IO-pure (async) and `filter/em/surface/` math-pure (sync, numpy) for testability
- Never trust raw mid — always flow through `canonical_mid.py` (paper §5.1 outlier hygiene)
- Diagnostics are **mandatory**, not optional — if Ljung–Box fails, fix the filter before trusting the surface
