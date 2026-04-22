# `mm/hedge/` — Cross-Event, Calendar, and Synthetic Hedges

**All files here are staged — do not build ahead of the stage gate.**

## Files and stages

- `beta.py` — **Stage 2** (after Stage 1 paper-trading validation)
  Cross-event β-hedge (paper §4.4):
  `β̃_{i←j} = α · (S'_i / S'_j) · ρ_ij` with shrinkage `α ∈ [0.5, 1]`, clamp when `S'_k → 0`.
  Co-jump correction: `Δβ^{jump} = ∫ Δp_i Δp_j ν_{ij}(dz_i, dz_j) / (S'_j² σ_b^{j,2})`.
  Integrates with `mm/refresh_loop.py` step 5.

- `calendar.py` — **Stage 3**
  Variance strip sizing (paper §4.3): `N^{x-var} ≈ −ν̂_b / σ_b`. Used when a listed near-dated variance contract is available, otherwise synthesized via `synth_strip.py`.

- `synth_strip.py` — **Stage 3**
  Static replication of variance / corridor / first-passage notes from a basket of vanilla Polymarket contracts at adjacent moneyness and maturity (paper §3.4). This is the work the paper calls "coherent derivative layer"; since Polymarket does not list these products natively, we synthesize them.

## Design notes

- Hedges only make sense if the hedge instrument is liquid. Paper §4.7: "Quote where you can hedge. If no liquid proxy exists for a bucket, carry less exposure and charge more spread in that bucket." The refresh loop should consult a **hedgeability flag** before posting large size.
- All hedge instructions route through `exec/order_router.py` like any other order — we never bypass the ledger.
- Co-jump detection comes from `core/surface/corr.py`. Around known jump windows (e.g., election night, CPI print), the paper recommends **over-hedging diffusive correlation** (larger `α`) and carrying first-passage notes if available.
