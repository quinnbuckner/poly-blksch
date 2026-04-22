# `core/surface/` — Belief-Volatility Surface + Correlations (Paper §5.3, §5.4)

Smooth point estimates from `core/em/` into a stable surface over `(τ, m)` coordinates, and compute de-jumped cross-event correlations.

## Planned files

- `smooth.py` — tensor-product B-spline (or thin-plate) fit of `σ̂_b(τ, m)` with penalized least squares `min Σ w_g (σ̂_b(g) − σ_b(τ_g, m_g))² + α ‖∇² σ_b‖²`.
  Shape constraints:
  - Nonnegativity (squared-link or barrier)
  - Edge stability — damp curvature at extreme `m` (realized variance scales `∝ p²(1-p)²` so natural damping near boundaries, but surface itself shouldn't explode)
  - Term smoothness — regularize `∂_τ σ_b`, relaxing at known scheduled news dates
  Apply the same smoother to `λ(τ, m)` and `s²_J(τ, m)` to produce the jump layer.
- `corr.py` — de-jumped `ρ̂_ij(τ, m)` on intervals with no jumps in either series (`max(γ_i, γ_j) < τ_J`):
  `Cov̂(d) = (1/W) Σ_u S'_i(u) S'_j(u) Δx̂_i Δx̂_j`. Map to `(τ, m)` cells and smooth.
  Co-jump moments: count common jump events and estimate their second moment.

## Outputs

- `SurfacePoint(token_id, tau, m, sigma_b, lambda_, s2_j, uncertainty, ts)` — one per (token, grid cell)
- `CorrelationEntry(token_id_i, token_id_j, rho, co_jump_lambda, co_jump_m2, ts)` — one per pair

## Surface coordinates

The paper (§5.3) allows `m = x` (logit) or `m = min(p, 1-p)` (distance-to-boundary). We use `m = x` since it aligns with the kernel and avoids reshaping between spaces.

## Downstream use

- Track B reads `SurfacePoint` to get the `σ̄²_b` window average used in AS reservation/spread (paper §4.2)
- Track B reads `CorrelationEntry` for cross-event β-hedges (Stage 2, paper §4.4)
