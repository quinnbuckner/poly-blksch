# `core/em/` — EM for Diffusion vs. Jumps (Paper §5.2, §3.2)

Separate diffusive belief volatility `σ̂_b` from jump intensity `λ̂` and jump second moment `ŝ²_J` using a Gaussian+jump increment mixture. Then enforce the risk-neutral drift.

## Planned files

- `increments.py` — on a grid with step `Δ`, `Δx_t ~ N(μ_t Δ, σ_b²(t)Δ)` w.p. `1 - λ_tΔ`, else `Z_t ~ f_J(·; θ_t)`. Builds the mixture likelihood per tick.
- `jumps.py` — E-step: posterior jump responsibilities `γ_t` (paper §5.2); cross-check with bi-power variation test.
  M-step: updates for `σ̂_b²`, `λ̂`, `ŝ²_J` (paper eq 11–12).
- `rn_drift.py` — after each M-step, recompute `μ(t, x)` from the martingale restriction (paper §3.2 eq 3):
  ```
  μ(t,x) = −( ½ S''(x) σ_b²(t,x) + ∫[S(x+z) − S(x) − S'(x) χ(z)] ν_t(dz) ) / S'(x)
  ```
  Monte Carlo the jump compensator with `config.calibration.mc_draws_per_step` draws. Outer EM loop: ~6 global passes, then rolling 400s window.

## Numerical hazards

- `S'(x) → 0` near `p = 0` or `1` — clip by `config.calibration.sprime_clip` to avoid division blow-up
- Cap `|μ|` by `config.calibration.mu_cap_per_sec` to prevent runaway drift
- Symmetric Gaussian jumps are the default; switch to double-exponential or tempered-stable if Q–Q diagnostics reject Gaussian

## Tests

- Unit: on synthetic jump-diffusion with known `(σ_b, λ, s_J)`, EM recovers within 10%
- Unit: martingale check — simulate `dx_t` with computed `μ`, verify `E[p_{t+Δ} | F_t] ≈ p_t` on Monte Carlo
