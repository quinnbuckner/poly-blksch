# `core/filter/` — Microstructure Filtering (Paper §5.1)

Recover the latent logit `x̂_t` from noisy Polymarket mid/bid/ask/trade data.

## Planned files

- `canonical_mid.py` — trade-weighted mid `p̃_t = (1/Z_t) Σ w_u (b_u+a_u)/2`, clipping to `[ε, 1-ε]`, outlier hygiene (drop crossed/locked books, drop isolated spikes). Resamples to a uniform grid (default 1 Hz; see `config.calibration.kf_grid_hz`).
- `microstruct.py` — heteroskedastic measurement noise model, paper eq (10):
  `σ_η²(t) = a_0 + a_1 s_t² + a_2 d_t⁻¹ + a_3 r_t + a_4 ι_t²`
  fit by robust regression on short-horizon squared innovations.
- `kalman.py` — heteroskedastic Kalman filter on `y = logit(p̃)` with measurement variance `σ_η²(t)` from `microstruct.py`. Local-level transition; no fixed drift (the EM step replaces it later with the RN drift). UKF fallback for `p` pinned near 0/1 or very frequent jumps.

## Outputs

`LogitState(token_id, x_hat, sigma_eta2, ts)` fed into `core/em/`.

## Tests (mandatory)

- Unit: recover known `x_t` on synthetic path with injected noise matching eq (10)
- Diagnostics (paper §5.1): Ljung–Box on residuals, Q–Q plot away from jump times, realized-p-variance matches `Σ S'(x)² σ_b²`
