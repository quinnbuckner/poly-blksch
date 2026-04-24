# CATALOG — Living Inventory

Status legend: ✅ built & tested · 🟡 built, needs tests · 🛠 in progress · ⬜ pending · 🚫 blocked

**Update this file on every commit.** If you add, remove, or change the status of a file, reflect it here.

## Stage tracker

| Stage | Scope | Status |
|---|---|---|
| 0 | Scaffold + Track A calibration | ✅ CLEARED — multi-seed §6 replication gate passes (fac7d13) |
| 1 | Tracks A+B+C on paper engine | 🛠 ready to launch — `python -m blksch.app --mode=paper` wired; `scripts/paper_soak.py` 72h harness ready |
| 2 | Live CLOB orders | ⬜ gated on Stage-1 72h acceptance |
| 3 | Cross-event β-hedges | ⬜ code built, `hedge_enabled=False` |
| 4 | Synthetic variance/corridor strips | ⬜ code built, `calendar_hedge_enabled=False`, `synth_strip_enabled=False` |

## Root

| File | Status | Notes |
|---|---|---|
| `README.md` | ✅ | Project overview |
| `ARCHITECTURE.md` | ✅ | Three-track split, contracts, paper-to-module map |
| `HANDOFF.md` | ✅ | Fresh-session orientation |
| `CATALOG.md` | ✅ | This file |
| `pyproject.toml` | ✅ | Deps + entrypoint |
| `.gitignore` | ✅ | Excludes .env, data/, *.parquet, *.db |

## Config (`config/`)

| File | Status | Notes |
|---|---|---|
| `bot.yaml` | ✅ | Seeded from plan's Defaults table |
| `markets.yaml` | ✅ | Screener stub + correlation pair hints |
| `README.md` | ✅ | What each knob controls |

## Shared contracts (`src/blksch/`)

| File | Status | Notes |
|---|---|---|
| `schemas.py` | ✅ | Pydantic models for all inter-track messages |
| `app.py` | ✅ | Full asyncio orchestration: screener → polyclient WS → canonical_mid → Kalman → background em_calibrate → RefreshLoop → paper_engine/clob_client. `--mode=live` gated on `--live-ack`. |
| `README.md` | ✅ | Package overview |

## Track A — Data & Calibration (`src/blksch/core/`)

| Module | Status | Paper ref | Notes |
|---|---|---|---|
| `ingest/polyclient.py` | ✅ | — | Async REST + CLOB WS; rate-limited; emits BookSnap/TradeTick |
| `ingest/store.py` | ✅ | — | Async append-only Parquet; partitioned by (stream, token_id, UTC date); size-cap rotation; `read_range()` |
| `ingest/screener.py` | ✅ | — | Top-N volume+depth screener; TTL cache; correlation-pair resolution |
| `filter/canonical_mid.py` | ✅ | §5.1 | Trade-weighted mid + clipping + K·IQR outlier reject + grid LOCF |
| `filter/microstruct.py` | ✅ | §5.1 eq 10 | Heteroskedastic σ_η²(t); OLS fit; forward-fill widening; floor |
| `filter/kalman.py` | ✅ | §5.1 | Heteroskedastic scalar KF on y=logit(p̃); UKF blend near boundary; gain-clip divergence safeguards. **Known pathology:** \|x\|>4.5 inflates σ̂_b² 30–70% (see memory/project_boundary_regime_em_inflation.md) |
| `filter/ewma_var.py` | ✅ | — | Streaming jump-aware EWMA σ̂_b²(u) forecast component (H=90s default); production σ̂_b² for the §6 gate forecast |
| `em/increments.py` | ✅ | §5.2 | Mixture E-step: `γ_t` posteriors via log-sum-exp; frozen `MixtureParams` |
| `em/jumps.py` | ✅ | §5.2 eq 11–12 | M-step λ̂ / ŝ²_J; jump-timestamp extraction; bi-power variation cross-check |
| `em/rn_drift.py` | ✅ | §3.2 eq 3 | μ(t,x) via MC grid compile; `em_calibrate` outer loop; BV warm-start |
| `surface/smooth.py` | ✅ | §5.3 | Tensor B-spline surface; shape-constrained (nonneg, edge-stable) |
| `surface/corr.py` | ✅ | §5.4 | De-jumped ρ̂ + co-jump moments; jump-window masking |
| `diagnostics.py` | ✅ | §5.1 / §6.1 | Ljung–Box, Q–Q, realized-vs-implied variance consistency |

## Track B — Quoting Engine (`src/blksch/mm/`)

| Module | Status | Paper ref | Notes |
|---|---|---|---|
| `greeks.py` | ✅ | §4.1 | Δ_x, Γ_x, ν_b (x-var + p-var), ν_ρ |
| `quote.py` | ✅ | §4.2 eq 8–9 | AS reservation + spread in logit; δ_p floor; inventory cap |
| `guards.py` | ✅ | §4.2 | VPIN toxicity / news window + pause / queue discipline |
| `refresh_loop.py` | ✅ | §4.5 | 100–500 ms asyncio cycle; per-token state; pluggable sinks |
| `pnl.py` | ✅ | §4.6 | Δ–Γ–ν_b–ν_ρ–jump attribution; realized vs expected (dp)² |
| `limits.py` | ✅ | §4.6 | Feed-gap / vol spike / pickoff / drawdown / swing-zone Γ |
| `hedge/beta.py` | 🟡 | §4.4 | Cross-event β-hedge — built, runtime-flagged `hedge_enabled=False` until Stage-1 paper gate |
| `hedge/calendar.py` | 🟡 | §4.3 | Variance-strip sizing — built, `calendar_hedge_enabled=False` until Stage-3 gate; synth legs now resolve via synth_strip when `synth_strip_enabled=True` |
| `hedge/synth_strip.py` | 🟡 | §3.4 | Static replication of variance/corridor from vanilla basket — built, `synth_strip_enabled=False` until Stage-3 gate |

## Track C — Execution & Infra (`src/blksch/exec/`)

| Module | Status | Paper ref | Notes |
|---|---|---|---|
| `clob_client.py` | ✅ | — | Async adapter; prefers `py-clob-client`, `HttpCLOBClient` fallback for read-only |
| `signer.py` | ✅ | — | EIP-712 typed-data signing (Polymarket CTF Exchange v1); recovered-signer unit tests |
| `ledger.py` | ✅ | — | SQLite (WAL) positions/fills/orders; signed-qty WAP accounting; `reconcile()` helper |
| `paper_engine.py` | ✅ | — | Conservative book-through + trade-tick matching; queue haircut; feed-gap halt |
| `order_router.py` | ✅ | — | Idempotent `sync_quote`/place/cancel/replace; retry/backoff; `live_ack` gate |
| `dashboard.py` | ✅ | — | Rich `Live` terminal layout + Flask `/api/state` JSON |

## Tests

| Path | Status | Notes |
|---|---|---|
| `tests/unit/` | ✅ | All modules covered; property-fuzz suite at `tests/unit/property/` (35 Hypothesis tests); snapshot regressions at `tests/unit/snapshot/` (25 golden fixtures, 1e-10 tolerance); fast suite total: 1086 passed / 15 skipped / 0 failed |
| `tests/integration/` | ✅ | Track A calibration, Track B quote/hedge/ablation/calendar/synth_strip, Track C paper_engine, PnL reconciliation, adversarial kill-switches, app.py wiring (7 scenarios), calibration_dryrun offline |
| `tests/contract/` | ✅ | polyclient + clob_client + testnet contract tests; fixtures recorded from live Polymarket (3b8371d) |
| `tests/pipeline/` | ✅ | `test_paper_sec6_replication.py`: multi-seed median + robustness + causality (3 passing); kill_switch + live_paper_trade still stubs |
| `tests/fixtures/` | ✅ | Synthetic RN-consistent path generator; contract fixtures recorded; snapshot fixtures pinned |

## Change log

- **Day 0 (Apr 22 2026)** — Repo scaffolded. Pydantic schemas written. Config seeded. All three track folders created with READMEs. Nothing wired yet. Committed and pushed to `origin main`.
- **Apr 22 2026 — Track B Stage 1 complete.** Shipped `mm/greeks.py`, `mm/quote.py`, `mm/guards.py`, `mm/pnl.py`, `mm/limits.py`, `mm/refresh_loop.py` (paper §4.1, §4.2 eq 8-9, §4.5, §4.6). 110 unit+integration tests green. Critical integration test `tests/integration/test_track_b_quote.py` verifies expected Quote on fixed inputs, toxicity/news spread widening, feed-gap + inventory-cap pulls. Stage 2/3 (`mm/hedge/*`) still stubbed.
- **Apr 22 2026 — Track A `polyclient.py`.** `core/ingest/polyclient.py` landed: async aiohttp + websockets client for Polymarket Gamma/CLOB REST and the CLOB `market` WS channel. Emits `BookSnap` and `TradeTick`. In-memory book state + diff application for `price_change` events. Reconnect with exponential backoff. Rate-limiter ported from `polyarb_v1.0/src/api.py`. 20 unit tests on parsers + limiter.
- **Apr 22 2026 — Track C Stage 1 complete.** Shipped `exec/clob_client.py` (py-clob-client adapter + aiohttp fallback for read-only use), `exec/signer.py` (EIP-712 Polymarket CTF Exchange v1), `exec/ledger.py` (SQLite WAL; signed-qty WAP accounting; `reconcile()` helper), `exec/paper_engine.py` (book-through + trade-tick matching; configurable queue haircut; feed-gap halt), `exec/order_router.py` (idempotent `sync_quote` with replace-on-change; retry/backoff; `live_ack` gate), `exec/dashboard.py` (Rich `Live` layout + Flask `/api/state`). 31 new unit tests and the `test_track_c_paper_engine.py` integration gate are green (161 total tests pass). Stage-2 live promotion stays gated on `RouterConfig.live_ack=True`.
- **Apr 22 2026 — Track A `store.py`.** `core/ingest/store.ParquetStore`: async append-only Parquet writer partitioned as `<root>/<stream>/<token_id>/<YYYY-MM-DD>/part-NNNNN.parquet`. `append_book` / `append_trade` wrap `asyncio.to_thread` so the ingest path never blocks; flushing is bounded by a configurable size cap (default 128 MB) that rotates to a new part file. `read_range(token_id, start_ts, end_ts, *, stream)` returns a pandas DataFrame for downstream calibration; includes in-memory buffer by default. 12 unit tests covering round-trip, date/token partitioning, rotation, empty-range reads, and buffered reads.
- **Apr 22 2026 — Track A `screener.py`.** `core/ingest/screener.Screener`: top-N liquidity picker scored by a volume/depth composite (depth-dominant at 0.7 / 0.3 by default). Consumes `PolyClient.list_markets` (added in this module — pages through Gamma) and `PolyClient.get_book` for ±5% depth sampling. `ScreenerFilters` maps directly to `config/markets.yaml`. Correlation pair hints are resolved against the scanned universe — missing-leg pairs are dropped. TTL cache (default 300 s) avoids hammering Gamma on every refresh loop. 18 unit tests covering filter boundaries, scoring monotonicity, `top_n` truncation, TTL hit/miss, `force` bypass, and pair resolution (207 total unit tests).
- **Apr 22 2026 — Track A `canonical_mid.py`.** `core/filter/canonical_mid.CanonicalMidFilter`: stream-or-batch §5.1 conditioner. Emits `CanonicalMid` at the configured `kf_grid_hz` (default 1 Hz) with trade-notional VWAP when trades in the sliding window (default 30 s), book-mid otherwise; clamps to `[eps, 1-eps]` (default 1e-5). Outlier hygiene rejects `|Δlogit|` > `K·IQR` of the trailing delta-logit window (K=4 default) — rejections are logged, the last accepted p̃ is carried forward and the output is flagged `rejected_outlier=True`. Grid cadence uses LOCF across empty bins; empty/crossed books are skipped as hygiene fails. Introduces `CanonicalMid` (non-schema dataclass, Track A-internal). 20 unit tests covering pure helpers, constructor validation, VWAP/book-mid fallback, trade-window purge, clipping at p→0 and p→1, outlier reject + recover, grid cadence, forward-fill flags, hygiene drops, and snap=None tick mode (223 total unit tests).
- **Apr 23 2026 — Track A `microstruct.py`.** `core/filter/microstruct.MicrostructModel`: §5.1 eq (10) heteroskedastic measurement-noise model `σ_η²(t) = a₀ + a₁·(spread/2)² + a₂·(1/depth) + a₃·|rate| + a₄·|imbalance|`. `extract_features(book, trades)` builds the covariate vector using top-K per-side depth (K=5 default), a 30 s trade-rate window, and signed imbalance. `MicrostructModel.fit` runs ridge-regularized OLS (λ=1e-8 default) on `(squared_innovation ~ features)` — the EM loop supplies innovations so the regression is decoupled from any particular x̂_t estimator. `variance()` clamps to a `sigma_floor` (1e-6) and, when `forward_filled=True`, widens σ² by a configurable multiplicative factor (10× default) so downstream Kalman gain drops on stale ticks. 19 unit tests: config validation, feature extraction (imbalance, trade rate, empty book), coefficient recovery on 4 k-sample synthetic set, variance monotonicity in all four covariates, forward-fill widening at the default and a custom factor, floor activation on perfect books, graceful degenerate inputs, and strict positivity even with pathological negative coefficients (313 total unit tests).
- **Apr 23 2026 — Track A `kalman.py`.** `core/filter/kalman.KalmanFilter`: scalar heteroskedastic KF on `y_t = logit(p̃_t) = x_t + η_t`. Process noise `Q_t = σ_b²·dt` (σ_b from the constructor now; from `em/rn_drift.py` in production). Measurement variance `R_t` is pulled from `MicrostructModel.variance(book, trades, forward_filled=…)` per step via a `VarianceOracle` protocol. Inside `p ∈ [0, p_low] ∪ [p_high, 1]` (defaults 0.02 / 0.98) the filter blends in a scalar UKF using three sigma points mapped through `S(x)` — the p-space predictive variance is back-translated to x-space via the Jacobian and used as the effective R when it exceeds the microstruct value. Blend weight is quadratic in margin-to-edge so the posterior is continuous across `p_low`/`p_high`. Divergence safety: `K ∈ [0, 1]` clip, innovation-variance and posterior-variance floors, long-gap `dt` cap (default 60 s). Emits existing `LogitState(token_id, x_hat, sigma_eta2, ts)` — the published `sigma_eta2` is the raw microstruct value; the UKF-inflated R stays internal. 15 unit tests: constructor validation; 3σ-coverage recovery on 2 k synthetic ticks at ≥ 95 %; MSE ≤ ½ × raw-observation MSE; Ljung–Box whiteness on normalized innovations (α = 0.05, 3 k ticks); smooth boundary sweep (max consecutive-step jump < 1σ); continuous effective-R at p_low; linear-KF behavior in the interior; divergence protection under alternating σ_η² of `1e-12` and `1e12`; gain-clip correctness; LogitState faithfulness; forward-fill flag propagation; `max_dt_sec` cap under long gaps (349 total unit tests).
- **Apr 23 2026 — Track A `em/increments.py`.** `core/em/increments`: E-step of the jump-diffusion EM loop (paper §5.2). Freezes the `MixtureParams` contract (`sigma_b, s_J, lambda_jump, mu`) that `em/jumps.py` and `em/rn_drift.py` will import. `compute_posteriors(increments, dts, params)` returns `γ_t = P(jump | Δx̂_t)` via log-sum-exp over a Gaussian+Bernoulli-jump mixture — stable under 600 s gaps where `λΔt` saturates. `log_likelihood` exposes the mixture LL for convergence monitoring and M-step cross-checks. `mark_jumps(γ, threshold=0.7)` applies the paper's τ_J cutoff. 17 unit tests: frozen-dataclass validation, Gaussian kernel cross-check with scipy, pure-diffusion (λ=0 or s_J=0) shortcut, zero-`dt` handling, shape validation, 600 s-gap finiteness, `λΔt` saturation, ≥ 80 % jump recall on synthetic JD (`s_J/σ_b√Δt ~ 33`), < 0.5 % false-positive on pure diffusion, log-likelihood peak near true `σ_b` on a 21-point grid, reduction to Gaussian LL at `λ=0`, and `mark_jumps` threshold semantics (383 total unit tests).
- **Apr 23 2026 — Track A `em/jumps.py`.** `core/em/jumps`: M-step for the jump parameters. Also extends `em/increments` with a `PosteriorResult` dataclass and an `e_step(states, params)` wrapper that jumps / rn_drift consume. `m_step_jumps(states, posteriors, params)` returns a frozen `JumpEstimate(lambda_hat, s_J_sq_hat, jump_timestamps, log_likelihood)` — λ̂ is clipped to `1/Δt_min` (one jump per step ceiling); ŝ²_J is drift-corrected `Σ γ (Δx̂ − μ·Δt)² / Σ γ`, floored at `1e-8`; when `Σ γ ≈ 0` the branch collapses to `λ̂ = 0` with `ŝ²_J` held at the prior. `jump_timestamps` lists the ends of increments crossing the τ_J threshold — this is the frozen contract Window C's `surface/corr.py` imports (plain `list[datetime]`). Also exposes `bipower_variance(increments) = (π/2)·Σ|Δx_t|·|Δx_{t-1}|` as an independent diffusion-variance estimator for jump-decomposition diagnostics. 17 new unit tests: dataclass validation, paper §6-scale (6 k-sample) recovery of λ within 20 % and ŝ²_J within 30 %, stability at higher jump rates, exact 10-of-10 timestamp recovery from injected jumps, threshold-passthrough, empty-gamma degenerate branch, empty-states branch, s_J² floor, λ̂ clip to 1/Δt, BV vs RV robustness ratio, BV within 20 % on a rare-jump regime, BV shape validation, π/2 scale check, full E→M→E monotonicity (LL non-decreasing), state/posterior length mismatch validation (446 total unit tests).
- **Apr 23 2026 — Track A `em/rn_drift.py`.** `core/em/rn_drift`: closes the §3.2 eq (3) risk-neutral drift loop and wires the **outer EM orchestrator** `em_calibrate`. `compile_mu_fn(sigma_b, lambda_jump, s_J)` precomputes ``μ(x) = -[½·S''(x)·σ_b² + λ·E_z(S(x+z)−S(x)−S'(x)·z)] / max(S'(x), sprime_clip)`` on a logit grid (default 401 points on [-5, 5]) via a 2 000-sample MC compensator, then serves by linear interpolation — the online Kalman never pays the MC cost. Numerical guards: ``|μ| ≤ mu_cap_per_sec=0.25``, ``S'(x)`` floored at ``sprime_clip=1e-4``, deterministic seed. Under ``λ=0`` the MC branch is skipped and μ collapses to the analytic ``-½·(1-2p)·σ_b²`` — verified at 100 logit points. `em_calibrate(states, initial_params, …)` iterates ``E-step → m_step_jumps → m_step_sigma_b (paper eq 11) → compile_mu_fn`` with LL-based convergence (``|ΔLL/LL| < tol`` or ``max_iters=50``). Returns a frozen `CalibrationResult(final_params, jumps, mu_fn, log_likelihood_history, converged, iters)`. 17 unit tests: config validation; closed-form pure-diffusion match at 100 grid points; zero drift at ``p=0.5``; s_J-independence at ``λ=0``; **the paper's martingale check** — simulate a JD path with the computed drift and verify ``|E[p_{t+60 s}] - p₀| < 0.005`` at ``p ∈ {0.2, 0.5, 0.8}`` over 10 k paths; ``|μ|`` cap activation; finite μ at ``x = ±5``; deterministic output for identical seed; seed-sensitivity sanity; pathological inputs (``s_J=1e-9, λ=1e3``) stay finite; MC std < 5 % at K=2000; em_calibrate convergence within 30 % of truth for ``(σ_b, λ, s_J)`` on 6 k-sample synthetic JD from mis-specified priors; LL history non-decreasing; hyperparameter validation; short-input graceful handling; ``mu_fn`` exposes grid for diagnostics (463 total unit tests). **Track A critical chain complete.**
