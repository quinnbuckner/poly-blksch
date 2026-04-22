# CATALOG — Living Inventory

Status legend: ✅ built & tested · 🟡 built, needs tests · 🛠 in progress · ⬜ pending · 🚫 blocked

**Update this file on every commit.** If you add, remove, or change the status of a file, reflect it here.

## Stage tracker

| Stage | Scope | Status |
|---|---|---|
| 0 | Scaffold + Track A calibration | 🛠 scaffold complete; Track A ingest underway; Track B Stage-1 done |
| 1 | Tracks A+B+C on paper engine | 🛠 Track B Stage-1 complete (quote/guards/pnl/limits/refresh_loop); awaiting A & C |
| 2 | Live CLOB orders | ⬜ |
| 3 | Cross-event β-hedges | ⬜ |
| 4 | Synthetic variance/corridor strips | ⬜ |

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
| `app.py` | ✅ | Stub entrypoint; prints "not wired yet" |
| `README.md` | ✅ | Package overview |

## Track A — Data & Calibration (`src/blksch/core/`)

| Module | Status | Paper ref | Notes |
|---|---|---|---|
| `ingest/polyclient.py` | ⬜ | — | REST + WS for Polymarket CLOB |
| `ingest/store.py` | ⬜ | — | Parquet tick store |
| `ingest/screener.py` | ⬜ | — | Top-liquidity picker |
| `filter/canonical_mid.py` | ⬜ | §5.1 | Trade-weighted mid + outlier hygiene |
| `filter/microstruct.py` | ⬜ | §5.1 eq 10 | Heteroskedastic noise model |
| `filter/kalman.py` | ⬜ | §5.1 | Heteroskedastic KF / UKF |
| `em/increments.py` | ⬜ | §5.2 | Gaussian+jump mixture |
| `em/jumps.py` | ⬜ | §5.2 eq 11–12 | Posterior jump responsibilities |
| `em/rn_drift.py` | ⬜ | §3.2 eq 3 | Risk-neutral drift enforcement |
| `surface/smooth.py` | ⬜ | §5.3 | Tensor B-spline surface |
| `surface/corr.py` | ⬜ | §5.4 | De-jumped ρ̂ + co-jumps |
| `diagnostics.py` | ⬜ | §5.1 | Ljung–Box, Q–Q, variance checks |

## Track B — Quoting Engine (`src/blksch/mm/`)

| Module | Status | Paper ref | Notes |
|---|---|---|---|
| `greeks.py` | ✅ | §4.1 | Δ_x, Γ_x, ν_b (x-var + p-var), ν_ρ |
| `quote.py` | ✅ | §4.2 eq 8–9 | AS reservation + spread in logit; δ_p floor; inventory cap |
| `guards.py` | ✅ | §4.2 | VPIN toxicity / news window + pause / queue discipline |
| `refresh_loop.py` | ✅ | §4.5 | 100–500 ms asyncio cycle; per-token state; pluggable sinks |
| `pnl.py` | ✅ | §4.6 | Δ–Γ–ν_b–ν_ρ–jump attribution; realized vs expected (dp)² |
| `limits.py` | ✅ | §4.6 | Feed-gap / vol spike / pickoff / drawdown / swing-zone Γ |
| `hedge/beta.py` | ⬜ | §4.4 | Cross-event β-hedge (Stage 2) |
| `hedge/calendar.py` | ⬜ | §4.3 | Variance-strip sizing (Stage 3) |
| `hedge/synth_strip.py` | ⬜ | §3.4 | Synthetic variance/corridor (Stage 3) |

## Track C — Execution & Infra (`src/blksch/exec/`)

| Module | Status | Paper ref | Notes |
|---|---|---|---|
| `clob_client.py` | ⬜ | — | Polymarket CLOB REST + auth |
| `signer.py` | ⬜ | — | EIP-712 typed-data signing |
| `ledger.py` | ⬜ | — | Positions / fills / PnL |
| `paper_engine.py` | ⬜ | — | Simulated matching engine |
| `order_router.py` | ⬜ | — | Idempotent place/cancel/replace |
| `dashboard.py` | ⬜ | — | Rich terminal + Flask live view |

## Tests

| Path | Status | Notes |
|---|---|---|
| `tests/unit/` | 🟡 | Track B complete (greeks/quote/guards/pnl/limits); A & C pending |
| `tests/integration/test_track_b_quote.py` | ✅ | Fixed-input Quote + toxicity widen + feed-gap pull + news widen + inventory cap |
| `tests/pipeline/` | ⬜ | End-to-end including Sec 6 replication |
| `tests/fixtures/` | ⬜ | Recorded book snapshots, synthetic paths |

## Change log

- **Day 0 (Apr 22 2026)** — Repo scaffolded. Pydantic schemas written. Config seeded. All three track folders created with READMEs. Nothing wired yet. Committed and pushed to `origin main`.
- **Apr 22 2026 — Track B Stage 1 complete.** Shipped `mm/greeks.py`, `mm/quote.py`, `mm/guards.py`, `mm/pnl.py`, `mm/limits.py`, `mm/refresh_loop.py` (paper §4.1, §4.2 eq 8-9, §4.5, §4.6). 110 unit+integration tests green. Critical integration test `tests/integration/test_track_b_quote.py` verifies expected Quote on fixed inputs, toxicity/news spread widening, feed-gap + inventory-cap pulls. Stage 2/3 (`mm/hedge/*`) still stubbed.
