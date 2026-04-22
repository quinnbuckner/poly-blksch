# Architecture

## Three-track split

The bot is divided into three packages that talk only through the Pydantic contracts in `src/blksch/schemas.py`. This lets three Claude Code windows work in parallel without blocking each other.

```
┌───────────────────────────┐
│  TRACK A  core/           │  Data & Calibration
│  Polymarket WS/REST  ─►   │  → BookSnap, TradeTick
│  canonical mid + KF  ─►   │  → LogitState
│  EM (jumps)          ─►   │  → SurfacePoint, CorrelationEntry
└───────────────────────────┘
              │  (SurfacePoint, CorrelationEntry, LogitState)
              ▼
┌───────────────────────────┐
│  TRACK B  mm/             │  Quoting Engine
│  Greeks + AS-in-logit ─►  │  → Quote
│  toxicity/news guards     │
│  β-hedges (stage 2)       │
│  variance strips (stage 3)│
└───────────────────────────┘
              │  (Quote, HedgeInstruction)
              ▼
┌───────────────────────────┐
│  TRACK C  exec/           │  Execution & Infra
│  CLOB client (live) OR    │
│  paper_engine (sim)   ─►  │  → Order, Fill, Position
│  ledger + dashboard       │
└───────────────────────────┘
              │  (Position, Fill)
              └──── feeds back to Track B inventory state
```

## Shared contracts (`src/blksch/schemas.py`)

These are **frozen interfaces** — any change must be coordinated across all three tracks.

| Contract | Producer | Consumer(s) | Purpose |
|---|---|---|---|
| `BookSnap` | Track A | B, C | L2 book snapshot |
| `TradeTick` | Track A | B, C | Executed trade |
| `LogitState` | Track A | B | Filtered x̂_t + noise variance |
| `SurfacePoint` | Track A | B | (τ, m) point estimate of σ̂_b, λ̂, ŝ²_J |
| `CorrelationEntry` | Track A | B | De-jumped ρ̂_ij + co-jump moments |
| `Quote` | Track B | C | Target bid/ask in both x and p units |
| `Order` | Track C | — | Live or paper order descriptor |
| `Fill` | Track C | B | Executed fill |
| `Position` | Track C | B | Inventory mark and PnL |

## Data flow per refresh cycle (§4.5, 100–500 ms)

1. **Track A** pushes latest `LogitState`, `SurfacePoint`, `CorrelationEntry` via an asyncio event bus.
2. **Track B** refresh loop:
   1. Pull current surface + inventory from `Position`
   2. Compute `r_x`, `δ_x` (§4.2 eq 8–9)
   3. Apply boundary floor `δ_p` and inventory cap `q_max`
   4. Apply toxicity / news guards — widen or pull
   5. Emit `Quote` to Track C
3. **Track C** order router:
   1. Diff new `Quote` against in-flight orders → cancel/replace/place
   2. Route to either `paper_engine` (stage 1) or `clob_client` (stage 2+)
   3. On `Fill`, update `ledger` → emit new `Position`

## Paper sections mapped to modules

| Paper section | Module |
|---|---|
| §3.2 — RN drift for logit JD | `core/em/rn_drift.py` |
| §3.3 — cross-event correlation / co-jumps | `core/surface/corr.py` |
| §3.4 — prototype derivatives | `mm/hedge/synth_strip.py` (stage 3) |
| §4.1 — Greeks in logit | `mm/greeks.py` |
| §4.2 — Avellaneda–Stoikov in logit | `mm/quote.py` |
| §4.3 — calendar hedges | `mm/hedge/calendar.py` (stage 3) |
| §4.4 — cross-event β-hedges | `mm/hedge/beta.py` (stage 2) |
| §4.5 — refresh loop | `mm/refresh_loop.py` |
| §4.6 — PnL attribution + kill-switches | `mm/pnl.py`, `mm/limits.py` |
| §5.1 — data conditioning | `core/filter/canonical_mid.py`, `core/filter/microstruct.py` |
| §5.1 — heteroskedastic KF | `core/filter/kalman.py` |
| §5.2 — EM for jumps | `core/em/increments.py`, `core/em/jumps.py` |
| §5.3 — surface smoothing | `core/surface/smooth.py` |
| §5.4 — de-jumped correlation | `core/surface/corr.py` |
| §6 — evaluation protocol | `tests/pipeline/test_paper_sec6_replication.py` |

## Configuration

All tunable parameters live in `config/bot.yaml`. Markets and screener filters in `config/markets.yaml`. Seeds come from the plan's *Defaults & Config Seeds* table — expect to tune `γ`, `k`, and `q_max` on paper-trading data before live promotion.
