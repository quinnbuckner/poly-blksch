# `blksch/` — Package Overview

Top-level Python package for the bot. Each subpackage is owned by one of the three tracks (see `../../ARCHITECTURE.md`).

## Layout

```
blksch/
├── schemas.py        # FROZEN shared Pydantic contracts (all tracks consume)
├── app.py            # entrypoint (paper|live mode selector)
├── core/             # TRACK A — data & calibration
├── mm/               # TRACK B — quoting & hedges
└── exec/             # TRACK C — order execution & ledger
```

## Shared contracts (`schemas.py`)

These are the integration glue. **Do not modify without coordinating across all three tracks.**

| Contract | Producer | Consumers |
|---|---|---|
| `BookSnap`, `TradeTick` | Track A | B, C |
| `LogitState` | Track A | B |
| `SurfacePoint` | Track A | B |
| `CorrelationEntry` | Track A | B |
| `Quote`, `HedgeInstruction` | Track B | C |
| `Order`, `Fill`, `Position` | Track C | B |

## Conventions

- All timestamps are timezone-aware `datetime` UTC.
- Probability `p ∈ (0, 1)` — clip to `[ε, 1-ε]` (`ε=1e-5`) before `logit`.
- Logit `x = log(p / (1-p))` and `S(x) = 1 / (1 + exp(-x))`. `S'(x) = p(1-p)`, `S''(x) = p(1-p)(1-2p)`.
- Risk-neutral measure ℚ. Discounted `p_t` is a ℚ-martingale. The drift on `x_t` is computed (not free) from the paper's eq (3) to enforce this.
- Inventory `q` is signed: positive = long YES shares.
