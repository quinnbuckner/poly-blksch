# Tests

Three levels, each with its own folder. CI gates unit + integration on every commit; pipeline tests run nightly and pre-stage-promotion.

## Layout

```
tests/
├── unit/          pure math, synthetic inputs, fast (seconds)
├── integration/   per-track with fixtures (seconds to ~1 minute)
├── pipeline/      end-to-end, including Sec 6 replication (minutes)
└── fixtures/      recorded book snapshots, synthetic RN-consistent paths, known test vectors
```

## Running

```bash
pytest tests/unit -v                       # fast feedback
pytest tests/unit tests/integration -v     # CI gate
pytest tests/pipeline -v                   # nightly / pre-promotion
pytest --cov=blksch --cov-report=term-missing
```

## Critical tests (stage gates)

### Stage 0 → 1 gate: `tests/pipeline/test_paper_sec6_replication.py`

Reproduces the paper's §6 setup — synthetic RN-consistent path, 1 Hz, N=6000 steps, forecast horizon h=60s — runs the full Track A pipeline causally, and asserts MSE/MAE/QLIKE match the RN-JD row of Table 1 within 10%:

- `MSE_all ≈ 70.28`
- `MAE_all ≈ 1.59`
- `QLIKE_all ≈ 1.46`

Until this passes, **Track A is not trustworthy** and Tracks B/C must operate against fixtures only.

### Stage 1 → 2 gate: `tests/pipeline/test_live_paper_trade.py`

1-hour dry run against live WS feed with `paper_engine`. Asserts:
- No crashes
- Quotes always inside the book (never cross)
- Inventory within `q_max` caps
- PnL attribution sums reconcile within `config.pnl.reconcile_tolerance_usd`

### Stage kill-switch: `tests/pipeline/test_kill_switch.py`

Injects feed gap, volatility spike, and repeated pick-off scenarios. Asserts auto-pause fires within one refresh cycle for each.

## Unit test responsibilities by track

**Track A:**
- `test_kalman.py` — recover known `x_t` on synthetic path (eq 10 noise)
- `test_em.py` — recover known `(σ_b, λ, s_J)` on synthetic JD
- `test_rn_drift.py` — martingale check via Monte Carlo
- `test_surface_smooth.py` — shape constraints hold

**Track B:**
- `test_greeks.py` — identities `Δ_x = p(1-p)`, `Γ_x = p(1-p)(1-2p)`, `S` round-trip
- `test_quote.py` — reservation skew sign with inventory; spread monotonicity in γ/σ/(T-t); boundary floor activation
- `test_limits.py` — kill-switch thresholds fire correctly
- `test_pnl.py` — attribution identity holds on a scripted sequence

**Track C:**
- `test_signer.py` — EIP-712 signature matches known Polymarket test vectors
- `test_ledger.py` — PnL reconciliation on a sequence of fills
- `test_paper_engine.py` — conservative queue model produces expected fills on scripted book sequence
