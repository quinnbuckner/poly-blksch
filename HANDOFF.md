# Handoff Guide — Picking Up Work in a Fresh Session

If you're a fresh Claude Code session (or a new collaborator) resuming this project, read these **in order**:

## 1. Core documents

1. `README.md` — what the bot does, quick start
2. `ARCHITECTURE.md` — three-track split, data flow, shared contracts
3. `CATALOG.md` — **current status** of every module (built / in progress / pending)
4. `/Users/quinnbuckner/.claude/plans/i-want-to-make-validated-crane.md` — full implementation plan with staging gates
5. `/Users/quinnbuckner/Downloads/Toward Black-Scholes for Prediction Markets.pdf` — the paper we're implementing

## 2. The paper maps to specific modules

See the table at the bottom of `ARCHITECTURE.md`. Before touching a module, re-read the paper section it implements — the code is meaningless without the math.

## 3. Where to work

The repo is split into three parallel tracks. Each has its own folder with a README describing its scope and current state:

- **Track A** — `src/blksch/core/README.md`  (data + calibration)
- **Track B** — `src/blksch/mm/README.md`    (quoting + hedges)
- **Track C** — `src/blksch/exec/README.md`  (orders + paper/live execution)

If you work on one track in isolation, you can stub the outputs of the other two using the Pydantic contracts in `src/blksch/schemas.py`. **Never modify those contracts without coordinating across all three tracks** — they are the integration glue.

## 4. Before you change anything

1. `git pull` and read `CATALOG.md` for the latest status
2. Identify which stage (0–4) we're in — see the staging table in `README.md`
3. Re-read the paper section for the module you're touching
4. Check `tests/` for existing fixtures that constrain behavior
5. **Do not skip ahead.** If a stage gate hasn't passed, don't build the next stage's modules.

## 5. Before you commit

1. Run `pytest tests/unit tests/integration -v` and keep them green
2. Update `CATALOG.md` — mark what you built, what's still pending
3. Update the track-level `README.md` if the module you built changes its interface
4. Commit with a message describing the paper section and stage, e.g. `mm/quote: implement §4.2 eq (8-9) reservation + spread (Stage 1)`
5. Push to `origin main` (https://github.com/quinnbuckner/poly-blksch)

## 6. Common hazards

- **Log-odds explosion near 0/1.** Clip `p` to `[ε, 1−ε]` (ε=1e-5) before taking `logit`. Watch for `S'(x) → 0` which makes hedge ratios explode; the inventory cap in `mm/quote.py` exists for exactly this reason.
- **Kalman filter divergence.** If `p` is pinned near 0/1 for long stretches, switch to UKF or particle smoother (paper §5.1 note).
- **EM convergence.** Paper uses ~6 global EM steps then a rolling 400s window. Don't run it to convergence per-tick — that's too expensive.
- **Fee model.** Polymarket's fee schedule changes. Stage 2 (live) is gated on confirming the current maker/taker fees and wiring them into `mm/pnl.py` expected edge.
- **Testnet vs mainnet.** CLOB addresses and chain IDs differ. `config/bot.yaml` must select.

## 7. Running the paper's §6 replication test

This is the **calibration correctness gate** for Stage 0 → Stage 1 promotion:

```bash
pytest tests/pipeline/test_paper_sec6_replication.py -v
```

Target metrics (paper Table 1, RN-JD row): `MSE ≈ 70.28`, `MAE ≈ 1.59`, `QLIKE ≈ 1.46`. Stay within 10% and the calibration pipeline is trustworthy.

## 8. Credentials & secrets

Never commit `.env`. Stage 0–1 needs no credentials. Stage 2+ requires:

```
POLY_PRIVATE_KEY=...
POLY_API_KEY=...
POLY_API_SECRET=...
POLY_API_PASSPHRASE=...
POLY_FUNDER_ADDRESS=...
```

Derive these via the Polymarket onboarding flow documented at https://docs.polymarket.com/.

## 9. Useful neighbor code

- `../mm-v1.0/polyarb_v1.0/src/api.py` — reference for HTTP session pooling and rate-limiting (reuse pattern, not the module)
- `../mm-v1.0/polyarb_v1.0/src/database.py` — SQLite schema reference for `exec/ledger.py`
