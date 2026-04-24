# `scripts/`

Manual / one-shot scripts. Not part of the `blksch` package — run directly:

```bash
python scripts/<name>.py [args]
```

## Files

| Script | When to run | Stage gate |
|---|---|---|
| `calibration_dryrun.py` | Before each paper-soak launch; picks a market via `--auto` and emits GREEN/YELLOW/RED verdict | **Stage 0 → Stage 1 entry** |
| `paper_soak.py` | 72h supervised paper-trade run; polls `/api/state`, aggregates hourly reports, evaluates acceptance criteria | **Stage 1 acceptance** |
| `reconcile_drill.py` | Hourly during soak (and always before any live run) — ledger ↔ venue drift detector, 6 discrepancy kinds | Stage-1 monitoring / pre-Stage-2 |
| `record_contract_fixtures.py` | One-shot live-RO recorder for `tests/contract/fixtures/*`; re-run to refresh schema | Stage-2 prep |
| `live_ro_auth_check.py` | Before signing_canary — verify auth plumbing without placing orders | Pre-canary |
| `signing_canary.py` | $1 far-from-market order + cancel on testnet then mainnet | **Stage 1 → Stage 2 gate** |

## Safety rules

1. Any script that places orders must require a `--i-mean-it` flag and load credentials from `.env` — no hardcoded keys
2. Any script that touches shared infrastructure (db, dashboard server) must accept a `--dry-run` flag
3. Scripts must log a structured `{script, args, ts, user}` header on start so post-hoc audit works
