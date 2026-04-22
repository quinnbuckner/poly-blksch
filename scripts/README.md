# `scripts/`

Manual / one-shot scripts. Not part of the `blksch` package — run directly:

```bash
python scripts/<name>.py [args]
```

## Files

| Script | When to run | Stage gate |
|---|---|---|
| `signing_canary.py` | Before first `--mode=live` | **Stage 1 → Stage 2 gate** |

## Safety rules

1. Any script that places orders must require a `--i-mean-it` flag and load credentials from `.env` — no hardcoded keys
2. Any script that touches shared infrastructure (db, dashboard server) must accept a `--dry-run` flag
3. Scripts must log a structured `{script, args, ts, user}` header on start so post-hoc audit works
