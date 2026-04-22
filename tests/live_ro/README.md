# `tests/live_ro/`

**Live read-only** smoke tests against Polymarket's public endpoints. These are manual / on-demand — they hit the network, so they're **excluded from default CI**.

## Purpose

Prove Polymarket's API hasn't drifted out from under us in ways a contract-test recording wouldn't catch:
- Endpoint moved / deprecated
- Auth requirements changed on previously-public endpoints
- WS channel renamed
- TLS cert issues

If these start failing unexpectedly, it's a signal to pull a fresh contract fixture and check release notes.

## Marker

All tests here use `@pytest.mark.live_ro`. They are **skipped by default**. Opt in explicitly:

```bash
pytest tests/live_ro -v -m live_ro
```

## What tests belong here

Only **read-only, unauthenticated** endpoints. Anything that signs or authenticates goes through `scripts/signing_canary.py` — a separate, gated harness.

## Safety

- No credentials required; these tests should never need a `.env`
- Tests should take < 10 seconds each to avoid hammering Polymarket
- Use well-known liquid token IDs — don't pick new ones each run (hot cache)

## When to run

- Before flipping Stage 1 paper-trading onto a new environment
- Pre-commit for changes to `polyclient.py` or `clob_client.py`
- When a contract test fails (to distinguish "API changed" from "our parser broke")
- Nightly, as a canary
