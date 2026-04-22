# `tests/contract/`

Contract tests pin our parsers against **recorded real responses** from Polymarket. They run offline (no network), fast (seconds), and catch the single most common production failure: the exchange changes a response field and our Pydantic models silently drop or reject it.

## Marker

All tests here use `@pytest.mark.contract`. They run in default CI (no `-m` filter needed) because they're offline and fast.

## When to update the recorded fixtures

Record a fresh fixture when:

1. Polymarket announces an API change
2. A `live_ro` smoke test starts failing
3. You're onboarding a new endpoint

## How to record a fixture

```bash
# Track A owns polyclient; Track C owns clob_client. Whichever client fetches
# the response, pipe its raw JSON to a fixture file:
python - <<'PY'
import json
import httpx
r = httpx.get("https://clob.polymarket.com/book?token_id=<known liquid token>")
with open("tests/contract/fixtures/book_sample.json", "w") as f:
    json.dump(r.json(), f, indent=2)
PY
```

**Important:** scrub any wallet addresses, API keys, or PII before committing. Markets and tokens are public, orders are not.

## File layout

```
tests/contract/
├── README.md                           (this file)
├── test_polyclient_contracts.py        Track A polyclient parse tests
├── test_clob_client_contracts.py       Track C clob_client parse tests (pending)
└── fixtures/
    ├── book_sample.json                recorded GET /book response
    ├── markets_sample.json             recorded GET /markets page
    ├── trades_sample.json              recorded GET /trades response
    └── ws_book_messages.jsonl          recorded WS book channel messages
```

## What contract tests do NOT cover

- **Correctness of our logic after parsing** — that's unit/integration territory
- **Liveness of the endpoint** — `tests/live_ro/` covers that
- **Signed endpoints** — see `scripts/signing_canary.py`

## Failure handling

A contract test failure almost always means Polymarket changed the response schema. Procedure:
1. Re-record the fixture
2. Diff old vs. new fixture
3. Update our Pydantic model only if the change is additive (new optional field) or mandatory (field type changed)
4. Add a migration note in `CATALOG.md` change log
