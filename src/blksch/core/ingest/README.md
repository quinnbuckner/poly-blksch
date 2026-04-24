# `core/ingest/` — Market Data Ingestion

IO layer for Polymarket. Keep this async and IO-pure — no math here.

## Planned files

- `polyclient.py` — REST (Gamma + CLOB) + WS subscriber. Reuse the rate-limiter / session-pooling pattern from `../../../../../mm-v1.0/polyarb_v1.0/src/api.py`. Extend with `websockets` for L2 book and trade channels.
- `store.py` — Parquet tick store (one file per (token_id, hour)). `pyarrow` backend. Lean schema matching `schemas.BookSnap` / `schemas.TradeTick`.
- `screener.py` — top-liquidity picker. Reads `config/markets.yaml`, ranks markets by composite of volume + depth + spread tightness, re-screens every `rescreen_every_sec`.

## Polymarket API surface (from docs.polymarket.com)

- Gamma REST: `GET /markets` (list/paginate), `GET /markets/{slug}`
- CLOB REST: `GET /book?token_id=...`, `GET /prices`, `GET /trades`
- CLOB WS: `book` channel, `trades` channel (subscribe by `token_id`)

## Rate-limit discipline

Polymarket's rate limits are not officially documented for all endpoints; prior work (`mm-v1.0/polyarb_v1.0`) ran at 12 req/s on Gamma with no trouble. Keep the rate-limiter conservative (~10 req/s default) and use WS for hot book updates.

## Mock-mode rehearsals (no network)

Any `token_id` matching the prefixes in `polyclient.MOCK_TOKEN_PREFIXES` — currently `0xmock` and `mock:`, case-insensitive — routes through `polyclient._mock_stream_market` instead of the real endpoints. So `0xmock`, `0xMOCKfoo`, and `mock:btc-70k` all qualify.

Used for:

- `paper_soak.py --token-id 0xmock` — full rehearsal of the Stage-1 supervisor without any live network. Produces a deterministic GBM-driven quote stream so the dashboard, ledger, and PnL plumbing all see traffic.
- Unit tests of downstream modules (filter, calibration, dashboard) that need a synthetic `BookSnap` / `TradeTick` generator without pulling in the paper-§6 synthetic-path fixture machinery.

Routing lives inside `PolyClient.stream_market` and `PolyClient.get_book`, so callers do not need to change anything — pass a mock `token_id` and the client transparently serves it. Mixing mock and real `token_id`s in a single `stream_market` call is a `ValueError`; multiplexing a live WS with a synthetic stream would give false quote-uptime readings.

Determinism knobs (sensible defaults; tune only when byte-reproducible test output is required):

- `MOCK_POLYCLIENT_SEED` env var (default `42`) — RNG seed.
- `MockPolyClient(seed=..., interval_s=..., sigma_b=..., trade_probability=..., start_ts=...)` — per-instance overrides for tests that need to pin both the RNG and timestamps.

See `polyclient.py` (module-level `MOCK_*` constants + `MockPolyClient`) and `tests/unit/test_polyclient.py` (the five `mock_polyclient_*` / `is_mock_token` tests) for the full contract.
