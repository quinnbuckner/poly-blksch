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
