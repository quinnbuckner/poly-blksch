# TRACK C — `exec/` — Execution & Infrastructure

**Owner:** Window 3

**Responsibility:** submit orders to either `paper_engine` (Stage 1) or the Polymarket CLOB (Stage 2+). Maintain the authoritative ledger of positions, fills, fees, and PnL. Provide the live dashboard.

**Status:** ✅ Stage 1 complete. Paper engine, ledger, signer, router, and dashboard are shipped with unit + integration tests green. Stage 2 (live CLOB routing) is wired but gated on `RouterConfig.live_ack=True` — no real orders can be placed until that flag is set.

> Note on the name `exec`: shadows Python's `exec()` builtin only inside `blksch.exec`. Not actually a problem — Python treats it as a regular identifier — but don't `from blksch.exec import *`.

## Inputs and outputs

**Consumes:**
- `Quote`, `HedgeInstruction` from Track B
- `BookSnap` from Track A (paper engine needs it to simulate matches)

**Emits:**
- `Order` (for logging / dashboard)
- `Fill` → Track B ledger state
- `Position` → Track B refresh loop

## Files

| File | Stage | Status | Notes |
|---|---|---|---|
| `clob_client.py` | 1 (read), 2 (write) | ✅ | `make_clob_client()` auto-picks `PyCLOBClient` (py-clob-client adapter, async via `to_thread`) or `HttpCLOBClient` (aiohttp read-only fallback). Returns `BookSnap` / `Order` — never raw upstream types. |
| `signer.py` | 2 | ✅ | EIP-712 Polymarket CTF Exchange v1 typed-data signer; deterministic, signature recovery verified. Used only when falling back off `py-clob-client`. |
| `ledger.py` | 1 | ✅ | SQLite (WAL) ledger for orders/fills/positions/marks. Signed-qty WAP accounting with sign-flip handling; fees reduce realized PnL. `reconcile()` is a pure-Python cross-check used by tests. |
| `paper_engine.py` | 1 | ✅ | Conservative matcher: fills on book trade-through or opposite-aggressor TradeTick; `queue_haircut` (default 0.5) models queue position; halts on `feed_gap_sec`. |
| `order_router.py` | 1 | ✅ | Idempotent `sync_quote(Quote)` aligns resting orders to the target; `place`/`cancel`/`replace`/`cancel_all` primitives; exponential backoff + jitter retry. Live mode refuses to construct unless `RouterConfig.live_ack=True`. |
| `dashboard.py` | 1 | ✅ | `RichDashboard` (asyncio `rich.Live` loop) + `FlaskDashboard` (`/api/state`, `/api/pnl`, `/api/health`). Both read from a shared `DashboardContext` — never mutate bot state. |

## Interfaces

* **Schemas (frozen):** router consumes `Quote`; router/engine emit `Order`/`Fill`; ledger returns `Position`. See `blksch/schemas.py`.
* **Public entry points:** `OrderRouter.sync_quote(Quote)` is the only call Track B's refresh loop needs. `PaperEngine.on_book(BookSnap)` / `PaperEngine.on_trade(TradeTick)` are the only calls Track A's polyclient needs.
* **Mode routing:** Stage 1 wires `OrderRouter(paper_backend=PaperEngine(...))`; Stage 2 swaps to `OrderRouter(live_backend=make_clob_client(...), config=RouterConfig(mode="live", live_ack=True))`.

## Build order (historic)

1. ✅ `clob_client.py` + `signer.py`
2. ✅ `ledger.py`
3. ✅ `paper_engine.py`
4. ✅ `order_router.py`
5. ✅ `dashboard.py`

## Paper engine (Stage 1) matching model

Conservative: fills our resting quote only when the market **trades through** our price (i.e., there's an aggressor that would have consumed our queue position). We model a haircut to account for queue position — assume we're behind 50% of the resting size at our level.

- Our `BUY @ p_bid` fills on trades where `aggressor_side = SELL` and `trade.price ≤ p_bid`, up to our size minus queue haircut.
- Our `SELL @ p_ask` fills on trades where `aggressor_side = BUY` and `trade.price ≥ p_ask`, symmetrically.
- No marketable-order simulation (we don't cross the book in paper mode).

This is intentionally pessimistic — real fills may be better, and Stage 2 will recalibrate.

## Live promotion (Stage 2) safety

- Orders routed to `clob_client` only when `mode == "live"` **and** an explicit `--live-ack` flag is passed on `app.py`
- Log mode prominently on every quote emission
- Size cap: `config.inventory.q_max_notional_usd` is the per-side ceiling; no override in code
- `.env` required for live — read via `python-dotenv`, never hardcode

## Reconciliation

Every N minutes, reconcile the ledger against the CLOB's reported positions and open orders. Any drift beyond `config.pnl.reconcile_tolerance_usd` triggers an alert and (optionally) a kill-switch.
