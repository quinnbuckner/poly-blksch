# TRACK C — `exec/` — Execution & Infrastructure

**Owner:** Window 3

**Responsibility:** submit orders to either `paper_engine` (Stage 1) or the Polymarket CLOB (Stage 2+). Maintain the authoritative ledger of positions, fills, fees, and PnL. Provide the live dashboard.

**Status:** ⬜ pending. Day-0 scaffold only.

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

| File | Stage | Notes |
|---|---|---|
| `clob_client.py` | 1 (read), 2 (write) | Polymarket CLOB REST. Try `py-clob-client` first; fall back to custom if stale |
| `signer.py` | 2 | EIP-712 typed-data signing (`eth-account`) — only if not using py-clob-client |
| `ledger.py` | 1 | SQLite: positions, fills, fees, realized/unrealized PnL |
| `paper_engine.py` | 1 | Simulated matching against live L2 book with conservative queue model |
| `order_router.py` | 1 | Idempotent place / cancel / replace, retry/backoff, routes paper vs. live |
| `dashboard.py` | 1 | Rich terminal view + optional Flask endpoint |

## Build order

See the plan's Window-3 prompt. Ship each with unit tests:

1. `clob_client.py` (read-only endpoints first)
2. `ledger.py`
3. `paper_engine.py`
4. `order_router.py`
5. `dashboard.py`

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
