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

---

## Operator runbook

> **Intended audience:** the person on watch while the bot is running. Read top to bottom before touching anything; every step has a "what to verify" line.

### 0. One-time setup

1. `python3.14 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev,clob]'`
2. Copy `.env.example` → `.env` once live creds are in flight. Stage 1 does not need a `.env`.
3. `pytest tests/unit tests/integration tests/contract -q` — must be green before any live work.

### 1. Run paper mode (Stage 1)

```bash
python -m blksch.app --mode=paper --market=<token_id>
```

* Dashboard: `RichDashboard` prints to the terminal at ~1 Hz. Optionally `python -m blksch.exec.dashboard --flask --port 5055` to expose `/api/state`.
* Ledger DB lands under `data/blksch_paper.db` (sqlite, WAL). Safe to `rm` between runs.
* **Verify:** quotes inside the book, `inventory_q` mean-reverting, `PnL.fees_usd == 0` (maker rebates not modelled), no `engine: HALTED` banner.

### 2. Signed pre-flight (before the canary)

```bash
python scripts/live_ro_auth_check.py                # prints the plan, exit 0
python scripts/live_ro_auth_check.py --i-mean-it    # signed GET /orders
```

* Loads `.env` (python-dotenv) → validates `POLY_PRIVATE_KEY` and `POLY_FUNDER_ADDRESS` → constructs `PyCLOBClient` → derives L2 API creds if missing → calls signed `get_orders`.
* Zero side effects — if this fails, the signing pipeline is broken.
* **Verify:** log line `SIGNED GET /orders OK — maker has N open order(s)`, exit 0.

### 3. Signing canary (Stage-1 → Stage-2 gate)

```bash
python scripts/signing_canary.py --token-id <TOKEN>                   # dry-run
python scripts/signing_canary.py --token-id <TOKEN> --i-mean-it       # mainnet
python scripts/signing_canary.py --token-id <TOKEN> --i-mean-it --testnet
```

* Places a $1-notional BUY far below the book (`--far-price 0.01`), sleeps 3 s, cancels.
* Uses `OrderSide.BUY` + a `canary-…` client_id routed through `PyCLOBClient.place_order`.
* **If POST /order fails:** investigate signing or gas / MATIC balance. No cleanup needed.
* **If DELETE /order fails:** the log prints the `venue_id` — cancel manually via the Polymarket UI before retrying.
* **Verify:** log line `CANARY PASSED`. Exit 0.

### 4. Flip to live mode (Stage 2 — promoted only after ≥72 h clean paper runs)

```bash
# 1. Do NOT flip mode in config yet. The router refuses to construct without live_ack.
python -m blksch.app --mode=live --market=<token_id> --live-ack
```

* `--live-ack` wires `RouterConfig(mode="live", live_ack=True)`. Without it, `OrderRouter.__init__` raises at boot — that is by design.
* Confirm `ORDER place [mode=live]` log lines appear on every emission.
* **Verify:** `Position.realized_pnl_usd` advancing in the dashboard; reconciler runs every `pnl.reconcile_tolerance_usd`-driven interval without drift alerts.

### 5. Kill-switches — where they live and how they fire

| Trigger | Source | What it does | How to clear |
|---|---|---|---|
| Feed gap (`feed_gap_sec`, default 3) | `PaperEngine._gap_check` → `state.halted=True` | Suppresses all fills; router-driven placements are rejected | Feed recovery + `paper_engine.resume()` via REPL, or process restart |
| Volatility spike (Z-score on σ̂_b) | `mm/limits.py` (Track B) | Widens quotes → pauses | Track B clears when realized σ normalizes |
| Repeated pick-offs | `mm/limits.py` (Track B) | Pauses for `repeated_pickoff_window_sec` | Auto-clears on window expiry |
| Max drawdown USD | `mm/limits.py` (Track B) | Full shutdown | Explicit operator restart after investigation |
| Live-ack missing | `OrderRouter.__init__` | Raises on boot | Pass `--live-ack` intentionally |

### 6. `PaperEngine.state.halted` semantics

* `state.halted = True` ⇒ `on_book` / `on_trade` stop generating fills and `place_order` returns status `REJECTED`.
* `state.halt_reason` is a short string (e.g. `"feed_gap 5.2s > 3.0s"`). Always log it.
* Clearing: the halt is **not self-healing**. Either:
  1. Call `engine.resume()` from a REPL / debug endpoint after you confirm the upstream issue is gone (feed healthy, spread sane, no stale positions).
  2. Restart the process. The ledger survives restarts — resuming from a persistent SQLite state is the default.
* **Do not** blanket-resume without checking `ledger.open_orders()` — stale orders may still be live on the venue if you restart after a live-mode halt.

### 7. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `py-clob-client OK` but `place_order` raises `RuntimeError: signed endpoints require ...` | `.env` not loaded / missing `POLY_PRIVATE_KEY` | `load_dotenv()` before client construction; re-run `live_ro_auth_check.py` |
| Canary POST ok, DELETE times out | Upstream latency or venue-id shape drift | Copy venue_id from log, cancel via UI, bump canary retry window |
| `engine=HALTED feed_gap ...` immediately after start | First tick older than `feed_gap_sec` vs `now()` — clock skew or stale WS replay | Sync system clock; disable replay-from-disk feed |
| Dashboard shows stale `quotes` but rising `fills_count` | Track B refresh loop stopped; router still reacting to old state | Check `mm/refresh_loop` log for exceptions; restart the loop task |
| PnL drift between `ledger.pnl()` and `reconcile(fills, mark)` | WAP sign convention mismatch in a caller | `reconcile` is the ground truth — diff the two, file a bug |
