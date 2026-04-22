# config/

Runtime configuration for the bot. All values are **seeds** — tune on paper-trading data before Stage 2 (live) promotion.

## Files

- `bot.yaml` — risk parameters, calibration knobs, refresh cadence, kill-switch thresholds
- `markets.yaml` — screener filters, correlation-pair hints, blocklist

## Key knobs and where they bite

| Knob | File | Module that reads it | Paper |
|---|---|---|---|
| `quoting.gamma` | `bot.yaml` | `mm/quote.py` | §4.2 eq 8–9 |
| `quoting.k` | `bot.yaml` | `mm/quote.py` | §4.2 |
| `boundary.eps` | `bot.yaml` | `core/filter/canonical_mid.py`, `mm/quote.py` | §5.1 |
| `inventory.q_max_notional_usd` | `bot.yaml` | `mm/quote.py`, `mm/limits.py` | §4.2 |
| `loop.refresh_ms` | `bot.yaml` | `mm/refresh_loop.py` | §4.5 |
| `calibration.em_window_sec` | `bot.yaml` | `core/em/*` | §6.4 |
| `limits.feed_gap_sec` | `bot.yaml` | `mm/limits.py` | §4.6 |
| `screener.top_n` | `markets.yaml` | `core/ingest/screener.py` | — |

## Tuning order

1. Run §6 replication test (Stage 0 gate) with `calibration` defaults. If it fails, tune `em_window_sec`, `em_global_steps`, `mc_draws_per_step`.
2. Paper-trade (Stage 1) and observe inventory half-life. Tune `quoting.gamma` so inventory mean-reverts within ~1/10th of average time-to-resolution.
3. Fit `quoting.k` from realized fill distance vs. order-arrival intensity (log-linear regression on post-run data).
4. Adjust `inventory.q_max_notional_usd` upward only after Stage 2 shows stable PnL attribution.
5. `limits.*` are safety catches — tighten, never loosen, without explicit sign-off.
