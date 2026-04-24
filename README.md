# blksch — Polymarket Market-Making Bot

A Python market-making bot for [Polymarket](https://polymarket.com) that implements the pricing kernel, calibration pipeline, and quoting strategy from:

> Dalen, S. (2026). *Toward Black–Scholes for Prediction Markets: A Unified Kernel and Market-Maker's Handbook.* arXiv:2510.15205v2.

## What it does

1. **Calibrates** a logit jump-diffusion model with risk-neutral drift to live Polymarket CLOB mid/bid/ask/trade streams (paper §3.2, §5).
2. **Quotes** continuously on vanilla YES/NO contracts using an Avellaneda–Stoikov reservation price and spread in logit units, with inventory caps that tighten near the 0/1 boundaries (paper §4.1–4.2).
3. **Hedges** cross-event exposure with correlated Polymarket markets (paper §4.4) — *Stage 2*.
4. **Synthesizes** variance/corridor strips from portfolios of vanilla contracts (paper §3.4, §4.3) — *Stage 3*.

Paper trading first; live promotion is gated on test results at each stage.

## Orientation (read these first)

| File | Purpose |
|---|---|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Three-track split, data flow, shared contracts |
| [`HANDOFF.md`](./HANDOFF.md) | How to pick up work in a fresh Claude Code session |
| [`CATALOG.md`](./CATALOG.md) | Living inventory of what's built and what's pending |
| `~/.claude/plans/i-want-to-make-validated-crane.md` | Full implementation plan |

## Quick start

```bash
git clone https://github.com/quinnbuckner/poly-blksch.git
cd poly-blksch
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,clob]"
pytest tests/unit tests/integration tests/contract -q  # expect 1086/15/0
```

### Stage-1 operator path

```bash
# 1. Pre-flight
lsof -i :5055                                                            # dashboard port free?
python scripts/live_ro_auth_check.py                                     # live-RO sanity (no .env)

# 2. Pick a market (per-market GO/NO-GO)
python scripts/calibration_dryrun.py --auto --minutes 15 --out ./runs/dryrun-1/

# 3. 72h paper-soak (Stage-1 acceptance gate)
python scripts/paper_soak.py --i-mean-it --token-id <GREEN_TOKEN> --hours 72
```

Dashboard on `http://127.0.0.1:5055/api/state`. Final verdict in `soak_output/final_report.json`.

## Staging

| Stage | Scope | Gate |
|---|---|---|
| 0 | Scaffold + Track A calibration | ✅ Paper §6 replication (multi-seed median MSE in ±10%) |
| 1 | Tracks A+B+C on paper engine, single market | 72h paper run, positive edge, stable inventory |
| 2 | Live CLOB orders, small size | 1-week live run, PnL attribution reconciles |
| 3 | Cross-event β-hedges | Net cross-event vega within limit |
| 4 | Synthetic variance/corridor strips | Replication tracking error within bound |

## Status

**Stage 0 CLEARED; Stage 1 ready for paper-soak launch.** Fast suite 1086/15/0. See [`CATALOG.md`](./CATALOG.md) for current inventory and [`SESSION_RESUME.md`](./SESSION_RESUME.md) for recent history.
