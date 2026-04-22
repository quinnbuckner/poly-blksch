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
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/unit -v         # fast sanity check
python -m blksch.app --mode=paper  # (not wired yet at Day-0)
```

## Staging

| Stage | Scope | Gate |
|---|---|---|
| 0 | Scaffold + Track A calibration | Paper §6 replication test passes |
| 1 | Tracks A+B+C on paper engine, single market | 72h paper run, positive edge, stable inventory |
| 2 | Live CLOB orders, small size | 1-week live run, PnL attribution reconciles |
| 3 | Cross-event β-hedges | Net cross-event vega within limit |
| 4 | Synthetic variance/corridor strips | Replication tracking error within bound |

## Status

**Stage 0, Day 0 (scaffold).** See [`CATALOG.md`](./CATALOG.md) for current inventory.
