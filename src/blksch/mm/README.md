# TRACK B — `mm/` — Quoting Engine

**Owner:** Window 2

**Responsibility:** turn Track A's calibrated surface + Track C's inventory into `Quote` messages. Run the refresh loop. Decide hedges. Manage kill-switches and PnL attribution.

**Status:** ⬜ pending. Day-0 scaffold only.

## Inputs and outputs

**Consumes:**
- `SurfacePoint`, `CorrelationEntry`, `LogitState` from Track A
- `BookSnap` from Track A (for queue-aware sizing)
- `Position`, `Fill` from Track C

**Emits:**
- `Quote` → Track C order router
- `HedgeInstruction` → Track C order router (stages 2–3)

## Files

| File | Paper § | Stage | Notes |
|---|---|---|---|
| `greeks.py` | 4.1 | 1 | `Δ_x = p(1-p)`, `Γ_x = p(1-p)(1-2p)`, `ν_b`, `ν_ρ` |
| `quote.py` | 4.2 eq 8–9 | 1 | Reservation + spread in logit; boundary floor; inventory cap |
| `guards.py` | 4.2 | 1 | Toxicity (VPIN), news window, queue discipline |
| `refresh_loop.py` | 4.5 | 1 | 100–500 ms asyncio cycle wiring A→B→C |
| `pnl.py` | 4.6 | 1 | Δ–Γ–ν_b–ν_ρ–jump attribution |
| `limits.py` | 4.6 | 1 | Kill-switches, auto-pause |
| `hedge/beta.py` | 4.4 | 2 | Cross-event β-hedge |
| `hedge/calendar.py` | 4.3 | 3 | Variance-strip sizing |
| `hedge/synth_strip.py` | 3.4 | 3 | Synthetic variance/corridor from vanilla basket |

## Build order

See the plan's Window-2 prompt. Ship each with unit tests before moving on:

1. `greeks.py` + `quote.py` (pure functions, heavy unit testing)
2. `guards.py`
3. `pnl.py` + `limits.py`
4. `refresh_loop.py` (wires it all)
5. **Stage 2:** `hedge/beta.py`
6. **Stage 3:** `hedge/calendar.py`, `hedge/synth_strip.py`

## Rules of engagement

- `quote.py` is a **pure function** — no IO, no network. Takes surface + inventory + book, returns `Quote`. Makes unit testing trivial and keeps the refresh loop fast.
- Boundary handling is critical. As `p → 0` or `1`:
  - `δ_p ≈ S'(x) δ_x → 0`, so the spread in p auto-compresses — floor it at `config.boundary.delta_p_floor_ticks`
  - Inventory cap `q_max ∝ 1/max(S'(x), ε)` — tightens as you approach the boundary
- Do not over-engineer hedges before Stage 1 validates vanilla MM. Stubs that return "no hedge" are fine in the refresh loop until Stage 2.
- If the kill-switch fires, **log why** and emit a structured alert — silent auto-pauses are hard to debug.
