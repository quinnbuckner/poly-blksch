"""Live monitoring for the bot.

Two entry points:

* :class:`RichDashboard` — terminal ``rich.Live`` view. Updates at ~1 Hz,
  showing inventory, PnL, last-N fills, current quotes, and kill-switch
  status. Good enough for dev / paper-trading.
* :class:`FlaskDashboard` — optional JSON endpoints + a minimal static HTML
  shell. Mirrors the pattern from ``../mm-v1.0/polyarb_v1.0/scripts/dashboard_server.py``
  so frontend users can poll ``/api/state``.

Both read from a shared :class:`DashboardContext` whose state is populated
by the refresh loop (Track B) and the paper engine (Track C). Nothing here
mutates bot state — the dashboard is purely an observer.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from blksch.schemas import Fill, Position, Quote, SurfacePoint

from .ledger import Ledger, PnLSnapshot
from .paper_engine import PaperEngineState

log = logging.getLogger(__name__)

_MAX_FILL_HISTORY = 50


@dataclass
class DashboardContext:
    """Shared observer state. Readers must treat it as read-only."""

    ledger: Ledger
    mode: str = "paper"
    quotes: dict[str, Quote] = field(default_factory=dict)  # token_id -> latest Quote
    surface: dict[str, SurfacePoint] = field(default_factory=dict)  # token_id -> latest
    engine_state: PaperEngineState | None = None
    recent_fills: deque[Fill] = field(default_factory=lambda: deque(maxlen=_MAX_FILL_HISTORY))
    kill_switches: dict[str, bool] = field(default_factory=dict)

    def on_quote(self, quote: Quote) -> None:
        self.quotes[quote.token_id] = quote

    def on_surface(self, point: SurfacePoint) -> None:
        self.surface[point.token_id] = point

    def on_fill(self, fill: Fill) -> None:
        self.recent_fills.append(fill)

    def snapshot_dict(self) -> dict[str, Any]:
        pnl = self.ledger.pnl()
        positions = self.ledger.all_positions()
        return {
            "ts": datetime.now(UTC).isoformat(),
            "mode": self.mode,
            "pnl": {
                "realized_usd": round(pnl.realized_usd, 4),
                "unrealized_usd": round(pnl.unrealized_usd, 4),
                "fees_usd": round(pnl.fees_usd, 4),
                "total_usd": round(pnl.total_usd, 4),
            },
            "positions": [_position_dict(p) for p in positions],
            "quotes": {k: _quote_dict(q) for k, q in self.quotes.items()},
            "surface": {k: _surface_dict(s) for k, s in self.surface.items()},
            "engine": _engine_dict(self.engine_state) if self.engine_state else None,
            "kill_switches": dict(self.kill_switches),
            "recent_fills": [_fill_dict(f) for f in list(self.recent_fills)[-10:]],
        }


# ---------------------------------------------------------------------------
# Rich terminal dashboard
# ---------------------------------------------------------------------------


class RichDashboard:
    """Render the dashboard to the terminal via ``rich.Live``."""

    def __init__(self, ctx: DashboardContext, *, refresh_hz: float = 1.0):
        from rich.console import Console  # imported lazily

        self.ctx = ctx
        self.refresh_hz = refresh_hz
        self._console = Console()

    def _build_layout(self):
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        snap = self.ctx.snapshot_dict()

        # Header
        mode_text = Text()
        mode_text.append(f"blksch — mode=", style="bold")
        mode_text.append(snap["mode"], style="bold yellow" if snap["mode"] == "live" else "bold cyan")
        mode_text.append(f"   ts={snap['ts']}", style="dim")

        # PnL table
        pnl = snap["pnl"]
        pnl_t = Table(show_header=False, box=None, pad_edge=False)
        pnl_t.add_row("Realized",   _fmt_usd(pnl["realized_usd"]))
        pnl_t.add_row("Unrealized", _fmt_usd(pnl["unrealized_usd"]))
        pnl_t.add_row("Fees",       _fmt_usd(-pnl["fees_usd"]))
        pnl_t.add_row("Total",      _fmt_usd(pnl["total_usd"]))

        # Positions table
        pos_t = Table(title="Positions", expand=True)
        pos_t.add_column("token_id")
        pos_t.add_column("qty",       justify="right")
        pos_t.add_column("avg_entry", justify="right")
        pos_t.add_column("mark",      justify="right")
        pos_t.add_column("uPnL",      justify="right")
        for p in snap["positions"]:
            pos_t.add_row(
                p["token_id"][:12] + "…" if len(p["token_id"]) > 12 else p["token_id"],
                f"{p['qty']:+.2f}",
                f"{p['avg_entry']:.4f}",
                f"{p['mark']:.4f}",
                _fmt_usd(p["unrealized_pnl_usd"]),
            )

        # Quotes table
        q_t = Table(title="Quotes (latest per token)", expand=True)
        q_t.add_column("token_id")
        q_t.add_column("bid",       justify="right")
        q_t.add_column("ask",       justify="right")
        q_t.add_column("size_bid",  justify="right")
        q_t.add_column("size_ask",  justify="right")
        q_t.add_column("inv_q",     justify="right")
        for k, q in snap["quotes"].items():
            q_t.add_row(
                k[:12] + "…" if len(k) > 12 else k,
                f"{q['p_bid']:.4f}",
                f"{q['p_ask']:.4f}",
                f"{q['size_bid']:.2f}",
                f"{q['size_ask']:.2f}",
                f"{q['inventory_q']:+.2f}",
            )

        # Kill-switches
        ks_text = Text()
        for name, tripped in snap["kill_switches"].items():
            style = "bold red" if tripped else "green"
            ks_text.append(f" {name}={'TRIPPED' if tripped else 'ok'}", style=style)
        engine = snap["engine"]
        if engine is not None:
            style = "bold red" if engine.get("halted") else "green"
            ks_text.append(
                f" engine={'HALTED' if engine.get('halted') else 'ok'}",
                style=style,
            )
            if engine.get("halt_reason"):
                ks_text.append(f" ({engine['halt_reason']})", style="red dim")

        # Recent fills
        f_t = Table(title="Recent fills", expand=True)
        f_t.add_column("ts")
        f_t.add_column("token_id")
        f_t.add_column("side")
        f_t.add_column("price", justify="right")
        f_t.add_column("size",  justify="right")
        for f in snap["recent_fills"]:
            f_t.add_row(
                f["ts"][11:19],
                f["token_id"][:10] + "…" if len(f["token_id"]) > 10 else f["token_id"],
                f["side"],
                f"{f['price']:.4f}",
                f"{f['size']:.2f}",
            )

        layout = Layout()
        layout.split_column(
            Layout(Panel(mode_text, border_style="blue"), size=3),
            Layout(name="top", ratio=1),
            Layout(name="mid", ratio=1),
            Layout(Panel(ks_text, title="Kill-switches", border_style="magenta"), size=3),
            Layout(Panel(f_t, border_style="white"), ratio=1),
        )
        layout["top"].split_row(
            Layout(Panel(pnl_t, title="PnL", border_style="green")),
            Layout(Panel(pos_t, border_style="cyan")),
        )
        layout["mid"].update(Panel(q_t, border_style="yellow"))
        return layout

    async def run(self, stop_event=None) -> None:
        import asyncio as _asyncio

        from rich.live import Live

        stop = stop_event or _asyncio.Event()
        with Live(self._build_layout(), console=self._console, refresh_per_second=self.refresh_hz, screen=False) as live:
            while not stop.is_set():
                live.update(self._build_layout())
                try:
                    await _asyncio.wait_for(stop.wait(), timeout=1.0 / self.refresh_hz)
                except _asyncio.TimeoutError:
                    continue


# ---------------------------------------------------------------------------
# Flask JSON dashboard
# ---------------------------------------------------------------------------


class FlaskDashboard:
    """Minimal JSON-over-HTTP window onto the ledger. Import-optional."""

    def __init__(self, ctx: DashboardContext):
        from flask import Flask, jsonify

        self.ctx = ctx
        self.app = Flask(__name__)

        @self.app.route("/api/state")
        def _state():  # pragma: no cover — exercised via integration-style test
            return jsonify(self.ctx.snapshot_dict())

        @self.app.route("/api/pnl")
        def _pnl():  # pragma: no cover
            return jsonify(self.ctx.snapshot_dict()["pnl"])

        @self.app.route("/api/health")
        def _health():  # pragma: no cover
            return jsonify({"status": "ok", "ts": datetime.now(UTC).isoformat(),
                            "mode": self.ctx.mode})

    def run(self, host: str = "127.0.0.1", port: int = 5055) -> None:  # pragma: no cover
        log.info("FlaskDashboard listening on %s:%d", host, port)
        self.app.run(host=host, port=port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _fmt_usd(x: float) -> str:
    style = "green" if x >= 0 else "red"
    return f"[{style}]${x:+.4f}[/{style}]"


def _position_dict(p: Position) -> dict[str, Any]:
    return {
        "token_id": p.token_id,
        "qty": p.qty,
        "avg_entry": p.avg_entry,
        "mark": p.mark,
        "realized_pnl_usd": p.realized_pnl_usd,
        "unrealized_pnl_usd": p.unrealized_pnl_usd,
    }


def _quote_dict(q: Quote) -> dict[str, Any]:
    return {
        "token_id": q.token_id,
        "p_bid": q.p_bid, "p_ask": q.p_ask,
        "x_bid": q.x_bid, "x_ask": q.x_ask,
        "size_bid": q.size_bid, "size_ask": q.size_ask,
        "half_spread_x": q.half_spread_x,
        "reservation_x": q.reservation_x,
        "inventory_q": q.inventory_q,
        "ts": q.ts.isoformat(),
    }


def _surface_dict(s: SurfacePoint) -> dict[str, Any]:
    return {
        "token_id": s.token_id,
        "tau": s.tau, "m": s.m,
        "sigma_b": s.sigma_b,
        "lambda": s.lambda_,
        "s2_j": s.s2_j,
        "uncertainty": s.uncertainty,
        "ts": s.ts.isoformat(),
    }


def _engine_dict(state: PaperEngineState) -> dict[str, Any]:
    return {
        "halted": state.halted,
        "halt_reason": state.halt_reason,
        "last_book_ts": state.last_book_ts.isoformat() if state.last_book_ts else None,
        "last_trade_ts": state.last_trade_ts.isoformat() if state.last_trade_ts else None,
        "fills_count": state.fills_count,
    }


def _fill_dict(f: Fill) -> dict[str, Any]:
    return {
        "order_client_id": f.order_client_id,
        "token_id": f.token_id,
        "side": f.side.value,
        "price": f.price,
        "size": f.size,
        "fee_usd": f.fee_usd,
        "ts": f.ts.isoformat(),
    }
