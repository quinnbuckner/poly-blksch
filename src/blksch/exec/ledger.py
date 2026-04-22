"""Authoritative ledger of orders, fills, positions, and PnL.

Backed by SQLite (WAL) so paper-engine, order-router and dashboard can all
read from the same file without coordination.

Accounting conventions
----------------------

* Positions are **signed**: positive ``qty`` means long YES shares, negative
  means net short YES (achieved on Polymarket by selling to a buyer at a
  higher price than our entry, or via minting/splitting).
* Weighted-average cost basis. When a fill reduces the position past zero,
  the PnL on the closed portion is realized and the leftover quantity takes
  the fill price as the new basis.
* Fees are subtracted from realized PnL at fill time. No queue rebates are
  modeled (paper's §4.2 assumes zero maker rebate on Polymarket — revisit in
  Stage 2 if the fee schedule changes).
* Unrealized PnL is computed against a separately recorded ``mark`` per
  token_id — typically the last BookSnap mid from Track A.

Everything public returns ``blksch.schemas`` models so callers are decoupled
from SQLite row shapes.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from blksch.schemas import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_id    TEXT PRIMARY KEY,
    venue_id     TEXT,
    token_id     TEXT NOT NULL,
    side         TEXT NOT NULL,
    price        REAL NOT NULL,
    size         REAL NOT NULL,
    status       TEXT NOT NULL,
    created_ts   TEXT NOT NULL,
    updated_ts   TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_token   ON orders(token_id);
CREATE INDEX IF NOT EXISTS idx_orders_venue   ON orders(venue_id);
CREATE INDEX IF NOT EXISTS idx_orders_status  ON orders(status);

CREATE TABLE IF NOT EXISTS fills (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    order_client_id   TEXT NOT NULL,
    order_venue_id    TEXT,
    token_id          TEXT NOT NULL,
    side              TEXT NOT NULL,
    price             REAL NOT NULL,
    size              REAL NOT NULL,
    fee_usd           REAL NOT NULL DEFAULT 0,
    ts                TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fills_token ON fills(token_id);
CREATE INDEX IF NOT EXISTS idx_fills_ts    ON fills(ts);

CREATE TABLE IF NOT EXISTS positions (
    token_id          TEXT PRIMARY KEY,
    qty               REAL NOT NULL DEFAULT 0,
    avg_entry         REAL NOT NULL DEFAULT 0,
    realized_pnl_usd  REAL NOT NULL DEFAULT 0,
    updated_ts        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS marks (
    token_id   TEXT PRIMARY KEY,
    mark       REAL NOT NULL,
    ts         TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class PnLSnapshot:
    realized_usd: float
    unrealized_usd: float
    fees_usd: float

    @property
    def total_usd(self) -> float:
        return self.realized_usd + self.unrealized_usd


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _parse_iso(s: str) -> datetime:
    # sqlite-stored strings are always written via _iso above
    return datetime.fromisoformat(s)


class Ledger:
    """Thread-safe SQLite ledger. Intended for a single bot process.

    Use :meth:`in_memory` for tests; :meth:`open` for persistent files.
    """

    def __init__(self, db_path: str | Path | None):
        self._lock = threading.RLock()
        self._path = ":memory:" if db_path is None else str(db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL") if self._path != ":memory:" else None
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    # -- factory helpers ---------------------------------------------------

    @classmethod
    def in_memory(cls) -> "Ledger":
        return cls(None)

    @classmethod
    def open(cls, path: str | Path) -> "Ledger":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return cls(path)

    # -- orders ------------------------------------------------------------

    def record_order(self, order: Order) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO orders "
                "(client_id, venue_id, token_id, side, price, size, status, created_ts, updated_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    order.client_id,
                    order.venue_id,
                    order.token_id,
                    order.side.value,
                    order.price,
                    order.size,
                    order.status.value,
                    _iso(order.created_ts),
                    _iso(order.updated_ts) if order.updated_ts else None,
                ),
            )

    def update_order_status(
        self,
        client_id: str,
        status: OrderStatus,
        *,
        venue_id: str | None = None,
        updated_ts: datetime | None = None,
    ) -> None:
        ts = updated_ts or datetime.now(UTC)
        with self._lock:
            cur = self._conn.execute(
                "UPDATE orders SET status=?, "
                "venue_id=COALESCE(?, venue_id), "
                "updated_ts=? WHERE client_id=?",
                (status.value, venue_id, _iso(ts), client_id),
            )
            if cur.rowcount == 0:
                log.warning("update_order_status: unknown client_id %s", client_id)

    def get_order(self, client_id: str) -> Order | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT client_id, venue_id, token_id, side, price, size, status, created_ts, updated_ts "
                "FROM orders WHERE client_id=?",
                (client_id,),
            ).fetchone()
        if not row:
            return None
        return Order(
            client_id=row[0],
            venue_id=row[1],
            token_id=row[2],
            side=OrderSide(row[3]),
            price=row[4],
            size=row[5],
            status=OrderStatus(row[6]),
            created_ts=_parse_iso(row[7]),
            updated_ts=_parse_iso(row[8]) if row[8] else None,
        )

    def open_orders(self, token_id: str | None = None) -> list[Order]:
        open_statuses = (OrderStatus.PENDING.value, OrderStatus.OPEN.value, OrderStatus.PARTIALLY_FILLED.value)
        sql = (
            "SELECT client_id, venue_id, token_id, side, price, size, status, created_ts, updated_ts "
            "FROM orders WHERE status IN (?, ?, ?)"
        )
        args: list = list(open_statuses)
        if token_id is not None:
            sql += " AND token_id=?"
            args.append(token_id)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [
            Order(
                client_id=r[0],
                venue_id=r[1],
                token_id=r[2],
                side=OrderSide(r[3]),
                price=r[4],
                size=r[5],
                status=OrderStatus(r[6]),
                created_ts=_parse_iso(r[7]),
                updated_ts=_parse_iso(r[8]) if r[8] else None,
            )
            for r in rows
        ]

    # -- fills & position accounting --------------------------------------

    def apply_fill(self, fill: Fill) -> Position:
        """Record a fill and update the corresponding position / realized PnL.

        Weighted-average cost basis with sign flips handled. Returns the new
        :class:`Position` snapshot (with ``mark`` defaulted to the fill price
        if no mark has been recorded yet).
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO fills "
                "(order_client_id, order_venue_id, token_id, side, price, size, fee_usd, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fill.order_client_id,
                    fill.order_venue_id,
                    fill.token_id,
                    fill.side.value,
                    fill.price,
                    fill.size,
                    fill.fee_usd,
                    _iso(fill.ts),
                ),
            )

            row = self._conn.execute(
                "SELECT qty, avg_entry, realized_pnl_usd FROM positions WHERE token_id=?",
                (fill.token_id,),
            ).fetchone()
            qty = row[0] if row else 0.0
            avg = row[1] if row else 0.0
            realized = row[2] if row else 0.0

            delta = fill.size if fill.side is OrderSide.BUY else -fill.size
            new_qty, new_avg, pnl_delta = _apply_fill_to_position(
                qty, avg, delta, fill.price
            )
            # fees always reduce realized PnL
            realized_new = realized + pnl_delta - fill.fee_usd

            self._conn.execute(
                "INSERT INTO positions (token_id, qty, avg_entry, realized_pnl_usd, updated_ts) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(token_id) DO UPDATE SET "
                "  qty=excluded.qty, avg_entry=excluded.avg_entry, "
                "  realized_pnl_usd=excluded.realized_pnl_usd, updated_ts=excluded.updated_ts",
                (fill.token_id, new_qty, new_avg, realized_new, _iso(fill.ts)),
            )

            mark_row = self._conn.execute(
                "SELECT mark FROM marks WHERE token_id=?", (fill.token_id,)
            ).fetchone()
            mark = mark_row[0] if mark_row else fill.price

        return Position(
            token_id=fill.token_id,
            qty=new_qty,
            avg_entry=max(0.0, min(1.0, new_avg)) if new_qty != 0 else 0.0,
            mark=mark,
            realized_pnl_usd=realized_new,
        )

    def fills(self, token_id: str | None = None) -> list[Fill]:
        sql = (
            "SELECT order_client_id, order_venue_id, token_id, side, price, size, fee_usd, ts "
            "FROM fills"
        )
        args: list = []
        if token_id is not None:
            sql += " WHERE token_id=?"
            args.append(token_id)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [
            Fill(
                order_client_id=r[0],
                order_venue_id=r[1],
                token_id=r[2],
                side=OrderSide(r[3]),
                price=r[4],
                size=r[5],
                fee_usd=r[6],
                ts=_parse_iso(r[7]),
            )
            for r in rows
        ]

    # -- marks & position reads -------------------------------------------

    def update_mark(self, token_id: str, mark: float, ts: datetime | None = None) -> None:
        ts = ts or datetime.now(UTC)
        if not (0.0 <= mark <= 1.0):
            raise ValueError(f"mark must be in [0,1], got {mark}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO marks (token_id, mark, ts) VALUES (?, ?, ?) "
                "ON CONFLICT(token_id) DO UPDATE SET mark=excluded.mark, ts=excluded.ts",
                (token_id, mark, _iso(ts)),
            )

    def get_position(self, token_id: str) -> Position:
        with self._lock:
            pos_row = self._conn.execute(
                "SELECT qty, avg_entry, realized_pnl_usd FROM positions WHERE token_id=?",
                (token_id,),
            ).fetchone()
            mark_row = self._conn.execute(
                "SELECT mark FROM marks WHERE token_id=?", (token_id,)
            ).fetchone()

        if not pos_row:
            return Position(
                token_id=token_id, qty=0.0, avg_entry=0.0,
                mark=mark_row[0] if mark_row else 0.5, realized_pnl_usd=0.0,
            )
        qty, avg, realized = pos_row
        mark = mark_row[0] if mark_row else (avg if qty != 0 else 0.5)
        return Position(
            token_id=token_id,
            qty=qty,
            avg_entry=max(0.0, min(1.0, avg)) if qty != 0 else 0.0,
            mark=mark,
            realized_pnl_usd=realized,
        )

    def all_positions(self) -> list[Position]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT token_id FROM positions WHERE qty != 0"
            ).fetchall()
        return [self.get_position(r[0]) for r in rows]

    # -- PnL ---------------------------------------------------------------

    def pnl(self) -> PnLSnapshot:
        with self._lock:
            realized = self._conn.execute(
                "SELECT COALESCE(SUM(realized_pnl_usd), 0) FROM positions"
            ).fetchone()[0]
            fees = self._conn.execute(
                "SELECT COALESCE(SUM(fee_usd), 0) FROM fills"
            ).fetchone()[0]
        unrealized = sum(p.unrealized_pnl_usd for p in self.all_positions())
        return PnLSnapshot(
            realized_usd=float(realized),
            unrealized_usd=float(unrealized),
            fees_usd=float(fees),
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Pure accounting core — testable without sqlite
# ---------------------------------------------------------------------------


def _apply_fill_to_position(
    qty: float, avg: float, delta: float, price: float
) -> tuple[float, float, float]:
    """Apply a signed fill (delta) at ``price`` to a position ``(qty, avg)``.

    Returns ``(new_qty, new_avg, realized_pnl_delta)``. Fees are handled by
    the caller.

    Sign conventions:
    * ``qty > 0`` = long; ``qty < 0`` = short.
    * ``delta > 0`` = a BUY; ``delta < 0`` = a SELL.
    """
    if qty == 0:
        return delta, price, 0.0
    if (qty > 0 and delta > 0) or (qty < 0 and delta < 0):
        # same direction — merge weighted average
        new_qty = qty + delta
        new_avg = (abs(qty) * avg + abs(delta) * price) / abs(new_qty)
        return new_qty, new_avg, 0.0
    # opposite direction — closing portion
    close_size = min(abs(qty), abs(delta))
    sign_qty = 1.0 if qty > 0 else -1.0
    pnl = close_size * (price - avg) * sign_qty
    new_qty = qty + delta
    if abs(new_qty) < 1e-12:
        return 0.0, 0.0, pnl
    if (new_qty > 0 and qty > 0) or (new_qty < 0 and qty < 0):
        # still on the same side; avg basis unchanged for remainder
        return new_qty, avg, pnl
    # flipped — remainder takes the fill price as its new basis
    return new_qty, price, pnl


def reconcile(fills: Iterable[Fill], mark: float) -> PnLSnapshot:
    """Replay a fill sequence from scratch. Used by tests to cross-check the
    Ledger's SQL-backed accounting against a pure-python computation."""
    qty = 0.0
    avg = 0.0
    realized = 0.0
    fees = 0.0
    for f in fills:
        delta = f.size if f.side is OrderSide.BUY else -f.size
        qty, avg, pnl_delta = _apply_fill_to_position(qty, avg, delta, f.price)
        realized += pnl_delta - f.fee_usd
        fees += f.fee_usd
    unrealized = qty * (mark - avg) if qty != 0 else 0.0
    return PnLSnapshot(realized_usd=realized, unrealized_usd=unrealized, fees_usd=fees)
