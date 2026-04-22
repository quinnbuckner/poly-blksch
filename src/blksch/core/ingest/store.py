"""Append-only Parquet tick store for BookSnap and TradeTick streams.

The ingest loop (``polyclient``) pushes every ``BookSnap`` and ``TradeTick`` it
receives into this store. Downstream calibration (filter/EM/surface) reads
through :meth:`ParquetStore.read_range`.

Design notes
------------
- Files are partitioned as ``<root>/<stream>/<token_id>/<YYYY-MM-DD>/part-*.parquet``.
  Each flush writes a new part file, so "rotation at size cap" is enforced by
  flushing when the per-partition in-memory buffer estimate exceeds
  ``rotate_bytes``. Files never grow past one flush.
- All disk I/O runs via :func:`asyncio.to_thread` so the event loop — and the
  WS ingest path in particular — never blocks on fsync/parquet-encode.
- The store does **not** extend :class:`BookSnap` / :class:`TradeTick`. We only
  flatten them into a PyArrow schema at write time.
"""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blksch.schemas import BookSnap, PriceLevel, TradeSide, TradeTick

Stream = Literal["book", "trade"]

DEFAULT_ROTATE_BYTES = 128 * 1024 * 1024  # 128 MB

_LEVEL_TYPE = pa.struct([
    pa.field("price", pa.float64()),
    pa.field("size", pa.float64()),
])

_BOOK_SCHEMA = pa.schema([
    pa.field("token_id", pa.string()),
    pa.field("ts", pa.timestamp("ns", tz="UTC")),
    pa.field("seq", pa.int64(), nullable=True),
    pa.field("bids", pa.list_(_LEVEL_TYPE)),
    pa.field("asks", pa.list_(_LEVEL_TYPE)),
])

_TRADE_SCHEMA = pa.schema([
    pa.field("token_id", pa.string()),
    pa.field("ts", pa.timestamp("ns", tz="UTC")),
    pa.field("price", pa.float64()),
    pa.field("size", pa.float64()),
    pa.field("aggressor_side", pa.string()),
])

_PART_RE = re.compile(r"part-(\d{5,})\.parquet$")


# ---------------------------------------------------------------------------
# Row builders (pure)
# ---------------------------------------------------------------------------


def _levels(levels: list[PriceLevel]) -> list[dict[str, float]]:
    return [{"price": lv.price, "size": lv.size} for lv in levels]


def _book_row(snap: BookSnap) -> dict:
    return {
        "token_id": snap.token_id,
        "ts": snap.ts,
        "seq": snap.seq,
        "bids": _levels(snap.bids),
        "asks": _levels(snap.asks),
    }


def _trade_row(tick: TradeTick) -> dict:
    return {
        "token_id": tick.token_id,
        "ts": tick.ts,
        "price": tick.price,
        "size": tick.size,
        "aggressor_side": tick.aggressor_side.value,
    }


def _rows_to_table(stream: Stream, rows: list[dict]) -> pa.Table:
    schema = _BOOK_SCHEMA if stream == "book" else _TRADE_SCHEMA
    return pa.Table.from_pylist(rows, schema=schema)


def _estimate_book_bytes(snap: BookSnap) -> int:
    # Very rough: 16 bytes per level + 64 bytes overhead.
    return 64 + 16 * (len(snap.bids) + len(snap.asks))


def _estimate_trade_bytes() -> int:
    return 64


# ---------------------------------------------------------------------------
# Partition key helpers
# ---------------------------------------------------------------------------


def _partition_dir(root: Path, stream: Stream, token_id: str, d: date) -> Path:
    return root / stream / token_id / d.isoformat()


def _next_part_index(dir_: Path) -> int:
    if not dir_.exists():
        return 0
    max_idx = -1
    for entry in dir_.iterdir():
        m = _PART_RE.search(entry.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _as_utc_date(ts: datetime) -> date:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).date()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class _Buffer:
    rows: list[dict]
    approx_bytes: int


class ParquetStore:
    """Append-only Parquet store partitioned by (stream, token_id, UTC date).

    Usage::

        store = ParquetStore(Path("./data"))
        await store.append_book(snap)
        await store.append_trade(tick)
        await store.flush()  # or await store.close()

        df = store.read_range(token_id, t0, t1, stream="book")
    """

    def __init__(
        self,
        root: Path | str,
        *,
        rotate_bytes: int = DEFAULT_ROTATE_BYTES,
    ) -> None:
        if rotate_bytes <= 0:
            raise ValueError("rotate_bytes must be positive")
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._rotate_bytes = rotate_bytes
        self._buffers: dict[tuple[Stream, str, date], _Buffer] = {}
        # Plain threading.Lock — all mutations happen on asyncio.to_thread
        # workers (i.e. a real threadpool), not on the event loop itself.
        self._lock = threading.Lock()

    # -- public async API -----------------------------------------------------

    async def append_book(self, snap: BookSnap) -> None:
        await asyncio.to_thread(self._append_book_sync, snap)

    async def append_trade(self, tick: TradeTick) -> None:
        await asyncio.to_thread(self._append_trade_sync, tick)

    async def flush(self) -> None:
        await asyncio.to_thread(self._flush_all_sync)

    async def close(self) -> None:
        await self.flush()

    # -- sync writes ----------------------------------------------------------

    def _append_book_sync(self, snap: BookSnap) -> None:
        self._append_row("book", snap.token_id, snap.ts, _book_row(snap), _estimate_book_bytes(snap))

    def _append_trade_sync(self, tick: TradeTick) -> None:
        self._append_row("trade", tick.token_id, tick.ts, _trade_row(tick), _estimate_trade_bytes())

    def _append_row(
        self,
        stream: Stream,
        token_id: str,
        ts: datetime,
        row: dict,
        row_bytes: int,
    ) -> None:
        key: tuple[Stream, str, date] = (stream, token_id, _as_utc_date(ts))
        with self._lock:
            buf = self._buffers.setdefault(key, _Buffer(rows=[], approx_bytes=0))
            buf.rows.append(row)
            buf.approx_bytes += row_bytes
            if buf.approx_bytes >= self._rotate_bytes:
                self._flush_key_locked(key)

    def _flush_all_sync(self) -> None:
        with self._lock:
            for key in list(self._buffers.keys()):
                self._flush_key_locked(key)

    def _flush_key_locked(self, key: tuple[Stream, str, date]) -> None:
        """Flush one partition buffer. Caller holds the lock."""
        buf = self._buffers.pop(key, None)
        if buf is None or not buf.rows:
            return
        stream, token_id, d = key
        dir_ = _partition_dir(self._root, stream, token_id, d)
        dir_.mkdir(parents=True, exist_ok=True)
        idx = _next_part_index(dir_)
        path = dir_ / f"part-{idx:05d}.parquet"
        table = _rows_to_table(stream, buf.rows)
        pq.write_table(table, path, compression="snappy")

    # -- read -----------------------------------------------------------------

    def read_range(
        self,
        token_id: str,
        start_ts: datetime,
        end_ts: datetime,
        *,
        stream: Stream = "book",
        include_buffered: bool = True,
    ) -> pd.DataFrame:
        """Load all records for ``token_id`` with ``start_ts <= ts < end_ts``.

        If ``include_buffered`` is True (default), the in-memory buffer for the
        matching partition is included. Callers who want only durable data
        should pass ``include_buffered=False``.
        """
        if start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=UTC)
        if end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=UTC)
        if end_ts <= start_ts:
            return self._empty_frame(stream)

        start_d = _as_utc_date(start_ts)
        end_d = _as_utc_date(end_ts - pd.Timedelta(nanoseconds=1))  # end is exclusive
        token_root = self._root / stream / token_id
        if not token_root.exists() and not include_buffered:
            return self._empty_frame(stream)

        frames: list[pd.DataFrame] = []
        if token_root.exists():
            for day_dir in sorted(token_root.iterdir()):
                if not day_dir.is_dir():
                    continue
                try:
                    d = date.fromisoformat(day_dir.name)
                except ValueError:
                    continue
                if d < start_d or d > end_d:
                    continue
                for part in sorted(day_dir.glob("part-*.parquet")):
                    frames.append(pq.read_table(part).to_pandas())

        if include_buffered:
            with self._lock:
                for (bstream, btid, bdate), buf in self._buffers.items():
                    if bstream != stream or btid != token_id:
                        continue
                    if bdate < start_d or bdate > end_d:
                        continue
                    if buf.rows:
                        frames.append(_rows_to_table(stream, buf.rows).to_pandas())

        if not frames:
            return self._empty_frame(stream)
        df = pd.concat(frames, ignore_index=True)
        # Enforce ts filter (partition pruning was coarse).
        mask = (df["ts"] >= pd.Timestamp(start_ts)) & (df["ts"] < pd.Timestamp(end_ts))
        return df.loc[mask].sort_values("ts").reset_index(drop=True)

    def _empty_frame(self, stream: Stream) -> pd.DataFrame:
        schema = _BOOK_SCHEMA if stream == "book" else _TRADE_SCHEMA
        return pa.Table.from_pylist([], schema=schema).to_pandas()

    # -- test/introspection helpers ------------------------------------------

    def list_parts(self, token_id: str, *, stream: Stream = "book") -> list[Path]:
        """Return all durable part files for ``token_id`` (sorted)."""
        token_root = self._root / stream / token_id
        if not token_root.exists():
            return []
        parts: list[Path] = []
        for day_dir in sorted(token_root.iterdir()):
            if day_dir.is_dir():
                parts.extend(sorted(day_dir.glob("part-*.parquet")))
        return parts

    @property
    def root(self) -> Path:
        return self._root


__all__ = [
    "DEFAULT_ROTATE_BYTES",
    "ParquetStore",
    "Stream",
]

# Re-exported for test convenience.
_ = (TradeSide,)
