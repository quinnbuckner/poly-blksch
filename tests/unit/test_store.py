"""Unit tests for ``core/ingest/store.ParquetStore``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from blksch.core.ingest.store import ParquetStore
from blksch.schemas import BookSnap, PriceLevel, TradeSide, TradeTick


# ---------- Round-trip ----------


async def test_book_write_read_roundtrip(tmp_path: Path, book_mid_50: BookSnap) -> None:
    store = ParquetStore(tmp_path)
    await store.append_book(book_mid_50)
    await store.flush()

    df = store.read_range(
        book_mid_50.token_id,
        book_mid_50.ts - timedelta(seconds=1),
        book_mid_50.ts + timedelta(seconds=1),
        stream="book",
    )
    assert len(df) == 1
    row = df.iloc[0]
    assert row["token_id"] == book_mid_50.token_id
    # ts round-trip preserves UTC.
    assert pd.Timestamp(row["ts"]).to_pydatetime() == book_mid_50.ts
    # Nested levels survive.
    bids = row["bids"]
    assert bids[0]["price"] == pytest.approx(book_mid_50.bids[0].price)
    assert bids[0]["size"] == pytest.approx(book_mid_50.bids[0].size)


async def test_trade_write_read_roundtrip(tmp_path: Path, buy_trade: TradeTick) -> None:
    store = ParquetStore(tmp_path)
    await store.append_trade(buy_trade)
    await store.close()

    df = store.read_range(
        buy_trade.token_id,
        buy_trade.ts - timedelta(seconds=1),
        buy_trade.ts + timedelta(seconds=1),
        stream="trade",
    )
    assert len(df) == 1
    assert df.iloc[0]["aggressor_side"] == TradeSide.BUY.value
    assert df.iloc[0]["price"] == pytest.approx(buy_trade.price)


# ---------- Partitioning ----------


async def test_partition_by_date(tmp_path: Path, token_id: str) -> None:
    store = ParquetStore(tmp_path)
    day1 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    day2 = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
    for ts in (day1, day2):
        await store.append_book(
            BookSnap(
                token_id=token_id,
                bids=[PriceLevel(price=0.49, size=10)],
                asks=[PriceLevel(price=0.51, size=10)],
                ts=ts,
            )
        )
    await store.flush()

    token_root = tmp_path / "book" / token_id
    date_dirs = sorted(p.name for p in token_root.iterdir() if p.is_dir())
    assert date_dirs == ["2026-04-22", "2026-04-23"]


async def test_partition_by_token(tmp_path: Path, now: datetime) -> None:
    store = ParquetStore(tmp_path)
    for tid in ("tok-A", "tok-B"):
        await store.append_book(
            BookSnap(
                token_id=tid,
                bids=[PriceLevel(price=0.49, size=10)],
                asks=[PriceLevel(price=0.51, size=10)],
                ts=now,
            )
        )
    await store.flush()

    book_root = tmp_path / "book"
    token_dirs = sorted(p.name for p in book_root.iterdir() if p.is_dir())
    assert token_dirs == ["tok-A", "tok-B"]


# ---------- Rotation ----------


async def test_rotation_at_size_cap(tmp_path: Path, token_id: str, now: datetime) -> None:
    """Exceed a tiny rotate_bytes cap; expect multiple part files."""
    # Each book row estimates ~96 bytes (2 bids + 2 asks + 64 base). Set the
    # cap to 100 bytes so every append that isn't the very first forces a
    # flush.
    store = ParquetStore(tmp_path, rotate_bytes=100)
    base = BookSnap(
        token_id=token_id,
        bids=[PriceLevel(price=0.49, size=10), PriceLevel(price=0.48, size=20)],
        asks=[PriceLevel(price=0.51, size=10), PriceLevel(price=0.52, size=20)],
        ts=now,
    )
    for i in range(5):
        await store.append_book(base.model_copy(update={"ts": now + timedelta(milliseconds=i)}))
    await store.flush()

    parts = store.list_parts(token_id, stream="book")
    # At least two parts because we exceeded the cap repeatedly; the final
    # flush is the tail partition.
    assert len(parts) >= 2, f"expected rotation to produce >=2 parts, got {parts}"


async def test_rotation_requires_positive_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ParquetStore(tmp_path, rotate_bytes=0)


# ---------- Empty-range read ----------


async def test_empty_range_returns_empty_frame(tmp_path: Path, token_id: str) -> None:
    store = ParquetStore(tmp_path)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    df = store.read_range(token_id, t0, t0 + timedelta(seconds=1), stream="book")
    assert df.empty
    # Columns should still be present so downstream code can reason about shape.
    assert {"token_id", "ts", "bids", "asks"}.issubset(df.columns)


async def test_reversed_range_returns_empty(tmp_path: Path, token_id: str) -> None:
    store = ParquetStore(tmp_path)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    df = store.read_range(token_id, t0 + timedelta(seconds=1), t0, stream="trade")
    assert df.empty


async def test_read_excludes_outside_range(tmp_path: Path, token_id: str) -> None:
    store = ParquetStore(tmp_path)
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    for i in range(3):
        await store.append_book(
            BookSnap(
                token_id=token_id,
                bids=[PriceLevel(price=0.49, size=10)],
                asks=[PriceLevel(price=0.51, size=10)],
                ts=t0 + timedelta(seconds=i),
            )
        )
    await store.flush()

    # Only the middle record should survive [t0+1, t0+2).
    df = store.read_range(token_id, t0 + timedelta(seconds=1), t0 + timedelta(seconds=2), stream="book")
    assert len(df) == 1
    assert pd.Timestamp(df.iloc[0]["ts"]).to_pydatetime() == t0 + timedelta(seconds=1)


# ---------- Buffered reads ----------


async def test_read_range_includes_buffered_by_default(tmp_path: Path, book_mid_50: BookSnap) -> None:
    store = ParquetStore(tmp_path)
    await store.append_book(book_mid_50)
    # Not flushed — nothing on disk yet.
    assert store.list_parts(book_mid_50.token_id, stream="book") == []

    df = store.read_range(
        book_mid_50.token_id,
        book_mid_50.ts - timedelta(seconds=1),
        book_mid_50.ts + timedelta(seconds=1),
        stream="book",
    )
    assert len(df) == 1


async def test_read_range_can_exclude_buffered(tmp_path: Path, book_mid_50: BookSnap) -> None:
    store = ParquetStore(tmp_path)
    await store.append_book(book_mid_50)
    df = store.read_range(
        book_mid_50.token_id,
        book_mid_50.ts - timedelta(seconds=1),
        book_mid_50.ts + timedelta(seconds=1),
        stream="book",
        include_buffered=False,
    )
    assert df.empty


# ---------- Interleaved append + flush ----------


async def test_multiple_flushes_produce_multiple_parts(
    tmp_path: Path, token_id: str, now: datetime
) -> None:
    store = ParquetStore(tmp_path)
    for i in range(3):
        await store.append_book(
            BookSnap(
                token_id=token_id,
                bids=[PriceLevel(price=0.49, size=10)],
                asks=[PriceLevel(price=0.51, size=10)],
                ts=now + timedelta(milliseconds=i),
            )
        )
        await store.flush()
    parts = store.list_parts(token_id, stream="book")
    assert len(parts) == 3
    # All parts read back.
    df = store.read_range(token_id, now - timedelta(seconds=1), now + timedelta(seconds=1), stream="book")
    assert len(df) == 3
