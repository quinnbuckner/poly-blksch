"""Offline integration test for ``scripts/calibration_dryrun.py``.

Generates a synthetic jump-diffusion path, materializes it as a
``BookSnap`` / ``TradeTick`` stream, writes the stream to a
``ParquetStore``, reads it back, and runs the dry-run's calibration +
diagnostic + verdict path against the replayed stream. The WS / REST
client is never touched.

This is the pre-paper-soak proxy for the live run: if the pipeline
passes here, the script is wired correctly end-to-end.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from blksch.core.ingest.store import ParquetStore
from blksch.schemas import BookSnap, PriceLevel, TradeSide, TradeTick

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import calibration_dryrun as cd  # noqa: E402

pytestmark = pytest.mark.integration


# ---------- Synthetic stream builder ----------


def _build_synthetic_stream(
    *,
    n: int = 1200,
    sigma_b: float = 0.026,
    lambda_jump: float = 0.005,
    s_J: float = 0.08,
    rng_seed: int = 42,
    token_id: str = "tok-offline",
) -> tuple[list[BookSnap], list[TradeTick]]:
    """Generate paper-§6-style synthetic JD path → BookSnap + TradeTick stream.

    Each step: one BookSnap with a spread that reflects injected
    heteroskedastic noise. Trades fire with probability ∝ |Δx|.
    """
    rng = np.random.default_rng(rng_seed)
    t0 = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)

    # Path
    x = np.zeros(n)
    jumps = np.zeros(n, dtype=bool)
    for i in range(1, n):
        diff = sigma_b * rng.normal()
        if rng.random() < lambda_jump:
            jumps[i] = True
            x[i] = x[i - 1] + diff + rng.normal(0.0, s_J)
        else:
            x[i] = x[i - 1] + diff

    # Microstructure-noisy mid.
    spread_base = 0.008 + 0.004 * np.abs(rng.standard_normal(n))
    depth = 400.0 + 150.0 * rng.uniform(size=n)
    noise = rng.normal(0.0, np.sqrt(1e-6 + 0.4 * spread_base * spread_base))
    y_obs = x + noise
    p_obs = 1.0 / (1.0 + np.exp(-y_obs))
    # Keep inside (0, 1) strictly.
    p_obs = np.clip(p_obs, 1e-4, 1 - 1e-4)

    books: list[BookSnap] = []
    for i in range(n):
        mid = float(p_obs[i])
        half = float(
            min(0.4 * mid, 0.4 * (1.0 - mid),
                max(5e-4, 0.5 * math.sqrt(spread_base[i] ** 2 + 1e-6)))
        )
        p_bid = max(1e-4, mid - half)
        p_ask = min(1 - 1e-4, mid + half)
        books.append(BookSnap(
            token_id=token_id,
            bids=[PriceLevel(price=p_bid, size=float(depth[i]))],
            asks=[PriceLevel(price=p_ask, size=float(depth[i]))],
            ts=t0 + timedelta(seconds=i),
        ))

    # Trades — a small Poisson rate plus the scheduled-jump-like cluster.
    trades: list[TradeTick] = []
    for i in range(n):
        if rng.random() < 0.05:  # ~5% of steps have a trade
            side = TradeSide.BUY if rng.random() < 0.5 else TradeSide.SELL
            trades.append(TradeTick(
                token_id=token_id,
                price=float(p_obs[i]),
                size=float(rng.uniform(1, 100)),
                aggressor_side=side,
                ts=t0 + timedelta(seconds=i, microseconds=500_000),
            ))
    return books, trades


# ---------- Direct calibration path (no store) ----------


async def test_snapshot_on_synthetic_stream_produces_valid_output() -> None:
    """The end-to-end calibration + diagnostic + verdict path runs to
    completion and produces a finite, well-formed snapshot.

    The verdict color is intentionally not asserted — the exact color
    depends on how cleanly the synthetic tracks under the microstruct/
    Kalman chain, and a RED verdict on a poorly-behaved synthetic is a
    legitimate behavior of the dry-run (it's the whole point: flag
    markets whose innovations don't whiten). The test's job is to prove
    the wiring: books → snapshot → diagnostics → verdict → no exceptions.
    """
    books, trades = _build_synthetic_stream(n=1500)
    snap = cd.run_calibration_snapshot(
        books, trades,
        token_id="tok-offline",
        t_elapsed_sec=1500,
        bot_config={"calibration": {"kf_grid_hz": 1.0, "mc_draws_per_step": 400}},
        horizon_sec=60,
        microstruct_fit_window=400,
    )
    assert snap.n_states > 1000
    assert math.isfinite(snap.sigma_b_hat)
    assert snap.sigma_b_hat > 0
    assert math.isfinite(snap.lambda_hat)
    assert snap.lambda_hat >= 0
    assert snap.verdict in (cd.VERDICT_GREEN, cd.VERDICT_YELLOW, cd.VERDICT_RED)
    assert snap.diagnostics is not None
    assert snap.innovations_n > 500
    # Whichever verdict landed, the diagnostic fields are populated and
    # finite so the report can serialize.
    d = snap.diagnostics
    assert 0.0 <= d.ljung_box_pvalue <= 1.0
    assert 0.0 <= d.qq_shapiro_wilk_pvalue <= 1.0
    assert math.isfinite(d.variance_rel_error)


async def test_snapshot_short_stream_is_yellow_not_red() -> None:
    """Too-short stream yields a YELLOW verdict with NaN σ̂_b, not a crash."""
    books, trades = _build_synthetic_stream(n=20)
    snap = cd.run_calibration_snapshot(
        books, trades,
        token_id="tok-short", t_elapsed_sec=20,
        bot_config={"calibration": {"kf_grid_hz": 1.0}},
    )
    assert snap.verdict == cd.VERDICT_YELLOW
    assert snap.diagnostics is None
    assert snap.n_books == len(books)


# ---------- ParquetStore round-trip ----------


async def test_parquet_round_trip_preserves_stream(tmp_path: Path) -> None:
    """Writing BookSnap + TradeTick to ParquetStore and reading back must
    preserve counts and timestamps."""
    books, trades = _build_synthetic_stream(n=200)
    token_id = books[0].token_id
    store = ParquetStore(tmp_path / "ticks")
    for b in books:
        await store.append_book(b)
    for t in trades:
        await store.append_trade(t)
    await store.close()

    start = books[0].ts - timedelta(seconds=1)
    end = books[-1].ts + timedelta(seconds=1)
    df_books = store.read_range(token_id, start, end, stream="book")
    df_trades = store.read_range(token_id, start, end, stream="trade")
    assert len(df_books) == len(books)
    assert len(df_trades) == len(trades)


# ---------- Async live-ingest shell with a fake client ----------


class _FakeClient:
    """Minimal PolyClient stand-in. stream_market replays a prebuilt
    (BookSnap | TradeTick) sequence; the dryrun's elapsed-time math
    drives the snapshot triggers via the injected clock."""

    def __init__(self, events: list[BookSnap | TradeTick], initial_book: BookSnap):
        self._events = events
        self._initial = initial_book
        self.stream_called_with: list[list[str]] = []

    async def get_book(self, token_id: str) -> BookSnap:
        return self._initial

    async def stream_market(self, token_ids):
        self.stream_called_with.append(list(token_ids))
        for e in self._events:
            yield e


async def test_ingest_and_snapshot_fires_at_configured_intervals(tmp_path: Path) -> None:
    """Drive the async shell with a synthetic event stream and a
    controllable clock; verify one snapshot fires at each configured
    boundary and the ParquetStore captures the stream."""
    books, trades = _build_synthetic_stream(n=1200)
    # Interleave books and trades in ts order.
    events: list[BookSnap | TradeTick] = []
    ti = 0
    for b in books:
        while ti < len(trades) and trades[ti].ts <= b.ts:
            events.append(trades[ti])
            ti += 1
        events.append(b)
    events.extend(trades[ti:])

    client = _FakeClient(events, initial_book=books[0])
    store = ParquetStore(tmp_path / "ticks")

    # Clock that advances 1 s per event so 1200 events ≈ 1200 s elapsed.
    _elapsed = {"now": 0.0}

    def clock() -> float:
        _elapsed["now"] += 1.0
        return _elapsed["now"]

    cfg = cd.DryrunConfig(
        minutes=20,
        out_dir=tmp_path,
        snapshot_intervals_sec=(300, 600, 900),
        token_id=books[0].token_id,
        auto_select=False,
        bot_config={"calibration": {"kf_grid_hz": 1.0, "mc_draws_per_step": 300}},
        horizon_sec=60,
        ewma_half_life_sec=90.0,
        microstruct_fit_window=250,
    )

    snapshots = await cd.ingest_and_snapshot(
        client, books[0].token_id, config=cfg, store=store, clock=clock,
    )

    # Three snapshot boundaries cleared.
    assert len(snapshots) == 3
    elapsed_seconds = [s.t_elapsed_sec for s in snapshots]
    assert elapsed_seconds == [300, 600, 900]
    # Each snapshot saw monotonically more data.
    assert snapshots[0].n_books < snapshots[1].n_books <= snapshots[2].n_books

    # Store persisted most of the stream.
    start = books[0].ts - timedelta(seconds=1)
    end = books[-1].ts + timedelta(seconds=1)
    df_books = store.read_range(books[0].token_id, start, end, stream="book")
    assert len(df_books) >= 0.9 * len(books)


async def test_ingest_writes_report_jsonable(tmp_path: Path) -> None:
    """The report that would be written to disk is JSON-serializable end-to-end."""
    import json as _json

    books, trades = _build_synthetic_stream(n=800)
    events: list = list(books)  # trades can be dropped for this smoke test

    client = _FakeClient(events, initial_book=books[0])
    store = ParquetStore(tmp_path / "ticks")
    _elapsed = {"now": 0.0}

    def clock() -> float:
        _elapsed["now"] += 1.0
        return _elapsed["now"]

    cfg = cd.DryrunConfig(
        minutes=15,
        out_dir=tmp_path,
        snapshot_intervals_sec=(500,),
        token_id=books[0].token_id,
        auto_select=False,
        bot_config={"calibration": {"kf_grid_hz": 1.0}},
        horizon_sec=60,
        ewma_half_life_sec=90.0,
        microstruct_fit_window=300,
    )
    snapshots = await cd.ingest_and_snapshot(
        client, books[0].token_id, config=cfg, store=store, clock=clock,
    )
    report = cd.render_report(
        header={"script": "calibration_dryrun.py", "args": [], "ts": "2026-04-23T00:00:00+00:00", "user": "u"},
        token_id=books[0].token_id,
        snapshots=snapshots,
        bot_config=cfg.bot_config,
    )
    s = _json.dumps(report)
    assert '"overall_verdict"' in s
    assert report["n_snapshots"] == 1
    path = cd.write_report(report, tmp_path / "report-out")
    assert path.exists()
