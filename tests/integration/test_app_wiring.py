"""End-to-end wiring test for ``src/blksch/app.py``.

Stubs the PolyClient with a scripted BookSnap/TradeTick stream and drives
``app.run()`` in ``--mode=paper`` against an in-memory Ledger. No live
network. Three scenarios:

  1. First ``Quote`` hits the paper-engine book (i.e. a resting order lands
     in the Ledger) within a short window of the first scripted BookSnap.
  2. A scripted crossing BookSnap triggers a Fill and the Ledger's position
     on that token becomes non-zero.
  3. An externally-injected ``asyncio.Event`` (simulating SIGINT) makes
     ``app.run()`` return cleanly — no dangling tasks, no open client.

Live smoke against Polymarket belongs in ``tests/live_ro/``; nothing here
hits the network.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from blksch.app import RunArgs, run
from blksch.exec.ledger import Ledger
from blksch.schemas import BookSnap, PriceLevel, TradeSide, TradeTick


REPO_ROOT = Path(__file__).resolve().parents[2]
BOT_YAML = REPO_ROOT / "config" / "bot.yaml"
MARKETS_YAML = REPO_ROOT / "config" / "markets.yaml"
TOKEN = "0xmocktoken01"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _snap(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    *,
    ts: datetime,
    token_id: str = TOKEN,
) -> BookSnap:
    return BookSnap(
        token_id=token_id,
        bids=[PriceLevel(price=p, size=s) for p, s in bids],
        asks=[PriceLevel(price=p, size=s) for p, s in asks],
        ts=ts,
    )


@dataclass
class ScriptedEvent:
    """(delay before emitting, event)."""

    delay_sec: float
    event: BookSnap | TradeTick


class MockPolyClient:
    """Stand-in for ``core.ingest.polyclient.PolyClient`` that emits a
    scripted sequence with real-time spacing between yields.

    Only the subset of the client interface that ``app.run()`` touches when
    `--tokens` is set (skipping the screener) is implemented — namely
    ``stream_market`` and ``close``. ``start`` is never called because the
    caller passes in an already-constructed client.
    """

    def __init__(self, events: Iterable[ScriptedEvent], *, start_delay: float = 0.0) -> None:
        self._events = list(events)
        self._start_delay = start_delay
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def start(self) -> None:  # never called when injected — symmetry only
        pass

    async def list_markets(self, **kw: Any) -> list[dict[str, Any]]:
        return []

    async def get_book(self, token_id: str) -> BookSnap:  # pragma: no cover — screener path
        raise NotImplementedError("MockPolyClient.get_book is not wired")

    def stream_market(
        self, token_ids: list[str] | tuple[str, ...], **kw: Any,
    ) -> AsyncIterator[BookSnap | TradeTick]:
        return self._gen(set(token_ids))

    async def _gen(
        self, token_ids: set[str],
    ) -> AsyncIterator[BookSnap | TradeTick]:
        if self._start_delay:
            await asyncio.sleep(self._start_delay)
        for scripted in self._events:
            if scripted.delay_sec > 0:
                await asyncio.sleep(scripted.delay_sec)
            if scripted.event.token_id not in token_ids:
                continue
            yield scripted.event
        # After the scripted sequence, park on a long sleep so the stream
        # doesn't close itself out (app.run is responsible for cancelling).
        try:
            await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            raise


def _run_args(
    *,
    max_runtime_sec: float | None = None,
    refresh_ms_override: int | None = None,
    tokens: list[str] | None = None,
) -> RunArgs:
    return RunArgs(
        mode="paper",
        config_path=BOT_YAML,
        markets_path=MARKETS_YAML,
        log_level="WARNING",
        tokens=tokens if tokens is not None else [TOKEN],
        rich_dashboard="off",  # keep pytest's stdout quiet
        max_runtime_sec=max_runtime_sec,
    )


def _quiet_balanced_stream(
    *, tick_count: int = 6, step_sec: float = 0.6,
) -> list[ScriptedEvent]:
    """A book hovering tightly around p ≈ 0.5. Safe to quote against —
    neither side crosses the algorithm's bid/ask.

    Real-time spacing (0.6 s/tick) guarantees the canonical_mid's 1 Hz grid
    closes at least twice before the test assertions run.
    """
    t0 = datetime.now(UTC)
    events: list[ScriptedEvent] = []
    for i in range(tick_count):
        ts = t0 + timedelta(seconds=i * step_sec)
        delay = step_sec if i > 0 else 0.0
        events.append(ScriptedEvent(
            delay,
            _snap([(0.49, 200.0)], [(0.51, 200.0)], ts=ts),
        ))
    return events


# ---------------------------------------------------------------------------
# (1) First quote lands in the ledger's order book
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_quote_lands_in_ledger_within_3s(caplog: pytest.LogCaptureFixture) -> None:
    """After the first few scripted BookSnaps (with the canonical_mid grid
    closing at 1 Hz), the refresh loop should emit a Quote and the
    OrderRouter should place resting orders into the Ledger.
    """
    caplog.set_level(logging.WARNING, logger="blksch")
    events = _quiet_balanced_stream(tick_count=8, step_sec=0.5)
    mock = MockPolyClient(events)
    ledger = Ledger.in_memory()
    stop_event = asyncio.Event()

    task = asyncio.create_task(run(
        _run_args(max_runtime_sec=5.0),
        client=mock, ledger=ledger, stop_event=stop_event,
    ))

    try:
        # Poll up to ~3 s for the first resting order to hit the ledger.
        order_seen = False
        for _ in range(30):
            await asyncio.sleep(0.1)
            if ledger.open_orders():
                order_seen = True
                break
        assert order_seen, "no resting orders after ~3 s of scripted book feed"
    finally:
        stop_event.set()
        await asyncio.wait_for(task, timeout=4.0)


# ---------------------------------------------------------------------------
# (2) Crossing BookSnap → Fill → Position updates the Ledger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scripted_crossing_produces_fill_and_position() -> None:
    """After the algorithm places a bid near 0.49–0.50, a BookSnap whose
    best ask dips below the bid should trigger a PaperEngine fill and leave
    a non-zero position on the Ledger.

    We keep the book inside the canonical_mid outlier window by taking the
    ask down only 2 ticks (0.47), with plenty of depth. The paper engine's
    queue-haircut means we won't fill the full posted size — partial is fine
    for the assertion.
    """
    t0 = datetime.now(UTC)
    warmup = [
        ScriptedEvent(0.0, _snap([(0.49, 200.0)], [(0.51, 200.0)], ts=t0)),
        ScriptedEvent(0.6, _snap([(0.49, 200.0)], [(0.51, 200.0)],
                                 ts=t0 + timedelta(seconds=0.6))),
        ScriptedEvent(0.6, _snap([(0.49, 200.0)], [(0.51, 200.0)],
                                 ts=t0 + timedelta(seconds=1.2))),
        ScriptedEvent(0.6, _snap([(0.49, 200.0)], [(0.51, 200.0)],
                                 ts=t0 + timedelta(seconds=1.8))),
        ScriptedEvent(0.6, _snap([(0.49, 200.0)], [(0.51, 200.0)],
                                 ts=t0 + timedelta(seconds=2.4))),
    ]
    # Give the refresh loop at least one cycle after warm-up to post quotes,
    # then cross the bid from the ask side with plenty of depth.
    crossing = [
        ScriptedEvent(0.8, _snap([(0.45, 200.0)], [(0.47, 200.0)],
                                 ts=t0 + timedelta(seconds=3.2))),
        ScriptedEvent(0.3, _snap([(0.46, 200.0)], [(0.48, 200.0)],
                                 ts=t0 + timedelta(seconds=3.5))),
    ]

    mock = MockPolyClient(warmup + crossing)
    ledger = Ledger.in_memory()
    stop_event = asyncio.Event()

    task = asyncio.create_task(run(
        _run_args(max_runtime_sec=6.0),
        client=mock, ledger=ledger, stop_event=stop_event,
    ))

    try:
        # Wait up to ~5 s for a non-zero position to appear.
        got_position = False
        for _ in range(50):
            await asyncio.sleep(0.1)
            pos = ledger.get_position(TOKEN)
            if pos is not None and abs(pos.qty) > 0:
                got_position = True
                break
        assert got_position, (
            f"expected non-zero position on {TOKEN} after crossing book; "
            f"ledger state: {ledger.pnl()}"
        )
    finally:
        stop_event.set()
        await asyncio.wait_for(task, timeout=5.0)


# ---------------------------------------------------------------------------
# (3) Graceful shutdown on externally-injected stop_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graceful_shutdown_on_stop_event() -> None:
    """Setting the injected asyncio.Event simulates SIGINT; app.run() must
    return within a few seconds and leave the injected Ledger usable."""
    events = _quiet_balanced_stream(tick_count=4, step_sec=0.5)
    mock = MockPolyClient(events)
    ledger = Ledger.in_memory()
    stop_event = asyncio.Event()

    task = asyncio.create_task(run(
        _run_args(max_runtime_sec=15.0),  # generous — we'll signal stop well before this
        client=mock, ledger=ledger, stop_event=stop_event,
    ))

    # Let the graph spin up and settle.
    await asyncio.sleep(1.0)
    assert not task.done()

    # Simulate SIGINT.
    stop_event.set()

    # app.run must exit within a reasonable window (stop_event → refresh
    # drain → task cancellation → router cancel_all → client.close).
    await asyncio.wait_for(task, timeout=5.0)

    assert task.done()
    assert task.exception() is None
    # PaperEngine / Ledger should still be readable after clean shutdown.
    _ = ledger.pnl()
    assert mock.closed is False  # test-owned client; app.run only closes when it created it


# ---------------------------------------------------------------------------
# (4) argparse sanity — live mode without --live-ack is refused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_mode_requires_live_ack(tmp_path: Path) -> None:
    args = RunArgs(
        mode="live",
        config_path=BOT_YAML,
        markets_path=MARKETS_YAML,
        log_level="WARNING",
        tokens=[TOKEN],
        rich_dashboard="off",
        live_ack=False,
    )
    with pytest.raises(SystemExit):
        await run(args, client=MockPolyClient([]), ledger=Ledger.in_memory())


# ---------------------------------------------------------------------------
# (5) Ownership contract — externally-provided ledger survives cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_ledger_is_not_closed_by_cleanup() -> None:
    """Regression for the early-exit cleanup path: when a test (or any
    outer caller) passes its own ``Ledger``, ``run()`` must NOT close it
    on any exit path — clean shutdown, screener failure, or empty-tokens
    early return. The ledger must stay readable afterwards.

    We trip the "empty tokens" early-exit path by passing an empty
    ``tokens`` list.  Any other exit path exercises the same
    ``_cleanup(client if owned_client else None, ledger if owned_ledger
    else None)`` contract.
    """
    ledger = Ledger.in_memory()

    args = RunArgs(
        mode="paper",
        config_path=BOT_YAML,
        markets_path=MARKETS_YAML,
        log_level="WARNING",
        tokens=[],  # empty — triggers the early-exit cleanup branch
        rich_dashboard="off",
    )

    await run(args, client=MockPolyClient([]), ledger=ledger, stop_event=asyncio.Event())

    # Ledger must still be usable — if app.run wrongly closed it we'd get
    # sqlite3.ProgrammingError("Cannot operate on a closed database.").
    snap = ledger.pnl()
    assert snap.realized_usd == 0.0
    assert snap.unrealized_usd == 0.0


@pytest.mark.asyncio
async def test_screener_failure_does_not_close_injected_ledger() -> None:
    """Bug #1 regression: screener failure used to leak owned resources or
    close the injected ledger. The fix routes through `_cleanup` with
    None for non-owned resources; the injected ledger must be intact and
    the exception must propagate to the caller so the operator sees it.
    """

    class RaisingClient(MockPolyClient):
        async def list_markets(self, **kw):  # type: ignore[override]
            raise RuntimeError("simulated screener failure")

    ledger = Ledger.in_memory()
    args = RunArgs(
        mode="paper",
        config_path=BOT_YAML,
        markets_path=MARKETS_YAML,
        log_level="WARNING",
        tokens=None,  # force screener path
        rich_dashboard="off",
    )

    with pytest.raises(RuntimeError, match="simulated screener failure"):
        await run(args, client=RaisingClient([]), ledger=ledger, stop_event=asyncio.Event())

    # Injected ledger must survive the failure path.
    snap = ledger.pnl()
    assert snap.realized_usd == 0.0
    assert snap.unrealized_usd == 0.0


# ---------------------------------------------------------------------------
# (6) Config error messages include the YAML path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_config_key_reports_dotted_path(tmp_path: Path) -> None:
    """Bug #3 regression: missing nested keys in bot.yaml used to raise bare
    ``KeyError`` with no location context. Each access now goes through
    ``_require()`` and raises a ``RuntimeError`` that names the full
    dotted path."""
    bad_yaml = tmp_path / "bot.yaml"
    bad_yaml.write_text(
        # Valid top-level keys + malformed quoting subtree (missing `gamma`).
        "quoting:\n  k: 1.5\n"
        "boundary:\n  eps: 1.0e-5\n  delta_p_floor_ticks: 1\n"
        "loop:\n  refresh_ms: 250\n"
        "calibration: {}\n"
        "limits: {}\n"
    )
    args = RunArgs(
        mode="paper",
        config_path=bad_yaml,
        markets_path=MARKETS_YAML,
        log_level="WARNING",
        tokens=[TOKEN],
        rich_dashboard="off",
    )
    with pytest.raises(RuntimeError, match=r"bot\.quoting\.gamma"):
        await run(args, client=MockPolyClient([]), ledger=Ledger.in_memory())
