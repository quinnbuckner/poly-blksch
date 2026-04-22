"""Dashboard context: surface publication, staleness computation,
Rich-layout rendering with and without Track A data.

The dashboard must render safely before Track A publishes — a missing
upstream is an expected startup state, not an error.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from blksch.exec.dashboard import (
    DashboardContext,
    RichDashboard,
    StalenessThresholds,
    _fmt_age,
)
from blksch.exec.ledger import Ledger
from blksch.schemas import (
    BookSnap,
    PriceLevel,
    Quote,
    SurfacePoint,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _snap(token="tok", ts=None) -> BookSnap:
    return BookSnap(
        token_id=token,
        bids=[PriceLevel(price=0.48, size=100)],
        asks=[PriceLevel(price=0.52, size=100)],
        ts=ts or _now(),
    )


def _surface(token="tok", ts=None) -> SurfacePoint:
    return SurfacePoint(
        token_id=token,
        tau=60.0, m=0.0, sigma_b=0.35,
        **{"lambda": 0.12}, s2_j=0.04,
        uncertainty=0.02, ts=ts or _now(),
    )


def _quote(token="tok", ts=None) -> Quote:
    return Quote(
        token_id=token,
        p_bid=0.48, p_ask=0.52,
        x_bid=-0.08, x_ask=0.08,
        size_bid=10, size_ask=10,
        half_spread_x=0.08, reservation_x=0.0,
        inventory_q=0.0, ts=ts or _now(),
    )


@pytest.fixture()
def ctx() -> DashboardContext:
    return DashboardContext(ledger=Ledger.in_memory())


# ---------------------------------------------------------------------------
# Publication hooks
# ---------------------------------------------------------------------------


def test_on_book_records_timestamp_per_token(ctx: DashboardContext):
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    ctx.on_book(_snap(token="a", ts=t0))
    ctx.on_book(_snap(token="b", ts=t0 + timedelta(seconds=1)))
    assert ctx.last_book_ts == {"a": t0, "b": t0 + timedelta(seconds=1)}
    # Dashboard does NOT retain the full book — memory stays bounded.
    assert "bids" not in ctx.__dict__


def test_on_surface_stores_latest_per_token(ctx: DashboardContext):
    ctx.on_surface(_surface(token="a"))
    s2 = _surface(token="a")
    ctx.on_surface(s2)
    assert ctx.surface["a"] is s2


# ---------------------------------------------------------------------------
# Staleness computation
# ---------------------------------------------------------------------------


def test_staleness_empty_when_nothing_published(ctx: DashboardContext):
    assert ctx.staleness() == {}


def test_staleness_reports_none_for_missing_publisher(ctx: DashboardContext):
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    ctx.on_book(_snap(token="a", ts=t0))
    # no surface for "a"
    stale = ctx.staleness(now=t0 + timedelta(seconds=7))
    assert stale["a"]["book_age_sec"] == pytest.approx(7.0)
    assert stale["a"]["surface_age_sec"] is None


def test_staleness_flags_asymmetric_quote_without_surface(ctx: DashboardContext):
    """If the quoting engine is live but calibration isn't, the token must
    still show up so the operator can spot the mismatch."""
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    ctx.on_quote(_quote(token="a", ts=t0))
    stale = ctx.staleness(now=t0 + timedelta(seconds=3))
    assert "a" in stale
    assert stale["a"]["book_age_sec"] is None
    assert stale["a"]["surface_age_sec"] is None


def test_staleness_includes_all_tracked_tokens(ctx: DashboardContext):
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    ctx.on_book(_snap(token="a", ts=t0))
    ctx.on_surface(_surface(token="b", ts=t0))
    ctx.on_quote(_quote(token="c", ts=t0))
    assert set(ctx.staleness().keys()) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# snapshot_dict contract
# ---------------------------------------------------------------------------


def test_snapshot_dict_contains_staleness_block(ctx: DashboardContext):
    ctx.on_book(_snap(token="t"))
    ctx.on_surface(_surface(token="t"))
    snap = ctx.snapshot_dict()
    assert "staleness" in snap
    assert "t" in snap["staleness"]
    entry = snap["staleness"]["t"]
    assert set(entry) == {"book_age_sec", "surface_age_sec"}
    assert entry["book_age_sec"] is not None
    assert entry["surface_age_sec"] is not None


def test_snapshot_dict_surface_carries_sigma_and_lambda(ctx: DashboardContext):
    ctx.on_surface(_surface(token="t"))
    snap = ctx.snapshot_dict()
    assert snap["surface"]["t"]["sigma_b"] == pytest.approx(0.35)
    assert snap["surface"]["t"]["lambda"] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# _fmt_age color thresholds
# ---------------------------------------------------------------------------


def test_fmt_age_none_is_dim():
    out = _fmt_age(None, StalenessThresholds())
    assert "dim" in out
    assert "never" in out


def test_fmt_age_fresh_warn_stale_transitions():
    t = StalenessThresholds(fresh_sec=5.0, warn_sec=30.0)
    assert "green" in _fmt_age(2.0, t)
    assert "yellow" in _fmt_age(10.0, t)
    assert "bold red" in _fmt_age(60.0, t)


def test_fmt_age_unit_scales():
    t = StalenessThresholds(fresh_sec=5, warn_sec=30)
    assert _fmt_age(3.1, t).endswith("3.1s[/green]")
    assert "2.0m" in _fmt_age(120.0, t)
    assert "1.5h" in _fmt_age(5400.0, t)


# ---------------------------------------------------------------------------
# Rich layout render — must not crash pre-Track-A, or post-Track-A
# ---------------------------------------------------------------------------


def test_rich_layout_builds_before_any_data(ctx: DashboardContext):
    dash = RichDashboard(ctx)
    layout = dash._build_layout()
    # Render to a Console capture so rendering itself is exercised.
    from rich.console import Console
    buf = Console(record=True, width=240, height=120)
    buf.print(layout)
    text = buf.export_text()
    assert "waiting on Track A" in text
    assert "no publishers yet" in text


def test_rich_layout_builds_with_full_data(ctx: DashboardContext):
    t0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    ctx.on_book(_snap(token="tok1", ts=t0))
    ctx.on_surface(_surface(token="tok1", ts=t0))
    ctx.on_quote(_quote(token="tok1", ts=t0))

    dash = RichDashboard(ctx)
    layout = dash._build_layout()
    from rich.console import Console
    buf = Console(record=True, width=240, height=120)
    buf.print(layout)
    text = buf.export_text()
    # Surface fields visible
    assert "σ̂_b" in text or "sigma" in text.lower() or "0.3500" in text
    # Staleness pane populated with finite ages (not "never")
    assert "never" not in text.split("Data staleness")[-1].split("Recent fills")[0]
