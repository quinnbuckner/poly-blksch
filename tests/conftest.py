"""Shared pytest fixtures used across unit, integration, and pipeline tests.

Fixtures here are intentionally deterministic and small — they should load in
milliseconds so unit tests stay fast. Heavier synthetic data (e.g. the paper's
§6 RN-consistent path) lives in `tests/fixtures/synthetic.py`.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from blksch.schemas import (
    BookSnap,
    CorrelationEntry,
    Fill,
    LogitState,
    Order,
    OrderSide,
    Position,
    PriceLevel,
    Quote,
    SurfacePoint,
    TradeSide,
    TradeTick,
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    """Single timestamp for deterministic test runs."""
    return datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def token_id() -> str:
    return "test-token-yes"


@pytest.fixture
def peer_token_id() -> str:
    return "test-token-no"


# ---------------------------------------------------------------------------
# Math helpers (exposed as fixtures so tests can assert identities)
# ---------------------------------------------------------------------------


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@pytest.fixture
def logit_fn():
    return logit


@pytest.fixture
def sigmoid_fn():
    return sigmoid


# ---------------------------------------------------------------------------
# Book / trade fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def book_mid_50(token_id: str, now: datetime) -> BookSnap:
    """Balanced book, mid = 0.50, spread = 0.02."""
    return BookSnap(
        token_id=token_id,
        bids=[PriceLevel(price=0.49, size=1000), PriceLevel(price=0.48, size=2000)],
        asks=[PriceLevel(price=0.51, size=1000), PriceLevel(price=0.52, size=2000)],
        ts=now,
        seq=1,
    )


@pytest.fixture
def book_mid_80(token_id: str, now: datetime) -> BookSnap:
    """Skewed book, mid = 0.80, near boundary where S'(x) shrinks."""
    return BookSnap(
        token_id=token_id,
        bids=[PriceLevel(price=0.79, size=500), PriceLevel(price=0.78, size=1000)],
        asks=[PriceLevel(price=0.81, size=500), PriceLevel(price=0.82, size=1000)],
        ts=now,
        seq=1,
    )


@pytest.fixture
def buy_trade(token_id: str, now: datetime) -> TradeTick:
    return TradeTick(
        token_id=token_id,
        price=0.51,
        size=100,
        aggressor_side=TradeSide.BUY,
        ts=now,
    )


@pytest.fixture
def sell_trade(token_id: str, now: datetime) -> TradeTick:
    return TradeTick(
        token_id=token_id,
        price=0.49,
        size=100,
        aggressor_side=TradeSide.SELL,
        ts=now,
    )


# ---------------------------------------------------------------------------
# Calibration fixtures (what Track A emits)
# ---------------------------------------------------------------------------


@pytest.fixture
def logit_state_mid(token_id: str, now: datetime) -> LogitState:
    return LogitState(token_id=token_id, x_hat=0.0, sigma_eta2=0.01, ts=now)


@pytest.fixture
def surface_point_mid(token_id: str, now: datetime) -> SurfacePoint:
    """Representative point mid-book, 1-hour time-to-resolution."""
    return SurfacePoint(
        token_id=token_id,
        tau=3600.0,
        m=0.0,
        sigma_b=0.5,
        **{"lambda": 0.05},
        s2_j=0.04,
        uncertainty=0.05,
        ts=now,
    )


@pytest.fixture
def surface_point_boundary(token_id: str, now: datetime) -> SurfacePoint:
    """Near-boundary point where S'(x) is small and inventory cap tightens."""
    return SurfacePoint(
        token_id=token_id,
        tau=3600.0,
        m=logit(0.9),
        sigma_b=0.5,
        **{"lambda": 0.1},
        s2_j=0.09,
        uncertainty=0.1,
        ts=now,
    )


@pytest.fixture
def correlation_entry(token_id: str, peer_token_id: str, now: datetime) -> CorrelationEntry:
    return CorrelationEntry(
        token_id_i=token_id,
        token_id_j=peer_token_id,
        rho=0.6,
        co_jump_lambda=0.02,
        co_jump_m2=0.05,
        ts=now,
    )


# ---------------------------------------------------------------------------
# Position / order / fill fixtures (what Track C emits)
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_position(token_id: str) -> Position:
    return Position(token_id=token_id, qty=0.0, avg_entry=0.5, mark=0.5)


@pytest.fixture
def long_position(token_id: str) -> Position:
    return Position(token_id=token_id, qty=100.0, avg_entry=0.48, mark=0.50)


@pytest.fixture
def short_position(token_id: str) -> Position:
    return Position(token_id=token_id, qty=-100.0, avg_entry=0.52, mark=0.50)


@pytest.fixture
def sample_order(token_id: str, now: datetime) -> Order:
    return Order(
        token_id=token_id,
        side=OrderSide.BUY,
        price=0.49,
        size=100,
        client_id="test-client-1",
        created_ts=now,
    )


@pytest.fixture
def sample_fill(token_id: str, now: datetime) -> Fill:
    return Fill(
        order_client_id="test-client-1",
        order_venue_id="venue-1",
        token_id=token_id,
        side=OrderSide.BUY,
        price=0.49,
        size=100,
        fee_usd=0.05,
        ts=now,
    )


# ---------------------------------------------------------------------------
# Quote fixture (what Track B emits)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_quote(token_id: str, now: datetime) -> Quote:
    return Quote(
        token_id=token_id,
        p_bid=0.49,
        p_ask=0.51,
        x_bid=-0.04,
        x_ask=0.04,
        size_bid=100,
        size_ask=100,
        half_spread_x=0.04,
        reservation_x=0.0,
        inventory_q=0.0,
        ts=now,
    )


# ---------------------------------------------------------------------------
# Timeline helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def timeline():
    """Yield a sequence of increasing timestamps 1 second apart."""

    def _gen(start: datetime, n: int, step_sec: float = 1.0):
        return [start + timedelta(seconds=i * step_sec) for i in range(n)]

    return _gen
