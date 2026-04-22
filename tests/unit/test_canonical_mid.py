"""Unit tests for `core/filter/canonical_mid.py` (paper §5.1).

Validates the trade-weighted mid, boundary clipping, outlier hygiene, and
uniform-grid resampling. Skipped until Track A lands the module.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

canonical_mid = pytest.importorskip(
    "blksch.core.filter.canonical_mid",
    reason="core/filter/canonical_mid.py not yet implemented",
)


pytestmark = pytest.mark.unit


class TestCanonicalMid:
    def test_trade_weighted_mid_weights_by_size_and_spread(self) -> None:
        """p_tilde = (1/Z) Sum w_u (b_u + a_u)/2, weights monotone in size and
        inverse spread. Paper §5.1."""
        pytest.skip("Stub: fill in when canonical_mid lands")

    def test_clips_to_epsilon_boundary(self) -> None:
        """p_tilde clamped to [eps, 1 - eps] with eps = 1e-5 to avoid exploding logits."""
        pytest.skip("Stub")

    def test_drops_crossed_and_locked_books(self) -> None:
        """A crossed book (bid >= ask) must not produce a mid — flagged as halt."""
        pytest.skip("Stub")

    def test_removes_isolated_spikes(self) -> None:
        """Spikes that revert within one tick + one update are filtered."""
        pytest.skip("Stub")

    def test_resamples_to_uniform_grid_with_lastobs_and_vwap(self) -> None:
        """At 1 Hz (default), intra-bin averaging uses within-bin VWAP and
        last-observation-carried-forward across empty bins."""
        pytest.skip("Stub")

    def test_ignores_updates_below_tick_size(self) -> None:
        """De-bounce flicker: ignore bid/ask changes smaller than the tick."""
        pytest.skip("Stub")
