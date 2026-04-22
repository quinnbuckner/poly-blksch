"""Kill-switch behavior under fault injection (paper §4.6).

Stage 1 required: every kill-switch must fire within one refresh cycle of
its triggering condition, and every firing must produce an audit-log entry
naming the cause.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.pipeline


try:  # pragma: no cover
    from blksch.mm.limits import KillSwitch  # type: ignore[attr-defined]  # noqa: F401
    from blksch.exec.paper_engine import PaperEngine  # noqa: F401
    _READY = True
except ImportError:
    _READY = False


@pytest.mark.skipif(not _READY, reason="mm/limits or exec/paper_engine not importable")
class TestKillSwitchFaults:
    def test_feed_gap_triggers_pause(self) -> None:
        """Inject a feed_gap_sec + 1 gap, assert paused=True within one refresh."""
        pytest.skip("Stub: wire kill-switch harness")

    def test_volatility_spike_triggers_pause(self) -> None:
        """Inject sigma_b spike of Z = volatility_spike_z, assert pause."""
        pytest.skip("Stub")

    def test_repeated_pickoffs_trigger_pause(self) -> None:
        """N adverse fills (paper's 'repeated pick-offs') within the window
        must pause; log must identify the fill sequence."""
        pytest.skip("Stub")

    def test_drawdown_triggers_pause(self) -> None:
        """Realized PnL breaching -max_drawdown_usd must pause immediately."""
        pytest.skip("Stub")

    def test_pause_cancels_all_resting_orders(self) -> None:
        """Any paused kill-switch must cancel all resting orders in Track C."""
        pytest.skip("Stub")

    def test_pause_reason_is_in_audit_log(self) -> None:
        """Every pause must write a structured log entry with {reason, ts,
        snapshot of triggering metric}."""
        pytest.skip("Stub")

    def test_resume_requires_explicit_ack(self) -> None:
        """Auto-pauses do NOT auto-resume — resume requires an explicit API call
        (safety: prevent thundering-herd re-enable after transient)."""
        pytest.skip("Stub")
