"""Unit tests for ``scripts/calibration_dryrun.py``.

Covers the pure, test-friendly pieces: CLI parsing, safety checks,
verdict logic, and JSON serialization. The async live-ingest shell
(``ingest_and_snapshot``, ``main_async``) is exercised offline by
``tests/integration/test_calibration_dryrun_offline.py``.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# The script lives outside the src/ package; put it on sys.path.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import calibration_dryrun as cd  # noqa: E402

pytestmark = pytest.mark.unit


# ---------- CLI ----------


def test_parse_args_explicit_token() -> None:
    cfg = cd.parse_args(
        ["--token-id", "tok-abc", "--minutes", "10",
         "--out", "./runs/x", "--snapshot-every", "5"]
    )
    assert cfg.token_id == "tok-abc"
    assert cfg.auto_select is False
    assert cfg.minutes == 10
    # 5-min stride × 10-min run → snapshots at 300 s and 600 s.
    assert cfg.snapshot_intervals_sec == (300, 600)
    assert cfg.out_dir == Path("./runs/x")
    assert cfg.i_mean_it is False


def test_parse_args_auto_mode() -> None:
    cfg = cd.parse_args(["--auto", "--minutes", "15", "--out", "./runs/auto"])
    assert cfg.auto_select is True
    assert cfg.token_id is None
    # default snapshot_every=5 → 5, 10, 15 min.
    assert cfg.snapshot_intervals_sec == (300, 600, 900)


def test_parse_args_i_mean_it_flag() -> None:
    cfg = cd.parse_args(["--auto", "--minutes", "1", "--out", "./x", "--i-mean-it"])
    assert cfg.i_mean_it is True


def test_parse_args_rejects_both_token_and_auto() -> None:
    with pytest.raises(SystemExit):
        cd.parse_args(["--token-id", "t", "--auto", "--minutes", "5", "--out", "./x"])


def test_parse_args_requires_one_source() -> None:
    with pytest.raises(SystemExit):
        cd.parse_args(["--minutes", "5", "--out", "./x"])


def test_parse_args_short_duration_yields_end_only_snapshot() -> None:
    # duration < snapshot stride → single end-snapshot.
    cfg = cd.parse_args(
        ["--auto", "--minutes", "2", "--out", "./x", "--snapshot-every", "5"]
    )
    assert cfg.snapshot_intervals_sec == (120,)


# ---------- Safety ----------


def test_check_out_dir_safety_allows_adhoc_path(tmp_path: Path) -> None:
    # A tmp_path under pytest is never the shared ./data/ dir.
    cd.check_out_dir_safety(tmp_path, i_mean_it=False)


def test_check_out_dir_safety_requires_flag_for_shared_data(tmp_path: Path, monkeypatch) -> None:
    # Simulate a run from a repo whose cwd contains a shared data/ dir.
    fake_repo = tmp_path / "repo"
    data_dir = fake_repo / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.chdir(fake_repo)
    with pytest.raises(SystemExit, match="--i-mean-it"):
        cd.check_out_dir_safety(data_dir / "dryrun-xyz", i_mean_it=False)
    # With the flag, it passes.
    cd.check_out_dir_safety(data_dir / "dryrun-xyz", i_mean_it=True)


def test_check_out_dir_safety_missing_shared_is_noop(tmp_path: Path, monkeypatch) -> None:
    # No shared data/ dir exists → nothing to warn about.
    monkeypatch.chdir(tmp_path)
    cd.check_out_dir_safety(tmp_path / "out", i_mean_it=False)


# ---------- Header ----------


def test_log_header_shape() -> None:
    header = cd.log_header("calibration_dryrun.py", ["--auto", "--minutes", "1"])
    assert set(header) == {"script", "args", "ts", "user"}
    assert header["script"] == "calibration_dryrun.py"
    assert header["args"] == ["--auto", "--minutes", "1"]
    # timestamp should ISO-parse.
    from datetime import datetime
    datetime.fromisoformat(header["ts"])


# ---------- Verdict ----------


def _mk_diag(
    *,
    lb_p: float = 0.5,
    sw_pass: bool = True,
    var_rel: float = 0.05,
) -> cd.DiagnosticsReport:
    return cd.DiagnosticsReport(
        ljung_box_lags=20,
        ljung_box_statistic=5.0,
        ljung_box_pvalue=lb_p,
        ljung_box_pass=lb_p > 0.05,
        qq_quantile_correlation=0.99,
        qq_shapiro_wilk_pvalue=0.3 if sw_pass else 0.001,
        qq_shapiro_wilk_pass=sw_pass,
        realized_variance=0.5,
        implied_variance=0.5 / (1.0 + var_rel) if var_rel > 0 else 0.5,
        variance_rel_error=var_rel,
        variance_tolerance_pct=0.2,
        variance_consistency_pass=var_rel <= 0.2,
    )


def test_verdict_green_when_all_pass() -> None:
    d = _mk_diag(lb_p=0.5, sw_pass=True, var_rel=0.05)
    assert cd.verdict_for_snapshot(d, forecast_qlike=0.5) == cd.VERDICT_GREEN


def test_verdict_red_on_serial_correlation() -> None:
    d = _mk_diag(lb_p=0.001)
    assert cd.verdict_for_snapshot(d, forecast_qlike=0.5) == cd.VERDICT_RED


def test_verdict_red_on_big_variance_error() -> None:
    d = _mk_diag(var_rel=0.7)
    assert cd.verdict_for_snapshot(d, forecast_qlike=0.5) == cd.VERDICT_RED


def test_verdict_yellow_on_borderline_ljung_box() -> None:
    d = _mk_diag(lb_p=0.03)  # between RED and YELLOW thresholds
    assert cd.verdict_for_snapshot(d, forecast_qlike=0.5) == cd.VERDICT_YELLOW


def test_verdict_yellow_on_shapiro_fail() -> None:
    d = _mk_diag(sw_pass=False)
    assert cd.verdict_for_snapshot(d, forecast_qlike=0.5) == cd.VERDICT_YELLOW


def test_verdict_yellow_on_borderline_variance() -> None:
    d = _mk_diag(var_rel=0.35)
    assert cd.verdict_for_snapshot(d, forecast_qlike=0.5) == cd.VERDICT_YELLOW


def test_verdict_yellow_on_high_forecast_qlike() -> None:
    d = _mk_diag()
    assert cd.verdict_for_snapshot(d, forecast_qlike=3.0) == cd.VERDICT_YELLOW


def test_verdict_yellow_when_diagnostics_missing() -> None:
    assert cd.verdict_for_snapshot(None, forecast_qlike=float("nan")) == cd.VERDICT_YELLOW


# ---------- Report serialization ----------


def test_as_jsonable_handles_dataclass_and_numpy() -> None:
    import numpy as np

    @dataclass
    class Dummy:
        a: int
        b: list[float]

    nested = {"x": Dummy(a=1, b=[1.0, 2.0]), "arr": np.array([1, 2, 3])}
    js = cd._as_jsonable(nested)
    # round-trip through json to catch any non-serializable leftover.
    s = json.dumps(js)
    assert '"a": 1' in s
    assert '[1, 2, 3]' in s


def test_as_jsonable_coerces_non_finite_floats_to_null() -> None:
    assert cd._as_jsonable(float("nan")) is None
    assert cd._as_jsonable(float("inf")) is None
    assert cd._as_jsonable(0.5) == 0.5


def test_render_report_shape_and_overall_verdict() -> None:
    d_green = _mk_diag()
    d_yellow = _mk_diag(var_rel=0.3)
    snaps = [
        cd.CalibrationSnapshot(
            t_elapsed_sec=300, n_books=100, n_trades=10, n_states=95,
            sigma_b_hat=0.05, lambda_hat=0.01, s_J_sq_hat=0.003,
            em_converged=True, em_iters=5,
            diagnostics=d_green,
            self_forecast_mse=1.0, self_forecast_mae=0.5, self_forecast_qlike=0.3,
            innovations_n=90, verdict=cd.VERDICT_GREEN,
        ),
        cd.CalibrationSnapshot(
            t_elapsed_sec=600, n_books=200, n_trades=20, n_states=195,
            sigma_b_hat=0.05, lambda_hat=0.01, s_J_sq_hat=0.003,
            em_converged=True, em_iters=4,
            diagnostics=d_yellow,
            self_forecast_mse=1.5, self_forecast_mae=0.7, self_forecast_qlike=0.8,
            innovations_n=190, verdict=cd.VERDICT_YELLOW,
        ),
    ]
    report = cd.render_report(
        header={"script": "x", "args": [], "ts": "2026-04-23T00:00:00+00:00", "user": "u"},
        token_id="tok-1",
        snapshots=snaps,
        bot_config={"calibration": {"kf_grid_hz": 1.0}},
    )
    assert report["overall_verdict"] == cd.VERDICT_YELLOW
    assert report["token_id"] == "tok-1"
    assert report["n_snapshots"] == 2
    # JSON serializable.
    json.dumps(report)


def test_render_report_red_overrides_other_verdicts() -> None:
    snaps = [
        cd.CalibrationSnapshot(
            t_elapsed_sec=300, n_books=1, n_trades=0, n_states=1,
            sigma_b_hat=0.05, lambda_hat=0.01, s_J_sq_hat=0.003,
            em_converged=True, em_iters=1,
            diagnostics=_mk_diag(), self_forecast_mse=1.0,
            self_forecast_mae=0.5, self_forecast_qlike=0.2,
            innovations_n=100, verdict=cd.VERDICT_GREEN,
        ),
        cd.CalibrationSnapshot(
            t_elapsed_sec=600, n_books=2, n_trades=0, n_states=2,
            sigma_b_hat=0.05, lambda_hat=0.01, s_J_sq_hat=0.003,
            em_converged=True, em_iters=1,
            diagnostics=_mk_diag(lb_p=0.0001),
            self_forecast_mse=1.0, self_forecast_mae=0.5, self_forecast_qlike=0.2,
            innovations_n=100, verdict=cd.VERDICT_RED,
        ),
    ]
    report = cd.render_report(
        header={"script": "x", "args": [], "ts": "2026-04-23T00:00:00+00:00", "user": "u"},
        token_id="t",
        snapshots=snaps,
        bot_config={},
    )
    assert report["overall_verdict"] == cd.VERDICT_RED


def test_render_report_all_green_when_all_green() -> None:
    snap = cd.CalibrationSnapshot(
        t_elapsed_sec=300, n_books=1, n_trades=0, n_states=1,
        sigma_b_hat=0.05, lambda_hat=0.01, s_J_sq_hat=0.003,
        em_converged=True, em_iters=1,
        diagnostics=_mk_diag(),
        self_forecast_mse=1.0, self_forecast_mae=0.5, self_forecast_qlike=0.2,
        innovations_n=100, verdict=cd.VERDICT_GREEN,
    )
    report = cd.render_report(
        header={"script": "x", "args": [], "ts": "2026-04-23T00:00:00+00:00", "user": "u"},
        token_id="t",
        snapshots=[snap],
        bot_config={},
    )
    assert report["overall_verdict"] == cd.VERDICT_GREEN


def test_render_report_empty_snapshots_is_yellow() -> None:
    report = cd.render_report(
        header={"script": "x", "args": [], "ts": "2026-04-23T00:00:00+00:00", "user": "u"},
        token_id="t",
        snapshots=[],
        bot_config={},
    )
    assert report["overall_verdict"] == cd.VERDICT_YELLOW
    assert report["n_snapshots"] == 0


def test_write_report_creates_file(tmp_path: Path) -> None:
    report = {"header": {"script": "x"}, "overall_verdict": "GREEN", "snapshots": []}
    path = cd.write_report(report, tmp_path / "subdir")
    assert path.exists()
    assert json.loads(path.read_text())["overall_verdict"] == "GREEN"


def test_console_summary_renders_each_snapshot() -> None:
    snap = cd.CalibrationSnapshot(
        t_elapsed_sec=300, n_books=100, n_trades=10, n_states=95,
        sigma_b_hat=0.05, lambda_hat=0.01, s_J_sq_hat=0.003,
        em_converged=True, em_iters=5,
        diagnostics=_mk_diag(),
        self_forecast_mse=1.0, self_forecast_mae=0.5, self_forecast_qlike=0.3,
        innovations_n=90, verdict=cd.VERDICT_GREEN,
    )
    report = cd.render_report(
        header={"script": "x", "args": [], "ts": "2026-04-23T00:00:00+00:00", "user": "u"},
        token_id="tok-1",
        snapshots=[snap],
        bot_config={},
    )
    text = cd.console_summary(report)
    assert "tok-1" in text
    assert "GREEN" in text
    assert "t+  300s" in text


# ---------- Self-forecast metrics on a trivial synthetic series ----------


def test_self_forecast_metrics_on_short_series_returns_nan() -> None:
    from datetime import UTC, datetime, timedelta

    from blksch.schemas import LogitState

    t0 = datetime(2026, 4, 23, tzinfo=UTC)
    states = [
        LogitState(token_id="t", x_hat=0.0, sigma_eta2=0.01,
                   ts=t0 + timedelta(seconds=i))
        for i in range(10)  # < horizon=60
    ]
    mse, mae, qlike, n = cd._self_forecast_metrics(
        states, horizon_sec=60, ewma_half_life_sec=90.0,
    )
    assert math.isnan(mse) and math.isnan(mae) and math.isnan(qlike)
    assert n == 0
