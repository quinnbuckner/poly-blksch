"""Argument parsing + dry-run behavior for Track C operator scripts.

These scripts can destroy money (`signing_canary.py` places a real order) or
leak creds (`live_ro_auth_check.py` constructs a signing client), so the
dry-run path needs to stay rock-solid: no network, no env validation,
printable plan, exit 0.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _import_script(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    # Use a mangled module name so we don't clash with anything in `scripts`
    # that might already be importable via sys.path shenanigans.
    spec = importlib.util.spec_from_file_location(f"blksch_scripts_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# live_ro_auth_check
# ---------------------------------------------------------------------------


def test_live_ro_auth_check_dry_run_requires_no_env(monkeypatch, caplog):
    mod = _import_script("live_ro_auth_check")
    # Wipe any creds the developer might have in env — dry-run must not care.
    for k in (*mod.REQUIRED_ENV, *mod.OPTIONAL_L2_ENV):
        monkeypatch.delenv(k, raising=False)

    caplog.set_level(logging.WARNING)
    rc = mod.main([])  # no --i-mean-it
    assert rc == 0
    text = caplog.text
    assert "Dry-run" in text
    assert "GET /orders" in text
    assert "--i-mean-it" in text


def test_live_ro_auth_check_armed_without_creds_fails_fast(monkeypatch, caplog):
    mod = _import_script("live_ro_auth_check")
    for k in (*mod.REQUIRED_ENV, *mod.OPTIONAL_L2_ENV):
        monkeypatch.delenv(k, raising=False)
    # Also block python-dotenv from loading a developer's local .env.
    monkeypatch.setattr(mod, "_load_env", lambda: None)

    caplog.set_level(logging.ERROR)
    rc = mod.main(["--i-mean-it"])
    assert rc == 2
    assert "Missing required env vars" in caplog.text


# ---------------------------------------------------------------------------
# signing_canary
# ---------------------------------------------------------------------------


def test_signing_canary_help_works_without_env():
    mod = _import_script("signing_canary")
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0


def test_signing_canary_dry_run_prints_plan(monkeypatch, caplog):
    mod = _import_script("signing_canary")
    caplog.set_level(logging.WARNING)
    rc = mod.main(["--token-id", "0xdeadbeef"])  # no --i-mean-it
    assert rc == 0
    assert "Dry-run" in caplog.text
    assert "--i-mean-it" in caplog.text
