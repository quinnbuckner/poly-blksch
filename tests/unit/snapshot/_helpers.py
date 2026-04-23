"""Canonicalize-and-compare plumbing for ``tests/unit/snapshot/``.

Pattern borrowed from ``tests/unit/test_paper_soak_determinism.py``: every
float is stored as its ``repr()`` so round-tripping through JSON preserves
full IEEE 754 precision, and every dict key is sorted at every depth so
key-insertion-order drift can't hide behind byte-identity comparisons.

On a snapshot test, the observed value is canonicalized the same way and
compared to the committed fixture:

* string / int / bool fields must be exactly equal;
* float fields (stored as repr-strings) must agree within a configurable
  absolute + relative tolerance — default ``1e-10`` per the plan.

To regenerate a fixture, delete it and rerun the test, **or** set the
environment variable ``SNAPSHOT_UPDATE=1``. A missing fixture is *never*
silently created during a CI run — the test fails hard with an
``AssertionError`` pointing at the fixture path, so an operator rerun is
always an explicit decision.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TOL = 1.0e-10
SNAPSHOT_UPDATE_ENV = "SNAPSHOT_UPDATE"

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def _encode_float(x: float) -> str:
    """Store floats as ``repr()`` strings. Full precision, platform-stable."""
    if math.isnan(x):
        return "__nan__"
    if math.isinf(x):
        return "__inf__" if x > 0 else "__-inf__"
    return repr(float(x))


def canonicalize(obj: Any) -> Any:
    """Recursively normalize an object into a JSON-safe form with sorted
    dict keys and repr-encoded floats."""
    # Pydantic v2 models expose model_dump(); keep them from being
    # mis-handled as "iterables of BaseModel instances".
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return canonicalize(obj.model_dump(mode="json"))
        except Exception:
            pass

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return canonicalize(dataclasses.asdict(obj))

    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": [int(x) for x in obj.shape],
            "dtype": str(obj.dtype),
            "data": [_encode_float(float(x)) for x in obj.ravel().tolist()]
            if obj.dtype.kind in ("f", "c")
            else [int(x) if isinstance(x, (int, np.integer)) else x
                  for x in obj.ravel().tolist()],
        }

    if isinstance(obj, dict):
        return {
            str(k): canonicalize(obj[k])
            for k in sorted(obj.keys(), key=lambda x: str(x))
        }

    if isinstance(obj, (list, tuple, set, frozenset)):
        items = list(obj) if not isinstance(obj, (set, frozenset)) else sorted(obj, key=repr)
        return [canonicalize(v) for v in items]

    if isinstance(obj, np.floating):
        return _encode_float(float(obj))
    if isinstance(obj, float):
        return _encode_float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj

    # Datetimes + enums → str; unknown objects → repr
    return repr(obj) if obj.__class__.__module__ != "builtins" else obj


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _is_float_repr(s: str) -> tuple[bool, float]:
    """If ``s`` is a repr-encoded float sentinel or a parseable float,
    return (True, decoded); else (False, 0.0). Does NOT treat plain int
    strings as floats — the canonicalizer keeps ints as ints."""
    if s == "__nan__":
        return True, math.nan
    if s == "__inf__":
        return True, math.inf
    if s == "__-inf__":
        return True, -math.inf
    if not s or s[0].isalpha():
        return False, 0.0
    # Cheap guard: reject ASCII that isn't obviously a float.
    has_dot_or_e = any(c in s for c in ".eE")
    allowed = all(c in "0123456789+-.eE" for c in s)
    if not (has_dot_or_e and allowed):
        return False, 0.0
    try:
        return True, float(s)
    except ValueError:
        return False, 0.0


def _float_close(a: float, b: float, *, tol: float) -> tuple[bool, float, float]:
    if math.isnan(a) and math.isnan(b):
        return True, 0.0, 0.0
    if math.isinf(a) or math.isinf(b):
        return a == b, 0.0, 0.0
    diff = abs(a - b)
    rel = diff / max(abs(a), abs(b), 1e-300)
    return (diff <= tol or rel <= tol), diff, rel


def _compare(observed: Any, expected: Any, *, tol: float, path: str) -> None:
    if type(observed) != type(expected):
        # Allow int vs int JSON round-trip (json parses ints as int fine).
        raise AssertionError(
            f"{path}: type {type(observed).__name__} vs {type(expected).__name__}\n"
            f"  observed={observed!r}\n  expected={expected!r}"
        )
    if isinstance(observed, dict):
        o_keys = set(observed.keys())
        e_keys = set(expected.keys())
        if o_keys != e_keys:
            missing = sorted(e_keys - o_keys)
            extra = sorted(o_keys - e_keys)
            raise AssertionError(
                f"{path}: key set differs  missing={missing}  extra={extra}"
            )
        for k in sorted(observed.keys()):
            _compare(observed[k], expected[k], tol=tol, path=f"{path}.{k}")
        return
    if isinstance(observed, list):
        if len(observed) != len(expected):
            raise AssertionError(
                f"{path}: length {len(observed)} vs {len(expected)}"
            )
        for i, (o, e) in enumerate(zip(observed, expected)):
            _compare(o, e, tol=tol, path=f"{path}[{i}]")
        return
    if isinstance(observed, str):
        ok_obs, obs_f = _is_float_repr(observed)
        ok_exp, exp_f = _is_float_repr(expected)
        if ok_obs and ok_exp:
            close, diff, rel = _float_close(obs_f, exp_f, tol=tol)
            if not close:
                raise AssertionError(
                    f"{path}: float drift observed={observed} expected={expected} "
                    f"abs_diff={diff:.3e} rel_diff={rel:.3e} tol={tol:.1e}"
                )
            return
        if observed != expected:
            raise AssertionError(
                f"{path}: string {observed!r} vs {expected!r}"
            )
        return
    if observed != expected:
        raise AssertionError(f"{path}: {observed!r} vs {expected!r}")


def assert_matches_snapshot(
    observed: Any,
    snapshot_name: str,
    *,
    tol: float = DEFAULT_TOL,
) -> None:
    """Compare a value against its committed snapshot.

    ``snapshot_name`` is interpreted relative to ``tests/unit/snapshot/fixtures/``
    and must not include a leading ``fixtures/`` prefix. Regenerate a
    fixture by deleting the file or setting ``SNAPSHOT_UPDATE=1`` in the
    environment.
    """
    canon = canonicalize(observed)
    path = FIXTURES_DIR / snapshot_name
    update = os.environ.get(SNAPSHOT_UPDATE_ENV) in ("1", "true", "yes")

    if update or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(canon, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        )
        if not update:
            # Fail loudly so an operator rerun is explicit; the fixture
            # is written either way so the next rerun under SNAPSHOT_UPDATE=1
            # is a no-op diff.
            raise AssertionError(
                f"{snapshot_name}: fixture did not exist; wrote initial snapshot. "
                f"Review {path} and rerun with SNAPSHOT_UPDATE=1 if the "
                f"values look right."
            )
        return

    expected = json.loads(path.read_text())
    _compare(canon, expected, tol=tol, path=snapshot_name)


__all__ = [
    "DEFAULT_TOL",
    "FIXTURES_DIR",
    "SNAPSHOT_UPDATE_ENV",
    "assert_matches_snapshot",
    "canonicalize",
]
