"""Tests for shared ingest helpers (api/_ingest_common.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))
from _ingest_common import (
    decode_bytes,
    find_any,
    find_col,
    float_or_none,
    get_cell,
    int_or_none,
    resolve_db_path,
)


# ---------------------------------------------------------------------------
# decode_bytes
# ---------------------------------------------------------------------------
def test_decode_plain_utf8() -> None:
    assert decode_bytes(b"hello") == "hello"


def test_decode_utf8_bom() -> None:
    assert decode_bytes(b"\xef\xbb\xbfhello") == "hello"


def test_decode_cp1252_fallback() -> None:
    # Norwegian å is 0xE5 in cp1252 (not valid utf-8 in isolation)
    text = decode_bytes(b"h\xe5llo")
    assert "å" in text


# ---------------------------------------------------------------------------
# find_any / find_col
# ---------------------------------------------------------------------------
def test_find_any_single_needle() -> None:
    assert find_any(["date", "value"], "date") == 0
    assert find_any(["date", "value"], "value") == 1


def test_find_any_multiple_needles_all_must_match() -> None:
    header = ["historic glucose mmol/l", "scan glucose mmol/l"]
    assert find_any(header, "historic", "mmol") == 0
    assert find_any(header, "scan", "mmol") == 1


def test_find_any_no_match_returns_none() -> None:
    assert find_any(["a", "b"], "c") is None


def test_find_col_index_zero_is_preserved() -> None:
    # Regression: "a or b" returns b when a is 0. find_col must return 0.
    assert find_col(["timestamp", "value"], ("timestamp",), ("date",)) == 0


def test_find_col_tries_alternatives_in_order() -> None:
    header = ["device", "serial", "tidsstempel"]
    # First alternative ("timestamp") misses; second ("tidsstempel") hits
    assert find_col(header, ("timestamp",), ("tidsstempel",)) == 2


def test_find_col_no_match_returns_none() -> None:
    assert find_col(["a", "b"], ("c",), ("d",)) is None


# ---------------------------------------------------------------------------
# get_cell / int_or_none / float_or_none
# ---------------------------------------------------------------------------
def test_get_cell_trims_and_preserves_content() -> None:
    assert get_cell(["  hello  ", "x"], 0) == "hello"


def test_get_cell_returns_none_on_empty() -> None:
    assert get_cell(["", "x"], 0) is None


def test_get_cell_returns_none_on_out_of_range() -> None:
    assert get_cell(["a"], 5) is None
    assert get_cell(["a"], None) is None


def test_int_or_none_handles_comma_decimal() -> None:
    assert int_or_none("72,0") == 72
    assert int_or_none("72.5") == 72
    assert int_or_none("72") == 72


def test_int_or_none_returns_none_on_empty_or_invalid() -> None:
    assert int_or_none(None) is None
    assert int_or_none("") is None
    assert int_or_none("not a number") is None


def test_float_or_none_handles_comma_decimal() -> None:
    assert float_or_none("5,6") == 5.6
    assert float_or_none("5.6") == 5.6


def test_float_or_none_returns_none_on_empty_or_invalid() -> None:
    assert float_or_none(None) is None
    assert float_or_none("") is None
    assert float_or_none("nope") is None


# ---------------------------------------------------------------------------
# resolve_db_path
# ---------------------------------------------------------------------------
def test_resolve_db_path_override_wins() -> None:
    # Override wins even when profile is set
    result = resolve_db_path(
        db_override="/tmp/custom.db", profile_name="henrik"
    )
    assert result == "/tmp/custom.db"


def test_resolve_db_path_profile_lookup() -> None:
    # Uses real profiles.py — "mitch" is a known profile
    result = resolve_db_path(profile_name="mitch")
    assert result.endswith("mitch.db")


def test_resolve_db_path_unknown_profile_raises() -> None:
    try:
        resolve_db_path(profile_name="nonexistent_profile_xyz")
    except SystemExit as e:
        assert "unknown profile" in str(e)
        return
    raise AssertionError("expected SystemExit for unknown profile")


def test_resolve_db_path_default_falls_back_to_config() -> None:
    # No override, no profile → config.DATABASE_PATH
    from config import DATABASE_PATH

    result = resolve_db_path()
    assert result == str(DATABASE_PATH)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all() -> int:
    tests = [
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    failures: list[tuple[str, BaseException]] = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except BaseException as e:  # noqa: BLE001
            failures.append((name, e))
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
