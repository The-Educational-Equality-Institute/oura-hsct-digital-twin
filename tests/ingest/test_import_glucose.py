"""Tests for the glucose CSV parser and upsert logic.

Runs as a standalone script OR under pytest. Uses synthetic LibreView-format
CSV fixtures (no real patient data) to validate parser robustness against
common format variants the parser will encounter on first real export.

    python api/test_import_glucose.py       # direct run, stdlib only
    pytest api/test_import_glucose.py       # works too, if pytest installed
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))
from import_glucose import (
    MG_DL_TO_MMOL_L,
    _parse_timestamp,
    init_glucose_readings_table,
    init_symptom_events_table,
    parse_libreview_csv,
    upsert_glucose_readings,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures — no real patient data
# ---------------------------------------------------------------------------
LIBREVIEW_ENGLISH_MMOL = dedent(
    """\
    Patient Name,Test Patient

    Device,Serial Number,Device Timestamp,Record Type,Historic Glucose mmol/L,Scan Glucose mmol/L
    FreeStyle Libre 3 Plus,ABC123,21-04-2026 14:30,0,5.6,
    FreeStyle Libre 3 Plus,ABC123,21-04-2026 14:31,0,5.7,
    FreeStyle Libre 3 Plus,ABC123,21-04-2026 14:35,1,,6.2
    FreeStyle Libre 3 Plus,ABC123,21-04-2026 14:40,2,,
    """
)

LIBREVIEW_ENGLISH_MGDL = dedent(
    """\
    Patient Name,Test Patient

    Device,Serial Number,Device Timestamp,Record Type,Historic Glucose mg/dL,Scan Glucose mg/dL
    FreeStyle Libre 3 Plus,XYZ789,21-04-2026 09:00,0,108,
    FreeStyle Libre 3 Plus,XYZ789,21-04-2026 09:01,0,110,
    """
)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
def test_parse_timestamp_dd_mm_yyyy() -> None:
    assert _parse_timestamp("21-04-2026 14:30") == "2026-04-21T14:30:00"


def test_parse_timestamp_iso_with_seconds() -> None:
    assert _parse_timestamp("2026-04-21 14:30:15") == "2026-04-21T14:30:15"


def test_parse_timestamp_unrecognized_raises() -> None:
    try:
        _parse_timestamp("not-a-timestamp")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unrecognized timestamp")


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def test_parse_english_mmol_csv() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "libreview.csv"
        csv_path.write_text(LIBREVIEW_ENGLISH_MMOL)

        readings = parse_libreview_csv(csv_path)

    # 3 rows with record_type in {0, 1}; row with type 2 is filtered
    assert len(readings) == 3, f"expected 3 readings, got {len(readings)}"
    assert readings[0]["glucose_mmol_l"] == 5.6
    assert readings[0]["record_type"] == 0
    assert readings[0]["sensor_serial"] == "ABC123"
    assert readings[0]["source"] == "libreview_csv"
    assert readings[2]["record_type"] == 1
    assert readings[2]["glucose_mmol_l"] == 6.2


def test_parse_mgdl_converts_to_mmol() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "libreview_us.csv"
        csv_path.write_text(LIBREVIEW_ENGLISH_MGDL)

        readings = parse_libreview_csv(csv_path)

    assert len(readings) == 2
    expected = 108 / MG_DL_TO_MMOL_L
    assert abs(readings[0]["glucose_mmol_l"] - expected) < 0.001


def test_sensor_serial_override() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "libreview.csv"
        csv_path.write_text(LIBREVIEW_ENGLISH_MMOL)

        readings = parse_libreview_csv(csv_path, sensor_serial="OVERRIDE")

    assert all(r["sensor_serial"] == "OVERRIDE" for r in readings)


def test_missing_header_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "bad.csv"
        csv_path.write_text("just some random text,without any header\n")

        try:
            parse_libreview_csv(csv_path)
        except ValueError as e:
            assert "Timestamp" in str(e) or "Tidsstempel" in str(e)
            return
    raise AssertionError("expected ValueError for missing header")


def test_utf8_bom_handled() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "libreview_bom.csv"
        csv_path.write_bytes(b"\xef\xbb\xbf" + LIBREVIEW_ENGLISH_MMOL.encode("utf-8"))

        readings = parse_libreview_csv(csv_path)

    assert len(readings) == 3


# ---------------------------------------------------------------------------
# Schema init + upsert idempotency
# ---------------------------------------------------------------------------
def test_upsert_idempotent_on_duplicate() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_glucose_readings_table(conn)
            readings = [
                {
                    "timestamp": "2026-04-21T14:30:00",
                    "glucose_mmol_l": 5.6,
                    "record_type": 0,
                    "sensor_serial": "ABC123",
                    "source": "libreview_csv",
                }
            ]

            inserted1, skipped1 = upsert_glucose_readings(conn, readings)
            inserted2, skipped2 = upsert_glucose_readings(conn, readings)

            assert (inserted1, skipped1) == (1, 0)
            assert (inserted2, skipped2) == (0, 1)

            count = conn.execute("SELECT COUNT(*) FROM glucose_readings").fetchone()[0]
            assert count == 1
        finally:
            conn.close()


def test_init_tables_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            # Calling each init function twice must not error
            init_glucose_readings_table(conn)
            init_glucose_readings_table(conn)
            init_symptom_events_table(conn)
            init_symptom_events_table(conn)

            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "glucose_readings" in tables
            assert "symptom_events" in tables
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Runner for direct invocation (pytest auto-discovers the same functions)
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
        except BaseException as e:  # noqa: BLE001 — test runner needs to catch all
            failures.append((name, e))
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
