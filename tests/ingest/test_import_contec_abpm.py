"""Tests for the Contec ABPM50 importer.

Synthetic fixtures cover the most likely column structures for Contec ABPM
CSV exports. Real format validation happens within 24h after the first real
export arrives.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))
from import_contec_abpm import (
    classify_day_night,
    compute_map,
    detect_artifact,
    init_contec_abpm_tables,
    parse_abpm_csv,
    record_session_summary,
    upsert_abpm_readings,
)


ABPM_CSV_SEPARATE_DATE_TIME = dedent(
    """\
    No.,Date,Time,SYS,DIA,HR,MAP,Code
    1,2026-04-28,09:00,125,78,72,93.7,0
    2,2026-04-28,09:30,132,82,75,98.7,0
    3,2026-04-28,22:30,118,70,64,86.0,0
    4,2026-04-29,03:00,108,65,58,79.3,0
    """
)

ABPM_CSV_COMBINED_DATETIME = dedent(
    """\
    Datetime,Systolic,Diastolic,Pulse,Error
    2026-04-28T09:00:00,125,78,72,0
    2026-04-28T09:30:00,132,82,75,0
    """
)

ABPM_CSV_WITH_ARTIFACT = dedent(
    """\
    No.,Date,Time,SYS,DIA,HR,Code
    1,2026-04-28,09:00,125,78,72,0
    2,2026-04-28,09:30,300,90,80,5
    3,2026-04-28,10:00,120,115,75,0
    """
)


# ---------------------------------------------------------------------------
# Unit-level functions
# ---------------------------------------------------------------------------
def test_compute_map_standard_formula() -> None:
    # MAP = DIA + (SYS - DIA) / 3
    assert compute_map(120, 80) == 93.3
    assert compute_map(140, 90) == 106.7


def test_classify_day_night_esh_convention() -> None:
    assert classify_day_night(datetime(2026, 4, 28, 9, 0)) == "day"
    assert classify_day_night(datetime(2026, 4, 28, 21, 59)) == "day"
    assert classify_day_night(datetime(2026, 4, 28, 22, 0)) == "night"
    assert classify_day_night(datetime(2026, 4, 29, 3, 0)) == "night"
    assert classify_day_night(datetime(2026, 4, 29, 6, 59)) == "night"
    assert classify_day_night(datetime(2026, 4, 29, 7, 0)) == "day"


def test_detect_artifact_clean_reading() -> None:
    now = datetime(2026, 4, 28, 12, 0)
    artifact, reason = detect_artifact(120, 80, 72, datetime(2026, 4, 28, 9, 0), now)
    assert artifact is False
    assert reason is None


def test_detect_artifact_sys_out_of_range() -> None:
    now = datetime(2026, 4, 28, 12, 0)
    artifact, reason = detect_artifact(300, 80, 72, datetime(2026, 4, 28, 9, 0), now)
    assert artifact is True
    assert "sys_out_of_range" in reason


def test_detect_artifact_narrow_pulse_pressure() -> None:
    now = datetime(2026, 4, 28, 12, 0)
    artifact, reason = detect_artifact(120, 115, 72, datetime(2026, 4, 28, 9, 0), now)
    assert artifact is True
    assert "narrow_pulse_pressure" in reason


def test_detect_artifact_future_timestamp() -> None:
    now = datetime(2026, 4, 28, 12, 0)
    artifact, reason = detect_artifact(120, 80, 72, datetime(2048, 1, 1), now)
    assert artifact is True
    assert "future_timestamp" in reason


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def test_parse_separate_date_time() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "abpm.csv"
        csv_path.write_text(ABPM_CSV_SEPARATE_DATE_TIME)

        readings = parse_abpm_csv(
            csv_path, session_id="TEST", now=datetime(2026, 4, 29, 15, 0)
        )

    assert len(readings) == 4
    assert readings[0]["sys"] == 125
    assert readings[0]["dia"] == 78
    assert readings[0]["bpm"] == 72
    assert readings[0]["map_mmhg"] == 93.7
    assert readings[0]["pulse_pressure"] == 47
    assert readings[0]["day_night"] == "day"
    assert readings[0]["is_artifact"] == 0
    assert readings[2]["day_night"] == "night"  # 22:30
    assert readings[3]["day_night"] == "night"  # 03:00
    assert readings[0]["datetime"] == "2026-04-28T09:00:00"


def test_parse_combined_datetime() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "abpm.csv"
        csv_path.write_text(ABPM_CSV_COMBINED_DATETIME)

        readings = parse_abpm_csv(
            csv_path, session_id="TEST", now=datetime(2026, 4, 29, 15, 0)
        )

    assert len(readings) == 2
    assert readings[0]["sys"] == 125
    # MAP computed since not in export
    assert readings[0]["map_mmhg"] == compute_map(125, 78)


def test_parse_computes_map_when_missing() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "abpm.csv"
        csv_path.write_text(ABPM_CSV_COMBINED_DATETIME)

        readings = parse_abpm_csv(
            csv_path, session_id="TEST", now=datetime(2026, 4, 29, 15, 0)
        )

    for r in readings:
        assert r["map_mmhg"] == compute_map(r["sys"], r["dia"])


def test_parse_flags_artifacts() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "abpm.csv"
        csv_path.write_text(ABPM_CSV_WITH_ARTIFACT)

        readings = parse_abpm_csv(
            csv_path, session_id="TEST", now=datetime(2026, 4, 29, 15, 0)
        )

    assert len(readings) == 3
    assert readings[0]["is_artifact"] == 0
    assert readings[1]["is_artifact"] == 1  # sys=300
    assert readings[2]["is_artifact"] == 1  # narrow pulse pressure (120-115=5)


def test_parse_missing_required_columns_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "bad.csv"
        csv_path.write_text("Date,Time,HR\n2026-04-28,09:00,72\n")

        try:
            parse_abpm_csv(csv_path, session_id="TEST")
        except ValueError as e:
            assert "SYS" in str(e) or "DIA" in str(e)
            return
    raise AssertionError("expected ValueError for missing SYS/DIA columns")


# ---------------------------------------------------------------------------
# Schema + upsert
# ---------------------------------------------------------------------------
def test_init_tables_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_contec_abpm_tables(conn)
            init_contec_abpm_tables(conn)

            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "contec_abpm_readings" in tables
            assert "contec_abpm_sessions" in tables
        finally:
            conn.close()


def test_upsert_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_contec_abpm_tables(conn)
            readings = [
                {
                    "datetime": "2026-04-28T09:00:00",
                    "session_id": "S1",
                    "sequence_number": 1,
                    "sys": 125,
                    "dia": 78,
                    "bpm": 72,
                    "map_mmhg": 93.7,
                    "pulse_pressure": 47,
                    "error_code": 0,
                    "day_night": "day",
                    "is_artifact": 0,
                    "artifact_reason": None,
                    "source_file": "a.csv",
                }
            ]
            inserted1, skipped1 = upsert_abpm_readings(conn, readings)
            inserted2, skipped2 = upsert_abpm_readings(conn, readings)

            assert (inserted1, skipped1) == (1, 0)
            assert (inserted2, skipped2) == (0, 1)
        finally:
            conn.close()


def test_session_summary_counts() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_contec_abpm_tables(conn)
            readings = [
                {
                    "datetime": "2026-04-28T09:00:00",
                    "session_id": "S1",
                    "sequence_number": 1,
                    "sys": 125,
                    "dia": 78,
                    "bpm": 72,
                    "map_mmhg": 93.7,
                    "pulse_pressure": 47,
                    "error_code": 0,
                    "day_night": "day",
                    "is_artifact": 0,
                    "artifact_reason": None,
                    "source_file": "a.csv",
                },
                {
                    "datetime": "2026-04-29T03:00:00",
                    "session_id": "S1",
                    "sequence_number": 2,
                    "sys": 110,
                    "dia": 68,
                    "bpm": 58,
                    "map_mmhg": 82.0,
                    "pulse_pressure": 42,
                    "error_code": 0,
                    "day_night": "night",
                    "is_artifact": 0,
                    "artifact_reason": None,
                    "source_file": "a.csv",
                },
            ]
            record_session_summary(conn, "S1", readings, "a.csv")

            row = conn.execute(
                """
                SELECT reading_count, day_readings, night_readings
                FROM contec_abpm_sessions WHERE session_id='S1'
                """
            ).fetchone()
            assert row == (2, 1, 1)
        finally:
            conn.close()


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
