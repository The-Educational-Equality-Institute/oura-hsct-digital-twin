"""Tests for the Checkme O2 Max importer.

Synthetic fixtures exercise likely column layouts for Viatom/Wellue Checkme
PC software CSV exports. Real format validation happens within 24h after the
first real file arrives.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))
from import_checkme_spo2 import (
    _parse_time_field,
    detect_spo2_artifact,
    init_checkme_spo2_tables,
    parse_checkme_csv,
    record_session_summary,
    upsert_spo2_readings,
)


CHECKME_CSV_ABSOLUTE = dedent(
    """\
    Timestamp,SpO2,PR,PI,Motion
    2026-04-28T22:00:00,97,65,2.1,0
    2026-04-28T22:00:01,97,64,2.1,0
    2026-04-28T22:00:02,96,64,2.0,0
    2026-04-28T22:00:03,88,66,1.8,1
    """
)

CHECKME_CSV_RELATIVE = dedent(
    """\
    Time,SpO2,Pulse Rate,Perfusion Index
    00:00:00,97,65,2.1
    00:00:01,97,64,2.1
    00:00:02,96,64,2.0
    """
)

CHECKME_CSV_SEMICOLON_COMMA = dedent(
    """\
    Timestamp;SpO2;PR;PI
    2026-04-28T22:00:00;97,5;65;2,1
    """
)


# ---------------------------------------------------------------------------
# Unit-level
# ---------------------------------------------------------------------------
def test_parse_time_field_absolute() -> None:
    dt = _parse_time_field("2026-04-28T22:00:00")
    assert dt == datetime(2026, 4, 28, 22, 0, 0)


def test_parse_time_field_relative_with_anchor() -> None:
    start = datetime(2026, 4, 28, 22, 0, 0)
    dt = _parse_time_field("00:00:15", session_start=start)
    assert dt == datetime(2026, 4, 28, 22, 0, 15)


def test_parse_time_field_relative_without_anchor_returns_none() -> None:
    dt = _parse_time_field("00:00:15", session_start=None)
    assert dt is None


def test_parse_time_field_empty_returns_none() -> None:
    assert _parse_time_field("") is None


def test_detect_artifact_clean() -> None:
    artifact, reason = detect_spo2_artifact(97, 65, 0)
    assert artifact is False
    assert reason is None


def test_detect_artifact_missing_spo2() -> None:
    artifact, reason = detect_spo2_artifact(None, 65, 0)
    assert artifact is True
    assert "no_spo2" in reason


def test_detect_artifact_spo2_below_floor() -> None:
    artifact, reason = detect_spo2_artifact(30, 65, 0)
    assert artifact is True
    assert "spo2_out_of_range" in reason


def test_detect_artifact_motion_flag() -> None:
    artifact, reason = detect_spo2_artifact(97, 65, 1)
    assert artifact is True
    assert "motion" in reason


def test_detect_artifact_pulse_rate_out_of_range() -> None:
    artifact, reason = detect_spo2_artifact(97, 15, 0)
    assert artifact is True
    assert "pulse_rate" in reason


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def test_parse_absolute_timestamps() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "checkme.csv"
        csv_path.write_text(CHECKME_CSV_ABSOLUTE)

        readings = parse_checkme_csv(csv_path, session_id="TEST")

    assert len(readings) == 4
    assert readings[0]["spo2"] == 97
    assert readings[0]["pulse_rate"] == 65
    assert readings[0]["perfusion_index"] == 2.1
    assert readings[0]["motion_flag"] == 0
    assert readings[0]["is_artifact"] == 0
    # Row 4 has motion_flag=1 → artifact
    assert readings[3]["is_artifact"] == 1
    assert "motion" in readings[3]["artifact_reason"]


def test_parse_relative_timestamps_with_anchor() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "checkme.csv"
        csv_path.write_text(CHECKME_CSV_RELATIVE)

        start = datetime(2026, 4, 28, 22, 0, 0)
        readings = parse_checkme_csv(
            csv_path, session_id="TEST", session_start=start
        )

    assert len(readings) == 3
    assert readings[0]["timestamp"] == "2026-04-28T22:00:00"
    assert readings[2]["timestamp"] == "2026-04-28T22:00:02"


def test_parse_semicolon_comma_decimal() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "checkme.csv"
        csv_path.write_text(CHECKME_CSV_SEMICOLON_COMMA)

        readings = parse_checkme_csv(csv_path, session_id="TEST")

    assert len(readings) == 1
    assert readings[0]["spo2"] == 97.5
    assert readings[0]["perfusion_index"] == 2.1


def test_parse_missing_spo2_column_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "bad.csv"
        csv_path.write_text("Timestamp,PR\n2026-04-28T22:00:00,65\n")

        try:
            parse_checkme_csv(csv_path, session_id="TEST")
        except ValueError as e:
            assert "SpO2" in str(e) or "spo2" in str(e).lower()
            return
    raise AssertionError("expected ValueError for missing SpO2 column")


# ---------------------------------------------------------------------------
# Schema + upsert
# ---------------------------------------------------------------------------
def test_init_tables_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_checkme_spo2_tables(conn)
            init_checkme_spo2_tables(conn)

            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "checkme_spo2_continuous" in tables
            assert "checkme_spo2_sessions" in tables
        finally:
            conn.close()


def test_upsert_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_checkme_spo2_tables(conn)
            readings = [
                {
                    "timestamp": "2026-04-28T22:00:00",
                    "session_id": "S1",
                    "spo2": 97.0,
                    "pulse_rate": 65,
                    "perfusion_index": 2.1,
                    "motion_flag": 0,
                    "is_artifact": 0,
                    "artifact_reason": None,
                    "source_file": "a.csv",
                }
            ]
            inserted1, skipped1 = upsert_spo2_readings(conn, readings)
            inserted2, skipped2 = upsert_spo2_readings(conn, readings)

            assert (inserted1, skipped1) == (1, 0)
            assert (inserted2, skipped2) == (0, 1)
        finally:
            conn.close()


def test_session_summary_stats() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_checkme_spo2_tables(conn)
            readings = [
                {
                    "timestamp": "2026-04-28T22:00:00",
                    "session_id": "S1",
                    "spo2": 97.0,
                    "pulse_rate": 65,
                    "perfusion_index": 2.1,
                    "motion_flag": 0,
                    "is_artifact": 0,
                    "artifact_reason": None,
                    "source_file": "a.csv",
                },
                {
                    "timestamp": "2026-04-28T22:00:01",
                    "session_id": "S1",
                    "spo2": 88.0,
                    "pulse_rate": 66,
                    "perfusion_index": 1.5,
                    "motion_flag": 0,
                    "is_artifact": 0,
                    "artifact_reason": None,
                    "source_file": "a.csv",
                },
                {
                    "timestamp": "2026-04-28T22:00:02",
                    "session_id": "S1",
                    "spo2": 40.0,  # will be artifact-flagged elsewhere; here pre-flagged
                    "pulse_rate": 65,
                    "perfusion_index": 1.0,
                    "motion_flag": 1,
                    "is_artifact": 1,
                    "artifact_reason": "motion_artifact",
                    "source_file": "a.csv",
                },
            ]
            record_session_summary(conn, "S1", readings, "a.csv")

            row = conn.execute(
                """
                SELECT sample_count, artifact_count, mean_spo2, min_spo2,
                       time_below_90_pct
                FROM checkme_spo2_sessions WHERE session_id='S1'
                """
            ).fetchone()
            sample_count, artifact_count, mean_spo2, min_spo2, time_below_90 = row

            assert sample_count == 3
            assert artifact_count == 1
            # Only clean readings (97, 88) contribute to summaries
            assert abs(mean_spo2 - 92.5) < 0.001
            assert min_spo2 == 88.0
            # 88 < 90, 97 >= 90 → 1/2 = 50%
            assert abs(time_below_90 - 50.0) < 0.001
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
