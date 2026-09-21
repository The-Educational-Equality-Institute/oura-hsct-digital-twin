"""Tests for the Viatom TH12 ECG importer.

Uses synthetic fixtures to exercise parser paths. The real TH12 export format
is not yet validated; these tests guard against regressions in the flexible-
detection logic so the 24-hour fix window after first real export stays narrow.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))
from import_viatom_ecg import (
    KNOWN_EVENT_TYPES,
    _normalize_event_type,
    _parse_timestamp,
    init_viatom_ecg_tables,
    parse_events_csv,
    parse_recording_xml,
    upsert_events,
    upsert_recording,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
EVENTS_CSV_ABSOLUTE_TIME = dedent(
    """\
    Timestamp,Event Type,Severity,Lead,Duration,HR,Confidence,Notes
    2026-04-28T13:15:22,PVC,mild,V2,0.08,78,0.92,isolated
    2026-04-28T13:20:45,afib,moderate,II,45.2,142,0.88,sustained
    2026-04-28T13:45:00,ST_depression,severe,V5,120.5,96,0.76,
    """
)

EVENTS_CSV_OFFSET_ONLY = dedent(
    """\
    Offset Seconds,Event,Lead,HR
    15.5,PVC,V1,88
    120.0,tachycardia,II,130
    """
)

EVENTS_CSV_SEMICOLON_COMMA_DECIMAL = dedent(
    """\
    Timestamp;Event;Confidence
    2026-04-28T14:00:00;afib;0,95
    """
)

RECORDING_XML = dedent(
    """\
    <?xml version="1.0"?>
    <EcgRecording>
      <SessionInfo>
        <StartTime>2026-04-28T13:00:00</StartTime>
        <EndTime>2026-04-29T13:00:00</EndTime>
        <Duration>86400</Duration>
        <SampleRate>250</SampleRate>
        <Channels>12</Channels>
      </SessionInfo>
      <Device>
        <SerialNumber>TH12-XYZ-001</SerialNumber>
        <FirmwareVersion>2.3.1</FirmwareVersion>
      </Device>
    </EcgRecording>
    """
)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
def test_parse_timestamp_iso() -> None:
    assert _parse_timestamp("2026-04-28T13:15:22") == "2026-04-28T13:15:22"


def test_parse_timestamp_space_separator() -> None:
    assert _parse_timestamp("2026-04-28 13:15:22") == "2026-04-28T13:15:22"


def test_parse_timestamp_with_microseconds() -> None:
    # Viatom timestamps sometimes include fractional seconds
    iso = _parse_timestamp("2026-04-28T13:15:22.500")
    assert iso.startswith("2026-04-28T13:15:22")


def test_parse_timestamp_empty_raises() -> None:
    try:
        _parse_timestamp("")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty timestamp")


# ---------------------------------------------------------------------------
# Event-type normalization (maps Viatom aliases to our vocabulary)
# ---------------------------------------------------------------------------
def test_normalize_afib_aliases() -> None:
    assert _normalize_event_type("AF") == "afib"
    assert _normalize_event_type("Atrial Fibrillation") == "afib"
    assert _normalize_event_type("atrial_flutter") == "afib"


def test_normalize_pvc_alias() -> None:
    assert _normalize_event_type("Premature Ventricular Contraction") == "pvc"


def test_normalize_st_abbreviations() -> None:
    assert _normalize_event_type("ST_elev") == "st_elevation"
    assert _normalize_event_type("ST_dep") == "st_depression"


def test_unknown_event_mapped_to_other() -> None:
    assert _normalize_event_type("some_weird_new_event") == "other"


def test_normalize_preserves_known_types() -> None:
    # All documented event types should round-trip unchanged
    for known in KNOWN_EVENT_TYPES:
        assert _normalize_event_type(known) == known


# ---------------------------------------------------------------------------
# CSV event parsing
# ---------------------------------------------------------------------------
def test_parse_events_absolute_time() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "events.csv"
        csv_path.write_text(EVENTS_CSV_ABSOLUTE_TIME)

        events = parse_events_csv(csv_path, recording_id="R1")

    assert len(events) == 3
    assert events[0]["event_type"] == "pvc"
    assert events[0]["severity"] == "mild"
    assert events[0]["lead"] == "V2"
    assert events[0]["confidence"] == 0.92
    assert events[1]["event_type"] == "afib"
    assert events[2]["event_type"] == "st_depression"


def test_parse_events_offset_only() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "events.csv"
        csv_path.write_text(EVENTS_CSV_OFFSET_ONLY)

        start = datetime(2026, 4, 28, 13, 0, 0)
        events = parse_events_csv(csv_path, recording_id="R2", recording_start=start)

    assert len(events) == 2
    assert events[0]["offset_seconds"] == 15.5
    # 13:00:00 + 15.5 s = 13:00:15.5 → ISO "2026-04-28T13:00:15.500000"
    assert events[0]["timestamp"].startswith("2026-04-28T13:00:15")


def test_parse_events_offset_without_start_skips() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "events.csv"
        csv_path.write_text(EVENTS_CSV_OFFSET_ONLY)

        events = parse_events_csv(csv_path, recording_id="R3", recording_start=None)

    # Without a recording start we can't derive absolute timestamps; rows skipped
    assert events == []


def test_parse_events_semicolon_comma_decimal() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "events.csv"
        csv_path.write_text(EVENTS_CSV_SEMICOLON_COMMA_DECIMAL)

        events = parse_events_csv(csv_path, recording_id="R4")

    assert len(events) == 1
    assert events[0]["confidence"] == 0.95


def test_parse_events_missing_type_column_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "events.csv"
        csv_path.write_text("Timestamp,HR\n2026-04-28T13:00:00,75\n")

        try:
            parse_events_csv(csv_path, recording_id="R5")
        except ValueError as e:
            assert "event-type" in str(e) or "event type" in str(e).lower()
            return
    raise AssertionError("expected ValueError for missing event-type column")


# ---------------------------------------------------------------------------
# XML metadata parsing
# ---------------------------------------------------------------------------
def test_parse_recording_xml() -> None:
    with tempfile.TemporaryDirectory() as td:
        xml_path = Path(td) / "session_R100.xml"
        xml_path.write_text(RECORDING_XML)

        rec = parse_recording_xml(xml_path)

    assert rec["recording_id"] == "session_R100"
    assert rec["start_datetime"] == "2026-04-28T13:00:00"
    assert rec["end_datetime"] == "2026-04-29T13:00:00"
    assert rec["duration_seconds"] == 86400
    assert rec["sample_rate_hz"] == 250
    assert rec["leads_count"] == 12
    assert rec["device_serial"] == "TH12-XYZ-001"
    assert rec["firmware_version"] == "2.3.1"


def test_parse_recording_xml_missing_start_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        xml_path = Path(td) / "bad.xml"
        xml_path.write_text("<root><foo>bar</foo></root>")

        try:
            parse_recording_xml(xml_path)
        except ValueError as e:
            assert "start" in str(e).lower()
            return
    raise AssertionError("expected ValueError for missing start element")


# ---------------------------------------------------------------------------
# Schema + upsert
# ---------------------------------------------------------------------------
def test_init_tables_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_viatom_ecg_tables(conn)
            init_viatom_ecg_tables(conn)

            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "viatom_ecg_recordings" in tables
            assert "viatom_ecg_events" in tables
        finally:
            conn.close()


def test_upsert_recording_updates_on_conflict() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_viatom_ecg_tables(conn)
            rec = {
                "recording_id": "R1",
                "start_datetime": "2026-04-28T13:00:00",
                "end_datetime": None,
                "duration_seconds": 3600,
                "sample_rate_hz": 250,
                "leads_count": 12,
                "device_serial": "SN-1",
                "firmware_version": "1.0",
                "raw_file_path": None,
                "source_file": "a.xml",
            }
            upsert_recording(conn, rec)

            # Update with new duration
            rec["duration_seconds"] = 7200
            upsert_recording(conn, rec)

            row = conn.execute(
                "SELECT COUNT(*), MAX(duration_seconds) FROM viatom_ecg_recordings"
            ).fetchone()
            assert row[0] == 1
            assert row[1] == 7200
        finally:
            conn.close()


def test_upsert_events_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        conn = sqlite3.connect(str(Path(td) / "test.db"))
        try:
            init_viatom_ecg_tables(conn)
            events = [
                {
                    "recording_id": "R1",
                    "timestamp": "2026-04-28T13:15:22",
                    "offset_seconds": 15.5,
                    "event_type": "pvc",
                    "severity": "mild",
                    "lead": "V2",
                    "duration_seconds": 0.08,
                    "hr_at_event": 78,
                    "confidence": 0.92,
                    "notes": None,
                }
            ]

            inserted1, skipped1 = upsert_events(conn, events)
            inserted2, skipped2 = upsert_events(conn, events)

            assert (inserted1, skipped1) == (1, 0)
            assert (inserted2, skipped2) == (0, 1)
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
