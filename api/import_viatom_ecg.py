#!/usr/bin/env python3
"""Viatom TH12 12-lead ECG Holter importer.

Ingests recording metadata and AI-detected event logs produced by Viatom's
PC software (or the TH12's on-device AI). Raw waveform binaries are kept as
files on disk (too large for SQLite); only path + recording metadata is stored
in the DB so analysis scripts can load waveforms on demand.

Format status (as of 2026-04-16):
    The Viatom TH12 PC software export format has NOT been validated against
    a real export. The parser is built from published-spec patterns used across
    the Viatom/Wellue device family (ER1, ER2, Duo, TH12). When the first real
    export arrives (~2026-04-28), validate and fix within 24h.

    Expected files per recording:
      - <name>.xml or <name>.json    session metadata + AI event list
      - <name>.csv                   event list (alternative flat format)
      - <name>.bin / .dat            raw waveform (stored as file, NOT parsed here)
      - <name>.pdf                   clinician report (not parsed)

Event vocabulary (Viatom AI categories):
    afib, pvc, pac, bradycardia, tachycardia, pause, st_depression,
    st_elevation, long_qt, noise, wide_qrs, heart_block, other

Usage:
    python api/import_viatom_ecg.py --init-only
    python api/import_viatom_ecg.py --events path/to/events.csv --recording-id R1
    python api/import_viatom_ecg.py --metadata path/to/session.xml --profile mitch
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Iterable, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ingest_common import (
    decode_bytes,
    find_any,
    find_col,
    float_or_none,
    get_cell,
    int_or_none,
    resolve_db_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("import_viatom_ecg")


# Known Viatom AI event vocabulary — extend here if real exports introduce new types.
KNOWN_EVENT_TYPES = frozenset(
    {
        "afib",
        "pac",
        "pvc",
        "bradycardia",
        "tachycardia",
        "pause",
        "st_depression",
        "st_elevation",
        "long_qt",
        "noise",
        "wide_qrs",
        "heart_block",
        "ischemia",
        "other",
    }
)


class EcgEvent(TypedDict):
    recording_id: str
    timestamp: str
    offset_seconds: float | None
    event_type: str
    severity: str | None
    lead: str | None
    duration_seconds: float | None
    hr_at_event: int | None
    confidence: float | None
    notes: str | None


class EcgRecording(TypedDict):
    recording_id: str
    start_datetime: str
    end_datetime: str | None
    duration_seconds: int | None
    sample_rate_hz: int | None
    leads_count: int | None
    device_serial: str | None
    firmware_version: str | None
    raw_file_path: str | None
    source_file: str


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def init_viatom_ecg_tables(conn: sqlite3.Connection) -> None:
    """Create Viatom ECG tables + indexes. Idempotent."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS viatom_ecg_recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id TEXT UNIQUE NOT NULL,
            start_datetime TEXT NOT NULL,
            end_datetime TEXT,
            duration_seconds INTEGER,
            sample_rate_hz INTEGER,
            leads_count INTEGER,
            device_serial TEXT,
            firmware_version TEXT,
            raw_file_path TEXT,
            source_file TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS viatom_ecg_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            offset_seconds REAL,
            event_type TEXT NOT NULL,
            severity TEXT,
            lead TEXT,
            duration_seconds REAL,
            hr_at_event INTEGER,
            confidence REAL,
            notes TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(recording_id, timestamp, event_type)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_viatom_ecg_events_timestamp
        ON viatom_ecg_events(timestamp)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_viatom_ecg_events_type
        ON viatom_ecg_events(event_type)
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y%m%dT%H%M%S",
)


def _parse_timestamp(ts: str) -> str:
    """Parse a Viatom timestamp string to ISO 8601."""
    ts = ts.strip()
    if not ts:
        raise ValueError("empty timestamp")
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(ts, fmt).isoformat()
        except ValueError:
            continue
    # ISO fromisoformat as a final fallback
    try:
        return datetime.fromisoformat(ts).isoformat()
    except ValueError as e:
        raise ValueError(f"Unrecognized timestamp format: {ts!r}") from e


def _normalize_event_type(raw: str) -> str:
    """Map Viatom's event strings (various casings/underscores) to the vocab."""
    s = raw.strip().lower().replace(" ", "_").replace("-", "_")
    # Common aliases from published Viatom docs
    aliases = {
        "atrial_fibrillation": "afib",
        "af": "afib",
        "atrial_flutter": "afib",
        "premature_ventricular_contraction": "pvc",
        "premature_atrial_contraction": "pac",
        "ventricular_tachycardia": "tachycardia",
        "sinus_tachycardia": "tachycardia",
        "sinus_bradycardia": "bradycardia",
        "st_elev": "st_elevation",
        "st_dep": "st_depression",
        "qt_prolongation": "long_qt",
    }
    return aliases.get(s, s if s in KNOWN_EVENT_TYPES else "other")


# ---------------------------------------------------------------------------
# CSV events parser
# ---------------------------------------------------------------------------
def parse_events_csv(
    csv_path: Path,
    recording_id: str,
    recording_start: datetime | None = None,
) -> list[EcgEvent]:
    """Parse a CSV of AI-detected ECG events.

    Accepts flexible column naming since Viatom's CSV header format is not yet
    validated. Header tokens that match (case-insensitive substring):

        - timestamp / time / datetime      → absolute event time
        - offset / seconds / elapsed       → seconds from recording start
        - event / type / arrhythmia        → event type
        - severity / level                 → severity label
        - lead / channel                   → ECG lead (I, II, V1..V6, aVR, aVL, aVF)
        - duration                         → event duration in seconds
        - hr / heart_rate / bpm            → HR at event
        - confidence / probability         → AI confidence 0-1
        - note / comment                   → free text

    If `timestamp` is missing but `offset_seconds` + `recording_start` exist,
    absolute timestamp is derived.
    """
    raw = csv_path.read_bytes()
    text = decode_bytes(raw)
    lines = text.splitlines()
    if not lines:
        return []

    # Pick separator
    first_nonempty = next((line for line in lines if line.strip()), "")
    sep = ";" if first_nonempty.count(";") > first_nonempty.count(",") else ","

    reader = csv.reader(lines, delimiter=sep)
    try:
        header = next(reader)
    except StopIteration:
        return []
    header_lower = [h.strip().lower() for h in header]

    idx_ts = find_col(header_lower, ("timestamp",), ("datetime",), ("time",))
    idx_offset = find_col(header_lower, ("offset",), ("elapsed",))
    idx_type = find_col(header_lower, ("event",), ("arrhythmia",), ("type",))
    idx_sev = find_col(header_lower, ("severity",), ("level",))
    idx_lead = find_col(header_lower, ("lead",), ("channel",))
    idx_dur = find_col(header_lower, ("duration",))
    idx_hr = find_col(header_lower, ("bpm",), ("heart_rate",), ("hr",))
    idx_conf = find_col(header_lower, ("confidence",), ("probability",))
    idx_note = find_col(header_lower, ("note",), ("comment",))

    if idx_type is None:
        raise ValueError(f"No event-type column found in CSV header: {header}")
    if idx_ts is None and idx_offset is None:
        raise ValueError(
            f"No timestamp/offset column in CSV header: {header}. "
            "Need either absolute time or offset-from-start + --recording-start."
        )

    events: list[EcgEvent] = []
    for row in reader:
        if not row or idx_type >= len(row):
            continue
        event_type_raw = row[idx_type].strip()
        if not event_type_raw:
            continue

        ts_iso: str | None = None
        offset_s: float | None = None

        if idx_offset is not None and idx_offset < len(row) and row[idx_offset].strip():
            try:
                offset_s = float(row[idx_offset].replace(",", "."))
            except ValueError:
                offset_s = None

        if idx_ts is not None and idx_ts < len(row) and row[idx_ts].strip():
            try:
                ts_iso = _parse_timestamp(row[idx_ts])
            except ValueError as e:
                log.warning("Skipping row, unparseable timestamp: %s", e)
                continue
        elif offset_s is not None and recording_start is not None:
            from datetime import timedelta as _td

            ts_iso = (recording_start + _td(seconds=offset_s)).isoformat()
        else:
            continue

        events.append(
            {
                "recording_id": recording_id,
                "timestamp": ts_iso,
                "offset_seconds": offset_s,
                "event_type": _normalize_event_type(event_type_raw),
                "severity": get_cell(row, idx_sev),
                "lead": get_cell(row, idx_lead),
                "duration_seconds": float_or_none(get_cell(row, idx_dur)),
                "hr_at_event": int_or_none(get_cell(row, idx_hr)),
                "confidence": float_or_none(get_cell(row, idx_conf)),
                "notes": get_cell(row, idx_note),
            }
        )

    return events


# ---------------------------------------------------------------------------
# XML / JSON session-metadata parser
# ---------------------------------------------------------------------------
def parse_recording_xml(xml_path: Path) -> EcgRecording:
    """Parse a Viatom session metadata XML.

    The exact tag layout is not yet validated against a real TH12 export. This
    parser uses lenient tag lookup: it descends the tree searching for known
    tag-name patterns, case-insensitive. If the first real file uses a totally
    different structure, override this function.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _find_text(*name_parts: str) -> str | None:
        for elem in root.iter():
            tag = elem.tag.lower()
            if all(p in tag for p in name_parts):
                return (elem.text or "").strip() or None
        return None

    start = _find_text("start") or _find_text("begin")
    end = _find_text("end") or _find_text("stop")
    duration = _find_text("duration")
    rate = _find_text("sample", "rate") or _find_text("frequency")
    leads = _find_text("lead", "count") or _find_text("channels")
    serial = _find_text("serial")
    firmware = _find_text("firmware") or _find_text("version")

    if not start:
        raise ValueError(
            f"No start-time element found in {xml_path}. "
            "XML structure may differ from expected Viatom format."
        )

    return {
        "recording_id": xml_path.stem,
        "start_datetime": _parse_timestamp(start),
        "end_datetime": _parse_timestamp(end) if end else None,
        "duration_seconds": int(float(duration)) if duration else None,
        "sample_rate_hz": int(float(rate)) if rate else None,
        "leads_count": int(float(leads)) if leads else None,
        "device_serial": serial,
        "firmware_version": firmware,
        "raw_file_path": None,
        "source_file": str(xml_path),
    }


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------
def upsert_recording(conn: sqlite3.Connection, rec: EcgRecording) -> int:
    """Insert or update recording metadata by recording_id."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO viatom_ecg_recordings (
            recording_id, start_datetime, end_datetime, duration_seconds,
            sample_rate_hz, leads_count, device_serial, firmware_version,
            raw_file_path, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(recording_id) DO UPDATE SET
            start_datetime = excluded.start_datetime,
            end_datetime = excluded.end_datetime,
            duration_seconds = excluded.duration_seconds,
            sample_rate_hz = excluded.sample_rate_hz,
            leads_count = excluded.leads_count,
            device_serial = excluded.device_serial,
            firmware_version = excluded.firmware_version,
            raw_file_path = excluded.raw_file_path,
            source_file = excluded.source_file
        """,
        (
            rec["recording_id"],
            rec["start_datetime"],
            rec["end_datetime"],
            rec["duration_seconds"],
            rec["sample_rate_hz"],
            rec["leads_count"],
            rec["device_serial"],
            rec["firmware_version"],
            rec["raw_file_path"],
            rec["source_file"],
        ),
    )
    conn.commit()
    return cur.lastrowid


def upsert_events(
    conn: sqlite3.Connection, events: Iterable[EcgEvent]
) -> tuple[int, int]:
    """Insert events, ignoring duplicates on (recording_id, timestamp, event_type)."""
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for e in events:
        cur.execute(
            """
            INSERT OR IGNORE INTO viatom_ecg_events (
                recording_id, timestamp, offset_seconds, event_type, severity,
                lead, duration_seconds, hr_at_event, confidence, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["recording_id"],
                e["timestamp"],
                e["offset_seconds"],
                e["event_type"],
                e["severity"],
                e["lead"],
                e["duration_seconds"],
                e["hr_at_event"],
                e["confidence"],
                e["notes"],
            ),
        )
        if cur.rowcount:
            inserted += 1
        else:
            skipped += 1
    conn.commit()
    return inserted, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create tables without importing data",
    )
    parser.add_argument("--metadata", help="Path to session XML (or JSON)")
    parser.add_argument("--events", help="Path to AI events CSV")
    parser.add_argument(
        "--recording-id",
        help="Recording ID (required when --events is given without --metadata)",
    )
    parser.add_argument(
        "--recording-start",
        help="ISO 8601 start time (required when events CSV has only offsets)",
    )
    parser.add_argument(
        "--raw-file",
        help="Path to raw waveform .bin/.dat (recorded in metadata, NOT parsed)",
    )
    parser.add_argument("--db", help="Database path override")
    parser.add_argument("--profile", "-p", help="Patient profile name")

    args = parser.parse_args()

    db_path = resolve_db_path(args.db, args.profile)
    conn = sqlite3.connect(db_path)
    try:
        init_viatom_ecg_tables(conn)
        if args.init_only:
            print(f"Initialized Viatom ECG tables in {db_path}")
            return 0

        recording_start: datetime | None = None
        recording_id: str | None = args.recording_id

        if args.metadata:
            metadata_path = Path(args.metadata)
            if not metadata_path.exists():
                print(f"Error: metadata file not found: {metadata_path}", file=sys.stderr)
                return 1
            rec = parse_recording_xml(metadata_path)
            if args.raw_file:
                rec["raw_file_path"] = str(Path(args.raw_file).resolve())
            upsert_recording(conn, rec)
            recording_id = rec["recording_id"]
            recording_start = datetime.fromisoformat(rec["start_datetime"])
            print(f"Imported recording metadata: {recording_id}")

        if args.events:
            if not recording_id:
                print(
                    "Error: --recording-id is required when importing events "
                    "without --metadata",
                    file=sys.stderr,
                )
                return 1
            events_path = Path(args.events)
            if not events_path.exists():
                print(f"Error: events CSV not found: {events_path}", file=sys.stderr)
                return 1
            if args.recording_start and recording_start is None:
                recording_start = datetime.fromisoformat(args.recording_start)

            events = parse_events_csv(
                events_path, recording_id, recording_start=recording_start
            )
            inserted, skipped = upsert_events(conn, events)
            print(
                f"Parsed {len(events)} events, inserted {inserted}, "
                f"skipped {skipped} duplicates"
            )
    finally:
        conn.close()

    print(f"Database: {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
