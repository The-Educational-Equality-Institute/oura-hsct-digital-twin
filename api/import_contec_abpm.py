#!/usr/bin/env python3
"""Contec ABPM50 ambulatory blood pressure importer.

24-hour ABPM readings (~50-100 per session) from the Contec ABPM50 PC software
CSV export. Data structure mirrors OMRON M7 (api/import_omron.py) but keeps a
separate device-namespaced table so queries can distinguish spot-check vs
ambulatory measurements without inference.

Format status (as of 2026-04-16):
    Contec ABPM50 CSV export is NOT yet validated against a real file. The
    parser is built from general ABPM CSV conventions and Contec product
    documentation. When the first real export arrives (~2026-04-28), validate
    against the actual column structure and fix within 24h.

    Expected columns (Contec-style):
      No. / #         sequence number
      Date            YYYY-MM-DD or DD-MM-YYYY
      Time            HH:MM or HH:MM:SS
      (or combined "Date Time" / "Datetime" column)
      SYS / Systolic  mmHg
      DIA / Diastolic mmHg
      MAP             mmHg (if present; otherwise computed)
      HR / Pulse      bpm
      Code / Error    0 = OK, other = error code
      Position        optional: sitting/lying/standing

Enrichment at ingest (parallels OMRON):
  - MAP (computed if not in export)
  - Pulse pressure
  - Day/night classification (ESH convention: day 07:00-21:59, night 22:00-06:59)
  - Artifact flag (implausible values, narrow pulse pressure)

Usage:
    python api/import_contec_abpm.py --init-only
    python api/import_contec_abpm.py --csv path/to/abpm_export.csv
    python api/import_contec_abpm.py --csv export.csv --session-id HENRIK-2026-04-28 --profile henrik
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from dataclasses import dataclass
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
log = logging.getLogger("import_contec_abpm")


# ESH ambulatory BP conventions (European Society of Hypertension)
DAY_START_HOUR = 7
NIGHT_START_HOUR = 22

# Physiologic plausibility bands (parallels OMRON)
SYS_MIN, SYS_MAX = 60, 260
DIA_MIN, DIA_MAX = 30, 200
BPM_MIN, BPM_MAX = 25, 220
MIN_PULSE_PRESSURE = 20


class AbpmReading(TypedDict):
    datetime: str
    session_id: str
    sequence_number: int | None
    sys: int
    dia: int
    bpm: int | None
    map_mmhg: float
    pulse_pressure: int
    error_code: int | None
    day_night: str
    is_artifact: int
    artifact_reason: str | None
    source_file: str | None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def init_contec_abpm_tables(conn: sqlite3.Connection) -> None:
    """Create Contec ABPM tables + indexes. Idempotent."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS contec_abpm_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            session_id TEXT NOT NULL,
            sequence_number INTEGER,
            sys INTEGER NOT NULL,
            dia INTEGER NOT NULL,
            bpm INTEGER,
            map_mmhg REAL NOT NULL,
            pulse_pressure INTEGER NOT NULL,
            error_code INTEGER,
            day_night TEXT,
            is_artifact INTEGER DEFAULT 0,
            artifact_reason TEXT,
            source_file TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, datetime)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS contec_abpm_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            start_datetime TEXT NOT NULL,
            end_datetime TEXT,
            reading_count INTEGER,
            artifact_count INTEGER,
            day_readings INTEGER,
            night_readings INTEGER,
            device_serial TEXT,
            source_file TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_contec_abpm_datetime
        ON contec_abpm_readings(datetime)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_contec_abpm_session
        ON contec_abpm_readings(session_id)
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%Y%m%d",
)
_TIME_FORMATS = (
    "%H:%M:%S",
    "%H:%M",
)
_DATETIME_FORMATS = tuple(
    f"{d} {t}" for d in _DATE_FORMATS for t in _TIME_FORMATS
) + (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
)


def _parse_datetime(date_str: str, time_str: str | None = None) -> datetime:
    """Parse Contec date+time into a datetime.

    Accepts either a combined datetime string or separate date + time.
    """
    if time_str:
        combined = f"{date_str.strip()} {time_str.strip()}"
        for fmt in _DATETIME_FORMATS:
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                continue
    # Single combined field
    s = date_str.strip()
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Final ISO fallback
    try:
        return datetime.fromisoformat(s)
    except ValueError as e:
        raise ValueError(f"Unrecognized date/time: {date_str!r} {time_str!r}") from e


def classify_day_night(dt: datetime) -> str:
    """ESH convention: day 07:00-21:59, night 22:00-06:59."""
    return "day" if DAY_START_HOUR <= dt.hour < NIGHT_START_HOUR else "night"


def compute_map(sys_v: int, dia_v: int) -> float:
    """Mean arterial pressure = DIA + (SYS - DIA) / 3 (standard clinical formula)."""
    return round(dia_v + (sys_v - dia_v) / 3.0, 1)


def detect_artifact(
    sys_v: int,
    dia_v: int,
    bpm: int | None,
    dt: datetime,
    now: datetime,
) -> tuple[bool, str | None]:
    """Return (is_artifact, reason). None reason means clean reading."""
    if dt.year > now.year + 1:
        return True, f"future_timestamp:{dt.year}"
    if not SYS_MIN <= sys_v <= SYS_MAX:
        return True, f"sys_out_of_range:{sys_v}"
    if not DIA_MIN <= dia_v <= DIA_MAX:
        return True, f"dia_out_of_range:{dia_v}"
    if bpm is not None and not BPM_MIN <= bpm <= BPM_MAX:
        return True, f"bpm_out_of_range:{bpm}"
    if sys_v - dia_v < MIN_PULSE_PRESSURE:
        return True, f"narrow_pulse_pressure:{sys_v - dia_v}"
    return False, None


def parse_abpm_csv(
    csv_path: Path,
    session_id: str,
    now: datetime | None = None,
) -> list[AbpmReading]:
    """Parse a Contec ABPM CSV export into AbpmReading records.

    Flexible column detection — substring matching on header names. Accepts
    separate Date + Time columns OR a combined Datetime column. Accepts MAP
    from export OR computes it.
    """
    now = now or datetime.now()

    raw = csv_path.read_bytes()
    text = decode_bytes(raw)
    lines = text.splitlines()
    if not lines:
        return []

    sample = next((ln for ln in lines if ln.strip()), "")
    sep = ";" if sample.count(";") > sample.count(",") else ","

    reader = csv.reader(lines, delimiter=sep)
    try:
        header = next(reader)
    except StopIteration:
        return []
    header_lower = [h.strip().lower() for h in header]

    idx_no = find_col(header_lower, ("no",), ("number",), ("#",), ("seq",))
    idx_datetime = find_col(
        header_lower, ("datetime",), ("date time",), ("date_time",)
    )
    idx_date = find_col(header_lower, ("date",))
    idx_time = find_col(header_lower, ("time",))
    idx_sys = find_col(header_lower, ("sys",), ("systolic",))
    idx_dia = find_col(header_lower, ("dia",), ("diastolic",))
    idx_map = find_col(header_lower, ("map",), ("mean",))
    idx_bpm = find_col(header_lower, ("bpm",), ("pulse",), ("hr",), ("heart",))
    idx_code = find_col(header_lower, ("code",), ("error",), ("status",))

    # Avoid conflict: "datetime" matches the "date" substring search too; we
    # prefer the combined field if it exists, so only use separate date/time
    # when there is no combined field.
    if idx_datetime is not None:
        idx_date = idx_datetime
        idx_time = None

    if idx_sys is None or idx_dia is None:
        raise ValueError(
            f"Missing required SYS/DIA columns in ABPM CSV header: {header}"
        )
    if idx_date is None:
        raise ValueError(
            f"Missing Date/Datetime column in ABPM CSV header: {header}"
        )

    readings: list[AbpmReading] = []
    for row in reader:
        if not row or idx_sys >= len(row):
            continue
        try:
            sys_v = int(float(row[idx_sys].replace(",", ".")))
            dia_v = int(float(row[idx_dia].replace(",", ".")))
        except (ValueError, IndexError):
            continue

        try:
            dt = _parse_datetime(
                row[idx_date],
                row[idx_time] if idx_time is not None and idx_time < len(row) else None,
            )
        except ValueError as e:
            log.warning("Skipping row with unparseable datetime: %s", e)
            continue

        bpm = int_or_none(get_cell(row, idx_bpm))
        code = int_or_none(get_cell(row, idx_code))
        seq = int_or_none(get_cell(row, idx_no))

        map_val = float_or_none(get_cell(row, idx_map))
        if map_val is None:
            map_val = compute_map(sys_v, dia_v)

        artifact, reason = detect_artifact(sys_v, dia_v, bpm, dt, now)

        readings.append(
            {
                "datetime": dt.isoformat(),
                "session_id": session_id,
                "sequence_number": seq,
                "sys": sys_v,
                "dia": dia_v,
                "bpm": bpm,
                "map_mmhg": map_val,
                "pulse_pressure": sys_v - dia_v,
                "error_code": code,
                "day_night": classify_day_night(dt),
                "is_artifact": 1 if artifact else 0,
                "artifact_reason": reason,
                "source_file": csv_path.name,
            }
        )

    return readings


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------
def upsert_abpm_readings(
    conn: sqlite3.Connection, readings: Iterable[AbpmReading]
) -> tuple[int, int]:
    """Insert readings, ignoring duplicates on (session_id, datetime)."""
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for r in readings:
        cur.execute(
            """
            INSERT OR IGNORE INTO contec_abpm_readings (
                datetime, session_id, sequence_number, sys, dia, bpm,
                map_mmhg, pulse_pressure, error_code, day_night,
                is_artifact, artifact_reason, source_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["datetime"],
                r["session_id"],
                r["sequence_number"],
                r["sys"],
                r["dia"],
                r["bpm"],
                r["map_mmhg"],
                r["pulse_pressure"],
                r["error_code"],
                r["day_night"],
                r["is_artifact"],
                r["artifact_reason"],
                r["source_file"],
            ),
        )
        if cur.rowcount:
            inserted += 1
        else:
            skipped += 1
    conn.commit()
    return inserted, skipped


def record_session_summary(
    conn: sqlite3.Connection,
    session_id: str,
    readings: list[AbpmReading],
    source_file: str,
    device_serial: str | None = None,
) -> None:
    """Compute and upsert session-level summary for a batch of readings."""
    if not readings:
        return
    starts = [r["datetime"] for r in readings]
    artifacts = sum(r["is_artifact"] for r in readings)
    day = sum(1 for r in readings if r["day_night"] == "day")
    night = sum(1 for r in readings if r["day_night"] == "night")

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO contec_abpm_sessions (
            session_id, start_datetime, end_datetime, reading_count,
            artifact_count, day_readings, night_readings, device_serial, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            start_datetime = MIN(excluded.start_datetime, start_datetime),
            end_datetime = MAX(excluded.end_datetime, end_datetime),
            reading_count = excluded.reading_count,
            artifact_count = excluded.artifact_count,
            day_readings = excluded.day_readings,
            night_readings = excluded.night_readings,
            source_file = excluded.source_file
        """,
        (
            session_id,
            min(starts),
            max(starts),
            len(readings),
            artifacts,
            day,
            night,
            device_serial,
            source_file,
        ),
    )
    conn.commit()


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
    parser.add_argument("--csv", help="Path to Contec ABPM CSV export")
    parser.add_argument(
        "--session-id",
        help="Session identifier (default: derived from CSV start time)",
    )
    parser.add_argument("--device-serial", help="Device serial number (optional)")
    parser.add_argument("--db", help="Database path override")
    parser.add_argument("--profile", "-p", help="Patient profile name")

    args = parser.parse_args()

    db_path = resolve_db_path(args.db, args.profile)
    conn = sqlite3.connect(db_path)
    try:
        init_contec_abpm_tables(conn)
        if args.init_only:
            print(f"Initialized Contec ABPM tables in {db_path}")
            return 0

        if not args.csv:
            parser.error("either --csv or --init-only is required")

        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
            return 1

        session_id = args.session_id or f"abpm-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        readings = parse_abpm_csv(csv_path, session_id=session_id)
        if not readings:
            print(f"No valid readings parsed from {csv_path}")
            return 1

        if not args.session_id:
            # Derive a nicer session id from the actual start time
            start_iso = min(r["datetime"] for r in readings)
            session_id = f"abpm-{start_iso.replace(':', '').replace('-', '')[:15]}"
            for r in readings:
                r["session_id"] = session_id

        inserted, skipped = upsert_abpm_readings(conn, readings)
        record_session_summary(
            conn,
            session_id=session_id,
            readings=readings,
            source_file=csv_path.name,
            device_serial=args.device_serial,
        )
    finally:
        conn.close()

    print(f"Parsed {len(readings)} ABPM readings from {csv_path}")
    print(f"Session: {session_id}")
    print(f"Inserted: {inserted}  |  Skipped (duplicates): {skipped}")
    print(f"Database: {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
