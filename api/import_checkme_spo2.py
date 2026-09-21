#!/usr/bin/env python3
"""Checkme O2 Max (Pro) continuous pulse oximeter importer.

Wrist-worn continuous SpO2 + pulse rate + perfusion index, typically 1 Hz
resolution over an overnight ~8-hour session (~28,800 samples). Imported from
the Checkme PC Evaluation Software CSV export.

Data separation rationale:
    oura_spo2 is a per-night summary (one row per date). checkme_spo2_continuous
    is a time series — different shape, kept in its own table. Coupling analyses
    join on timestamp when both exist.

Format status (as of 2026-04-16):
    Checkme O2 Max PC software CSV export has NOT been validated against a real
    file. Parser is built from Viatom/Wellue Checkme-family conventions. When
    the first real file arrives (~2026-04-28), validate and fix within 24h.

    Expected columns:
      Time / Timestamp         absolute or relative time
      SpO2 / Oxygen            %
      PR / Pulse / Heart Rate  bpm
      PI / Perfusion Index     %
      Motion / Movement        0/1 flag (optional)

Usage:
    python api/import_checkme_spo2.py --init-only
    python api/import_checkme_spo2.py --csv path/to/checkme_export.csv
    python api/import_checkme_spo2.py --csv export.csv --session-start 2026-04-28T22:00:00
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
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
log = logging.getLogger("import_checkme_spo2")


# Physiologic plausibility bands
SPO2_MIN = 50.0  # below is almost always artifact (off-finger, low perfusion)
SPO2_MAX = 100.0
PR_MIN = 25
PR_MAX = 220


class Spo2Reading(TypedDict):
    timestamp: str
    session_id: str
    spo2: float | None
    pulse_rate: int | None
    perfusion_index: float | None
    motion_flag: int | None
    is_artifact: int
    artifact_reason: str | None
    source_file: str | None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def init_checkme_spo2_tables(conn: sqlite3.Connection) -> None:
    """Create Checkme SpO2 tables + indexes. Idempotent."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS checkme_spo2_continuous (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            spo2 REAL,
            pulse_rate INTEGER,
            perfusion_index REAL,
            motion_flag INTEGER,
            is_artifact INTEGER DEFAULT 0,
            artifact_reason TEXT,
            source_file TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, timestamp)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS checkme_spo2_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            start_datetime TEXT NOT NULL,
            end_datetime TEXT,
            duration_seconds INTEGER,
            sample_count INTEGER,
            artifact_count INTEGER,
            mean_spo2 REAL,
            min_spo2 REAL,
            time_below_90_pct REAL,
            device_serial TEXT,
            source_file TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_checkme_spo2_timestamp
        ON checkme_spo2_continuous(timestamp)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_checkme_spo2_session
        ON checkme_spo2_continuous(session_id)
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_ABSOLUTE_TIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
)

_RELATIVE_TIME_FORMATS = (
    "%H:%M:%S",
    "%M:%S",
)


def _parse_time_field(
    raw: str, session_start: datetime | None = None
) -> datetime | None:
    """Parse either an absolute timestamp or a relative HH:MM:SS offset.

    If `raw` is a relative time (no date), `session_start` must be provided to
    anchor it into a full datetime.
    """
    raw = raw.strip()
    if not raw:
        return None

    # Absolute formats first
    for fmt in _ABSOLUTE_TIME_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass

    # Relative formats — need session_start
    if session_start is not None:
        for fmt in _RELATIVE_TIME_FORMATS:
            try:
                t = datetime.strptime(raw, fmt)
                # Convert to offset and add to session start
                offset = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
                return session_start + offset
            except ValueError:
                continue

    return None


def detect_spo2_artifact(
    spo2: float | None,
    pulse_rate: int | None,
    motion: int | None,
) -> tuple[bool, str | None]:
    if spo2 is None:
        return True, "no_spo2_reading"
    if not SPO2_MIN <= spo2 <= SPO2_MAX:
        return True, f"spo2_out_of_range:{spo2}"
    if pulse_rate is not None and not PR_MIN <= pulse_rate <= PR_MAX:
        return True, f"pulse_rate_out_of_range:{pulse_rate}"
    if motion == 1:
        return True, "motion_artifact"
    return False, None


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------
def parse_checkme_csv(
    csv_path: Path,
    session_id: str,
    session_start: datetime | None = None,
) -> list[Spo2Reading]:
    """Parse a Checkme O2 CSV export into Spo2Reading records.

    Flexible header detection: looks for Time/Timestamp, SpO2/Oxygen, PR/Pulse,
    PI/Perfusion, Motion. Rows with no valid time or no SpO2 are skipped with
    a warning.
    """
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

    idx_time = find_col(header_lower, ("timestamp",), ("datetime",), ("time",))
    idx_spo2 = find_col(header_lower, ("spo2",), ("sao2",), ("oxygen",))
    idx_pr = find_col(
        header_lower, ("pr",), ("pulse", "rate"), ("heart", "rate"), ("bpm",)
    )
    # Be careful: "pulse" alone could match "pulse rate" too; order matters
    if idx_pr is None:
        idx_pr = find_col(header_lower, ("pulse",))
    idx_pi = find_col(header_lower, ("pi",), ("perfusion",))
    idx_motion = find_col(header_lower, ("motion",), ("movement",))

    if idx_time is None:
        raise ValueError(f"No time column found in Checkme CSV header: {header}")
    if idx_spo2 is None:
        raise ValueError(f"No SpO2 column found in Checkme CSV header: {header}")

    readings: list[Spo2Reading] = []
    skipped_time = 0
    for row in reader:
        if not row or idx_time >= len(row):
            continue

        dt = _parse_time_field(row[idx_time], session_start=session_start)
        if dt is None:
            skipped_time += 1
            continue

        spo2 = float_or_none(get_cell(row, idx_spo2))
        pr = int_or_none(get_cell(row, idx_pr))
        pi = float_or_none(get_cell(row, idx_pi))
        motion = int_or_none(get_cell(row, idx_motion))

        artifact, reason = detect_spo2_artifact(spo2, pr, motion)

        readings.append(
            {
                "timestamp": dt.isoformat(),
                "session_id": session_id,
                "spo2": spo2,
                "pulse_rate": pr,
                "perfusion_index": pi,
                "motion_flag": motion,
                "is_artifact": 1 if artifact else 0,
                "artifact_reason": reason,
                "source_file": csv_path.name,
            }
        )

    if skipped_time:
        log.warning("Skipped %d rows with unparseable time fields", skipped_time)
    return readings


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------
def upsert_spo2_readings(
    conn: sqlite3.Connection, readings: Iterable[Spo2Reading]
) -> tuple[int, int]:
    """Insert readings, ignoring duplicates on (session_id, timestamp)."""
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for r in readings:
        cur.execute(
            """
            INSERT OR IGNORE INTO checkme_spo2_continuous (
                timestamp, session_id, spo2, pulse_rate, perfusion_index,
                motion_flag, is_artifact, artifact_reason, source_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["timestamp"],
                r["session_id"],
                r["spo2"],
                r["pulse_rate"],
                r["perfusion_index"],
                r["motion_flag"],
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
    readings: list[Spo2Reading],
    source_file: str,
    device_serial: str | None = None,
) -> None:
    """Compute and upsert session summary (mean/min SpO2, time<90%)."""
    if not readings:
        return

    clean = [
        r for r in readings if r["is_artifact"] == 0 and r["spo2"] is not None
    ]
    timestamps = [r["timestamp"] for r in readings]
    if clean:
        spo2_vals = [r["spo2"] for r in clean if r["spo2"] is not None]
        mean_spo2 = sum(spo2_vals) / len(spo2_vals) if spo2_vals else None
        min_spo2 = min(spo2_vals) if spo2_vals else None
        below_90 = sum(1 for v in spo2_vals if v < 90)
        time_below_90_pct = 100.0 * below_90 / len(spo2_vals) if spo2_vals else 0.0
    else:
        mean_spo2 = min_spo2 = None
        time_below_90_pct = None

    start_dt = datetime.fromisoformat(min(timestamps))
    end_dt = datetime.fromisoformat(max(timestamps))
    duration = int((end_dt - start_dt).total_seconds())

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO checkme_spo2_sessions (
            session_id, start_datetime, end_datetime, duration_seconds,
            sample_count, artifact_count, mean_spo2, min_spo2,
            time_below_90_pct, device_serial, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            start_datetime = excluded.start_datetime,
            end_datetime = excluded.end_datetime,
            duration_seconds = excluded.duration_seconds,
            sample_count = excluded.sample_count,
            artifact_count = excluded.artifact_count,
            mean_spo2 = excluded.mean_spo2,
            min_spo2 = excluded.min_spo2,
            time_below_90_pct = excluded.time_below_90_pct,
            source_file = excluded.source_file
        """,
        (
            session_id,
            start_dt.isoformat(),
            end_dt.isoformat(),
            duration,
            len(readings),
            sum(r["is_artifact"] for r in readings),
            mean_spo2,
            min_spo2,
            time_below_90_pct,
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
    parser.add_argument("--csv", help="Path to Checkme O2 CSV export")
    parser.add_argument(
        "--session-id",
        help="Session identifier (default: derived from CSV start time)",
    )
    parser.add_argument(
        "--session-start",
        help="ISO 8601 start time (required if CSV only has relative HH:MM:SS)",
    )
    parser.add_argument("--device-serial", help="Device serial number (optional)")
    parser.add_argument("--db", help="Database path override")
    parser.add_argument("--profile", "-p", help="Patient profile name")

    args = parser.parse_args()

    db_path = resolve_db_path(args.db, args.profile)
    conn = sqlite3.connect(db_path)
    try:
        init_checkme_spo2_tables(conn)
        if args.init_only:
            print(f"Initialized Checkme SpO2 tables in {db_path}")
            return 0

        if not args.csv:
            parser.error("either --csv or --init-only is required")

        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
            return 1

        session_start = (
            datetime.fromisoformat(args.session_start) if args.session_start else None
        )
        session_id = (
            args.session_id or f"checkme-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        )

        readings = parse_checkme_csv(
            csv_path, session_id=session_id, session_start=session_start
        )
        if not readings:
            print(f"No readings parsed from {csv_path}")
            return 1

        if not args.session_id:
            start_iso = min(r["timestamp"] for r in readings)
            session_id = f"checkme-{start_iso.replace(':', '').replace('-', '')[:15]}"
            for r in readings:
                r["session_id"] = session_id

        inserted, skipped = upsert_spo2_readings(conn, readings)
        record_session_summary(
            conn,
            session_id=session_id,
            readings=readings,
            source_file=csv_path.name,
            device_serial=args.device_serial,
        )
    finally:
        conn.close()

    print(f"Parsed {len(readings)} SpO2 samples from {csv_path}")
    print(f"Session: {session_id}")
    print(f"Inserted: {inserted}  |  Skipped (duplicates): {skipped}")
    print(f"Database: {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
