#!/usr/bin/env python3
"""Glucose data import utility (FreeStyle Libre / LibreView CSV).

Imports CGM data into the same SQLite database as Oura data so coupling
analyses can join on timestamp. Primary path is LibreView CSV export;
LibreLink-Up API integration can be added later without schema changes.

Usage:
    # Initialize schema only (before sensor arrives)
    python api/import_glucose.py --init-only

    # Import a LibreView CSV export into the default database
    python api/import_glucose.py --csv path/to/libreview_export.csv

    # Import into a specific patient profile's database
    python api/import_glucose.py --csv export.csv --profile mitch
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
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

# Conversion: 1 mmol/L = 18.0182 mg/dL (glucose)
MG_DL_TO_MMOL_L = 18.0182

# LibreView record type codes — only types 0 and 1 carry glucose values.
# 0 = historic (automatic 1-minute sensor reading)
# 1 = scan (user-initiated tap)
# 2 = blood strip test, 3 = rapid insulin, 4 = food, 5 = long insulin, 6 = notes
RECORD_TYPE_HISTORIC = 0
RECORD_TYPE_SCAN = 1
_GLUCOSE_RECORD_TYPES = {RECORD_TYPE_HISTORIC, RECORD_TYPE_SCAN}


class GlucoseReading(TypedDict):
    timestamp: str  # ISO 8601
    glucose_mmol_l: float
    record_type: int
    sensor_serial: str | None
    source: str


# ---------------------------------------------------------------------------
# Schema initialization (non-breaking additions to oura.db)
# ---------------------------------------------------------------------------
def init_glucose_readings_table(conn: sqlite3.Connection) -> None:
    """Create glucose_readings table and index if missing."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS glucose_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            glucose_mmol_l REAL NOT NULL,
            record_type INTEGER,
            sensor_serial TEXT,
            source TEXT NOT NULL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, sensor_serial)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_glucose_readings_timestamp
        ON glucose_readings(timestamp)
        """
    )
    conn.commit()


def init_symptom_events_table(conn: sqlite3.Connection) -> None:
    """Create symptom_events table and indexes if missing.

    symptom_events complements glucose_readings: manual symptom timestamps
    are required for H3 (postprandial chest-pain coincidence) and any
    symptom-coupling analysis. See reports/cgm_hypotheses_pre_registered.md.
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS symptom_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symptom_type TEXT NOT NULL,
            severity INTEGER,
            context TEXT,
            note TEXT,
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_symptom_events_timestamp
        ON symptom_events(timestamp)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_symptom_events_type
        ON symptom_events(symptom_type)
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# LibreView CSV parsing
# ---------------------------------------------------------------------------
_TIMESTAMP_FORMATS = (
    "%d-%m-%Y %H:%M",
    "%m-%d-%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",  # Norwegian locale sometimes uses dots
)


def _parse_timestamp(ts: str) -> str:
    """Parse a LibreView timestamp string to ISO 8601.

    LibreView format varies by account locale (DD-MM-YYYY vs MM-DD-YYYY is the
    most common ambiguity). Tries each known format in order.
    """
    ts = ts.strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(ts, fmt).isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp format: {ts!r}")


def _detect_separator(sample_line: str) -> str:
    """Pick ';' for Norwegian/EU CSV or ',' for English CSV."""
    return ";" if sample_line.count(";") > sample_line.count(",") else ","


def _find_header_line(lines: list[str]) -> int:
    """LibreView exports have 1-3 metadata rows before the column header.

    Locate the header by scanning for a line containing 'Timestamp' (English)
    or 'Tidsstempel' (Norwegian).
    """
    for i, line in enumerate(lines):
        lower = line.lower()
        if "timestamp" in lower or "tidsstempel" in lower:
            return i
    raise ValueError(
        "Could not find header row with 'Timestamp' / 'Tidsstempel'. "
        "Is this a LibreView CSV export?"
    )


def parse_libreview_csv(
    csv_path: Path,
    sensor_serial: str | None = None,
) -> list[GlucoseReading]:
    """Parse a LibreView CSV export into glucose reading records.

    Handles these real-world format variants:
      - UTF-8 BOM (common LibreView quirk on Windows)
      - 1-3 metadata rows before the column header
      - 'Historic Glucose mmol/L' and 'Scan Glucose mmol/L' (EU default)
      - 'Historic Glucose mg/dL' and 'Scan Glucose mg/dL' (US default)
      - Comma-decimal values with semicolon separator (Norwegian locale)
      - Mixed DD-MM-YYYY / MM-DD-YYYY timestamp formats

    Filters out record_type not in {0, 1} (blood strip tests, insulin, food, notes).

    Raises ValueError with an actionable message if the file cannot be recognized
    as a LibreView export.
    """
    raw = csv_path.read_bytes()
    text = decode_bytes(raw)
    lines = text.splitlines()

    if not lines:
        raise ValueError(f"{csv_path} is empty")

    first_nonempty = next((line for line in lines if line.strip()), "")
    separator = _detect_separator(first_nonempty)

    header_idx = _find_header_line(lines)
    reader = csv.reader(lines[header_idx:], delimiter=separator)
    try:
        header = next(reader)
    except StopIteration as e:
        raise ValueError(f"{csv_path} has a header row but no data rows") from e

    header_lower = [h.strip().lower() for h in header]

    ts_idx = find_col(header_lower, ("timestamp",), ("tidsstempel",))
    rt_idx = find_col(header_lower, ("record", "type"), ("posttype",))
    serial_idx = find_col(header_lower, ("serial",), ("serienummer",))
    hist_mmol_idx = find_any(header_lower, "historic", "mmol")
    hist_mgdl_idx = find_any(header_lower, "historic", "mg/dl")
    scan_mmol_idx = find_any(header_lower, "scan", "mmol")
    scan_mgdl_idx = find_any(header_lower, "scan", "mg/dl")

    if ts_idx is None:
        raise ValueError(f"No timestamp column found in header: {header}")
    if not any(
        idx is not None
        for idx in (hist_mmol_idx, hist_mgdl_idx, scan_mmol_idx, scan_mgdl_idx)
    ):
        raise ValueError(
            f"No glucose column (historic/scan, mmol/L or mg/dL) found in header: {header}"
        )

    readings: list[GlucoseReading] = []
    for row in reader:
        if not row or ts_idx >= len(row) or not row[ts_idx].strip():
            continue

        # Filter by record type when available
        if rt_idx is not None and rt_idx < len(row):
            try:
                rt = int(row[rt_idx])
            except ValueError:
                continue
            if rt not in _GLUCOSE_RECORD_TYPES:
                continue
        else:
            rt = RECORD_TYPE_HISTORIC

        glucose_mmol = _extract_glucose(
            row,
            mmol_indices=(hist_mmol_idx, scan_mmol_idx),
            mgdl_indices=(hist_mgdl_idx, scan_mgdl_idx),
        )
        if glucose_mmol is None:
            continue

        try:
            ts_iso = _parse_timestamp(row[ts_idx])
        except ValueError as e:
            logging.warning("Skipping row with unparseable timestamp: %s", e)
            continue

        serial = sensor_serial
        if serial is None and serial_idx is not None and serial_idx < len(row):
            serial = row[serial_idx].strip() or None

        readings.append(
            {
                "timestamp": ts_iso,
                "glucose_mmol_l": glucose_mmol,
                "record_type": rt,
                "sensor_serial": serial,
                "source": "libreview_csv",
            }
        )

    return readings


def _extract_glucose(
    row: list[str],
    mmol_indices: tuple[int | None, ...],
    mgdl_indices: tuple[int | None, ...],
) -> float | None:
    """Return glucose in mmol/L from whichever column carries a value.

    Prefers mmol/L columns; falls back to mg/dL and converts.
    Accepts comma-decimal (Norwegian) and period-decimal values.
    """
    for idx in mmol_indices:
        if idx is None or idx >= len(row):
            continue
        val = row[idx].strip().replace(",", ".")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
    for idx in mgdl_indices:
        if idx is None or idx >= len(row):
            continue
        val = row[idx].strip().replace(",", ".")
        if val:
            try:
                return float(val) / MG_DL_TO_MMOL_L
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------
def upsert_glucose_readings(
    conn: sqlite3.Connection,
    readings: Iterable[GlucoseReading],
) -> tuple[int, int]:
    """Insert glucose readings, ignoring duplicates on (timestamp, sensor_serial).

    Returns (inserted, skipped) counts. Idempotent: re-importing the same CSV
    is safe.
    """
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for r in readings:
        cur.execute(
            """
            INSERT OR IGNORE INTO glucose_readings
                (timestamp, glucose_mmol_l, record_type, sensor_serial, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                r["timestamp"],
                r["glucose_mmol_l"],
                r["record_type"],
                r["sensor_serial"],
                r["source"],
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
    parser = argparse.ArgumentParser(
        description="Import FreeStyle Libre glucose data from LibreView CSV export"
    )
    parser.add_argument(
        "--csv",
        help="Path to LibreView CSV export (omit with --init-only)",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create glucose_readings and symptom_events tables without importing data",
    )
    parser.add_argument(
        "--sensor-serial",
        help="Override sensor serial number (default: read from CSV if column present)",
    )
    parser.add_argument("--db", help="Database path (overrides config and --profile)")
    parser.add_argument(
        "--profile",
        "-p",
        help="Patient profile name from profiles.py (default: config.py DATABASE_PATH)",
    )

    args = parser.parse_args()

    if not args.init_only and not args.csv:
        parser.error("either --csv or --init-only is required")

    db_path = resolve_db_path(args.db, args.profile)
    conn = sqlite3.connect(db_path)
    try:
        init_glucose_readings_table(conn)
        init_symptom_events_table(conn)
        if args.init_only:
            print(f"Initialized glucose_readings and symptom_events in {db_path}")
            return 0

        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
            return 1

        readings = parse_libreview_csv(csv_path, sensor_serial=args.sensor_serial)
        inserted, skipped = upsert_glucose_readings(conn, readings)
    finally:
        conn.close()

    print(f"Parsed {len(readings)} glucose readings from {csv_path}")
    print(f"Inserted: {inserted}  |  Skipped (duplicates): {skipped}")
    print(f"Database: {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
