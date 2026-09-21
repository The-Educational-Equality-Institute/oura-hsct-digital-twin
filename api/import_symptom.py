#!/usr/bin/env python3
"""CLI for logging symptom events during the CGM trial.

Appends timestamped symptom records to the symptom_events table for later
coupling analysis against glucose_readings and Oura HRV/HR streams.
See reports/cgm_hypotheses_pre_registered.md for how these events are used.

Usage:
    python api/import_symptom.py \\
        --symptom chest_pain --severity 4 \\
        --context "30min postprandial" --note "after lunch"

    python api/import_symptom.py --symptom dizziness --severity 3 --profile mitch

    python api/import_symptom.py --symptom nausea --timestamp 2026-04-22T13:15:00
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ingest_common import resolve_db_path
from import_glucose import init_symptom_events_table


# Controlled vocabulary — extend as needed during trial.
# Keeping these explicit prevents typo-driven fragmentation of the dataset.
SYMPTOM_TYPES = (
    "chest_pain",
    "nausea",
    "dizziness",
    "fatigue",
    "palpitations",
    "shortness_of_breath",
    "headache",
    "sweating",
    "tremor",
    "confusion",
    "other",
)


def insert_symptom(
    conn: sqlite3.Connection,
    timestamp: str,
    symptom_type: str,
    severity: int | None,
    context: str | None,
    note: str | None,
) -> int:
    """Insert a symptom event and return its rowid."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO symptom_events
            (timestamp, symptom_type, severity, context, note)
        VALUES (?, ?, ?, ?, ?)
        """,
        (timestamp, symptom_type, severity, context, note),
    )
    conn.commit()
    return cur.lastrowid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log a symptom event to the symptom_events table"
    )
    parser.add_argument(
        "--symptom",
        required=True,
        choices=SYMPTOM_TYPES,
        help=f"Symptom type (choices: {', '.join(SYMPTOM_TYPES)})",
    )
    parser.add_argument("--severity", type=int, help="Severity 0-10")
    parser.add_argument(
        "--context",
        help="Short context tag, e.g. '30min postprandial', 'at_rest', 'post_exercise'",
    )
    parser.add_argument("--note", help="Free-text note")
    parser.add_argument(
        "--timestamp",
        help="ISO 8601 timestamp (default: now in local time)",
    )
    parser.add_argument("--db", help="Database path (overrides config and --profile)")
    parser.add_argument(
        "--profile", "-p", help="Patient profile name from profiles.py"
    )

    args = parser.parse_args()

    if args.severity is not None and not (0 <= args.severity <= 10):
        print("Error: severity must be between 0 and 10", file=sys.stderr)
        return 1

    ts = args.timestamp or datetime.now().isoformat(timespec="seconds")

    db_path = resolve_db_path(args.db, args.profile)
    conn = sqlite3.connect(db_path)
    try:
        init_symptom_events_table(conn)
        rowid = insert_symptom(
            conn,
            timestamp=ts,
            symptom_type=args.symptom,
            severity=args.severity,
            context=args.context,
            note=args.note,
        )
    finally:
        conn.close()

    print(f"Logged symptom event #{rowid}: {args.symptom} at {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
