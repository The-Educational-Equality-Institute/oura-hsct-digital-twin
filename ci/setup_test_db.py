#!/usr/bin/env python3
"""Create a minimal SQLite database for CI smoke tests.

Creates the required tables (oura_heart_rate, oura_sleep_periods,
oura_readiness) with one dummy row each so validate_config() passes.

Reuses init_database() from import_oura.py when available, otherwise
falls back to minimal CREATE TABLE statements.

Usage:
    python ci/setup_test_db.py              # default: data/oura.db
    python ci/setup_test_db.py path/to/db   # custom path
"""
import sqlite3
import sys
from pathlib import Path


def create_minimal_tables(conn: sqlite3.Connection) -> None:
    """Create the 3 tables that validate_config() checks for."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS oura_heart_rate (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT UNIQUE,
            bpm INTEGER,
            source TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS oura_sleep_periods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_id TEXT UNIQUE,
            day TEXT,
            type TEXT,
            average_hrv REAL,
            average_heart_rate REAL,
            average_breath REAL,
            total_sleep_duration INTEGER,
            rem_sleep_duration INTEGER,
            deep_sleep_duration INTEGER,
            light_sleep_duration INTEGER,
            awake_time INTEGER,
            efficiency INTEGER,
            latency INTEGER,
            restless_periods INTEGER,
            lowest_heart_rate INTEGER,
            bedtime_start TEXT,
            bedtime_end TEXT,
            time_in_bed INTEGER,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS oura_readiness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            score INTEGER,
            temperature_deviation REAL,
            activity_balance INTEGER,
            body_temperature INTEGER,
            hrv_balance INTEGER,
            previous_day_activity INTEGER,
            previous_night INTEGER,
            recovery_index INTEGER,
            resting_heart_rate INTEGER,
            sleep_balance INTEGER,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/oura.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    # Try to reuse the full schema from import_oura.py
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from api.import_oura import init_database

        conn.close()
        conn = init_database(db_path)
        print("Created DB with full schema from init_database()")
    except (ImportError, Exception) as e:
        print(f"Falling back to minimal schema: {e}")
        create_minimal_tables(conn)

    # Insert one dummy row per required table so validate_config() passes
    conn.execute(
        "INSERT OR IGNORE INTO oura_heart_rate (timestamp, bpm) "
        "VALUES ('2026-01-01T00:00:00', 70)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO oura_sleep_periods (period_id, day) "
        "VALUES ('ci-dummy', '2026-01-01')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO oura_readiness (date, score) "
        "VALUES ('2026-01-01', 80)"
    )
    conn.commit()

    # Verify
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    size = Path(db_path).stat().st_size
    print(f"Test DB: {db_path} ({size} bytes, {len(tables)} tables)")
    conn.close()


if __name__ == "__main__":
    main()
