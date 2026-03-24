#!/usr/bin/env python3
"""
Oura Ring Data Import Utility

Import Oura Ring data to SQLite database (oura.db) for analysis.
Includes sleep, readiness, activity, heart rate, HRV (from sleep periods),
SpO2, stress, sleep time, workouts, resilience, and cardiovascular age.

Usage:
    python api/import_oura.py --days 30
    python api/import_oura.py --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from oura_ring import OuraClient

# Load .env: try repo root first, then parent directory
_REPO_ENV = Path(__file__).resolve().parent.parent / ".env"
_PARENT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_REPO_ENV)
load_dotenv(_PARENT_ENV)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_PATH

API_URL = "https://api.ouraring.com"


def get_database_path() -> str:
    """Get database path from config or environment override."""
    return os.getenv("DATABASE_PATH", str(DATABASE_PATH))


def _unpack_time_series(
    base_timestamp_str: str, interval: int, items: list, endpoint: str = "unknown"
) -> list[tuple[str, float]]:
    """Convert Oura time-series {timestamp, interval, items[]} into (timestamp, value) pairs.

    Args:
        base_timestamp_str: ISO 8601 timestamp for the first item.
        interval: Seconds between consecutive items.
        items: List of numeric values (may contain None).
        endpoint: Name of the API endpoint (for logging).

    Returns:
        List of (timestamp_str, value) tuples. Skips None and values <= 0.
    """
    results = []
    skipped = 0
    try:
        # Parse the base timestamp, handling timezone offsets
        base_ts = datetime.fromisoformat(base_timestamp_str)
    except (ValueError, TypeError):
        logging.warning(
            "Unparseable base timestamp '%s' in %s, skipping entire series (%d items)",
            base_timestamp_str,
            endpoint,
            len(items) if items else 0,
        )
        return results

    for i, value in enumerate(items):
        if value is None or value <= 0:
            skipped += 1
            continue
        ts = base_ts + timedelta(seconds=i * interval)
        results.append((ts.isoformat(), float(value)))

    if skipped:
        logging.warning(
            "Skipped %d unparseable/null timestamps in %s", skipped, endpoint
        )

    return results


def _fetch_paginated(
    session: requests.Session, endpoint: str, start_date: str, end_date: str
) -> list[dict[str, Any]]:
    """Direct API call for endpoints not in oura-ring library.

    Uses the client's authenticated session to call the Oura API directly.
    Handles pagination via next_token.

    Args:
        session: Authenticated requests.Session (from client.session).
        endpoint: API path after /v2/usercollection/ (e.g. "daily_resilience").
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.

    Returns:
        List of record dicts.
    """
    url = f"{API_URL}/v2/usercollection/{endpoint}"
    params = {"start_date": start_date, "end_date": end_date}
    all_data: list[dict[str, Any]] = []
    max_retries = 3

    while True:
        for attempt in range(max_retries):
            resp = session.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logging.warning(
                    "Rate limited on %s, waiting %ds (attempt %d/%d)",
                    endpoint,
                    retry_after,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(retry_after)
                continue
            if resp.status_code == 401:
                logging.error(
                    "Oura API returned 401 Unauthorized. Token may be expired. "
                    "Run: python api/oura_oauth2_setup.py"
                )
                sys.exit(1)
            break  # Got a non-429, non-401 response
        else:
            # Exhausted retries on 429
            logging.error("Rate limit retries exhausted for %s", endpoint)
            resp.raise_for_status()

        if resp.status_code == 404:
            # Endpoint not available (e.g. ring generation too old)
            return []
        resp.raise_for_status()
        body = resp.json()
        all_data.extend(body.get("data", []))
        next_token = body.get("next_token")
        if next_token:
            params["next_token"] = next_token
        else:
            break

    return all_data


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize database with Oura tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Sleep data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_sleep (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            score INTEGER,
            total_sleep_duration INTEGER,
            rem_sleep_duration INTEGER,
            deep_sleep_duration INTEGER,
            light_sleep_duration INTEGER,
            awake_time INTEGER,
            efficiency INTEGER,
            latency INTEGER,
            restless_periods INTEGER,
            bedtime_start TEXT,
            bedtime_end TEXT,
            hr_lowest INTEGER,
            hr_average REAL,
            hrv_average REAL,
            breath_average REAL,
            temperature_delta REAL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Readiness data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_readiness (
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
        )
    """)

    # Activity data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            score INTEGER,
            active_calories INTEGER,
            total_calories INTEGER,
            steps INTEGER,
            daily_movement INTEGER,
            inactive_time INTEGER,
            rest_time INTEGER,
            low_activity_time INTEGER,
            medium_activity_time INTEGER,
            high_activity_time INTEGER,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Heart rate samples (5-minute intervals)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_heart_rate (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            bpm INTEGER,
            source TEXT,
            UNIQUE(timestamp)
        )
    """)

    # HRV samples (5-minute RMSSD from sleep periods)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_hrv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            rmssd REAL,
            source TEXT,
            UNIQUE(timestamp)
        )
    """)

    # Sleep periods  - detailed per-period data (multiple per day possible)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_sleep_periods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_id TEXT UNIQUE NOT NULL,
            day TEXT NOT NULL,
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
        )
    """)

    # SpO2 daily averages + breathing disturbance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_spo2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            spo2_average REAL,
            breathing_disturbance_index REAL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Stress daily summary
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_stress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            stress_high INTEGER,
            recovery_high INTEGER,
            day_summary TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Sleep time recommendations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_sleep_time (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            optimal_bedtime_start_offset INTEGER,
            optimal_bedtime_end_offset INTEGER,
            recommendation TEXT,
            status TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Workouts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workout_id TEXT UNIQUE NOT NULL,
            day TEXT NOT NULL,
            activity TEXT,
            calories REAL,
            distance REAL,
            intensity TEXT,
            start_datetime TEXT,
            end_datetime TEXT,
            source TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Resilience (direct API  - not in oura-ring library)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_resilience (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            level TEXT,
            contributors_sleep_recovery REAL,
            contributors_daytime_recovery REAL,
            contributors_stress REAL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Cardiovascular age (direct API  - not in oura-ring library)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_cardiovascular_age (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            vascular_age REAL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migrate: add breathing_disturbance_index to oura_spo2 if missing
    try:
        cursor.execute("SELECT breathing_disturbance_index FROM oura_spo2 LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute(
            "ALTER TABLE oura_spo2 ADD COLUMN breathing_disturbance_index REAL"
        )

    # Sleep HR time-series (5-second resolution from sleep periods)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_sleep_hr_timeseries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            bpm INTEGER,
            UNIQUE(period_id, timestamp)
        )
    """)

    # Sleep phase epochs (5-minute staging from sleep periods)
    # Values: 1=deep, 2=light, 3=REM, 4=awake
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_sleep_epochs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_id TEXT NOT NULL,
            epoch_index INTEGER NOT NULL,
            phase INTEGER NOT NULL,
            UNIQUE(period_id, epoch_index)
        )
    """)

    # Sleep movement (30-second classification from sleep periods)
    # Values: 1=no motion, 2=restless, 3=tossing/turning, 4=active
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_sleep_movement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_id TEXT NOT NULL,
            movement_index INTEGER NOT NULL,
            classification INTEGER NOT NULL,
            UNIQUE(period_id, movement_index)
        )
    """)

    # VO2 Max (direct API  - not in oura-ring library)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_vo2_max (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            vo2_max REAL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Rest mode periods  - illness/recovery tracking (GVHD flare correlation)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_rest_mode (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_id TEXT UNIQUE NOT NULL,
            start_day TEXT NOT NULL,
            end_day TEXT,
            start_date TEXT,
            end_date TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Sessions  - guided breathing/meditation (controlled-condition HRV)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            day TEXT NOT NULL,
            type TEXT,
            start_datetime TEXT,
            end_datetime TEXT,
            heart_rate_average REAL,
            hrv_average REAL,
            mood TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Enhanced tags  - symptoms, meds, notes (subjective-objective correlation)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_id TEXT UNIQUE NOT NULL,
            day TEXT NOT NULL,
            timestamp TEXT,
            tag_type_code TEXT,
            comment TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Ring configuration  - hardware/firmware provenance for evidence chain
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_ring_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id TEXT UNIQUE NOT NULL,
            color TEXT,
            design TEXT,
            firmware_version TEXT,
            hardware_type TEXT,
            set_up_at TEXT,
            size INTEGER,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Personal info  - baseline demographics (no date range)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oura_personal_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            weight REAL,
            height REAL,
            biological_sex TEXT,
            email_hash TEXT,  -- SHA-256 prefix of email (PII redacted)
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn


def import_oura_data(
    conn: sqlite3.Connection, client: OuraClient, start_date: str, end_date: str
) -> dict:
    """
    Import Oura data for the specified date range.

    Returns:
        Dictionary with counts of imported records by type
    """
    cursor = conn.cursor()
    stats = {
        "sleep": 0,
        "readiness": 0,
        "activity": 0,
        "heart_rate": 0,
        "sleep_periods": 0,
        "hrv": 0,
        "sleep_hrv_updated": 0,
        "sleep_hr_timeseries": 0,
        "sleep_epochs": 0,
        "sleep_movement": 0,
        "spo2": 0,
        "stress": 0,
        "sleep_time": 0,
        "workouts": 0,
        "resilience": 0,
        "cardiovascular_age": 0,
        "vo2_max": 0,
        "rest_mode": 0,
        "sessions": 0,
        "tags": 0,
        "ring_config": 0,
        "personal_info": 0,
    }

    # ---- 1. Daily sleep (existing) ----
    try:
        sleep_data = client.get_daily_sleep(start_date, end_date)
        records = (
            sleep_data if isinstance(sleep_data, list) else sleep_data.get("data", [])
        )
        for record in records:
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_sleep (
                    date, score, total_sleep_duration, rem_sleep_duration,
                    deep_sleep_duration, light_sleep_duration, awake_time,
                    efficiency, latency, restless_periods, bedtime_start,
                    bedtime_end, hr_lowest, hr_average, hrv_average,
                    breath_average, temperature_delta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.get("day"),
                    record.get("score"),
                    record.get("total_sleep_duration"),
                    record.get("rem_sleep_duration"),
                    record.get("deep_sleep_duration"),
                    record.get("light_sleep_duration"),
                    record.get("awake_time"),
                    record.get("efficiency"),
                    record.get("latency"),
                    record.get("restless_periods"),
                    record.get("bedtime_start"),
                    record.get("bedtime_end"),
                    record.get("lowest_heart_rate"),
                    record.get("average_heart_rate"),
                    record.get("average_hrv"),
                    record.get("average_breath"),
                    record.get("readiness", {}).get("temperature_deviation"),
                ),
            )
            stats["sleep"] += 1
    except Exception as e:
        logging.warning("Could not import sleep data: %s", e)

    # ---- 2. Readiness (existing) ----
    try:
        readiness_data = client.get_daily_readiness(start_date, end_date)
        records = (
            readiness_data
            if isinstance(readiness_data, list)
            else readiness_data.get("data", [])
        )
        for record in records:
            contributors = record.get("contributors", {})
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_readiness (
                    date, score, temperature_deviation, activity_balance,
                    body_temperature, hrv_balance, previous_day_activity,
                    previous_night, recovery_index, resting_heart_rate,
                    sleep_balance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.get("day"),
                    record.get("score"),
                    record.get("temperature_deviation"),
                    contributors.get("activity_balance"),
                    contributors.get("body_temperature"),
                    contributors.get("hrv_balance"),
                    contributors.get("previous_day_activity"),
                    contributors.get("previous_night"),
                    contributors.get("recovery_index"),
                    contributors.get("resting_heart_rate"),
                    contributors.get("sleep_balance"),
                ),
            )
            stats["readiness"] += 1
    except Exception as e:
        logging.warning("Could not import readiness data: %s", e)

    # ---- 3. Activity (existing) ----
    try:
        activity_data = client.get_daily_activity(start_date, end_date)
        records = (
            activity_data
            if isinstance(activity_data, list)
            else activity_data.get("data", [])
        )
        for record in records:
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_activity (
                    date, score, active_calories, total_calories, steps,
                    daily_movement, inactive_time, rest_time,
                    low_activity_time, medium_activity_time, high_activity_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.get("day"),
                    record.get("score"),
                    record.get("active_calories"),
                    record.get("total_calories"),
                    record.get("steps"),
                    record.get("daily_movement"),
                    record.get("sedentary_time"),
                    record.get("resting_time"),
                    record.get("low_activity_time"),
                    record.get("medium_activity_time"),
                    record.get("high_activity_time"),
                ),
            )
            stats["activity"] += 1
    except Exception as e:
        logging.warning("Could not import activity data: %s", e)

    # ---- 4. Heart rate (existing) ----
    try:
        hr_data = client.get_heart_rate(start_date, end_date)
        records = hr_data if isinstance(hr_data, list) else hr_data.get("data", [])
        for record in records:
            cursor.execute(
                """
                INSERT OR IGNORE INTO oura_heart_rate (timestamp, bpm, source)
                VALUES (?, ?, ?)
            """,
                (record.get("timestamp"), record.get("bpm"), record.get("source")),
            )
            stats["heart_rate"] += 1
    except Exception as e:
        logging.warning("Could not import heart rate data: %s", e)

    # ---- 5. Sleep periods + HRV fix (NEW) ----
    # This is the critical fix: get_sleep_periods() provides per-period HRV data
    # that get_daily_sleep() does not include.
    try:
        periods = client.get_sleep_periods(start_date, end_date)
        records = periods if isinstance(periods, list) else periods.get("data", [])

        # Track best HRV per day for updating oura_sleep
        # Prefer long_sleep type; keep highest average_hrv if multiple
        best_hrv_per_day: dict[str, tuple[float, str]] = {}  # day -> (hrv, type)

        for record in records:
            period_id = record.get("id")
            day = record.get("day")
            sleep_type = record.get("type", "unknown")
            avg_hrv = record.get("average_hrv")

            if not period_id or not day:
                continue

            # Insert detailed sleep period record
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_sleep_periods (
                    period_id, day, type, average_hrv, average_heart_rate,
                    average_breath, total_sleep_duration, rem_sleep_duration,
                    deep_sleep_duration, light_sleep_duration, awake_time,
                    efficiency, latency, restless_periods, lowest_heart_rate,
                    bedtime_start, bedtime_end, time_in_bed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    period_id,
                    day,
                    sleep_type,
                    avg_hrv,
                    record.get("average_heart_rate"),
                    record.get("average_breath"),
                    record.get("total_sleep_duration"),
                    record.get("rem_sleep_duration"),
                    record.get("deep_sleep_duration"),
                    record.get("light_sleep_duration"),
                    record.get("awake_time"),
                    record.get("efficiency"),
                    record.get("latency"),
                    record.get("restless_periods"),
                    record.get("lowest_heart_rate"),
                    record.get("bedtime_start"),
                    record.get("bedtime_end"),
                    record.get("time_in_bed"),
                ),
            )
            stats["sleep_periods"] += 1

            # Unpack HRV time-series items into individual RMSSD rows
            hrv_data = record.get("hrv", {})
            if hrv_data and hrv_data.get("items"):
                pairs = _unpack_time_series(
                    hrv_data.get("timestamp", ""),
                    hrv_data.get("interval", 300),
                    hrv_data["items"],
                    endpoint="sleep_periods/hrv",
                )
                for ts, rmssd in pairs:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO oura_hrv (timestamp, rmssd, source)
                        VALUES (?, ?, ?)
                    """,
                        (ts, rmssd, f"sleep_period:{sleep_type}"),
                    )
                    stats["hrv"] += 1

            # Unpack HR time-series from sleep period (5-second resolution)
            hr_data = record.get("heart_rate", {})
            if hr_data and hr_data.get("items"):
                hr_pairs = _unpack_time_series(
                    hr_data.get("timestamp", ""),
                    hr_data.get("interval", 5),
                    hr_data["items"],
                    endpoint="sleep_periods/heart_rate",
                )
                for ts, bpm in hr_pairs:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO oura_sleep_hr_timeseries (period_id, timestamp, bpm)
                        VALUES (?, ?, ?)
                    """,
                        (period_id, ts, int(bpm)),
                    )
                    stats["sleep_hr_timeseries"] += 1

            # Extract sleep phase epochs (5-minute staging)
            sleep_phases = record.get("sleep_phase_5_min")
            if sleep_phases:
                for idx, phase_char in enumerate(sleep_phases):
                    if phase_char.isdigit():
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO oura_sleep_epochs (period_id, epoch_index, phase)
                            VALUES (?, ?, ?)
                        """,
                            (period_id, idx, int(phase_char)),
                        )
                        stats["sleep_epochs"] += 1

            # Extract movement data (30-second classification)
            movement = record.get("movement_30_sec")
            if movement:
                for idx, move_char in enumerate(movement):
                    if move_char.isdigit():
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO oura_sleep_movement (period_id, movement_index, classification)
                            VALUES (?, ?, ?)
                        """,
                            (period_id, idx, int(move_char)),
                        )
                        stats["sleep_movement"] += 1

            # Track best HRV per day for oura_sleep update
            if avg_hrv is not None and avg_hrv > 0:
                existing = best_hrv_per_day.get(day)
                if existing is None:
                    best_hrv_per_day[day] = (avg_hrv, sleep_type)
                else:
                    # Prefer long_sleep; if both same type, keep higher HRV
                    if sleep_type == "long_sleep" and existing[1] != "long_sleep":
                        best_hrv_per_day[day] = (avg_hrv, sleep_type)
                    elif sleep_type == existing[1] and avg_hrv > existing[0]:
                        best_hrv_per_day[day] = (avg_hrv, sleep_type)

        # Update oura_sleep.hrv_average with best per-day HRV from sleep periods
        for day, (hrv_val, _) in best_hrv_per_day.items():
            cursor.execute(
                """
                UPDATE oura_sleep SET hrv_average = ? WHERE date = ? AND (hrv_average IS NULL OR hrv_average = 0)
            """,
                (hrv_val, day),
            )
            if cursor.rowcount > 0:
                stats["sleep_hrv_updated"] += 1

    except Exception as e:
        logging.warning("Could not import sleep periods/HRV data: %s", e)

    # ---- 6. SpO2 (NEW) ----
    try:
        spo2_data = client.get_daily_spo2(start_date, end_date)
        records = (
            spo2_data if isinstance(spo2_data, list) else spo2_data.get("data", [])
        )
        for record in records:
            spo2_pct = record.get("spo2_percentage", {})
            avg = spo2_pct.get("average") if isinstance(spo2_pct, dict) else None
            bdi = record.get("breathing_disturbance_index")
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_spo2 (date, spo2_average, breathing_disturbance_index)
                VALUES (?, ?, ?)
            """,
                (record.get("day"), avg, bdi),
            )
            stats["spo2"] += 1
    except Exception as e:
        logging.warning("Could not import SpO2 data: %s", e)

    # ---- 7. Stress (NEW) ----
    try:
        stress_data = client.get_daily_stress(start_date, end_date)
        records = (
            stress_data
            if isinstance(stress_data, list)
            else stress_data.get("data", [])
        )
        for record in records:
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_stress (date, stress_high, recovery_high, day_summary)
                VALUES (?, ?, ?, ?)
            """,
                (
                    record.get("day"),
                    record.get("stress_high"),
                    record.get("recovery_high"),
                    record.get("day_summary"),
                ),
            )
            stats["stress"] += 1
    except Exception as e:
        logging.warning("Could not import stress data: %s", e)

    # ---- 8. Sleep time (NEW) ----
    try:
        sleep_time_data = client.get_sleep_time(start_date, end_date)
        records = (
            sleep_time_data
            if isinstance(sleep_time_data, list)
            else sleep_time_data.get("data", [])
        )
        for record in records:
            bedtime = record.get("optimal_bedtime", {})
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_sleep_time (
                    date, optimal_bedtime_start_offset, optimal_bedtime_end_offset,
                    recommendation, status
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    record.get("day"),
                    bedtime.get("start_offset") if isinstance(bedtime, dict) else None,
                    bedtime.get("end_offset") if isinstance(bedtime, dict) else None,
                    record.get("recommendation"),
                    record.get("status"),
                ),
            )
            stats["sleep_time"] += 1
    except Exception as e:
        logging.warning("Could not import sleep time data: %s", e)

    # ---- 9. Workouts (NEW) ----
    try:
        workout_data = client.get_workouts(start_date, end_date)
        records = (
            workout_data
            if isinstance(workout_data, list)
            else workout_data.get("data", [])
        )
        for record in records:
            workout_id = record.get("id")
            if not workout_id:
                continue
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_workouts (
                    workout_id, day, activity, calories, distance,
                    intensity, start_datetime, end_datetime, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    workout_id,
                    record.get("day"),
                    record.get("activity"),
                    record.get("calories"),
                    record.get("distance"),
                    record.get("intensity"),
                    record.get("start_datetime"),
                    record.get("end_datetime"),
                    record.get("source"),
                ),
            )
            stats["workouts"] += 1
    except Exception as e:
        logging.warning("Could not import workout data: %s", e)

    # ---- 10. Resilience (NEW  - direct API) ----
    try:
        resilience_data = _fetch_paginated(
            client.session, "daily_resilience", start_date, end_date
        )
        for record in resilience_data:
            contributors = record.get("contributors", {})
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_resilience (
                    date, level, contributors_sleep_recovery,
                    contributors_daytime_recovery, contributors_stress
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    record.get("day"),
                    record.get("level"),
                    contributors.get("sleep_recovery")
                    if isinstance(contributors, dict)
                    else None,
                    contributors.get("daytime_recovery")
                    if isinstance(contributors, dict)
                    else None,
                    contributors.get("stress")
                    if isinstance(contributors, dict)
                    else None,
                ),
            )
            stats["resilience"] += 1
    except Exception as e:
        logging.warning("Could not import resilience data: %s", e)

    # ---- 11. Cardiovascular age (NEW  - direct API) ----
    try:
        cv_data = _fetch_paginated(
            client.session, "daily_cardiovascular_age", start_date, end_date
        )
        for record in cv_data:
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_cardiovascular_age (date, vascular_age)
                VALUES (?, ?)
            """,
                (
                    record.get("day"),
                    record.get("vascular_age"),
                ),
            )
            stats["cardiovascular_age"] += 1
    except Exception as e:
        logging.warning("Could not import cardiovascular age data: %s", e)

    # ---- 12. VO2 Max (NEW  - direct API) ----
    try:
        vo2_data = _fetch_paginated(client.session, "vo2_max", start_date, end_date)
        for record in vo2_data:
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_vo2_max (date, vo2_max)
                VALUES (?, ?)
            """,
                (
                    record.get("day"),
                    record.get("vo2_max"),
                ),
            )
            stats["vo2_max"] += 1
    except Exception as e:
        logging.warning("Could not import VO2 Max data: %s", e)

    # ---- 13. Rest mode periods (library method) ----
    try:
        rest_data = client.get_rest_mode_period(start_date, end_date)
        records = (
            rest_data if isinstance(rest_data, list) else rest_data.get("data", [])
        )
        for record in records:
            period_id = record.get("id")
            if not period_id:
                continue
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_rest_mode (
                    period_id, start_day, end_day, start_date, end_date
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    period_id,
                    record.get("start_day"),
                    record.get("end_day"),
                    record.get("start_date"),
                    record.get("end_date"),
                ),
            )
            stats["rest_mode"] += 1
    except Exception as e:
        logging.warning("Could not import rest mode data: %s", e)

    # ---- 14. Sessions  - guided breathing/meditation (library method) ----
    try:
        session_data = client.get_sessions(start_date, end_date)
        records = (
            session_data
            if isinstance(session_data, list)
            else session_data.get("data", [])
        )
        for record in records:
            session_id = record.get("id")
            if not session_id:
                continue
            hr_data = record.get("heart_rate", {})
            hrv_data = record.get("hrv", {})
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_sessions (
                    session_id, day, type, start_datetime, end_datetime,
                    heart_rate_average, hrv_average, mood
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    record.get("day"),
                    record.get("type"),
                    record.get("start_datetime"),
                    record.get("end_datetime"),
                    hr_data.get("average") if isinstance(hr_data, dict) else None,
                    hrv_data.get("average") if isinstance(hrv_data, dict) else None,
                    record.get("mood"),
                ),
            )
            stats["sessions"] += 1
    except Exception as e:
        logging.warning("Could not import session data: %s", e)

    # ---- 15. Enhanced tags  - symptoms, meds, notes (library method) ----
    try:
        tag_data = client.get_enhanced_tag(start_date, end_date)
        records = tag_data if isinstance(tag_data, list) else tag_data.get("data", [])
        for record in records:
            tag_id = record.get("id")
            if not tag_id:
                continue
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_tags (
                    tag_id, day, timestamp, tag_type_code, comment
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    tag_id,
                    record.get("day"),
                    record.get("timestamp"),
                    record.get("tag_type_code"),
                    record.get("comment"),
                ),
            )
            stats["tags"] += 1
    except Exception as e:
        logging.warning("Could not import enhanced tag data: %s", e)

    # ---- 16. Ring configuration (library method, no date range) ----
    try:
        config_data = client.get_ring_configuration()
        records = (
            config_data
            if isinstance(config_data, list)
            else config_data.get("data", [])
        )
        for record in records:
            config_id = record.get("id")
            if not config_id:
                continue
            cursor.execute(
                """
                INSERT OR REPLACE INTO oura_ring_config (
                    config_id, color, design, firmware_version,
                    hardware_type, set_up_at, size
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    config_id,
                    record.get("color"),
                    record.get("design"),
                    record.get("firmware_version"),
                    record.get("hardware_type"),
                    record.get("set_up_at"),
                    record.get("size"),
                ),
            )
            stats["ring_config"] += 1
    except Exception as e:
        logging.warning("Could not import ring configuration: %s", e)

    # ---- 17. Personal info (library method, no date range) ----
    try:
        info = client.get_personal_info()
        # get_personal_info returns a single record, not a list
        if isinstance(info, dict):
            data = info.get("data", info) if "data" in info else info
            cursor.execute("DELETE FROM oura_personal_info")  # Single-row table
            raw_email = data.get("email")
            email_hash = hashlib.sha256(raw_email.encode()).hexdigest()[:16] if raw_email else None
            cursor.execute(
                """
                INSERT INTO oura_personal_info (age, weight, height, biological_sex, email_hash)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    data.get("age"),
                    data.get("weight"),
                    data.get("height"),
                    data.get("biological_sex"),
                    email_hash,
                ),
            )
            stats["personal_info"] += 1
    except Exception as e:
        logging.warning("Could not import personal info: %s", e)

    conn.commit()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Import Oura Ring data to SQLite database"
    )
    parser.add_argument("--days", "-d", type=int, help="Import last N days of data")
    parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD)")
    parser.add_argument("--db", help="Database path (default: from DATABASE_PATH env)")

    args = parser.parse_args()

    # Get Oura token  - prefer OAuth2 access token, fall back to PAT
    oura_token = os.getenv("OURA_ACCESS_TOKEN") or os.getenv("OURA_PAT")
    token_type = "OAuth2" if os.getenv("OURA_ACCESS_TOKEN") else "PAT"
    if not oura_token or oura_token == "your_oura_personal_access_token":
        print("Error: No Oura API token configured in the parent .env file")
        print("Option 1 (recommended): python api/oura_oauth2_setup.py")
        print(
            "Option 2 (legacy): Set OURA_PAT at https://cloud.ouraring.com/personal-access-tokens"
        )
        return 1
    print(f"Using {token_type} authentication")

    # Determine date range
    if args.days:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    elif args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        # Default to last 30 days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"Importing Oura data from {start_date} to {end_date}")

    # Initialize database
    db_path = args.db or get_database_path()
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = init_database(db_path)

    try:
        # Create Oura client with retry adapter (F30)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)

        client = OuraClient(personal_access_token=oura_token)
        # OuraClient uses `session`, so attach the retry-enabled session there.
        client.session = session
        client.session.headers.update({"Authorization": f"Bearer {oura_token}"})

        # Pre-flight auth check (F31)
        logging.info("Verifying API access...")
        try:
            preflight = session.get(
                f"{API_URL}/v2/usercollection/personal_info",
                timeout=10,
            )
            if preflight.status_code == 401:
                logging.error("Authentication failed (HTTP 401). Token may be expired.")
                logging.error("Try: python api/oura_oauth2_setup.py --refresh")
                conn.close()
                return 1
            elif preflight.status_code != 200:
                logging.warning(
                    "Pre-flight check returned HTTP %d — proceeding anyway",
                    preflight.status_code,
                )
            else:
                logging.info("API access verified (HTTP 200)")
        except requests.RequestException as e:
            logging.warning("Pre-flight check failed: %s — proceeding anyway", e)

        stats = import_oura_data(conn, client, start_date, end_date)
    finally:
        conn.close()

    print("\nImport complete:")
    print(f"  Sleep records:          {stats['sleep']}")
    print(f"  Readiness records:      {stats['readiness']}")
    print(f"  Activity records:       {stats['activity']}")
    print(f"  Heart rate samples:     {stats['heart_rate']}")
    print(f"  Sleep periods:          {stats['sleep_periods']}")
    print(f"  HRV (RMSSD) samples:    {stats['hrv']}")
    print(f"  Sleep HRV backfilled:   {stats['sleep_hrv_updated']}")
    print(f"  Sleep HR time-series:   {stats['sleep_hr_timeseries']}")
    print(f"  Sleep epochs (5-min):   {stats['sleep_epochs']}")
    print(f"  Sleep movement (30s):   {stats['sleep_movement']}")
    print(f"  SpO2 records:           {stats['spo2']}")
    print(f"  Stress records:         {stats['stress']}")
    print(f"  Sleep time records:     {stats['sleep_time']}")
    print(f"  Workout records:        {stats['workouts']}")
    print(f"  Resilience records:     {stats['resilience']}")
    print(f"  Cardiovascular age:     {stats['cardiovascular_age']}")
    print(f"  VO2 Max records:        {stats['vo2_max']}")
    print(f"  Rest mode periods:      {stats['rest_mode']}")
    print(f"  Sessions (breathing):   {stats['sessions']}")
    print(f"  Enhanced tags:          {stats['tags']}")
    print(f"  Ring configurations:    {stats['ring_config']}")
    print(f"  Personal info:          {stats['personal_info']}")
    print(f"\nData saved to: {db_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
