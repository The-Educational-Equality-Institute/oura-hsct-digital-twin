#!/usr/bin/env python3
"""OMRON M7 Intelli IT AFib (HEM-7380T1) blood pressure data ingester.

Reads CSV dumps produced by the Windows-side BLE runner (`omron_pull.py`) and
writes them into `oura.db` alongside the Oura time-series. Derives clinical
fields at ingest time:

    - Mean arterial pressure (MAP)
    - Pulse pressure (SYS - DIA)
    - AM/PM classification
    - Triplet grouping (OMRON AFib-mode = 3 readings within ~5 min)
    - AFib candidate flag (triplet with >=2 IHB positives)
    - Artifact flag (implausible values, narrow pulse pressure, future timestamps)

Usage:
    python api/import_omron.py                          # default profile, default inbox
    python api/import_omron.py --profile henrik
    python api/import_omron.py --inbox /mnt/c/Users/ovehe/omron-bridge/out

The ingester is idempotent: UNIQUE(user_slot, datetime) dedupes re-reads of the
device's 100-slot ring buffer. CSV files are moved to `<inbox>/ingested/` after
successful import so they aren't re-processed.

Companion:
    Windows runner that produces the CSVs: C:\\Users\\ovehe\\omron-bridge\\omron_pull.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH  # noqa: E402
from profiles import PROFILES  # noqa: E402  # used for --profile flag

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("import_omron")

DEFAULT_INBOX = Path("/mnt/c/Users/ovehe/omron-bridge/out")
TRIPLET_WINDOW = timedelta(minutes=5)

# OMRON M7 AFib (HEM-7380T1) per-reading record byte format:
#   sys (raw + 25), dia, bpm, ihb flag, mov flag, datetime
# All fields are stored as-decoded; derived fields are computed here.

CSV_FIELDNAMES = {"datetime", "dia", "sys", "bpm", "mov", "ihb"}


def init_database(db_path: Path) -> sqlite3.Connection:
    """Create OMRON tables if they don't exist. Idempotent."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS omron_bp_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            user_slot INTEGER NOT NULL,
            sys INTEGER NOT NULL,
            dia INTEGER NOT NULL,
            bpm INTEGER NOT NULL,
            ihb INTEGER NOT NULL,
            mov INTEGER NOT NULL,
            map_mmhg REAL NOT NULL,
            pulse_pressure INTEGER NOT NULL,
            am_pm TEXT,
            triplet_id TEXT,
            triplet_seq INTEGER,
            afib_candidate INTEGER DEFAULT 0,
            is_artifact INTEGER DEFAULT 0,
            artifact_reason TEXT,
            source_file TEXT,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_slot, datetime)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS omron_bp_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pulled_at TEXT NOT NULL,
            device_model TEXT,
            device_mac TEXT,
            device_serial TEXT,
            firmware_version TEXT,
            manufacturer TEXT,
            hardware_revision TEXT,
            software_revision TEXT,
            system_id TEXT,
            battery_pct INTEGER,
            reading_count INTEGER,
            source_file TEXT UNIQUE
        )
    """)

    # Safe migration: add columns on older databases created before these fields existed.
    for col, col_type in (
        ("hardware_revision", "TEXT"),
        ("software_revision", "TEXT"),
        ("system_id", "TEXT"),
    ):
        try:
            cur.execute(f"SELECT {col} FROM omron_bp_sessions LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute(f"ALTER TABLE omron_bp_sessions ADD COLUMN {col} {col_type}")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_omron_bp_datetime ON omron_bp_readings(datetime)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_omron_bp_triplet ON omron_bp_readings(triplet_id)")

    conn.commit()
    return conn


def is_artifact(sys_v: int, dia_v: int, bpm: int, dt: datetime, now: datetime) -> tuple[bool, str | None]:
    """Return (is_artifact, reason). None reason means clean."""
    # Future dates / unsynced clock (M7 ships with 2048-ish dates)
    if dt.year > now.year + 1:
        return True, f"future_timestamp:{dt.year}"
    # Physiologically implausible
    if sys_v < 60 or sys_v > 260:
        return True, f"sys_out_of_range:{sys_v}"
    if dia_v < 30 or dia_v > 200:
        return True, f"dia_out_of_range:{dia_v}"
    if bpm < 25 or bpm > 220:
        return True, f"bpm_out_of_range:{bpm}"
    # Sys must exceed dia by >= 20 (narrow pulse pressure = cuff artifact on M7)
    if sys_v - dia_v < 20:
        return True, f"narrow_pulse_pressure:{sys_v - dia_v}"
    return False, None


def classify_am_pm(dt: datetime) -> str:
    """Classify reading by clinical window. ESH morning window is pre-breakfast / first 2h after waking."""
    h = dt.hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"


def group_triplets(rows: list[dict[str, Any]]) -> None:
    """Mutates rows: assigns triplet_id, triplet_seq, afib_candidate.

    OMRON AFib mode takes 3 consecutive measurements ~30-60s apart, though in
    practice the recorded intervals are 1-3 minutes. Group any run of readings
    from the same user_slot within TRIPLET_WINDOW as one triplet.
    """
    # Sort per-user by datetime
    by_user: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_user.setdefault(r["user_slot"], []).append(r)
    for user_rows in by_user.values():
        user_rows.sort(key=lambda r: r["_dt"])

    for user_slot, user_rows in by_user.items():
        i = 0
        while i < len(user_rows):
            j = i
            # Extend window while consecutive deltas stay within TRIPLET_WINDOW
            while (
                j + 1 < len(user_rows)
                and user_rows[j + 1]["_dt"] - user_rows[j]["_dt"] <= TRIPLET_WINDOW
            ):
                j += 1
            group = user_rows[i : j + 1]
            if len(group) >= 2:
                # Treat as a triplet (or duplicate reading, or triplicate + bonus)
                # Use earliest datetime as triplet ID anchor
                triplet_id = f"u{user_slot}_{group[0]['_dt'].strftime('%Y%m%dT%H%M%S')}"
                ihb_count = sum(r["ihb"] for r in group)
                afib = 1 if (len(group) >= 3 and ihb_count >= 2) else 0
                for seq, r in enumerate(group, start=1):
                    r["triplet_id"] = triplet_id
                    r["triplet_seq"] = seq
                    r["afib_candidate"] = afib
            else:
                group[0]["triplet_id"] = None
                group[0]["triplet_seq"] = None
                group[0]["afib_candidate"] = 0
            i = j + 1


def parse_csv(csv_path: Path, user_slot: int) -> list[dict[str, Any]]:
    """Read omblepy's per-user CSV. Returns rows with parsed datetime."""
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or set(reader.fieldnames) != CSV_FIELDNAMES:
            raise ValueError(
                f"Unexpected CSV columns in {csv_path.name}: got {reader.fieldnames!r}, "
                f"expected {CSV_FIELDNAMES!r}"
            )
        for raw in reader:
            dt = datetime.fromisoformat(raw["datetime"])
            sys_v = int(raw["sys"])
            dia_v = int(raw["dia"])
            bpm = int(raw["bpm"])
            ihb = int(raw["ihb"])
            mov = int(raw["mov"])
            rows.append({
                "_dt": dt,
                "datetime": dt.isoformat(),
                "user_slot": user_slot,
                "sys": sys_v,
                "dia": dia_v,
                "bpm": bpm,
                "ihb": ihb,
                "mov": mov,
            })
    return rows


def enrich(rows: list[dict[str, Any]], now: datetime) -> None:
    """Mutates rows in-place: adds derived and artifact fields."""
    for r in rows:
        sys_v, dia_v = r["sys"], r["dia"]
        r["map_mmhg"] = round(dia_v + (sys_v - dia_v) / 3.0, 1)
        r["pulse_pressure"] = sys_v - dia_v
        r["am_pm"] = classify_am_pm(r["_dt"])
        artifact, reason = is_artifact(sys_v, dia_v, r["bpm"], r["_dt"], now)
        r["is_artifact"] = 1 if artifact else 0
        r["artifact_reason"] = reason
    group_triplets(rows)


def insert_readings(
    conn: sqlite3.Connection, rows: list[dict[str, Any]], source_file: str
) -> tuple[int, int]:
    """Insert readings; returns (inserted, skipped_duplicates)."""
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for r in rows:
        try:
            cur.execute(
                """
                INSERT INTO omron_bp_readings (
                    datetime, user_slot, sys, dia, bpm, ihb, mov,
                    map_mmhg, pulse_pressure, am_pm,
                    triplet_id, triplet_seq, afib_candidate,
                    is_artifact, artifact_reason, source_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["datetime"], r["user_slot"], r["sys"], r["dia"], r["bpm"],
                    r["ihb"], r["mov"], r["map_mmhg"], r["pulse_pressure"], r["am_pm"],
                    r.get("triplet_id"), r.get("triplet_seq"), r.get("afib_candidate", 0),
                    r["is_artifact"], r["artifact_reason"], source_file,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1
    conn.commit()
    return inserted, skipped


def record_session(
    conn: sqlite3.Connection, session_json: Path | None, csv_name: str, reading_count: int
) -> None:
    """Persist session metadata if the runner wrote a companion .json."""
    meta: dict[str, Any] = {}
    if session_json and session_json.exists():
        try:
            meta = json.loads(session_json.read_text())
        except json.JSONDecodeError:
            log.warning("Session JSON malformed: %s", session_json)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO omron_bp_sessions (
                pulled_at, device_model, device_mac, device_serial,
                firmware_version, manufacturer, hardware_revision, software_revision,
                system_id, battery_pct, reading_count, source_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.get("pulled_at", datetime.now().isoformat()),
                meta.get("device_model"),
                meta.get("device_mac"),
                meta.get("device_serial"),
                meta.get("firmware_version"),
                meta.get("manufacturer"),
                meta.get("hardware_revision"),
                meta.get("software_revision"),
                meta.get("system_id"),
                meta.get("battery_pct"),
                reading_count,
                csv_name,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Already recorded
        pass


def process_inbox(inbox: Path, conn: sqlite3.Connection) -> dict[str, int]:
    """Process every *.csv in the inbox. Returns summary stats."""
    stats = {"files": 0, "inserted": 0, "duplicates": 0, "artifacts": 0}
    if not inbox.exists():
        log.warning("Inbox does not exist: %s", inbox)
        return stats

    ingested_dir = inbox / "ingested"
    ingested_dir.mkdir(exist_ok=True)

    now = datetime.now()
    for csv_path in sorted(inbox.glob("*.csv")):
        if csv_path.name.startswith("_"):
            continue  # skip test artifacts

        # omblepy convention: user1.csv = slot 1, user2.csv = slot 2
        # Our runner convention: omron_YYYYMMDD_HHMMSS_user1.csv etc.
        # Detect user slot from filename.
        name = csv_path.stem.lower()
        if "user2" in name:
            user_slot = 2
        elif "user1" in name:
            user_slot = 1
        else:
            log.warning("Can't determine user slot from %s; defaulting to 1", csv_path.name)
            user_slot = 1

        try:
            rows = parse_csv(csv_path, user_slot)
        except ValueError as e:
            log.error("Skipping %s: %s", csv_path.name, e)
            continue
        if not rows:
            log.info("Empty CSV (no readings): %s", csv_path.name)
            # Still move to ingested so we don't re-process
            shutil.move(str(csv_path), str(ingested_dir / csv_path.name))
            stats["files"] += 1
            continue

        enrich(rows, now)
        inserted, duplicates = insert_readings(conn, rows, csv_path.name)

        # Session metadata lives in a sibling .json with the same stem prefix
        session_json = csv_path.with_suffix(".json")
        if not session_json.exists():
            # Try stripping the _userN suffix
            base = csv_path.stem.rsplit("_user", 1)[0]
            alt = csv_path.parent / f"{base}.json"
            session_json = alt if alt.exists() else None

        record_session(conn, session_json, csv_path.name, len(rows))

        artifact_count = sum(1 for r in rows if r["is_artifact"])
        stats["files"] += 1
        stats["inserted"] += inserted
        stats["duplicates"] += duplicates
        stats["artifacts"] += artifact_count

        log.info(
            "%s: %d rows (user %d), inserted %d, dup %d, artifacts %d",
            csv_path.name, len(rows), user_slot, inserted, duplicates, artifact_count,
        )

        # Archive the CSV
        shutil.move(str(csv_path), str(ingested_dir / csv_path.name))

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, choices=list(PROFILES.keys()),
                        help="Use profile's database (e.g. 'mitch'). Default: config.DATABASE_PATH.")
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX,
                        help=f"Directory to scan for omblepy CSVs (default: {DEFAULT_INBOX})")
    parser.add_argument("--db", type=Path, default=None,
                        help="Override database path (wins over --profile)")
    args = parser.parse_args()

    if args.db:
        db_path = args.db
    elif args.profile:
        db_path = Path(PROFILES[args.profile]["database"])
    else:
        db_path = DATABASE_PATH
    log.info("Profile=%s  DB=%s  Inbox=%s", args.profile or "(default)", db_path, args.inbox)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_database(db_path)
    try:
        stats = process_inbox(args.inbox, conn)
    finally:
        conn.close()

    log.info(
        "Done. %d files, %d inserted, %d duplicates, %d artifacts",
        stats["files"], stats["inserted"], stats["duplicates"], stats["artifacts"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
