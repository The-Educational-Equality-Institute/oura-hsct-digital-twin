"""Shared helpers for device-data importers.

All device importers (glucose, ECG, ABPM, SpO2, and any future CSV-based
importers) follow the same pattern: bytes → text decode → CSV parse with
flexible column detection → typed records → idempotent upsert. These helpers
centralize the cross-cutting parts so a bug fix (e.g. the 0-is-falsy column
detection pitfall) only needs one change.

Intentionally NOT shared (each importer keeps its own):
    - timestamp parsers (vendor date-format quirks vary)
    - schema init / table DDL
    - vendor-specific artifact detection
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH


# ---------------------------------------------------------------------------
# Text decoding
# ---------------------------------------------------------------------------
def decode_bytes(raw: bytes) -> str:
    """Decode bytes using common vendor encodings.

    Tries utf-8-sig (handles BOM common on Windows), utf-8, cp1252, latin-1.
    Raises ValueError if none succeed.
    """
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode bytes as utf-8 / cp1252 / latin-1")


# ---------------------------------------------------------------------------
# Flexible column detection
# ---------------------------------------------------------------------------
def find_any(header_lower: list[str], *needles: str) -> int | None:
    """Return first column index where every needle substring appears."""
    for i, col in enumerate(header_lower):
        if all(n in col for n in needles):
            return i
    return None


def find_col(header_lower: list[str], *alternatives: tuple[str, ...]) -> int | None:
    """Try each AND-group of needles; return first match.

    Using tuples avoids the 0-is-falsy bug of chaining ``a or b`` when the
    matching column sits at index 0 of the header row.

    >>> find_col(["timestamp", "value"], ("timestamp",), ("date",))
    0
    """
    for alt in alternatives:
        idx = find_any(header_lower, *alt)
        if idx is not None:
            return idx
    return None


# ---------------------------------------------------------------------------
# Cell extraction + typed parsing (accepts comma-decimal European format)
# ---------------------------------------------------------------------------
def get_cell(row: list[str], idx: int | None) -> str | None:
    """Safely extract and trim a cell value, returning None for missing/empty."""
    if idx is None or idx >= len(row):
        return None
    val = row[idx].strip()
    return val or None


def int_or_none(v: str | None) -> int | None:
    """Parse optional integer, accepting comma decimals (e.g. '72,0' → 72)."""
    if not v:
        return None
    try:
        return int(float(v.replace(",", ".")))
    except ValueError:
        return None


def float_or_none(v: str | None) -> float | None:
    """Parse optional float, accepting comma decimals (e.g. '5,6' → 5.6)."""
    if not v:
        return None
    try:
        return float(v.replace(",", "."))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Database path resolution
# ---------------------------------------------------------------------------
def resolve_db_path(
    db_override: str | Path | None = None,
    profile_name: str | None = None,
) -> str:
    """Resolve the target SQLite database path for an ingest run.

    Precedence (highest first):
      1. explicit ``db_override`` (CLI --db flag)
      2. ``profile_name`` → profiles.PROFILES[name]["database"]
      3. ``config.DATABASE_PATH``

    Raises SystemExit with an actionable message if ``profile_name`` doesn't
    exist in the PROFILES dict.
    """
    if db_override:
        return str(db_override)
    if profile_name:
        from profiles import PROFILES

        profile = PROFILES.get(profile_name)
        if profile is None:
            raise SystemExit(
                f"Error: unknown profile '{profile_name}'. "
                f"Available: {', '.join(PROFILES.keys())}"
            )
        return str(profile["database"])
    return str(DATABASE_PATH)
