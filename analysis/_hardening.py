"""Shared hardening utilities for Oura analysis scripts.

Provides safe database connections, DataFrame validation, and report section
isolation so that one failing section does not kill an entire report.
"""
from __future__ import annotations

import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Database connection guard
# ---------------------------------------------------------------------------

def safe_connect(path: Path, *, read_only: bool = True) -> sqlite3.Connection:
    """Open a SQLite connection with existence and integrity checks.

    Returns a connection on success. Prints a clear error and calls
    sys.exit(1) if the file is missing or corrupt.
    """
    if not path.exists():
        print(f"ERROR: Database file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        uri = f"file:{path}?mode=ro" if read_only else str(path)
        conn = sqlite3.connect(uri, uri=read_only)
        conn.row_factory = sqlite3.Row
        # Quick integrity probe
        conn.execute("SELECT 1")
        return conn
    except sqlite3.DatabaseError as exc:
        print(f"ERROR: Database corrupt or unreadable: {path}\n  {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Safe pd.read_sql wrapper
# ---------------------------------------------------------------------------

def safe_read_sql(
    sql: str,
    conn: sqlite3.Connection,
    *,
    label: str = "",
    required: bool = False,
) -> pd.DataFrame:
    """Run pd.read_sql_query with empty-table and error handling.

    Parameters
    ----------
    sql : str
        SQL query string.
    conn : sqlite3.Connection
        Database connection.
    label : str
        Human-readable table/query label for log messages.
    required : bool
        If True and the table returns 0 rows, print a warning to stderr.

    Returns
    -------
    pd.DataFrame
        The result, which may be empty.
    """
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as exc:
        tag = f" [{label}]" if label else ""
        print(f"WARNING: SQL query failed{tag}: {exc}", file=sys.stderr)
        return pd.DataFrame()

    if df.empty and required:
        tag = f" [{label}]" if label else ""
        print(f"WARNING: Query returned 0 rows{tag}", file=sys.stderr)

    return df


# ---------------------------------------------------------------------------
# Column existence check
# ---------------------------------------------------------------------------

def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    label: str = "",
) -> list[str]:
    """Return the subset of *columns* that actually exist in *df*.

    Logs a warning for any missing columns.
    """
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        tag = f" [{label}]" if label else ""
        print(f"WARNING: Missing columns{tag}: {missing}", file=sys.stderr)
    return present


# ---------------------------------------------------------------------------
# Safe date parsing
# ---------------------------------------------------------------------------

def safe_to_datetime(
    series: pd.Series,
    *,
    utc: bool = False,
    label: str = "",
) -> pd.Series:
    """Parse dates with errors='coerce' so bad values become NaT."""
    try:
        return pd.to_datetime(series, errors="coerce", utc=utc)
    except Exception as exc:
        tag = f" [{label}]" if label else ""
        print(f"WARNING: Date parsing failed{tag}: {exc}", file=sys.stderr)
        return pd.Series(pd.NaT, index=series.index)


# ---------------------------------------------------------------------------
# Division-by-zero guard
# ---------------------------------------------------------------------------

def safe_divide(
    numerator: Any,
    denominator: Any,
    *,
    fill: float = 0.0,
) -> Any:
    """Element-wise division that replaces div-by-zero with *fill*.

    Works with scalars, numpy arrays, and pandas Series.
    """
    if isinstance(denominator, (pd.Series, pd.DataFrame)):
        denom = denominator.replace(0, np.nan)
        result = numerator / denom
        return result.fillna(fill)
    elif isinstance(denominator, np.ndarray):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denominator != 0, numerator / denominator, fill)
        return result
    else:
        # scalar
        if denominator == 0:
            return fill
        return numerator / denominator


# ---------------------------------------------------------------------------
# Report section isolation
# ---------------------------------------------------------------------------

def safe_section(
    name: str,
    func: Callable[..., Any],
    *args: Any,
    fallback: Any = None,
    **kwargs: Any,
) -> Any:
    """Call *func* inside a try/except, returning *fallback* on failure.

    On error, prints a clear message with the section name and traceback.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        print(f"WARNING: Section '{name}' failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return fallback


def section_html_or_placeholder(
    name: str,
    func: Callable[..., str],
    *args: Any,
    **kwargs: Any,
) -> str:
    """Call *func* to produce an HTML snippet; on failure return a placeholder."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        print(f"WARNING: HTML section '{name}' failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return (
            f'<div style="padding:20px;margin:20px 0;background:#fff3f3;'
            f'border:2px solid #e63946;border-radius:8px;">'
            f'<h3>Section failed: {name}</h3>'
            f'<p style="color:#666;">{exc}</p></div>'
        )
