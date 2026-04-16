"""Canonical analytic frames for the digital twin.

Single-point data access for all analysis scripts. Replaces the pattern of
per-script SQL + ad-hoc pandas resampling with named frames at declared cadences:

    load_frame("1d", start, end, cols=[...])   # daily civic-date aligned
    load_frame("1h", start, end, cols=[...])   # hourly
    # frame_1min deferred until CGM + Checkme SpO2 data land

Internally:
  * Raw data lives in data/oura.db (unchanged)
  * DuckDB opens oura.db via the sqlite_scanner extension (read-only, zero-copy)
  * Analytic views + baseline statistics are persisted in data/analytic.duckdb
  * Daily frame is built lazily on first call and cached in the analytic.duckdb
    as table `frame_1d` with an `rebuilt_at` sentinel row

Baseline z-scoring is frozen at [DATA_START, TREATMENT_START) - the pre-rux
window - so that downstream models never leak post-treatment variance into
"normal" references.

Cadence rules (enforced via assert_cadence):
  "1d" - one row per civic date (local TZ, see PATIENT_TIMEZONE in config)
  "1h" - one row per hour, UTC-start aligned

Design note: follows the plan in build-prompts/13-multimodal-fusion.md (section
2.1 option B: DuckDB over SQLite). Writes NEVER go through this module.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH, DATA_START, TREATMENT_START  # noqa: E402

ANALYTIC_DB = DATABASE_PATH.parent / "analytic.duckdb"

# Frame cadence keys
CADENCE_1D = "1d"
CADENCE_1H = "1h"
CADENCES = (CADENCE_1D, CADENCE_1H)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def _connect_analytic(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open DuckDB and attach oura.db read-only via sqlite_scanner."""
    con = duckdb.connect(str(ANALYTIC_DB), read_only=read_only)
    con.execute("INSTALL sqlite")
    con.execute("LOAD sqlite")
    # Attach SQLite read-only so we can't accidentally corrupt raw data
    con.execute(f"ATTACH '{DATABASE_PATH}' AS raw (TYPE sqlite, READ_ONLY)")
    return con


# ---------------------------------------------------------------------------
# Frame builders - idempotent, safe to call repeatedly
# ---------------------------------------------------------------------------
_FRAME_1D_SQL = """
CREATE OR REPLACE TABLE frame_1d AS
WITH
    oura_read AS (
        SELECT
            CAST(date AS DATE) AS date,
            score               AS oura_readiness_score,
            temperature_deviation AS oura_tempdev,
            resting_heart_rate  AS oura_hr_rest,
            hrv_balance         AS oura_hrv_balance
        FROM raw.oura_readiness
    ),
    oura_sleep AS (
        SELECT
            CAST(date AS DATE) AS date,
            score                     AS oura_sleep_score,
            efficiency                AS oura_sleep_efficiency,
            total_sleep_duration      AS oura_sleep_sec,
            rem_sleep_duration        AS oura_rem_sec,
            deep_sleep_duration       AS oura_deep_sec,
            light_sleep_duration      AS oura_light_sec,
            awake_time                AS oura_awake_sec,
            hr_lowest                 AS oura_hr_lowest,
            hr_average                AS oura_hr_sleep_avg,
            hrv_average               AS oura_rmssd_sleep_avg,
            breath_average            AS oura_breath_avg,
            temperature_delta         AS oura_temp_delta
        FROM raw.oura_sleep
    ),
    oura_spo2 AS (
        SELECT
            CAST(date AS DATE)              AS date,
            spo2_average                    AS oura_spo2_avg,
            breathing_disturbance_index     AS oura_bdi
        FROM raw.oura_spo2
    ),
    oura_activity AS (
        SELECT
            CAST(date AS DATE) AS date,
            score              AS oura_activity_score,
            steps              AS oura_steps,
            active_calories    AS oura_active_cal,
            low_activity_time  AS oura_low_active_sec,
            medium_activity_time AS oura_med_active_sec,
            high_activity_time AS oura_high_active_sec
        FROM raw.oura_activity
    ),
    oura_stress AS (
        SELECT
            CAST(date AS DATE) AS date,
            stress_high   AS oura_stress_high,
            recovery_high AS oura_recovery_high
        FROM raw.oura_stress
    ),
    bp_daily AS (
        SELECT
            CAST(datetime AS DATE) AS date,
            COUNT(*)        AS bp_reading_count,
            AVG(sys)        AS bp_sys_mean,
            AVG(dia)        AS bp_dia_mean,
            AVG(bpm)        AS bp_pulse_mean,
            MAX(sys)        AS bp_sys_max,
            MAX(dia)        AS bp_dia_max,
            SUM(ihb)        AS bp_ihb_count,
            AVG(CAST(ihb AS DOUBLE)) AS bp_ihb_rate,
            AVG(map_mmhg)   AS bp_map_mean,
            AVG(pulse_pressure) AS bp_pp_mean,
            MAX(CASE WHEN afib_candidate = 1 THEN 1 ELSE 0 END) AS bp_afib_candidate_day
        FROM raw.omron_bp_readings
        WHERE is_artifact = 0
        GROUP BY 1
    ),
    bp_am AS (
        SELECT
            CAST(datetime AS DATE) AS date,
            AVG(sys) AS bp_sys_am_mean,
            AVG(dia) AS bp_dia_am_mean,
            AVG(bpm) AS bp_pulse_am_mean,
            COUNT(*) AS bp_am_n
        FROM raw.omron_bp_readings
        WHERE is_artifact = 0 AND am_pm = 'morning'
        GROUP BY 1
    ),
    bp_pm AS (
        SELECT
            CAST(datetime AS DATE) AS date,
            AVG(sys) AS bp_sys_pm_mean,
            AVG(dia) AS bp_dia_pm_mean,
            AVG(bpm) AS bp_pulse_pm_mean,
            COUNT(*) AS bp_pm_n
        FROM raw.omron_bp_readings
        WHERE is_artifact = 0 AND am_pm = 'evening'
        GROUP BY 1
    ),
    date_spine AS (
        SELECT DISTINCT date FROM oura_read
        UNION SELECT DISTINCT date FROM oura_sleep
        UNION SELECT DISTINCT date FROM oura_spo2
        UNION SELECT DISTINCT date FROM oura_activity
        UNION SELECT DISTINCT date FROM bp_daily
    )
SELECT
    d.date,
    r.oura_readiness_score, r.oura_tempdev, r.oura_hr_rest, r.oura_hrv_balance,
    s.oura_sleep_score, s.oura_sleep_efficiency, s.oura_sleep_sec,
    s.oura_rem_sec, s.oura_deep_sec, s.oura_light_sec, s.oura_awake_sec,
    s.oura_hr_lowest, s.oura_hr_sleep_avg, s.oura_rmssd_sleep_avg,
    s.oura_breath_avg, s.oura_temp_delta,
    sp.oura_spo2_avg, sp.oura_bdi,
    act.oura_activity_score, act.oura_steps, act.oura_active_cal,
    act.oura_low_active_sec, act.oura_med_active_sec, act.oura_high_active_sec,
    st.oura_stress_high, st.oura_recovery_high,
    b.bp_reading_count, b.bp_sys_mean, b.bp_dia_mean, b.bp_pulse_mean,
    b.bp_sys_max, b.bp_dia_max, b.bp_ihb_count, b.bp_ihb_rate,
    b.bp_map_mean, b.bp_pp_mean, b.bp_afib_candidate_day,
    bam.bp_sys_am_mean, bam.bp_dia_am_mean, bam.bp_pulse_am_mean, bam.bp_am_n,
    bpm.bp_sys_pm_mean, bpm.bp_dia_pm_mean, bpm.bp_pulse_pm_mean, bpm.bp_pm_n
FROM date_spine d
LEFT JOIN oura_read     r  ON d.date = r.date
LEFT JOIN oura_sleep    s  ON d.date = s.date
LEFT JOIN oura_spo2     sp ON d.date = sp.date
LEFT JOIN oura_activity act ON d.date = act.date
LEFT JOIN oura_stress   st ON d.date = st.date
LEFT JOIN bp_daily      b  ON d.date = b.date
LEFT JOIN bp_am         bam ON d.date = bam.date
LEFT JOIN bp_pm         bpm ON d.date = bpm.date
ORDER BY d.date
"""

_FRAME_1H_SQL = """
CREATE OR REPLACE TABLE frame_1h AS
WITH
    hr_hour AS (
        SELECT
            date_trunc('hour', CAST(timestamp AS TIMESTAMP)) AS ts_hour,
            AVG(bpm) AS oura_hr_mean,
            MIN(bpm) AS oura_hr_min,
            MAX(bpm) AS oura_hr_max,
            STDDEV_POP(CAST(bpm AS DOUBLE)) AS oura_hr_sd,
            COUNT(*) AS oura_hr_n
        FROM raw.oura_heart_rate
        GROUP BY 1
    ),
    hrv_hour AS (
        SELECT
            date_trunc('hour', CAST(timestamp AS TIMESTAMP)) AS ts_hour,
            AVG(rmssd) AS oura_rmssd_mean,
            MIN(rmssd) AS oura_rmssd_min,
            MAX(rmssd) AS oura_rmssd_max,
            COUNT(*)   AS oura_rmssd_n
        FROM raw.oura_hrv
        GROUP BY 1
    ),
    bp_hour AS (
        SELECT
            date_trunc('hour', CAST(datetime AS TIMESTAMP)) AS ts_hour,
            AVG(sys) AS bp_sys_mean,
            AVG(dia) AS bp_dia_mean,
            AVG(bpm) AS bp_pulse_mean,
            MAX(ihb) AS bp_ihb_flag,
            COUNT(*) AS bp_n
        FROM raw.omron_bp_readings
        WHERE is_artifact = 0
        GROUP BY 1
    )
SELECT
    COALESCE(h.ts_hour, v.ts_hour, b.ts_hour) AS ts_hour,
    h.oura_hr_mean, h.oura_hr_min, h.oura_hr_max, h.oura_hr_sd, h.oura_hr_n,
    v.oura_rmssd_mean, v.oura_rmssd_min, v.oura_rmssd_max, v.oura_rmssd_n,
    b.bp_sys_mean, b.bp_dia_mean, b.bp_pulse_mean, b.bp_ihb_flag, b.bp_n
FROM hr_hour h
FULL OUTER JOIN hrv_hour v USING (ts_hour)
FULL OUTER JOIN bp_hour  b USING (ts_hour)
ORDER BY ts_hour
"""

_BASELINES_SQL = f"""
CREATE OR REPLACE TABLE v_baselines AS
SELECT
    'frame_1d' AS frame,
    col,
    baseline_mean,
    baseline_sd,
    baseline_n,
    window_start,
    window_end,
    CURRENT_TIMESTAMP AS computed_at
FROM (
    SELECT UNNEST(['oura_rmssd_sleep_avg','oura_hr_rest','oura_hr_lowest',
                   'oura_sleep_efficiency','oura_spo2_avg','oura_temp_delta',
                   'bp_sys_mean','bp_dia_mean','bp_pulse_mean']) AS col
) cols,
LATERAL (
    SELECT
        AVG(v) AS baseline_mean,
        STDDEV_POP(v) AS baseline_sd,
        COUNT(v) AS baseline_n,
        DATE '{DATA_START}' AS window_start,
        DATE '{TREATMENT_START}' AS window_end
    FROM (
        SELECT CASE cols.col
            WHEN 'oura_rmssd_sleep_avg' THEN oura_rmssd_sleep_avg
            WHEN 'oura_hr_rest'         THEN oura_hr_rest
            WHEN 'oura_hr_lowest'       THEN oura_hr_lowest
            WHEN 'oura_sleep_efficiency' THEN oura_sleep_efficiency
            WHEN 'oura_spo2_avg'        THEN oura_spo2_avg
            WHEN 'oura_temp_delta'      THEN oura_temp_delta
            WHEN 'bp_sys_mean'          THEN bp_sys_mean
            WHEN 'bp_dia_mean'          THEN bp_dia_mean
            WHEN 'bp_pulse_mean'        THEN bp_pulse_mean
        END AS v
        FROM frame_1d
        WHERE date >= DATE '{DATA_START}'
          AND date <  DATE '{TREATMENT_START}'
    )
)
"""


def rebuild(verbose: bool = False) -> None:
    """Rebuild frame_1d, frame_1h, and v_baselines in analytic.duckdb.

    Safe to call repeatedly. Writes happen inside a single transaction.
    """
    con = _connect_analytic(read_only=False)
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(_FRAME_1D_SQL)
        con.execute(_FRAME_1H_SQL)
        con.execute(_BASELINES_SQL)
        con.execute("""
            CREATE OR REPLACE TABLE frame_meta AS
            SELECT CURRENT_TIMESTAMP AS rebuilt_at
        """)
        con.execute("COMMIT")
        if verbose:
            n1d = con.execute("SELECT COUNT(*) FROM frame_1d").fetchone()[0]
            n1h = con.execute("SELECT COUNT(*) FROM frame_1h").fetchone()[0]
            nb = con.execute("SELECT COUNT(*) FROM v_baselines").fetchone()[0]
            print(f"Built frame_1d={n1d} rows, frame_1h={n1h} rows, baselines={nb}")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------
def _ensure_built() -> None:
    """Build analytic tables if they don't exist yet."""
    con = _connect_analytic(read_only=False)
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'frame_1d'"
        ).fetchone()[0]
        if not exists:
            con.close()
            rebuild(verbose=False)
    finally:
        try:
            con.close()
        except Exception:
            pass


def load_frame(
    cadence: str,
    start: Optional[date | str] = None,
    end: Optional[date | str] = None,
    cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load an analytic frame at the requested cadence.

    Args:
        cadence: "1d" or "1h"
        start: inclusive lower bound (date for 1d, datetime for 1h; None = no bound)
        end: exclusive upper bound (None = no bound)
        cols: subset of columns to return (None = all). `date` or `ts_hour`
              is always included.

    Returns:
        pandas DataFrame with DatetimeIndex keyed by date (1d) or ts_hour (1h).
    """
    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}, got {cadence!r}")
    _ensure_built()

    table = f"frame_{cadence}"
    key = "date" if cadence == CADENCE_1D else "ts_hour"
    where = []
    params: list = []
    if start is not None:
        where.append(f"{key} >= ?")
        params.append(start)
    if end is not None:
        where.append(f"{key} < ?")
        params.append(end)
    wclause = ("WHERE " + " AND ".join(where)) if where else ""

    con = _connect_analytic(read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {table} {wclause} ORDER BY {key}", params).df()
    finally:
        con.close()

    if cols is not None:
        cols_list = list(cols)
        keep = [key] + [c for c in cols_list if c != key and c in df.columns]
        df = df[keep]
    df = df.set_index(key)
    return df


def load_baselines() -> pd.DataFrame:
    """Return the frozen pre-treatment baseline means and SDs."""
    _ensure_built()
    con = _connect_analytic(read_only=True)
    try:
        return con.execute("SELECT * FROM v_baselines").df()
    finally:
        con.close()


def with_baseline_z(df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Add z-scored versions of the given columns, using the frozen baselines.

    Columns that are not in the baselines table are skipped silently (they can
    still be used untransformed). The z-score columns are named `<col>_z`.
    """
    baselines = load_baselines().set_index("col")
    out = df.copy()
    target = list(cols) if cols else [c for c in df.columns if c in baselines.index]
    for col in target:
        if col not in baselines.index or col not in df.columns:
            continue
        mean = baselines.loc[col, "baseline_mean"]
        sd = baselines.loc[col, "baseline_sd"]
        if pd.isna(mean) or pd.isna(sd) or sd == 0:
            continue
        out[f"{col}_z"] = (df[col] - mean) / sd
    return out


def assert_cadence(df: pd.DataFrame, expected: str) -> None:
    """Raise if the DataFrame's index is not at the expected cadence.

    Cheap sanity check before feeding df to a model that assumed a cadence.
    """
    if expected not in CADENCES:
        raise ValueError(f"unknown cadence {expected!r}")
    if len(df) < 2:
        return
    gaps = df.index.to_series().diff().dropna()
    if expected == CADENCE_1D:
        bad = gaps[(gaps < timedelta(hours=20)) | (gaps > timedelta(hours=28))]
        if not bad.empty:
            raise AssertionError(
                f"cadence=1d but found {len(bad)} non-daily gaps (first: {bad.iloc[0]})"
            )
    else:  # 1h
        bad = gaps[(gaps < timedelta(minutes=45)) | (gaps > timedelta(minutes=75))]
        if not bad.empty:
            raise AssertionError(
                f"cadence=1h but found {len(bad)} non-hourly gaps (first: {bad.iloc[0]})"
            )


# ---------------------------------------------------------------------------
# CLI entrypoint - rebuild and summarise
# ---------------------------------------------------------------------------
def _main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of analytic.duckdb")
    parser.add_argument("--show", action="store_true",
                        help="Print a summary of the built frames")
    args = parser.parse_args()

    if args.rebuild or not ANALYTIC_DB.exists():
        rebuild(verbose=True)
    if args.show or True:  # always summarise when run
        for cadence in CADENCES:
            df = load_frame(cadence)
            print(f"\nframe_{cadence}: {len(df)} rows, {len(df.columns)} cols")
            if len(df):
                print(f"  range: {df.index.min()}  ->  {df.index.max()}")
                missing = df.isna().mean().sort_values(ascending=False).head(5)
                print("  top 5 columns by missingness:")
                for c, frac in missing.items():
                    print(f"    {c:40s} {frac*100:5.1f}%")
        baselines = load_baselines()
        print(f"\nv_baselines: {len(baselines)} columns tracked")
        if len(baselines):
            print(baselines[["col", "baseline_mean", "baseline_sd", "baseline_n"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
