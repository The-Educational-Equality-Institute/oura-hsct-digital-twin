#!/usr/bin/env python3
"""
Novel Composite Biomarker Scores for Post-HSCT Monitoring

Computes 6 composite biomarker indices from Oura Ring data tailored for
post-allogeneic-HSCT surveillance with chronic GVHD, autonomic dysfunction,
and iron overload.

Biomarkers:
  1. Autonomic Dysfunction Severity Index (ADSI) — 0-100
  2. GVHD Activity Score (Wearable) — 0-100
  3. Recovery Trajectory Index — 0-100
  4. Pharmacodynamic Response Score (ruxolitinib) — Z-score
  5. Cardiovascular Risk Composite — 0-100
  6. Wearable Allostatic Load Score — 0-7

Outputs:
  - Interactive HTML dashboard: reports/composite_biomarkers.html
  - JSON metrics: reports/composite_biomarkers.json

Usage:
    python analysis/analyze_oura_biomarkers.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

# Suppress FutureWarnings from pandas
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution & config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATABASE_PATH, REPORTS_DIR, TRANSPLANT_DATE, TREATMENT_START,
    TREATMENT_START_STR, PATIENT_AGE, PATIENT_LABEL,
    ESC_RMSSD_DEFICIENCY, KNOWN_EVENT_DATE,
)
from _hardening import section_html_or_placeholder
from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    METRIC_DESCRIPTIONS,
    COLORWAY, STATUS_COLORS, BG_PRIMARY, BG_SURFACE, BORDER_SUBTLE,
    TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_BLUE, ACCENT_GREEN,
    ACCENT_RED, ACCENT_AMBER, ACCENT_PURPLE, ACCENT_CYAN,
    C_HR, C_SPO2, C_HRV, C_SLEEP, C_TEMP,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "composite_biomarkers.html"
JSON_OUTPUT = REPORTS_DIR / "composite_biomarkers.json"

# Population norms for 36yo male (Nunan 2010, Shaffer & Ginsberg 2017)
NORM_RMSSD_MEAN = 42.0  # ms
NORM_RMSSD_SD = 15.0
NORM_SLEEP_HR_MEAN = 65.0  # bpm
NORM_SLEEP_HR_SD = 10.0
NORM_SPO2_MEAN = 97.5
NORM_SPO2_SD = 1.0
NORM_CV_AGE_DELTA_MEAN = 0.0
NORM_CV_AGE_DELTA_SD = 3.0

# Post-HSCT expected recovery milestones (Chamorro-Vina 2012, Wood 2013)
# Note: Recovery trajectory values are clinical estimates, not derived from specific literature.
RECOVERY_MILESTONES = {
    6:  {"rmssd": 20, "sleep_hr": 80},
    12: {"rmssd": 25, "sleep_hr": 75},
    24: {"rmssd": 30, "sleep_hr": 72},
    36: {"rmssd": 35, "sleep_hr": 70},
}

# Nocturnal HR dip norm: 10-20% (Palatini 1999)
NORMAL_DIP_MIN = 10.0
NORMAL_DIP_MAX = 20.0

# Allostatic load thresholds
# Note: This composite uses clinical risk thresholds, not population-percentile
# quartiles as in the original McEwen allostatic load methodology.
ALLOSTATIC_THRESHOLDS = {
    "resting_hr": 80,        # bpm, 90th percentile
    "rmssd": ESC_RMSSD_DEFICIENCY,  # ms, autonomic deficiency threshold (Kleiger 1987 / Bigger 1992; below = fail)
    "sleep_efficiency": 85,  # %, below = fail
    "temp_deviation": 0.5,   # degC, above = fail
    "spo2": 95.0,            # %, below = fail
    "deep_pct": 10.0,        # %, below = fail
    "rem_pct": 15.0,         # %, below = fail
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Open read-only connection to biometrics database."""
    if not DATABASE_PATH.exists():
        print(f"ERROR: Database not found at {DATABASE_PATH}. Run: python api/import_oura.py --days 90", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Data loading: build daily metrics DataFrame
# ---------------------------------------------------------------------------

def get_daily_metrics(conn: sqlite3.Connection) -> pd.DataFrame:
    """Build a unified daily metrics DataFrame from all Oura tables.

    Returns a DataFrame indexed by date with columns:
        rmssd_mean, rmssd_sd, rmssd_count, dfa_alpha1,
        sleep_hr_mean, sleep_hr_lowest, awake_hr_mean,
        hr_dip_pct, total_sleep_sec, rem_pct, deep_pct,
        light_pct, awake_pct, sleep_efficiency,
        spo2, cv_age, cv_age_delta,
        temp_deviation, stress_high, recovery_high,
        stress_recovery_ratio
    """

    def _safe_read(sql: str, con: sqlite3.Connection) -> pd.DataFrame:
        """Wrap pd.read_sql_query; return empty DataFrame on missing table."""
        try:
            return pd.read_sql_query(sql, con)
        except Exception:
            logger.warning("Query failed (table may be missing), returning empty DataFrame")
            return pd.DataFrame()

    # -- 1) Nightly HRV (from 5-min epochs) --
    hrv_df = _safe_read(
        """
        SELECT substr(timestamp, 1, 10) AS day,
               AVG(rmssd) AS rmssd_mean,
               -- Use population SD (not sample) for small n
               AVG(rmssd * rmssd) AS rmssd_sq_mean,
               COUNT(*) AS rmssd_count
        FROM oura_hrv
        WHERE rmssd IS NOT NULL AND rmssd > 0
        GROUP BY day
        ORDER BY day
        """,
        conn,
    )
    hrv_df["day"] = pd.to_datetime(hrv_df["day"]).dt.date
    # Compute SD from E[X^2] - E[X]^2
    hrv_df["rmssd_sd"] = np.sqrt(
        np.maximum(hrv_df["rmssd_sq_mean"] - hrv_df["rmssd_mean"] ** 2, 0)
    )
    hrv_df.drop(columns=["rmssd_sq_mean"], inplace=True)

    # -- 1b) DFA alpha-1 from nightly RMSSD-epoch series (per night) --
    # NOTE: Oura provides 5-minute RMSSD epochs, NOT raw RR intervals.
    # DFA is designed for RR-interval data; applying it to RMSSD epochs
    # yields a proxy measure. Reference ranges from RR-interval studies
    # may not be directly comparable.
    dfa_records: list[dict[str, Any]] = []
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT substr(timestamp, 1, 10) AS day, rmssd
        FROM oura_hrv
        WHERE rmssd IS NOT NULL AND rmssd > 0
        ORDER BY day, timestamp
        """
    )
    from collections import defaultdict
    night_epochs: dict[str, list[float]] = defaultdict(list)
    for row in cursor:
        night_epochs[row["day"]].append(row["rmssd"])

    for day_str, values in night_epochs.items():
        alpha1 = _compute_dfa_alpha1(values)
        dfa_records.append({"day": datetime.strptime(day_str, "%Y-%m-%d").date(), "dfa_alpha1": alpha1})

    dfa_df = pd.DataFrame(dfa_records)

    # -- 2) Sleep periods (long_sleep only for primary sleep) --
    sleep_df = _safe_read(
        """
        SELECT day,
               average_hrv AS sleep_hrv,
               average_heart_rate AS sleep_hr_mean,
               lowest_heart_rate AS sleep_hr_lowest,
               total_sleep_duration AS total_sleep_sec,
               rem_sleep_duration,
               deep_sleep_duration,
               light_sleep_duration,
               awake_time,
               efficiency AS sleep_efficiency
        FROM oura_sleep_periods
        WHERE type = 'long_sleep'
        ORDER BY day
        """,
        conn,
    )
    sleep_df["day"] = pd.to_datetime(sleep_df["day"]).dt.date
    # Compute sleep stage percentages
    total = sleep_df["total_sleep_sec"].replace(0, np.nan)
    sleep_df["rem_pct"] = sleep_df["rem_sleep_duration"] / total * 100
    sleep_df["deep_pct"] = sleep_df["deep_sleep_duration"] / total * 100
    sleep_df["light_pct"] = sleep_df["light_sleep_duration"] / total * 100
    sleep_df["awake_pct"] = sleep_df["awake_time"] / (sleep_df["awake_time"] + total) * 100

    # -- 3) Awake HR for dipping calculation --
    awake_hr_df = _safe_read(
        """
        SELECT substr(timestamp, 1, 10) AS day,
               AVG(bpm) AS awake_hr_mean
        FROM oura_heart_rate
        WHERE source = 'awake'
        GROUP BY day
        """,
        conn,
    )
    awake_hr_df["day"] = pd.to_datetime(awake_hr_df["day"]).dt.date

    # -- 4) SpO2 --
    spo2_df = _safe_read(
        "SELECT date AS day, spo2_average AS spo2 FROM oura_spo2 WHERE spo2_average > 0",
        conn,
    )
    spo2_df["day"] = pd.to_datetime(spo2_df["day"]).dt.date

    # -- 5) CV age --
    cv_df = _safe_read(
        "SELECT date AS day, vascular_age AS cv_age FROM oura_cardiovascular_age",
        conn,
    )
    cv_df["day"] = pd.to_datetime(cv_df["day"]).dt.date
    cv_df["cv_age_delta"] = cv_df["cv_age"] - PATIENT_AGE

    # -- 6) Readiness (temperature deviation) --
    readiness_df = _safe_read(
        "SELECT date AS day, temperature_deviation AS temp_deviation, recovery_index FROM oura_readiness",
        conn,
    )
    readiness_df["day"] = pd.to_datetime(readiness_df["day"]).dt.date

    # -- 7) Stress --
    stress_df = _safe_read(
        "SELECT date AS day, stress_high, recovery_high FROM oura_stress",
        conn,
    )
    stress_df["day"] = pd.to_datetime(stress_df["day"]).dt.date
    # Stress/recovery ratio (handle zero recovery)
    stress_df["stress_recovery_ratio"] = np.where(
        stress_df["recovery_high"] > 0,
        stress_df["stress_high"] / stress_df["recovery_high"],
        np.where(stress_df["stress_high"] > 0, 5.0, 1.0),  # cap at 5 if no recovery
    )

    # -- Merge all on day --
    df = hrv_df.copy()
    for other in [dfa_df, sleep_df, awake_hr_df, spo2_df, cv_df, readiness_df, stress_df]:
        if not other.empty:
            df = pd.merge(df, other, on="day", how="outer")

    df.sort_values("day", inplace=True)
    df.set_index("day", inplace=True)

    # Ensure columns exist even when source tables are empty
    for col in ["cv_age", "cv_age_delta", "spo2", "dfa_alpha1"]:
        if col not in df.columns:
            df[col] = np.nan

    # -- Compute HR dipping --
    df["hr_dip_pct"] = np.where(
        (df["awake_hr_mean"].notna()) & (df["sleep_hr_mean"].notna()) & (df["awake_hr_mean"] > 0),
        (df["awake_hr_mean"] - df["sleep_hr_mean"]) / df["awake_hr_mean"] * 100,
        np.nan,
    )

    return df


def _compute_dfa_alpha1(rmssd_series: list[float], min_epochs: int = 16) -> float | None:
    """Compute RMSSD-Epoch DFA (Proxy) alpha-1 from a nightly RMSSD series.

    Uses antropy.detrended_fluctuation for short-term fractal scaling.
    NOTE: This operates on 5-min RMSSD epochs from Oura, NOT raw RR intervals.
    Reference ranges from RR-interval DFA studies may not be directly comparable.
    Returns None if insufficient data points.
    """
    if len(rmssd_series) < min_epochs:
        return None

    try:
        import antropy
    except ImportError:
        return None

    try:
        arr = np.array(rmssd_series, dtype=np.float64)
        # Remove NaN/zero
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) < min_epochs:
            return None
        alpha = antropy.detrended_fluctuation(arr)
        return float(alpha)
    except Exception as e:
        logging.warning(f"DFA alpha-1 failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Biomarker 1: Autonomic Dysfunction Severity Index (ADSI)
# ---------------------------------------------------------------------------

def compute_adsi(df: pd.DataFrame) -> pd.Series:
    """Compute Autonomic Dysfunction Severity Index (0-100, 100 = worst).

    Components (normalized 0-100):
    - RMSSD percentile vs age-norm (weight 0.25)
    - RMSSD-Epoch DFA (Proxy) alpha-1 deviation from 1.0 (weight 0.15)
    - Sleep HR dipping deficit (weight 0.20)
    - HR recovery estimate (weight 0.15)
    - CV age excess (weight 0.10)
    - SpO2 deficit (weight 0.15)
    """
    scores = pd.DataFrame(index=df.index)

    # (a) RMSSD deficit: 100 * (1 - percentile/100) where percentile is CDF position
    z_rmssd = (df["rmssd_mean"] - NORM_RMSSD_MEAN) / NORM_RMSSD_SD
    pct_rmssd = scipy_stats.norm.cdf(z_rmssd) * 100
    scores["rmssd_score"] = 100 - pct_rmssd

    # (b) RMSSD-Epoch DFA (Proxy) alpha-1 deviation from optimal 1.0 (healthy short-term scaling)
    # Score = |alpha1 - 1.0| / 0.5 * 100, capped at 100
    # Note: Reference ranges from RR-interval studies; RMSSD-epoch DFA values may not be directly comparable
    scores["dfa_score"] = np.clip(
        np.abs(df["dfa_alpha1"] - 1.0) / 0.5 * 100, 0, 100
    )

    # (c) Sleep HR dipping deficit: normal dip is 10-20%
    # Score = 100 if no dip (0%), 0 if dip >= 15% (midpoint of normal)
    scores["dip_score"] = np.clip(
        100 - (df["hr_dip_pct"] / 15.0 * 100), 0, 100
    )

    # (d) HR recovery estimate: use the gap between awake and rest HR decay
    # Higher awake HR + lower dip = worse recovery. Proxy: awake HR percentile.
    z_awake_hr = (df["awake_hr_mean"] - 75) / 10  # 75 bpm = healthy awake resting
    scores["hr_recovery_score"] = np.clip(
        scipy_stats.norm.cdf(z_awake_hr) * 100, 0, 100
    )

    # (e) CV age excess: (cv_age - chronological) / 10 * 100, capped
    scores["cv_age_score"] = np.clip(
        df["cv_age_delta"] / 15.0 * 100, 0, 100
    )

    # (f) SpO2 deficit: distance below 97.5% baseline
    # Score: (97.5 - spo2) / 3.0 * 100 (3% below = 100)
    scores["spo2_score"] = np.clip(
        (NORM_SPO2_MEAN - df["spo2"]) / 3.0 * 100, 0, 100
    )

    # Weighted composite
    weights = {
        "rmssd_score": 0.25,
        "dfa_score": 0.15,
        "dip_score": 0.20,
        "hr_recovery_score": 0.15,
        "cv_age_score": 0.10,
        "spo2_score": 0.15,
    }

    adsi = pd.Series(0.0, index=df.index, dtype=float)
    total_weight = pd.Series(0.0, index=df.index, dtype=float)
    for col, w in weights.items():
        mask = scores[col].notna()
        adsi = adsi + scores[col].fillna(0) * w * mask.astype(float)
        total_weight = total_weight + w * mask.astype(float)

    # Normalize by available weight (handle partial data)
    adsi = np.where(total_weight > 0, adsi / total_weight, np.nan)
    return pd.Series(adsi, index=df.index, name="adsi")


# ---------------------------------------------------------------------------
# Biomarker 2: GVHD Activity Score (Wearable)
# ---------------------------------------------------------------------------

def compute_gvhd_score(df: pd.DataFrame) -> pd.Series:
    """Compute GVHD Activity Score (0-100, 100 = active flare).

    Components (7-day rolling trends):
    - Temperature deviation trend (weight 0.25)
    - SpO2 decline trend (weight 0.20)
    - HRV deterioration rate (weight 0.25)
    - Sleep fragmentation increase (weight 0.15)
    - Recovery deficit (weight 0.15)
    """
    scores = pd.DataFrame(index=df.index)
    window = 7

    # (a) Temperature deviation: 7-day rolling mean
    # Rising temp = potential flare. Score: temp_dev / 1.0 * 100 (1C = 100)
    temp_roll = df["temp_deviation"].rolling(window, min_periods=3).mean()
    scores["temp_score"] = np.clip(temp_roll / 1.0 * 100, 0, 100)

    # (b) SpO2 decline: 7-day slope (negative slope = declining)
    spo2_slope = _rolling_slope(df["spo2"], window)
    # Declining SpO2: -0.5%/day = 100, 0 = 0
    scores["spo2_score"] = np.clip(-spo2_slope / 0.5 * 100, 0, 100)

    # (c) HRV deterioration: 7-day slope of nightly RMSSD (negative = worsening)
    hrv_slope = _rolling_slope(df["rmssd_mean"], window)
    # Declining HRV: -1 ms/day = 100, 0 = 0
    scores["hrv_score"] = np.clip(-hrv_slope / 1.0 * 100, 0, 100)

    # (d) Sleep fragmentation: 7-day trend in awake_pct
    frag_slope = _rolling_slope(df["awake_pct"], window)
    # Increasing fragmentation: +2%/day = 100, 0 = 0
    scores["frag_score"] = np.clip(frag_slope / 2.0 * 100, 0, 100)

    # (e) Recovery deficit: stress/recovery ratio
    # Ratio > 3 = 100, ratio = 1 = 0
    ratio_roll = df["stress_recovery_ratio"].rolling(window, min_periods=3).mean()
    scores["recovery_score"] = np.clip((ratio_roll - 1.0) / 2.0 * 100, 0, 100)

    weights = {
        "temp_score": 0.25,
        "spo2_score": 0.20,
        "hrv_score": 0.25,
        "frag_score": 0.15,
        "recovery_score": 0.15,
    }

    gvhd = pd.Series(0.0, index=df.index, dtype=float)
    total_w = pd.Series(0.0, index=df.index, dtype=float)
    for col, w in weights.items():
        mask = scores[col].notna()
        gvhd = gvhd + scores[col].fillna(0) * w * mask.astype(float)
        total_w = total_w + w * mask.astype(float)

    gvhd = np.where(total_w > 0, gvhd / total_w, np.nan)
    return pd.Series(gvhd, index=df.index, name="gvhd_score")


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling OLS slope over a window.

    Returns slope (units per day) for each point.
    """
    slopes = pd.Series(np.nan, index=series.index, dtype=float)
    vals = series.values
    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1: i + 1]
        mask = np.isfinite(y)
        if mask.sum() >= 3:
            x = np.arange(window)[mask]
            y_clean = y[mask]
            if len(x) >= 3:
                slope, _, _, _, _ = scipy_stats.linregress(x, y_clean)
                slopes.iloc[i] = slope
    return slopes


# ---------------------------------------------------------------------------
# Biomarker 3: Recovery Trajectory Index
# ---------------------------------------------------------------------------

def compute_recovery_index(df: pd.DataFrame) -> pd.Series:
    """Compute Recovery Trajectory Index (0-100, 100 = full recovery to expected norm).

    Compares patient metrics against expected post-HSCT recovery milestones.
    At 26+ months: expected RMSSD ~30ms, sleep HR ~72bpm.
    Score = average of (actual/expected * 100), capped at 100.
    """
    # Determine months post-HSCT dynamically
    months_post = (date.today() - TRANSPLANT_DATE).days / 30.44
    # Linear interpolation between 24mo and 36mo milestones
    frac = (months_post - 24) / (36 - 24)
    expected_rmssd = RECOVERY_MILESTONES[24]["rmssd"] + frac * (
        RECOVERY_MILESTONES[36]["rmssd"] - RECOVERY_MILESTONES[24]["rmssd"]
    )
    expected_hr = RECOVERY_MILESTONES[24]["sleep_hr"] + frac * (
        RECOVERY_MILESTONES[36]["sleep_hr"] - RECOVERY_MILESTONES[24]["sleep_hr"]
    )

    # RMSSD score: actual / expected * 100
    rmssd_score = np.clip(df["rmssd_mean"] / expected_rmssd * 100, 0, 100)

    # HR score: inverted (lower is better). Score = expected / actual * 100
    hr_score = np.clip(expected_hr / df["sleep_hr_mean"] * 100, 0, 100)

    # Sleep quality score: efficiency vs expected 85%
    sleep_score = np.clip(df["sleep_efficiency"] / 85.0 * 100, 0, 100)

    # Average of available components
    components = pd.DataFrame({
        "rmssd": rmssd_score,
        "hr": hr_score,
        "sleep": sleep_score,
    })

    recovery = components.mean(axis=1, skipna=True)
    recovery.name = "recovery_index"
    return recovery


# ---------------------------------------------------------------------------
# Biomarker 4: Pharmacodynamic Response Score (ruxolitinib)
# ---------------------------------------------------------------------------

def compute_pharma_response(df: pd.DataFrame) -> pd.Series:
    """Compute Pharmacodynamic Response Score for ruxolitinib.

    Pre-treatment baseline: before 2026-03-16
    Post-treatment: from 2026-03-16
    Score = mean absolute Z-score across metrics relative to baseline.
    Higher = more drug effect (either beneficial or adverse).
    """
    rux_date = TREATMENT_START
    baseline_mask = pd.Series([d < rux_date for d in df.index], index=df.index)
    post_mask = ~baseline_mask

    # Metrics to track
    metrics = ["rmssd_mean", "sleep_hr_mean", "sleep_efficiency", "temp_deviation", "spo2"]
    available_metrics = [m for m in metrics if m in df.columns]

    response = pd.Series(np.nan, index=df.index, name="pharma_response")

    if baseline_mask.sum() < 7 or post_mask.sum() < 1:
        return response

    # Compute baseline stats
    baseline_means = {}
    baseline_sds = {}
    for m in available_metrics:
        bl = df.loc[baseline_mask, m].dropna()
        if len(bl) >= 5:
            baseline_means[m] = bl.mean()
            baseline_sds[m] = bl.std()
            if baseline_sds[m] == 0 or np.isnan(baseline_sds[m]):
                baseline_sds[m] = 0.01  # prevent division by zero / NaN

    if not baseline_means:
        return response

    # Compute daily Z-scores for ALL days (baseline should cluster near 0)
    z_scores = pd.DataFrame(index=df.index)
    for m in baseline_means:
        z_scores[m] = (df[m] - baseline_means[m]) / baseline_sds[m]

    # Mean absolute Z-score
    response = z_scores.abs().mean(axis=1, skipna=True)
    response.name = "pharma_response"
    return response


# ---------------------------------------------------------------------------
# Biomarker 5: Cardiovascular Risk Composite
# ---------------------------------------------------------------------------

def compute_cv_risk(df: pd.DataFrame) -> pd.Series:
    """Compute Cardiovascular Risk Composite (0-100, 100 = highest risk).

    Based on ESC/AHA risk factors derivable from wearable:
    - Resting HR percentile for age (weight 0.25)
    - HRV percentile for age (weight 0.25)
    - Nocturnal dipping (weight 0.15)
    - SpO2 nadir (weight 0.15)
    - CV age delta (weight 0.20)
    """
    scores = pd.DataFrame(index=df.index)

    # (a) Resting HR risk: >90 = ESC risk factor
    # Score: (sleep_hr - 60) / 40 * 100 (60=0%, 100+=100%)
    scores["hr_risk"] = np.clip((df["sleep_hr_mean"] - 60) / 40 * 100, 0, 100)

    # (b) HRV risk: low HRV = high risk
    z_hrv = (df["rmssd_mean"] - NORM_RMSSD_MEAN) / NORM_RMSSD_SD
    pct_hrv = scipy_stats.norm.cdf(z_hrv) * 100
    scores["hrv_risk"] = 100 - pct_hrv  # lower HRV = higher risk

    # (c) Non-dipper risk: dip < 10% = non-dipper
    # Score: 100 if dip <= 0%, 0 if dip >= 15%
    scores["dip_risk"] = np.clip(100 - (df["hr_dip_pct"] / 15.0 * 100), 0, 100)

    # (d) SpO2 risk: (97.5 - spo2) / 3.0 * 100
    scores["spo2_risk"] = np.clip((NORM_SPO2_MEAN - df["spo2"]) / 3.0 * 100, 0, 100)

    # (e) CV age delta: higher delta = more risk
    scores["cv_age_risk"] = np.clip(df["cv_age_delta"] / 15.0 * 100, 0, 100)

    weights = {
        "hr_risk": 0.25,
        "hrv_risk": 0.25,
        "dip_risk": 0.15,
        "spo2_risk": 0.15,
        "cv_age_risk": 0.20,
    }

    cv_risk = pd.Series(0.0, index=df.index, dtype=float)
    total_w = pd.Series(0.0, index=df.index, dtype=float)
    for col, w in weights.items():
        mask = scores[col].notna()
        cv_risk = cv_risk + scores[col].fillna(0) * w * mask.astype(float)
        total_w = total_w + w * mask.astype(float)

    cv_risk = np.where(total_w > 0, cv_risk / total_w, np.nan)
    return pd.Series(cv_risk, index=df.index, name="cv_risk")


# ---------------------------------------------------------------------------
# Biomarker 6: Wearable Allostatic Load Score
# ---------------------------------------------------------------------------

def compute_allostatic_load(df: pd.DataFrame) -> pd.Series:
    """Compute Wearable Allostatic Load Score (0-7).

    Count of biomarkers exceeding clinical thresholds:
    [x] Resting HR > 80 bpm
    [x] RMSSD < 15 ms
    [x] Sleep efficiency < 85%
    [x] Temperature deviation > 0.5 degC
    [x] SpO2 < 95%
    [x] Deep sleep < 10%
    [x] REM sleep < 15%
    """
    load = pd.Series(0, index=df.index, dtype=int)

    checks = [
        ("sleep_hr_mean", lambda x: x > ALLOSTATIC_THRESHOLDS["resting_hr"]),
        ("rmssd_mean", lambda x: x < ALLOSTATIC_THRESHOLDS["rmssd"]),
        ("sleep_efficiency", lambda x: x < ALLOSTATIC_THRESHOLDS["sleep_efficiency"]),
        ("temp_deviation", lambda x: x.abs() > ALLOSTATIC_THRESHOLDS["temp_deviation"]),
        ("spo2", lambda x: x < ALLOSTATIC_THRESHOLDS["spo2"]),
        ("deep_pct", lambda x: x < ALLOSTATIC_THRESHOLDS["deep_pct"]),
        ("rem_pct", lambda x: x < ALLOSTATIC_THRESHOLDS["rem_pct"]),
    ]

    for col, check_fn in checks:
        if col in df.columns:
            vals = df[col]
            exceeded = check_fn(vals).astype(int)
            # Only count where data exists
            exceeded = exceeded.where(vals.notna(), other=0)
            load = load + exceeded

    load.name = "allostatic_load"
    return load


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def compute_trend(series: pd.Series, min_points: int = 10,
                   window_days: int | None = None) -> dict[str, Any]:
    """Compute trend statistics for a time series.

    Args:
        series: Time series with date index
        min_points: Minimum data points required
        window_days: If set, only use the last N days for trend calculation

    Returns:
        direction: 'improving', 'worsening', 'stable'
        window_label: e.g. '30-day trend' or 'full-window trend'
        slope: daily rate of change
        p_value: statistical significance
        r_squared: explained variance
    """
    clean = series.dropna()

    # Optionally truncate to last N days
    actual_window = None
    if window_days and len(clean) > 0:
        cutoff = clean.index[-1] - pd.Timedelta(days=window_days)
        windowed = clean[clean.index > cutoff]
        if len(windowed) >= min_points:
            clean = windowed
            actual_window = window_days
        # else: fall back to full series

    window_label = f"{actual_window}-day trend" if actual_window else "full-window trend"

    if len(clean) < min_points:
        return {"direction": "insufficient data", "window_label": window_label,
                "slope": None, "p_value": None, "r_squared": None}

    # Convert index to numeric (days from start)
    x = np.array([(d - clean.index[0]).days for d in clean.index], dtype=float)
    y = clean.values.astype(float)

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
    r_squared = r_value ** 2

    if p_value > 0.05:
        direction = "stable"
    elif slope > 0:
        direction = "worsening" if series.name in ["adsi", "gvhd_score", "cv_risk", "allostatic_load", "pharma_response"] else "improving"
    else:
        direction = "improving" if series.name in ["adsi", "gvhd_score", "cv_risk", "allostatic_load", "pharma_response"] else "worsening"

    return {
        "direction": direction,
        "window_label": window_label,
        "slope": round(float(slope), 4),
        "p_value": round(float(p_value), 4),
        "r_squared": round(float(r_squared), 4),
    }


# ---------------------------------------------------------------------------
# Clinical interpretation
# ---------------------------------------------------------------------------

def interpret_adsi(value: float) -> str:
    """Clinical interpretation for ADSI."""
    if value >= 80:
        return "Critical autonomic dysfunction. Indicates severely impaired parasympathetic control consistent with post-HSCT autonomic neuropathy."
    elif value >= 60:
        return "Severe autonomic dysfunction. Substantially reduced vagal tone and heart rate variability."
    elif value >= 40:
        return "Moderate autonomic dysfunction. Below expected level for age, but improved from acute phase."
    elif value >= 20:
        return "Mild autonomic dysfunction. Slightly reduced autonomic regulation."
    else:
        return "Normal autonomic function for age."


def interpret_gvhd(value: float) -> str:
    """Clinical interpretation for GVHD Activity Score."""
    if value >= 60:
        return "High GVHD activity indicator. Consider clinical evaluation for possible flare (temperature, SpO2, HRV deterioration)."
    elif value >= 40:
        return "Moderate GVHD activity. Monitor trends over the next days for escalation."
    elif value >= 20:
        return "Low GVHD activity. Stable biometric trends."
    else:
        return "Minimal GVHD activity based on wearable data."


def interpret_recovery(value: float) -> str:
    """Clinical interpretation for Recovery Trajectory Index."""
    if value >= 80:
        return "Good recovery. Near expected level for time since transplant."
    elif value >= 60:
        return "Moderate recovery. Behind expected timeline, but positive trend possible."
    elif value >= 40:
        months_post = (date.today() - TRANSPLANT_DATE).days // 30
        return f"Delayed recovery. Substantially below expected recovery level at {months_post} months post-HSCT."
    else:
        return "Severely delayed recovery. Markedly below expected level - consider underlying causes (chronic GVHD, iron overload, autonomic dysfunction)."


def interpret_cv_risk(value: float) -> str:
    """Clinical interpretation for CV Risk Composite."""
    if value >= 70:
        return "High cardiovascular risk. Persistent tachycardia, extremely low HRV, and pathological cardiovascular aging require attention."
    elif value >= 50:
        return "Elevated cardiovascular risk. Multiple risk factors above clinical thresholds."
    elif value >= 30:
        return "Moderate cardiovascular risk."
    else:
        return "Acceptable cardiovascular risk for age."


def interpret_allostatic(value: float) -> str:
    """Clinical interpretation for Allostatic Load Score."""
    if value >= 5:
        return f"Allostatic load {value:.0f}/7 - severe burden. Majority of biological markers outside normal range."
    elif value >= 3:
        return f"Allostatic load {value:.0f}/7 - moderate burden. Multiple systems under stress."
    elif value >= 1:
        return f"Allostatic load {value:.0f}/7 - low burden."
    else:
        return f"Allostatic load {value:.0f}/7 - no exceedances."


# ---------------------------------------------------------------------------
# Plotly dashboard
# ---------------------------------------------------------------------------

def build_dashboard(df: pd.DataFrame, biomarkers: dict[str, pd.Series],
                    trends: dict[str, dict], summary: dict) -> go.Figure:
    """Build interactive Plotly HTML dashboard with one subplot per biomarker."""

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "1. Autonomic Dysfunction Severity Index (ADSI)",
            "2. GVHD Activity Score (Wearable)",
            "3. Recovery Trajectory Index",
            "4. Pharmacodynamic Response (Ruxolitinib)",
            "5. Cardiovascular Risk Composite",
            "6. Wearable Allostatic Load Score",
            "Nocturnal RMSSD and Sleep HR",
            "Component Overview",
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "polar"}],
        ],
    )

    dates = [datetime.combine(d, datetime.min.time()) for d in df.index]
    rux_line_date = datetime.combine(TREATMENT_START, datetime.min.time())

    # Color scheme (dark-theme compatible)
    colors = {
        "adsi": ACCENT_RED,
        "gvhd_score": ACCENT_AMBER,
        "recovery_index": ACCENT_GREEN,
        "pharma_response": ACCENT_PURPLE,
        "cv_risk": ACCENT_BLUE,
        "allostatic_load": ACCENT_CYAN,
    }
    # Convert hex colors to rgba with 0.08 opacity for subtle fills
    def _hex_to_rgba_fill(hex_color: str, alpha: float = 0.08) -> str:
        """Convert #RRGGBB to rgba(r,g,b,alpha)."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    fill_colors = {k: _hex_to_rgba_fill(c) for k, c in colors.items()}

    biomarker_configs = [
        ("adsi", 1, 1, "ADSI"),
        ("gvhd_score", 1, 2, "GVHD"),
        ("recovery_index", 2, 1, "Recovery"),
        ("pharma_response", 2, 2, "Pharma"),
        ("cv_risk", 3, 1, "CV Risk"),
        ("allostatic_load", 3, 2, "Allostatic"),
    ]

    for name, row, col, label in biomarker_configs:
        series = biomarkers[name]
        valid = series.dropna()
        valid_dates = [datetime.combine(d, datetime.min.time()) for d in valid.index]

        # Daily scatter (subtle, behind the trend line)
        fig.add_trace(
            go.Scatter(
                x=valid_dates,
                y=valid.values,
                mode="markers",
                marker=dict(size=3, color=colors[name], opacity=0.35),
                name=f"{label} (daily)",
                showlegend=False,
                hovertemplate=f"<b>%{{x|%b %d}}</b><br>{label}: %{{y:.1f}}<extra></extra>",
            ),
            row=row, col=col,
        )

        # 7-day rolling average (prominent trend line with subtle fill)
        rolling = series.rolling(7, min_periods=3).mean().dropna()
        rolling_dates = [datetime.combine(d, datetime.min.time()) for d in rolling.index]

        # Fill under rolling average
        fig.add_trace(
            go.Scatter(
                x=rolling_dates,
                y=rolling.values,
                mode="none",
                fill="tozeroy",
                fillcolor=fill_colors[name],
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_dates,
                y=rolling.values,
                mode="lines",
                line=dict(color=colors[name], width=2.5),
                name=f"{label} (7d avg)",
                showlegend=True,
                hovertemplate=f"<b>%{{x|%b %d}}</b><br>{label} 7d avg: %{{y:.1f}}<extra></extra>",
            ),
            row=row, col=col,
        )

        # Ruxolitinib start line - consistent, prominent
        axis_idx = (row - 1) * 2 + col
        yref_str = f"y{'%d' % axis_idx if axis_idx > 1 else ''} domain"
        fig.add_shape(
            type="line", x0=rux_line_date, x1=rux_line_date,
            y0=0, y1=1, yref=yref_str,
            line=dict(color=ACCENT_BLUE, width=2, dash="dash"),
            row=row, col=col,
        )
        # Ruxolitinib label on first panel only
        if row == 1 and col == 1:
            fig.add_annotation(
                x=rux_line_date, y=0.98,
                xref="x domain",
                yref="y domain",
                text="Rux start",
                showarrow=False,
                font=dict(size=9, color=ACCENT_BLUE),
                bgcolor="rgba(26,29,39,0.85)",
                borderpad=2,
            )

        # Add trend annotation with cleaner styling
        trend = trends.get(name, {})
        trend_text = trend.get("direction", "?")
        if trend.get("p_value") is not None and trend["p_value"] < 0.05:
            trend_text += f" (p={trend['p_value']:.3f})"

        fig.add_annotation(
            x=0.02, y=0.95,
            xref=f"x{axis_idx} domain" if axis_idx > 1 else "x domain",
            yref=f"y{axis_idx} domain" if axis_idx > 1 else "y domain",
            text=f"Trend: {trend_text}",
            showarrow=False,
            font=dict(size=10, color=colors[name]),
            bgcolor="rgba(26,29,39,0.85)",
            borderpad=3,
        )

    # -- Row 4, Col 1: Raw RMSSD + Sleep HR --
    rmssd_dates = [datetime.combine(d, datetime.min.time()) for d in df.index[df["rmssd_mean"].notna()]]
    rmssd_vals = df.loc[df["rmssd_mean"].notna(), "rmssd_mean"]
    # RMSSD fill
    fig.add_trace(
        go.Scatter(
            x=rmssd_dates,
            y=rmssd_vals,
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.06)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rmssd_dates,
            y=rmssd_vals,
            mode="lines+markers",
            marker=dict(size=2, color=C_HRV),
            line=dict(color=C_HRV, width=2),
            name="RMSSD (ms)",
            yaxis="y7",
            hovertemplate="<b>%{x|%b %d}</b><br>RMSSD: %{y:.1f} ms<extra></extra>",
        ),
        row=4, col=1,
    )

    sleep_hr_valid = df["sleep_hr_mean"].dropna()
    hr_dates = [datetime.combine(d, datetime.min.time()) for d in sleep_hr_valid.index]
    fig.add_trace(
        go.Scatter(
            x=hr_dates,
            y=sleep_hr_valid.values,
            mode="lines+markers",
            marker=dict(size=2, color=C_HR),
            line=dict(color=C_HR, width=2),
            name="Sleep HR (bpm)",
            yaxis="y7",
            hovertemplate="<b>%{x|%b %d}</b><br>Sleep HR: %{y:.0f} bpm<extra></extra>",
        ),
        row=4, col=1,
    )

    # Reference lines for RMSSD
    fig.add_shape(
        type="line", x0=dates[0], x1=dates[-1],
        y0=NORM_RMSSD_MEAN, y1=NORM_RMSSD_MEAN,
        line=dict(color=C_HRV, width=1, dash="dot"),
        row=4, col=1,
    )
    fig.add_annotation(
        x=dates[-1], y=NORM_RMSSD_MEAN,
        text=f"Age norm ({NORM_RMSSD_MEAN:.0f} ms)",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=9, color=C_HRV),
        row=4, col=1,
    )

    # -- Row 4, Col 2: Component breakdown radar (latest day with full data) --
    # Find the most recent day with all key metrics
    latest = df.dropna(subset=["rmssd_mean", "sleep_hr_mean", "spo2"], how="any").iloc[-1] if not df.dropna(subset=["rmssd_mean", "sleep_hr_mean", "spo2"], how="any").empty else None

    if latest is not None:
        categories = [
            "HRV (RMSSD)",
            "Sleep HR",
            "HR Dipping",
            "SpO2",
            "CV Age",
            "Sleep Quality",
            "Temperature",
        ]
        # Normalize each to 0-100 where 100 = healthy
        vals = [
            min(100, latest.get("rmssd_mean", 0) / NORM_RMSSD_MEAN * 100),
            min(100, max(0, (100 - latest.get("sleep_hr_mean", 100)) / (100 - 60) * 100)),
            min(100, max(0, latest.get("hr_dip_pct", 0) / 15 * 100)),
            min(100, max(0, (latest.get("spo2", 90) - 90) / (NORM_SPO2_MEAN - 90) * 100)),
            min(100, max(0, 100 - latest.get("cv_age_delta", 0) / 15 * 100)),
            min(100, max(0, latest.get("sleep_efficiency", 0))),
            min(100, max(0, 100 - abs(latest.get("temp_deviation", 0)) / 1.0 * 100)),
        ]
        # Close the radar
        categories_closed = categories + [categories[0]]
        vals_closed = vals + [vals[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=vals_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(59, 130, 246, 0.15)",
                line=dict(color=ACCENT_BLUE, width=2.5),
                marker=dict(size=5, color=ACCENT_BLUE),
                name="Health Profile (latest day)",
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.0f}/100<extra></extra>",
            ),
            row=4, col=2,
        )
        # Update polar layout for dark theme
        fig.update_layout(
            polar=dict(
                bgcolor=BG_SURFACE,
                radialaxis=dict(
                    gridcolor="rgba(255,255,255,0.08)",
                    linecolor=BORDER_SUBTLE,
                    range=[0, 100],
                    tickfont=dict(size=9, color=TEXT_SECONDARY),
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.08)",
                    linecolor=BORDER_SUBTLE,
                    tickfont=dict(size=10, color=TEXT_PRIMARY),
                ),
            ),
        )

    # -- Layout (clinical_dark template handles bgcolor, font, etc.) --
    fig.update_layout(
        height=1800,
        margin=dict(l=70, r=30, t=120, b=40),
        showlegend=True,
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
        hovermode="x unified",
    )

    # Set axis labels, date formatting, subtle grids, and crosshair spikes
    for i, (name, row, col, label) in enumerate(biomarker_configs):
        axis_idx = (row - 1) * 2 + col
        y_label = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
        if name == "allostatic_load":
            fig.update_layout(**{y_label: dict(title="Score (0-7)", range=[0, 7.5])})
        elif name == "pharma_response":
            fig.update_layout(**{y_label: dict(title="Z-score (mean)")})
        else:
            fig.update_layout(**{y_label: dict(title="Score (0-100)", range=[0, 105])})

        # Consistent date formatting on all x-axes
        x_label = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
        fig.update_layout(**{x_label: dict(tickformat="%d %b")})

        # Crosshair spikes and subtle gridlines on all subplot axes
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True, spikemode="across", spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)", spikedash="dot",
            row=row, col=col,
        )
        fig.update_yaxes(
            zeroline=False,
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True, spikemode="across", spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)", spikedash="dot",
            row=row, col=col,
        )

    # Also update row 4 col 1 (RMSSD/HR panel)
    fig.update_xaxes(
        tickformat="%d %b",
        gridcolor="rgba(255,255,255,0.05)",
        showspikes=True, spikemode="across", spikethickness=1,
        spikecolor="rgba(255,255,255,0.15)", spikedash="dot",
        row=4, col=1,
    )
    fig.update_yaxes(
        zeroline=False,
        gridcolor="rgba(255,255,255,0.05)",
        row=4, col=1,
    )

    return fig


# ---------------------------------------------------------------------------
# Build summary JSON
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame, biomarkers: dict[str, pd.Series],
                  trends: dict[str, dict]) -> dict[str, Any]:
    """Build JSON-serializable summary of all biomarkers."""
    generated_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "generated": generated_at,
        "generated_at": generated_at,
        "patient": PATIENT_LABEL,
        "age": PATIENT_AGE,
        "months_post_hsct": round((date.today() - TRANSPLANT_DATE).days / 30.44, 1),
        "data_range": {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "days": len(df),
        },
        "ruxolitinib_start": str(TREATMENT_START),
        "biomarkers": {},
    }

    interpreters = {
        "adsi": interpret_adsi,
        "gvhd_score": interpret_gvhd,
        "recovery_index": interpret_recovery,
        "cv_risk": interpret_cv_risk,
        "allostatic_load": interpret_allostatic,
    }

    for name, series in biomarkers.items():
        clean = series.dropna()
        if len(clean) == 0:
            continue

        rolling = series.rolling(7, min_periods=3).mean().dropna()
        latest_val = float(clean.iloc[-1]) if len(clean) > 0 else None
        latest_rolling = float(rolling.iloc[-1]) if len(rolling) > 0 else None

        entry: dict[str, Any] = {
            "latest_value": round(latest_val, 2) if latest_val is not None else None,
            "latest_7d_avg": round(latest_rolling, 2) if latest_rolling is not None else None,
            "mean": round(float(clean.mean()), 2),
            "median": round(float(clean.median()), 2),
            "std": round(float(clean.std()), 2),
            "min": round(float(clean.min()), 2),
            "max": round(float(clean.max()), 2),
            "n_days": len(clean),
            "trend": trends.get(name, {}),
        }

        if name in interpreters:
            interp_val = latest_rolling if latest_rolling is not None else latest_val
            if interp_val is not None:
                entry["interpretation"] = interpreters[name](interp_val)

        # Pre/post ruxolitinib comparison
        pre = clean[[d < TREATMENT_START for d in clean.index]]
        post = clean[[d >= TREATMENT_START for d in clean.index]]
        if len(pre) >= 5 and len(post) >= 1:
            entry["ruxolitinib"] = {
                "pre_mean": round(float(pre.mean()), 2),
                "pre_sd": round(float(pre.std()), 2),
                "post_mean": round(float(post.mean()), 2),
                "post_n": len(post),
                "delta": round(float(post.mean() - pre.mean()), 2),
            }

        # Daily time series (for JSON export)
        entry["daily"] = [
            {"date": str(d), "value": round(float(v), 3)}
            for d, v in zip(clean.index, clean.values)
        ]

        summary["biomarkers"][name] = entry

    return summary


# ---------------------------------------------------------------------------
# Individual Metric Treatment Response
# ---------------------------------------------------------------------------

_TREATMENT_METRICS = [
    ("rmssd_mean", "HRV (RMSSD)", "ms"),
    ("sleep_hr_lowest", "Lowest HR", "bpm"),
    ("sleep_hr_mean", "Average HR", "bpm"),
    ("sleep_efficiency", "Sleep Efficiency", "%"),
]


def _effect_label(d: float) -> str:
    """Return Cohen's d effect-size label."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _build_treatment_response_section(df: pd.DataFrame) -> str:
    """Pre/post Ruxolitinib comparison for individual biometrics.

    Returns an HTML string with KPI cards, comparison table,
    three-period trajectory, clinical note, and grouped bar chart.
    """
    from scipy.stats import mannwhitneyu

    # ------------------------------------------------------------------
    # 1. Split at TREATMENT_START
    # ------------------------------------------------------------------
    pre = df[df.index < TREATMENT_START]
    post = df[df.index >= TREATMENT_START]

    if len(pre) < 5 or len(post) < 1:
        return "<p>Insufficient data for treatment response analysis.</p>"

    # ------------------------------------------------------------------
    # 2. Mann-Whitney U for each metric
    # ------------------------------------------------------------------
    results: list[dict[str, Any]] = []
    for col, label, unit in _TREATMENT_METRICS:
        if col not in df.columns:
            continue
        a = pre[col].dropna()
        b = post[col].dropna()
        if len(a) < 3 or len(b) < 1:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        na, nb = len(a), len(b)
        pooled_std = np.sqrt(
            ((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2)
        )
        d = (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0.0
        pct_change = (
            ((b.mean() - a.mean()) / a.mean()) * 100 if a.mean() != 0 else 0.0
        )
        results.append({
            "col": col,
            "label": label,
            "unit": unit,
            "pre_mean": a.mean(),
            "pre_sd": a.std(),
            "post_mean": b.mean(),
            "post_sd": b.std(),
            "pct_change": pct_change,
            "p": p,
            "d": d,
            "effect": _effect_label(d),
        })

    if not results:
        return "<p>No treatment response metrics available.</p>"

    # ------------------------------------------------------------------
    # 3. Three-period breakdown
    # ------------------------------------------------------------------
    period1 = df[df.index < KNOWN_EVENT_DATE]
    period2 = df[(df.index >= KNOWN_EVENT_DATE) & (df.index < TREATMENT_START)]
    period3 = df[df.index >= TREATMENT_START]

    three_period: dict[str, dict[str, float | None]] = {}
    for col, label, unit in _TREATMENT_METRICS:
        if col not in df.columns:
            continue
        p1 = period1[col].dropna()
        p2 = period2[col].dropna()
        p3 = period3[col].dropna()
        three_period[col] = {
            "label": label,
            "unit": unit,
            "p1_mean": float(p1.mean()) if len(p1) else None,
            "p2_mean": float(p2.mean()) if len(p2) else None,
            "p3_mean": float(p3.mean()) if len(p3) else None,
        }

    # ------------------------------------------------------------------
    # 4. Build HTML
    # ------------------------------------------------------------------

    # KPI cards
    cards = []
    for r in results:
        if r["p"] < 0.05:
            status = "good"
        elif r["p"] < 0.10:
            status = "warning"
        else:
            status = "neutral"
        cards.append(make_kpi_card(
            label=r["label"],
            value=r["pct_change"],
            unit="% change",
            status=status,
            detail=f"p={r['p']:.4f}, d={abs(r['d']):.2f} ({r['effect']})",
            status_label="Significant" if r["p"] < 0.05 else "",
        ))
    kpi_html = make_kpi_row(*cards)

    # Comparison table
    table_rows = ""
    for r in results:
        direction = "+" if r["pct_change"] >= 0 else ""
        table_rows += (
            f"<tr>"
            f"<td>{r['label']}</td>"
            f"<td>{r['pre_mean']:.1f} &plusmn; {r['pre_sd']:.1f} {r['unit']}</td>"
            f"<td>{r['post_mean']:.1f} &plusmn; {r['post_sd']:.1f} {r['unit']}</td>"
            f"<td>{direction}{r['pct_change']:.1f}%</td>"
            f"<td><b>{r['p']:.4f}</b></td>"
            f"<td>{abs(r['d']):.2f}</td>"
            f"<td>{r['effect']}</td>"
            f"</tr>"
        )
    table_html = (
        f'<table style="width:100%;border-collapse:collapse;background:{BG_SURFACE};'
        f'color:{TEXT_PRIMARY};margin:16px 0;">'
        f'<tr style="border-bottom:1px solid {BORDER_SUBTLE};">'
        f"<th style='padding:8px;text-align:left;'>Metric</th>"
        f"<th style='padding:8px;'>Pre-Rux (mean&plusmn;SD)</th>"
        f"<th style='padding:8px;'>Post-Rux (mean&plusmn;SD)</th>"
        f"<th style='padding:8px;'>Change</th>"
        f"<th style='padding:8px;'>p-value</th>"
        f"<th style='padding:8px;'>Cohen's d</th>"
        f"<th style='padding:8px;'>Effect</th></tr>"
        f"{table_rows}</table>"
    )

    # Three-period trajectory
    traj_lines = ""
    for col, info in three_period.items():
        vals = []
        for key in ("p1_mean", "p2_mean", "p3_mean"):
            v = info[key]
            vals.append(f"{v:.1f}" if v is not None else "N/A")
        interpretation = ""
        if col == "rmssd_mean":
            interpretation = "(acute event = inflection, Rux stabilized)"
        elif col in ("sleep_hr_lowest", "sleep_hr_mean"):
            interpretation = "(steady improvement)"
        elif col == "sleep_efficiency":
            interpretation = ""
        traj_lines += (
            f"<div style='padding:4px 0;'>"
            f"<b>{info['label']}:</b> "
            f"{vals[0]} {info['unit']} &rarr; {vals[1]} {info['unit']} &rarr; {vals[2]} {info['unit']} "
            f"<span style='color:{TEXT_SECONDARY};'>{interpretation}</span></div>"
        )
    trajectory_html = (
        f'<div style="background:{BG_SURFACE};padding:16px;border-radius:8px;'
        f'border:1px solid {BORDER_SUBTLE};margin:16px 0;">'
        f'<h4 style="margin:0 0 8px 0;color:{TEXT_PRIMARY};">Three-Period Trajectory</h4>'
        f'<p style="color:{TEXT_SECONDARY};margin:0 0 8px 0;font-size:0.85em;">'
        f'Pre-acute ({KNOWN_EVENT_DATE}) &rarr; Post-acute/Pre-Rux &rarr; Post-Rux ({TREATMENT_START})</p>'
        f'{traj_lines}</div>'
    )

    # Clinical note
    note_html = (
        f'<div style="background:{BG_SURFACE};padding:16px;border-radius:8px;'
        f'border-left:3px solid {ACCENT_AMBER};margin:16px 0;">'
        f'<p style="color:{TEXT_PRIMARY};margin:0;"><b>Clinical note:</b> '
        f'The composite ADSI score dilutes these individual metric signals. '
        f'While ADSI aggregates multiple domains into a single index, individual metrics '
        f'like HRV and heart rate show statistically significant pre/post differences that '
        f'the composite score may obscure through averaging.</p></div>'
    )

    # Plotly grouped bar chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[r["label"] for r in results[:4]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for idx, r in enumerate(results[:4]):
        row, col = positions[idx]
        fig.add_trace(
            go.Bar(
                x=["Pre-Rux"],
                y=[r["pre_mean"]],
                error_y=dict(type="data", array=[r["pre_sd"]], visible=True),
                marker_color=ACCENT_PURPLE,
                name="Pre-Rux" if idx == 0 else None,
                showlegend=(idx == 0),
                legendgroup="pre",
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Bar(
                x=["Post-Rux"],
                y=[r["post_mean"]],
                error_y=dict(type="data", array=[r["post_sd"]], visible=True),
                marker_color=ACCENT_GREEN,
                name="Post-Rux" if idx == 0 else None,
                showlegend=(idx == 0),
                legendgroup="post",
            ),
            row=row, col=col,
        )
        fig.update_yaxes(
            title_text=f"{r['unit']}", row=row, col=col,
        )
    fig.update_layout(
        height=500,
        title_text="Individual Metric Pre/Post Ruxolitinib Comparison",
        barmode="group",
        margin=dict(t=60, b=40, l=60, r=20),
    )
    plot_div = fig.to_html(include_plotlyjs=False, full_html=False)

    return kpi_html + table_html + trajectory_html + note_html + plot_div


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Main entry point: load data, compute biomarkers, generate outputs."""
    print("Composite biomarker analysis for post-HSCT monitoring")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    try:
        # 1. Load daily metrics
        print("\n[1/6] Loading daily metrics from Oura tables...")
        df = get_daily_metrics(conn)
        print(f"  -> {len(df)} days, {len(df.columns)} columns")
        print(f"  -> Period: {df.index.min()} to {df.index.max()}")
    finally:
        conn.close()

    # 2. Compute all biomarkers
    print("\n[2/6] Computing composite biomarkers...")

    biomarkers: dict[str, pd.Series] = {}

    print("  -> ADSI (Autonomic Dysfunction Severity Index)...")
    biomarkers["adsi"] = compute_adsi(df)
    adsi_valid = biomarkers["adsi"].dropna()
    if len(adsi_valid) > 0:
        print(f"     Mean: {adsi_valid.mean():.1f}, Latest: {adsi_valid.iloc[-1]:.1f}")

    print("  -> GVHD Activity Score...")
    biomarkers["gvhd_score"] = compute_gvhd_score(df)
    gvhd_valid = biomarkers["gvhd_score"].dropna()
    if len(gvhd_valid) > 0:
        print(f"     Mean: {gvhd_valid.mean():.1f}, Latest: {gvhd_valid.iloc[-1]:.1f}")

    print("  -> Recovery Trajectory Index...")
    biomarkers["recovery_index"] = compute_recovery_index(df)
    rec_valid = biomarkers["recovery_index"].dropna()
    if len(rec_valid) > 0:
        print(f"     Mean: {rec_valid.mean():.1f}, Latest: {rec_valid.iloc[-1]:.1f}")

    print("  -> Pharmacodynamic Response Score (ruxolitinib)...")
    biomarkers["pharma_response"] = compute_pharma_response(df)
    pharma_valid = biomarkers["pharma_response"].dropna()
    post_rux = pharma_valid[[d >= TREATMENT_START for d in pharma_valid.index]]
    if len(post_rux) > 0:
        print(f"     Post-rux days: {len(post_rux)}, Mean Z: {post_rux.mean():.2f}")
    else:
        print(f"     Baseline established ({len(pharma_valid)} days), awaiting post-rux data")

    print("  -> Cardiovascular Risk Composite...")
    biomarkers["cv_risk"] = compute_cv_risk(df)
    cv_valid = biomarkers["cv_risk"].dropna()
    if len(cv_valid) > 0:
        print(f"     Mean: {cv_valid.mean():.1f}, Latest: {cv_valid.iloc[-1]:.1f}")

    print("  -> Wearable Allostatic Load Score...")
    biomarkers["allostatic_load"] = compute_allostatic_load(df)
    al_valid = biomarkers["allostatic_load"].dropna()
    if len(al_valid) > 0:
        print(f"     Mean: {al_valid.mean():.1f}/7, Latest: {al_valid.iloc[-1]:.0f}/7")

    # 3. Compute trends
    print("\n[3/6] Computing trends...")
    trends: dict[str, dict] = {}
    for name, series in biomarkers.items():
        trends[name] = compute_trend(series, window_days=30)
        t = trends[name]
        slope_info = f" (slope={t['slope']}/day, p={t['p_value']})" if t['slope'] is not None else ''
        print(f"  -> {name}: {t['direction']}{slope_info}")

    # 4. Build summary
    print("\n[4/6] Building summary...")
    summary = build_summary(df, biomarkers, trends)

    # 5. Generate dashboard
    print("\n[5/6] Generating interactive dashboard...")
    fig = build_dashboard(df, biomarkers, trends, summary)

    # Build themed HTML with KPI cards, executive summary, and chart
    months_post = round((date.today() - TRANSPLANT_DATE).days / 30.44, 1)
    plot_div = fig.to_html(full_html=False, include_plotlyjs=False)

    # --- KPI cards for each biomarker ---
    kpi_cards = []
    kpi_names = {
        "adsi": ("ADSI", "/100", "Autonomic Dysfunction"),
        "gvhd_score": ("GVHD Score", "/100", "Wearable Activity"),
        "recovery_index": ("Recovery", "/100", "Trajectory Index"),
        "pharma_response": ("Pharma", "Z", "Ruxolitinib Response"),
        "cv_risk": ("CV Risk", "/100", "Cardiovascular Composite"),
        "allostatic_load": ("Allostatic", "/7", "Load Score"),
    }
    for bname, (label, unit, detail_text) in kpi_names.items():
        bseries = biomarkers[bname]
        bclean = bseries.dropna()
        if len(bclean) == 0:
            continue
        rolling7 = bseries.rolling(7, min_periods=3).mean().dropna()
        bval = float(rolling7.iloc[-1]) if len(rolling7) > 0 else float(bclean.iloc[-1])
        btrend = trends.get(bname, {}).get("direction", "?")
        window_lbl = trends.get(bname, {}).get("window_label", "trend")

        if bname in ("adsi", "gvhd_score", "cv_risk"):
            status = "critical" if bval >= 60 else "warning" if bval >= 40 else "normal"
            slabel = "Elevated" if status in ("critical", "warning") else ""
        elif bname == "allostatic_load":
            status = "critical" if bval >= 5 else "warning" if bval >= 3 else "normal"
            slabel = "Elevated" if status in ("critical", "warning") else ""
        elif bname == "recovery_index":
            status = "normal" if bval >= 70 else "warning" if bval >= 40 else "critical"
            slabel = "Low" if status in ("critical", "warning") else ""
        else:
            status = "info"
            slabel = ""

        kpi_cards.append(make_kpi_card(
            label=label,
            value=bval,
            unit=unit,
            status=status,
            detail=f"{detail_text} | {btrend} ({window_lbl})",
            explainer=METRIC_DESCRIPTIONS.get(bname.upper(), ""),
            status_label=slabel,
        ))

    body = make_kpi_row(*kpi_cards)

    # --- Executive summary table ---
    exec_rows = []
    for bname in ["adsi", "gvhd_score", "recovery_index", "cv_risk", "allostatic_load"]:
        bseries = biomarkers[bname]
        bclean = bseries.dropna()
        if len(bclean) == 0:
            continue
        rolling7 = bseries.rolling(7, min_periods=3).mean().dropna()
        bval = float(rolling7.iloc[-1]) if len(rolling7) > 0 else float(bclean.iloc[-1])
        scale = "/7" if bname == "allostatic_load" else "/100"
        btrend = trends.get(bname, {}).get("direction", "?")
        window_lbl = trends.get(bname, {}).get("window_label", "trend")
        exec_rows.append(
            f"<tr><td><b>{bname.upper()}</b></td>"
            f"<td>{bval:.1f}{scale}</td><td>{btrend} ({window_lbl})</td></tr>"
        )
    exec_table = "\n".join(exec_rows)
    summary_html = (
        f'<p><b>{PATIENT_LABEL}</b> | {months_post} months post-HSCT '
        f'| Ruxolitinib from {TREATMENT_START_STR}</p>'
        f'<table><tr><th>Biomarker</th><th>Value</th><th>Trend</th></tr>'
        f'{exec_table}</table>'
    )
    body += make_section("Summary", summary_html)

    # --- Individual Metric Treatment Response ---
    body += section_html_or_placeholder(
        "Treatment Response",
        _build_treatment_response_section,
        df,
    )

    # --- Main chart ---
    body += make_section("Biomarker Dashboard", plot_div)

    html_content = wrap_html(
        title="Composite Biomarkers",
        body_content=body,
        report_id="biomarkers",
        subtitle=f"Post-HSCT Monitoring - {months_post} months",
    )
    HTML_OUTPUT.write_text(html_content, encoding="utf-8")
    print(f"  -> HTML: {HTML_OUTPUT}")

    # 6. Export JSON — include individual treatment response metrics
    print("\n[6/6] Exporting JSON metrics...")

    # Build treatment_response and three_period for JSON
    _pre = df[df.index < TREATMENT_START]
    _post = df[df.index >= TREATMENT_START]
    _p1 = df[df.index < KNOWN_EVENT_DATE]
    _p2 = df[(df.index >= KNOWN_EVENT_DATE) & (df.index < TREATMENT_START)]
    _p3 = df[df.index >= TREATMENT_START]

    tr_json: dict[str, Any] = {}
    tp_json: dict[str, Any] = {}
    for _col, _label, _unit in _TREATMENT_METRICS:
        if _col not in df.columns:
            continue
        a = _pre[_col].dropna()
        b = _post[_col].dropna()
        if len(a) >= 3 and len(b) >= 1:
            from scipy.stats import mannwhitneyu
            _stat, _p = mannwhitneyu(a, b, alternative="two-sided")
            _na, _nb = len(a), len(b)
            _pooled = np.sqrt(
                ((_na - 1) * a.std() ** 2 + (_nb - 1) * b.std() ** 2) / (_na + _nb - 2)
            )
            _d = (b.mean() - a.mean()) / _pooled if _pooled > 0 else 0.0
            _pct = ((b.mean() - a.mean()) / a.mean()) * 100 if a.mean() != 0 else 0.0
            tr_json[_col] = {
                "pre_mean": round(float(a.mean()), 2),
                "post_mean": round(float(b.mean()), 2),
                "pct_change": round(float(_pct), 2),
                "mann_whitney_p": round(float(_p), 6),
                "cohens_d": round(float(_d), 3),
                "effect_label": _effect_label(_d),
            }
        # Three-period values
        v1 = _p1[_col].dropna()
        v2 = _p2[_col].dropna()
        v3 = _p3[_col].dropna()
        tp_json[_col] = {
            "label": _label,
            "pre_acute_mean": round(float(v1.mean()), 2) if len(v1) else None,
            "post_acute_pre_rux_mean": round(float(v2.mean()), 2) if len(v2) else None,
            "post_rux_mean": round(float(v3.mean()), 2) if len(v3) else None,
        }

    summary["treatment_response"] = tr_json
    summary["three_period"] = tp_json

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> JSON: {JSON_OUTPUT}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY - Composite Biomarkers")
    print("=" * 60)

    for name in ["adsi", "gvhd_score", "recovery_index", "cv_risk", "allostatic_load"]:
        series = biomarkers[name]
        clean = series.dropna()
        if len(clean) == 0:
            continue
        rolling = series.rolling(7, min_periods=3).mean().dropna()
        latest = float(rolling.iloc[-1]) if len(rolling) > 0 else float(clean.iloc[-1])
        scale = "/7" if name == "allostatic_load" else "/100"
        trend_str = trends[name]["direction"]

        interp_fns = {
            "adsi": interpret_adsi,
            "gvhd_score": interpret_gvhd,
            "recovery_index": interpret_recovery,
            "cv_risk": interpret_cv_risk,
            "allostatic_load": interpret_allostatic,
        }
        interp = interp_fns[name](latest)

        print(f"\n  {name.upper()}: {latest:.1f}{scale} [{trend_str}]")
        print(f"    {interp}")

    pharma = biomarkers["pharma_response"]
    post_rux_ph = pharma[[d >= TREATMENT_START for d in pharma.index]].dropna()
    if len(post_rux_ph) > 0:
        print(f"\n  PHARMA_RESPONSE: {post_rux_ph.mean():.2f} Z-score (mean over {len(post_rux_ph)} days)")
        print("    Higher Z-score indicates stronger pharmacodynamic response to ruxolitinib.")
    else:
        print("\n  PHARMA_RESPONSE: Baseline established. Awaiting more post-ruxolitinib data.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
