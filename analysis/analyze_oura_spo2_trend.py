#!/usr/bin/env python3
"""
SpO2 Trend Analysis for BOS Early Detection in Post-HSCT Patient

Continuous nocturnal SpO2 monitoring via Oura Ring (finger sensor) for early
detection of bronchiolitis obliterans syndrome (BOS) - the leading cause of
late non-relapse mortality after allogeneic HSCT.

Analyses:
  1. Nightly SpO2 trend with linear regression and prediction intervals
  2. Desaturation event frequency (absolute + relative thresholds)
  3. SpO2 night-to-night variability (SD, CV, rolling)
  4. SpO2-HR coupling analysis (compensatory response assessment)
  5. DLCO-SpO2 correlation (clinical PFT overlay)
  6. Breathing Disturbance Index (BDI) trend
  7. SpO2-Temperature coupling
  8. Composite BOS Risk Score
  9. Pre/Post Ruxolitinib comparison

Output:
  - Interactive HTML report: reports/spo2_bos_screening.html
  - JSON metrics: reports/spo2_bos_metrics.json
  - Stdout summary with key findings

Usage:
    python analysis/analyze_oura_spo2_trend.py
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH, REPORTS_DIR, TREATMENT_START,
    PATIENT_LABEL, BASELINE_DAYS,
)
from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    COLORWAY, STATUS_COLORS, BG_PRIMARY, BG_SURFACE, BORDER_SUBTLE,
    TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_BLUE, ACCENT_RED, ACCENT_AMBER,
    ACCENT_GREEN, ACCENT_PURPLE, ACCENT_CYAN, ACCENT_ORANGE, C_SPO2,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "spo2_bos_screening.html"
JSON_OUTPUT = REPORTS_DIR / "spo2_bos_metrics.json"

# SpO2 thresholds
SPO2_ABSOLUTE_THRESHOLD = 94.0      # Desaturation cutoff (%)
SPO2_CONCERN_SLOPE = -0.02          # %/day -> 1% decline per 50 days
SPO2_NORMAL_RANGE = (95.0, 100.0)   # Normal adult range

# BDI thresholds (events/hour)
BDI_NORMAL = 5.0
BDI_MILD = 15.0
BDI_MODERATE = 30.0

# BOS risk score weights
BOS_WEIGHTS = {
    "spo2_slope": 0.30,
    "spo2_variability": 0.20,
    "desaturation_freq": 0.20,
    "bdi": 0.15,
    "hr_decoupling": 0.15,
}

# DLCO measurements (from medical records)
DLCO_MEASUREMENTS = [
    # (date, DLCO%, context)
    (date(2024, 3, 21), 71.0, "Baseline post-HSCT"),
    (date(2025, 3, 20), 89.0, "Improvement"),
    (date(2025, 12, 17), 67.0, "Concerning decline"),
]

# Color aliases for local semantics (mapped from theme palette)
C_CAUTION = ACCENT_AMBER
C_OK = ACCENT_GREEN
C_CRITICAL = ACCENT_RED
C_WARNING = ACCENT_AMBER
C_BLUE = ACCENT_CYAN       # SpO2 data points
C_DARK = TEXT_SECONDARY     # Muted text / secondary elements
C_RUXI = ACCENT_PURPLE     # Ruxolitinib marker


# ---------------------------------------------------------------------------
# JSON / display helpers
# ---------------------------------------------------------------------------

def _fmt_nan(val: Any, fmt: str = "", fallback: str = "N/A") -> str:
    """Format a value for HTML display, returning *fallback* for NaN/None."""
    if val is None:
        return fallback
    try:
        if math.isnan(val):
            return fallback
    except (TypeError, ValueError):
        pass
    return f"{val:{fmt}}" if fmt else str(val)


def _is_finite_number(val: Any) -> bool:
    """Return True when *val* is a finite real number."""
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _sanitize_nan(obj: Any) -> Any:
    """Recursively replace float NaN/Inf with None for valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_spo2(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load nightly SpO2 data, filtering out no-data sentinels."""
    df = pd.read_sql_query(
        "SELECT date, spo2_average, breathing_disturbance_index "
        "FROM oura_spo2 WHERE spo2_average > 0 ORDER BY date",
        conn,
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_nightly_hr(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load nightly mean HR from oura_heart_rate (rest source) grouped by day."""
    df = pd.read_sql_query(
        "SELECT substr(timestamp,1,10) as day, AVG(bpm) as mean_hr, "
        "MIN(bpm) as min_hr, MAX(bpm) as max_hr, COUNT(*) as n_readings "
        "FROM oura_heart_rate WHERE source='rest' "
        "GROUP BY day ORDER BY day",
        conn,
    )
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


def load_sleep_periods(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load sleep period summaries."""
    df = pd.read_sql_query(
        "SELECT day, average_heart_rate, lowest_heart_rate, average_breath "
        "FROM oura_sleep_periods ORDER BY day",
        conn,
    )
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


def load_readiness(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load readiness scores with temperature deviation."""
    df = pd.read_sql_query(
        "SELECT date, score, temperature_deviation "
        "FROM oura_readiness WHERE temperature_deviation IS NOT NULL ORDER BY date",
        conn,
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Analysis 1: SpO2 Linear Trend
# ---------------------------------------------------------------------------
def analyze_spo2_trend(spo2: pd.DataFrame) -> dict[str, Any]:
    """Fit linear regression to nightly SpO2 and compute prediction intervals."""
    if len(spo2) < 2:
        return {
            "slope_pct_per_day": 0.0,
            "slope_pct_per_month": 0.0,
            "slope_95ci": [0.0, 0.0],
            "p_value": 1.0,
            "r_squared": 0.0,
            "intercept": float(spo2["spo2_average"].iloc[0]) if len(spo2) == 1 else 0.0,
            "concern_level": "INSUFFICIENT_DATA",
            "predictions": {},
            "model_x_days": [],
            "model_fitted": [],
        }

    dates = pd.to_datetime(spo2["date"])
    x_days = (dates - dates.min()).dt.days.values.astype(float)
    y = spo2["spo2_average"].values

    X = add_constant(x_days)
    model = OLS(y, X).fit()

    slope = model.params[1]
    intercept = model.params[0]
    slope_ci = model.conf_int(alpha=0.05)[1]
    p_value = model.pvalues[1]
    r_squared = model.rsquared

    # Prediction intervals for 30/60/90 days ahead
    max_day = x_days.max()
    future_days = [30, 60, 90]
    predictions = {}
    for fd in future_days:
        x_pred = max_day + fd
        pred = intercept + slope * x_pred
        # Prediction interval
        exog_pred = np.array([[1.0, x_pred]])  # [const, x_value]
        se_pred = model.get_prediction(exog_pred).summary_frame(alpha=0.05)
        predictions[f"{fd}d"] = {
            "predicted_spo2": round(pred, 2),
            "ci_lower": round(se_pred["obs_ci_lower"].iloc[0], 2),
            "ci_upper": round(se_pred["obs_ci_upper"].iloc[0], 2),
        }

    # Concern level
    if slope < SPO2_CONCERN_SLOPE:
        concern = "HIGH"
    elif slope < SPO2_CONCERN_SLOPE / 2:
        concern = "MODERATE"
    else:
        concern = "LOW"

    return {
        "slope_pct_per_day": round(slope, 5),
        "slope_pct_per_month": round(slope * 30, 3),
        "slope_95ci": [round(slope_ci[0], 5), round(slope_ci[1], 5)],
        "p_value": round(p_value, 6),
        "r_squared": round(r_squared, 4),
        "intercept": round(intercept, 3),
        "concern_level": concern,
        "predictions": predictions,
        "model_x_days": x_days.tolist(),
        "model_fitted": model.fittedvalues.tolist(),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Desaturation Events
# ---------------------------------------------------------------------------
def analyze_desaturation_events(spo2: pd.DataFrame) -> dict[str, Any]:
    """Count desaturation events using absolute and relative thresholds."""
    _empty = {"baseline_mean": np.nan, "baseline_sd": np.nan,
              "relative_threshold": np.nan, "absolute_desaturation_count": 0,
              "absolute_desaturation_pct": 0.0, "relative_desaturation_count": 0,
              "relative_desaturation_pct": 0.0, "first_half_desat_rate": 0.0,
              "second_half_desat_rate": 0.0, "frequency_increasing": False,
              "rolling_7d_abs": [], "rolling_7d_rel": []}
    if spo2.empty:
        return _empty
    vals = spo2["spo2_average"].values
    if len(vals) == 0:
        return _empty
    dates = spo2["date"].values

    # Personal baseline from first BASELINE_DAYS of data
    baseline_mask = np.arange(len(vals)) < BASELINE_DAYS
    baseline_vals = vals[baseline_mask]
    baseline_mean = float(np.mean(baseline_vals))
    baseline_sd = float(np.std(baseline_vals, ddof=1))
    relative_threshold = baseline_mean - 2 * baseline_sd

    # Absolute desaturation (< 94%)
    abs_desat = vals < SPO2_ABSOLUTE_THRESHOLD
    abs_desat_count = int(abs_desat.sum())

    # Relative desaturation (< personal baseline - 2SD)
    rel_desat = vals < relative_threshold
    rel_desat_count = int(rel_desat.sum())

    # 7-day rolling count of absolute desaturations
    spo2_copy = spo2.copy()
    spo2_copy["abs_desat"] = abs_desat.astype(int)
    spo2_copy["rel_desat"] = rel_desat.astype(int)
    spo2_copy["abs_desat_7d"] = spo2_copy["abs_desat"].rolling(7, min_periods=1).sum()
    spo2_copy["rel_desat_7d"] = spo2_copy["rel_desat"].rolling(7, min_periods=1).sum()

    # Trend in desaturation frequency
    # Compare first half vs second half
    mid = len(vals) // 2
    first_half_rate = abs_desat[:mid].sum() / mid if mid > 0 else 0
    second_half_rate = abs_desat[mid:].sum() / (len(vals) - mid) if len(vals) > mid else 0

    return {
        "baseline_mean": round(baseline_mean, 3),
        "baseline_sd": round(baseline_sd, 3),
        "relative_threshold": round(relative_threshold, 3),
        "absolute_desaturation_count": abs_desat_count,
        "absolute_desaturation_pct": round(100 * abs_desat_count / len(vals), 1),
        "relative_desaturation_count": rel_desat_count,
        "relative_desaturation_pct": round(100 * rel_desat_count / len(vals), 1),
        "first_half_desat_rate": round(first_half_rate, 3),
        "second_half_desat_rate": round(second_half_rate, 3),
        "frequency_increasing": second_half_rate > first_half_rate * 1.5,
        "rolling_7d_abs": spo2_copy["abs_desat_7d"].tolist(),
        "rolling_7d_rel": spo2_copy["rel_desat_7d"].tolist(),
    }


# ---------------------------------------------------------------------------
# Analysis 3: SpO2 Variability
# ---------------------------------------------------------------------------
def analyze_spo2_variability(spo2: pd.DataFrame) -> dict[str, Any]:
    """Assess night-to-night SpO2 variability."""
    _empty = {"overall_sd": np.nan, "overall_mean": np.nan, "cv_pct": np.nan,
              "diff_sd": np.nan, "sd_first_half": np.nan, "sd_second_half": np.nan,
              "variability_increasing": False, "rolling_sd_7d": []}
    if spo2.empty:
        return _empty
    vals = spo2["spo2_average"].values
    if len(vals) == 0:
        return _empty

    overall_sd = float(np.std(vals, ddof=1))
    overall_mean = float(np.mean(vals))
    if overall_mean == 0 or np.isnan(overall_mean):
        cv = np.nan
    else:
        cv = (overall_sd / overall_mean) * 100

    # 7-day rolling SD
    spo2_copy = spo2.copy()
    spo2_copy["rolling_sd_7d"] = spo2_copy["spo2_average"].rolling(7, min_periods=3).std()

    # Night-to-night differences
    diffs = np.diff(vals)
    diff_sd = float(np.std(diffs, ddof=1))

    # Compare variability first vs second half
    mid = len(vals) // 2
    sd_first = float(np.std(vals[:mid], ddof=1)) if mid > 2 else 0
    sd_second = float(np.std(vals[mid:], ddof=1)) if len(vals) - mid > 2 else 0

    return {
        "overall_sd": round(overall_sd, 3),
        "overall_mean": round(overall_mean, 3),
        "cv_pct": round(cv, 3),
        "diff_sd": round(diff_sd, 3),
        "sd_first_half": round(sd_first, 3),
        "sd_second_half": round(sd_second, 3),
        "variability_increasing": sd_second > sd_first * 1.3,
        "rolling_sd_7d": spo2_copy["rolling_sd_7d"].tolist(),
    }


# ---------------------------------------------------------------------------
# Analysis 4: SpO2-HR Coupling
# ---------------------------------------------------------------------------
def analyze_spo2_hr_coupling(
    spo2: pd.DataFrame, nightly_hr: pd.DataFrame, sleep: pd.DataFrame
) -> dict[str, Any]:
    """Assess SpO2-HR compensatory coupling (normally inverse)."""
    # Merge SpO2 with sleep HR data
    merged = pd.merge(
        spo2[["date", "spo2_average"]],
        sleep[["day", "average_heart_rate", "lowest_heart_rate"]].rename(columns={"day": "date"}),
        on="date",
        how="inner",
    )
    merged = merged.dropna(subset=["spo2_average", "average_heart_rate"]).copy()

    if len(merged) < 5:
        return {"error": "Insufficient overlapping SpO2-HR data", "n_paired": len(merged)}

    spo2_vals = merged["spo2_average"].values
    hr_vals = merged["average_heart_rate"].values

    # Correlation is undefined for constant or otherwise non-finite series.
    # Treat this as insufficient data rather than directional coupling.
    if np.nanstd(spo2_vals) == 0 or np.nanstd(hr_vals) == 0:
        return {
            "n_paired_nights": len(merged),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
            "coupling_status": "INSUFFICIENT DATA",
            "interpretation": (
                "Insufficient variation to assess SpO2-HR coupling reliably."
            ),
            "coupling_assessable": False,
            "rolling_correlation": [],
        }

    # Overall correlation
    pearson_r, pearson_p = stats.pearsonr(spo2_vals, hr_vals)
    spearman_r, spearman_p = stats.spearmanr(spo2_vals, hr_vals)

    if not all(
        _is_finite_number(v)
        for v in (pearson_r, pearson_p, spearman_r, spearman_p)
    ):
        return {
            "n_paired_nights": len(merged),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
            "coupling_status": "INSUFFICIENT DATA",
            "interpretation": (
                "Correlation could not be estimated reliably from the available "
                "SpO2-HR overlap."
            ),
            "coupling_assessable": False,
            "rolling_correlation": [],
        }

    # Rolling 14-day correlation
    rolling_corr = []
    window = 14
    for i in range(window, len(merged) + 1):
        subset = merged.iloc[i - window:i]
        if len(subset) >= 5:
            if (
                np.nanstd(subset["spo2_average"]) == 0
                or np.nanstd(subset["average_heart_rate"]) == 0
            ):
                continue
            r, _ = stats.pearsonr(
                subset["spo2_average"], subset["average_heart_rate"]
            )
            if _is_finite_number(r):
                rolling_corr.append({
                    "date": str(subset["date"].iloc[-1]),
                    "correlation": round(r, 3),
                })

    # Decoupling assessment
    # In healthy: SpO2 drop -> HR increase (negative correlation)
    # Decoupling: correlation near zero or positive
    if pearson_r > 0:
        coupling_status = "DECOUPLED (positive - abnormal)"
        interpretation = (
            "Positive SpO2-HR coupling suggests loss of the expected compensatory "
            "inverse response."
        )
    elif pearson_r > -0.2:
        coupling_status = "WEAK (possible decoupling)"
        interpretation = (
            "Weak inverse coupling suggests a blunted compensatory response."
        )
    else:
        coupling_status = "NORMAL (inverse relationship)"
        interpretation = (
            "Inverse coupling is present and consistent with the expected "
            "compensatory response."
        )

    return {
        "n_paired_nights": len(merged),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 6),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 6),
        "coupling_status": coupling_status,
        "interpretation": interpretation,
        "coupling_assessable": True,
        "rolling_correlation": rolling_corr,
    }


# ---------------------------------------------------------------------------
# Analysis 5: DLCO-SpO2 Correlation
# ---------------------------------------------------------------------------
def analyze_dlco_spo2(spo2: pd.DataFrame) -> dict[str, Any]:
    """Overlay DLCO measurements with concurrent SpO2 data."""
    spo2_date_range = (spo2["date"].min(), spo2["date"].max())

    overlapping = []
    for dlco_date, dlco_pct, context in DLCO_MEASUREMENTS:
        # Check if we have SpO2 data within +/- 7 days
        nearby = spo2[
            (spo2["date"] >= dlco_date - timedelta(days=7))
            & (spo2["date"] <= dlco_date + timedelta(days=7))
        ]
        if len(nearby) > 0:
            overlapping.append({
                "date": str(dlco_date),
                "dlco_pct": dlco_pct,
                "context": context,
                "concurrent_spo2_mean": round(nearby["spo2_average"].mean(), 2),
                "concurrent_spo2_n": len(nearby),
            })

    # All DLCO points for display even if not overlapping
    all_dlco = [
        {"date": str(d), "dlco_pct": v, "context": c}
        for d, v, c in DLCO_MEASUREMENTS
    ]

    return {
        "spo2_data_range": [str(spo2_date_range[0]), str(spo2_date_range[1])],
        "dlco_measurements": all_dlco,
        "overlapping": overlapping,
        "has_overlap": len(overlapping) > 0,
    }


# ---------------------------------------------------------------------------
# Analysis 6: Breathing Disturbance Index
# ---------------------------------------------------------------------------
def analyze_bdi(spo2: pd.DataFrame) -> dict[str, Any]:
    """Analyze Breathing Disturbance Index trend."""
    bdi_data = spo2.dropna(subset=["breathing_disturbance_index"]).copy()

    if len(bdi_data) < 3:
        return {"available": False, "n_readings": len(bdi_data)}

    vals = bdi_data["breathing_disturbance_index"].values
    if len(vals) == 0:
        return {"available": False, "n_readings": 0}
    dates = pd.to_datetime(bdi_data["date"])
    x_days = (dates - dates.min()).dt.days.values.astype(float)

    # Linear trend
    if len(x_days) >= 3:
        slope, intercept, r_val, p_val, se = stats.linregress(x_days, vals)
    else:
        slope = intercept = r_val = p_val = se = 0.0

    # Classification
    mean_bdi = float(np.mean(vals))
    if mean_bdi < BDI_NORMAL:
        bdi_status = "NORMAL"
    elif mean_bdi < BDI_MILD:
        bdi_status = "MILD ELEVATION"
    elif mean_bdi < BDI_MODERATE:
        bdi_status = "MODERATE"
    else:
        bdi_status = "SEVERE"

    # Nights above threshold
    elevated_nights = int((vals >= BDI_NORMAL).sum())
    elevated_pct = round(100 * elevated_nights / len(vals), 1) if len(vals) > 0 else 0.0

    # 7-day rolling
    bdi_data["bdi_7d"] = bdi_data["breathing_disturbance_index"].rolling(7, min_periods=3).mean()

    return {
        "available": True,
        "n_readings": len(bdi_data),
        "mean_bdi": round(mean_bdi, 2),
        "median_bdi": round(float(np.median(vals)), 2),
        "min_bdi": round(float(vals.min()), 1),
        "max_bdi": round(float(vals.max()), 1),
        "sd_bdi": round(float(np.std(vals, ddof=1)), 2),
        "bdi_status": bdi_status,
        "elevated_nights": elevated_nights,
        "elevated_pct": elevated_pct,
        "trend_slope": round(slope, 4),
        "trend_p_value": round(p_val, 6),
        "trend_r_squared": round(r_val**2, 4),
        "rolling_7d": bdi_data["bdi_7d"].tolist(),
        "dates": [str(d) for d in bdi_data["date"]],
        "values": vals.tolist(),
    }


# ---------------------------------------------------------------------------
# Analysis 7: SpO2-Temperature Coupling
# ---------------------------------------------------------------------------
def analyze_spo2_temp_coupling(
    spo2: pd.DataFrame, readiness: pd.DataFrame
) -> dict[str, Any]:
    """Correlate nightly SpO2 with temperature deviation."""
    merged = pd.merge(
        spo2[["date", "spo2_average"]],
        readiness[["date", "temperature_deviation"]],
        on="date",
        how="inner",
    )

    if len(merged) < 5:
        return {"error": "Insufficient overlapping data", "n_paired": len(merged)}

    spo2_vals = merged["spo2_average"].values
    temp_vals = merged["temperature_deviation"].values

    pearson_r, pearson_p = stats.pearsonr(spo2_vals, temp_vals)
    spearman_r, spearman_p = stats.spearmanr(spo2_vals, temp_vals)

    # Interpretation
    if pearson_r < -0.3:
        interpretation = "Inverse coupling: inflammation/fever associated with lower SpO2"
    elif pearson_r > 0.3:
        interpretation = "Positive coupling: unexpected, may indicate compensatory mechanism"
    else:
        interpretation = "Weak coupling: SpO2 relatively independent of temperature"

    return {
        "n_paired": len(merged),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 6),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 6),
        "interpretation": interpretation,
        "merged_dates": [str(d) for d in merged["date"]],
        "merged_spo2": merged["spo2_average"].tolist(),
        "merged_temp": merged["temperature_deviation"].tolist(),
    }


# ---------------------------------------------------------------------------
# Analysis 8: Composite BOS Risk Score
# ---------------------------------------------------------------------------
def compute_bos_risk_score(
    trend: dict, variability: dict, desat: dict, bdi: dict, coupling: dict
) -> dict[str, Any]:
    """Compute composite BOS risk score (0-100)."""
    scores: dict[str, float] = {}

    # 1. SpO2 slope component (0-100)
    slope = trend["slope_pct_per_day"]
    if slope >= 0:
        scores["spo2_slope"] = 0.0
    elif slope >= SPO2_CONCERN_SLOPE:
        # Linear interpolation between 0 and concern threshold
        scores["spo2_slope"] = min(100, abs(slope / SPO2_CONCERN_SLOPE) * 50)
    else:
        # Beyond concern threshold
        scores["spo2_slope"] = min(100, 50 + abs(slope / SPO2_CONCERN_SLOPE) * 25)

    # 2. Variability component (0-100)
    cv = variability["cv_pct"]
    # CV < 0.5% = normal, > 1.5% = concerning
    if cv < 0.5:
        scores["spo2_variability"] = 0.0
    elif cv < 1.0:
        scores["spo2_variability"] = (cv - 0.5) / 0.5 * 50
    else:
        scores["spo2_variability"] = min(100, 50 + (cv - 1.0) / 1.0 * 50)

    # 3. Desaturation frequency (0-100)
    desat_pct = desat["absolute_desaturation_pct"]
    if desat_pct == 0:
        scores["desaturation_freq"] = 0.0
    elif desat_pct < 5:
        scores["desaturation_freq"] = desat_pct / 5 * 40
    elif desat_pct < 15:
        scores["desaturation_freq"] = 40 + (desat_pct - 5) / 10 * 30
    else:
        scores["desaturation_freq"] = min(100, 70 + (desat_pct - 15) / 20 * 30)

    # 4. BDI component (0-100)
    if bdi.get("available"):
        mean_bdi = bdi["mean_bdi"]
        if mean_bdi < BDI_NORMAL:
            scores["bdi"] = mean_bdi / BDI_NORMAL * 30
        elif mean_bdi < BDI_MILD:
            scores["bdi"] = 30 + (mean_bdi - BDI_NORMAL) / (BDI_MILD - BDI_NORMAL) * 40
        else:
            scores["bdi"] = min(100, 70 + (mean_bdi - BDI_MILD) / BDI_MILD * 30)
    else:
        scores["bdi"] = 25.0  # Unknown = moderate default

    # 5. HR decoupling component (0-100)
    if not coupling.get("coupling_assessable", True):
        scores["hr_decoupling"] = 25.0
    elif "pearson_r" in coupling and _is_finite_number(coupling["pearson_r"]):
        r = coupling["pearson_r"]
        # Normal: r < -0.3 (inverse). Concerning: r > 0
        if r < -0.3:
            scores["hr_decoupling"] = 0.0
        elif r < 0:
            scores["hr_decoupling"] = (r + 0.3) / 0.3 * 50
        else:
            scores["hr_decoupling"] = min(100, 50 + r / 0.5 * 50)
    else:
        scores["hr_decoupling"] = 25.0

    # Weighted composite
    composite = sum(scores[k] * BOS_WEIGHTS[k] for k in BOS_WEIGHTS)

    # Risk classification
    if composite < 20:
        risk_level = "LOW"
        recommendation = "Continue monitoring. No immediate action required."
    elif composite < 40:
        risk_level = "MODERATE"
        recommendation = "Consider spirometry within 2-4 weeks. Closer clinical follow-up."
    elif composite < 60:
        risk_level = "ELEVATED"
        recommendation = "Spirometry within 1-2 weeks. Consider HRCT thorax. Discuss with transplant team."
    else:
        risk_level = "HIGH"
        recommendation = "Urgent spirometry. HRCT thorax. Immediate contact with transplant team."

    return {
        "composite_score": round(composite, 1),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "component_scores": {k: round(v, 1) for k, v in scores.items()},
        "weights": BOS_WEIGHTS,
    }


# ---------------------------------------------------------------------------
# Analysis 9: Pre/Post Ruxolitinib
# ---------------------------------------------------------------------------
def analyze_ruxolitinib_effect(spo2: pd.DataFrame) -> dict[str, Any]:
    """Compare SpO2 before and after ruxolitinib start."""
    pre = spo2[spo2["date"] < TREATMENT_START]["spo2_average"].values
    post = spo2[spo2["date"] >= TREATMENT_START]["spo2_average"].values

    result: dict[str, Any] = {
        "ruxolitinib_start": str(TREATMENT_START),
        "pre_n": len(pre),
        "post_n": len(post),
    }

    if len(post) < 2:
        result["status"] = "INSUFFICIENT_POST_DATA"
        result["note"] = (
            f"Only {len(post)} night(s) after ruxolitinib start. "
            "At least 7-14 nights needed for meaningful comparison."
        )
        if len(pre) > 0:
            result["pre_mean"] = round(float(np.mean(pre)), 3)
            result["pre_sd"] = round(float(np.std(pre, ddof=1)), 3)
        if len(post) > 0:
            result["post_mean"] = round(float(np.mean(post)), 3)
        return result

    result["pre_mean"] = round(float(np.mean(pre)), 3)
    result["pre_sd"] = round(float(np.std(pre, ddof=1)), 3)
    result["post_mean"] = round(float(np.mean(post)), 3)
    result["post_sd"] = round(float(np.std(post, ddof=1)), 3)
    result["diff"] = round(result["post_mean"] - result["pre_mean"], 3)

    # Mann-Whitney U test (non-parametric, robust for small samples)
    u_stat, u_p = stats.mannwhitneyu(pre, post, alternative="two-sided")
    result["mann_whitney_u"] = round(float(u_stat), 2)
    result["mann_whitney_p"] = round(float(u_p), 6)

    # Effect size (rank-biserial correlation)
    n1, n2 = len(pre), len(post)
    rbc = 1 - (2 * u_stat) / (n1 * n2)
    result["effect_size_rbc"] = round(rbc, 4)

    # Interpretation
    if u_p < 0.05:
        if result["diff"] > 0:
            result["interpretation"] = "Statistically significant SpO2 improvement after ruxolitinib"
        else:
            result["interpretation"] = "Statistically significant SpO2 deterioration after ruxolitinib"
    else:
        result["interpretation"] = "No statistically significant change after ruxolitinib (yet)"

    result["status"] = "ANALYZED"
    return result


# ---------------------------------------------------------------------------
# HTML Report Builder
# ---------------------------------------------------------------------------
def build_html_report(
    spo2: pd.DataFrame,
    trend: dict,
    desat: dict,
    variability: dict,
    coupling: dict,
    dlco: dict,
    bdi: dict,
    temp_coupling: dict,
    bos_risk: dict,
    ruxi: dict,
    sleep: pd.DataFrame,
) -> str:
    """Build interactive Plotly HTML report."""

    dates = [str(d) for d in spo2["date"]]
    vals = spo2["spo2_average"].values

    # ===================================================================
    # Figure 1: SpO2 Trend with Regression
    # ===================================================================
    fig1 = go.Figure()

    _y_min = min(92, min(vals) - 1)
    _y_max = max(vals) + 1

    # Normal range band (95-100%) - very subtle green
    fig1.add_shape(
        type="rect", x0=dates[0], x1=dates[-1], y0=95, y1=100,
        fillcolor="rgba(16, 185, 129, 0.04)", line=dict(width=0),
        layer="below",
    )

    # Below-95% danger zone - very subtle red tint
    fig1.add_shape(
        type="rect", x0=dates[0], x1=dates[-1], y0=_y_min, y1=95,
        fillcolor="rgba(239, 68, 68, 0.03)", line=dict(width=0),
        layer="below",
    )

    # Gradient fill below SpO2 line (cyan to transparent)
    fig1.add_trace(go.Scatter(
        x=dates, y=vals,
        mode="none",
        fill="tozeroy",
        fillcolor="rgba(6, 182, 212, 0.08)",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Raw data - prominent cyan line (clinical monitor standard for SpO2)
    fig1.add_trace(go.Scatter(
        x=dates, y=vals,
        mode="markers+lines",
        name="Nightly SpO2",
        marker=dict(size=6, color=C_SPO2, line=dict(width=0)),
        line=dict(width=2.5, color=C_SPO2),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>SpO2: %{y:.1f}%<extra></extra>",
    ))

    # Regression line
    x_numeric = trend["model_x_days"]
    fitted = trend["model_fitted"]
    fig1.add_trace(go.Scatter(
        x=dates, y=fitted,
        mode="lines",
        name=f"Trend ({trend['slope_pct_per_month']:+.3f}%/mo)",
        line=dict(width=2, color=C_CRITICAL, dash="dash"),
        hovertemplate="<b>%{x|%b %d}</b><br>Trend: %{y:.2f}%<extra></extra>",
    ))

    # Thresholds - refined styling
    fig1.add_hline(
        y=SPO2_ABSOLUTE_THRESHOLD, line_dash="dot",
        line_color="rgba(239, 68, 68, 0.5)", line_width=1,
        annotation_text="94% desaturation", annotation_position="bottom right",
        annotation=dict(font=dict(size=10, color="rgba(239, 68, 68, 0.7)")),
    )
    fig1.add_hline(
        y=95, line_dash="dot",
        line_color="rgba(245, 158, 11, 0.4)", line_width=1,
        annotation_text="95% lower normal",
        annotation=dict(font=dict(size=10, color="rgba(245, 158, 11, 0.6)")),
    )

    # Ruxolitinib start
    ruxi_date = str(TREATMENT_START)
    if ruxi_date >= dates[0]:
        fig1.add_shape(type="line", x0=ruxi_date, x1=ruxi_date, y0=0, y1=1,
                       yref="paper", line=dict(color=C_RUXI, width=2, dash="dashdot"))
        fig1.add_annotation(x=ruxi_date, y=1.02, yref="paper",
                            text="Ruxolitinib start", showarrow=False,
                            font=dict(color=C_RUXI, size=11))

    # Prediction intervals
    pred_dates = []
    pred_means = []
    pred_lowers = []
    pred_uppers = []
    last_date = pd.to_datetime(dates[-1])
    for fd_str, pred_data in trend["predictions"].items():
        fd = int(fd_str.replace("d", ""))
        pred_date = last_date + timedelta(days=fd)
        pred_dates.append(str(pred_date.date()))
        pred_means.append(pred_data["predicted_spo2"])
        pred_lowers.append(pred_data["ci_lower"])
        pred_uppers.append(pred_data["ci_upper"])

    fig1.add_trace(go.Scatter(
        x=pred_dates, y=pred_means,
        mode="markers",
        name="Prediction (30/60/90d)",
        marker=dict(size=10, color=C_WARNING, symbol="diamond",
                    line=dict(width=1, color="rgba(245, 158, 11, 0.5)")),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Predicted: %{y:.1f}%<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=pred_dates + pred_dates[::-1],
        y=pred_uppers + pred_lowers[::-1],
        fill="toself",
        fillcolor="rgba(245, 158, 11, 0.08)",
        line=dict(color="rgba(245, 158, 11, 0)"),
        name="95% prediction interval",
        hoverinfo="skip",
    ))

    fig1.update_layout(
        xaxis_title="Date", yaxis_title="SpO2 (%)",
        yaxis=dict(range=[_y_min, _y_max]),
        xaxis=dict(showspikes=True, spikemode="across", spikethickness=1,
                   spikecolor="rgba(156, 163, 175, 0.3)", spikedash="dot"),
        yaxis_showspikes=True, yaxis_spikemode="across", yaxis_spikethickness=1,
        yaxis_spikecolor="rgba(156, 163, 175, 0.3)", yaxis_spikedash="dot",
        hovermode="x unified",
        height=500,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )

    # ===================================================================
    # Figure 2: Desaturation Events + Rolling Count
    # ===================================================================
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Nightly SpO2 with Desaturation Thresholds",
                        "7-Day Rolling Desaturation Events"],
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )

    # Marker sizes scaled by severity (deeper desaturation = larger marker)
    _desat_sizes = []
    _desat_colors = []
    _desat_opacities = []
    _rel_thresh = desat["relative_threshold"]
    for v in vals:
        if v < SPO2_ABSOLUTE_THRESHOLD:
            # Severe desaturation: size proportional to depth below 94%
            depth = SPO2_ABSOLUTE_THRESHOLD - v
            _desat_sizes.append(max(10, min(18, 10 + depth * 4)))
            _desat_colors.append(C_CRITICAL)
            _desat_opacities.append(min(1.0, 0.7 + depth * 0.1))
        elif v < _rel_thresh:
            _desat_sizes.append(8)
            _desat_colors.append(C_CAUTION)
            _desat_opacities.append(0.8)
        else:
            _desat_sizes.append(5)
            _desat_colors.append(C_SPO2)
            _desat_opacities.append(0.5)

    fig2.add_trace(go.Scatter(
        x=dates, y=vals, mode="markers",
        marker=dict(size=_desat_sizes, color=_desat_colors,
                    opacity=_desat_opacities,
                    line=dict(width=1, color="rgba(255,255,255,0.15)")),
        name="SpO2 (severity-coded)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>SpO2: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    # Threshold lines with subtle styling
    fig2.add_hline(y=SPO2_ABSOLUTE_THRESHOLD, line_dash="dot",
                   line_color="rgba(239, 68, 68, 0.5)", line_width=1, row=1, col=1,
                   annotation_text="94%", annotation_position="bottom right",
                   annotation=dict(font=dict(size=9, color="rgba(239, 68, 68, 0.6)")))
    fig2.add_hline(y=_rel_thresh, line_dash="dot",
                   line_color="rgba(245, 158, 11, 0.4)", line_width=1, row=1, col=1,
                   annotation_text=f"Personal baseline-2SD ({_rel_thresh:.1f}%)",
                   annotation_position="bottom right",
                   annotation=dict(font=dict(size=9, color="rgba(245, 158, 11, 0.5)")))

    # Rolling count with fill
    fig2.add_trace(go.Scatter(
        x=dates, y=desat["rolling_7d_abs"],
        mode="lines", name="<94% events (7d)",
        line=dict(color=C_CRITICAL, width=2),
        fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.08)",
        hovertemplate="<b>%{x|%b %d}</b><br>Events (7d): %{y:.0f}<extra></extra>",
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=dates, y=desat["rolling_7d_rel"],
        mode="lines", name="<baseline-2SD events (7d)",
        line=dict(color=C_CAUTION, width=2, dash="dash"),
        hovertemplate="<b>%{x|%b %d}</b><br>Relative events (7d): %{y:.0f}<extra></extra>",
    ), row=2, col=1)

    # Ruxolitinib marker
    if ruxi_date >= dates[0]:
        for row in [1, 2]:
            fig2.add_shape(type="line", x0=ruxi_date, x1=ruxi_date, y0=0, y1=1,
                           yref=f"y{row} domain" if row > 1 else "y domain",
                           xref=f"x{row}" if row > 1 else "x",
                           line=dict(color=C_RUXI, width=2, dash="dashdot"))

    # Crosshair spikes
    fig2.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                      spikecolor="rgba(156, 163, 175, 0.3)", spikedash="dot")
    fig2.update_yaxes(showspikes=True, spikemode="across", spikethickness=1,
                      spikecolor="rgba(156, 163, 175, 0.3)", spikedash="dot")

    fig2.update_layout(
        height=700,
        margin=dict(l=50, r=30, t=100, b=40),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        hovermode="x unified",
    )

    # ===================================================================
    # Figure 3: SpO2 Variability
    # ===================================================================
    fig3 = go.Figure()
    rolling_sd = variability["rolling_sd_7d"]
    # Filter out NaN
    valid_sd = [(d, s) for d, s in zip(dates, rolling_sd) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    if valid_sd:
        sd_dates, sd_vals = zip(*valid_sd)
        # Subtle fill below the line
        fig3.add_trace(go.Scatter(
            x=list(sd_dates), y=list(sd_vals),
            mode="lines",
            name="7-Day Rolling SD",
            line=dict(color=C_SPO2, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.06)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>SD: %{y:.3f}%<extra></extra>",
        ))

    # Threshold lines - refined
    fig3.add_hline(
        y=0.5, line_dash="dot", line_color="rgba(16, 185, 129, 0.4)", line_width=1,
        annotation_text="Low (<0.5%)",
        annotation=dict(font=dict(size=10, color="rgba(16, 185, 129, 0.6)")),
    )
    fig3.add_hline(
        y=1.0, line_dash="dot", line_color="rgba(245, 158, 11, 0.4)", line_width=1,
        annotation_text="Moderate (1.0%)",
        annotation=dict(font=dict(size=10, color="rgba(245, 158, 11, 0.6)")),
    )

    if ruxi_date >= dates[0]:
        fig3.add_shape(type="line", x0=ruxi_date, x1=ruxi_date, y0=0, y1=1,
                       yref="paper", line=dict(color=C_RUXI, width=2, dash="dashdot"))
        fig3.add_annotation(x=ruxi_date, y=1.02, yref="paper",
                            text="Ruxolitinib", showarrow=False,
                            font=dict(color=C_RUXI, size=10))

    fig3.update_layout(
        xaxis_title="Date", yaxis_title="Standard Deviation (%)",
        xaxis=dict(showspikes=True, spikemode="across", spikethickness=1,
                   spikecolor="rgba(156, 163, 175, 0.3)", spikedash="dot"),
        yaxis=dict(showspikes=True, spikemode="across", spikethickness=1,
                   spikecolor="rgba(156, 163, 175, 0.3)", spikedash="dot"),
        hovermode="x unified",
        height=400,
        margin=dict(l=50, r=30, t=50, b=40),
    )

    # ===================================================================
    # Figure 4: SpO2-HR Coupling
    # ===================================================================
    fig4 = make_subplots(
        rows=1, cols=2,
        subplot_titles=["SpO2 vs Nightly Mean HR", "Rolling Correlation (14d)"],
        horizontal_spacing=0.12,
    )

    # Scatter plot
    merged_sleep = pd.merge(
        spo2[["date", "spo2_average"]],
        sleep[["day", "average_heart_rate"]].rename(columns={"day": "date"}),
        on="date", how="inner",
    )
    if len(merged_sleep) > 3:
        fig4.add_trace(go.Scatter(
            x=merged_sleep["average_heart_rate"],
            y=merged_sleep["spo2_average"],
            mode="markers",
            marker=dict(size=7, color=C_SPO2, opacity=0.5,
                        line=dict(width=0.5, color="rgba(6, 182, 212, 0.3)")),
            name="Nightly pairs",
            text=[str(d) for d in merged_sleep["date"]],
            hovertemplate="<b>%{text}</b><br>HR: %{x:.1f} bpm<br>SpO2: %{y:.1f}%<extra></extra>",
        ), row=1, col=1)

        # Trendline with confidence band
        _hr_valid = merged_sleep["average_heart_rate"].dropna()
        _sp_valid = merged_sleep["spo2_average"].reindex(_hr_valid.index).dropna()
        _hr_valid = _hr_valid.reindex(_sp_valid.index)
        if len(_hr_valid) > 2 and _hr_valid.std() > 0:
            try:
                z = np.polyfit(_hr_valid.values, _sp_valid.values, 1)
                p = np.poly1d(z)
                hr_range = np.linspace(_hr_valid.min(), _hr_valid.max(), 50)
                _fitted_vals = p(hr_range)

                # Compute confidence band (approximate using residual SE)
                _residuals = _sp_valid.values - p(_hr_valid.values)
                _se_resid = float(np.std(_residuals, ddof=2))
                _ci_upper = _fitted_vals + 1.96 * _se_resid
                _ci_lower = _fitted_vals - 1.96 * _se_resid

                # Confidence band
                fig4.add_trace(go.Scatter(
                    x=np.concatenate([hr_range, hr_range[::-1]]).tolist(),
                    y=np.concatenate([_ci_upper, _ci_lower[::-1]]).tolist(),
                    fill="toself",
                    fillcolor="rgba(239, 68, 68, 0.06)",
                    line=dict(color="rgba(239, 68, 68, 0)"),
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=1)

                trendline_name = (
                    f"r={_fmt_nan(coupling.get('pearson_r'))}"
                    if coupling.get("coupling_assessable", True)
                    else "Trendline"
                )
                fig4.add_trace(go.Scatter(
                    x=hr_range, y=_fitted_vals,
                    mode="lines", name=trendline_name,
                    line=dict(color=C_CRITICAL, dash="dash", width=2),
                    hovertemplate="HR: %{x:.1f} bpm<br>Fitted SpO2: %{y:.2f}%<extra></extra>",
                ), row=1, col=1)
            except (np.linalg.LinAlgError, FloatingPointError):
                pass  # Skip trendline if SVD fails

    # Rolling correlation with color-coded fill
    if coupling.get("rolling_correlation"):
        rc = coupling["rolling_correlation"]
        _rc_dates = [r["date"] for r in rc]
        _rc_vals = [r["correlation"] for r in rc]
        # Color segments: green when negative (normal), amber/red when positive (abnormal)
        _rc_colors = [C_OK if v < -0.2 else C_CAUTION if v < 0 else C_CRITICAL for v in _rc_vals]

        fig4.add_trace(go.Scatter(
            x=_rc_dates, y=_rc_vals,
            mode="lines",
            name="14d correlation",
            line=dict(color=C_SPO2, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.06)",
            hovertemplate="<b>%{x|%b %d}</b><br>r = %{y:.3f}<extra></extra>",
        ), row=1, col=2)
        fig4.add_hline(y=0, line_dash="dot", line_color="rgba(156, 163, 175, 0.3)",
                       line_width=1, row=1, col=2)
        fig4.add_hline(y=-0.3, line_dash="dot", line_color="rgba(16, 185, 129, 0.4)",
                       line_width=1, row=1, col=2,
                       annotation_text="Normal inverse coupling",
                       annotation=dict(font=dict(size=9, color="rgba(16, 185, 129, 0.6)")))

    fig4.update_xaxes(title_text="Mean HR (bpm)", row=1, col=1)
    fig4.update_yaxes(title_text="SpO2 (%)", row=1, col=1)
    fig4.update_xaxes(title_text="Date", showspikes=True, spikemode="across",
                      spikethickness=1, spikecolor="rgba(156,163,175,0.3)",
                      spikedash="dot", row=1, col=2)
    fig4.update_yaxes(title_text="Pearson r", row=1, col=2)
    fig4.update_layout(
        height=450,
        margin=dict(l=50, r=30, t=100, b=40),
        showlegend=True,
        hovermode="closest",
    )

    # ===================================================================
    # Figure 5: BDI Trend
    # ===================================================================
    fig5 = go.Figure()
    if bdi.get("available"):
        # Individual readings as scatter
        fig5.add_trace(go.Scatter(
            x=bdi["dates"], y=bdi["values"],
            mode="markers",
            name="BDI (events/hour)",
            marker=dict(size=5, color=C_SPO2, opacity=0.4,
                        line=dict(width=0)),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>BDI: %{y:.1f} events/hr<extra></extra>",
        ))
        # Rolling 7d average - prominent trend line
        valid_rolling = [(d, v) for d, v in zip(bdi["dates"], bdi["rolling_7d"])
                         if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if valid_rolling:
            rd, rv = zip(*valid_rolling)
            fig5.add_trace(go.Scatter(
                x=list(rd), y=list(rv),
                mode="lines", name="7d rolling mean",
                line=dict(color=C_SPO2, width=2.5),
                fill="tozeroy", fillcolor="rgba(6, 182, 212, 0.06)",
                hovertemplate="<b>%{x|%b %d}</b><br>7d mean: %{y:.1f} events/hr<extra></extra>",
            ))

        # Warning threshold lines - refined dash styling
        fig5.add_hline(
            y=BDI_NORMAL, line_dash="dot",
            line_color="rgba(16, 185, 129, 0.4)", line_width=1,
            annotation_text="Normal <5",
            annotation=dict(font=dict(size=10, color="rgba(16, 185, 129, 0.6)")),
        )
        fig5.add_hline(
            y=BDI_MILD, line_dash="dot",
            line_color="rgba(245, 158, 11, 0.4)", line_width=1,
            annotation_text="Mild <15",
            annotation=dict(font=dict(size=10, color="rgba(245, 158, 11, 0.6)")),
        )

        if ruxi_date >= bdi["dates"][0]:
            fig5.add_shape(type="line", x0=ruxi_date, x1=ruxi_date, y0=0, y1=1,
                           yref="paper", line=dict(color=C_RUXI, width=2, dash="dashdot"))
            fig5.add_annotation(x=ruxi_date, y=1.02, yref="paper",
                                text="Ruxolitinib", showarrow=False,
                                font=dict(color=C_RUXI, size=10))
    else:
        fig5.add_annotation(text="Insufficient BDI data", xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False, font=dict(size=16))

    fig5.update_layout(
        xaxis_title="Date", yaxis_title="BDI (events/hour)",
        xaxis=dict(showspikes=True, spikemode="across", spikethickness=1,
                   spikecolor="rgba(156, 163, 175, 0.3)", spikedash="dot"),
        yaxis=dict(showspikes=True, spikemode="across", spikethickness=1,
                   spikecolor="rgba(156, 163, 175, 0.3)", spikedash="dot"),
        hovermode="x unified",
        height=400,
        margin=dict(l=50, r=30, t=50, b=40),
    )

    # ===================================================================
    # Figure 6: SpO2-Temperature Coupling
    # ===================================================================
    fig6 = go.Figure()
    if "merged_dates" in temp_coupling:
        # Color markers by temperature deviation (warm = orange, cool = blue)
        _temp_vals_for_color = temp_coupling["merged_temp"]
        fig6.add_trace(go.Scatter(
            x=temp_coupling["merged_temp"],
            y=temp_coupling["merged_spo2"],
            mode="markers",
            marker=dict(
                size=8, opacity=0.6,
                color=temp_coupling["merged_temp"],
                colorscale=[[0, ACCENT_BLUE], [0.5, C_SPO2], [1, ACCENT_ORANGE]],
                colorbar=dict(title=dict(text="\u0394T (\u00b0C)", side="right"),
                              thickness=12, len=0.6,
                              tickfont=dict(size=10)),
                line=dict(width=0.5, color="rgba(255,255,255,0.1)"),
            ),
            text=temp_coupling["merged_dates"],
            hovertemplate="<b>%{text}</b><br>Temp dev: %{x:.2f}\u00b0C<br>SpO2: %{y:.1f}%<extra></extra>",
            name="Nightly pairs",
        ))
        # Regression line with confidence band
        if len(temp_coupling["merged_temp"]) > 3:
            try:
                _t_arr = np.array(temp_coupling["merged_temp"])
                _s_arr = np.array(temp_coupling["merged_spo2"])
                z = np.polyfit(_t_arr, _s_arr, 1)
                p = np.poly1d(z)
                t_range = np.linspace(_t_arr.min(), _t_arr.max(), 50)
                _fitted_temp = p(t_range)

                # Confidence band
                _resid = _s_arr - p(_t_arr)
                _se = float(np.std(_resid, ddof=2))
                _ci_u = _fitted_temp + 1.96 * _se
                _ci_l = _fitted_temp - 1.96 * _se
                fig6.add_trace(go.Scatter(
                    x=np.concatenate([t_range, t_range[::-1]]).tolist(),
                    y=np.concatenate([_ci_u, _ci_l[::-1]]).tolist(),
                    fill="toself",
                    fillcolor="rgba(249, 115, 22, 0.06)",
                    line=dict(color="rgba(249, 115, 22, 0)"),
                    showlegend=False, hoverinfo="skip",
                ))

                fig6.add_trace(go.Scatter(
                    x=t_range, y=_fitted_temp,
                    mode="lines",
                    name=f"r={_fmt_nan(temp_coupling.get('pearson_r'))}",
                    line=dict(color=ACCENT_ORANGE, dash="dash", width=2),
                    hovertemplate="\u0394T: %{x:.2f}\u00b0C<br>Fitted SpO2: %{y:.2f}%<extra></extra>",
                ))
            except (np.linalg.LinAlgError, FloatingPointError):
                pass
    else:
        fig6.add_annotation(text="Insufficient data", xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False, font=dict(size=16))

    fig6.update_layout(
        xaxis_title="Temperature Deviation (\u00b0C)", yaxis_title="SpO2 (%)",
        hovermode="closest",
        height=400,
        margin=dict(l=50, r=30, t=50, b=40),
    )

    # ===================================================================
    # Figure 7: BOS Risk Score Dashboard (clinical monitoring style)
    # ===================================================================
    fig7 = go.Figure()

    components = bos_risk["component_scores"]
    comp_names = {
        "spo2_slope": "SpO2 Trend",
        "spo2_variability": "Variability",
        "desaturation_freq": "Desaturations",
        "bdi": "BDI",
        "hr_decoupling": "HR Decoupling",
    }

    _comp_labels = [comp_names.get(k, k) for k in components]
    _comp_vals = list(components.values())
    _comp_colors = [
        C_OK if v < 20 else
        "rgba(16, 185, 129, 0.7)" if v < 30 else
        C_CAUTION if v < 50 else
        C_WARNING if v < 70 else
        C_CRITICAL
        for v in _comp_vals
    ]

    # Background risk zone bands
    fig7.add_shape(type="rect", x0=-0.5, x1=len(_comp_labels) - 0.5,
                   y0=0, y1=20, fillcolor="rgba(16, 185, 129, 0.04)",
                   line=dict(width=0), layer="below")
    fig7.add_shape(type="rect", x0=-0.5, x1=len(_comp_labels) - 0.5,
                   y0=20, y1=40, fillcolor="rgba(245, 158, 11, 0.03)",
                   line=dict(width=0), layer="below")
    fig7.add_shape(type="rect", x0=-0.5, x1=len(_comp_labels) - 0.5,
                   y0=40, y1=60, fillcolor="rgba(249, 115, 22, 0.03)",
                   line=dict(width=0), layer="below")
    fig7.add_shape(type="rect", x0=-0.5, x1=len(_comp_labels) - 0.5,
                   y0=60, y1=100, fillcolor="rgba(239, 68, 68, 0.03)",
                   line=dict(width=0), layer="below")

    fig7.add_trace(go.Bar(
        x=_comp_labels,
        y=_comp_vals,
        marker=dict(
            color=_comp_colors,
            line=dict(width=1, color="rgba(255,255,255,0.08)"),
        ),
        text=[f"<b>{v:.0f}</b>" for v in _comp_vals],
        textposition="outside",
        textfont=dict(size=13),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}/100<extra></extra>",
    ))

    # Composite score line
    _composite = bos_risk["composite_score"]
    _risk_level = bos_risk["risk_level"]
    _composite_color = (
        C_OK if _composite < 20 else
        C_CAUTION if _composite < 40 else
        C_WARNING if _composite < 60 else
        C_CRITICAL
    )
    fig7.add_hline(
        y=_composite, line_dash="dash", line_color=_composite_color, line_width=2,
        annotation_text=f"Composite: {_composite:.0f}/100 ({_risk_level})",
        annotation=dict(font=dict(size=12, color=_composite_color, weight="bold")),
    )

    # Risk zone labels on right margin
    for _zone_y, _zone_label, _zone_color in [
        (10, "LOW", "rgba(16, 185, 129, 0.3)"),
        (30, "MODERATE", "rgba(245, 158, 11, 0.3)"),
        (50, "ELEVATED", "rgba(249, 115, 22, 0.3)"),
        (80, "HIGH", "rgba(239, 68, 68, 0.3)"),
    ]:
        fig7.add_annotation(
            x=1.02, y=_zone_y, xref="paper", yref="y",
            text=_zone_label, showarrow=False,
            font=dict(size=9, color=_zone_color),
            xanchor="left",
        )

    fig7.update_layout(
        yaxis_title="Component Score (0-100)",
        yaxis=dict(range=[0, 110]),
        height=450,
        margin=dict(l=50, r=80, t=50, b=40),
    )

    # ===================================================================
    # Figure 8: Pre/Post Ruxolitinib (dramatic comparison)
    # ===================================================================
    fig8 = make_subplots(
        rows=1, cols=2,
        subplot_titles=["SpO2 Timeline (Pre/Post)", "Distribution Comparison"],
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.08,
    )

    # --- Left panel: time series ---
    # Pre-treatment background tint
    if ruxi_date >= dates[0]:
        fig8.add_shape(
            type="rect", x0=dates[0], x1=ruxi_date, y0=0, y1=1,
            yref="y domain", fillcolor="rgba(156, 163, 175, 0.03)",
            line=dict(width=0), layer="below", row=1, col=1,
        )
        # Post-treatment background tint
        fig8.add_shape(
            type="rect", x0=ruxi_date, x1=dates[-1], y0=0, y1=1,
            yref="y domain", fillcolor="rgba(139, 92, 246, 0.03)",
            line=dict(width=0), layer="below", row=1, col=1,
        )

    # Pre-treatment data
    _pre_mask = [d < TREATMENT_START for d in spo2["date"]]
    _post_mask = [d >= TREATMENT_START for d in spo2["date"]]
    _pre_dates = [d for d, m in zip(dates, _pre_mask) if m]
    _pre_vals = [v for v, m in zip(vals, _pre_mask) if m]
    _post_dates = [d for d, m in zip(dates, _post_mask) if m]
    _post_vals = [v for v, m in zip(vals, _post_mask) if m]

    if _pre_dates:
        fig8.add_trace(go.Scatter(
            x=_pre_dates, y=_pre_vals,
            mode="markers+lines",
            marker=dict(size=5, color=TEXT_SECONDARY, opacity=0.6),
            line=dict(width=1.5, color="rgba(156, 163, 175, 0.4)"),
            name="Pre-ruxolitinib",
            hovertemplate="<b>%{x|%b %d}</b><br>SpO2: %{y:.1f}%<extra>Pre</extra>",
        ), row=1, col=1)

    if _post_dates:
        fig8.add_trace(go.Scatter(
            x=_post_dates, y=_post_vals,
            mode="markers+lines",
            marker=dict(size=7, color=C_RUXI, opacity=0.9,
                        line=dict(width=1, color="rgba(139, 92, 246, 0.3)")),
            line=dict(width=2.5, color=C_RUXI),
            name="Post-ruxolitinib",
            hovertemplate="<b>%{x|%b %d}</b><br>SpO2: %{y:.1f}%<extra>Post</extra>",
        ), row=1, col=1)

    # Mean lines
    if ruxi.get("pre_mean"):
        fig8.add_hline(
            y=ruxi["pre_mean"], line_dash="dash",
            line_color="rgba(156, 163, 175, 0.5)", line_width=1,
            annotation_text=f"Pre: {ruxi['pre_mean']:.2f}%",
            annotation=dict(font=dict(size=10, color=TEXT_SECONDARY)),
            row=1, col=1,
        )
    if ruxi.get("post_mean"):
        fig8.add_hline(
            y=ruxi["post_mean"], line_dash="dash",
            line_color="rgba(139, 92, 246, 0.6)", line_width=1,
            annotation_text=f"Post: {ruxi['post_mean']:.2f}%",
            annotation=dict(font=dict(size=10, color=C_RUXI)),
            row=1, col=1,
        )

    # Treatment start line
    if ruxi_date >= dates[0]:
        fig8.add_shape(type="line", x0=ruxi_date, x1=ruxi_date, y0=0, y1=1,
                       yref="y domain", line=dict(color=C_RUXI, width=3, dash="dashdot"),
                       row=1, col=1)
        fig8.add_annotation(x=ruxi_date, y=1.05, yref="y domain",
                            text="Ruxolitinib 10mg BID", showarrow=False,
                            font=dict(color=C_RUXI, size=11, weight="bold"),
                            row=1, col=1)

    # --- Right panel: box plots for distribution comparison ---
    if _pre_vals:
        fig8.add_trace(go.Box(
            y=_pre_vals,
            name="Pre",
            marker=dict(color=TEXT_SECONDARY, opacity=0.6),
            line=dict(color=TEXT_SECONDARY),
            fillcolor="rgba(156, 163, 175, 0.15)",
            boxmean="sd",
            hovertemplate="SpO2: %{y:.1f}%<extra>Pre</extra>",
        ), row=1, col=2)

    if _post_vals:
        fig8.add_trace(go.Box(
            y=_post_vals,
            name="Post",
            marker=dict(color=C_RUXI, opacity=0.8),
            line=dict(color=C_RUXI),
            fillcolor="rgba(139, 92, 246, 0.15)",
            boxmean="sd",
            hovertemplate="SpO2: %{y:.1f}%<extra>Post</extra>",
        ), row=1, col=2)

    fig8.update_xaxes(title_text="Date", showspikes=True, spikemode="across",
                      spikethickness=1, spikecolor="rgba(156,163,175,0.3)",
                      spikedash="dot", row=1, col=1)
    fig8.update_yaxes(title_text="SpO2 (%)", row=1, col=1)
    fig8.update_yaxes(title_text="SpO2 (%)", row=1, col=2)
    fig8.update_layout(
        height=450,
        margin=dict(l=50, r=30, t=100, b=40),
        hovermode="x unified",
    )

    # ===================================================================
    # Assemble full HTML using theme system
    # ===================================================================
    data_period = f"{dates[0]} to {dates[-1]}"
    n_nights = len(vals)

    # Risk badge color
    risk_colors = {"LOW": C_OK, "MODERATE": C_CAUTION, "ELEVATED": C_WARNING, "HIGH": C_CRITICAL}
    risk_color = risk_colors.get(bos_risk["risk_level"], C_DARK)

    # Concern level badge color
    concern_colors = {"HIGH": ACCENT_RED, "MODERATE": ACCENT_AMBER, "LOW": ACCENT_GREEN}

    # --- KPI Cards ---
    bos_status = "critical" if bos_risk["composite_score"] > 50 else "warning" if bos_risk["composite_score"] > 30 else "normal"
    bos_lbl = "Elevated" if bos_status in ("critical", "warning") else ""
    if trend["slope_pct_per_day"] < SPO2_CONCERN_SLOPE:
        trend_status = "critical"
        trend_lbl = "Declining"
    elif trend["concern_level"] == "MODERATE":
        trend_status = "warning"
        trend_lbl = "Moderate"
    else:
        trend_status = "normal"
        trend_lbl = ""
    spo2_status = "normal" if variability["overall_mean"] >= 95 else "warning"
    spo2_lbl = "Low" if spo2_status in ("critical", "warning") else ""
    desat_status = "critical" if desat["absolute_desaturation_pct"] > 10 else "warning" if desat["absolute_desaturation_pct"] > 3 else "normal"
    desat_lbl = "Elevated" if desat_status in ("critical", "warning") else ""
    bdi_status_kpi = "warning" if bdi.get("bdi_status") in ("MILD ELEVATION", "MODERATE", "SEVERE") else "normal"
    bdi_lbl = "Elevated" if bdi_status_kpi in ("critical", "warning") else ""
    coupling_r = coupling.get("pearson_r")
    if not coupling.get("coupling_assessable", True):
        coupling_status_kpi = "neutral"
    elif _is_finite_number(coupling_r) and coupling_r > -0.2:
        coupling_status_kpi = "warning"
    else:
        coupling_status_kpi = "normal"
    coupling_lbl = "Weak" if coupling_status_kpi == "warning" else ""

    body = make_kpi_row(
        make_kpi_card(
            "BOS Risk Score", f"{bos_risk['composite_score']:.0f}", "/100",
            status=bos_status,
            detail=bos_risk["risk_level"],
            status_label=bos_lbl,
        ),
        make_kpi_card(
            "SpO2 Mean", variability["overall_mean"], "%",
            status=spo2_status,
            detail=f"SD: {variability['overall_sd']:.2f}%, CV: {variability['cv_pct']:.2f}%",
            status_label=spo2_lbl,
        ),
        make_kpi_card(
            "SpO2 Trend", f"{trend['slope_pct_per_month']:+.2f}", "%/mo",
            status=trend_status,
            detail=f"p={trend['p_value']:.4f}, R\u00b2={trend['r_squared']:.3f}",
            status_label=trend_lbl,
        ),
        make_kpi_card(
            "Desaturations (<94%)", f"{desat['absolute_desaturation_count']}/{n_nights}", "",
            status=desat_status,
            detail=f"{desat['absolute_desaturation_pct']:.1f}% of nights",
            status_label=desat_lbl,
        ),
        make_kpi_card(
            "BDI Mean", bdi.get("mean_bdi", "N/A"), "",
            status=bdi_status_kpi,
            detail=bdi.get("bdi_status", "Unavailable"),
            status_label=bdi_lbl,
        ),
        make_kpi_card(
            "SpO2-HR Coupling", f"r={_fmt_nan(coupling.get('pearson_r'))}", "",
            status=coupling_status_kpi,
            detail=coupling.get("coupling_status", "N/A"),
            status_label=coupling_lbl,
        ),
    )

    # --- Recommendation callout ---
    body += (
        '<div class="odt-narrative">'
        f'<h3 style="margin-bottom:8px;color:{TEXT_PRIMARY}">Recommendation</h3>'
        f'<p style="margin-bottom:8px"><strong>{bos_risk["recommendation"]}</strong></p>'
        f'<p style="font-size:0.8125rem;color:{TEXT_SECONDARY}"><em>Note: SpO2 monitoring is supplementary '
        'screening and cannot substitute for pulmonary function testing (spirometry). Normal SpO2 does not '
        'exclude early-stage BOS (bronchiolitis obliterans syndrome).</em></p>'
        '</div>'
    )

    # --- Section 1: SpO2 Nightly Trend ---
    prediction_rows = ''.join(
        f"<tr><td>{k}</td><td>{v['predicted_spo2']:.1f}%</td>"
        f"<td>[{v['ci_lower']:.1f}, {v['ci_upper']:.1f}]%</td></tr>"
        for k, v in trend["predictions"].items()
    )
    concern_color = concern_colors.get(trend["concern_level"], ACCENT_GREEN)
    trend_interp = (
        f'<p><strong>Slope:</strong> {trend["slope_pct_per_day"]:.5f}%/day '
        f'({trend["slope_pct_per_month"]:+.3f}%/month). '
        f'95% CI: [{trend["slope_95ci"][0]:.5f}, {trend["slope_95ci"][1]:.5f}]. '
        f'p={trend["p_value"]:.4f}.</p>'
        '<p><strong>Prediction:</strong></p>'
        '<table><tr><th>Time Horizon</th><th>Predicted SpO2</th><th>95% PI</th></tr>'
        f'{prediction_rows}</table>'
        f'<p><strong>Concern Level:</strong> '
        f'<span class="risk-badge" style="background:{concern_color}">'
        f'{trend["concern_level"]}</span></p>'
    )
    body += make_section(
        "1. SpO2 Nightly Trend",
        f'<div id="chart1" class="chart-box">Loading...</div>{trend_interp}',
    )

    # --- Section 2: Desaturation Events ---
    freq_change = (
        f'- <span style="color:{ACCENT_RED}">INCREASING FREQUENCY</span>'
        if desat["frequency_increasing"] else "- Stable"
    )
    desat_interp = (
        f'<p><strong>Absolute threshold (&lt;94%):</strong> '
        f'{desat["absolute_desaturation_count"]} nights ({desat["absolute_desaturation_pct"]:.1f}%)</p>'
        f'<p><strong>Relative threshold (&lt;{desat["relative_threshold"]:.2f}%, baseline-2SD):</strong> '
        f'{desat["relative_desaturation_count"]} nights ({desat["relative_desaturation_pct"]:.1f}%)</p>'
        f'<p><strong>Frequency trend:</strong> First half: {desat["first_half_desat_rate"]:.1%} '
        f'vs second half: {desat["second_half_desat_rate"]:.1%} {freq_change}</p>'
    )
    body += make_section(
        "2. Desaturation Events",
        f'<div id="chart2" class="chart-box">Loading...</div>{desat_interp}',
    )

    # --- Section 3: SpO2 Variability ---
    var_change = (
        f'- <span style="color:{ACCENT_RED}">INCREASING VARIABILITY</span>'
        if variability["variability_increasing"] else "- Stable"
    )
    var_interp = (
        f'<p><strong>Overall SD:</strong> {variability["overall_sd"]:.3f}% | '
        f'<strong>CV:</strong> {variability["cv_pct"]:.3f}%</p>'
        f'<p><strong>Night-to-night difference SD:</strong> {variability["diff_sd"]:.3f}%</p>'
        f'<p><strong>First half SD:</strong> {variability["sd_first_half"]:.3f}% vs '
        f'<strong>second half:</strong> {variability["sd_second_half"]:.3f}% {var_change}</p>'
    )
    body += make_section(
        "3. SpO2 Variability",
        f'<div id="chart3" class="chart-box">Loading...</div>{var_interp}',
    )

    # --- Section 4: SpO2-HR Coupling ---
    coupling_note = (
        f'<p><strong>Interpretation:</strong> {coupling.get("interpretation", "N/A")}</p>'
    )
    if coupling.get("coupling_assessable", True):
        coupling_note += (
            f'<p style="font-size:0.8125rem;color:{TEXT_SECONDARY}">Normal: Inverse relationship '
            '(SpO2 down -> HR up, compensatory). Decoupling (positive or zero correlation) '
            '= autonomic dysfunction or respiratory failure.</p>'
        )
    else:
        coupling_note += (
            f'<p style="font-size:0.8125rem;color:{TEXT_SECONDARY}">Directional coupling is not '
            'interpreted when the overlapping series lacks enough variation for a reliable '
            'correlation estimate. The BOS score therefore uses a neutral HR-decoupling component.</p>'
        )
    coupling_interp = (
        f'<p><strong>Pearson r:</strong> {_fmt_nan(coupling.get("pearson_r"))} '
        f'(p={_fmt_nan(coupling.get("pearson_p"))})</p>'
        f'<p><strong>Spearman rho:</strong> {_fmt_nan(coupling.get("spearman_r"))} '
        f'(p={_fmt_nan(coupling.get("spearman_p"))})</p>'
        f'<p><strong>Status:</strong> {coupling.get("coupling_status", "N/A")}</p>'
        f'{coupling_note}'
    )
    body += make_section(
        "4. SpO2-HR Coupling",
        f'<div id="chart4" class="chart-box">Loading...</div>{coupling_interp}',
    )

    # --- Section 5: DLCO-SpO2 Correlation ---
    dlco_source = dlco.get("overlapping", []) or dlco.get("dlco_measurements", [])
    dlco_rows = ''.join(
        f'<tr><td>{m["date"]}</td><td>{m["dlco_pct"]:.0f}%</td>'
        f'<td>{m["context"]}</td><td>{m.get("concurrent_spo2_mean", "No data")}</td></tr>'
        for m in dlco_source
    )
    dlco_trajectory = ' -> '.join(f"{v:.0f}% ({c})" for _, v, c in DLCO_MEASUREMENTS)
    dlco_overlap_note = (
        'SpO2 data overlaps with DLCO measurements.'
        if dlco.get("has_overlap")
        else 'SpO2 monitoring started after the last DLCO measurement. Direct correlation not possible, '
             'but trend direction is consistent with DLCO decline.'
    )
    dlco_html = (
        '<p><strong>DLCO Measurements:</strong></p>'
        '<table><tr><th>Date</th><th>DLCO%</th><th>Context</th><th>Concurrent SpO2</th></tr>'
        f'{dlco_rows}</table>'
        f'<p><strong>DLCO Trajectory:</strong> {dlco_trajectory}</p>'
        f'<p>{dlco_overlap_note}</p>'
    )
    body += make_section("5. DLCO-SpO2 Correlation", dlco_html)

    # --- Section 6: BDI ---
    if bdi.get("available"):
        bdi_interp = (
            f'<p><strong>Mean BDI:</strong> {bdi.get("mean_bdi", "N/A")} '
            f'events/hour (status: {bdi.get("bdi_status", "N/A")})</p>'
            f'<p><strong>Elevated nights (>=5):</strong> {bdi.get("elevated_nights", "N/A")} '
            f'of {bdi.get("n_readings", 0)} ({bdi.get("elevated_pct", "N/A")}%)</p>'
            f'<p><strong>BDI trend:</strong> slope={bdi.get("trend_slope", "N/A")}/day '
            f'(p={bdi.get("trend_p_value", "N/A")})</p>'
        )
    else:
        bdi_interp = '<p>Insufficient BDI data available.</p>'
    body += make_section(
        "6. Breathing Disturbance Index (BDI)",
        f'<div id="chart5" class="chart-box">Loading...</div>{bdi_interp}',
    )

    # --- Section 7: SpO2-Temperature Coupling ---
    temp_interp = (
        f'<p><strong>Pearson r:</strong> {temp_coupling.get("pearson_r", "N/A")} '
        f'(p={temp_coupling.get("pearson_p", "N/A")})</p>'
        f'<p><strong>Interpretation:</strong> {temp_coupling.get("interpretation", "N/A")}</p>'
    )
    body += make_section(
        "7. SpO2-Temperature Coupling",
        f'<div id="chart6" class="chart-box">Loading...</div>{temp_interp}',
    )

    # --- Section 8: BOS Risk Score Dashboard ---
    components = bos_risk["component_scores"]
    comp_names = {
        "spo2_slope": "SpO2 Trend",
        "spo2_variability": "SpO2 Variability",
        "desaturation_freq": "Desaturation Frequency",
        "bdi": "Breathing Disturbance Index (BDI)",
        "hr_decoupling": "HR Decoupling",
    }
    risk_table_rows = ''.join(
        f'<tr><td>{comp_names.get(k, k)}</td><td>{v:.0f}/100</td>'
        f'<td>{BOS_WEIGHTS[k]:.0%}</td><td>{v * BOS_WEIGHTS[k]:.1f}</td></tr>'
        for k, v in components.items()
    )
    risk_interp = (
        f'<p><strong>Composite Score:</strong> {bos_risk["composite_score"]:.0f}/100 - '
        f'<span class="risk-badge" style="background:{risk_color}">{bos_risk["risk_level"]}</span></p>'
        '<table><tr><th>Component</th><th>Score</th><th>Weight</th><th>Contribution</th></tr>'
        f'{risk_table_rows}</table>'
    )
    body += make_section(
        "8. BOS Risk Score",
        f'<div id="chart7" class="chart-box">Loading...</div>{risk_interp}',
    )

    # --- Section 9: Pre/Post Ruxolitinib ---
    ruxi_stats = (
        f'<p><strong>Pre-ruxolitinib ({ruxi.get("pre_n", 0)} nights):</strong> '
        f'{ruxi.get("pre_mean", "N/A")}% (SD: {ruxi.get("pre_sd", "N/A")}%)</p>'
        f'<p><strong>Post-ruxolitinib ({ruxi.get("post_n", 0)} nights):</strong> '
        f'{ruxi.get("post_mean", "N/A")}%</p>'
        f'<p><strong>Status:</strong> {ruxi.get("interpretation", ruxi.get("note", "N/A"))}</p>'
    )
    if ruxi.get("status") == "ANALYZED":
        ruxi_stats += (
            f'<p><strong>Mann-Whitney U:</strong> {ruxi.get("mann_whitney_u", "N/A")} '
            f'(p={ruxi.get("mann_whitney_p", "N/A")}), '
            f'effect size (rbc)={ruxi.get("effect_size_rbc", "N/A")}</p>'
        )
    body += make_section(
        "9. Pre/Post Ruxolitinib",
        f'<div id="chart8" class="chart-box">Loading...</div>{ruxi_stats}',
    )

    # --- Extra CSS for report-specific elements ---
    extra_css = f"""
.risk-badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    color: white;
    font-weight: 700;
    font-size: 0.8125rem;
}}
.finding {{
    background: rgba(245,158,11,0.1);
    border-left: 4px solid {ACCENT_AMBER};
    padding: 12px 16px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
}}
.finding.critical {{
    background: rgba(239,68,68,0.1);
    border-left-color: {ACCENT_RED};
}}
.finding.ok {{
    background: rgba(16,185,129,0.1);
    border-left-color: {ACCENT_GREEN};
}}
"""

    # --- Build JS for chart rendering ---
    chart_js_parts = []
    for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8], 1):
        fig_json = fig.to_json()
        chart_js_parts.append(
            f"var fig{i} = {fig_json};\n"
            f"var chart{i}El = document.getElementById('chart{i}');\n"
            f"if (chart{i}El) {{\n"
            f"  chart{i}El.innerHTML = '';\n"
            f"  chart{i}El.style.height = (fig{i}.layout.height || 450) + 'px';\n"
            f"  Plotly.newPlot('chart{i}', fig{i}.data, fig{i}.layout, "
            f"{{responsive: true, displayModeBar: true, displaylogo: false, "
            f"modeBarButtonsToRemove: ['lasso2d', 'select2d']}}).then((graphDiv) => {{\n"
            f"    window.__odtEnhancePlotly?.(graphDiv);\n"
            f"    Plotly.Plots.resize(graphDiv);\n"
            f"  }});\n"
            f"}}\n"
        )
    extra_js = "\n".join(chart_js_parts)

    # --- Assemble full page ---
    html = wrap_html(
        title="SpO2 & BOS Screening",
        body_content=body,
        report_id="spo2",
        subtitle=(
            f"Post-HSCT Bronchiolitis Obliterans Early Detection - "
            f"{data_period} ({n_nights} nights)"
        ),
        extra_css=extra_css,
        extra_js=extra_js,
    )

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Run all SpO2 BOS screening analyses."""
    print("=" * 70)
    print("SpO2 BOS SCREENING - Post-HSCT Bronchiolitis Obliterans Early Detection")
    print("=" * 70)

    if not DATABASE_PATH.exists():
        print(f"ERROR: Database not found: {DATABASE_PATH}")
        return 1

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    try:
        # Load data
        print("\n[1/10] Loading data...")
        spo2 = load_spo2(conn)
        nightly_hr = load_nightly_hr(conn)
        sleep = load_sleep_periods(conn)
        readiness = load_readiness(conn)

        if spo2.empty:
            print("  WARNING: No SpO2 data found. Skipping SpO2/BOS analysis.")
            conn.close()
            return
        print(f"  SpO2: {len(spo2)} nights ({spo2['date'].min()} to {spo2['date'].max()})")
        print(f"  HR (nightly): {len(nightly_hr)} days")
        print(f"  Sleep periods: {len(sleep)} periods")
        print(f"  Readiness: {len(readiness)} days")

        # Run analyses
        print("\n[2/10] SpO2 linear trend...")
        trend = analyze_spo2_trend(spo2)
        print(f"  Slope: {trend['slope_pct_per_day']:.5f}%/day ({trend['slope_pct_per_month']:+.3f}%/mo)")
        print(f"  p={trend['p_value']:.4f}, R2={trend['r_squared']:.4f}")
        print(f"  Concern level: {trend['concern_level']}")

        print("\n[3/10] Desaturation events...")
        desat = analyze_desaturation_events(spo2)
        print(f"  Absolute (<94%): {desat['absolute_desaturation_count']} nights ({desat['absolute_desaturation_pct']:.1f}%)")
        print(f"  Relative (<{desat['relative_threshold']:.2f}%): {desat['relative_desaturation_count']} nights ({desat['relative_desaturation_pct']:.1f}%)")
        print(f"  Frequency increasing: {desat['frequency_increasing']}")

        print("\n[4/10] SpO2 variability...")
        variability = analyze_spo2_variability(spo2)
        print(f"  SD: {variability['overall_sd']:.3f}%, CV: {variability['cv_pct']:.3f}%")
        print(f"  Variability increasing: {variability['variability_increasing']}")

        print("\n[5/10] SpO2-HR coupling...")
        coupling = analyze_spo2_hr_coupling(spo2, nightly_hr, sleep)
        if "error" not in coupling:
            print(f"  Pearson r={coupling['pearson_r']:.4f} (p={coupling['pearson_p']:.4f})")
            print(f"  Status: {coupling['coupling_status']}")
        else:
            print(f"  {coupling['error']}")

        print("\n[6/10] DLCO-SpO2 correlation...")
        dlco = analyze_dlco_spo2(spo2)
        print(f"  Overlapping measurements: {len(dlco.get('overlapping', []))}")

        print("\n[7/10] Breathing Disturbance Index...")
        bdi = analyze_bdi(spo2)
        if bdi.get("available"):
            print(f"  Mean BDI: {bdi['mean_bdi']:.1f} ({bdi['bdi_status']})")
            print(f"  Elevated nights: {bdi['elevated_nights']}/{bdi['n_readings']} ({bdi['elevated_pct']:.1f}%)")
        else:
            print("  Insufficient BDI data")

        print("\n[8/10] SpO2-temperature coupling...")
        temp_coupling = analyze_spo2_temp_coupling(spo2, readiness)
        if "error" not in temp_coupling:
            print(f"  Pearson r={temp_coupling['pearson_r']:.4f} (p={temp_coupling['pearson_p']:.4f})")
            print(f"  {temp_coupling['interpretation']}")
        else:
            print(f"  {temp_coupling['error']}")

        print("\n[9/10] BOS risk score...")
        bos_risk = compute_bos_risk_score(trend, variability, desat, bdi, coupling)
        print(f"  COMPOSITE SCORE: {bos_risk['composite_score']:.0f}/100 - {bos_risk['risk_level']}")
        print(f"  Component scores: {bos_risk['component_scores']}")
        print(f"  Recommendation: {bos_risk['recommendation']}")

        print("\n[10/10] Pre/post ruxolitinib...")
        ruxi = analyze_ruxolitinib_effect(spo2)
        print(f"  Pre: {ruxi.get('pre_n', 0)} nights, mean={ruxi.get('pre_mean', 'N/A')}%")
        print(f"  Post: {ruxi.get('post_n', 0)} nights, mean={ruxi.get('post_mean', 'N/A')}%")
        print(f"  Status: {ruxi.get('interpretation', ruxi.get('note', 'N/A'))}")

        # Build HTML report
        print("\nBuilding HTML report...")
        html = build_html_report(
            spo2, trend, desat, variability, coupling, dlco, bdi, temp_coupling, bos_risk, ruxi, sleep
        )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        HTML_OUTPUT.write_text(html, encoding="utf-8")
        print(f"  HTML report: {HTML_OUTPUT}")

        # Build JSON metrics
        metrics = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_range": {
                "start": str(spo2["date"].min()),
                "end": str(spo2["date"].max()),
                "n_nights": len(spo2),
            },
            "trend": {
                "slope_pct_per_day": trend["slope_pct_per_day"],
                "slope_pct_per_month": trend["slope_pct_per_month"],
                "slope_95ci": trend["slope_95ci"],
                "p_value": trend["p_value"],
                "r_squared": trend["r_squared"],
                "concern_level": trend["concern_level"],
                "predictions": trend["predictions"],
            },
            "desaturation": {
                "absolute_count": desat["absolute_desaturation_count"],
                "absolute_pct": desat["absolute_desaturation_pct"],
                "relative_count": desat["relative_desaturation_count"],
                "relative_pct": desat["relative_desaturation_pct"],
                "baseline_mean": desat["baseline_mean"],
                "baseline_sd": desat["baseline_sd"],
                "relative_threshold": desat["relative_threshold"],
                "frequency_increasing": desat["frequency_increasing"],
            },
            "variability": {
                "overall_sd": variability["overall_sd"],
                "cv_pct": variability["cv_pct"],
                "diff_sd": variability["diff_sd"],
                "variability_increasing": variability["variability_increasing"],
            },
            "spo2_hr_coupling": {
                "pearson_r": coupling.get("pearson_r"),
                "pearson_p": coupling.get("pearson_p"),
                "spearman_r": coupling.get("spearman_r"),
                "spearman_p": coupling.get("spearman_p"),
                "coupling_status": coupling.get("coupling_status"),
                "interpretation": coupling.get("interpretation"),
                "coupling_assessable": coupling.get("coupling_assessable"),
            },
            "bdi": {
                "available": bdi.get("available", False),
                "mean": bdi.get("mean_bdi"),
                "median": bdi.get("median_bdi"),
                "status": bdi.get("bdi_status"),
                "elevated_pct": bdi.get("elevated_pct"),
                "trend_slope": bdi.get("trend_slope"),
            },
            "temp_coupling": {
                "pearson_r": temp_coupling.get("pearson_r"),
                "pearson_p": temp_coupling.get("pearson_p"),
                "interpretation": temp_coupling.get("interpretation"),
            },
            "bos_risk": {
                "composite_score": bos_risk["composite_score"],
                "risk_level": bos_risk["risk_level"],
                "recommendation": bos_risk["recommendation"],
                "component_scores": bos_risk["component_scores"],
            },
            "ruxolitinib": {
                "start_date": str(TREATMENT_START),
                "pre_n": ruxi.get("pre_n"),
                "post_n": ruxi.get("post_n"),
                "pre_mean": ruxi.get("pre_mean"),
                "post_mean": ruxi.get("post_mean"),
                "status": ruxi.get("status"),
                "mann_whitney_p": ruxi.get("mann_whitney_p"),
                "effect_size": ruxi.get("effect_size_rbc"),
                "interpretation": ruxi.get("interpretation", ruxi.get("note")),
            },
        }

        def _default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                v = float(o)
                return None if math.isnan(v) or math.isinf(v) else v
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

        JSON_OUTPUT.write_text(json.dumps(_sanitize_nan(metrics), indent=2, ensure_ascii=False, default=_default), encoding="utf-8")
        print(f"  JSON metrics: {JSON_OUTPUT}")
    finally:
        conn.close()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY - KEY FINDINGS")
    print("=" * 70)
    print(f"  SpO2 mean:             {variability['overall_mean']:.1f}% (SD {variability['overall_sd']:.2f}%)")
    print(f"  SpO2 trend:            {trend['slope_pct_per_month']:+.3f}%/month (p={trend['p_value']:.4f})")
    print(f"  Desaturations (<94%):  {desat['absolute_desaturation_count']}/{len(spo2)} nights ({desat['absolute_desaturation_pct']:.1f}%)")
    print(f"  BDI:                   {bdi.get('mean_bdi', 'N/A')} ({bdi.get('bdi_status', 'N/A')})")
    print(f"  SpO2-HR coupling:      r={_fmt_nan(coupling.get('pearson_r'))} ({coupling.get('coupling_status', 'N/A')})")
    print(f"  BOS risk score:        {bos_risk['composite_score']:.0f}/100 ({bos_risk['risk_level']})")
    print(f"  Recommendation:        {bos_risk['recommendation']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
