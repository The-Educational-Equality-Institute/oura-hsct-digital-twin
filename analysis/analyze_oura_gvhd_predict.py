#!/usr/bin/env python3
"""
GVHD Flare Prediction System from Oura Ring Wearable Data

Implements a multi-stream prediction system for chronic GVHD flare detection
in a post-HSCT patient using wearable biometric data from Oura Ring.

Analyses:
  1. Temperature Fluctuation Analysis - regime change detection, variability
  2. Multi-Stream GVHD Composite Score - weighted 6-stream daily index
  3. Recurrent Switching Linear Dynamical System (rSLDS) - 4 discrete states
     with continuous linear dynamics within each state (Linderman et al. 2017)
  4. Early Warning Alert System - retrospective validation against known events
  5. Predictive Feature Importance - mutual information ranking
  6. BOS Risk Integration - pulmonary GVHD + systemic composite

The rSLDS is a strict upgrade over the previous HMM: it models both discrete
regime switching AND continuous linear dynamics within each regime. The recurrent
transitions allow the discrete state to depend on the continuous latent state,
capturing how gradual physiological deterioration triggers state transitions.

See config.py for patient details.
Known event date and treatment dates are loaded from config.

Output:
  - Interactive HTML report: reports/gvhd_prediction_report.html
  - JSON metrics: reports/gvhd_prediction_metrics.json

Usage:
    python analysis/analyze_oura_gvhd_predict.py
"""

from __future__ import annotations

import html as _html_escape_mod
import io
import json
import random
import sqlite3
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress ssm/tqdm progress bars
import os as _os

_os.environ["TQDM_DISABLE"] = "1"

try:
    import ssm as _ssm_module

    SSM_AVAILABLE = True
except (ImportError, Exception) as _ssm_err:
    SSM_AVAILABLE = False

try:
    from hmmlearn.hmm import GaussianHMM as _HMMLearnGaussianHMM

    HMMLEARN_AVAILABLE = True
except (ImportError, Exception) as _hmm_err:
    HMMLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Path resolution and config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATABASE_PATH,
    REPORTS_DIR,
    TREATMENT_START,
    KNOWN_EVENT_DATE,
    HEV_DIAGNOSIS_DATE,
    PATIENT_LABEL,
    DATA_START as _DATA_START_DATE,
)
from _hardening import safe_divide
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    BG_PRIMARY,
    BG_ELEVATED,
    BORDER_SUBTLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_PURPLE,
    ACCENT_CYAN,
)
from _bos_risk import load_bos_risk

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "gvhd_prediction_report.html"
JSON_OUTPUT = REPORTS_DIR / "gvhd_prediction_metrics.json"

# ---------------------------------------------------------------------------
# Clinical context
# ---------------------------------------------------------------------------
RUXOLITINIB_START = str(TREATMENT_START)  # JAK inhibitor for cGVHD
KNOWN_EVENT_DATE = str(KNOWN_EVENT_DATE)  # Convert date->str for comparisons
HEV_DIAGNOSIS_DATE = str(HEV_DIAGNOSIS_DATE)  # Convert date->str
HEV_DIAGNOSIS = HEV_DIAGNOSIS_DATE  # Hepatitis E diagnosed
DATA_START = str(_DATA_START_DATE)  # String form for SQL queries
# DATA_END is resolved dynamically from the database at load time.
# Use _get_data_end() instead of DATA_END directly to ensure lazy resolution.
DATA_END: str | None = None  # set by _get_data_end() before first use


def _get_data_end() -> str:
    """Lazy getter that resolves DATA_END on first access."""
    global DATA_END
    if DATA_END is None:
        DATA_END = _resolve_data_end()
    return DATA_END


def _resolve_data_end() -> str:
    """Query database for the latest available date across key tables."""
    import sqlite3 as _sql

    with _sql.connect(str(DATABASE_PATH)) as conn:
        row = conn.execute(
            "SELECT MAX(d) FROM ("
            "  SELECT MAX(substr(timestamp,1,10)) AS d FROM oura_heart_rate"
            "  UNION ALL SELECT MAX(day) FROM oura_sleep_periods"
            "  UNION ALL SELECT MAX(date) FROM oura_readiness"
            ")"
        ).fetchone()
    if row and row[0]:
        return row[0]
    raise RuntimeError(
        "Unable to determine latest available Oura date from the database"
    )


# BOS risk payload — loaded at runtime from canonical SpO2/BOS output
_BOS_RISK = load_bos_risk(REPORTS_DIR)
try:
    BOS_RISK_SCORE = (
        float(_BOS_RISK.get("composite_score"))
        if _BOS_RISK.get("composite_score") is not None
        else None
    )
except (TypeError, ValueError):
    BOS_RISK_SCORE = None
BOS_RISK_LEVEL = _BOS_RISK.get("risk_level")
BOS_RISK_RECOMMENDATION = _BOS_RISK.get("recommendation")

# GVHD composite score weights (sum = 1.0)
WEIGHTS = {
    "temperature": 0.25,
    "spo2": 0.15,
    "hrv": 0.20,
    "sleep_frag": 0.15,
    "resting_hr": 0.15,
    "activity": 0.10,
}

# rSLDS configuration (recurrent Switching Linear Dynamical System)
N_STATES = 4
STATE_NAMES = ["Remission", "Pre-flare", "Active Flare", "Recovery"]
STATE_COLORS = [ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, ACCENT_BLUE]
RSLDS_BASE_SEED = 42

# Alert thresholds
YELLOW_PREFLARE_PROB = 0.30
YELLOW_CONSEC_DAYS = 3
RED_PREFLARE_PROB = 0.50
RED_FLARE_PROB = 0.20

# Visual palette (used by chart traces — NOT the HTML layout)
COLORS = {
    "pre": ACCENT_BLUE,
    "post": ACCENT_RED,
    "warning": ACCENT_AMBER,
    "alert": ACCENT_RED,
    "recovery": ACCENT_GREEN,
    "bg": BG_PRIMARY,
}

# ---------------------------------------------------------------------------
# Global metrics collector
# ---------------------------------------------------------------------------
metrics: dict[str, Any] = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "patient": PATIENT_LABEL,
    "data_range": {"start": DATA_START, "end": None},
    "known_event": KNOWN_EVENT_DATE,
}
figures: list[go.Figure] = []
progress_log: list[str] = []


def log(tag: str, msg: str) -> None:
    """Print progress with tag prefix."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{tag}] {ts} - {msg}"
    print(line)
    progress_log.append(line)


# ===========================================================================
# Section 1: Data Loading
# ===========================================================================
def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all Oura tables into a dict of DataFrames."""
    log("DATA", "Loading Oura biometric data from SQLite...")
    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    data = {}

    # Readiness (temperature deviation is the KEY signal)
    data["readiness"] = pd.read_sql_query(
        """SELECT date, temperature_deviation, score, recovery_index,
                  resting_heart_rate, hrv_balance, body_temperature
           FROM oura_readiness
           WHERE date >= ? AND date <= ?
           ORDER BY date""",
        conn,
        params=(DATA_START, _get_data_end()),
        parse_dates=["date"],
    )
    log("DATA", f"  Readiness: {len(data['readiness'])} rows")

    # Sleep periods (main nightly aggregates)
    data["sleep"] = pd.read_sql_query(
        """SELECT day as date, average_hrv, average_heart_rate, average_breath,
                  total_sleep_duration, rem_sleep_duration, deep_sleep_duration,
                  light_sleep_duration, awake_time, efficiency, lowest_heart_rate,
                  restless_periods, period_id
           FROM oura_sleep_periods
           WHERE day >= ? AND day <= ?
           ORDER BY day""",
        conn,
        params=(DATA_START, _get_data_end()),
        parse_dates=["date"],
    )
    log("DATA", f"  Sleep periods: {len(data['sleep'])} rows")

    # SpO2 (filter sentinel 0 values)
    data["spo2"] = pd.read_sql_query(
        """SELECT date, spo2_average, breathing_disturbance_index
           FROM oura_spo2
           WHERE date >= ? AND date <= ? AND spo2_average > 0
           ORDER BY date""",
        conn,
        params=(DATA_START, _get_data_end()),
        parse_dates=["date"],
    )
    log("DATA", f"  SpO2: {len(data['spo2'])} rows (after filtering zeros)")

    # HRV 5-minute intervals
    data["hrv"] = pd.read_sql_query(
        """SELECT timestamp, rmssd FROM oura_hrv
           WHERE substr(timestamp, 1, 10) >= ? AND substr(timestamp, 1, 10) <= ?
           ORDER BY timestamp""",
        conn,
        params=(DATA_START, _get_data_end()),
    )
    # Parse timestamps with timezone awareness
    data["hrv"]["timestamp"] = pd.to_datetime(data["hrv"]["timestamp"], utc=True)
    data["hrv"]["date"] = data["hrv"]["timestamp"].dt.date
    log("DATA", f"  HRV intervals: {len(data['hrv'])} rows")

    # Heart rate
    data["heart_rate"] = pd.read_sql_query(
        """SELECT timestamp, bpm FROM oura_heart_rate
           WHERE substr(timestamp, 1, 10) >= ? AND substr(timestamp, 1, 10) <= ?
           ORDER BY timestamp""",
        conn,
        params=(DATA_START, _get_data_end()),
    )
    data["heart_rate"]["timestamp"] = pd.to_datetime(
        data["heart_rate"]["timestamp"], utc=True
    )
    data["heart_rate"]["date"] = data["heart_rate"]["timestamp"].dt.date
    log("DATA", f"  Heart rate: {len(data['heart_rate'])} rows")

    # Activity
    data["activity"] = pd.read_sql_query(
        """SELECT date, score, steps FROM oura_activity
           WHERE date >= ? AND date <= ?
           ORDER BY date""",
        conn,
        params=(DATA_START, _get_data_end()),
        parse_dates=["date"],
    )
    log("DATA", f"  Activity: {len(data['activity'])} rows")

    # Stress
    data["stress"] = pd.read_sql_query(
        """SELECT date, stress_high, recovery_high FROM oura_stress
           WHERE date >= ? AND date <= ?
           ORDER BY date""",
        conn,
        params=(DATA_START, _get_data_end()),
        parse_dates=["date"],
    )
    log("DATA", f"  Stress: {len(data['stress'])} rows")

    # Sleep epochs (for fragmentation analysis)
    data["epochs"] = pd.read_sql_query(
        """SELECT e.period_id, e.epoch_index, e.phase
           FROM oura_sleep_epochs e
           JOIN oura_sleep_periods sp ON e.period_id = sp.period_id
           WHERE sp.day >= ? AND sp.day <= ?
           ORDER BY e.period_id, e.epoch_index""",
        conn,
        params=(DATA_START, _get_data_end()),
    )
    log("DATA", f"  Sleep epochs: {len(data['epochs'])} rows")

    conn.close()
    log("DATA", "Data loading complete.")
    return data


# ===========================================================================
# Section 2: Build Daily Feature Matrix
# ===========================================================================
def build_daily_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all streams into a single daily feature matrix.
    Each row = one calendar day with all biometric features.
    """
    log("FEATURES", "Building daily feature matrix...")

    # Date range
    dates = pd.date_range(DATA_START, _get_data_end(), freq="D")
    daily = pd.DataFrame({"date": dates})

    # --- Temperature deviation (from readiness) ---
    temp = data["readiness"][["date", "temperature_deviation"]].copy()
    temp = temp.rename(columns={"temperature_deviation": "temp_dev"})
    daily = daily.merge(temp, on="date", how="left")

    # --- Readiness score ---
    rdns = data["readiness"][["date", "score"]].copy()
    rdns = rdns.rename(columns={"score": "readiness_score"})
    daily = daily.merge(rdns, on="date", how="left")

    # --- HRV (daily median of 5-min RMSSD) ---
    hrv_daily = (
        data["hrv"]
        .groupby("date")
        .agg(
            hrv_median=("rmssd", "median"),
            hrv_mean=("rmssd", "mean"),
            hrv_std=("rmssd", "std"),
            hrv_min=("rmssd", "min"),
            hrv_count=("rmssd", "count"),
        )
        .reset_index()
    )
    hrv_daily["date"] = pd.to_datetime(hrv_daily["date"])
    daily = daily.merge(hrv_daily, on="date", how="left")

    # --- Sleep HRV and HR (from sleep periods, take primary) ---
    # For days with multiple sleep periods, take the longest
    sleep_sorted = data["sleep"].sort_values(
        ["date", "total_sleep_duration"], ascending=[True, False]
    )
    sleep_primary = sleep_sorted.drop_duplicates(subset="date", keep="first")
    sleep_cols = sleep_primary[
        [
            "date",
            "average_hrv",
            "average_heart_rate",
            "average_breath",
            "total_sleep_duration",
            "rem_sleep_duration",
            "deep_sleep_duration",
            "light_sleep_duration",
            "awake_time",
            "efficiency",
            "lowest_heart_rate",
            "restless_periods",
            "period_id",
        ]
    ].copy()
    sleep_cols = sleep_cols.rename(
        columns={
            "average_hrv": "sleep_hrv",
            "average_heart_rate": "sleep_hr",
            "average_breath": "sleep_breath",
            "total_sleep_duration": "sleep_total_sec",
            "rem_sleep_duration": "rem_sec",
            "deep_sleep_duration": "deep_sec",
            "light_sleep_duration": "light_sec",
            "awake_time": "awake_sec",
            "efficiency": "sleep_eff",
            "lowest_heart_rate": "lowest_hr",
            "restless_periods": "restless",
        }
    )
    daily = daily.merge(sleep_cols, on="date", how="left")

    # --- Sleep architecture ratios ---
    total = daily["sleep_total_sec"].replace(0, np.nan)
    daily["rem_pct"] = daily["rem_sec"] / total * 100
    daily["deep_pct"] = daily["deep_sec"] / total * 100
    daily["light_pct"] = daily["light_sec"] / total * 100
    daily["awake_pct"] = daily["awake_sec"] / (total + daily["awake_sec"]) * 100
    daily["sleep_hours"] = total / 3600

    # --- Sleep fragmentation index ---
    # From epochs: count phase transitions per hour of sleep
    epoch_frag = compute_epoch_fragmentation(data["epochs"], data["sleep"])
    daily = daily.merge(epoch_frag, on="date", how="left")

    # --- SpO2 ---
    spo2 = data["spo2"][["date", "spo2_average", "breathing_disturbance_index"]].copy()
    daily = daily.merge(spo2, on="date", how="left")

    # --- Activity ---
    act = data["activity"][["date", "score", "steps"]].copy()
    act = act.rename(columns={"score": "activity_score", "steps": "steps"})
    daily = daily.merge(act, on="date", how="left")

    # --- Stress ---
    stress = data["stress"][["date", "stress_high", "recovery_high"]].copy()
    daily = daily.merge(stress, on="date", how="left")

    # --- Daily resting HR (from heart_rate: take 10th percentile as resting) ---
    hr_daily = (
        data["heart_rate"]
        .groupby("date")
        .agg(
            hr_p10=("bpm", lambda x: np.nanpercentile(x, 10)),
            hr_median=("bpm", "median"),
            hr_mean=("bpm", "mean"),
        )
        .reset_index()
    )
    hr_daily["date"] = pd.to_datetime(hr_daily["date"])
    daily = daily.merge(hr_daily, on="date", how="left")

    # --- Derived: night-to-night temperature gradient ---
    daily["temp_gradient"] = daily["temp_dev"].diff()

    # --- Derived: rolling 7-day temperature SD (variability) ---
    daily["temp_var_7d"] = daily["temp_dev"].rolling(7, min_periods=3).std()

    # --- Derived: rolling 7-day HRV mean ---
    daily["hrv_7d"] = daily["hrv_median"].rolling(7, min_periods=3).mean()

    # --- Derived: rolling 7-day HR ---
    daily["hr_7d"] = daily["sleep_hr"].rolling(7, min_periods=3).mean()

    daily = daily.set_index("date")
    log(
        "FEATURES",
        f"Daily feature matrix: {daily.shape[0]} days x {daily.shape[1]} features",
    )
    log("FEATURES", f"  Missing data: {daily.isnull().sum().sum()} total NaN cells")
    return daily


def compute_epoch_fragmentation(
    epochs: pd.DataFrame, sleep: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute sleep fragmentation index from epoch phase transitions.
    Epoch phases: 1=deep, 2=light, 3=REM, 4=awake.
    Fragmentation = number of phase transitions / hours of sleep.
    """
    # Map period_id to date
    pid_date = sleep[["period_id", "date"]].drop_duplicates()

    frag_rows = []
    for pid, group in epochs.groupby("period_id"):
        phases = group.sort_values("epoch_index")["phase"].values
        transitions = np.sum(np.diff(phases) != 0) if len(phases) > 1 else 0
        # Epochs are 5-minute intervals
        hours = len(phases) * 5 / 60
        frag_idx = transitions / hours if hours > 0 else np.nan
        # Count awake intrusions (phase 4)
        awake_epochs = np.sum(phases == 4)
        awake_pct_epoch = awake_epochs / len(phases) * 100 if len(phases) > 0 else 0
        frag_rows.append(
            {
                "period_id": pid,
                "frag_index": frag_idx,
                "transitions": transitions,
                "awake_epochs_pct": awake_pct_epoch,
            }
        )

    frag_df = pd.DataFrame(frag_rows)
    frag_df = frag_df.merge(pid_date, on="period_id", how="left")

    # Merge sleep duration so we can pick the longest period (matching build_daily_features)
    pid_dur = sleep[["period_id", "total_sleep_duration"]].drop_duplicates()
    frag_df = frag_df.merge(pid_dur, on="period_id", how="left")

    # Take primary period per day (longest sleep duration, matching build_daily_features)
    frag_daily = frag_df.sort_values(
        ["date", "total_sleep_duration"], ascending=[True, False]
    ).drop_duplicates(subset="date", keep="first")[
        ["date", "frag_index", "transitions", "awake_epochs_pct"]
    ]
    return frag_daily


# ===========================================================================
# Section 3: Temperature Fluctuation Analysis
# ===========================================================================
def analyze_temperature(daily: pd.DataFrame) -> dict[str, Any]:
    """
    Detailed temperature deviation analysis.
    GVHD causes micro-inflammatory temperature signatures visible
    as increased variability and regime changes.
    """
    log("TEMP", "Analyzing temperature fluctuation patterns...")
    temp = daily["temp_dev"].dropna()

    result: dict[str, Any] = {}

    # Basic statistics
    result["stats"] = {
        "mean": float(temp.mean()),
        "std": float(temp.std()),
        "min": float(temp.min()),
        "max": float(temp.max()),
        "median": float(temp.median()),
        "iqr": float(temp.quantile(0.75) - temp.quantile(0.25)),
    }
    log("TEMP", f"  Mean temp deviation: {result['stats']['mean']:.3f} °C")
    log("TEMP", f"  Temperature SD: {result['stats']['std']:.3f} °C")

    # Rolling variability (7-day SD)
    var_7d = daily["temp_var_7d"].dropna()
    result["variability_7d"] = {
        "mean": float(var_7d.mean()),
        "max": float(var_7d.max()),
        "max_date": str(var_7d.idxmax().date()) if not var_7d.empty else None,
    }
    log(
        "TEMP",
        f"  Peak 7d variability: {result['variability_7d']['max']:.3f} °C on {result['variability_7d']['max_date']}",
    )

    # Autocorrelation structure (lag 1-7)
    acf_values = []
    for lag in range(1, 8):
        if len(temp) > lag:
            acf = temp.autocorr(lag=lag)
            acf_values.append(float(acf) if not np.isnan(acf) else 0.0)
        else:
            acf_values.append(0.0)
    result["autocorrelation"] = {f"lag_{i + 1}": v for i, v in enumerate(acf_values)}
    log("TEMP", f"  Lag-1 autocorrelation: {acf_values[0]:.3f}")

    # Night-to-night gradient analysis
    grad = daily["temp_gradient"].dropna()
    result["gradient"] = {
        "mean_abs": float(grad.abs().mean()),
        "max_positive": float(grad.max()),
        "max_positive_date": str(grad.idxmax().date()) if not grad.empty else None,
        "max_negative": float(grad.min()),
    }

    # Regime change detection using CUSUM on temperature deviation
    temp_vals = temp.values
    mu = np.mean(temp_vals[:14])  # Baseline from first 2 weeks
    sigma = np.std(temp_vals[:14]) if np.std(temp_vals[:14]) > 0.01 else 0.1
    k = 0.5 * sigma
    h = 4.0 * sigma

    cusum_pos = np.zeros(len(temp_vals))
    cusum_neg = np.zeros(len(temp_vals))
    regime_changes = []
    for i in range(1, len(temp_vals)):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + temp_vals[i] - mu - k)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - temp_vals[i] + mu - k)
        reset_logged = False
        if cusum_pos[i] > h:
            regime_changes.append(str(temp.index[i].date()))
            reset_logged = True
            cusum_pos[i] = 0
        if cusum_neg[i] > h:
            if not reset_logged:
                regime_changes.append(str(temp.index[i].date()))
            cusum_neg[i] = 0

    result["regime_changes"] = regime_changes
    result["cusum"] = {
        "positive": cusum_pos.tolist(),
        "negative": cusum_neg.tolist(),
        "dates": [str(d.date()) for d in temp.index],
    }
    log("TEMP", f"  Regime changes detected: {len(regime_changes)} at {regime_changes}")

    # Pre-event temperature pattern (7 days before Feb 9: days -6 through 0 inclusive)
    event_date = pd.Timestamp(KNOWN_EVENT_DATE)
    pre_window = daily.loc[
        (daily.index >= event_date - timedelta(days=6)) & (daily.index <= event_date),
        "temp_dev",
    ]
    if not pre_window.empty:
        result["pre_event_pattern"] = {
            "mean": float(pre_window.mean()),
            "trajectory": [
                {"date": str(d.date()), "value": float(v)}
                for d, v in pre_window.items()
                if not np.isnan(v)
            ],
        }
        log(
            "TEMP",
            f"  Pre-event (7d) mean temp: {result['pre_event_pattern']['mean']:.3f} °C",
        )

    # Build temperature figure
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Temperature Deviation (Daily)",
            "7-Day Temperature Variability (Rolling SD)",
            "Night-to-Night Temperature Gradient",
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3],
    )

    # Panel 1: Raw temperature deviation with gradient fill
    # Upper fill (positive deviations - fever territory)
    temp_dev_vals = daily["temp_dev"].copy()
    temp_pos = temp_dev_vals.clip(lower=0)
    temp_neg = temp_dev_vals.clip(upper=0)

    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=temp_pos,
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.08)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=temp_neg,
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.08)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Highlight fever episodes (temp_dev > 0.5 °C) with background shading
    fever_mask = daily["temp_dev"] > 0.5
    if fever_mask.any():
        fever_starts = []
        in_fever = False
        for i, (idx, is_fever) in enumerate(fever_mask.items()):
            if is_fever and not in_fever:
                fever_starts.append(idx)
                in_fever = True
            elif not is_fever and in_fever:
                fever_starts.append(idx)
                in_fever = False
        if in_fever:
            fever_starts.append(daily.index[-1])
        for j in range(0, len(fever_starts) - 1, 2):
            fig.add_vrect(
                x0=fever_starts[j],
                x1=fever_starts[j + 1],
                fillcolor="rgba(239, 68, 68, 0.06)",
                line=dict(width=0),
                row=1,
                col=1,
            )

    # Main temperature line
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["temp_dev"],
            mode="lines+markers",
            marker=dict(size=4, color=ACCENT_BLUE, line=dict(width=0)),
            line=dict(width=1.8, color=ACCENT_BLUE),
            name="Temp Deviation",
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>Deviation: %{y:+.2f} °C<br><extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    # Zero baseline
    fig.add_hline(
        y=0, line_dash="dot", line_color=TEXT_SECONDARY, opacity=0.4, row=1, col=1
    )
    # Clinical thresholds with refined dash patterns
    fig.add_hline(
        y=0.5,
        line_dash="dashdot",
        line_color="rgba(239, 68, 68, 0.35)",
        line_width=1,
        row=1,
        col=1,
        annotation_text="Fever threshold (+0.5 °C)",
        annotation_position="bottom right",
        annotation_font=dict(size=9, color=ACCENT_RED),
    )
    fig.add_hline(
        y=-0.5,
        line_dash="dashdot",
        line_color="rgba(59, 130, 246, 0.35)",
        line_width=1,
        row=1,
        col=1,
        annotation_text="Hypothermia (-0.5 °C)",
        annotation_position="top right",
        annotation_font=dict(size=9, color=ACCENT_BLUE),
    )
    _add_event_markers(fig, row=1, show_labels=True)

    # Panel 2: Rolling variability with refined gradient fill
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["temp_var_7d"],
            mode="lines",
            fill="tozeroy",
            line=dict(color=ACCENT_AMBER, width=2),
            fillcolor="rgba(245, 158, 11, 0.12)",
            name="7d Variability",
            hovertemplate=(
                "<b>%{x|%b %d}</b><br>7-day SD: %{y:.3f} °C<br><extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    _add_event_markers(fig, row=2, show_labels=False)

    # Panel 3: Gradient with color-coded severity
    grad_vals = daily["temp_gradient"].fillna(0)
    grad_colors = []
    for v in grad_vals:
        av = abs(v)
        if av > 0.5:
            grad_colors.append(ACCENT_RED)
        elif av > 0.3:
            grad_colors.append(ACCENT_AMBER)
        else:
            grad_colors.append("rgba(59, 130, 246, 0.6)")

    fig.add_trace(
        go.Bar(
            x=daily.index,
            y=daily["temp_gradient"],
            marker_color=grad_colors,
            marker_line=dict(width=0),
            name="Nightly Gradient",
            hovertemplate=(
                "<b>%{x|%b %d}</b><br>Gradient: %{y:+.3f} °C/night<br><extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )
    _add_event_markers(fig, row=3, show_labels=False)

    fig.update_layout(
        height=900,
        showlegend=False,
        margin=dict(t=136),
        hovermode="x unified",
    )
    # Subtle gridlines and crosshair spikes for all axes
    for row_i in range(1, 4):
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikesnap="cursor",
            spikedash="dot",
            row=row_i,
            col=1,
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)",
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikesnap="cursor",
            spikedash="dot",
            row=row_i,
            col=1,
        )
    fig.update_yaxes(title_text="Temperature deviation (°C)", row=1, col=1)
    fig.update_yaxes(title_text="SD (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Δ°C/night", row=3, col=1)

    figures.append(fig)
    log("TEMP", "Temperature analysis complete.")
    return result


def _add_event_markers(fig: go.Figure, row: int, show_labels: bool = False) -> None:
    """Add known clinical event markers to a subplot with refined styling."""
    events = [
        (KNOWN_EVENT_DATE, "Acute Event", ACCENT_RED, "dot"),
        (RUXOLITINIB_START, "Ruxolitinib", ACCENT_GREEN, "dashdot"),
        (HEV_DIAGNOSIS, "HEV Dx", ACCENT_AMBER, "longdashdot"),
    ]
    label_y = [0.06, 0.14, 0.22]
    for i, (date_str, label, color, dash_style) in enumerate(events):
        fig.add_shape(
            type="line",
            x0=date_str,
            x1=date_str,
            y0=0,
            y1=1,
            yref=f"y{row} domain" if row > 1 else "y domain",
            line=dict(color=color, dash=dash_style, width=1.5),
            opacity=0.7,
            row=row,
            col=1,
        )
        if show_labels:
            fig.add_annotation(
                x=date_str,
                y=label_y[i] if i < len(label_y) else 0.06,
                yref=f"y{row} domain" if row > 1 else "y domain",
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                xshift=6,
                font=dict(size=8, color=color, family="Inter, sans-serif"),
                bgcolor="rgba(15, 17, 23, 0.75)",
                borderpad=2,
                row=row,
                col=1,
            )


# ===========================================================================
# Section 4: Multi-Stream GVHD Composite Score
# ===========================================================================
def compute_gvhd_composite(daily: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """
    Compute daily GVHD composite score from 6 biometric streams.
    Each component normalized to 0-100 where higher = more GVHD-like.
    """
    log("COMPOSITE", "Computing multi-stream GVHD composite score...")

    # Compute baselines from first 14 days of valid data
    baseline_end = daily.index.min() + timedelta(days=14)
    baseline = daily.loc[daily.index <= baseline_end]

    components = pd.DataFrame(index=daily.index)
    component_stats: dict[str, Any] = {}

    # 1. Temperature: higher absolute deviation + variability = worse
    # Normalize: percentile rank within series, amplified by abs deviation
    temp_abs = daily["temp_dev"].abs()
    temp_bl_mean = baseline["temp_dev"].abs().mean()
    temp_bl_std = baseline["temp_dev"].abs().std()
    if temp_bl_std < 0.01:
        temp_bl_std = 0.1
    components["temperature"] = np.clip(
        (temp_abs - temp_bl_mean) / temp_bl_std * 25 + 50, 0, 100
    )
    component_stats["temperature"] = {
        "baseline_mean": float(temp_bl_mean),
        "baseline_std": float(temp_bl_std),
        "weight": WEIGHTS["temperature"],
    }

    # 2. SpO2: lower = worse (pulmonary GVHD / BOS)
    spo2_bl = baseline["spo2_average"].dropna()
    spo2_bl_mean = spo2_bl.mean() if not spo2_bl.empty else 96.5
    spo2_bl_std = spo2_bl.std() if not spo2_bl.empty and spo2_bl.std() > 0 else 0.5
    # Invert: lower SpO2 -> higher score
    spo2_z = -(daily["spo2_average"] - spo2_bl_mean) / spo2_bl_std
    components["spo2"] = np.clip(spo2_z * 25 + 50, 0, 100)
    # Fill missing SpO2 days with neutral 50
    components["spo2"] = components["spo2"].fillna(50)
    component_stats["spo2"] = {
        "baseline_mean": float(spo2_bl_mean),
        "available_days": int(daily["spo2_average"].notna().sum()),
        "weight": WEIGHTS["spo2"],
    }

    # 3. HRV: lower = worse (autonomic dysfunction)
    hrv_bl = baseline["hrv_median"].dropna()
    hrv_bl_mean = hrv_bl.mean() if not hrv_bl.empty else 10
    hrv_bl_std = hrv_bl.std() if not hrv_bl.empty and hrv_bl.std() > 0 else 3
    # Invert: lower HRV -> higher score
    hrv_z = -(daily["hrv_median"] - hrv_bl_mean) / hrv_bl_std
    components["hrv"] = np.clip(hrv_z * 25 + 50, 0, 100)
    component_stats["hrv"] = {
        "baseline_mean": float(hrv_bl_mean),
        "baseline_std": float(hrv_bl_std),
        "weight": WEIGHTS["hrv"],
    }

    # 4. Sleep fragmentation: higher = worse
    frag_bl = baseline["frag_index"].dropna()
    frag_bl_mean = frag_bl.mean() if not frag_bl.empty else 5
    frag_bl_std = frag_bl.std() if not frag_bl.empty and frag_bl.std() > 0 else 1
    frag_z = (daily["frag_index"] - frag_bl_mean) / frag_bl_std
    components["sleep_frag"] = np.clip(frag_z * 25 + 50, 0, 100)
    component_stats["sleep_frag"] = {
        "baseline_mean": float(frag_bl_mean),
        "weight": WEIGHTS["sleep_frag"],
    }

    # 5. Resting HR: higher = worse (cardiac stress, inflammation)
    hr_bl = baseline["sleep_hr"].dropna()
    hr_bl_mean = hr_bl.mean() if not hr_bl.empty else 90
    hr_bl_std = hr_bl.std() if not hr_bl.empty and hr_bl.std() > 0 else 5
    hr_z = (daily["sleep_hr"] - hr_bl_mean) / hr_bl_std
    components["resting_hr"] = np.clip(hr_z * 25 + 50, 0, 100)
    component_stats["resting_hr"] = {
        "baseline_mean": float(hr_bl_mean),
        "weight": WEIGHTS["resting_hr"],
    }

    # 6. Activity score: lower = worse (functional decline)
    act_bl = baseline["activity_score"].dropna()
    act_bl_mean = act_bl.mean() if not act_bl.empty else 55
    act_bl_std = act_bl.std() if not act_bl.empty and act_bl.std() > 0 else 10
    # Invert: lower activity -> higher score
    act_z = -(daily["activity_score"] - act_bl_mean) / act_bl_std
    components["activity"] = np.clip(act_z * 25 + 50, 0, 100)
    component_stats["activity"] = {
        "baseline_mean": float(act_bl_mean),
        "weight": WEIGHTS["activity"],
    }

    # --- Weighted composite ---
    composite = pd.Series(0.0, index=daily.index)
    for stream, weight in WEIGHTS.items():
        if stream in components.columns:
            composite += components[stream].fillna(50) * weight

    # Rolling 7-day smoothed composite
    composite_7d = composite.rolling(7, min_periods=3).mean()

    # Backtest: correlation with known events (7-day window: days -6 through 0 inclusive)
    event_date = pd.Timestamp(KNOWN_EVENT_DATE)
    pre_event_composite = composite.loc[
        (composite.index >= event_date - timedelta(days=6))
        & (composite.index <= event_date)
    ]
    post_event_composite = composite.loc[
        (composite.index > event_date)
        & (composite.index <= event_date + timedelta(days=7))
    ]

    result = {
        "component_stats": component_stats,
        "composite_stats": {
            "mean": float(composite.mean()),
            "std": float(composite.std()),
            "max": float(composite.max()),
            "max_date": str(composite.idxmax().date()),
            "min": float(composite.min()),
            "min_date": str(composite.idxmin().date()),
        },
        "pre_event_mean": float(pre_event_composite.mean())
        if not pre_event_composite.empty
        else None,
        "post_event_mean": float(post_event_composite.mean())
        if not post_event_composite.empty
        else None,
        "daily_scores": {
            str(d.date()): round(float(v), 1)
            for d, v in composite.items()
            if not np.isnan(v)
        },
    }

    log("COMPOSITE", f"  Composite mean: {result['composite_stats']['mean']:.1f}")
    log(
        "COMPOSITE",
        f"  Peak composite: {result['composite_stats']['max']:.1f} on {result['composite_stats']['max_date']}",
    )
    if result["pre_event_mean"] is not None:
        log("COMPOSITE", f"  Pre-event (7d) mean: {result['pre_event_mean']:.1f}")
    if result["post_event_mean"] is not None:
        log("COMPOSITE", f"  Post-event (7d) mean: {result['post_event_mean']:.1f}")

    # --- Figure: Composite score timeline ---
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "GVHD Composite Score (Daily + 7-Day Rolling)",
            "Component Breakdown (Stacked Weighted Contribution)",
        ),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    # Panel 1: Composite - daily markers with severity coloring
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=composite,
            mode="markers",
            marker=dict(
                size=5,
                color=composite,
                colorscale=[
                    [0, "rgba(16, 185, 129, 0.7)"],
                    [0.4, "rgba(245, 158, 11, 0.7)"],
                    [0.7, "rgba(239, 68, 68, 0.7)"],
                    [1, "rgba(239, 68, 68, 1.0)"],
                ],
                cmin=30,
                cmax=80,
                colorbar=dict(
                    title=dict(text="Score", font=dict(size=10)),
                    x=1.02,
                    thickness=12,
                    len=0.4,
                    tickfont=dict(size=9),
                    outlinewidth=0,
                ),
                line=dict(width=0),
            ),
            name="Daily Score",
            hovertemplate=(
                "<b>%{x|%b %d}</b><br>Composite: %{y:.1f}/100<br><extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    # Bold 7-day rolling average as the hero line
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=composite_7d,
            mode="lines",
            line=dict(color="#FFFFFF", width=3),
            name="7-Day Rolling Avg",
            hovertemplate=("<b>%{x|%b %d}</b><br>7d Avg: %{y:.1f}<br><extra></extra>"),
        ),
        row=1,
        col=1,
    )
    # Alert threshold with risk zone shading
    fig.add_hline(
        y=65,
        line_dash="dash",
        line_color=ACCENT_RED,
        opacity=0.4,
        line_width=1,
        row=1,
        col=1,
        annotation_text="Alert (65)",
        annotation_position="bottom right",
        annotation_font=dict(size=9, color=ACCENT_RED),
    )
    fig.add_hrect(
        y0=65, y1=100, fillcolor="rgba(239, 68, 68, 0.04)", line_width=0, row=1, col=1
    )
    _add_event_markers(fig, row=1, show_labels=True)

    # Panel 2: Stacked component breakdown with clean transitions
    stream_config = [
        ("temperature", ACCENT_RED, "Temperature (25%)"),
        ("hrv", ACCENT_PURPLE, "HRV (20%)"),
        ("resting_hr", ACCENT_AMBER, "Resting HR (15%)"),
        ("sleep_frag", ACCENT_BLUE, "Sleep Frag (15%)"),
        ("spo2", ACCENT_GREEN, "SpO2 (15%)"),
        ("activity", ACCENT_CYAN, "Activity (10%)"),
    ]
    for stream, color, label in stream_config:
        if stream in components.columns:
            # Convert hex to rgba for fill
            h = color.lstrip("#")
            r_c, g_c, b_c = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            fill_rgba = f"rgba({r_c},{g_c},{b_c},0.5)"

            fig.add_trace(
                go.Scatter(
                    x=daily.index,
                    y=components[stream] * WEIGHTS[stream],
                    mode="lines",
                    name=label,
                    line=dict(width=0.5, color=color),
                    fillcolor=fill_rgba,
                    stackgroup="one",
                    hovertemplate=(
                        f"<b>{label}</b><br>%{{x|%b %d}}: %{{y:.1f}}<br><extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        height=800,
        margin=dict(t=124),
        hovermode="x unified",
    )
    # Subtle gridlines and crosshair spikes
    for row_i in range(1, 3):
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikesnap="cursor",
            spikedash="dot",
            row=row_i,
            col=1,
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)",
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikesnap="cursor",
            spikedash="dot",
            row=row_i,
            col=1,
        )
    fig.update_yaxes(title_text="Composite Score (0-100)", row=1, col=1)
    fig.update_yaxes(title_text="Weighted Contribution", row=2, col=1)

    figures.append(fig)
    log("COMPOSITE", "Composite score analysis complete.")
    return composite, result


# ===========================================================================
# Section 5: Recurrent Switching Linear Dynamical System (rSLDS)
# ===========================================================================
#
# The rSLDS (Linderman et al. 2017) extends the HMM in two critical ways:
#   1. Within each discrete state, the latent dynamics follow a LINEAR
#      dynamical system (x_{t+1} = A_k * x_t + noise), capturing smooth
#      physiological trajectories within each GVHD regime.
#   2. Transitions between discrete states are RECURRENT: they depend on
#      the continuous latent state x_t, not just the previous discrete state.
#      This means gradual physiological deterioration (captured in x_t)
#      can trigger a regime change, modeling how pre-flare buildup leads
#      to acute events.
#
# Implementation uses the ssm library (Linderman Lab, Harvard/Stanford):
#   bash scripts/install_full_stack.sh
# Fallback: If ssm is unavailable, we use a Gaussian HMM via hmmlearn.
#
# Model specification:
#   K = 4 discrete states: Remission, Pre-flare, Active Flare, Recovery
#   D = 3 observation dimensions: composite score, temp deviation, HRV
#   N = D = 3 continuous latent dimensions (identity emission for rSLDS)
#   transitions = "recurrent" (x_t influences discrete state transitions)
#   dynamics = "gaussian" (linear dynamics within each state)
#   emissions = "gaussian_id" (latent = observed, identity mapping)
# ===========================================================================


_RSLDS_MAX_ITERS = 200  # Hard ceiling: stop early if Laplace-EM exceeds this
_EMISSION_VAR_FLOOR = 0.01  # Minimum emission variance to prevent singular covariance


def _fit_rslds(
    X: np.ndarray,
    n_states: int = 4,
    n_iters: int = 50,
    n_restarts: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[float], dict[str, Any]]:
    """
    Fit a recurrent Switching Linear Dynamical System using the ssm library.

    Parameters
    ----------
    X : (T, D) standardized observation matrix
    n_states : number of discrete states (K)
    n_iters : Laplace-EM iterations per restart (capped at _RSLDS_MAX_ITERS)
    n_restarts : number of random restarts (keep best ELBO)

    Returns
    -------
    state_probs : (T, K) posterior discrete state probabilities
    viterbi_path : (T,) most likely discrete state sequence
    elbos : list of ELBO values from the best run
    info : dict with model diagnostics (dynamics matrices, etc.)
    """
    import ssm

    # Enforce maximum iteration ceiling
    effective_iters = min(n_iters, _RSLDS_MAX_ITERS)
    if n_iters > _RSLDS_MAX_ITERS:
        log(
            "rSLDS",
            f"  Capping n_iters from {n_iters} to {_RSLDS_MAX_ITERS} (timeout guard)",
        )

    T, D = X.shape
    N = D  # Identity emission: latent dim = observation dim

    best_elbo = -np.inf
    best_result = None
    failed_restarts = 0

    for restart in range(n_restarts):
        seed = RSLDS_BASE_SEED + restart
        # ssm initialization is stochastic; pin each restart to a known seed so
        # repeated QA runs on unchanged data do not drift between clinical states.
        np.random.seed(seed)
        random.seed(seed)
        try:
            model = ssm.SLDS(
                N=N,
                K=n_states,
                D=D,
                M=0,
                transitions="recurrent",
                dynamics="gaussian",
                emissions="gaussian_id",
            )

            # Fit via Laplace-EM (variational inference for continuous states,
            # EM for discrete states and model parameters)
            elbos_arr, posterior = model.fit(
                X,
                method="laplace_em",
                num_iters=effective_iters,
                initialize=True,
            )

            final_elbo = float(elbos_arr[-1])
            log(
                "rSLDS",
                f"  Restart {restart + 1}/{n_restarts} (seed={seed}): final ELBO = {final_elbo:.1f}",
            )

            # --- ELBO convergence validation ---
            # Discard restarts where ELBO diverged to NaN or -inf
            if not np.isfinite(final_elbo):
                log(
                    "rSLDS",
                    f"  Restart {restart + 1}/{n_restarts}: ELBO non-finite ({final_elbo}), discarding",
                )
                failed_restarts += 1
                continue

            if final_elbo > best_elbo:
                best_elbo = final_elbo
                best_result = (model, elbos_arr, posterior)

        except Exception as e:
            log(
                "rSLDS",
                f"  Restart {restart + 1}/{n_restarts} (seed={seed}): failed ({e})",
            )
            failed_restarts += 1
            continue

    if best_result is None:
        raise RuntimeError(
            f"All {n_restarts} rSLDS restarts failed ({failed_restarts} failures). "
            "Falling back to HMM."
        )

    model, elbos_arr, posterior = best_result

    # --- Emission variance floor ---
    # Prevent near-singular emission covariance from producing garbage posteriors
    try:
        if hasattr(model.emissions, "Sigmas"):
            sigmas = np.array(model.emissions.Sigmas)
            for k in range(n_states):
                diag = np.diag(sigmas[k])
                if np.any(diag < _EMISSION_VAR_FLOOR):
                    log(
                        "rSLDS",
                        f"  State {k}: emission variance below floor, clamping to {_EMISSION_VAR_FLOOR}",
                    )
                    np.fill_diagonal(sigmas[k], np.maximum(diag, _EMISSION_VAR_FLOOR))
            model.emissions.Sigmas = sigmas
    except Exception as e:
        log("rSLDS", f"  Emission variance floor check skipped: {e}")

    # Extract discrete state probabilities from posterior
    state_probs = posterior.mean_discrete_states[0]  # (T, K)

    # Most likely discrete state sequence (Viterbi on the fitted model)
    x_mean = posterior.mean_continuous_states[0]  # (T, N)
    viterbi_path = model.most_likely_states(x_mean, X)  # (T,)

    # Extract dynamics matrices A_k for each state
    dynamics_info = {}
    if hasattr(model.dynamics, "As"):
        As = np.array(model.dynamics.As)  # (K, N, N)
        dynamics_info["dynamics_matrices"] = As.tolist()
        # Compute spectral radius of each A_k (stability indicator)
        for k in range(n_states):
            eigvals = np.linalg.eigvals(As[k])
            dynamics_info[f"state_{k}_spectral_radius"] = float(np.max(np.abs(eigvals)))

    elbos = [float(e) for e in elbos_arr]

    info = {
        "method": "rSLDS (Linderman ssm)",
        "n_restarts": n_restarts,
        "failed_restarts": failed_restarts,
        "best_final_elbo": best_elbo,
        "n_iters_requested": n_iters,
        "n_iters_effective": effective_iters,
        "latent_dim": N,
        "continuous_state_mean": x_mean.tolist(),
        **dynamics_info,
    }

    return state_probs, viterbi_path, elbos, info


def _fit_hmm_fallback(
    X: np.ndarray,
    n_states: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[float], dict[str, Any]]:
    """
    Fallback: fit a Gaussian HMM using hmmlearn when ssm is unavailable.

    Returns the same tuple signature as _fit_rslds for drop-in replacement.

    Raises RuntimeError if hmmlearn is not installed.
    """
    if not HMMLEARN_AVAILABLE:
        raise RuntimeError(
            "hmmlearn is not installed. Install with: pip install hmmlearn. "
            "Neither ssm nor hmmlearn is available -- cannot fit any state-space model."
        )
    from hmmlearn.hmm import GaussianHMM as HMMLearnGaussianHMM

    log("HMM", "ssm not available, using hmmlearn GaussianHMM")

    model = HMMLearnGaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        tol=1e-5,
        random_state=42,
        init_params="stmc",
    )

    # Set clinically-informed initial transition matrix
    model.startprob_ = np.array([0.6, 0.2, 0.1, 0.1])
    model.transmat_ = np.array(
        [
            [0.80, 0.15, 0.02, 0.03],
            [0.10, 0.60, 0.25, 0.05],
            [0.02, 0.08, 0.70, 0.20],
            [0.30, 0.10, 0.05, 0.55],
        ]
    )
    model.init_params = "mc"  # Only learn means and covariances from data

    stderr_buffer = io.StringIO()
    with redirect_stderr(stderr_buffer):
        model.fit(X)
    for line in stderr_buffer.getvalue().splitlines():
        if not line.startswith("Model is not converging."):
            print(line, file=sys.stderr)

    state_probs = model.predict_proba(X)
    viterbi_path = model.predict(X)

    # Collect convergence info
    scores = []
    if hasattr(model, "monitor_") and hasattr(model.monitor_, "history"):
        scores = [float(s) for s in model.monitor_.history]
    elif hasattr(model, "score"):
        scores = [float(model.score(X))]

    info = {
        "method": "GaussianHMM (hmmlearn)",
        "converged": bool(model.monitor_.converged)
        if hasattr(model, "monitor_")
        else True,
        "n_iter": int(model.monitor_.iter) if hasattr(model, "monitor_") else 200,
        "transition_matrix": model.transmat_.tolist(),
    }

    return state_probs, viterbi_path, scores, info


_MIN_DATAPOINTS = 21  # Minimum observations for model fitting (3 weeks)
_NAN_SKIP_THRESHOLD = 0.30  # Skip rSLDS if >30% of feature values are NaN


def run_rslds_analysis(
    daily: pd.DataFrame, composite: pd.Series
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Fit a recurrent Switching Linear Dynamical System (rSLDS) and produce
    state probabilities. Falls back to Gaussian HMM if ssm is unavailable
    or if rSLDS fitting fails (divergence, singular matrix, etc.).

    Uses composite score + temperature deviation + HRV as observations.
    The rSLDS adds continuous linear dynamics within each discrete state
    and recurrent (state-dependent) transitions, which is a strict upgrade
    over the previous HMM implementation.
    """
    # --- Import guard: ensure at least one backend is available ---
    if not SSM_AVAILABLE and not HMMLEARN_AVAILABLE:
        log("MODEL", "Neither ssm nor hmmlearn is installed.")
        log("MODEL", "  Install dependencies:  bash scripts/install_full_stack.sh")
        return (
            np.array([]),
            np.array([]),
            {
                "error": "No model backend available",
                "install_instructions": {
                    "full_stack": "bash scripts/install_full_stack.sh",
                },
            },
        )

    tag = "rSLDS" if SSM_AVAILABLE else "HMM"
    log(
        tag,
        f"Fitting {N_STATES}-state {'rSLDS' if SSM_AVAILABLE else 'Gaussian HMM (hmmlearn)'}...",
    )

    # Build observation matrix: (T, 3)
    obs_df = pd.DataFrame(
        {
            "composite": composite,
            "temp_dev": daily["temp_dev"],
            "hrv": daily["hrv_median"],
        }
    )

    # --- Missing data handling ---
    total_cells = obs_df.size
    nan_cells = int(obs_df.isna().sum().sum())
    nan_fraction = nan_cells / total_cells if total_cells > 0 else 0.0
    log(tag, f"  Missing data: {nan_cells}/{total_cells} cells ({nan_fraction:.1%})")

    if nan_fraction > _NAN_SKIP_THRESHOLD:
        log(
            tag,
            f"  WARNING: NaN fraction ({nan_fraction:.1%}) exceeds {_NAN_SKIP_THRESHOLD:.0%} threshold",
        )
        log(tag, "  Skipping rSLDS entirely due to excessive missing data")
        if HMMLEARN_AVAILABLE and SSM_AVAILABLE:
            log(tag, "  Attempting HMM with available data instead")
            # Let it fall through with reduced data -- HMM is more tolerant
        else:
            return (
                np.array([]),
                np.array([]),
                {
                    "error": f"Excessive missing data ({nan_fraction:.1%} > {_NAN_SKIP_THRESHOLD:.0%})",
                    "nan_cells": nan_cells,
                    "total_cells": total_cells,
                },
            )

    # Forward-fill then back-fill NaN
    imputed_before = int(obs_df.isna().sum().sum())
    obs_df = obs_df.ffill().bfill()
    imputed_after = int(obs_df.isna().sum().sum())
    n_imputed = imputed_before - imputed_after
    if n_imputed > 0:
        log(tag, f"  Imputed {n_imputed} NaN values via forward-fill + backward-fill")

    # Drop remaining NaN rows (track valid indices)
    valid_mask = obs_df.notna().all(axis=1)
    obs_clean = obs_df[valid_mask].copy()

    # --- Insufficient data guard ---
    if len(obs_clean) < _MIN_DATAPOINTS:
        log(
            tag,
            f"  WARNING: Only {len(obs_clean)} valid observations, need >= {_MIN_DATAPOINTS}",
        )
        log(
            tag,
            f"  rSLDS with {N_STATES} states and continuous dynamics requires at least {_MIN_DATAPOINTS} data points (3 weeks)",
        )
        return (
            np.array([]),
            np.array([]),
            {
                "error": f"Insufficient data ({len(obs_clean)} < {_MIN_DATAPOINTS})",
                "available_days": len(obs_clean),
                "minimum_required": _MIN_DATAPOINTS,
            },
        )

    # Standardize features
    obs_means = np.array(obs_clean.mean().to_numpy(dtype=np.float64), copy=True)
    obs_stds = np.array(obs_clean.std().to_numpy(dtype=np.float64), copy=True)
    obs_stds[obs_stds < 1e-6] = 1.0
    X = ((obs_clean.to_numpy(dtype=np.float64) - obs_means) / obs_stds).astype(
        np.float64
    )

    log(tag, f"  Observations: {X.shape[0]} days x {X.shape[1]} features")
    log(
        tag,
        f"  Feature means: composite={obs_means[0]:.1f}, temp={obs_means[1]:.3f}, hrv={obs_means[2]:.1f}",
    )

    # --- Fit model with fallback chain ---
    used_fallback = False

    if SSM_AVAILABLE and nan_fraction <= _NAN_SKIP_THRESHOLD:
        try:
            state_probs, viterbi_path, elbos, model_info = _fit_rslds(
                X,
                n_states=N_STATES,
                n_iters=50,
                n_restarts=3,
            )
        except RuntimeError as e:
            log("rSLDS", f"  rSLDS fitting failed: {e}")
            if HMMLEARN_AVAILABLE:
                log("rSLDS", "  Switching to HMM...")
                state_probs, viterbi_path, elbos, model_info = _fit_hmm_fallback(
                    X,
                    n_states=N_STATES,
                )
                model_info["rslds_failure_reason"] = str(e)
                used_fallback = True
                tag = "HMM"
            else:
                return (
                    np.array([]),
                    np.array([]),
                    {"error": f"rSLDS failed and hmmlearn not available: {e}"},
                )
        except Exception as e:
            log("rSLDS", f"  rSLDS unexpected error: {e}")
            if HMMLEARN_AVAILABLE:
                log("rSLDS", "  Switching to HMM...")
                state_probs, viterbi_path, elbos, model_info = _fit_hmm_fallback(
                    X,
                    n_states=N_STATES,
                )
                model_info["rslds_failure_reason"] = str(e)
                used_fallback = True
                tag = "HMM"
            else:
                return (
                    np.array([]),
                    np.array([]),
                    {"error": f"rSLDS failed and hmmlearn not available: {e}"},
                )
    else:
        state_probs, viterbi_path, elbos, model_info = _fit_hmm_fallback(
            X,
            n_states=N_STATES,
        )
        if SSM_AVAILABLE and nan_fraction > _NAN_SKIP_THRESHOLD:
            model_info["rslds_skipped_reason"] = (
                f"NaN fraction {nan_fraction:.1%} > {_NAN_SKIP_THRESHOLD:.0%}"
            )
        used_fallback = not SSM_AVAILABLE or nan_fraction > _NAN_SKIP_THRESHOLD

    # --- State assignment validation ---
    unique_states = np.unique(viterbi_path)
    if len(unique_states) == 1:
        log(
            tag,
            f"  WARNING: Degenerate solution -- all {len(viterbi_path)} observations assigned to state {unique_states[0]}",
        )
        if SSM_AVAILABLE and not used_fallback and HMMLEARN_AVAILABLE:
            log(tag, "  Re-trying with HMM to avoid degenerate solution...")
            state_probs, viterbi_path, elbos, model_info = _fit_hmm_fallback(
                X,
                n_states=N_STATES,
            )
            model_info["rslds_degenerate"] = True
            used_fallback = True
            tag = "HMM"
            unique_states = np.unique(viterbi_path)
        if len(unique_states) == 1:
            log(
                tag,
                "  WARNING: HMM also produced degenerate solution. Results may be unreliable.",
            )
            model_info["degenerate_warning"] = (
                "All observations assigned to a single state"
            )

    # Validate state indices are in valid range
    if np.any(viterbi_path < 0) or np.any(viterbi_path >= N_STATES):
        bad_indices = np.where((viterbi_path < 0) | (viterbi_path >= N_STATES))[0]
        log(
            tag,
            f"  WARNING: {len(bad_indices)} state indices out of range [0, {N_STATES - 1}], clamping",
        )
        viterbi_path = np.clip(viterbi_path, 0, N_STATES - 1)

    # Re-index to full date range (fill NaN days with uniform)
    full_probs = np.full((len(daily), N_STATES), 1.0 / N_STATES)
    full_viterbi = np.full(len(daily), -1, dtype=int)
    valid_indices = np.where(valid_mask.values)[0]
    full_probs[valid_indices] = state_probs
    full_viterbi[valid_indices] = viterbi_path

    # Ensure states are ordered by severity:
    # Sort states by mean composite score (state with highest mean = active flare)
    state_mean_composite = []
    for k in range(N_STATES):
        mask = viterbi_path == k
        if mask.any():
            state_mean_composite.append(obs_clean["composite"].values[mask].mean())
        else:
            state_mean_composite.append(0)

    # Re-map states: lowest composite mean = remission, highest = active flare
    severity_order = np.argsort(state_mean_composite)
    state_map = {severity_order[i]: i for i in range(N_STATES)}
    # 0=remission, 1=pre_flare, 2=active_flare, 3=recovery
    # Assign: lowest=remission, 2nd=recovery, 3rd=pre_flare, highest=active_flare
    if N_STATES == 4:
        assignment = [0, 3, 1, 2]  # remission, recovery, pre-flare, active-flare
        state_map = {severity_order[i]: assignment[i] for i in range(N_STATES)}

    remapped_probs = np.zeros_like(full_probs)
    remapped_viterbi = np.full(len(full_viterbi), -1, dtype=int)
    for old, new in state_map.items():
        remapped_probs[:, new] = full_probs[:, old]
        remapped_viterbi[full_viterbi == old] = new

    # Build result metrics
    model_label = "HMM" if used_fallback else ("rSLDS" if SSM_AVAILABLE else "HMM")
    result: dict[str, Any] = {
        "model_type": model_label,
        "convergence": {
            "iterations": len(elbos),
            "final_elbo": float(elbos[-1]) if elbos else None,
        },
        "model_info": model_info,
        "state_distribution": {},
        "elbo_history": elbos,
    }

    for k in range(N_STATES):
        days_in_state = np.sum(remapped_viterbi == k)
        result["state_distribution"][STATE_NAMES[k]] = {
            "days": int(days_in_state),
            "pct": float(safe_divide(days_in_state, len(remapped_viterbi)) * 100),
        }
    no_data_days = int(np.sum(remapped_viterbi == -1))
    if no_data_days > 0:
        result["state_distribution"]["No Data"] = {
            "days": no_data_days,
            "pct": float(safe_divide(no_data_days, len(remapped_viterbi)) * 100),
        }
    log(tag, f"  Convergence: {result['convergence']['iterations']} iterations")
    for name, info in result["state_distribution"].items():
        log(tag, f"  {name}: {info['days']} days ({info['pct']:.1f}%)")

    # Validate against known event
    event_idx = None
    event_date = pd.Timestamp(KNOWN_EVENT_DATE)
    for i, d in enumerate(daily.index):
        if d.date() == event_date.date():
            event_idx = i
            break
    if event_idx is not None:
        ev_state_idx = remapped_viterbi[event_idx]
        ev_state_name = STATE_NAMES[ev_state_idx] if ev_state_idx != -1 else "No Data"
        result["event_validation"] = {
            "date": KNOWN_EVENT_DATE,
            "viterbi_state": ev_state_name,
            "state_probs": {
                STATE_NAMES[k]: float(remapped_probs[event_idx, k])
                for k in range(N_STATES)
            },
        }
        log(tag, f"  Feb 9 state: {result['event_validation']['viterbi_state']}")
        log(tag, f"  Feb 9 probs: {result['event_validation']['state_probs']}")
    else:
        log(tag, "  WARNING: Could not locate Feb 9 event in daily index")

    # --- Figure: state probability heatmap ---
    model_short = "rSLDS" if not used_fallback else "HMM"
    convergence_label = (
        "ELBO Convergence (Laplace-EM)"
        if not used_fallback
        else "Log-Likelihood Convergence"
    )
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            f"{model_short} State Probabilities ({N_STATES} states x {len(daily)} days)",
            "Most Likely State (Viterbi Path)",
            convergence_label,
        ),
        vertical_spacing=0.10,
        row_heights=[0.45, 0.35, 0.20],
    )

    # Convert hex colors to rgba for fill
    def _hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
        """Convert #RRGGBB to rgba(r,g,b,a)."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # Panel 1: State probability bands with semantic risk colors
    for k in range(N_STATES):
        fig.add_trace(
            go.Scatter(
                x=daily.index,
                y=remapped_probs[:, k],
                mode="lines",
                name=STATE_NAMES[k],
                line=dict(color=STATE_COLORS[k], width=0.8),
                fill="tonexty" if k > 0 else "tozeroy",
                fillcolor=_hex_to_rgba(STATE_COLORS[k], 0.45),
                stackgroup="states",
                hovertemplate=(
                    f"<b>{STATE_NAMES[k]}</b><br>"
                    "%{x|%b %d}: %{y:.1%}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    _add_event_markers(fig, row=1, show_labels=True)

    # Panel 2: Viterbi path as colored bars with clean gaps
    for k in range(N_STATES):
        mask = remapped_viterbi == k
        if mask.sum() == 0:
            continue
        fig.add_trace(
            go.Bar(
                x=daily.index[mask],
                y=[1] * mask.sum(),
                marker_color=STATE_COLORS[k],
                marker_line=dict(width=0),
                name=f"Viterbi: {STATE_NAMES[k]}",
                showlegend=False,
                width=86400000,  # 1 day in ms
                hovertemplate=(
                    f"<b>{STATE_NAMES[k]}</b><br>%{{x|%b %d}}<br><extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    _add_event_markers(fig, row=2)

    # Panel 3: ELBO convergence with refined styling
    if elbos:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(elbos) + 1)),
                y=elbos,
                mode="lines+markers",
                marker=dict(size=3, color=ACCENT_BLUE, line=dict(width=0)),
                line=dict(color=ACCENT_BLUE, width=2),
                name="ELBO" if not used_fallback else "Log-Likelihood",
                hovertemplate=("Iteration %{x}<br>Value: %{y:.1f}<br><extra></extra>"),
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        height=950,
        barmode="stack",
        margin=dict(t=120),
        hovermode="x unified",
    )
    # Subtle gridlines and crosshair spikes
    for row_i in range(1, 4):
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikesnap="cursor",
            spikedash="dot",
            row=row_i,
            col=1,
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)",
            row=row_i,
            col=1,
        )
    fig.update_yaxes(title_text="P(state)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="State", showticklabels=False, row=2, col=1)
    fig.update_yaxes(
        title_text="ELBO" if not used_fallback else "Log-Likelihood", row=3, col=1
    )
    fig.update_xaxes(title_text="Iteration", row=3, col=1)

    figures.append(fig)
    log(tag, f"{'rSLDS' if SSM_AVAILABLE else 'HMM'} analysis complete.")
    return remapped_probs, remapped_viterbi, result


# ===========================================================================
# Section 6: Early Warning Alert System
# ===========================================================================
def evaluate_alerts(
    daily: pd.DataFrame,
    state_probs: np.ndarray,
    viterbi_path: np.ndarray,
    composite: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate early warning alerts based on rSLDS state probabilities.
    Retrospectively validate against known Feb 9 event.
    """
    log("ALERTS", "Evaluating early warning alert system...")

    if state_probs.size == 0:
        log("ALERTS", "  No rSLDS results available - skipping")
        return {"error": "No rSLDS data"}

    alerts: list[dict[str, Any]] = []
    dates = daily.index

    # Track consecutive pre-flare days for YELLOW alert
    consec_preflare = 0
    active_yellow = False

    for i in range(len(dates)):
        p_preflare = state_probs[i, 1]  # Pre-flare
        p_flare = state_probs[i, 2]  # Active flare
        date_str = str(dates[i].date())

        # RED alert: high pre-flare or any active flare probability
        if p_preflare > RED_PREFLARE_PROB or p_flare > RED_FLARE_PROB:
            alerts.append(
                {
                    "date": date_str,
                    "level": "RED",
                    "p_preflare": round(float(p_preflare), 3),
                    "p_flare": round(float(p_flare), 3),
                    "composite": round(float(composite.iloc[i]), 1)
                    if i < len(composite)
                    else None,
                }
            )
            consec_preflare = 0
            active_yellow = False
            continue

        # YELLOW alert: sustained pre-flare probability
        if p_preflare > YELLOW_PREFLARE_PROB:
            consec_preflare += 1
            if consec_preflare >= YELLOW_CONSEC_DAYS and not active_yellow:
                alerts.append(
                    {
                        "date": date_str,
                        "level": "YELLOW",
                        "p_preflare": round(float(p_preflare), 3),
                        "p_flare": round(float(p_flare), 3),
                        "consecutive_days": consec_preflare,
                        "composite": round(float(composite.iloc[i]), 1)
                        if i < len(composite)
                        else None,
                    }
                )
                active_yellow = True
        else:
            consec_preflare = 0
            active_yellow = False

    # --- Retrospective validation against Feb 9 ---
    event_date = pd.Timestamp(KNOWN_EVENT_DATE)
    event_idx = None
    for i, d in enumerate(dates):
        if d.date() == event_date.date():
            event_idx = i
            break

    validation: dict[str, Any] = {"event": KNOWN_EVENT_DATE}

    if event_idx is not None:
        # Find first RED alert before or on event date
        pre_event_reds = [
            a for a in alerts if a["level"] == "RED" and a["date"] <= KNOWN_EVENT_DATE
        ]
        pre_event_yellows = [
            a
            for a in alerts
            if a["level"] == "YELLOW" and a["date"] <= KNOWN_EVENT_DATE
        ]

        if pre_event_reds:
            first_red = pre_event_reds[0]
            lead_days = (event_date - pd.Timestamp(first_red["date"])).days
            validation["first_red_alert"] = first_red["date"]
            validation["red_lead_time_days"] = lead_days
            validation["detected"] = True
            log(
                "ALERTS",
                f"  First RED alert: {first_red['date']} ({lead_days}d before event)",
            )
        else:
            validation["detected"] = False
            log("ALERTS", "  No RED alert before Feb 9 event")

        if pre_event_yellows:
            first_yellow = pre_event_yellows[0]
            lead_days_y = (event_date - pd.Timestamp(first_yellow["date"])).days
            validation["first_yellow_alert"] = first_yellow["date"]
            validation["yellow_lead_time_days"] = lead_days_y
            log(
                "ALERTS",
                f"  First YELLOW alert: {first_yellow['date']} ({lead_days_y}d before event)",
            )

        # N=1 case study — descriptive detection statistics only.
        # Sensitivity/specificity require an external validation cohort
        # and cannot be computed from a single retrospective event.
        event_window = set()
        for delta in range(-3, 4):
            d = (event_date + timedelta(days=delta)).date()
            event_window.add(str(d))

        all_dates_set = {str(d.date()) for d in dates}
        red_dates = {a["date"] for a in alerts if a["level"] == "RED"}

        red_in_window = len(red_dates & event_window)
        red_outside_window = len(red_dates - event_window)

        validation["red_alerts_in_event_window"] = red_in_window
        validation["red_alerts_outside_event_window"] = red_outside_window
        validation["study_type"] = "N=1 retrospective case study"
        validation["validation_note"] = "External cohort validation required"
        log(
            "ALERTS",
            f"  N=1 case study: {red_in_window} RED alert(s) within ±3d of event, {red_outside_window} outside",
        )

    # Summary
    n_red = sum(1 for a in alerts if a["level"] == "RED")
    n_yellow = sum(1 for a in alerts if a["level"] == "YELLOW")
    log("ALERTS", f"  Total alerts: {n_red} RED, {n_yellow} YELLOW")

    result = {
        "alerts": alerts,
        "summary": {
            "n_red": n_red,
            "n_yellow": n_yellow,
            "thresholds": {
                "yellow_preflare_prob": YELLOW_PREFLARE_PROB,
                "yellow_consec_days": YELLOW_CONSEC_DAYS,
                "red_preflare_prob": RED_PREFLARE_PROB,
                "red_flare_prob": RED_FLARE_PROB,
            },
        },
        "validation": validation,
    }

    # --- Figure: Alert timeline ---
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Early Warning Alert Timeline",
            "Pre-flare Probability with Alert Thresholds",
        ),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.6],
    )

    # Panel 1: Composite score baseline with dramatic alert markers
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=composite,
            mode="lines",
            line=dict(color=TEXT_SECONDARY, width=1.2),
            name="Composite Score",
            hovertemplate=(
                "<b>%{x|%b %d}</b><br>Composite: %{y:.1f}<br><extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    # Separate RED and YELLOW alerts for legend clarity
    red_alerts = [a for a in alerts if a["level"] == "RED"]
    yellow_alerts = [a for a in alerts if a["level"] == "YELLOW"]

    if red_alerts:
        fig.add_trace(
            go.Scatter(
                x=[a["date"] for a in red_alerts],
                y=[a.get("composite", 50) for a in red_alerts],
                mode="markers",
                marker=dict(
                    size=14,
                    color=ACCENT_RED,
                    symbol="triangle-up",
                    line=dict(width=2, color="#FFFFFF"),
                    opacity=0.95,
                ),
                name="RED Alert",
                hovertemplate=(
                    "<b>RED ALERT</b><br>"
                    "%{x|%b %d}<br>"
                    "Composite: %{y:.1f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    if yellow_alerts:
        fig.add_trace(
            go.Scatter(
                x=[a["date"] for a in yellow_alerts],
                y=[a.get("composite", 50) for a in yellow_alerts],
                mode="markers",
                marker=dict(
                    size=11,
                    color=ACCENT_AMBER,
                    symbol="diamond",
                    line=dict(width=1.5, color="#FFFFFF"),
                    opacity=0.9,
                ),
                name="YELLOW Alert",
                hovertemplate=(
                    "<b>YELLOW ALERT</b><br>"
                    "%{x|%b %d}<br>"
                    "Composite: %{y:.1f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    _add_event_markers(fig, row=1)

    # Panel 2: Pre-flare probability over time with subtle gradient fills
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=state_probs[:, 1],
            mode="lines",
            line=dict(color=ACCENT_AMBER, width=2),
            name="P(Pre-flare)",
            fill="tozeroy",
            fillcolor="rgba(245, 158, 11, 0.10)",
            hovertemplate=(
                "<b>%{x|%b %d}</b><br>P(Pre-flare): %{y:.1%}<br><extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=state_probs[:, 2],
            mode="lines",
            line=dict(color=ACCENT_RED, width=2),
            name="P(Active Flare)",
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.10)",
            hovertemplate=(
                "<b>%{x|%b %d}</b><br>P(Active Flare): %{y:.1%}<br><extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    # Threshold lines with refined styling
    fig.add_hline(
        y=YELLOW_PREFLARE_PROB,
        line_dash="dash",
        line_color=ACCENT_AMBER,
        opacity=0.5,
        line_width=1,
        row=2,
        col=1,
        annotation_text=f"YELLOW ({YELLOW_PREFLARE_PROB})",
        annotation_position="bottom right",
        annotation_font=dict(size=9, color=ACCENT_AMBER),
    )
    fig.add_hline(
        y=RED_PREFLARE_PROB,
        line_dash="dash",
        line_color=ACCENT_RED,
        opacity=0.5,
        line_width=1,
        row=2,
        col=1,
        annotation_text=f"RED ({RED_PREFLARE_PROB})",
        annotation_position="top right",
        annotation_font=dict(size=9, color=ACCENT_RED),
    )
    _add_event_markers(fig, row=2, show_labels=False)

    fig.update_layout(
        height=700,
        margin=dict(t=124),
        hovermode="x unified",
    )
    # Subtle gridlines and crosshair spikes
    for row_i in range(1, 3):
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikesnap="cursor",
            spikedash="dot",
            row=row_i,
            col=1,
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)",
            row=row_i,
            col=1,
        )
    fig.update_yaxes(title_text="Composite Score", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)

    figures.append(fig)
    log("ALERTS", "Alert evaluation complete.")
    return result


# ===========================================================================
# Section 7: Predictive Feature Importance
# ===========================================================================
def compute_feature_importance(
    daily: pd.DataFrame,
    composite: pd.Series,
    viterbi_path: np.ndarray,
) -> dict[str, Any]:
    """
    Rank biometric features by predictive power for GVHD state.

    Uses the rSLDS/HMM Viterbi state path as the prediction target, NOT the
    composite score. The composite score is a direct weighted sum of the input
    features (temp_dev, hrv_median, sleep_hr, spo2, frag_index, activity_score),
    so using it as the target would create a circular dependency where features
    are ranked by how well they predict a function of themselves.

    The Viterbi path is derived from the rSLDS/HMM latent state model which
    learns state transitions from multivariate observation dynamics - a
    genuinely independent prediction target.

    Metrics:
      1. Point-biserial correlation with binary flare target (Viterbi state >= 2)
      2. Mutual information (discretized) with binary flare target
      3. Kruskal-Wallis H statistic across all 4 Viterbi states (normalized)
      4. Cohen's d effect size between non-flare and flare days
    """
    log("FEATURES", "Computing predictive feature importance...")
    log("FEATURES", "  Target: Viterbi state path (non-circular; independent of composite weights)")

    features = [
        ("temp_dev", "Temperature Deviation (\u00b0C)"),
        ("temp_var_7d", "Temperature Variability 7d (\u00b0C)"),
        ("temp_gradient", "Temperature Gradient (\u00b0C/night)"),
        ("hrv_median", "HRV RMSSD (ms)"),
        ("hrv_std", "HRV Variability (ms)"),
        ("sleep_hr", "Sleep Heart Rate (bpm)"),
        ("lowest_hr", "Lowest HR (bpm)"),
        ("sleep_eff", "Sleep Efficiency (%)"),
        ("frag_index", "Sleep Fragmentation (transitions/hr)"),
        ("rem_pct", "REM %"),
        ("deep_pct", "Deep Sleep %"),
        ("spo2_average", "SpO2 Average (%)"),
        ("activity_score", "Activity Score"),
        ("steps", "Steps"),
        ("readiness_score", "Readiness Score"),
        ("stress_high", "Stress High (sec)"),
        ("sleep_breath", "Breathing Rate (breaths/min)"),
    ]

    results = []

    # Target: binary flare indicator from Viterbi state path.
    # States: 0=Remission, 1=Pre-flare, 2=Active Flare, 3=Recovery
    # viterbi_path uses -1 for no-data days.
    # Binary target: state >= 2 (Active Flare or Recovery) = high-risk
    n_daily = len(daily)
    vp = viterbi_path[:n_daily] if len(viterbi_path) >= n_daily else np.pad(
        viterbi_path, (0, n_daily - len(viterbi_path)), constant_values=-1
    )

    binary_target = pd.Series(
        np.where(vp >= 2, 1, np.where(vp >= 0, 0, np.nan)),
        index=daily.index,
    )

    valid_target = binary_target.dropna()
    n_flare = int((valid_target == 1).sum())
    n_nonflare = int((valid_target == 0).sum())
    log("FEATURES", f"  Flare days: {n_flare}, non-flare days: {n_nonflare}")

    # Fallback: if too few flare days, broaden to state >= 1 (pre-flare+)
    if n_flare < 3 or n_nonflare < 3:
        log("FEATURES", "  WARNING: Insufficient state variability, broadening to state >= 1")
        binary_target = pd.Series(
            np.where(vp >= 1, 1, np.where(vp >= 0, 0, np.nan)),
            index=daily.index,
        )

    for col, label in features:
        if col not in daily.columns:
            continue
        feature = daily[col].copy()

        # Drop NaN pairs (feature NaN or invalid Viterbi state)
        valid = feature.notna() & binary_target.notna()
        if valid.sum() < 10:
            continue

        f_vals = feature[valid].values
        t_vals = binary_target[valid].values.astype(int)

        # 1. Point-biserial correlation with binary flare target
        if len(np.unique(t_vals)) > 1 and np.std(f_vals) > 0:
            corr, p_val = scipy_stats.pointbiserialr(t_vals, f_vals)
        else:
            corr, p_val = 0.0, 1.0

        # 2. Mutual information (discretized)
        mi = _mutual_information(f_vals, t_vals, n_bins=5)

        # 3. Kruskal-Wallis H across all Viterbi states (replaces circular
        #    composite correlation). Measures how well feature discriminates
        #    between ALL latent states, not just binary.
        vp_aligned = vp[valid.values]
        state_groups = [
            f_vals[vp_aligned == s]
            for s in range(N_STATES)
            if np.sum(vp_aligned == s) >= 2
        ]
        if len(state_groups) >= 2:
            try:
                h_stat, _h_pval = scipy_stats.kruskal(*state_groups)
                # Normalize: divide by scaling factor, cap at 1.0
                kw_score = min(h_stat / (max(len(state_groups) - 1, 1) * 50), 1.0)
            except ValueError:
                kw_score = 0.0
        else:
            kw_score = 0.0

        # 4. Effect size: Cohen's d between non-flare and flare days
        low_vals = f_vals[t_vals == 0]
        high_vals = f_vals[t_vals == 1]
        if len(low_vals) > 2 and len(high_vals) > 2:
            n_low, n_high = len(low_vals), len(high_vals)
            pooled_std = (
                np.sqrt(
                    (
                        (n_low - 1) * low_vals.std(ddof=1) ** 2
                        + (n_high - 1) * high_vals.std(ddof=1) ** 2
                    )
                    / (n_low + n_high - 2)
                )
                + 1e-6
            )
            cohens_d = (high_vals.mean() - low_vals.mean()) / pooled_std
        else:
            cohens_d = 0.0

        # Combined importance score (no circular composite correlation)
        importance = (
            abs(corr) * 0.30
            + mi * 0.30
            + kw_score * 0.20
            + min(abs(cohens_d) / 2, 1) * 0.20
        )

        results.append(
            {
                "feature": col,
                "label": label,
                "correlation": round(float(corr), 4),
                "p_value": round(float(p_val), 6),
                "mutual_info": round(float(mi), 4),
                "kruskal_wallis": round(float(kw_score), 4),
                "cohens_d": round(float(cohens_d), 4),
                "importance": round(float(importance), 4),
            }
        )

    # Sort by importance
    results.sort(key=lambda x: x["importance"], reverse=True)

    for r in results[:5]:
        log(
            "FEATURES",
            f"  {r['label']}: importance={r['importance']:.3f} (corr={r['correlation']:.3f}, MI={r['mutual_info']:.3f})",
        )

    # --- Figure: Feature importance ---
    fig = go.Figure()

    labels = [r["label"] for r in results]
    importances = [r["importance"] for r in results]

    # Gradient coloring based on importance value
    bar_colors = []
    bar_borders = []
    for r in results:
        imp = r["importance"]
        if imp > 0.4:
            bar_colors.append(ACCENT_RED)
            bar_borders.append("rgba(239, 68, 68, 0.6)")
        elif imp > 0.3:
            bar_colors.append(ACCENT_AMBER)
            bar_borders.append("rgba(245, 158, 11, 0.6)")
        elif imp > 0.2:
            bar_colors.append(ACCENT_BLUE)
            bar_borders.append("rgba(59, 130, 246, 0.6)")
        else:
            bar_colors.append("rgba(59, 130, 246, 0.5)")
            bar_borders.append("rgba(59, 130, 246, 0.3)")

    fig.add_trace(
        go.Bar(
            y=labels[::-1],
            x=importances[::-1],
            orientation="h",
            marker=dict(
                color=bar_colors[::-1],
                line=dict(color=bar_borders[::-1], width=1),
            ),
            text=[f"  {v:.3f}" for v in importances[::-1]],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_PRIMARY),
            hovertemplate=("<b>%{y}</b><br>Importance: %{x:.3f}<br><extra></extra>"),
        )
    )

    fig.update_layout(
        height=max(450, len(results) * 32 + 120),
        xaxis_title="Combined Importance Score",
        yaxis_title="",
        margin=dict(l=140, t=64, r=60, b=44),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", automargin=True),
    )

    figures.append(fig)
    log("FEATURES", "Feature importance analysis complete.")
    return {"rankings": results}


def _mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 5) -> float:
    """Compute mutual information between continuous x and binary y."""
    # Discretize x into bins
    try:
        x_bins = np.digitize(x, np.linspace(x.min(), x.max(), n_bins + 1)[1:-1])
    except ValueError:
        return 0.0

    # Joint and marginal distributions
    joint = np.zeros((n_bins, 2))
    for xi, yi in zip(x_bins, y):
        b = min(max(int(xi), 0), n_bins - 1)
        joint[b, int(yi)] += 1

    joint /= joint.sum() + 1e-10
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_bins):
        for j in range(2):
            if joint[i, j] > 1e-10 and px[i] > 1e-10 and py[j] > 1e-10:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))

    return max(mi, 0.0)


# ===========================================================================
# Section 8: BOS Risk Integration
# ===========================================================================
def bos_risk_integration(daily: pd.DataFrame, composite: pd.Series) -> dict[str, Any]:
    """
    Integrate BOS (Bronchiolitis Obliterans Syndrome) risk with systemic GVHD.
    BOS is the pulmonary manifestation of chronic GVHD.
    """
    log("BOS", "Integrating BOS risk with systemic GVHD assessment...")

    if BOS_RISK_SCORE is not None:
        log(
            "BOS",
            f"  Loaded BOS score from SpO2 analysis: {BOS_RISK_SCORE} ({BOS_RISK_LEVEL})",
        )
    else:
        log(
            "BOS",
            "  WARNING: spo2_bos_metrics.json unavailable — BOS score will be N/A",
        )

    result: dict[str, Any] = {
        "bos_risk_score": BOS_RISK_SCORE if BOS_RISK_SCORE is not None else "N/A",
        "bos_risk_level": BOS_RISK_LEVEL if BOS_RISK_LEVEL is not None else "N/A",
        "bos_recommendation": BOS_RISK_RECOMMENDATION
        if BOS_RISK_RECOMMENDATION
        else "N/A",
    }

    # SpO2 analysis for pulmonary component
    spo2 = daily["spo2_average"].dropna()
    if len(spo2) > 5:
        result["spo2_stats"] = {
            "mean": round(float(spo2.mean()), 2),
            "std": round(float(spo2.std()), 2),
            "min": round(float(spo2.min()), 2),
            "below_96": int((spo2 < 96).sum()),
            "below_95": int((spo2 < 95).sum()),
            "trend_slope": round(
                float(np.polyfit(range(len(spo2)), spo2.values, 1)[0]), 4
            ),
            "n_readings": len(spo2),
        }
        log(
            "BOS",
            f"  SpO2 mean: {result['spo2_stats']['mean']}%, readings: {result['spo2_stats']['n_readings']}",
        )
        log("BOS", f"  SpO2 < 96%: {result['spo2_stats']['below_96']} days")
        log("BOS", f"  SpO2 trend: {result['spo2_stats']['trend_slope']:.4f} %/day")
    else:
        result["spo2_stats"] = {
            "note": "Insufficient SpO2 data for pulmonary assessment"
        }
        log("BOS", "  Insufficient SpO2 data")

    # Breathing rate correlation with GVHD composite
    breath = daily["sleep_breath"].dropna()
    if len(breath) > 10:
        # Align on shared index
        shared_idx = breath.index.intersection(composite.dropna().index)
        if len(shared_idx) > 10:
            breath_corr = breath.loc[shared_idx].corr(composite.loc[shared_idx])
            result["breath_composite_correlation"] = round(
                float(breath_corr) if pd.notna(breath_corr) else 0.0, 3
            )
            log("BOS", f"  Breathing rate - composite correlation: {breath_corr:.3f}")

    # Combined pulmonary-systemic risk
    systemic_mean = float(composite.mean())
    pulmonary_factor = 1.0
    if "spo2_stats" in result and "mean" in result["spo2_stats"]:
        # Lower SpO2 increases risk
        if result["spo2_stats"]["mean"] < 96:
            pulmonary_factor = 1.2
        if result["spo2_stats"]["mean"] < 95:
            pulmonary_factor = 1.5

    if BOS_RISK_SCORE is not None:
        bos_val = BOS_RISK_SCORE
        combined_risk = systemic_mean * 0.7 + bos_val * 0.3 * pulmonary_factor
        result["combined_risk"] = {
            "systemic_component": round(systemic_mean, 1),
            "pulmonary_component": round(bos_val * pulmonary_factor, 1),
            "combined_score": round(combined_risk, 1),
            "interpretation": (
                "HIGH"
                if combined_risk > 60
                else "MODERATE"
                if combined_risk > 40
                else "LOW"
            ),
        }
    else:
        # Fallback: systemic-only (no BOS component available)
        result["combined_risk"] = {
            "systemic_component": round(systemic_mean, 1),
            "pulmonary_component": "N/A",
            "combined_score": round(systemic_mean, 1),
            "interpretation": (
                "HIGH"
                if systemic_mean > 60
                else "MODERATE"
                if systemic_mean > 40
                else "LOW"
            ),
            "note": "BOS component unavailable — systemic score only",
        }
    log(
        "BOS",
        f"  Combined risk: {result['combined_risk']['combined_score']:.1f} ({result['combined_risk']['interpretation']})",
    )

    # Ruxolitinib impact assessment
    rux_date = pd.Timestamp(RUXOLITINIB_START)
    pre_rux = composite[composite.index < rux_date]
    post_rux = composite[composite.index >= rux_date]
    if len(post_rux) >= 2:
        result["ruxolitinib_response"] = {
            "pre_mean": round(float(pre_rux.mean()), 1),
            "post_mean": round(float(post_rux.mean()), 1),
            "delta": round(float(post_rux.mean() - pre_rux.mean()), 1),
            "post_days": len(post_rux),
            "note": "Too early to assess (< 7 days)"
            if len(post_rux) < 7
            else "Preliminary assessment",
        }
        log(
            "BOS",
            f"  Ruxolitinib: pre={result['ruxolitinib_response']['pre_mean']}, "
            f"post={result['ruxolitinib_response']['post_mean']} ({result['ruxolitinib_response']['post_days']}d)",
        )

    log("BOS", "BOS risk integration complete.")
    return result


# ===========================================================================
# Section 9: HTML Report Generation
# ===========================================================================
def generate_html_report(
    temp_result: dict,
    composite_result: dict,
    rslds_result: dict,
    alert_result: dict,
    feature_result: dict,
    bos_result: dict,
) -> str:
    """Generate dark-themed HTML report using the shared design system."""
    log("REPORT", "Generating HTML report...")

    # --- Build figure HTML snippets ---
    fig_htmls: list[str] = []
    for fig in figures:
        fig_htmls.append(fig.to_html(full_html=False, include_plotlyjs=False))

    def _fig(idx: int) -> str:
        return fig_htmls[idx] if idx < len(fig_htmls) else "<p>No figure available</p>"

    # --- KPI cards ---
    peak_score = composite_result.get("composite_stats", {}).get("max", "N/A")
    peak_date = composite_result.get("composite_stats", {}).get("max_date", "N/A")

    viterbi_state = rslds_result.get("event_validation", {}).get("viterbi_state", "N/A")
    model_type = rslds_result.get("model_type", "rSLDS")
    model_info = rslds_result.get("model_info", {})
    state_model_label = "HMM" if "HMM" in model_type else "rSLDS"
    state_status = (
        "critical"
        if viterbi_state in ("Active Flare", "Pre-flare")
        else "normal"
        if viterbi_state in ("Remission", "Recovery")
        else "info"
    )
    state_label = (
        "Flare"
        if viterbi_state in ("Active Flare", "Pre-flare")
        else "Stable"
        if viterbi_state in ("Remission", "Recovery")
        else ""
    )

    red_in_window = alert_result.get("validation", {}).get(
        "red_alerts_in_event_window", 0
    )
    red_outside_window = alert_result.get("validation", {}).get(
        "red_alerts_outside_event_window", 0
    )
    total_red = alert_result.get("summary", {}).get("n_red", 0)
    red_label = "Detected" if total_red else "None"

    top_features = feature_result.get("rankings", [])[:3]
    top_feature_labels = (
        ", ".join(r["label"] for r in top_features)
        if top_features
        else "HRV / heart-rate features"
    )
    model_method = model_info.get("method", model_type)

    combined_score = bos_result.get("combined_risk", {}).get("combined_score", "N/A")
    combined_interp = bos_result.get("combined_risk", {}).get("interpretation", "N/A")
    # LOW composite does NOT mean "normal" — patient has confirmed multi-organ cGvHD.
    # A low wearable-derived score means the model under-captures organ burden,
    # not that the patient is healthy.
    combined_status = (
        "critical"
        if combined_interp == "HIGH"
        else "warning"
        if combined_interp in {"MODERATE", "ELEVATED", "LOW"}
        else "normal"
    )
    combined_label = (
        "High"
        if combined_interp == "HIGH"
        else "Elevated"
        if combined_interp in {"MODERATE", "ELEVATED"}
        else "Low — model limited"
    )

    peak_label = "Critical" if peak_score != "N/A" else ""

    body = make_kpi_row(
        make_kpi_card(
            "Peak GVHD Composite",
            peak_score if isinstance(peak_score, (int, float)) else peak_score,
            status="critical",
            detail=f"on {peak_date}",
            decimals=1,
            status_label=peak_label,
        ),
        make_kpi_card(
            f"{state_model_label} Feb 9 State",
            viterbi_state,
            status=state_status,
            detail=model_method,
            status_label=state_label,
        ),
        make_kpi_card(
            "RED Alerts in ±3d Window",
            f"{red_in_window}/{total_red}" if total_red else "0/0",
            status="warning" if total_red else "info",
            detail=(
                f"{red_outside_window} outside ±3d event window"
                + (" | retrospective only" if total_red else "")
            ),
            status_label=red_label,
        ),
        make_kpi_card(
            "Combined GVHD + BOS",
            combined_score
            if isinstance(combined_score, (int, float))
            else combined_score,
            status=combined_status,
            detail=f"{combined_interp} — wearable signals only (14 organ systems affected)",
            decimals=1,
            status_label=combined_label,
        ),
    )

    # --- Clinical context note ---
    body += (
        '<div class="clinical-note">'
        "<strong>Clinical Context:</strong> Chronic GVHD across 14 organ systems: "
        "skin (biopsy-verified), oral cavity (biopsy-verified), eyes (sicca), liver, GI tract, "
        "lungs (declining DLCO, BOS not excluded), joints/fascia, genitalia, heart (pericardial effusion, "
        "borderline EF), brain/CNS (white matter lesions 5x age norm), peripheral nerves (burning feet, "
        "tremor), autonomic nervous system (HRV 2nd percentile, IST), endocrine (secondary hypogonadism). "
        "NIH 2014 consensus: minimum moderate (3+ organs score &ge;1); clinically under-graded as mild by OUS. "
        "Known acute decompensation on Feb 9, 2026 (validation target). "
        f"{state_model_label} classified Feb 9 as <strong>{viterbi_state}</strong>, while RED alerts were mostly off-window "
        f"({red_outside_window}/{total_red} outside ±3d), so treat this as a retrospective state-classification signal. "
        f"Top ranked features in this run: {top_feature_labels}. "
        "Temperature deviation contributes inflammatory context but is not the leading feature. "
        "<strong>Note:</strong> The wearable-derived composite captures cardiovascular and sleep signals only. "
        "Organ involvement beyond what a ring can measure (eyes, mouth, skin, fascia, GI, genitalia, CNS) is not reflected in the score."
        "</div>"
    )

    # --- Section 1: Temperature ---
    sec1 = (
        '<div class="methodology">'
        "Temperature deviation from Oura measures deviations from personal baseline body temperature. "
        "GVHD causes micro-inflammatory signatures visible as increased night-to-night variability. "
        "CUSUM change-point detection identifies regime shifts in temperature dynamics."
        "</div>"
        + _fig(0)
        + f"<p>Regime changes detected: {', '.join(temp_result.get('regime_changes', [])) or 'None'}</p>"
    )
    body += make_section("1. Temperature Fluctuation Analysis", sec1)

    # --- Section 2: Composite ---
    sec2 = (
        '<div class="methodology">'
        "Six biometric streams weighted by clinical relevance: "
        "temperature (25%), HRV (20%), SpO2 (15%), sleep fragmentation (15%), resting HR (15%), activity (10%). "
        "Each component z-scored against first 14 days baseline, normalized to 0-100 (higher = more GVHD-like)."
        "</div>" + _fig(1)
    )
    body += make_section("2. Multi-Stream GVHD Composite Score", sec2)

    # --- Section 3: state model ---
    # State distribution table
    state_dist = rslds_result.get("state_distribution", {})
    state_rows = ""
    for name, info in state_dist.items():
        state_rows += (
            f"<tr><td>{name}</td>"
            f"<td style='text-align:center;'>{info['days']}</td>"
            f"<td style='text-align:center;'>{info['pct']:.1f}%</td></tr>"
        )
    state_table = (
        f"<h3>{state_model_label} State Distribution</h3>"
        "<table><thead><tr>"
        "<th>State</th><th>Days</th><th>Percentage</th>"
        "</tr></thead><tbody>" + state_rows + "</tbody></table>"
    )

    if "HMM" in model_type:
        fallback_reason = _html_escape_mod.escape(
            model_info.get("rslds_failure_reason")
            or model_info.get("rslds_skipped_reason")
            or "rSLDS path unavailable in this run"
        )
        sec3 = (
            '<div class="methodology">'
            "4-state Gaussian HMM (hmmlearn) fitted to composite score + temperature deviation + HRV "
            f"(3 features). rSLDS was unavailable in this run. Reason: {fallback_reason}."
            "</div>" + _fig(2) + state_table
        )
    else:
        sec3 = (
            '<div class="methodology">'
            "4-state rSLDS (Remission, Pre-flare, Active Flare, Recovery) fitted via Laplace-EM "
            "(Linderman et al. 2017). Each discrete state governs linear dynamics in a continuous "
            "latent space, with recurrent transitions that depend on the latent state. "
            "Observations: composite score + temperature deviation + HRV (3 features)."
            "</div>" + _fig(2) + state_table
        )
    body += make_section(f"3. {state_model_label} State Model", sec3)

    # --- Section 4: alerts ---
    alerts = alert_result.get("alerts", [])
    alert_rows = ""
    for a in alerts:
        row_cls = (
            ' class="alert-red"' if a["level"] == "RED" else ' class="alert-yellow"'
        )
        alert_rows += (
            f"<tr{row_cls}>"
            f"<td>{a['date']}</td>"
            f"<td style='font-weight:600;'>{a['level']}</td>"
            f"<td style='text-align:center;'>{a['p_preflare']}</td>"
            f"<td style='text-align:center;'>{a['p_flare']}</td>"
            f"<td style='text-align:center;'>{a.get('composite', 'N/A')}</td>"
            "</tr>"
        )
    alert_table = (
        f"<h3>Alert History ({len(alerts)} total)</h3>"
        "<table><thead><tr>"
        "<th>Date</th><th>Level</th><th>P(Pre-flare)</th><th>P(Flare)</th><th>Composite</th>"
        "</tr></thead><tbody>" + alert_rows + "</tbody></table>"
    )

    sec4 = (
        '<div class="methodology">'
        f"YELLOW alert: P(pre-flare) > {YELLOW_PREFLARE_PROB} for {YELLOW_CONSEC_DAYS}+ consecutive days.<br>"
        f"RED alert: P(pre-flare) > {RED_PREFLARE_PROB} OR P(active flare) > {RED_FLARE_PROB}.<br>"
        f"Retrospective validation against Feb 9, 2026 acute decompensation: {red_in_window}/{total_red} RED alerts "
        f"fell within ±3d of the event and {red_outside_window} occurred outside that window."
        "</div>" + _fig(3) + alert_table
    )
    body += make_section("4. Retrospective Alert Burden", sec4)

    # --- Section 5: Feature importance ---
    top_features = feature_result.get("rankings", [])[:7]
    feature_rows = ""
    for r in top_features:
        feature_rows += (
            f"<tr><td>{r['label']}</td>"
            f"<td style='text-align:center;'>{r['importance']:.3f}</td>"
            f"<td style='text-align:center;'>{r['correlation']:+.3f}</td>"
            f"<td style='text-align:center;'>{r['mutual_info']:.3f}</td>"
            f"<td style='text-align:center;'>{r['cohens_d']:+.2f}</td></tr>"
        )
    feature_table = (
        "<h3>Top Predictive Features for GVHD State</h3>"
        "<table><thead><tr>"
        "<th>Feature</th><th>Importance</th><th>Correlation</th>"
        "<th>Mutual Info</th><th>Cohen's d</th>"
        "</tr></thead><tbody>" + feature_rows + "</tbody></table>"
    )

    sec5 = (
        '<div class="methodology">'
        "Features ranked by combined score: point-biserial correlation with Viterbi flare state (30%), "
        "mutual information (30%), Kruskal-Wallis H across all 4 latent states (20%), "
        "Cohen's d effect size (20%). "
        "Binary target: Viterbi state &ge; 2 (Active Flare / Recovery). "
        "Target derived from rSLDS/HMM latent states, not the composite score, to avoid circularity."
        "</div>" + _fig(4) + feature_table
    )
    body += make_section("5. Predictive Feature Importance", sec5)

    # --- Section 6: BOS Risk ---
    spo2_mean = bos_result.get("spo2_stats", {}).get("mean", "N/A")
    spo2_trend = bos_result.get("spo2_stats", {}).get("trend_slope", "N/A")
    bos_score_display = bos_result.get("bos_risk_score", "N/A")
    bos_level_display = bos_result.get("bos_risk_level", "N/A")
    bos_recommendation = bos_result.get("bos_recommendation", "N/A")
    sec6 = (
        '<div class="methodology">'
        f"Bronchiolitis obliterans syndrome (BOS) is the pulmonary manifestation of chronic GVHD. "
        f"BOS risk score ({bos_score_display}/{bos_level_display}) loaded from SpO2/BOS analysis "
        "and integrated with systemic GVHD composite. "
        "SpO2 trend analysis provides supplementary pulmonary assessment."
        "</div>"
        "<ul>"
        f"<li>BOS Risk Score: {bos_result.get('bos_risk_score', 'N/A')} ({bos_result.get('bos_risk_level', 'N/A')})</li>"
        f"<li>Combined Risk: {combined_score} ({combined_interp})</li>"
        f"<li>SpO2 Mean: {spo2_mean}%</li>"
        f"<li>SpO2 Trend: {spo2_trend} %/day</li>"
        f"<li>BOS Recommendation: {bos_recommendation}</li>"
        "</ul>"
    )
    body += make_section("6. BOS Risk Integration", sec6)

    # --- Disclaimer ---
    body += (
        '<div class="disclaimer">'
        "<strong>Disclaimer:</strong> This analysis is for research purposes only and should not be used "
        "as a sole basis for clinical decisions. "
        "N=1 retrospective case study - all detection metrics are descriptive, not inferential. "
        "Sensitivity/specificity cannot be computed from a single patient. "
        "Validation requires an external multi-patient cohort. "
        "Temperature deviation from consumer wearables has limited precision compared to clinical thermometry. "
        "All clinical decisions should be made in consultation with the treating hematologist."
        "</div>"
    )

    # --- Extra CSS for report-specific classes ---
    extra_css = f"""
.clinical-note {{
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid {ACCENT_AMBER};
  border-radius: 8px;
  padding: 16px;
  margin: 0 0 24px 0;
  font-size: 0.875rem;
  color: {TEXT_PRIMARY};
  line-height: 1.6;
}}
.clinical-note strong {{ color: {ACCENT_AMBER}; }}
.methodology {{
  background: rgba(59, 130, 246, 0.08);
  border: 1px solid rgba(59, 130, 246, 0.25);
  border-radius: 8px;
  padding: 16px;
  margin: 0 0 16px 0;
  font-size: 0.8125rem;
  color: {TEXT_SECONDARY};
  line-height: 1.6;
}}
.disclaimer {{
  background: {BG_ELEVATED};
  border: 1px solid {BORDER_SUBTLE};
  border-radius: 8px;
  padding: 16px;
  margin-top: 24px;
  font-size: 0.75rem;
  color: {TEXT_SECONDARY};
  line-height: 1.6;
}}
.disclaimer strong {{ color: {TEXT_PRIMARY}; }}
.odt-section ul {{
  padding-left: 20px;
  font-size: 0.875rem;
  color: {TEXT_SECONDARY};
}}
.odt-section ul li {{ margin-bottom: 4px; }}
tr.alert-red td {{ background: rgba(239, 68, 68, 0.1); }}
tr.alert-yellow td {{ background: rgba(245, 158, 11, 0.1); }}
"""

    # --- Subtitle ---
    n_days = (pd.Timestamp(DATA_END) - pd.Timestamp(DATA_START)).days + 1
    post_days = max(0, (pd.Timestamp(DATA_END).date() - TREATMENT_START).days + 1)
    display_model_type = "HMM" if "HMM" in str(model_type) else "rSLDS"
    subtitle = (
        f"Oura Ring, {DATA_START} to {DATA_END} ({n_days} days) "
        f"| State model: {display_model_type} | Ruxolitinib started {RUXOLITINIB_START} | HEV diagnosed {HEV_DIAGNOSIS}"
    )

    # --- Assemble full page ---
    html = wrap_html(
        title="GvHD Prediction Model",
        body_content=body,
        report_id="gvhd",
        subtitle=subtitle,
        extra_css=extra_css,
        data_end=DATA_END,
        post_days=post_days,
    )

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    log("REPORT", f"HTML report written to {HTML_OUTPUT.name}")
    return str(HTML_OUTPUT)


# ===========================================================================
# Section 10: Main Pipeline
# ===========================================================================
def main() -> None:
    """Execute the full GVHD prediction pipeline."""
    data_end = _get_data_end()
    metrics["generated_at"] = datetime.now(timezone.utc).isoformat()
    metrics["data_range"] = {"start": DATA_START, "end": data_end}

    t0 = time.time()
    print("=" * 70)
    print("GVHD Flare Prediction System - Oura Ring Biometric Analysis")
    print(f"Patient: {PATIENT_LABEL}")
    print(f"Data range: {DATA_START} to {DATA_END} (dynamic)")
    print(f"Known event: {KNOWN_EVENT_DATE} (acute decompensation)")
    print("=" * 70)

    try:
        # 1. Load data
        data = load_all_data()

        # 2. Build daily feature matrix
        daily = build_daily_features(data)

        # 3. Temperature analysis
        temp_result = analyze_temperature(daily)
        metrics["temperature"] = temp_result

        # 4. GVHD composite score
        composite, composite_result = compute_gvhd_composite(daily)
        metrics["composite"] = composite_result

        # 5. rSLDS analysis (falls back to HMM if ssm unavailable)
        state_probs, viterbi_path, rslds_result = run_rslds_analysis(daily, composite)
        metrics["rslds"] = rslds_result

        # 6. Alert evaluation
        alert_result = evaluate_alerts(daily, state_probs, viterbi_path, composite)
        metrics["alerts"] = alert_result

        # 7. Feature importance
        feature_result = compute_feature_importance(daily, composite, viterbi_path)
        metrics["features"] = feature_result

        # 8. BOS risk integration
        bos_result = bos_risk_integration(daily, composite)
        metrics["bos"] = bos_result

        # 9. Generate report
        html_path = generate_html_report(
            temp_result,
            composite_result,
            rslds_result,
            alert_result,
            feature_result,
            bos_result,
        )

        # 10. Save metrics JSON
        metrics["study_type"] = "N=1 retrospective case study"
        metrics["validation_note"] = (
            "All detection metrics are descriptive. External cohort validation required."
        )
        metrics["runtime_seconds"] = round(time.time() - t0, 2)
        metrics["progress_log"] = progress_log
        with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        log("DONE", f"Metrics JSON written to {JSON_OUTPUT.name}")

        # Summary
        elapsed = time.time() - t0
        print("\n" + "=" * 70)
        print(f"GVHD Prediction Pipeline Complete ({elapsed:.1f}s)")
        print(f"  HTML report: {html_path}")
        print(f"  JSON metrics: {JSON_OUTPUT}")
        print(f"  Figures generated: {len(figures)}")

        # Key findings
        if composite_result.get("composite_stats"):
            print(
                f"\n  Peak GVHD score: {composite_result['composite_stats']['max']:.1f} on {composite_result['composite_stats']['max_date']}"
            )
        if rslds_result.get("event_validation"):
            ev = rslds_result["event_validation"]
            model_label = rslds_result.get("model_type", "state model")
            print(f"  Feb 9 {model_label} state: {ev['viterbi_state']}")
        if alert_result.get("validation", {}).get("detected"):
            val = alert_result["validation"]
            print(f"  Alert lead time: {val.get('red_lead_time_days', 'N/A')} days")
            print(
                f"  N=1 case study: {val.get('red_alerts_in_event_window', 0)} RED in event window, {val.get('red_alerts_outside_event_window', 0)} outside (external cohort validation required)"
            )
        if bos_result.get("combined_risk"):
            cr = bos_result["combined_risk"]
            print(
                f"  Combined GVHD+BOS risk: {cr['combined_score']:.1f} ({cr['interpretation']})"
            )

        print("=" * 70)

    except Exception as e:
        log("ERROR", f"Pipeline failed: {e}")
        traceback.print_exc()
        # Save partial metrics
        metrics["error"] = str(e)
        metrics["runtime_seconds"] = round(time.time() - t0, 2)
        with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        sys.exit(1)


if __name__ == "__main__":
    main()
