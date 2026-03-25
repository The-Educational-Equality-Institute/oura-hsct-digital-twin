#!/usr/bin/env python3
"""
Comprehensive Oura Ring Analysis

Produces a single interactive HTML dashboard covering ALL Oura endpoints:
  1. Executive Summary (key metrics at a glance)
  2. HRV Deep Dive (RMSSD trends, distribution, Poincare, circadian)
  3. Heart Rate Analysis (daily trends, circadian, tachycardia prevalence)
  4. Sleep Analysis (scores, durations, HR during sleep, breath rate)
  5. Readiness & Recovery (scores, HRV balance, recovery index)
  6. Resilience & Cardiovascular Age
  7. SpO2 Monitoring
  8. Stress / Recovery Balance
  9. Activity & Physical Capacity
 10. Clinical Implications Summary

Data sources:
  - oura.db (configured in config.py): Oura biometric tables
  - Clinical timeline events (optional)

Configuration:
  Shared project constants and paths are
  defined in config.py at the project root.

Usage:
    python analysis/analyze_oura_full.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths & patient config (all from config.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH, REPORTS_DIR,
    ESC_RMSSD_DEFICIENCY, NOCTURNAL_HR_ELEVATED, IST_HR_THRESHOLD,
    POPULATION_RMSSD_MEDIAN,
    NORM_RMSSD_P25, NORM_RMSSD_P75, TREATMENT_START,
)

from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    C_CRITICAL, C_WARNING, C_GOOD, C_CAUTION, C_NEUTRAL, C_BG_LIGHT,
    C_LIGHT, C_DARK, BG_ELEVATED, ACCENT_RED, ACCENT_BLUE, ACCENT_CYAN,
    ACCENT_AMBER, ACCENT_GREEN,
)

pio.templates.default = "clinical_dark"

# Derived constants from config ranges
NORM_RMSSD_P50 = POPULATION_RMSSD_MEDIAN

# Nocturnal HR norms based on healthy adult reference literature
NOCTURNAL_HR_DIP_NORMAL_LOW = 45  # bpm  - healthy nocturnal minimum
NOCTURNAL_HR_DIP_NORMAL_HIGH = 55  # bpm
SLEEP_AVG_HR_NORMAL = 65  # bpm  - average HR during sleep for healthy adult

# Supplementary color aliases (local to this script)
C_OK = C_GOOD
C_BLUE = ACCENT_BLUE
C_BG = C_BG_LIGHT


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def df_from(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        import logging
        logging.warning("Table not found or query failed, using empty DataFrame")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_all_data(bio: sqlite3.Connection) -> dict:
    """Load every relevant table into DataFrames."""
    d: dict[str, pd.DataFrame] = {}

    # --- Oura HRV (5-min RMSSD samples) ---
    d["oura_hrv"] = df_from(bio, "SELECT timestamp, rmssd, source FROM oura_hrv WHERE rmssd IS NOT NULL")
    if not d["oura_hrv"].empty:
        d["oura_hrv"]["timestamp"] = pd.to_datetime(d["oura_hrv"]["timestamp"], utc=True)
        d["oura_hrv"]["date"] = d["oura_hrv"]["timestamp"].dt.date
        d["oura_hrv"]["hour"] = d["oura_hrv"]["timestamp"].dt.hour

    # --- Oura Heart Rate ---
    d["oura_hr"] = df_from(bio, "SELECT timestamp, bpm, source FROM oura_heart_rate WHERE bpm IS NOT NULL")
    if not d["oura_hr"].empty:
        d["oura_hr"]["timestamp"] = pd.to_datetime(d["oura_hr"]["timestamp"], utc=True)
        d["oura_hr"]["date"] = d["oura_hr"]["timestamp"].dt.date
        d["oura_hr"]["hour"] = d["oura_hr"]["timestamp"].dt.hour

    # --- Oura Sleep (daily summary) ---
    d["oura_sleep"] = df_from(bio, """
        SELECT date, score, total_sleep_duration, rem_sleep_duration,
               deep_sleep_duration, light_sleep_duration, efficiency,
               hr_lowest, hr_average, hrv_average, breath_average, temperature_delta
        FROM oura_sleep WHERE score IS NOT NULL ORDER BY date
    """)
    if not d["oura_sleep"].empty:
        d["oura_sleep"]["date"] = pd.to_datetime(d["oura_sleep"]["date"])

    # --- Oura Sleep Periods (detailed) ---
    d["sleep_periods"] = df_from(bio, """
        SELECT day, type, total_sleep_duration, rem_sleep_duration,
               deep_sleep_duration, light_sleep_duration, awake_time,
               efficiency, average_heart_rate, lowest_heart_rate,
               average_hrv, average_breath, bedtime_start, bedtime_end, time_in_bed
        FROM oura_sleep_periods ORDER BY day
    """)
    if not d["sleep_periods"].empty:
        d["sleep_periods"]["day"] = pd.to_datetime(d["sleep_periods"]["day"])

    # --- Oura Readiness ---
    d["readiness"] = df_from(bio, """
        SELECT date, score, temperature_deviation, hrv_balance,
               recovery_index, resting_heart_rate, sleep_balance
        FROM oura_readiness WHERE score IS NOT NULL ORDER BY date
    """)
    if not d["readiness"].empty:
        d["readiness"]["date"] = pd.to_datetime(d["readiness"]["date"])

    # --- Oura Activity ---
    d["activity"] = df_from(bio, """
        SELECT date, score, active_calories, total_calories, steps,
               inactive_time, rest_time, low_activity_time,
               medium_activity_time, high_activity_time
        FROM oura_activity WHERE score IS NOT NULL ORDER BY date
    """)
    if not d["activity"].empty:
        d["activity"]["date"] = pd.to_datetime(d["activity"]["date"])

    # --- Oura SpO2 ---
    d["spo2"] = df_from(bio, "SELECT date, spo2_average FROM oura_spo2 WHERE spo2_average > 0 ORDER BY date")
    if not d["spo2"].empty:
        d["spo2"]["date"] = pd.to_datetime(d["spo2"]["date"])

    # --- Oura Stress ---
    d["stress"] = df_from(bio, "SELECT date, stress_high, recovery_high, day_summary FROM oura_stress ORDER BY date")
    if not d["stress"].empty:
        d["stress"]["date"] = pd.to_datetime(d["stress"]["date"])
        # Convert seconds to minutes
        d["stress"]["stress_min"] = d["stress"]["stress_high"] / 60
        d["stress"]["recovery_min"] = d["stress"]["recovery_high"] / 60

    # --- Oura Resilience ---
    d["resilience"] = df_from(bio, """
        SELECT date, level, contributors_sleep_recovery,
               contributors_daytime_recovery, contributors_stress
        FROM oura_resilience ORDER BY date
    """)
    if not d["resilience"].empty:
        d["resilience"]["date"] = pd.to_datetime(d["resilience"]["date"])

    # --- Oura Cardiovascular Age ---
    d["cv_age"] = df_from(bio, "SELECT date, vascular_age FROM oura_cardiovascular_age ORDER BY date")
    if not d["cv_age"].empty:
        d["cv_age"]["date"] = pd.to_datetime(d["cv_age"]["date"])

    # --- Samsung Health historical steps (pre-illness baseline) ---
    d["samsung_steps"] = df_from(bio, """
        SELECT date, steps FROM samsung_steps
        WHERE steps > 0 ORDER BY date
    """)
    if not d["samsung_steps"].empty:
        d["samsung_steps"]["date"] = pd.to_datetime(d["samsung_steps"]["date"])

    # --- Oura Workouts ---
    d["workouts"] = df_from(bio, """
        SELECT day, activity, calories, distance, intensity,
               start_datetime, end_datetime
        FROM oura_workouts ORDER BY day
    """)

    return d


def latest_observed_date(data: dict[str, pd.DataFrame]) -> date:
    """Return the latest observed date across loaded Oura tables."""
    candidates: list[date] = []
    for df in data.values():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for column in ("date", "day", "timestamp"):
            if column not in df.columns:
                continue
            series = pd.to_datetime(df[column], errors="coerce", utc=False).dropna()
            if not series.empty:
                candidates.append(series.max().date())
            break
    return max(candidates) if candidates else datetime.now().date()


def earliest_observed_date(data: dict[str, pd.DataFrame]) -> date:
    """Return the earliest observed date across loaded Oura tables."""
    candidates: list[date] = []
    for df in data.values():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for column in ("date", "day", "timestamp"):
            if column not in df.columns:
                continue
            series = pd.to_datetime(df[column], errors="coerce", utc=False).dropna()
            if not series.empty:
                candidates.append(series.min().date())
            break
    return min(candidates) if candidates else datetime.now().date()


# ---------------------------------------------------------------------------
# Compute aggregate statistics
# ---------------------------------------------------------------------------

def compute_stats(data: dict) -> dict:
    """Compute key statistics for the executive summary."""
    s: dict = {}

    # HRV stats
    hrv = data["oura_hrv"]
    if not hrv.empty:
        s["rmssd_mean"] = hrv["rmssd"].mean()
        s["rmssd_median"] = hrv["rmssd"].median()
        s["rmssd_p5"] = hrv["rmssd"].quantile(0.05)
        s["rmssd_p95"] = hrv["rmssd"].quantile(0.95)
        s["rmssd_min"] = hrv["rmssd"].min()
        s["rmssd_max"] = hrv["rmssd"].max()
        s["rmssd_std"] = hrv["rmssd"].std()
        s["rmssd_samples"] = len(hrv)
        daily_avg = hrv.groupby("date")["rmssd"].mean()
        s["rmssd_daily_mean"] = daily_avg.mean()
        s["rmssd_days_below_15"] = (daily_avg < ESC_RMSSD_DEFICIENCY).sum()
        s["rmssd_total_days"] = len(daily_avg)

    # HR stats (Oura)
    hr = data["oura_hr"]
    if not hr.empty:
        daily_hr = hr.groupby("date")["bpm"].agg(["mean", "min", "max"])
        s["hr_daily_mean"] = daily_hr["mean"].mean()
        s["hr_daily_min"] = daily_hr["min"].min()
        s["hr_daily_max"] = daily_hr["max"].max()
        s["hr_days_above_90"] = (daily_hr["mean"] > 90).sum()
        s["hr_total_days"] = len(daily_hr)
        s["hr_pct_tachycardic"] = 100 * s["hr_days_above_90"] / s["hr_total_days"]
        s["hr_total_samples"] = len(hr)

    # Readiness
    r = data["readiness"]
    if not r.empty:
        s["readiness_mean"] = r["score"].mean()
        s["readiness_min"] = r["score"].min()
        s["readiness_max"] = r["score"].max()
        hrv_bal = r["hrv_balance"].dropna()
        if not hrv_bal.empty:
            s["hrv_balance_mean"] = hrv_bal.mean()
            s["hrv_balance_min"] = hrv_bal.min()
            s["hrv_balance_max"] = hrv_bal.max()
        # CRITICAL: oura_readiness.resting_heart_rate is a CONTRIBUTOR SCORE (0-100),
        # NOT an actual heart rate. Values like 8, 16, 30 are scores. Use sleep_periods instead.
        rhr_score = r["resting_heart_rate"].dropna()
        if not rhr_score.empty:
            s["rhr_contributor_score_mean"] = rhr_score.mean()
            s["rhr_contributor_score_min"] = rhr_score.min()
            s["rhr_contributor_score_max"] = rhr_score.max()

    # Sleep
    sl = data["oura_sleep"]
    if not sl.empty:
        s["sleep_score_mean"] = sl["score"].mean()
        s["sleep_score_min"] = sl["score"].min()
        s["sleep_days_poor"] = (sl["score"] < 60).sum()
        s["sleep_total_days"] = len(sl)

    # Sleep periods for duration
    sp = data["sleep_periods"]
    if not sp.empty:
        long_sleep = sp[sp["type"] == "long_sleep"]
        if not long_sleep.empty:
            durations = long_sleep["total_sleep_duration"].dropna()
            s["sleep_duration_avg_hrs"] = (durations / 3600).mean()
            s["sleep_duration_min_hrs"] = (durations / 3600).min()
            s["sleep_duration_max_hrs"] = (durations / 3600).max()
            sleep_hr = long_sleep["average_heart_rate"].dropna()
            if not sleep_hr.empty:
                s["sleep_hr_mean"] = sleep_hr.mean()
            sleep_hrv = long_sleep["average_hrv"].dropna()
            if not sleep_hrv.empty:
                s["sleep_hrv_mean"] = sleep_hrv.mean()
            sleep_lowest = long_sleep["lowest_heart_rate"].dropna()
            if not sleep_lowest.empty:
                s["sleep_hr_lowest_mean"] = sleep_lowest.mean()

    # Nocturnal HR dip: use sleep_periods.lowest_heart_rate (actual bpm)
    if "sleep_hr_lowest_mean" in s:
        s["nocturnal_hr_dip_mean"] = s["sleep_hr_lowest_mean"]

    # Primary resting HR: use sleep period average HR (true resting measure).
    # IST criterion assessed against this, not the misleading readiness field.
    # Nocturnal HR concern: sleeping HR > 80 bpm is abnormal in healthy adult reference ranges.
    # IST criterion (24-hour mean > 90 bpm, HRS/EHRA 2015) applies only to 24-hour data.
    if "sleep_hr_mean" in s:
        s["resting_hr_primary"] = s["sleep_hr_mean"]
        s["ist_met"] = s["sleep_hr_mean"] > NOCTURNAL_HR_ELEVATED
    elif "hr_daily_mean" in s:
        s["resting_hr_primary"] = s["hr_daily_mean"]
        s["ist_met"] = s["hr_daily_mean"] > IST_HR_THRESHOLD  # 24-hour data: use IST criterion
    else:
        s["resting_hr_primary"] = 0
        s["ist_met"] = False

    # Activity
    act = data["activity"]
    if not act.empty:
        s["steps_mean"] = act["steps"].mean()
        s["steps_max"] = act["steps"].max()
        s["steps_days_under_2000"] = (act["steps"] < 2000).sum()
        s["steps_days_under_5000"] = (act["steps"] < 5000).sum()
        s["activity_score_mean"] = act["score"].mean()
        s["activity_total_days"] = len(act)

    # Historical Samsung Health peak capacity baseline
    samsung = data.get("samsung_steps")
    if samsung is not None and not samsung.empty:
        aug23 = samsung[(samsung["date"] >= "2023-08-01") & (samsung["date"] < "2023-09-01")]
        if not aug23.empty:
            s["pre_dx_peak_steps"] = int(aug23["steps"].max())
            s["pre_dx_days_over_10k"] = int((aug23["steps"] >= 10000).sum())
            if "steps_mean" in s and s["pre_dx_peak_steps"] > 0:
                s["steps_decline_pct"] = 100 * (1 - s["steps_mean"] / s["pre_dx_peak_steps"])

    # SpO2
    spo = data["spo2"]
    if not spo.empty:
        s["spo2_mean"] = spo["spo2_average"].mean()
        s["spo2_min"] = spo["spo2_average"].min()

    # Cardiovascular age
    cva = data["cv_age"]
    if not cva.empty:
        s["cv_age_mean"] = cva["vascular_age"].mean()
        s["cv_age_min"] = cva["vascular_age"].min()
        s["cv_age_max"] = cva["vascular_age"].max()
        s["cv_age_total_days"] = len(cva)

    # Resilience
    res = data["resilience"]
    if not res.empty:
        s["resilience_all_limited"] = (res["level"] == "limited").all()
        s["resilience_sleep_recovery_mean"] = res["contributors_sleep_recovery"].mean()
        s["resilience_daytime_recovery_mean"] = res["contributors_daytime_recovery"].mean()
        s["resilience_stress_mean"] = res["contributors_stress"].mean()

    return s


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

## Removed: fig_executive_summary (Plotly indicators) — replaced by make_kpi_card HTML cards


def fig_hrv_deep_dive(data: dict) -> go.Figure:
    """HRV deep dive: trends, distribution, Poincaré, circadian."""
    hrv = data["oura_hrv"]
    if hrv.empty:
        fig = go.Figure()
        fig.add_annotation(text="No HRV data available", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "RMSSD Daily Trend (5-min samples)",
            "RMSSD Distribution (histogram)",
            "Poincare Plot (RMSSD Epochs, SD1/SD2)",
            "Circadian Pattern (hourly)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # --- Panel 1: Daily trend ---
    daily = hrv.groupby("date")["rmssd"].agg(["mean", "min", "max", "std"]).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    # Gradient fill below RMSSD line
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["mean"],
        mode="lines", name="_fill",
        line=dict(width=0), showlegend=False,
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
        hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["mean"],
        mode="lines+markers", name="Daily Mean RMSSD",
        line=dict(color=C_CRITICAL, width=2),
        marker=dict(size=4, line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d}</b><br>RMSSD: %{y:.1f} ms<extra></extra>",
    ), row=1, col=1)

    # Min/max range
    fig.add_trace(go.Scatter(
        x=pd.concat([daily["date"], daily["date"][::-1]]),
        y=pd.concat([daily["max"], daily["min"][::-1]]),
        fill="toself", fillcolor="rgba(239,68,68,0.06)",
        line=dict(width=0), showlegend=True, name="Min-Max Range",
        hoverinfo="skip",
    ), row=1, col=1)

    if len(daily) >= 7:
        daily["roll7"] = daily["mean"].rolling(7, min_periods=3).mean()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["roll7"],
            mode="lines", name="7-day Average",
            line=dict(color=C_DARK, width=2.5, shape="spline"),
            hovertemplate="<b>%{x|%b %d}</b><br>7d avg: %{y:.1f} ms<extra></extra>",
        ), row=1, col=1)

    # Population normal range as subtle band
    fig.add_hrect(y0=NORM_RMSSD_P25, y1=NORM_RMSSD_P75, fillcolor=ACCENT_GREEN, opacity=0.06,
                  line_width=0, row=1, col=1)
    fig.add_annotation(
        text=f"Reference IQR ({NORM_RMSSD_P25}-{NORM_RMSSD_P75} ms)",
        xref="x domain", yref="y", x=0.98, y=(NORM_RMSSD_P25 + NORM_RMSSD_P75) / 2,
        showarrow=False, font=dict(size=10, color=ACCENT_GREEN),
        xanchor="right", opacity=0.7, row=1, col=1,
    )

    # Low-HRV reference threshold
    fig.add_hline(y=ESC_RMSSD_DEFICIENCY, line_dash="dash", line_color=ACCENT_RED, line_width=1,
                  row=1, col=1)
    fig.add_annotation(
        text=f"Low-HRV reference <{ESC_RMSSD_DEFICIENCY} ms",
        xref="x domain", yref="y", x=0.02, y=ESC_RMSSD_DEFICIENCY,
        showarrow=False, font=dict(size=10, color=ACCENT_RED),
        xanchor="left", yanchor="bottom", row=1, col=1,
    )

    # Treatment start line (ruxolitinib)
    fig.add_vline(x=pd.Timestamp(TREATMENT_START), line_dash="dash",
                  line_color=ACCENT_BLUE, line_width=1.5, opacity=0.7, row=1, col=1)
    fig.add_annotation(
        text="Ruxolitinib", x=pd.Timestamp(TREATMENT_START), yref="y domain", y=0.95,
        showarrow=False, font=dict(size=10, color=ACCENT_BLUE),
        textangle=-90, xanchor="right", row=1, col=1,
    )

    # --- Panel 2: Distribution ---
    fig.add_trace(go.Histogram(
        x=hrv["rmssd"], nbinsx=50, name="RMSSD Distribution",
        marker_color=C_CRITICAL, opacity=0.8,
        marker_line=dict(color="rgba(239,68,68,0.3)", width=0.5),
        hovertemplate="<b>%{x:.0f} ms</b><br>Count: %{y} samples<extra></extra>",
    ), row=1, col=2)

    fig.add_vline(x=hrv["rmssd"].mean(), line_dash="solid", line_color=C_DARK,
                  annotation_text=f"Mean {hrv['rmssd'].mean():.1f}", row=1, col=2)
    fig.add_vline(x=hrv["rmssd"].median(), line_dash="dot", line_color=C_BLUE,
                  annotation_text=f"Median {hrv['rmssd'].median():.0f}", row=1, col=2)
    fig.add_vline(x=ESC_RMSSD_DEFICIENCY, line_dash="dash", line_color=ACCENT_RED,
                  annotation_text="Low-HRV reference", row=1, col=2)

    # --- Panel 3: Poincaré plot ---
    rmssd_vals = hrv.sort_values("timestamp")["rmssd"].values
    if len(rmssd_vals) > 1:
        x_poincare = rmssd_vals[:-1]
        y_poincare = rmssd_vals[1:]

        fig.add_trace(go.Scattergl(
            x=x_poincare, y=y_poincare,
            mode="markers", name="Poincare",
            marker=dict(size=2, color=C_BLUE, opacity=0.35),
            hovertemplate="<b>Epoch n</b>: %{x:.0f} ms<br><b>Epoch n+1</b>: %{y:.0f} ms<extra></extra>",
        ), row=2, col=1)

        # SD1/SD2 calculation
        diff = y_poincare - x_poincare
        sd1 = np.std(diff) / np.sqrt(2)
        sd2 = np.sqrt(2 * np.std(rmssd_vals) ** 2 - sd1 ** 2) if 2 * np.std(rmssd_vals) ** 2 > sd1 ** 2 else 0

        fig.add_annotation(
            text=f"SD1={sd1:.1f} | SD2={sd2:.1f} | SD1/SD2={sd1/sd2:.2f}" if sd2 > 0 else f"SD1={sd1:.1f}",
            xref="x3 domain", yref="y3 domain",
            x=0.95, y=0.95, showarrow=False,
            font=dict(size=11, color=TEXT_PRIMARY),
            bgcolor=BG_ELEVATED,
        )

        # Identity line
        max_val = max(x_poincare.max(), y_poincare.max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", line=dict(color=TEXT_TERTIARY, dash="dot", width=1),
            showlegend=False,
        ), row=2, col=1)

    # --- Panel 4: Circadian pattern ---
    hourly = hrv.groupby("hour")["rmssd"].agg(["mean", "std", "count"]).reset_index()

    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["mean"],
        name="RMSSD per Hour",
        marker_color=[C_OK if v > 10 else C_CRITICAL for v in hourly["mean"]],
        marker_line=dict(color="rgba(255,255,255,0.08)", width=0.5),
        error_y=dict(type="data", array=hourly["std"], visible=True,
                     color="rgba(255,255,255,0.25)", thickness=1.5),
        hovertemplate="<b>%{x}:00</b><br>Mean: %{y:.1f} ms<br>Samples: %{customdata}<extra></extra>",
        customdata=hourly["count"],
    ), row=2, col=2)

    fig.add_hline(y=ESC_RMSSD_DEFICIENCY, line_dash="dash", line_color=ACCENT_RED, row=2, col=2)

    fig.update_xaxes(title_text="Date", tickformat="%d %b", row=1, col=1,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(title_text="RMSSD (ms)", row=1, col=2)
    fig.update_xaxes(title_text="RMSSD epoch n (ms)", row=2, col=1)
    fig.update_xaxes(title_text="Hour (0-23)", row=2, col=2)
    fig.update_yaxes(title_text="RMSSD (ms)", zeroline=False, row=1, col=1,
                     gridcolor="rgba(255,255,255,0.05)", gridwidth=1)
    fig.update_yaxes(title_text="Count", zeroline=False, row=1, col=2,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="RMSSD epoch n+1 (ms)", zeroline=False, row=2, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="RMSSD (ms)", zeroline=False, row=2, col=2,
                     gridcolor="rgba(255,255,255,0.05)")

    fig.update_layout(
        height=920, showlegend=True,
        margin=dict(l=64, r=34, t=124, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                    font=dict(size=12)),
    )
    return fig


def fig_heart_rate(data: dict) -> go.Figure:
    """Heart rate: daily trends, circadian, tachycardia distribution."""
    hr = data["oura_hr"]
    if hr.empty:
        fig = go.Figure()
        fig.add_annotation(text="No HR data available", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Daily Heart Rate (mean/min/max)",
            "Circadian Rhythm (hourly)",
            "Heart Rate Distribution",
            "Tachycardia Threshold Analysis",
        ),
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    # --- Daily trend ---
    daily = hr.groupby("date")["bpm"].agg(["mean", "min", "max"]).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    # Subtle tachycardia/bradycardia zones
    fig.add_hrect(y0=100, y1=150, fillcolor=ACCENT_RED, opacity=0.03,
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=90, y1=100, fillcolor=ACCENT_AMBER, opacity=0.03,
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=60, y1=80, fillcolor=ACCENT_GREEN, opacity=0.04,
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=30, y1=60, fillcolor=ACCENT_CYAN, opacity=0.03,
                  line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["mean"],
        mode="lines", name="Daily Mean HR",
        line=dict(color=C_CRITICAL, width=2),
        hovertemplate="<b>%{x|%b %d}</b><br>Mean HR: %{y:.0f} bpm<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pd.concat([daily["date"], daily["date"][::-1]]),
        y=pd.concat([daily["max"], daily["min"][::-1]]),
        fill="toself", fillcolor="rgba(239,68,68,0.06)",
        line=dict(width=0), showlegend=True, name="Min-Max Range",
        hoverinfo="skip",
    ), row=1, col=1)

    fig.add_hline(y=NOCTURNAL_HR_ELEVATED, line_dash="dash", line_color=ACCENT_AMBER, line_width=1,
                  row=1, col=1)
    fig.add_annotation(
        text=f"Nocturnal concern {NOCTURNAL_HR_ELEVATED} bpm", xref="x domain", yref="y",
        x=0.02, y=NOCTURNAL_HR_ELEVATED, showarrow=False,
        font=dict(size=10, color=ACCENT_AMBER), xanchor="left", yanchor="bottom",
        row=1, col=1,
    )

    # Treatment start
    fig.add_vline(x=pd.Timestamp(TREATMENT_START), line_dash="dash",
                  line_color=ACCENT_BLUE, line_width=1.5, opacity=0.7, row=1, col=1)

    # --- Circadian ---
    hourly = hr.groupby("hour")["bpm"].agg(["mean", "min", "max", "count"]).reset_index()

    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["mean"],
        mode="lines+markers", name="HR per Hour",
        line=dict(color=C_CRITICAL, width=2, shape="spline"),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.06)",
        marker=dict(size=5, line=dict(width=0)),
        hovertemplate="<b>%{x}:00</b><br>Mean HR: %{y:.0f} bpm<br>Samples: %{customdata}<extra></extra>",
        customdata=hourly["count"],
    ), row=1, col=2)

    fig.add_hline(y=NOCTURNAL_HR_ELEVATED, line_dash="dash", line_color=ACCENT_AMBER,
                  line_width=1, row=1, col=2)

    # --- Distribution ---
    fig.add_trace(go.Histogram(
        x=hr["bpm"], nbinsx=60, name="HR Distribution",
        marker_color=C_BLUE, opacity=0.8,
        marker_line=dict(color="rgba(59,130,246,0.3)", width=0.5),
        hovertemplate="<b>%{x:.0f} bpm</b><br>Count: %{y}<extra></extra>",
    ), row=2, col=1)

    fig.add_vline(x=hr["bpm"].mean(), line_dash="solid", line_color=C_DARK,
                  annotation_text=f"Mean {hr['bpm'].mean():.0f}", row=2, col=1)
    fig.add_vline(x=NOCTURNAL_HR_ELEVATED, line_dash="dash", line_color=ACCENT_AMBER,
                  annotation_text=f"Nocturnal concern {NOCTURNAL_HR_ELEVATED}", row=2, col=1)
    fig.add_vline(x=100, line_dash="dash", line_color=ACCENT_RED,
                  annotation_text="Tachycardia 100", row=2, col=1)

    # --- Tachycardia zone breakdown ---
    zones = {
        "<60 (bradycardia)": (hr["bpm"] < 60).sum(),
        "60-80 (normal)": ((hr["bpm"] >= 60) & (hr["bpm"] < 80)).sum(),
        "80-90 (elevated)": ((hr["bpm"] >= 80) & (hr["bpm"] < 90)).sum(),
        "90-100 (high)": ((hr["bpm"] >= 90) & (hr["bpm"] < 100)).sum(),
        "100-120 (tachycardia)": ((hr["bpm"] >= 100) & (hr["bpm"] < 120)).sum(),
        ">120 (severe)": (hr["bpm"] >= 120).sum(),
    }
    total = len(hr)
    colors = [ACCENT_CYAN, C_OK, C_CAUTION, C_WARNING, C_CRITICAL, ACCENT_RED]

    fig.add_trace(go.Bar(
        x=list(zones.keys()),
        y=[v / total * 100 for v in zones.values()],
        marker_color=colors,
        marker_line=dict(color="rgba(255,255,255,0.1)", width=1),
        text=[f"{v/total*100:.1f}%" for v in zones.values()],
        textposition="outside", textfont=dict(size=11),
        name="HR Zones",
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>%{y:.1f}% (%{customdata:,} samples)<extra></extra>",
        customdata=list(zones.values()),
    ), row=2, col=2)

    fig.update_xaxes(title_text="Date", tickformat="%d %b", row=1, col=1,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(title_text="Hour (0-23)", row=1, col=2)
    fig.update_xaxes(title_text="Heart Rate (bpm)", row=2, col=1)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=11), row=2, col=2)
    fig.update_yaxes(title_text="BPM", zeroline=False, row=1, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="BPM", zeroline=False, row=1, col=2,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Count", zeroline=False, row=2, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="% of Samples", zeroline=False, row=2, col=2,
                     gridcolor="rgba(255,255,255,0.05)")

    fig.update_layout(
        height=920, showlegend=True,
        margin=dict(l=64, r=34, t=124, b=74),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                    font=dict(size=12)),
    )
    return fig


def fig_sleep_analysis(data: dict) -> go.Figure:
    """Sleep analysis from oura_sleep + sleep_periods."""
    sl = data["oura_sleep"]
    sp = data["sleep_periods"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Sleep Score (daily)",
            "Sleep Duration & Efficiency",
            "Heart Rate During Sleep",
            "HRV & Respiratory Rate During Sleep",
        ),
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    # --- Panel 1: Sleep score ---
    if not sl.empty:
        colors = sl["score"].apply(
            lambda v: C_CRITICAL if v < 50 else (C_WARNING if v < 60 else (C_CAUTION if v < 70 else C_OK))
        )
        fig.add_trace(go.Bar(
            x=sl["date"], y=sl["score"],
            name="Sleep Score",
            marker_color=colors,
            marker_line=dict(color="rgba(255,255,255,0.08)", width=0.5),
            hovertemplate="<b>%{x|%b %d}</b><br>Sleep Score: %{y}<extra></extra>",
        ), row=1, col=1)

        if len(sl) >= 7:
            roll = sl.sort_values("date")["score"].rolling(7, min_periods=3).mean()
            fig.add_trace(go.Scatter(
                x=sl.sort_values("date")["date"], y=roll,
                mode="lines", name="7-day Average",
                line=dict(color=C_DARK, width=2.5, shape="spline"),
                hovertemplate="<b>%{x|%b %d}</b><br>7d avg: %{y:.0f}<extra></extra>",
            ), row=1, col=1)

        fig.add_hline(y=70, line_dash="dot", line_color=TEXT_TERTIARY, line_width=1,
                      row=1, col=1)
        fig.add_annotation(
            text="Good sleep (70)", xref="x domain", yref="y",
            x=0.98, y=70, showarrow=False, font=dict(size=10, color=TEXT_TERTIARY),
            xanchor="right", yanchor="bottom", row=1, col=1,
        )

    # --- Panel 2: Duration & efficiency from sleep_periods ---
    long_sleep = sp[sp["type"] == "long_sleep"].copy() if not sp.empty else pd.DataFrame()
    if not long_sleep.empty:
        long_sleep["hours"] = long_sleep["total_sleep_duration"] / 3600
        long_sleep = long_sleep.sort_values("day")

        fig.add_trace(go.Bar(
            x=long_sleep["day"], y=long_sleep["hours"],
            name="Sleep Hours",
            marker_color=C_BLUE, opacity=0.7,
            marker_line=dict(color="rgba(59,130,246,0.2)", width=0.5),
            hovertemplate="<b>%{x|%b %d}</b><br>Duration: %{y:.1f} hours<extra></extra>",
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=long_sleep["day"], y=long_sleep["efficiency"],
            mode="lines+markers", name="Efficiency %",
            line=dict(color=C_OK, width=2.5),
            marker=dict(size=4, line=dict(width=0)),
            yaxis="y4",
            hovertemplate="<b>%{x|%b %d}</b><br>Efficiency: %{y}%<extra></extra>",
        ), row=1, col=2)

        fig.add_hline(y=7, line_dash="dot", line_color=TEXT_TERTIARY, line_width=1,
                      row=1, col=2)
        fig.add_annotation(
            text="7 hrs recommended", xref="x2 domain", yref="y2",
            x=0.98, y=7, showarrow=False, font=dict(size=10, color=TEXT_TERTIARY),
            xanchor="right", yanchor="bottom", row=1, col=2,
        )

    # --- Panel 3: HR during sleep ---
    if not long_sleep.empty:
        fig.add_trace(go.Scatter(
            x=long_sleep["day"], y=long_sleep["average_heart_rate"],
            mode="lines+markers", name="Avg HR (sleep)",
            line=dict(color=C_CRITICAL, width=2),
            marker=dict(size=4, line=dict(width=0)),
            hovertemplate="<b>%{x|%b %d}</b><br>Avg HR: %{y:.0f} bpm<extra></extra>",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=long_sleep["day"], y=long_sleep["lowest_heart_rate"],
            mode="lines+markers", name="Lowest HR",
            line=dict(color=C_BLUE, width=2),
            marker=dict(size=4, line=dict(width=0)),
            hovertemplate="<b>%{x|%b %d}</b><br>Lowest HR: %{y:.0f} bpm<extra></extra>",
        ), row=2, col=1)

        fig.add_hline(y=80, line_dash="dot", line_color=ACCENT_AMBER, line_width=1,
                      row=2, col=1)
        fig.add_annotation(
            text="Elevated for sleep", xref="x3 domain", yref="y3",
            x=0.98, y=80, showarrow=False, font=dict(size=10, color=ACCENT_AMBER),
            xanchor="right", yanchor="bottom", row=2, col=1,
        )

    # --- Panel 4: HRV & Breath during sleep ---
    if not long_sleep.empty:
        fig.add_trace(go.Bar(
            x=long_sleep["day"], y=long_sleep["average_hrv"],
            name="Sleep HRV",
            marker_color=C_CRITICAL,
            marker_line=dict(color="rgba(239,68,68,0.2)", width=0.5),
            hovertemplate="<b>%{x|%b %d}</b><br>Sleep HRV: %{y:.0f} ms<extra></extra>",
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=long_sleep["day"], y=long_sleep["average_breath"],
            mode="lines+markers", name="Respiratory Rate",
            line=dict(color=C_OK, width=2),
            marker=dict(size=4, line=dict(width=0)),
            hovertemplate="<b>%{x|%b %d}</b><br>Breath: %{y:.1f} /min<extra></extra>",
        ), row=2, col=2)

        fig.add_hline(y=ESC_RMSSD_DEFICIENCY, line_dash="dash", line_color=ACCENT_RED,
                      line_width=1, row=2, col=2)
        fig.add_annotation(
            text=f"Deficiency {ESC_RMSSD_DEFICIENCY} ms",
            xref="x4 domain", yref="y4", x=0.02, y=ESC_RMSSD_DEFICIENCY,
            showarrow=False, font=dict(size=10, color=ACCENT_RED),
            xanchor="left", yanchor="bottom", row=2, col=2,
        )

    fig.update_xaxes(tickformat="%d %b", row=1, col=1,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=1, col=2,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=2, col=1,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=2, col=2,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_yaxes(title_text="Score", zeroline=False, row=1, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Hours / %", zeroline=False, row=1, col=2,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="BPM", zeroline=False, row=2, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="ms / breaths per min", zeroline=False, row=2, col=2,
                     gridcolor="rgba(255,255,255,0.05)")

    fig.update_layout(
        height=940, showlegend=True,
        margin=dict(l=64, r=34, t=124, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                    font=dict(size=12)),
    )
    return fig


def fig_readiness_recovery(data: dict) -> go.Figure:
    """Readiness scores, HRV balance, recovery index, temperature."""
    r = data["readiness"]
    if r.empty:
        fig = go.Figure()
        fig.add_annotation(text="No readiness data available", x=0.5, y=0.5, showarrow=False)
        return fig

    r = r.sort_values("date")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Readiness Score & Recovery Index",
            "HRV Balance & Sleep Balance",
            "Lowest Nocturnal HR (from sleep periods, actual bpm)",
            "Temperature Deviation",
        ),
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    # --- Readiness + Recovery ---
    fig.add_trace(go.Bar(
        x=r["date"], y=r["score"],
        name="Readiness Score",
        marker_color=r["score"].apply(
            lambda v: C_CRITICAL if v < 50 else (C_WARNING if v < 60 else (C_CAUTION if v < 70 else C_OK))
        ),
        marker_line=dict(color="rgba(255,255,255,0.08)", width=0.5),
        hovertemplate="<b>%{x|%b %d}</b><br>Readiness: %{y}/100<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=r["date"], y=r["recovery_index"],
        mode="lines+markers", name="Recovery Index",
        line=dict(color=C_DARK, width=2),
        marker=dict(size=4, line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d}</b><br>Recovery: %{y}<extra></extra>",
    ), row=1, col=1)

    # --- HRV Balance + Sleep Balance ---
    fig.add_trace(go.Scatter(
        x=r["date"], y=r["hrv_balance"],
        mode="lines+markers", name="HRV Balance",
        line=dict(color=C_CRITICAL, width=2),
        marker=dict(size=5,
                    color=r["hrv_balance"].apply(lambda v: C_CRITICAL if pd.notna(v) and v < 20 else C_BLUE),
                    line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d}</b><br>HRV Balance: %{y}/100<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=r["date"], y=r["sleep_balance"],
        mode="lines+markers", name="Sleep Balance",
        line=dict(color=C_OK, width=2),
        marker=dict(size=4, line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d}</b><br>Sleep Balance: %{y}/100<extra></extra>",
    ), row=1, col=2)

    fig.add_hline(y=50, line_dash="dot", line_color=TEXT_TERTIARY, line_width=1, row=1, col=2)
    fig.add_hrect(y0=0, y1=25, fillcolor=ACCENT_RED, opacity=0.04, line_width=0, row=1, col=2)

    # --- Lowest Nocturnal HR (from sleep_periods, actual bpm) ---
    # CRITICAL: oura_readiness.resting_heart_rate is a CONTRIBUTOR SCORE (0-100), NOT bpm!
    # Use sleep_periods.lowest_heart_rate for actual nocturnal HR minimum.
    sp = data.get("sleep_periods", pd.DataFrame())
    if not sp.empty and "lowest_heart_rate" in sp.columns:
        sp_long = sp[(sp["type"] == "long_sleep") & sp["lowest_heart_rate"].notna()].copy()
        sp_long["day"] = pd.to_datetime(sp_long["day"])
        if not sp_long.empty:
            fig.add_trace(go.Bar(
                x=sp_long["day"], y=sp_long["lowest_heart_rate"],
                name="Lowest Nocturnal HR",
                marker_color=sp_long["lowest_heart_rate"].apply(
                    lambda v: C_CRITICAL if v > 80 else (C_WARNING if v > NOCTURNAL_HR_DIP_NORMAL_HIGH else C_OK)
                ),
                marker_line=dict(color="rgba(255,255,255,0.08)", width=0.5),
                hovertemplate="<b>%{x|%b %d}</b><br>Lowest HR: %{y} bpm<extra></extra>",
            ), row=2, col=1)

            # Normal nocturnal dip zone
            fig.add_hrect(
                y0=NOCTURNAL_HR_DIP_NORMAL_LOW, y1=NOCTURNAL_HR_DIP_NORMAL_HIGH,
                fillcolor=ACCENT_GREEN, opacity=0.08, line_width=0,
                annotation_text=f"Normal nocturnal dip ({NOCTURNAL_HR_DIP_NORMAL_LOW}-{NOCTURNAL_HR_DIP_NORMAL_HIGH})",
                row=2, col=1,
            )
            avg_lowest = sp_long["lowest_heart_rate"].mean()
            fig.add_hline(y=avg_lowest, line_dash="solid", line_color=C_DARK,
                          annotation_text=f"Mean lowest {avg_lowest:.0f} bpm", row=2, col=1)

    # --- Temperature deviation ---
    td = r.dropna(subset=["temperature_deviation"])
    if not td.empty:
        fig.add_trace(go.Scatter(
            x=td["date"], y=td["temperature_deviation"],
            mode="lines+markers", name="Temp Deviation",
            line=dict(color=C_BLUE, width=2),
            marker=dict(size=4,
                        color=td["temperature_deviation"].apply(
                            lambda v: C_CRITICAL if abs(v) > 0.5 else C_OK
                        ),
                        line=dict(width=0)),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
            hovertemplate="<b>%{x|%b %d}</b><br>Temp: %{y:+.2f} C<extra></extra>",
        ), row=2, col=2)

        fig.add_hline(y=0, line_dash="solid", line_color=TEXT_TERTIARY, row=2, col=2)

    fig.update_xaxes(tickformat="%d %b", row=1, col=1,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=1, col=2,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=2, col=1,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=2, col=2,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_yaxes(title_text="Score", zeroline=False, row=1, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Score (0-100)", zeroline=False, row=1, col=2,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="BPM", zeroline=False, row=2, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Temperature (°C)", zeroline=False, row=2, col=2,
                     gridcolor="rgba(255,255,255,0.05)")

    fig.update_layout(
        height=940, showlegend=True,
        margin=dict(l=64, r=34, t=124, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                    font=dict(size=12)),
    )
    return fig


def fig_resilience_cv_age(data: dict) -> go.Figure:
    """Resilience contributors + cardiovascular age trend."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Resilience Contributors (all days: 'limited')",
            "Vascular Age vs Chronological Age",
        ),
        horizontal_spacing=0.12,
    )

    # --- Resilience ---
    res = data["resilience"]
    if not res.empty:
        res = res.sort_values("date")
        fig.add_trace(go.Scatter(
            x=res["date"], y=res["contributors_sleep_recovery"],
            mode="lines+markers", name="Sleep Recovery",
            line=dict(color=C_BLUE, width=2), marker=dict(size=4, line=dict(width=0)),
            hovertemplate="<b>%{x|%b %d}</b><br>Sleep Recovery: %{y:.1f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=res["date"], y=res["contributors_daytime_recovery"],
            mode="lines+markers", name="Daytime Recovery",
            line=dict(color=C_OK, width=2), marker=dict(size=4, line=dict(width=0)),
            hovertemplate="<b>%{x|%b %d}</b><br>Daytime Recovery: %{y:.1f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=res["date"], y=res["contributors_stress"],
            mode="lines+markers", name="Stress Mgmt",
            line=dict(color=C_WARNING, width=2), marker=dict(size=4, line=dict(width=0)),
            hovertemplate="<b>%{x|%b %d}</b><br>Stress Mgmt: %{y:.1f}<extra></extra>",
        ), row=1, col=1)

    # --- Cardiovascular Age ---
    cva = data["cv_age"]
    if not cva.empty:
        cva = cva.sort_values("date")
        fig.add_trace(go.Scatter(
            x=cva["date"], y=cva["vascular_age"],
            mode="lines+markers", name="Vascular Age",
            line=dict(color=C_CRITICAL, width=2),
            marker=dict(size=6, line=dict(width=0)),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.05)",
            hovertemplate="<b>%{x|%b %d}</b><br>Vascular Age: %{y:.0f} yr<extra></extra>",
        ), row=1, col=2)

        fig.add_hline(y=cva["vascular_age"].mean(), line_dash="dash", line_color=C_WARNING,
                      annotation_text=f"Mean vascular ({cva['vascular_age'].mean():.0f})", row=1, col=2)

    fig.update_xaxes(tickformat="%d %b", row=1, col=1,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=1, col=2,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_yaxes(title_text="Score", zeroline=False, row=1, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Age (years)", zeroline=False, row=1, col=2,
                     gridcolor="rgba(255,255,255,0.05)")

    fig.update_layout(
        height=520, showlegend=True,
        margin=dict(l=64, r=34, t=118, b=72),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
                    font=dict(size=12)),
    )
    return fig


def fig_spo2_stress(data: dict) -> go.Figure:
    """SpO2 monitoring + stress/recovery balance."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "SpO2 Average (nocturnal)",
            "Stress vs Recovery (minutes)",
        ),
        horizontal_spacing=0.12,
    )

    # --- SpO2 ---
    spo = data["spo2"]
    if not spo.empty:
        spo = spo.sort_values("date")
        fig.add_trace(go.Scatter(
            x=spo["date"], y=spo["spo2_average"],
            mode="lines+markers", name="SpO2",
            line=dict(color=C_BLUE, width=2),
            marker=dict(size=5,
                        color=spo["spo2_average"].apply(
                            lambda v: C_CRITICAL if v < 94 else (C_WARNING if v < 95.5 else C_OK)
                        ),
                        line=dict(width=0)),
            hovertemplate="<b>%{x|%b %d}</b><br>SpO2: %{y:.1f}%<extra></extra>",
        ), row=1, col=1)

        # Subtle warning zones
        fig.add_hrect(y0=93, y1=94, fillcolor=ACCENT_RED, opacity=0.04, line_width=0, row=1, col=1)
        fig.add_hrect(y0=94, y1=95, fillcolor=ACCENT_AMBER, opacity=0.03, line_width=0, row=1, col=1)

        fig.add_hline(y=95, line_dash="dash", line_color=ACCENT_AMBER, line_width=1,
                      row=1, col=1)
        fig.add_annotation(
            text="95% lower limit", xref="x domain", yref="y",
            x=0.02, y=95, showarrow=False, font=dict(size=10, color=ACCENT_AMBER),
            xanchor="left", yanchor="bottom", row=1, col=1,
        )
        fig.add_hline(y=94, line_dash="dash", line_color=ACCENT_RED, line_width=1,
                      row=1, col=1)
        fig.add_annotation(
            text="94% critical", xref="x domain", yref="y",
            x=0.02, y=94, showarrow=False, font=dict(size=10, color=ACCENT_RED),
            xanchor="left", yanchor="bottom", row=1, col=1,
        )

    # --- Stress/Recovery ---
    stress = data["stress"]
    if not stress.empty:
        stress = stress.sort_values("date")
        fig.add_trace(go.Bar(
            x=stress["date"], y=stress["stress_min"],
            name="Stress", marker_color=C_CRITICAL, opacity=0.7,
            marker_line=dict(color="rgba(239,68,68,0.2)", width=0.5),
            hovertemplate="<b>%{x|%b %d}</b><br>Stress: %{y:.0f} min<extra></extra>",
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=stress["date"], y=stress["recovery_min"],
            name="Recovery", marker_color=C_OK, opacity=0.7,
            marker_line=dict(color="rgba(16,185,129,0.2)", width=0.5),
            hovertemplate="<b>%{x|%b %d}</b><br>Recovery: %{y:.0f} min<extra></extra>",
        ), row=1, col=2)

    fig.update_yaxes(title_text="SpO2 %", range=[93, 98], zeroline=False, row=1, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Minutes", zeroline=False, row=1, col=2,
                     gridcolor="rgba(255,255,255,0.05)")

    fig.update_xaxes(tickformat="%d %b", row=1, col=1,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=1, col=2,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")

    fig.update_layout(
        height=520, showlegend=True, barmode="group",
        margin=dict(l=64, r=34, t=118, b=72),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
                    font=dict(size=12)),
    )
    return fig


def fig_activity(data: dict) -> go.Figure:
    """Activity: steps, scores, calories, time breakdown."""
    act = data["activity"]
    if act.empty:
        fig = go.Figure()
        fig.add_annotation(text="No activity data available", x=0.5, y=0.5, showarrow=False)
        return fig

    act = act.sort_values("date")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Daily Steps",
            "Activity Score",
            "Calorie Burn",
            "Time Distribution (hours)",
        ),
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    # --- Steps ---
    fig.add_trace(go.Bar(
        x=act["date"], y=act["steps"],
        name="Steps",
        marker_color=act["steps"].apply(
            lambda v: C_CRITICAL if v < 1000 else (
                C_WARNING if v < 2000 else (C_CAUTION if v < 5000 else C_OK))
        ),
        marker_line=dict(color="rgba(255,255,255,0.08)", width=0.5),
        hovertemplate="<b>%{x|%b %d}</b><br>Steps: %{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=5000, line_dash="dot", line_color=TEXT_TERTIARY, line_width=1,
                  row=1, col=1)
    fig.add_annotation(
        text="Sedentary (5,000)", xref="x domain", yref="y",
        x=0.98, y=5000, showarrow=False, font=dict(size=10, color=TEXT_TERTIARY),
        xanchor="right", yanchor="bottom", row=1, col=1,
    )
    fig.add_hline(y=2000, line_dash="dash", line_color=ACCENT_AMBER, line_width=1,
                  row=1, col=1)
    fig.add_annotation(
        text="Severe inactivity (2,000)", xref="x domain", yref="y",
        x=0.98, y=2000, showarrow=False, font=dict(size=10, color=ACCENT_AMBER),
        xanchor="right", yanchor="bottom", row=1, col=1,
    )

    # --- Activity Score ---
    fig.add_trace(go.Bar(
        x=act["date"], y=act["score"],
        name="Activity Score",
        marker_color=act["score"].apply(
            lambda v: C_CRITICAL if v < 50 else (C_WARNING if v < 60 else (C_CAUTION if v < 70 else C_OK))
        ),
        marker_line=dict(color="rgba(255,255,255,0.08)", width=0.5),
        hovertemplate="<b>%{x|%b %d}</b><br>Activity Score: %{y}<extra></extra>",
    ), row=1, col=2)

    if len(act) >= 7:
        roll = act["score"].rolling(7, min_periods=3).mean()
        fig.add_trace(go.Scatter(
            x=act["date"], y=roll,
            mode="lines", name="7-day Average",
            line=dict(color=C_DARK, width=2.5, shape="spline"),
            hovertemplate="<b>%{x|%b %d}</b><br>7d avg: %{y:.0f}<extra></extra>",
        ), row=1, col=2)

    # --- Calories ---
    fig.add_trace(go.Bar(
        x=act["date"], y=act["active_calories"],
        name="Active Cal", marker_color=C_WARNING,
        marker_line=dict(color="rgba(245,158,11,0.2)", width=0.5),
        hovertemplate="<b>%{x|%b %d}</b><br>Active: %{y:.0f} kcal<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=act["date"], y=act["total_calories"] - act["active_calories"],
        name="Basal Cal", marker_color=C_LIGHT,
        marker_line=dict(color="rgba(96,165,250,0.2)", width=0.5),
        hovertemplate="<b>%{x|%b %d}</b><br>Basal: %{y:.0f} kcal<extra></extra>",
    ), row=2, col=1)

    # --- Time breakdown (hours) ---
    fig.add_trace(go.Bar(
        x=act["date"], y=act["inactive_time"] / 3600,
        name="Inactive", marker_color=C_CRITICAL, opacity=0.7,
        marker_line=dict(color="rgba(255,255,255,0.05)", width=0.5),
        hovertemplate="<b>%{x|%b %d}</b><br>Inactive: %{y:.1f} hrs<extra></extra>",
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=act["date"], y=act["rest_time"] / 3600,
        name="Rest", marker_color=C_CAUTION, opacity=0.7,
        marker_line=dict(color="rgba(255,255,255,0.05)", width=0.5),
        hovertemplate="<b>%{x|%b %d}</b><br>Rest: %{y:.1f} hrs<extra></extra>",
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=act["date"], y=act["low_activity_time"] / 3600,
        name="Low Activity", marker_color=C_BLUE, opacity=0.7,
        marker_line=dict(color="rgba(255,255,255,0.05)", width=0.5),
        hovertemplate="<b>%{x|%b %d}</b><br>Low Activity: %{y:.1f} hrs<extra></extra>",
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=act["date"], y=(act["medium_activity_time"] + act["high_activity_time"]) / 3600,
        name="Med/High", marker_color=C_OK, opacity=0.7,
        marker_line=dict(color="rgba(255,255,255,0.05)", width=0.5),
        hovertemplate="<b>%{x|%b %d}</b><br>Med/High: %{y:.1f} hrs<extra></extra>",
    ), row=2, col=2)

    fig.update_xaxes(tickformat="%d %b", row=1, col=1,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=1, col=2,
                     showspikes=True, spikemode="across", spikethickness=0.5,
                     spikecolor=TEXT_TERTIARY, spikedash="dot")
    fig.update_xaxes(tickformat="%d %b", row=2, col=1)
    fig.update_xaxes(tickformat="%d %b", row=2, col=2)
    fig.update_yaxes(title_text="Steps", zeroline=False, row=1, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Score", zeroline=False, row=1, col=2,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Calories (kcal)", zeroline=False, row=2, col=1,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Hours", zeroline=False, row=2, col=2,
                     gridcolor="rgba(255,255,255,0.05)")

    fig.update_layout(
        height=940, showlegend=True,
        barmode="stack",
        margin=dict(l=64, r=34, t=124, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                    font=dict(size=12)),
    )
    return fig


def fig_rmssd_comparison_bar(stats: dict) -> go.Figure:
    """Reference comparison bar chart: patient vs norms."""
    patient_rmssd = stats.get("rmssd_mean", 8.5)

    cats = [
        f"Observed Mean<br>({patient_rmssd:.1f} ms)",
        f"Low-HRV Reference<br>(<{ESC_RMSSD_DEFICIENCY} ms)",
        f"Reference IQR<br>({NORM_RMSSD_P25}-{NORM_RMSSD_P75} ms)",
    ]
    vals = [patient_rmssd, ESC_RMSSD_DEFICIENCY, NORM_RMSSD_P50]
    errs = [0, 0, (NORM_RMSSD_P75 - NORM_RMSSD_P25) / 2]
    colors = [C_CRITICAL, C_WARNING, C_OK]

    fig = go.Figure(go.Bar(
        x=cats, y=vals,
        marker_color=colors,
        marker_line=dict(color="rgba(255,255,255,0.1)", width=1),
        error_y=dict(type="data", array=errs, visible=True, color="rgba(255,255,255,0.3)",
                     thickness=1.5),
        text=[f"{v:.0f} ms" for v in vals],
        textposition="outside", textfont=dict(size=13),
        hovertemplate="<b>%{x}</b><br>RMSSD: %{y:.0f} ms<extra></extra>",
    ))

    fig.update_layout(
        height=460,
        margin=dict(l=64, r=34, t=92, b=64),
        yaxis_title="RMSSD (ms)",
        yaxis_range=[0, 70],
        yaxis=dict(zeroline=False, gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Narrative text
# ---------------------------------------------------------------------------

def _bar_pct(val: float, lo: float, hi: float) -> float:
    """Clamp a value to [0, 100] as a percentage of [lo, hi]."""
    if hi == lo:
        return 50.0
    return max(0.0, min(100.0, (val - lo) / (hi - lo) * 100.0))


def clinical_narrative(stats: dict) -> str:
    """Build Clinical Summary HTML using cs-* design system (v2)."""
    # ---- Extract & compute ------------------------------------------------
    rmssd = stats.get("rmssd_mean", 0)
    rhr = stats.get("resting_hr_primary", 0)
    cv_age_mean = stats.get("cv_age_mean", 0)
    cv_age_days = stats.get("cv_age_total_days", 0)
    cv_age_min = stats.get("cv_age_min", 0)
    cv_age_max = stats.get("cv_age_max", 0)
    hrv_balance = stats.get("hrv_balance_mean", 0)
    steps = stats.get("steps_mean", 0)
    sleep_score = stats.get("sleep_score_mean", 0)
    spo2 = stats.get("spo2_mean", 0)
    spo2_min = stats.get("spo2_min", 0)
    rmssd_samples = stats.get("rmssd_samples", 0)
    rmssd_days = stats.get("rmssd_total_days", 0)
    ist_met = stats.get("ist_met", False)

    pct_below_norm = 100 * (1 - rmssd / NORM_RMSSD_P50)

    # Bar positions (percentage along the axis)
    rmssd_bar_max = 80
    rmssd_marker = _bar_pct(rmssd, 0, rmssd_bar_max)
    rmssd_norm_left = _bar_pct(NORM_RMSSD_P25, 0, rmssd_bar_max)
    rmssd_norm_w = _bar_pct(NORM_RMSSD_P75, 0, rmssd_bar_max) - rmssd_norm_left

    hr_bar_lo, hr_bar_hi = 40, 110
    hr_marker = _bar_pct(rhr, hr_bar_lo, hr_bar_hi)
    hr_norm_left = _bar_pct(50, hr_bar_lo, hr_bar_hi)
    hr_norm_w = _bar_pct(SLEEP_AVG_HR_NORMAL, hr_bar_lo, hr_bar_hi) - hr_norm_left

    bal_marker = _bar_pct(hrv_balance, 0, 100)
    bal_norm_left = 50
    bal_norm_w = 30

    step_bar_max = 10000
    step_marker = _bar_pct(steps, 0, step_bar_max)
    step_norm_left = _bar_pct(5000, 0, step_bar_max)
    step_norm_w = _bar_pct(8000, 0, step_bar_max) - step_norm_left

    rhr_sev = "critical" if ist_met else "warning"
    steps_sev = "critical" if steps < 3000 else "warning"

    # Historical step context from Samsung Health baseline
    peak_steps = stats.get("pre_dx_peak_steps")
    decline_pct = stats.get("steps_decline_pct")
    if peak_steps and decline_pct:
        steps_history_ctx = (
            f'<div class="cs-bar-context" style="margin-top:4px;color:{C_CRITICAL};font-weight:500">'
            f'Peak Samsung Health baseline: {peak_steps:,} steps/day '
            f'— current Oura-window mean is {decline_pct:.0f}% lower'
            f'</div>'
        )
    else:
        steps_history_ctx = ""

    rmssd_reference_text = (
        f"Mean RMSSD is {rmssd:.1f} ms, {pct_below_norm:.0f}% below the reference median "
        f"({NORM_RMSSD_P50} ms)."
        if rmssd < NORM_RMSSD_P50 else
        f"Mean RMSSD is {rmssd:.1f} ms, within or above the reference median "
        f"({NORM_RMSSD_P50} ms)."
    )

    narrative = f"""
      <!-- Verdict banner -->
      <div class="cs-verdict">
        <div class="cs-verdict-dot"></div>
        <div class="cs-verdict-text">
          <strong>This summary describes the wearable signals observed in the current data window.</strong>
          It does not assert specific clinical history or diagnosis.
          {rmssd_reference_text}
        </div>
      </div>

      <!-- Deviation bar grid -->
      <div class="cs-dev-grid">
        <div class="cs-dev-card">
          <div class="cs-dev-header">
            <span class="cs-dev-label">RMSSD (HRV)</span>
            <span class="cs-dev-pct critical">{pct_below_norm:.0f}% below median</span>
          </div>
          <div class="cs-dev-value">{rmssd:.1f} <span class="unit">ms</span></div>
          <div class="cs-bar">
            <div class="cs-bar-normal" style="left:{rmssd_norm_left:.1f}%;width:{rmssd_norm_w:.1f}%"></div>
            <div class="cs-bar-marker critical" style="left:{rmssd_marker:.1f}%"></div>
          </div>
          <div class="cs-bar-scale"><span>0</span><span>{rmssd_bar_max} ms</span></div>
          <div class="cs-bar-context">Normal: {NORM_RMSSD_P25}-{NORM_RMSSD_P75} ms (P25-P75)</div>
        </div>

        <div class="cs-dev-card">
          <div class="cs-dev-header">
            <span class="cs-dev-label">Sleep HR</span>
            <span class="cs-dev-pct {rhr_sev}">{'Elevated' if ist_met else 'Borderline'}</span>
          </div>
          <div class="cs-dev-value">{rhr:.0f} <span class="unit">bpm</span></div>
          <div class="cs-bar">
            <div class="cs-bar-normal" style="left:{hr_norm_left:.1f}%;width:{hr_norm_w:.1f}%"></div>
            <div class="cs-bar-marker {rhr_sev}" style="left:{hr_marker:.1f}%"></div>
          </div>
          <div class="cs-bar-scale"><span>{hr_bar_lo}</span><span>{hr_bar_hi} bpm</span></div>
          <div class="cs-bar-context">Normal sleep HR: 50-{SLEEP_AVG_HR_NORMAL} bpm</div>
        </div>

        <div class="cs-dev-card">
          <div class="cs-dev-header">
            <span class="cs-dev-label">HRV Balance</span>
            <span class="cs-dev-pct critical">Critically low</span>
          </div>
          <div class="cs-dev-value">{hrv_balance:.0f} <span class="unit">/100</span></div>
          <div class="cs-bar">
            <div class="cs-bar-normal" style="left:{bal_norm_left}%;width:{bal_norm_w}%"></div>
            <div class="cs-bar-marker critical" style="left:{bal_marker:.1f}%"></div>
          </div>
          <div class="cs-bar-scale"><span>0</span><span>100</span></div>
          <div class="cs-bar-context">Normal: 50-80 (Oura readiness component)</div>
        </div>

        <div class="cs-dev-card">
          <div class="cs-dev-header">
            <span class="cs-dev-label">Daily Steps</span>
            <span class="cs-dev-pct {steps_sev}">{steps:,.0f} avg</span>
          </div>
          <div class="cs-dev-value">{steps:,.0f} <span class="unit">steps</span></div>
          <div class="cs-bar">
            <div class="cs-bar-normal" style="left:{step_norm_left:.1f}%;width:{step_norm_w:.1f}%"></div>
            <div class="cs-bar-marker {steps_sev}" style="left:{step_marker:.1f}%"></div>
          </div>
          <div class="cs-bar-scale"><span>0</span><span>{step_bar_max:,}</span></div>
          <div class="cs-bar-context">Normal: 5,000-8,000 steps/day</div>
          {steps_history_ctx}
        </div>
      </div>

      <!-- Findings grid -->
      <div class="cs-findings-grid">
        <div class="cs-finding">
          <div class="cs-finding-header">
            <span class="cs-finding-title">Sleep</span>
            <span class="cs-sev moderate">Moderate</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Score</span>
            <span class="cs-metric-val warning">{sleep_score:.0f}/100</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Duration</span>
            <span class="cs-metric-val">{stats.get('sleep_duration_avg_hrs', 0):.1f} hrs</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Poor days</span>
            <span class="cs-metric-val warning">{stats.get('sleep_days_poor', 0)}/{stats.get('sleep_total_days', 0)}</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Sleep HRV</span>
            <span class="cs-metric-val critical">{stats.get('sleep_hrv_mean', 0):.0f} ms</span>
          </div>
        </div>

        <div class="cs-finding">
          <div class="cs-finding-header">
            <span class="cs-finding-title">Physical Capacity</span>
            <span class="cs-sev severe">Severe</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Activity score</span>
            <span class="cs-metric-val warning">{stats.get('activity_score_mean', 0):.0f}/100</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Days &lt;2k steps</span>
            <span class="cs-metric-val critical">{stats.get('steps_days_under_2000', 0)}</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Resilience</span>
            <span class="cs-metric-val critical">{'All limited' if stats.get('resilience_all_limited') else 'Variable'}</span>
          </div>
        </div>

        <div class="cs-finding">
          <div class="cs-finding-header">
            <span class="cs-finding-title">SpO2</span>
            <span class="cs-sev low-normal">Low Normal</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Average</span>
            <span class="cs-metric-val">{spo2:.1f}%</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Lowest</span>
            <span class="cs-metric-val warning">{spo2_min:.1f}%</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">BOS risk</span>
            <span class="cs-metric-val">Borderline</span>
          </div>
        </div>

        <div class="cs-finding">
          <div class="cs-finding-header">
            <span class="cs-finding-title">Vascular Age Estimate</span>
            <span class="cs-sev low-normal">Estimate</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Mean estimate</span>
            <span class="cs-metric-val">{cv_age_mean:.0f} yr</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Observed range</span>
            <span class="cs-metric-val">{cv_age_min:.0f}-{cv_age_max:.0f} yr</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Available days</span>
            <span class="cs-metric-val">{cv_age_days}</span>
          </div>
        </div>
      </div>

      <!-- Stat callouts -->
      <div class="cs-stats-row">
        <div class="cs-stat">
          <div class="cs-stat-number critical">{rmssd_samples:,}</div>
          <div class="cs-stat-label">RMSSD samples<br>over {rmssd_days} days</div>
        </div>
        <div class="cs-stat">
          <div class="cs-stat-number warning">{pct_below_norm:.0f}%</div>
          <div class="cs-stat-label">below reference<br>median</div>
        </div>
        <div class="cs-stat">
          <div class="cs-stat-number info">{cv_age_mean:.0f} yr</div>
          <div class="cs-stat-label">mean vascular age<br>estimate</div>
        </div>
      </div>

      <!-- Conclusion -->
      <div class="cs-conclusion">
        <strong>Wearable-derived snapshot for this observation window:</strong>
        RMSSD averages {rmssd:.1f} ms, sleep heart rate is {'elevated' if ist_met else 'borderline'}, activity averages {steps:,.0f} steps/day,
        and the Oura vascular age estimate averages {cv_age_mean:.0f} yr. These are descriptive wearable outputs only,
        not confirmed diagnoses or verified medical history.
      </div>

      <!-- Collapsible references -->
      <details class="cs-refs">
        <summary>References (4)</summary>
        <div class="cs-refs-inner">
          <ol>
            <li>Kleiger RE et al. Decreased heart rate variability and its association with increased mortality after acute myocardial infarction. Am J Cardiol 1987;59:256-62</li>
            <li>Bigger JT et al. Frequency domain measures of heart period variability and mortality after myocardial infarction. Circulation 1992;85:164-71</li>
            <li>Shaffer F, Ginsberg JP. An Overview of HRV Metrics and Norms. Front Public Health 2017;5:258</li>
            <li>Nunan D et al. Normal values for short-term HRV. Pacing Clin Electrophysiol 2010;33:1407-17</li>
          </ol>
        </div>
      </details>
    """
    return narrative


# ---------------------------------------------------------------------------
# Assemble full HTML report
# ---------------------------------------------------------------------------

def build_full_report(data: dict, stats: dict) -> str:
    """Combine all figures into a single HTML document using dark theme."""
    figs = [
        ("HRV Deep Dive", fig_hrv_deep_dive(data)),
        ("RMSSD Comparison", fig_rmssd_comparison_bar(stats)),
        ("Heart Rate", fig_heart_rate(data)),
        ("Sleep", fig_sleep_analysis(data)),
        ("Readiness & Recovery", fig_readiness_recovery(data)),
        ("Resilience & Vascular Age", fig_resilience_cv_age(data)),
        ("SpO2 & Stress", fig_spo2_stress(data)),
        ("Activity", fig_activity(data)),
    ]

    # --- KPI row ---
    rmssd_mean = stats.get("rmssd_mean", 0)
    rmssd_status = "critical" if rmssd_mean < ESC_RMSSD_DEFICIENCY else "warning"
    rmssd_label = "Low" if rmssd_status in ("critical", "warning") else ""
    rhr = stats.get("resting_hr_primary", 0)
    rhr_status = "critical" if stats.get("ist_met") else ("warning" if rhr > 80 else "normal")
    rhr_label = "Critical" if rhr_status == "critical" else ("Elevated" if rhr_status == "warning" else "")
    hrv_bal = stats.get("hrv_balance_mean", 50)
    hrv_bal_status = "critical" if hrv_bal < 25 else "warning"
    hrv_bal_label = "Low" if hrv_bal_status in ("critical", "warning") else ""
    readiness = stats.get("readiness_mean", 70)
    readiness_status = "warning" if readiness < 65 else "normal"
    readiness_label = "Low" if readiness_status == "warning" else ""

    kpi_row_1 = make_kpi_row(
        make_kpi_card("RMSSD", rmssd_mean, "ms", status=rmssd_status,
                      detail=f"Normal: {NORM_RMSSD_P25}-{NORM_RMSSD_P75} ms",
                      status_label=rmssd_label),
        make_kpi_card("Sleep HR", rhr, "bpm", status=rhr_status,
                      detail=f"{'Elevated (>80 bpm)' if stats.get('ist_met') else 'Normal range'}",
                      status_label=rhr_label),
        make_kpi_card("HRV Balance", hrv_bal, "/100", status=hrv_bal_status,
                      detail="Oura readiness contributor", decimals=0,
                      status_label=hrv_bal_label),
        make_kpi_card("Readiness", readiness, "/100", status=readiness_status,
                      detail="30-day average", decimals=0,
                      status_label=readiness_label),
    )

    sleep_score = stats.get("sleep_score_mean", 70)
    sleep_status = "warning" if sleep_score < 65 else "normal"
    sleep_label = "Low" if sleep_status == "warning" else ""
    steps = stats.get("steps_mean", 5000)
    steps_status = "critical" if steps < 2000 else "warning"
    steps_label = "Low" if steps_status in ("critical", "warning") else ""
    cv_age_mean = stats.get("cv_age_mean", 0)
    cv_status = "info"
    spo2 = stats.get("spo2_mean", 95)
    spo2_status = "normal" if spo2 >= 95 else "warning"
    spo2_label = "Low" if spo2_status == "warning" else ""

    kpi_row_2 = make_kpi_row(
        make_kpi_card("Sleep Score", sleep_score, "/100", status=sleep_status,
                      detail=f"{stats.get('sleep_days_poor', 0)}/{stats.get('sleep_total_days', 0)} days <60",
                      decimals=0,
                      status_label=sleep_label),
        make_kpi_card("Daily Steps", steps, "", status=steps_status,
                      detail=(f"Peak Samsung Health baseline {stats['pre_dx_peak_steps']:,}/day — {stats['steps_decline_pct']:.0f}% lower now"
                              if stats.get("pre_dx_peak_steps") else
                              f"{stats.get('steps_days_under_2000', 0)} days <2000"),
                      decimals=0,
                      status_label=steps_label),
        make_kpi_card("Vascular Age", cv_age_mean, "yr", status=cv_status,
                      detail=f"Oura estimate across {stats.get('cv_age_total_days', 0)} days", decimals=0),
        make_kpi_card("SpO2", spo2, "%", status=spo2_status,
                      detail=f"Lowest: {stats.get('spo2_min', 0):.1f}%",
                      status_label=spo2_label),
    )

    # --- Chart sections ---
    body = kpi_row_1 + kpi_row_2

    for title, fig in figs:
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
        body += make_section(title, chart_html)

    # --- Narrative section ---
    narrative = clinical_narrative(stats)
    body += make_section("Clinical Summary", narrative)

    # --- Assemble page ---
    subtitle = "Wearable-derived biometric summary for the current observation window"

    return wrap_html(
        title="Full Biometric Analysis",
        body_content=body,
        report_id="full_analysis",
        subtitle=subtitle,
        header_meta="",
        data_end=stats.get("data_end"),
        post_days=stats.get("post_days"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("  OURA FULL ANALYSIS  - Wearable Biometric Assessment")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    bio = connect(DATABASE_PATH)
    try:
        print("\nLoading data from all tables...")
        data = load_all_data(bio)

        for name, df in data.items():
            n = len(df) if isinstance(df, pd.DataFrame) else 0
            print(f"  {name}: {n} rows")
    finally:
        bio.close()

    print("\nCalculating statistics...")
    stats = compute_stats(data)
    stats["data_start"] = earliest_observed_date(data).isoformat()
    stats["data_end"] = latest_observed_date(data).isoformat()
    stats["post_days"] = max(
        0,
        (datetime.strptime(stats["data_end"], "%Y-%m-%d").date() - TREATMENT_START).days + 1,
    )

    print(f"\n  RMSSD mean: {stats.get('rmssd_mean', 0):.1f} ms ({stats.get('rmssd_samples', 0):,} samples)")
    print(f"  RMSSD median: {stats.get('rmssd_median', 0):.0f} ms")
    print(f"  HR daily mean: {stats.get('hr_daily_mean', 0):.0f} bpm ({stats.get('hr_pct_tachycardic', 0):.0f}% tachycardic)")
    print(f"  Sleep HR avg (primary resting HR): {stats.get('resting_hr_primary', 0):.0f} bpm ({'ELEVATED' if stats.get('ist_met') else 'ok'} vs {NOCTURNAL_HR_ELEVATED} bpm threshold)")
    print(f"  Nocturnal HR lowest (sleep_periods): {stats.get('nocturnal_hr_dip_mean', 0):.0f} bpm (normal: {NOCTURNAL_HR_DIP_NORMAL_LOW}-{NOCTURNAL_HR_DIP_NORMAL_HIGH})")
    print(f"  HRV Balance: {stats.get('hrv_balance_mean', 0):.0f}/100")
    print(f"  Sleep score: {stats.get('sleep_score_mean', 0):.0f}/100")
    peak = stats.get('pre_dx_peak_steps')
    decline = stats.get('steps_decline_pct')
    steps_ctx = f" (Peak Samsung Health baseline {peak:,}/day, {decline:.0f}% lower now)" if peak else ""
    print(f"  Steps avg: {stats.get('steps_mean', 0):,.0f}{steps_ctx}")
    print(
        f"  Vascular age: {stats.get('cv_age_mean', 0):.0f} "
        f"(range {stats.get('cv_age_min', 0):.0f}-{stats.get('cv_age_max', 0):.0f}, "
        f"{stats.get('cv_age_total_days', 0)} days)"
    )
    print(f"  SpO2 avg: {stats.get('spo2_mean', 0):.1f}%")

    print("\nGenerating HTML report...")
    html = build_full_report(data, stats)

    datestamp = datetime.now().strftime("%Y%m%d")
    output_path = REPORTS_DIR / f"oura_full_analysis_{datestamp}.html"
    output_path.write_text(html, encoding="utf-8")
    canonical_output_path = REPORTS_DIR / "oura_full_analysis.html"
    canonical_output_path.write_text(html, encoding="utf-8")
    print(f"\n  Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.0f} KB")
    print(f"  Canonical HTML: {canonical_output_path}")

    # Write JSON metrics alongside the HTML
    json_path = output_path.with_suffix(".json")
    json_metrics = {k: round(v, 2) if isinstance(v, float) else v
                    for k, v in stats.items()}
    generated_at = datetime.now(timezone.utc).isoformat()
    data_start = json_metrics.get("data_start")
    data_end = json_metrics.get("data_end")
    n_days = None
    if data_start and data_end:
        n_days = (
            datetime.strptime(str(data_end), "%Y-%m-%d")
            - datetime.strptime(str(data_start), "%Y-%m-%d")
        ).days + 1
    json_metrics["generated"] = generated_at
    json_metrics["generated_at"] = generated_at
    json_metrics["data_range"] = {
        "start": data_start,
        "end": data_end,
        "n_days": n_days,
    }
    json_metrics["report_html"] = str(output_path.name)
    json_metrics["canonical_html"] = str(canonical_output_path.name)
    json_path.write_text(json.dumps(json_metrics, indent=2, default=str), encoding="utf-8")
    canonical_json_path = REPORTS_DIR / "oura_full_analysis.json"
    canonical_json_metrics = dict(json_metrics)
    canonical_json_metrics["report_html"] = canonical_output_path.name
    canonical_json_metrics["date_stamped_html"] = output_path.name
    canonical_json_path.write_text(
        json.dumps(canonical_json_metrics, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  JSON metrics: {json_path}")
    print(f"  Canonical JSON: {canonical_json_path}")

    print("\n" + "=" * 70)
    print(f"  DONE: {output_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
