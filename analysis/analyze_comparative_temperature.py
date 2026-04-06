#!/usr/bin/env python3
"""Comparative Temperature Deviation Analysis.

Systematic temperature deviation analysis for both patients. Validated by
Mitchell's +13C flight day.  Can flag illness before symptoms appear.

Temperature delta from Oura represents nightly skin temperature deviation
from the wearer's personal baseline, measured in degrees Celsius.

Outputs:
  - Interactive HTML dashboard: reports/comparative_temperature_analysis.html
  - JSON metrics:               reports/comparative_temperature_metrics.json

Usage:
    python analysis/analyze_comparative_temperature.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution & config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    REPORTS_DIR,
    FONT_FAMILY,
    TREATMENT_START,
    KNOWN_EVENT_DATE,
)
from _comparative_utils import (
    PatientConfig,
    default_patients,
    load_patient_data,
    compare_distributions,
    effect_size_cohens_d,
    bootstrap_ci,
    PATIENT_COLORS,
)
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    disclaimer_banner,
    format_p_value,
    COLORWAY,
    STATUS_COLORS,
    BG_PRIMARY,
    BG_SURFACE,
    BG_ELEVATED,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_PURPLE,
    ACCENT_CYAN,
    ACCENT_ORANGE,
    ACCENT_PINK,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    BORDER_SUBTLE,
)
from _hardening import section_html_or_placeholder

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "comparative_temperature_analysis.html"
JSON_OUTPUT = REPORTS_DIR / "comparative_temperature_metrics.json"

# Temperature classification thresholds (degrees C deviation from baseline)
TEMP_FEVER = 1.0
TEMP_MILD_ELEVATION = 0.5
TEMP_NORMAL_LOW = -0.5
TEMP_HYPOTHERMIA = -1.0

# Known events for validation
KNOWN_EVENTS = {
    "mitch": [
        {"date": date(2022, 5, 4), "label": "+13C flight day", "expected_delta": 13.0},
    ],
    "henrik": [
        {"date": date(2026, 2, 9), "label": "Acute event (Feb 9)", "expected_delta": None},
    ],
}


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_data(
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load temperature and cross-reference data for both patients."""
    result: dict[str, dict[str, pd.DataFrame]] = {}

    for p in patients:
        pid = p.patient_id
        frames: dict[str, pd.DataFrame] = {}

        # Sleep temperature delta
        frames["sleep"] = load_patient_data(
            p, table="oura_sleep",
            columns="date, temperature_delta",
        )

        # Readiness temperature deviation + score
        frames["readiness"] = load_patient_data(
            p, table="oura_readiness",
            columns="date, score, temperature_deviation, recovery_index",
        )

        # Sleep periods for HR cross-reference
        frames["periods"] = load_patient_data(
            p, table="oura_sleep_periods",
            columns="day, average_heart_rate, lowest_heart_rate, average_hrv",
        )

        result[pid] = frames

    return result


# ---------------------------------------------------------------------------
# 2. Baseline Statistics
# ---------------------------------------------------------------------------

def compute_baseline_stats(
    data: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, dict[str, Any]]:
    """Per-patient descriptive statistics for temperature delta."""
    stats: dict[str, dict[str, Any]] = {}

    for pid, frames in data.items():
        sleep_df = frames["sleep"]
        if sleep_df.empty or "temperature_delta" not in sleep_df.columns:
            stats[pid] = {
                "mean": np.nan, "median": np.nan, "std": np.nan,
                "iqr": np.nan, "q25": np.nan, "q75": np.nan,
                "n_days": 0, "min": np.nan, "max": np.nan,
            }
            continue

        td = sleep_df["temperature_delta"].dropna()
        q25, q75 = td.quantile(0.25), td.quantile(0.75)
        stats[pid] = {
            "mean": float(td.mean()),
            "median": float(td.median()),
            "std": float(td.std()),
            "iqr": float(q75 - q25),
            "q25": float(q25),
            "q75": float(q75),
            "n_days": int(len(td)),
            "min": float(td.min()),
            "max": float(td.max()),
        }

    return stats


def compute_rolling(
    data: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, dict[str, pd.Series]]:
    """7-day and 30-day rolling means for temperature delta."""
    rolling: dict[str, dict[str, pd.Series]] = {}

    for pid, frames in data.items():
        sleep_df = frames["sleep"]
        if sleep_df.empty or "temperature_delta" not in sleep_df.columns:
            rolling[pid] = {"r7": pd.Series(dtype=float), "r30": pd.Series(dtype=float)}
            continue

        td = sleep_df["temperature_delta"].dropna()
        rolling[pid] = {
            "r7": td.rolling(7, min_periods=3).mean(),
            "r30": td.rolling(30, min_periods=10).mean(),
        }

    return rolling


# ---------------------------------------------------------------------------
# 3. Anomaly Detection
# ---------------------------------------------------------------------------

def classify_temperature(delta: float) -> str:
    """Classify a single temperature delta value."""
    if delta > TEMP_FEVER:
        return "fever"
    elif delta > TEMP_MILD_ELEVATION:
        return "mild_elevation"
    elif delta >= TEMP_NORMAL_LOW:
        return "normal"
    elif delta >= TEMP_HYPOTHERMIA:
        return "mild_low"
    else:
        return "hypothermia"


def detect_anomalies(
    data: dict[str, dict[str, pd.DataFrame]],
    baseline_stats: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Flag anomaly days (|temp_delta| > 2 SD) with classification and cross-metrics."""
    anomalies: dict[str, pd.DataFrame] = {}

    for pid, frames in data.items():
        sleep_df = frames["sleep"]
        if sleep_df.empty or "temperature_delta" not in sleep_df.columns:
            anomalies[pid] = pd.DataFrame()
            continue

        td = sleep_df["temperature_delta"].dropna()
        bs = baseline_stats[pid]
        threshold = 2.0 * bs["std"] if not np.isnan(bs["std"]) and bs["std"] > 0 else 1.0

        anomaly_mask = td.abs() > threshold
        anomaly_dates = td[anomaly_mask].index

        rows = []
        for dt in anomaly_dates:
            delta_val = float(td.loc[dt])
            row: dict[str, Any] = {
                "date": dt,
                "temperature_delta": delta_val,
                "z_score": (delta_val - bs["mean"]) / bs["std"] if bs["std"] > 0 else 0.0,
                "classification": classify_temperature(delta_val),
            }

            # Cross-reference HR
            periods_df = frames["periods"]
            if not periods_df.empty and dt in periods_df.index:
                pr = periods_df.loc[dt]
                if isinstance(pr, pd.DataFrame):
                    pr = pr.iloc[-1]
                row["avg_hr"] = pr.get("average_heart_rate", np.nan)
                row["lowest_hr"] = pr.get("lowest_heart_rate", np.nan)
                row["avg_hrv"] = pr.get("average_hrv", np.nan)
            else:
                row["avg_hr"] = np.nan
                row["lowest_hr"] = np.nan
                row["avg_hrv"] = np.nan

            # Cross-reference readiness
            readiness_df = frames["readiness"]
            if not readiness_df.empty and dt in readiness_df.index:
                rd = readiness_df.loc[dt]
                if isinstance(rd, pd.DataFrame):
                    rd = rd.iloc[-1]
                row["readiness_score"] = rd.get("score", np.nan)
            else:
                row["readiness_score"] = np.nan

            rows.append(row)

        if rows:
            adf = pd.DataFrame(rows).set_index("date").sort_index()
        else:
            adf = pd.DataFrame(
                columns=["temperature_delta", "z_score", "classification",
                          "avg_hr", "lowest_hr", "avg_hrv", "readiness_score"]
            )
        anomalies[pid] = adf

    return anomalies


def get_temp_spikes(
    data: dict[str, dict[str, pd.DataFrame]],
    threshold: float = TEMP_FEVER,
) -> dict[str, pd.DataFrame]:
    """List all days where temp_delta > threshold for each patient."""
    spikes: dict[str, pd.DataFrame] = {}

    for pid, frames in data.items():
        sleep_df = frames["sleep"]
        if sleep_df.empty or "temperature_delta" not in sleep_df.columns:
            spikes[pid] = pd.DataFrame()
            continue

        td = sleep_df["temperature_delta"].dropna()
        mask = td > threshold
        spike_dates = td[mask].sort_values(ascending=False)
        spikes[pid] = pd.DataFrame({
            "temperature_delta": spike_dates,
            "classification": spike_dates.map(classify_temperature),
        })

    return spikes


# ---------------------------------------------------------------------------
# 4. Known Event Validation
# ---------------------------------------------------------------------------

def validate_known_events(
    data: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    """Check temperature on known event dates."""
    results: list[dict[str, Any]] = []

    for pid, events in KNOWN_EVENTS.items():
        frames = data.get(pid)
        if frames is None:
            continue

        sleep_df = frames["sleep"]
        for ev in events:
            ev_date = pd.Timestamp(ev["date"])
            row: dict[str, Any] = {
                "patient": pid,
                "date": str(ev["date"]),
                "event": ev["label"],
                "expected_delta": ev["expected_delta"],
                "actual_delta": None,
                "validated": False,
            }

            if not sleep_df.empty and "temperature_delta" in sleep_df.columns:
                td = sleep_df["temperature_delta"].dropna()
                # Look for exact date or nearest date within 1 day
                if ev_date in td.index:
                    val = td.loc[ev_date]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        row["actual_delta"] = float(val)
                        row["validated"] = True
                else:
                    nearby = td.loc[
                        (td.index >= ev_date - pd.Timedelta(days=1)) &
                        (td.index <= ev_date + pd.Timedelta(days=1))
                    ]
                    if not nearby.empty:
                        val = nearby.iloc[0]
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            row["actual_delta"] = float(val)
                            row["validated"] = True

            results.append(row)

    return results


# ---------------------------------------------------------------------------
# 5. Early Illness Detection (lag correlations)
# ---------------------------------------------------------------------------

def compute_lag_correlations(
    data: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, dict[str, Any]]:
    """Spearman correlations at lags 0, 1, 2: temp[N] vs readiness[N+lag], temp[N] vs HR[N+lag]."""
    lag_results: dict[str, dict[str, Any]] = {}

    for pid, frames in data.items():
        sleep_df = frames["sleep"]
        readiness_df = frames["readiness"]
        periods_df = frames["periods"]

        if sleep_df.empty or "temperature_delta" not in sleep_df.columns:
            lag_results[pid] = {"temp_vs_readiness": {}, "temp_vs_hr": {}}
            continue

        td = sleep_df["temperature_delta"].dropna()
        result: dict[str, Any] = {"temp_vs_readiness": {}, "temp_vs_hr": {}}

        # Temp vs readiness score at lags 0, 1, 2
        if not readiness_df.empty and "score" in readiness_df.columns:
            rs = readiness_df["score"].dropna()
            for lag in [0, 1, 2]:
                shifted_rs = rs.shift(-lag)
                common = td.index.intersection(shifted_rs.dropna().index)
                if len(common) >= 10:
                    r, p = scipy_stats.spearmanr(td.loc[common], shifted_rs.loc[common])
                    result["temp_vs_readiness"][f"lag_{lag}"] = {
                        "rho": float(r), "p_value": float(p), "n": int(len(common)),
                    }

        # Temp vs average HR at lags 0, 1, 2
        if not periods_df.empty and "average_heart_rate" in periods_df.columns:
            hr = periods_df["average_heart_rate"].dropna()
            for lag in [0, 1, 2]:
                shifted_hr = hr.shift(-lag)
                common = td.index.intersection(shifted_hr.dropna().index)
                if len(common) >= 10:
                    r, p = scipy_stats.spearmanr(td.loc[common], shifted_hr.loc[common])
                    result["temp_vs_hr"][f"lag_{lag}"] = {
                        "rho": float(r), "p_value": float(p), "n": int(len(common)),
                    }

        lag_results[pid] = result

    return lag_results


# ---------------------------------------------------------------------------
# 6. Pre/Post Ruxolitinib Analysis (Henrik)
# ---------------------------------------------------------------------------

def rux_pre_post_analysis(
    data: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    """Compare Henrik temperature variability pre vs post Ruxolitinib."""
    result: dict[str, Any] = {"available": False}

    henrik_frames = data.get("henrik")
    if henrik_frames is None:
        return result

    sleep_df = henrik_frames["sleep"]
    if sleep_df.empty or "temperature_delta" not in sleep_df.columns:
        return result

    td = sleep_df["temperature_delta"].dropna()
    rux_date = pd.Timestamp(TREATMENT_START)

    pre = td[td.index < rux_date]
    post = td[td.index >= rux_date]

    if len(pre) < 5 or len(post) < 5:
        result["note"] = f"Insufficient data: pre={len(pre)}, post={len(post)}"
        return result

    result["available"] = True
    result["pre_n"] = int(len(pre))
    result["post_n"] = int(len(post))
    result["pre_mean"] = float(pre.mean())
    result["post_mean"] = float(post.mean())
    result["pre_sd"] = float(pre.std())
    result["post_sd"] = float(post.std())
    result["pre_abs_mean"] = float(pre.abs().mean())
    result["post_abs_mean"] = float(post.abs().mean())

    # Mann-Whitney U on |temp_delta|
    u_stat, u_p = scipy_stats.mannwhitneyu(
        pre.abs().values, post.abs().values, alternative="two-sided"
    )
    result["mannwhitney_u"] = float(u_stat)
    result["mannwhitney_p"] = float(u_p)

    # Levene's test for variance equality
    lev_stat, lev_p = scipy_stats.levene(pre.values, post.values)
    result["levene_stat"] = float(lev_stat)
    result["levene_p"] = float(lev_p)

    # Cohen's d on |temp_delta|
    result["cohens_d"] = float(effect_size_cohens_d(pre.abs(), post.abs()))

    # Variance ratio (F-test approximation)
    var_pre = float(pre.var())
    var_post = float(post.var())
    if var_post > 0:
        result["variance_ratio"] = float(var_pre / var_post)
    else:
        result["variance_ratio"] = np.nan

    result["sd_reduced"] = result["post_sd"] < result["pre_sd"]
    reduction_pct = (
        (result["pre_sd"] - result["post_sd"]) / result["pre_sd"] * 100
        if result["pre_sd"] > 0 else 0.0
    )
    result["sd_reduction_pct"] = float(reduction_pct)

    return result


# ---------------------------------------------------------------------------
# 7. Cross-Patient Comparison
# ---------------------------------------------------------------------------

def cross_patient_comparison(
    data: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    """Distribution comparison between the two patients."""
    result: dict[str, Any] = {}

    series_map: dict[str, pd.Series] = {}
    for pid, frames in data.items():
        sleep_df = frames["sleep"]
        if not sleep_df.empty and "temperature_delta" in sleep_df.columns:
            series_map[pid] = sleep_df["temperature_delta"].dropna()

    if "henrik" in series_map and "mitch" in series_map:
        h = series_map["henrik"]
        m = series_map["mitch"]
        result["comparison"] = compare_distributions(h, m)
        result["henrik_runs_hotter"] = float(h.mean()) > float(m.mean())
        result["henrik_more_variable"] = float(h.std()) > float(m.std())
    else:
        result["comparison"] = {"test_name": "insufficient_data"}

    return result


# ---------------------------------------------------------------------------
# 8. Visualization: Dual Timeline with Anomaly Markers
# ---------------------------------------------------------------------------

def fig_dual_timeline(
    data: dict[str, dict[str, pd.DataFrame]],
    rolling: dict[str, dict[str, pd.Series]],
    anomalies: dict[str, pd.DataFrame],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Both patients temp_delta over time. Rolling 7d mean, anomaly markers, normal band."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    for pid, frames in data.items():
        sleep_df = frames["sleep"]
        if sleep_df.empty or "temperature_delta" not in sleep_df.columns:
            continue

        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        td = sleep_df["temperature_delta"].dropna()

        # Raw scatter (faded)
        fig.add_trace(go.Scatter(
            x=td.index, y=td.values,
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.2),
            name=f"{p.display_name} (daily)",
            legendgroup=pid,
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f} C<extra>" + p.display_name + "</extra>",
        ))

        # Rolling 7d mean
        r7 = rolling.get(pid, {}).get("r7", pd.Series(dtype=float))
        if not r7.empty:
            fig.add_trace(go.Scatter(
                x=r7.index, y=r7.values,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{p.display_name} (7d avg)",
                legendgroup=pid,
            ))

        # Anomaly markers
        anom = anomalies.get(pid, pd.DataFrame())
        if not anom.empty and "temperature_delta" in anom.columns:
            fig.add_trace(go.Scatter(
                x=anom.index, y=anom["temperature_delta"].values,
                mode="markers",
                marker=dict(size=8, color=ACCENT_RED, symbol="circle-open", line=dict(width=2)),
                name=f"{p.display_name} anomaly",
                legendgroup=pid,
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f} C (anomaly)<extra></extra>",
            ))

    # Normal range band
    x_all = []
    for frames in data.values():
        s = frames["sleep"]
        if not s.empty and "temperature_delta" in s.columns:
            x_all.extend(s.index.tolist())
    if x_all:
        x_min, x_max = min(x_all), max(x_all)
        fig.add_shape(
            type="rect",
            x0=x_min, x1=x_max,
            y0=TEMP_NORMAL_LOW, y1=TEMP_MILD_ELEVATION,
            fillcolor="rgba(16, 185, 129, 0.08)",
            line=dict(width=0),
            layer="below",
        )
        # Band labels
        fig.add_annotation(
            x=x_max, y=TEMP_MILD_ELEVATION, text="Normal range",
            showarrow=False, font=dict(size=9, color=TEXT_TERTIARY),
            xanchor="right", yanchor="bottom",
        )

    # Event markers for each patient
    for p in patients:
        event_ts = pd.Timestamp(p.event_date)
        color = PATIENT_COLORS.get(p.patient_id, ACCENT_PURPLE)
        fig.add_shape(
            type="line",
            x0=event_ts, x1=event_ts, y0=0, y1=1, yref="paper",
            line=dict(color=color, width=1.5, dash="dash"), opacity=0.5,
        )
        fig.add_annotation(
            x=event_ts, y=1.02, yref="paper",
            text=p.event_label, showarrow=False,
            font=dict(size=9, color=color),
        )

    fig.update_layout(
        title=dict(text="Temperature Deviation Timeline", font=dict(size=16)),
        yaxis_title="Temperature Delta (C)",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Visualization: Temperature Distribution
# ---------------------------------------------------------------------------

def fig_temperature_distribution(
    data: dict[str, dict[str, pd.DataFrame]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Overlapping violin plots with normal range shading."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    for pid, frames in data.items():
        sleep_df = frames["sleep"]
        if sleep_df.empty or "temperature_delta" not in sleep_df.columns:
            continue

        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        td = sleep_df["temperature_delta"].dropna()

        fig.add_trace(go.Violin(
            y=td.values,
            name=p.display_name,
            marker_color=color,
            box_visible=True,
            meanline_visible=True,
            opacity=0.8,
        ))

    # Normal range rectangle (horizontal band)
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=1.5,
        y0=TEMP_NORMAL_LOW, y1=TEMP_MILD_ELEVATION,
        fillcolor="rgba(16, 185, 129, 0.08)",
        line=dict(color=ACCENT_GREEN, width=1, dash="dot"),
        layer="below",
    )
    fig.add_annotation(
        x=1.5, y=0.0, text="Normal",
        showarrow=False, font=dict(size=9, color=ACCENT_GREEN),
        xanchor="left",
    )

    fig.update_layout(
        title=dict(text="Temperature Delta Distribution", font=dict(size=16)),
        yaxis_title="Temperature Delta (C)",
        showlegend=True,
        margin=dict(l=60, r=20, t=50, b=40),
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# 10. Visualization: Pre/Post Ruxolitinib
# ---------------------------------------------------------------------------

def fig_rux_pre_post(
    data: dict[str, dict[str, pd.DataFrame]],
    rux_stats: dict[str, Any],
) -> go.Figure:
    """Box/violin for Henrik pre vs post Ruxolitinib temperature distribution."""
    fig = go.Figure()

    henrik_frames = data.get("henrik")
    if henrik_frames is None or not rux_stats.get("available"):
        fig.add_annotation(
            text="Insufficient data for pre/post Ruxolitinib comparison",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=TEXT_SECONDARY),
        )
        fig.update_layout(height=400)
        return fig

    sleep_df = henrik_frames["sleep"]
    td = sleep_df["temperature_delta"].dropna()
    rux_date = pd.Timestamp(TREATMENT_START)
    pre = td[td.index < rux_date]
    post = td[td.index >= rux_date]

    fig.add_trace(go.Violin(
        y=pre.values,
        name=f"Pre-Rux (n={len(pre)})",
        marker_color=ACCENT_AMBER,
        box_visible=True,
        meanline_visible=True,
        opacity=0.8,
        side="negative",
    ))

    fig.add_trace(go.Violin(
        y=post.values,
        name=f"Post-Rux (n={len(post)})",
        marker_color=ACCENT_CYAN,
        box_visible=True,
        meanline_visible=True,
        opacity=0.8,
        side="positive",
    ))

    # Annotation with stats
    p_val = rux_stats.get("mannwhitney_p", np.nan)
    d_val = rux_stats.get("cohens_d", 0.0)
    sd_pct = rux_stats.get("sd_reduction_pct", 0.0)
    ann_text = (
        f"Mann-Whitney p={format_p_value(p_val)}<br>"
        f"Cohen's d={d_val:.2f}<br>"
        f"SD change: {sd_pct:+.1f}%"
    )
    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text=ann_text, showarrow=False,
        font=dict(size=11, color=TEXT_SECONDARY),
        align="right", xanchor="right", yanchor="top",
        bgcolor=BG_ELEVATED, bordercolor=BORDER_SUBTLE, borderwidth=1,
    )

    fig.update_layout(
        title=dict(text="Henrik: Pre vs Post Ruxolitinib Temperature", font=dict(size=16)),
        yaxis_title="Temperature Delta (C)",
        violinmode="overlay",
        margin=dict(l=60, r=20, t=50, b=40),
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# 11. Visualization: Predictive Scatter (early warning)
# ---------------------------------------------------------------------------

def fig_predictive_scatter(
    data: dict[str, dict[str, pd.DataFrame]],
    lag_corrs: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Temp[N] vs Readiness[N+1] for both patients. Two subplots."""
    patient_list = [p for p in patients if p.patient_id in data]
    n_cols = min(len(patient_list), 2)
    if n_cols == 0:
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[p.display_name for p in patient_list[:n_cols]],
        horizontal_spacing=0.12,
    )

    for i, p in enumerate(patient_list[:n_cols], 1):
        pid = p.patient_id
        frames = data[pid]
        sleep_df = frames["sleep"]
        readiness_df = frames["readiness"]

        if (sleep_df.empty or "temperature_delta" not in sleep_df.columns
                or readiness_df.empty or "score" not in readiness_df.columns):
            continue

        td = sleep_df["temperature_delta"].dropna()
        rs = readiness_df["score"].dropna()

        # Align: temp[N] vs readiness[N+1]
        rs_shifted = rs.shift(-1).dropna()
        common = td.index.intersection(rs_shifted.index)
        if len(common) < 5:
            continue

        x_vals = td.loc[common].values
        y_vals = rs_shifted.loc[common].values

        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(size=4, color=color, opacity=0.4),
            name=p.display_name,
            showlegend=False,
            hovertemplate="Temp: %{x:.2f} C<br>Next-day readiness: %{y:.0f}<extra></extra>",
        ), row=1, col=i)

        # Spearman annotation
        lag1 = lag_corrs.get(pid, {}).get("temp_vs_readiness", {}).get("lag_1", {})
        rho = lag1.get("rho", np.nan)
        p_val = lag1.get("p_value", np.nan)
        if not np.isnan(rho):
            fig.add_annotation(
                x=0.5, y=0.98,
                xref=f"x{i} domain" if i > 1 else "x domain",
                yref=f"y{i} domain" if i > 1 else "y domain",
                text=f"Spearman r={rho:.3f}, p={format_p_value(p_val)}",
                showarrow=False,
                font=dict(size=10, color=TEXT_SECONDARY),
                bgcolor=BG_ELEVATED, bordercolor=BORDER_SUBTLE, borderwidth=1,
            )

        fig.update_xaxes(title_text="Temp Delta (C)", row=1, col=i)
        fig.update_yaxes(title_text="Next-Day Readiness", row=1, col=i)

    fig.update_layout(
        title=dict(text="Temperature as Early Warning: Temp[N] vs Readiness[N+1]", font=dict(size=16)),
        height=450,
        margin=dict(l=60, r=20, t=70, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# 12. Visualization: Anomaly Calendar Heatmap
# ---------------------------------------------------------------------------

def fig_calendar_heatmap(
    data: dict[str, dict[str, pd.DataFrame]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """GitHub-style calendar heatmap of temperature delta magnitude. One panel per patient."""
    patient_list = [p for p in patients if p.patient_id in data]
    n_rows = len(patient_list)
    if n_rows == 0:
        fig = go.Figure()
        fig.update_layout(height=300)
        return fig

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=[p.display_name for p in patient_list],
        vertical_spacing=0.15,
    )

    for row_idx, p in enumerate(patient_list, 1):
        pid = p.patient_id
        frames = data[pid]
        sleep_df = frames["sleep"]

        if sleep_df.empty or "temperature_delta" not in sleep_df.columns:
            continue

        td = sleep_df["temperature_delta"].dropna()
        if td.empty:
            continue

        # Build calendar data
        dates = td.index
        cal_df = pd.DataFrame({
            "date": dates,
            "value": td.values,
            "week": dates.isocalendar().week.values,
            "year": dates.year,
            "dow": dates.dayofweek,  # 0=Mon, 6=Sun
        })
        # Create year-week combo for x-axis
        cal_df["year_week"] = cal_df["year"].astype(str) + "-W" + cal_df["week"].astype(str).str.zfill(2)

        # Get unique year-weeks in order
        unique_weeks = cal_df.sort_values("date")["year_week"].unique()
        week_to_x = {w: i for i, w in enumerate(unique_weeks)}
        cal_df["x"] = cal_df["year_week"].map(week_to_x)

        # Clamp values for color scale
        max_abs = max(2.0, td.abs().quantile(0.95))

        color = PATIENT_COLORS.get(pid, ACCENT_BLUE)
        fig.add_trace(go.Heatmap(
            x=cal_df["x"],
            y=cal_df["dow"],
            z=cal_df["value"],
            colorscale=[
                [0.0, ACCENT_BLUE],
                [0.35, ACCENT_CYAN],
                [0.5, BG_ELEVATED],
                [0.65, ACCENT_AMBER],
                [1.0, ACCENT_RED],
            ],
            zmin=-max_abs,
            zmax=max_abs,
            showscale=(row_idx == 1),
            colorbar=dict(
                title="C",
                len=0.4,
                y=0.8,
            ) if row_idx == 1 else None,
            hovertemplate="Week %{x}, Day %{y}<br>Temp delta: %{z:.2f} C<extra></extra>",
        ), row=row_idx, col=1)

        # Label axes
        fig.update_yaxes(
            tickvals=list(range(7)),
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            row=row_idx, col=1,
        )
        # Sparse week labels
        n_weeks = len(unique_weeks)
        tick_step = max(1, n_weeks // 12)
        tick_vals = list(range(0, n_weeks, tick_step))
        tick_texts = [unique_weeks[i] if i < len(unique_weeks) else "" for i in tick_vals]
        fig.update_xaxes(
            tickvals=tick_vals,
            ticktext=tick_texts,
            row=row_idx, col=1,
        )

    fig.update_layout(
        title=dict(text="Temperature Calendar Heatmap", font=dict(size=16)),
        height=250 * n_rows + 80,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 13. HTML Assembly
# ---------------------------------------------------------------------------

def _build_event_table(events: list[dict[str, Any]]) -> str:
    """Build HTML table for known event validation."""
    if not events:
        return "<p>No known events to validate.</p>"

    rows_html = ""
    for ev in events:
        actual = ev.get("actual_delta")
        actual_str = f"{actual:.2f} C" if actual is not None else "No data"
        validated = "Yes" if ev.get("validated") else "No"
        status = "normal" if ev.get("validated") else "warning"
        status_color = STATUS_COLORS.get(status, TEXT_SECONDARY)

        rows_html += (
            f"<tr>"
            f"<td>{ev['patient'].title()}</td>"
            f"<td>{ev['date']}</td>"
            f"<td>{ev['event']}</td>"
            f"<td>{ev.get('expected_delta', 'N/A')}</td>"
            f"<td>{actual_str}</td>"
            f'<td style="color:{status_color}">{validated}</td>'
            f"</tr>"
        )

    return (
        '<table class="odt-table">'
        "<thead><tr>"
        "<th>Patient</th><th>Date</th><th>Event</th>"
        "<th>Expected</th><th>Actual</th><th>Validated</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )


def _build_lag_table(lag_corrs: dict[str, dict[str, Any]]) -> str:
    """Build HTML table for lag correlation results."""
    rows_html = ""
    for pid, corrs in lag_corrs.items():
        for target, target_label in [
            ("temp_vs_readiness", "Readiness Score"),
            ("temp_vs_hr", "Average HR"),
        ]:
            lag_data = corrs.get(target, {})
            for lag_key in ["lag_0", "lag_1", "lag_2"]:
                entry = lag_data.get(lag_key)
                if entry is None:
                    continue
                rho = entry["rho"]
                p_val = entry["p_value"]
                n = entry["n"]
                sig = "Yes" if p_val < 0.05 else "No"
                sig_color = ACCENT_GREEN if p_val < 0.05 else TEXT_SECONDARY
                rows_html += (
                    f"<tr>"
                    f"<td>{pid.title()}</td>"
                    f"<td>{target_label}</td>"
                    f"<td>{lag_key.replace('lag_', '')}</td>"
                    f"<td>{rho:.3f}</td>"
                    f"<td>{format_p_value(p_val)}</td>"
                    f"<td>{n}</td>"
                    f'<td style="color:{sig_color}">{sig}</td>'
                    f"</tr>"
                )

    if not rows_html:
        return "<p>Insufficient data for lag correlation analysis.</p>"

    return (
        '<table class="odt-table">'
        "<thead><tr>"
        "<th>Patient</th><th>Target</th><th>Lag (days)</th>"
        "<th>Spearman rho</th><th>p-value</th><th>n</th><th>Significant</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )


def _build_spikes_table(
    spikes: dict[str, pd.DataFrame],
    max_rows: int = 15,
) -> str:
    """Build HTML table showing top temperature spikes per patient."""
    rows_html = ""
    for pid, sdf in spikes.items():
        if sdf.empty:
            continue
        for dt, row in sdf.head(max_rows).iterrows():
            delta = row["temperature_delta"]
            cls = row["classification"]
            color = ACCENT_RED if cls == "fever" else ACCENT_AMBER
            rows_html += (
                f"<tr>"
                f"<td>{pid.title()}</td>"
                f"<td>{dt.strftime('%Y-%m-%d')}</td>"
                f'<td style="color:{color}">{delta:+.2f} C</td>'
                f"<td>{cls.replace('_', ' ').title()}</td>"
                f"</tr>"
            )

    if not rows_html:
        return "<p>No temperature spikes above +1.0 C found.</p>"

    return (
        '<table class="odt-table">'
        "<thead><tr>"
        "<th>Patient</th><th>Date</th><th>Temp Delta</th><th>Classification</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )


def build_html(
    data: dict[str, dict[str, pd.DataFrame]],
    baseline_stats: dict[str, dict[str, Any]],
    rolling: dict[str, dict[str, pd.Series]],
    anomalies: dict[str, pd.DataFrame],
    spikes: dict[str, pd.DataFrame],
    events: list[dict[str, Any]],
    lag_corrs: dict[str, dict[str, Any]],
    rux_stats: dict[str, Any],
    cross_patient: dict[str, Any],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Assemble the full HTML report."""
    sections: list[str] = []

    # ---- KPI Row ----
    def _kpi_section() -> str:
        h_stats = baseline_stats.get("henrik", {})
        m_stats = baseline_stats.get("mitch", {})
        h_anom = anomalies.get("henrik", pd.DataFrame())
        m_anom = anomalies.get("mitch", pd.DataFrame())

        h_mean = h_stats.get("mean", np.nan)
        m_mean = m_stats.get("mean", np.nan)
        h_sd = h_stats.get("std", np.nan)
        m_sd = m_stats.get("std", np.nan)
        total_anomalies = len(h_anom) + len(m_anom)

        cards = [
            make_kpi_card(
                "HENRIK MEAN TEMP",
                h_mean if not np.isnan(h_mean) else "N/A",
                unit="C",
                status="normal" if abs(h_mean) < 0.5 and not np.isnan(h_mean) else "warning",
                detail=f"n={h_stats.get('n_days', 0)} days",
            ),
            make_kpi_card(
                "MITCHELL MEAN TEMP",
                m_mean if not np.isnan(m_mean) else "N/A",
                unit="C",
                status="normal" if abs(m_mean) < 0.5 and not np.isnan(m_mean) else "warning",
                detail=f"n={m_stats.get('n_days', 0)} days",
            ),
            make_kpi_card(
                "HENRIK TEMP SD",
                h_sd if not np.isnan(h_sd) else "N/A",
                unit="C",
                decimals=3,
                status="info",
                detail=f"IQR={h_stats.get('iqr', 0):.3f}",
            ),
            make_kpi_card(
                "MITCHELL TEMP SD",
                m_sd if not np.isnan(m_sd) else "N/A",
                unit="C",
                decimals=3,
                status="info",
                detail=f"IQR={m_stats.get('iqr', 0):.3f}",
            ),
            make_kpi_card(
                "ANOMALY DAYS",
                total_anomalies,
                unit="days",
                status="warning" if total_anomalies > 10 else "normal",
                detail=f"H={len(h_anom)}, M={len(m_anom)}",
            ),
        ]
        return make_kpi_row(*cards)

    sections.append(section_html_or_placeholder("KPI Row", _kpi_section))

    # ---- Section 1: Temperature Trends ----
    def _trends_section() -> str:
        fig = fig_dual_timeline(data, rolling, anomalies, patients)
        chart = fig.to_html(include_plotlyjs=False, full_html=False)
        return make_section("Temperature Trends", chart, section_id="trends")

    sections.append(section_html_or_placeholder("Temperature Trends", _trends_section))

    # ---- Section 2: Distribution Comparison ----
    def _distribution_section() -> str:
        fig = fig_temperature_distribution(data, patients)
        chart = fig.to_html(include_plotlyjs=False, full_html=False)

        # Stats summary
        comp = cross_patient.get("comparison", {})
        stats_html = ""
        if comp.get("test_name") != "insufficient_data":
            stats_html = (
                f'<div style="padding:12px;margin-top:12px;background:{BG_ELEVATED};'
                f'border-radius:8px;font-size:13px;color:{TEXT_SECONDARY};">'
                f'<strong>Cross-patient comparison:</strong> '
                f'Mann-Whitney U p={format_p_value(comp.get("p_value"))}, '
                f'Cohen\'s d={comp.get("effect_size", 0):.2f} '
                f'({comp.get("effect_label", "n/a")}). '
            )
            if cross_patient.get("henrik_runs_hotter"):
                stats_html += "Henrik runs warmer on average. "
            else:
                stats_html += "Mitchell runs warmer on average. "
            if cross_patient.get("henrik_more_variable"):
                stats_html += "Henrik shows more temperature variability."
            else:
                stats_html += "Mitchell shows more temperature variability."
            stats_html += "</div>"

        return make_section(
            "Distribution Comparison",
            chart + stats_html,
            section_id="distribution",
        )

    sections.append(section_html_or_placeholder("Distribution Comparison", _distribution_section))

    # ---- Section 3: Ruxolitinib Effect ----
    def _rux_section() -> str:
        fig = fig_rux_pre_post(data, rux_stats)
        chart = fig.to_html(include_plotlyjs=False, full_html=False)

        summary = ""
        if rux_stats.get("available"):
            direction = "reduced" if rux_stats["sd_reduced"] else "increased"
            lev_p = rux_stats.get("levene_p", np.nan)
            summary = (
                f'<div style="padding:12px;margin-top:12px;background:{BG_ELEVATED};'
                f'border-radius:8px;font-size:13px;color:{TEXT_SECONDARY};">'
                f'Ruxolitinib (anti-inflammatory) {direction} temperature variability by '
                f'{abs(rux_stats["sd_reduction_pct"]):.1f}%. '
                f'Pre-Rux SD={rux_stats["pre_sd"]:.3f} C, Post-Rux SD={rux_stats["post_sd"]:.3f} C. '
                f'Levene\'s test for variance equality: p={format_p_value(lev_p)}. '
                f'|Temp delta| Mann-Whitney p={format_p_value(rux_stats.get("mannwhitney_p"))}.'
                f'</div>'
            )

        return make_section("Ruxolitinib Effect (Henrik)", chart + summary, section_id="rux")

    sections.append(section_html_or_placeholder("Ruxolitinib Effect", _rux_section))

    # ---- Section 4: Early Warning Model ----
    def _early_warning_section() -> str:
        fig = fig_predictive_scatter(data, lag_corrs, patients)
        chart = fig.to_html(include_plotlyjs=False, full_html=False)

        table = _build_lag_table(lag_corrs)
        explanation = (
            f'<div style="padding:12px;margin-top:12px;background:{BG_ELEVATED};'
            f'border-radius:8px;font-size:13px;color:{TEXT_SECONDARY};">'
            f'<strong>Method:</strong> Spearman rank correlations at lags 0, 1, and 2 days. '
            f'Lag 1 tests whether today\'s temperature predicts tomorrow\'s readiness or heart rate. '
            f'A significant negative correlation at lag 1 (temp up, readiness down next day) '
            f'suggests temperature as an early warning signal for declining health.'
            f'</div>'
        )

        return make_section(
            "Early Warning Model",
            chart + explanation + table,
            section_id="early-warning",
        )

    sections.append(section_html_or_placeholder("Early Warning Model", _early_warning_section))

    # ---- Section 5: Calendar Heatmap ----
    def _calendar_section() -> str:
        fig = fig_calendar_heatmap(data, patients)
        chart = fig.to_html(include_plotlyjs=False, full_html=False)
        return make_section("Temperature Calendar Heatmap", chart, section_id="calendar")

    sections.append(section_html_or_placeholder("Calendar Heatmap", _calendar_section))

    # ---- Section 6: Known Events Validation ----
    def _events_section() -> str:
        table = _build_event_table(events)
        spikes_html = "<h3>Top Temperature Spikes (&gt; +1.0 C)</h3>" + _build_spikes_table(spikes)
        return make_section(
            "Known Events Validation",
            table + spikes_html,
            section_id="events",
        )

    sections.append(section_html_or_placeholder("Known Events Validation", _events_section))

    # ---- Section 7: Clinical Interpretation ----
    def _interpretation_section() -> str:
        h_stats = baseline_stats.get("henrik", {})
        m_stats = baseline_stats.get("mitch", {})

        bullets = []
        # Temperature variability
        h_sd = h_stats.get("std", 0)
        m_sd = m_stats.get("std", 0)
        if h_sd > 0 and m_sd > 0:
            more_var = "Henrik" if h_sd > m_sd else "Mitchell"
            ratio = max(h_sd, m_sd) / min(h_sd, m_sd) if min(h_sd, m_sd) > 0 else 0
            bullets.append(
                f"<strong>Temperature variability:</strong> {more_var} shows {ratio:.1f}x "
                f"greater temperature variability (SD {max(h_sd, m_sd):.3f} vs {min(h_sd, m_sd):.3f} C)."
            )

        # Rux effect
        if rux_stats.get("available"):
            if rux_stats["sd_reduced"] and rux_stats.get("levene_p", 1) < 0.05:
                bullets.append(
                    f"<strong>Ruxolitinib effect:</strong> Statistically significant reduction in "
                    f"temperature variability post-treatment ({rux_stats['sd_reduction_pct']:.1f}% decrease), "
                    f"consistent with its anti-inflammatory mechanism."
                )
            elif rux_stats["sd_reduced"]:
                bullets.append(
                    f"<strong>Ruxolitinib effect:</strong> Temperature variability decreased by "
                    f"{rux_stats['sd_reduction_pct']:.1f}% post-treatment, but the change was not "
                    f"statistically significant (Levene p={format_p_value(rux_stats.get('levene_p'))})."
                )

        # Early warning potential
        for pid in ["henrik", "mitch"]:
            lag1 = lag_corrs.get(pid, {}).get("temp_vs_readiness", {}).get("lag_1", {})
            if lag1 and lag1.get("p_value", 1) < 0.05:
                rho = lag1["rho"]
                direction = "negative" if rho < 0 else "positive"
                bullets.append(
                    f"<strong>Early warning ({pid.title()}):</strong> Significant {direction} "
                    f"lag-1 correlation (rho={rho:.3f}) between temperature and next-day readiness, "
                    f"suggesting temperature deviations may predict readiness changes 24 hours in advance."
                )

        # Anomaly count context
        h_anom_n = len(anomalies.get("henrik", pd.DataFrame()))
        m_anom_n = len(anomalies.get("mitch", pd.DataFrame()))
        bullets.append(
            f"<strong>Anomalies:</strong> {h_anom_n + m_anom_n} total anomaly days detected "
            f"(Henrik: {h_anom_n}, Mitchell: {m_anom_n}) using 2-SD threshold."
        )

        content = "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
        return make_section("Clinical Interpretation", content, section_id="interpretation")

    sections.append(section_html_or_placeholder("Clinical Interpretation", _interpretation_section))

    # ---- Section 8: Methodology ----
    def _methodology_section() -> str:
        content = (
            f'<div style="font-size:13px;color:{TEXT_SECONDARY};line-height:1.6;">'
            "<p><strong>Data Source:</strong> Oura Ring sleep temperature deviation "
            "(temperature_delta) measures nightly skin temperature relative to the wearer's "
            "personal baseline, in degrees Celsius. Also incorporates readiness temperature_deviation.</p>"
            "<p><strong>Anomaly Detection:</strong> Days where |temperature_delta| exceeds 2 standard "
            "deviations from the patient's own mean are flagged as anomalies.</p>"
            "<p><strong>Classification:</strong> Fever (&gt;+1.0 C), mild elevation (+0.5 to +1.0 C), "
            "normal (-0.5 to +0.5 C), mild low (-1.0 to -0.5 C), hypothermia (&lt;-1.0 C).</p>"
            "<p><strong>Early Warning Model:</strong> Spearman rank correlations at lags 0, 1, and 2 "
            "days test whether temperature deviations predict changes in readiness score or heart rate "
            "1-2 days later.</p>"
            "<p><strong>Treatment Analysis:</strong> Pre/post Ruxolitinib comparison uses Mann-Whitney U "
            "on |temperature_delta| and Levene's test for variance equality.</p>"
            "<p><strong>Cross-patient Comparison:</strong> Mann-Whitney U test with Cohen's d effect size "
            "and 95% bootstrap confidence intervals (10,000 iterations).</p>"
            "</div>"
        )
        return make_section("Methodology", content, section_id="methodology")

    sections.append(section_html_or_placeholder("Methodology", _methodology_section))

    # ---- Disclaimer ----
    sections.append(disclaimer_banner())

    body = "\n".join(sections)

    # Determine data end date
    data_end = None
    for frames in data.values():
        for df in frames.values():
            if not df.empty:
                last = df.index.max()
                if data_end is None or last > data_end:
                    data_end = last

    return wrap_html(
        title="Comparative Temperature Analysis",
        body_content=body,
        report_id="comp_temperature",
        header_meta="Henrik (post-HSCT) vs Mitchell (post-Stroke)",
        data_end=data_end,
    )


# ---------------------------------------------------------------------------
# 14. JSON Export
# ---------------------------------------------------------------------------

def export_json(
    baseline_stats: dict[str, dict[str, Any]],
    anomalies: dict[str, pd.DataFrame],
    spikes: dict[str, pd.DataFrame],
    events: list[dict[str, Any]],
    lag_corrs: dict[str, dict[str, Any]],
    rux_stats: dict[str, Any],
    cross_patient: dict[str, Any],
) -> None:
    """Write metrics JSON to disk."""
    metrics: dict[str, Any] = {
        "report_id": "comp_temperature",
        "generated": datetime.now().isoformat(),
        "baseline_stats": baseline_stats,
        "anomaly_counts": {
            pid: len(adf) for pid, adf in anomalies.items()
        },
        "top_spikes": {},
        "known_events": events,
        "lag_correlations": lag_corrs,
        "rux_pre_post": {
            k: v for k, v in rux_stats.items()
            if not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))
        },
        "cross_patient": {
            k: v for k, v in cross_patient.items()
            if not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))
        },
    }

    # Serialize top spikes
    for pid, sdf in spikes.items():
        if sdf.empty:
            metrics["top_spikes"][pid] = []
            continue
        spike_list = []
        for dt, row in sdf.head(10).iterrows():
            spike_list.append({
                "date": dt.strftime("%Y-%m-%d"),
                "temperature_delta": float(row["temperature_delta"]),
                "classification": row["classification"],
            })
        metrics["top_spikes"][pid] = spike_list

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("JSON metrics written to %s", JSON_OUTPUT)


# ---------------------------------------------------------------------------
# 15. Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run comparative temperature analysis pipeline."""
    logger.info("[1/9] Loading patient data...")
    patients = default_patients()
    if patients[1] is None:
        print("Skipping: mitch.db not found (second patient data not available)")
        return 0
    data = load_data(patients)

    logger.info("[2/9] Computing baseline statistics...")
    baseline_stats = compute_baseline_stats(data)
    for pid, s in baseline_stats.items():
        logger.info(
            "  %s: mean=%.3f, SD=%.3f, n=%d",
            pid, s.get("mean", 0), s.get("std", 0), s.get("n_days", 0),
        )

    logger.info("[3/9] Computing rolling averages...")
    rolling = compute_rolling(data)

    logger.info("[4/9] Detecting anomalies...")
    anomalies = detect_anomalies(data, baseline_stats)
    for pid, adf in anomalies.items():
        logger.info("  %s: %d anomaly days", pid, len(adf))

    logger.info("[5/9] Collecting temperature spikes...")
    spikes = get_temp_spikes(data)

    logger.info("[6/9] Validating known events...")
    events = validate_known_events(data)
    for ev in events:
        logger.info(
            "  %s %s: actual=%s, validated=%s",
            ev["patient"], ev["event"],
            ev.get("actual_delta", "N/A"), ev.get("validated"),
        )

    logger.info("[7/9] Computing lag correlations (early warning)...")
    lag_corrs = compute_lag_correlations(data)

    logger.info("[8/9] Ruxolitinib pre/post analysis...")
    rux_stats = rux_pre_post_analysis(data)
    if rux_stats.get("available"):
        logger.info(
            "  Pre SD=%.3f, Post SD=%.3f, reduction=%.1f%%",
            rux_stats["pre_sd"], rux_stats["post_sd"], rux_stats["sd_reduction_pct"],
        )

    logger.info("[9/9] Cross-patient comparison...")
    cross_patient = cross_patient_comparison(data)

    # ---- Build HTML report ----
    logger.info("Generating HTML report...")
    html = build_html(
        data, baseline_stats, rolling, anomalies, spikes,
        events, lag_corrs, rux_stats, cross_patient, patients,
    )
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", HTML_OUTPUT)

    # ---- Export JSON ----
    logger.info("Exporting JSON metrics...")
    export_json(baseline_stats, anomalies, spikes, events, lag_corrs, rux_stats, cross_patient)

    logger.info("Comparative temperature analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
