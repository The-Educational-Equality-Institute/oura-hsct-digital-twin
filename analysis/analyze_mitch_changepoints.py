#!/usr/bin/env python3
"""Mitchell Changepoint Investigation.

Deep investigation of 14 automatically discovered changepoints from the
comparative treatment analysis.  Cross-references with known life events
(Australia trip, data gaps, stroke) to validate the detection system.

Outputs:
  - Interactive HTML dashboard: reports/mitch_changepoint_investigation.html
  - JSON metrics:               reports/mitch_changepoint_metrics.json

Usage:
    python analysis/analyze_mitch_changepoints.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution & config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import REPORTS_DIR, FONT_FAMILY
from profiles import PROFILES
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    COLORWAY,
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
from _hardening import safe_connect, safe_read_sql, section_html_or_placeholder

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "mitch_changepoint_investigation.html"
JSON_OUTPUT = REPORTS_DIR / "mitch_changepoint_metrics.json"

MITCH_DB = Path(PROFILES["mitch"]["database"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANGEPOINTS = [
    {"date": date(2022, 1, 23), "consensus": 9},
    {"date": date(2021, 5, 31), "consensus": 8},
    {"date": date(2021, 10, 29), "consensus": 8},
    {"date": date(2022, 5, 6), "consensus": 8},
    {"date": date(2023, 7, 6), "consensus": 8},
    {"date": date(2023, 12, 30), "consensus": 8},
    {"date": date(2022, 8, 8), "consensus": 7},
    {"date": date(2024, 6, 3), "consensus": 6},
    {"date": date(2025, 2, 3), "consensus": 6},
    {"date": date(2024, 1, 22), "consensus": 5},
    {"date": date(2025, 2, 11), "consensus": 5},
    {"date": date(2025, 3, 25), "consensus": 5},
    {"date": date(2021, 2, 20), "consensus": 2},
    {"date": date(2025, 12, 2), "consensus": 2},
]

DATA_GAPS = [
    (date(2022, 2, 19), date(2022, 5, 3)),
    (date(2022, 5, 4), date(2022, 8, 7)),
    (date(2022, 8, 8), date(2023, 4, 29)),
    (date(2023, 9, 7), date(2023, 12, 13)),
    (date(2024, 1, 25), date(2024, 6, 2)),
    (date(2024, 6, 5), date(2025, 1, 30)),
    (date(2025, 2, 11), date(2025, 3, 25)),
    (date(2025, 3, 26), date(2025, 12, 1)),
    (date(2025, 12, 3), date(2026, 3, 25)),
]

KNOWN_EVENTS = [
    (date(2021, 12, 18), "Arrived in Australia"),
    (date(2022, 5, 4), "Left Australia (24hr flight via Singapore)"),
    (date(2024, 12, 15), "Stroke (bilateral artery dissection)"),
]

AUSTRALIA_ARRIVAL = date(2021, 12, 18)
AUSTRALIA_DEPARTURE = date(2022, 5, 4)

WINDOW_DAYS = 14
GAP_ARTIFACT_THRESHOLD = 3

METRICS = [
    ("average_hrv", "HRV (RMSSD)", "ms"),
    ("average_heart_rate", "Avg Heart Rate", "bpm"),
    ("lowest_heart_rate", "Lowest HR", "bpm"),
    ("efficiency", "Sleep Efficiency", "%"),
    ("total_sleep_duration", "Total Sleep", "sec"),
    ("average_breath", "Breath Rate", "brpm"),
]

ACTIVITY_METRICS = [
    ("steps", "Daily Steps", "steps"),
    ("active_calories", "Active Calories", "kcal"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict[str, pd.DataFrame]:
    """Load sleep periods, readiness, and activity from mitch.db."""
    conn = safe_connect(MITCH_DB)
    try:
        sleep = safe_read_sql(
            "SELECT day, average_hrv, average_heart_rate, lowest_heart_rate, "
            "total_sleep_duration, efficiency, average_breath "
            "FROM oura_sleep_periods WHERE type='long_sleep' ORDER BY day",
            conn, label="sleep_periods",
        )
        readiness = safe_read_sql(
            "SELECT date, score, recovery_index, temperature_deviation "
            "FROM oura_readiness ORDER BY date",
            conn, label="readiness",
        )
        activity = safe_read_sql(
            "SELECT date, score, steps, active_calories "
            "FROM oura_activity ORDER BY date",
            conn, label="activity",
        )
    finally:
        conn.close()

    if not sleep.empty:
        sleep["day"] = pd.to_datetime(sleep["day"], errors="coerce")
        sleep = sleep.dropna(subset=["day"]).set_index("day")
        sleep = sleep[~sleep.index.duplicated(keep="first")]

    if not readiness.empty:
        readiness["date"] = pd.to_datetime(readiness["date"], errors="coerce")
        readiness = readiness.dropna(subset=["date"]).set_index("date")
        readiness = readiness[~readiness.index.duplicated(keep="first")]

    if not activity.empty:
        activity["date"] = pd.to_datetime(activity["date"], errors="coerce")
        activity = activity.dropna(subset=["date"]).set_index("date")
        activity = activity[~activity.index.duplicated(keep="first")]

    return {"sleep": sleep, "readiness": readiness, "activity": activity}


def _merge_daily(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all data into a single daily DataFrame."""
    sleep = data["sleep"].copy()
    activity = data["activity"][["steps", "active_calories"]].copy()
    readiness = data["readiness"][["score"]].rename(columns={"score": "readiness_score"}).copy()

    merged = sleep.join(activity, how="outer")
    merged = merged.join(readiness, how="outer")
    merged.index.name = "day"
    return merged.sort_index()


# ---------------------------------------------------------------------------
# Gap proximity analysis
# ---------------------------------------------------------------------------

def gap_distance(dt: date) -> tuple[int, str, tuple[date, date]]:
    """Return (distance_days, boundary_type, gap_tuple) for nearest gap."""
    min_dist = 999999
    boundary = "none"
    nearest_gap = DATA_GAPS[0]
    for g_start, g_end in DATA_GAPS:
        d_start = abs((dt - g_start).days)
        d_end = abs((dt - g_end).days)
        if d_start < min_dist:
            min_dist = d_start
            boundary = "gap_start"
            nearest_gap = (g_start, g_end)
        if d_end < min_dist:
            min_dist = d_end
            boundary = "gap_end"
            nearest_gap = (g_start, g_end)
    return min_dist, boundary, nearest_gap


def classify_changepoint(cp_date: date, consensus: int, delta_pct: dict[str, float]) -> str:
    """Classify a changepoint as VALIDATED, ARTIFACT, GENUINE, or INVESTIGATE."""
    dist, _, _ = gap_distance(cp_date)

    # Within Australia trip window
    if AUSTRALIA_ARRIVAL <= cp_date <= AUSTRALIA_DEPARTURE:
        return "VALIDATED"
    # Right after leaving Australia (within 7 days)
    if 0 <= (cp_date - AUSTRALIA_DEPARTURE).days <= 7:
        return "VALIDATED"

    # Near a gap boundary
    if dist <= GAP_ARTIFACT_THRESHOLD:
        return "ARTIFACT"

    # Genuine: far from gaps, clear metric shifts
    significant_shifts = sum(1 for v in delta_pct.values() if abs(v) > 10)
    if dist > WINDOW_DAYS and significant_shifts >= 2:
        return "GENUINE"

    return "INVESTIGATE"


def classify_type(cp_date: date, delta_pct: dict[str, float]) -> str:
    """Classify the nature of a changepoint."""
    if AUSTRALIA_ARRIVAL <= cp_date <= AUSTRALIA_DEPARTURE + timedelta(days=7):
        return "Travel/timezone"

    dist, _, _ = gap_distance(cp_date)
    if dist <= GAP_ARTIFACT_THRESHOLD:
        return "Data artifact"

    hrv_d = delta_pct.get("average_hrv", 0)
    hr_d = delta_pct.get("average_heart_rate", 0)
    eff_d = delta_pct.get("efficiency", 0)

    # Health event: HR up, HRV down, efficiency down
    if hr_d > 5 and hrv_d < -10:
        return "Health event"

    # Recovery milestone: HRV up, HR down
    if hrv_d > 10 and hr_d < -3:
        return "Recovery milestone"

    return "Lifestyle shift"


# ---------------------------------------------------------------------------
# Per-changepoint analysis
# ---------------------------------------------------------------------------

def analyze_changepoint(
    cp_date: date,
    consensus: int,
    daily: pd.DataFrame,
) -> dict[str, Any]:
    """Compute before/after stats for a single changepoint."""
    dt = pd.Timestamp(cp_date)
    before_start = dt - timedelta(days=WINDOW_DAYS)
    after_end = dt + timedelta(days=WINDOW_DAYS)

    before = daily.loc[(daily.index >= before_start) & (daily.index < dt)]
    after = daily.loc[(daily.index >= dt) & (daily.index <= after_end)]

    all_metrics = [m[0] for m in METRICS] + [m[0] for m in ACTIVITY_METRICS]
    metric_labels = {m[0]: m[1] for m in METRICS}
    metric_labels.update({m[0]: m[1] for m in ACTIVITY_METRICS})

    stats: dict[str, Any] = {}
    delta_pct: dict[str, float] = {}

    for col in all_metrics:
        b_vals = before[col].dropna() if col in before.columns else pd.Series(dtype=float)
        a_vals = after[col].dropna() if col in after.columns else pd.Series(dtype=float)

        b_mean = float(b_vals.mean()) if len(b_vals) > 0 else None
        a_mean = float(a_vals.mean()) if len(a_vals) > 0 else None
        b_std = float(b_vals.std()) if len(b_vals) > 1 else None
        a_std = float(a_vals.std()) if len(a_vals) > 1 else None

        delta = (a_mean - b_mean) if (a_mean is not None and b_mean is not None) else None
        pct = (delta / abs(b_mean) * 100) if (delta is not None and b_mean and b_mean != 0) else 0.0
        delta_pct[col] = pct

        stats[col] = {
            "label": metric_labels.get(col, col),
            "before_mean": round(b_mean, 2) if b_mean is not None else None,
            "after_mean": round(a_mean, 2) if a_mean is not None else None,
            "before_std": round(b_std, 2) if b_std is not None else None,
            "after_std": round(a_std, 2) if a_std is not None else None,
            "delta": round(delta, 2) if delta is not None else None,
            "delta_pct": round(pct, 1),
            "before_n": len(b_vals),
            "after_n": len(a_vals),
        }

    dist, boundary, nearest_gap = gap_distance(cp_date)
    validation = classify_changepoint(cp_date, consensus, delta_pct)
    cp_type = classify_type(cp_date, delta_pct)

    # Identify top shifting metric
    top_metric = max(delta_pct, key=lambda k: abs(delta_pct[k]))

    return {
        "date": cp_date.isoformat(),
        "consensus": consensus,
        "validation": validation,
        "type": cp_type,
        "gap_distance_days": dist,
        "gap_boundary": boundary,
        "nearest_gap": (nearest_gap[0].isoformat(), nearest_gap[1].isoformat()),
        "before_days": len(before),
        "after_days": len(after.loc[after.index > dt]) + (1 if dt in after.index else 0),
        "metrics": stats,
        "delta_pct": {k: round(v, 1) for k, v in delta_pct.items()},
        "top_shifting_metric": top_metric,
        "top_shift_pct": round(delta_pct[top_metric], 1),
    }


# ---------------------------------------------------------------------------
# Long-term trajectory segmentation
# ---------------------------------------------------------------------------

def segment_trajectory(daily: pd.DataFrame) -> list[dict[str, Any]]:
    """Split data into segments between gaps and compute stats."""
    if daily.empty:
        return []

    segments: list[dict[str, Any]] = []
    # Build wearing periods from gaps
    all_dates = daily.index.sort_values()
    first_date = all_dates.min().date()
    last_date = all_dates.max().date()

    boundaries = [first_date]
    for g_start, g_end in DATA_GAPS:
        if g_start > first_date and g_end < last_date:
            boundaries.append(g_start)
            boundaries.append(g_end + timedelta(days=1))
    boundaries.append(last_date + timedelta(days=1))

    for i in range(0, len(boundaries) - 1, 2):
        seg_start = pd.Timestamp(boundaries[i])
        seg_end = pd.Timestamp(boundaries[i + 1]) if i + 1 < len(boundaries) else pd.Timestamp(last_date)
        seg_data = daily.loc[(daily.index >= seg_start) & (daily.index < seg_end)]

        if seg_data.empty or len(seg_data) < 3:
            continue

        seg_stats: dict[str, Any] = {
            "start": seg_start.date().isoformat(),
            "end": (seg_end - timedelta(days=1)).date().isoformat(),
            "days": len(seg_data),
        }
        for col, label, unit in METRICS:
            if col in seg_data.columns:
                vals = seg_data[col].dropna()
                if len(vals) > 0:
                    seg_stats[col] = {
                        "mean": round(float(vals.mean()), 2),
                        "std": round(float(vals.std()), 2) if len(vals) > 1 else 0,
                        "label": label,
                        "unit": unit,
                    }
        segments.append(seg_stats)

    return segments


# ---------------------------------------------------------------------------
# Australia trip analysis
# ---------------------------------------------------------------------------

def australia_analysis(daily: pd.DataFrame) -> dict[str, Any]:
    """Analyze Mitchell's Australia stay (Dec 18, 2021 - Feb 18, 2022 data)."""
    aus_start = pd.Timestamp(AUSTRALIA_ARRIVAL)
    # Data gap starts Feb 19, so last usable data is Feb 18
    aus_data_end = pd.Timestamp(date(2022, 2, 18))
    aus = daily.loc[(daily.index >= aus_start) & (daily.index <= aus_data_end)]

    if aus.empty:
        return {"n_days": 0, "note": "No data available for Australia period"}

    # First week (jet lag window)
    jet_lag_end = aus_start + timedelta(days=7)
    jet_lag = daily.loc[(daily.index >= aus_start) & (daily.index < jet_lag_end)]

    # Post-adaptation (after first week)
    adapted = daily.loc[(daily.index >= jet_lag_end) & (daily.index <= aus_data_end)]

    # Pre-Australia baseline: 14 days before arrival
    pre_start = aus_start - timedelta(days=14)
    pre_aus = daily.loc[(daily.index >= pre_start) & (daily.index < aus_start)]

    result: dict[str, Any] = {
        "n_days": len(aus),
        "data_range": f"{aus.index.min().date()} to {aus.index.max().date()}",
        "jet_lag_days": len(jet_lag),
        "adapted_days": len(adapted),
        "pre_baseline_days": len(pre_aus),
    }

    for col, label, unit in METRICS:
        if col not in aus.columns:
            continue
        result[col] = {
            "label": label,
            "unit": unit,
            "overall_mean": round(float(aus[col].dropna().mean()), 2) if len(aus[col].dropna()) > 0 else None,
            "jet_lag_mean": round(float(jet_lag[col].dropna().mean()), 2) if len(jet_lag[col].dropna()) > 0 else None,
            "adapted_mean": round(float(adapted[col].dropna().mean()), 2) if len(adapted[col].dropna()) > 0 else None,
            "pre_baseline_mean": round(float(pre_aus[col].dropna().mean()), 2) if len(pre_aus[col].dropna()) > 0 else None,
        }

    return result


# ---------------------------------------------------------------------------
# Generate questions for INVESTIGATE items
# ---------------------------------------------------------------------------

def generate_questions(results: list[dict[str, Any]]) -> list[str]:
    """Auto-generate questions for Mitchell about INVESTIGATE changepoints."""
    questions: list[str] = []
    for r in results:
        if r["validation"] != "INVESTIGATE":
            continue
        cp_date = r["date"]
        top_metric = r["metrics"].get(r["top_shifting_metric"], {})
        label = top_metric.get("label", r["top_shifting_metric"])
        pct = r["top_shift_pct"]
        direction = "increased" if pct > 0 else "decreased"
        questions.append(
            f"Around {cp_date}: Your {label} {direction} by {abs(pct):.0f}%. "
            f"Did anything notable happen around this date? "
            f"(illness, travel, lifestyle change, medication, stress event?)"
        )
    return questions


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _embed(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _yref(row: int) -> str:
    """Return the correct yref for a subplot row (1-indexed)."""
    return "y domain" if row == 1 else f"y{row} domain"


def _fig_full_timeline(daily: pd.DataFrame, results: list[dict[str, Any]]) -> go.Figure:
    """Fig 1: Full 5-year HRV + HR timeline with changepoints and gaps."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["HRV (RMSSD)", "Average Heart Rate"],
        row_heights=[0.5, 0.5],
    )

    # HRV trace
    if "average_hrv" in daily.columns:
        hrv = daily["average_hrv"].dropna()
        fig.add_trace(
            go.Scatter(
                x=hrv.index, y=hrv.values,
                mode="lines",
                name="HRV",
                line=dict(color=ACCENT_PURPLE, width=1),
                opacity=0.7,
            ),
            row=1, col=1,
        )
        # 14-day rolling average
        hrv_roll = hrv.rolling(14, min_periods=3).mean()
        fig.add_trace(
            go.Scatter(
                x=hrv_roll.index, y=hrv_roll.values,
                mode="lines",
                name="HRV 14d avg",
                line=dict(color=ACCENT_PURPLE, width=2.5),
            ),
            row=1, col=1,
        )

    # HR trace
    if "average_heart_rate" in daily.columns:
        hr = daily["average_heart_rate"].dropna()
        fig.add_trace(
            go.Scatter(
                x=hr.index, y=hr.values,
                mode="lines",
                name="Avg HR",
                line=dict(color=ACCENT_GREEN, width=1),
                opacity=0.7,
            ),
            row=2, col=1,
        )
        hr_roll = hr.rolling(14, min_periods=3).mean()
        fig.add_trace(
            go.Scatter(
                x=hr_roll.index, y=hr_roll.values,
                mode="lines",
                name="HR 14d avg",
                line=dict(color=ACCENT_GREEN, width=2.5),
            ),
            row=2, col=1,
        )

    # Data gaps as gray shaded bands
    for g_start, g_end in DATA_GAPS:
        for row in [1, 2]:
            fig.add_shape(
                type="rect",
                x0=str(g_start), x1=str(g_end),
                y0=0, y1=1, yref=_yref(row),
                fillcolor="rgba(107,114,128,0.15)",
                line=dict(width=0),
                row=row, col=1,
            )

    # Changepoints as vertical lines
    validation_colors = {
        "VALIDATED": ACCENT_GREEN,
        "ARTIFACT": TEXT_TERTIARY,
        "GENUINE": ACCENT_BLUE,
        "INVESTIGATE": ACCENT_AMBER,
    }

    for r in results:
        cp_dt = r["date"]
        color = validation_colors.get(r["validation"], ACCENT_AMBER)
        for row in [1, 2]:
            fig.add_shape(
                type="line",
                x0=cp_dt, x1=cp_dt,
                y0=0, y1=1, yref=_yref(row),
                line=dict(color=color, width=1.5, dash="dot"),
                row=row, col=1,
            )
        # Label on top subplot only
        fig.add_annotation(
            x=cp_dt, y=1.02, yref=_yref(1),
            text=f"C{r['consensus']}",
            showarrow=False,
            font=dict(size=9, color=color),
            row=1, col=1,
        )

    # Known events
    for ev_date, ev_label in KNOWN_EVENTS:
        for row in [1, 2]:
            fig.add_shape(
                type="line",
                x0=str(ev_date), x1=str(ev_date),
                y0=0, y1=1, yref=_yref(row),
                line=dict(color=ACCENT_RED, width=2, dash="solid"),
                row=row, col=1,
            )
        fig.add_annotation(
            x=str(ev_date), y=1.06, yref=_yref(1),
            text=ev_label,
            showarrow=False,
            font=dict(size=10, color=ACCENT_RED),
        )

    fig.update_layout(
        height=550,
        font=dict(family=FONT_FAMILY, size=12),
        showlegend=True,
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_yaxes(title_text="ms", row=1, col=1)
    fig.update_yaxes(title_text="bpm", row=2, col=1)

    return fig


def _fig_top_changepoints(results: list[dict[str, Any]]) -> go.Figure:
    """Fig 2: Before/after grouped bars for top 6 changepoints."""
    top6 = sorted(results, key=lambda r: r["consensus"], reverse=True)[:6]
    display_metrics = ["average_hrv", "average_heart_rate", "efficiency"]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{r['date']} (C={r['consensus']})" for r in top6],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    metric_colors = {
        "average_hrv": ACCENT_PURPLE,
        "average_heart_rate": ACCENT_GREEN,
        "efficiency": ACCENT_BLUE,
    }

    for idx, r in enumerate(top6):
        row = idx // 3 + 1
        col = idx % 3 + 1

        before_vals = []
        after_vals = []
        labels = []

        for m_key in display_metrics:
            m = r["metrics"].get(m_key, {})
            b = m.get("before_mean")
            a = m.get("after_mean")
            if b is not None and a is not None:
                before_vals.append(b)
                after_vals.append(a)
                labels.append(m.get("label", m_key))

        if not labels:
            continue

        fig.add_trace(
            go.Bar(
                x=labels, y=before_vals,
                name="Before" if idx == 0 else None,
                marker_color="rgba(107,114,128,0.6)",
                showlegend=(idx == 0),
                legendgroup="before",
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Bar(
                x=labels, y=after_vals,
                name="After" if idx == 0 else None,
                marker_color=ACCENT_CYAN,
                showlegend=(idx == 0),
                legendgroup="after",
            ),
            row=row, col=col,
        )

    fig.update_layout(
        height=500,
        barmode="group",
        font=dict(family=FONT_FAMILY, size=11),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        margin=dict(l=50, r=30, t=60, b=60),
    )

    return fig


def _fig_australia(daily: pd.DataFrame) -> go.Figure:
    """Fig 3: Australia trip panel (Dec 2021 - Feb 2022)."""
    aus_start = pd.Timestamp(date(2021, 12, 4))  # 2 weeks before arrival
    aus_end = pd.Timestamp(date(2022, 2, 20))
    aus_data = daily.loc[(daily.index >= aus_start) & (daily.index <= aus_end)]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["HRV (RMSSD)", "Average Heart Rate", "Sleep Efficiency"],
        row_heights=[0.34, 0.33, 0.33],
    )

    arrival_str = str(AUSTRALIA_ARRIVAL)

    for row_idx, (col, label, unit) in enumerate(
        [("average_hrv", "HRV", "ms"), ("average_heart_rate", "HR", "bpm"), ("efficiency", "Efficiency", "%")],
        start=1,
    ):
        if col in aus_data.columns:
            vals = aus_data[col].dropna()
            color = [ACCENT_PURPLE, ACCENT_GREEN, ACCENT_BLUE][row_idx - 1]
            fig.add_trace(
                go.Scatter(
                    x=vals.index, y=vals.values,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                ),
                row=row_idx, col=1,
            )
        fig.update_yaxes(title_text=unit, row=row_idx, col=1)

        # Arrival line
        fig.add_shape(
            type="line",
            x0=arrival_str, x1=arrival_str,
            y0=0, y1=1, yref=_yref(row_idx),
            line=dict(color=ACCENT_RED, width=2, dash="solid"),
            row=row_idx, col=1,
        )

        # Jet lag zone (first 7 days)
        jet_lag_end = str(AUSTRALIA_ARRIVAL + timedelta(days=7))
        fig.add_shape(
            type="rect",
            x0=arrival_str, x1=jet_lag_end,
            y0=0, y1=1, yref=_yref(row_idx),
            fillcolor="rgba(239,68,68,0.08)",
            line=dict(width=0),
            row=row_idx, col=1,
        )

    # Changepoint 2022-01-23 marker
    cp_jan23 = str(date(2022, 1, 23))
    for row_idx in range(1, 4):
        fig.add_shape(
            type="line",
            x0=cp_jan23, x1=cp_jan23,
            y0=0, y1=1, yref=_yref(row_idx),
            line=dict(color=ACCENT_AMBER, width=2, dash="dot"),
            row=row_idx, col=1,
        )

    fig.add_annotation(
        x=arrival_str, y=1.08, yref=_yref(1),
        text="Arrived in Australia",
        showarrow=False,
        font=dict(size=10, color=ACCENT_RED),
    )
    fig.add_annotation(
        x=cp_jan23, y=1.08, yref=_yref(1),
        text="Changepoint (C=9)",
        showarrow=False,
        font=dict(size=10, color=ACCENT_AMBER),
    )

    fig.update_layout(
        height=550,
        font=dict(family=FONT_FAMILY, size=12),
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=40),
    )

    return fig


def _fig_gap_proximity(results: list[dict[str, Any]]) -> go.Figure:
    """Fig 4: Horizontal bar chart of gap distance per changepoint."""
    results_sorted = sorted(results, key=lambda r: r["date"])

    dates = [r["date"] for r in results_sorted]
    distances = [r["gap_distance_days"] for r in results_sorted]
    validations = [r["validation"] for r in results_sorted]

    color_map = {
        "ARTIFACT": TEXT_TERTIARY,
        "VALIDATED": ACCENT_GREEN,
        "GENUINE": ACCENT_BLUE,
        "INVESTIGATE": ACCENT_AMBER,
    }
    colors = [color_map.get(v, ACCENT_AMBER) for v in validations]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=[f"{d} (C={r['consensus']})" for d, r in zip(dates, results_sorted)],
            x=distances,
            orientation="h",
            marker_color=colors,
            text=[f"{d}d - {v}" for d, v in zip(distances, validations)],
            textposition="outside",
            textfont=dict(size=10),
        )
    )

    # Artifact threshold line
    fig.add_shape(
        type="line",
        x0=GAP_ARTIFACT_THRESHOLD, x1=GAP_ARTIFACT_THRESHOLD,
        y0=-0.5, y1=len(results_sorted) - 0.5,
        line=dict(color=ACCENT_RED, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=GAP_ARTIFACT_THRESHOLD, y=len(results_sorted) - 0.5,
        text=f"Artifact threshold ({GAP_ARTIFACT_THRESHOLD}d)",
        showarrow=False,
        font=dict(size=10, color=ACCENT_RED),
        yshift=15,
    )

    fig.update_layout(
        height=max(400, len(results_sorted) * 35 + 80),
        font=dict(family=FONT_FAMILY, size=12),
        xaxis_title="Days to nearest data gap boundary",
        margin=dict(l=150, r=80, t=30, b=50),
        showlegend=False,
    )

    return fig


def _fig_validation_table(results: list[dict[str, Any]]) -> str:
    """Fig 5: Validation summary as an HTML table."""
    validation_badges = {
        "VALIDATED": f'<span style="background:rgba(16,185,129,0.15);color:{ACCENT_GREEN};padding:2px 8px;border-radius:4px;font-size:12px;">VALIDATED</span>',
        "ARTIFACT": f'<span style="background:rgba(107,114,128,0.15);color:{TEXT_TERTIARY};padding:2px 8px;border-radius:4px;font-size:12px;">ARTIFACT</span>',
        "GENUINE": f'<span style="background:rgba(59,130,246,0.15);color:{ACCENT_BLUE};padding:2px 8px;border-radius:4px;font-size:12px;">GENUINE</span>',
        "INVESTIGATE": f'<span style="background:rgba(245,158,11,0.15);color:{ACCENT_AMBER};padding:2px 8px;border-radius:4px;font-size:12px;">INVESTIGATE</span>',
    }

    rows_html = ""
    for r in sorted(results, key=lambda x: x["consensus"], reverse=True):
        badge = validation_badges.get(r["validation"], r["validation"])
        top_m = r["metrics"].get(r["top_shifting_metric"], {})
        top_label = top_m.get("label", r["top_shifting_metric"])
        shift_color = ACCENT_GREEN if r["top_shift_pct"] > 0 else ACCENT_RED
        shift_arrow = "+" if r["top_shift_pct"] > 0 else ""

        rows_html += f"""<tr>
            <td style="font-weight:600;">{r['date']}</td>
            <td style="text-align:center;">{r['consensus']}</td>
            <td>{badge}</td>
            <td>{r['type']}</td>
            <td>{r['gap_distance_days']}d</td>
            <td>{top_label}</td>
            <td style="color:{shift_color};font-weight:600;">{shift_arrow}{r['top_shift_pct']:.0f}%</td>
            <td>{r['before_days']}d / {r['after_days']}d</td>
        </tr>"""

    return f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
            <tr style="border-bottom:2px solid {BORDER_SUBTLE};color:{TEXT_SECONDARY};">
                <th style="text-align:left;padding:10px 8px;">Date</th>
                <th style="text-align:center;padding:10px 8px;">Consensus</th>
                <th style="text-align:left;padding:10px 8px;">Validation</th>
                <th style="text-align:left;padding:10px 8px;">Type</th>
                <th style="text-align:left;padding:10px 8px;">Gap Dist.</th>
                <th style="text-align:left;padding:10px 8px;">Top Metric</th>
                <th style="text-align:left;padding:10px 8px;">Shift</th>
                <th style="text-align:left;padding:10px 8px;">Data (B/A)</th>
            </tr>
        </thead>
        <tbody style="color:{TEXT_PRIMARY};">{rows_html}</tbody>
    </table>
    </div>
    """


# ---------------------------------------------------------------------------
# Long-term trajectory table
# ---------------------------------------------------------------------------

def _trajectory_table(segments: list[dict[str, Any]]) -> str:
    """Render segments as an HTML table."""
    if not segments:
        return f'<p style="color:{TEXT_SECONDARY};">No segments to display.</p>'

    rows_html = ""
    for seg in segments:
        hrv_info = seg.get("average_hrv", {})
        hr_info = seg.get("average_heart_rate", {})
        eff_info = seg.get("efficiency", {})

        hrv_str = f"{hrv_info['mean']:.1f} +/- {hrv_info['std']:.1f}" if hrv_info else "N/A"
        hr_str = f"{hr_info['mean']:.1f} +/- {hr_info['std']:.1f}" if hr_info else "N/A"
        eff_str = f"{eff_info['mean']:.0f}%" if eff_info else "N/A"

        rows_html += f"""<tr>
            <td>{seg['start']}</td>
            <td>{seg['end']}</td>
            <td style="text-align:center;">{seg['days']}</td>
            <td style="text-align:center;">{hrv_str}</td>
            <td style="text-align:center;">{hr_str}</td>
            <td style="text-align:center;">{eff_str}</td>
        </tr>"""

    return f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
            <tr style="border-bottom:2px solid {BORDER_SUBTLE};color:{TEXT_SECONDARY};">
                <th style="text-align:left;padding:10px 8px;">Start</th>
                <th style="text-align:left;padding:10px 8px;">End</th>
                <th style="text-align:center;padding:10px 8px;">Days</th>
                <th style="text-align:center;padding:10px 8px;">HRV (ms)</th>
                <th style="text-align:center;padding:10px 8px;">HR (bpm)</th>
                <th style="text-align:center;padding:10px 8px;">Efficiency</th>
            </tr>
        </thead>
        <tbody style="color:{TEXT_PRIMARY};">{rows_html}</tbody>
    </table>
    </div>
    """


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(
    results: list[dict[str, Any]],
    daily: pd.DataFrame,
    segments: list[dict[str, Any]],
    aus_info: dict[str, Any],
    questions: list[str],
) -> str:
    """Assemble the full HTML report."""

    # Classification counts
    n_validated = sum(1 for r in results if r["validation"] == "VALIDATED")
    n_artifact = sum(1 for r in results if r["validation"] == "ARTIFACT")
    n_genuine = sum(1 for r in results if r["validation"] == "GENUINE")
    n_investigate = sum(1 for r in results if r["validation"] == "INVESTIGATE")

    # -- KPI row --
    kpi = make_kpi_row(
        make_kpi_card("TOTAL CHANGEPOINTS", len(results), status="info", decimals=0),
        make_kpi_card("VALIDATED", n_validated, status="good", decimals=0,
                      detail="Near known events"),
        make_kpi_card("ARTIFACT", n_artifact, status="neutral", decimals=0,
                      detail="Near gap boundaries"),
        make_kpi_card("GENUINE", n_genuine, status="info", decimals=0,
                      detail="Clear metric shifts"),
        make_kpi_card("INVESTIGATE", n_investigate, status="warning", decimals=0,
                      detail="Questions for Mitchell"),
    )

    # -- Executive Summary --
    top_consensus = max(r["consensus"] for r in results)
    top_cp = [r for r in results if r["consensus"] == top_consensus][0]

    summary_html = f"""
    <div style="color:{TEXT_SECONDARY};line-height:1.8;font-size:14px;">
    <p>Analysis of <strong style="color:{TEXT_PRIMARY};">14 changepoints</strong>
    discovered by the 4-method consensus algorithm across 5 years of Oura Ring data.
    Mitchell's data spans from early 2021 with significant gaps
    totaling approximately {sum((g[1] - g[0]).days for g in DATA_GAPS)} days of missing data
    across {len(DATA_GAPS)} gap periods.</p>

    <p>The highest-consensus changepoint (<strong style="color:{ACCENT_PURPLE};">C={top_cp['consensus']}</strong>
    on {top_cp['date']}) falls during Mitchell's Australia stay (Dec 2021 - May 2022),
    suggesting genuine physiological adaptation to the timezone shift and environment change.</p>

    <p>Of the 14 changepoints: <strong style="color:{ACCENT_GREEN};">{n_validated}</strong> validated
    against known events, <strong style="color:{TEXT_TERTIARY};">{n_artifact}</strong> classified as
    data artifacts (within {GAP_ARTIFACT_THRESHOLD} days of gap boundary),
    <strong style="color:{ACCENT_BLUE};">{n_genuine}</strong> genuine metric shifts, and
    <strong style="color:{ACCENT_AMBER};">{n_investigate}</strong> requiring follow-up with Mitchell.</p>
    </div>
    """

    # -- Section 1: Full Timeline --
    sec_timeline = section_html_or_placeholder(
        "Full Timeline",
        lambda: make_section(
            "5-Year Biometric Timeline with Changepoints",
            f"""<p style="color:{TEXT_SECONDARY};margin-bottom:12px;">
            Vertical lines mark changepoints (colored by validation status).
            Gray bands indicate data gap periods. Red lines mark known life events.
            Labels show consensus score (C=N).</p>"""
            + _embed(_fig_full_timeline(daily, results)),
            section_id="timeline",
        ),
    )

    # -- Section 2: Top Changepoints Detail --
    sec_top = section_html_or_placeholder(
        "Top Changepoints Detail",
        lambda: make_section(
            "Top 6 Changepoints: Before vs. After",
            f"""<p style="color:{TEXT_SECONDARY};margin-bottom:12px;">
            Comparison of key metrics in the {WINDOW_DAYS}-day window before and after
            each high-consensus changepoint. Grouped by HRV, Heart Rate, and Sleep Efficiency.</p>"""
            + _embed(_fig_top_changepoints(results)),
            section_id="top-changepoints",
        ),
    )

    # -- Section 3: Australia Trip --
    aus_content_parts = []
    if aus_info.get("n_days", 0) > 0:
        aus_content_parts.append(
            f"""<p style="color:{TEXT_SECONDARY};margin-bottom:12px;">
            Mitchell arrived in Australia on Dec 18, 2021. Data is available for
            {aus_info['n_days']} days ({aus_info.get('data_range', 'N/A')}). The red shaded zone
            marks the first 7 days (jet lag recovery window). The amber line marks
            the C=9 changepoint on Jan 23, 2022.</p>"""
        )

        # Stat cards for Australia
        aus_metrics_html = ""
        for col, label, unit in [("average_hrv", "HRV", "ms"), ("average_heart_rate", "HR", "bpm")]:
            m = aus_info.get(col, {})
            if m and m.get("pre_baseline_mean") is not None and m.get("overall_mean") is not None:
                delta = m["overall_mean"] - m["pre_baseline_mean"]
                sign = "+" if delta > 0 else ""
                aus_metrics_html += (
                    f'<div style="display:inline-block;margin:8px 12px 8px 0;padding:8px 16px;'
                    f'background:{BG_ELEVATED};border-radius:8px;font-size:13px;">'
                    f'<span style="color:{TEXT_SECONDARY};">{label}:</span> '
                    f'<span style="color:{TEXT_PRIMARY};font-weight:600;">'
                    f'Pre {m["pre_baseline_mean"]:.1f} -> Aus {m["overall_mean"]:.1f} '
                    f'({sign}{delta:.1f} {unit})</span></div>'
                )
        if aus_metrics_html:
            aus_content_parts.append(aus_metrics_html)

        aus_content_parts.append(_embed(_fig_australia(daily)))
    else:
        aus_content_parts.append(
            f'<p style="color:{TEXT_SECONDARY};">Insufficient data for Australia analysis.</p>'
        )

    sec_australia = section_html_or_placeholder(
        "Australia Trip Analysis",
        lambda: make_section(
            "Australia Trip Deep Dive (Dec 2021 - Feb 2022)",
            "\n".join(aus_content_parts),
            section_id="australia",
        ),
    )

    # -- Section 4: Gap Proximity --
    sec_gap = section_html_or_placeholder(
        "Gap Proximity Analysis",
        lambda: make_section(
            "Gap Proximity Analysis",
            f"""<p style="color:{TEXT_SECONDARY};margin-bottom:12px;">
            Distance (in days) from each changepoint to the nearest data gap boundary.
            Changepoints within {GAP_ARTIFACT_THRESHOLD} days of a gap are classified as
            potential artifacts. Color indicates validation status.</p>"""
            + _embed(_fig_gap_proximity(results)),
            section_id="gap-proximity",
        ),
    )

    # -- Section 5: Validation Summary --
    sec_validation = section_html_or_placeholder(
        "Validation Summary",
        lambda: make_section(
            "Changepoint Validation Summary",
            f"""<p style="color:{TEXT_SECONDARY};margin-bottom:12px;">
            All 14 changepoints ranked by consensus score with classification,
            dominant metric shift, and data availability.</p>"""
            + _fig_validation_table(results),
            section_id="validation",
        ),
    )

    # -- Section 6: Long-Term Trajectory --
    sec_trajectory = section_html_or_placeholder(
        "Long-Term Trajectory",
        lambda: make_section(
            "Long-Term Trajectory Segmentation",
            f"""<p style="color:{TEXT_SECONDARY};margin-bottom:12px;">
            Mitchell's biometric data split into wearing segments between gap periods.
            Shows baseline shifts across years.</p>"""
            + _trajectory_table(segments),
            section_id="trajectory",
        ),
    )

    # -- Section 7: Questions for Mitchell --
    if questions:
        q_items = "".join(
            f'<li style="margin:8px 0;color:{TEXT_PRIMARY};">{q}</li>'
            for q in questions
        )
        q_html = (
            f"""<p style="color:{TEXT_SECONDARY};margin-bottom:12px;">
            The following changepoints could not be explained by data gaps or known events.
            Answers from Mitchell would help validate the detection system.</p>
            <ol style="padding-left:20px;">{q_items}</ol>"""
        )
    else:
        q_html = f'<p style="color:{TEXT_SECONDARY};">No unresolved changepoints require follow-up.</p>'

    sec_questions = section_html_or_placeholder(
        "Questions for Mitchell",
        lambda: make_section(
            "Questions for Mitchell",
            q_html,
            section_id="questions",
        ),
    )

    body = "\n".join([
        kpi,
        make_section("Executive Summary", summary_html, section_id="summary"),
        sec_timeline,
        sec_top,
        sec_australia,
        sec_gap,
        sec_validation,
        sec_trajectory,
        sec_questions,
    ])

    return wrap_html(
        title="Mitch Changepoint Investigation",
        body_content=body,
        report_id="mitch_changepoints",
        header_meta="Mitchell — Changepoint Investigation",
        post_days=0,
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(
    results: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    aus_info: dict[str, Any],
    questions: list[str],
) -> None:
    """Write structured metrics JSON."""
    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "report": "mitch_changepoint_investigation",
        "total_changepoints": len(results),
        "classifications": {
            "validated": sum(1 for r in results if r["validation"] == "VALIDATED"),
            "artifact": sum(1 for r in results if r["validation"] == "ARTIFACT"),
            "genuine": sum(1 for r in results if r["validation"] == "GENUINE"),
            "investigate": sum(1 for r in results if r["validation"] == "INVESTIGATE"),
        },
        "changepoints": results,
        "trajectory_segments": segments,
        "australia_analysis": aus_info,
        "questions_for_mitchell": questions,
        "data_gaps": [
            {"start": g[0].isoformat(), "end": g[1].isoformat(), "days": (g[1] - g[0]).days}
            for g in DATA_GAPS
        ],
        "known_events": [
            {"date": e[0].isoformat(), "label": e[1]}
            for e in KNOWN_EVENTS
        ],
    }

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("JSON metrics written to %s", JSON_OUTPUT)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run Mitchell changepoint investigation pipeline."""
    if not MITCH_DB.exists():
        print("Skipping: mitch.db not found (second patient data not available)")
        return 0
    logger.info("[1/7] Loading Mitchell data from %s...", MITCH_DB)
    data = load_data()
    daily = _merge_daily(data)
    logger.info("  Loaded %d daily records (%s to %s)",
                len(daily),
                daily.index.min().date() if not daily.empty else "N/A",
                daily.index.max().date() if not daily.empty else "N/A")

    logger.info("[2/7] Analyzing 14 changepoints...")
    results: list[dict[str, Any]] = []
    for cp in CHANGEPOINTS:
        r = analyze_changepoint(cp["date"], cp["consensus"], daily)
        results.append(r)
        logger.info("  %s (C=%d): %s / %s / gap=%dd",
                     r["date"], r["consensus"], r["validation"], r["type"],
                     r["gap_distance_days"])

    logger.info("[3/7] Segmenting long-term trajectory...")
    segments = segment_trajectory(daily)
    logger.info("  Found %d wearing segments", len(segments))

    logger.info("[4/7] Australia trip analysis...")
    aus_info = australia_analysis(daily)
    logger.info("  Australia data: %d days", aus_info.get("n_days", 0))

    logger.info("[5/7] Generating questions for INVESTIGATE items...")
    questions = generate_questions(results)
    logger.info("  Generated %d questions", len(questions))

    logger.info("[6/7] Building HTML report...")
    html = build_html(results, daily, segments, aus_info, questions)
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", HTML_OUTPUT)

    logger.info("[7/7] Exporting JSON metrics...")
    export_json(results, segments, aus_info, questions)

    logger.info("Mitchell changepoint investigation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
