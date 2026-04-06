#!/usr/bin/env python3
"""Module 1: Autonomic Recovery Trajectories.

Compares HRV and resting HR recovery patterns between Henrik (post-HSCT)
and Mitchell (post-stroke), normalized to days-since-their-major-event.

Outputs:
  - Interactive HTML dashboard: reports/comparative_autonomic_report.html
  - JSON metrics:               reports/comparative_autonomic_metrics.json

Usage:
    python analysis/analyze_comparative_autonomic.py
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
    ESC_RMSSD_DEFICIENCY,
    POPULATION_RMSSD_MEDIAN,
    POPULATION_RMSSD_MEAN,
    POPULATION_RMSSD_SD,
    FONT_FAMILY,
    TREATMENT_START,
)
from _comparative_utils import (
    PatientConfig,
    default_patients,
    load_patient_data,
    zscore_normalize,
    zscore_both,
    percentile_of_self,
    find_date_overlap,
    align_by_event,
    days_since_event,
    compare_distributions,
    dual_patient_timeseries,
    dual_patient_distribution,
    event_aligned_comparison,
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
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_PURPLE,
    ACCENT_CYAN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from _hardening import safe_connect, safe_read_sql, section_html_or_placeholder

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "comparative_autonomic_report.html"
JSON_OUTPUT = REPORTS_DIR / "comparative_autonomic_metrics.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trend_direction(slope: float, p_value: float) -> str:
    """Classify trend direction from linear regression slope + p-value."""
    if p_value > 0.10 or np.isnan(p_value):
        return "stable"
    return "improving" if slope > 0 else "declining"


def _compute_linear_trend(series: pd.Series) -> dict:
    """Compute linear regression slope (per week) + p-value on a numeric series."""
    clean = series.dropna()
    if len(clean) < 5:
        return {"slope_per_week": np.nan, "p_value": np.nan, "direction": "insufficient data"}
    x = np.arange(len(clean), dtype=float)
    y = clean.values.astype(float)
    slope, intercept, r, p, se = scipy_stats.linregress(x, y)
    slope_per_week = slope * 7
    return {
        "slope_per_week": float(slope_per_week),
        "p_value": float(p),
        "direction": _trend_direction(slope, p),
    }


def _population_percentile(hrv_mean: float) -> float:
    """Map an HRV mean to population percentile using log-normal approximation."""
    if np.isnan(hrv_mean) or hrv_mean <= 0:
        return 0.0
    z = (hrv_mean - POPULATION_RMSSD_MEAN) / POPULATION_RMSSD_SD
    return float(scipy_stats.norm.cdf(z) * 100)


def _embed(fig: go.Figure) -> str:
    """Embed a Plotly figure as inline HTML (no JS bundle)."""
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _add_reference_line(
    fig: go.Figure,
    y_val: float,
    label: str,
    color: str,
    dash: str = "dash",
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add a horizontal reference line with annotation."""
    kwargs: dict[str, Any] = {}
    if row is not None and col is not None:
        kwargs["row"] = row
        kwargs["col"] = col
    fig.add_hline(
        y=y_val,
        line_dash=dash,
        line_color=color,
        line_width=1,
        opacity=0.6,
        annotation_text=label,
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color=color,
        **kwargs,
    )


def _add_event_vline(
    fig: go.Figure,
    x_val: Any,
    label: str,
    color: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add a vertical event marker using shape + annotation (never add_vline with annotation_text)."""
    yref = "paper"
    fig.add_shape(
        type="line",
        x0=x_val, x1=x_val,
        y0=0, y1=1, yref=yref,
        line=dict(color=color, width=1.5, dash="dash"),
        opacity=0.5,
        row=row, col=col,
    )
    fig.add_annotation(
        x=x_val, y=1.02, yref=yref,
        text=label,
        showarrow=False,
        font=dict(size=9, color=color),
        row=row, col=col,
    )


# ---------------------------------------------------------------------------
# [1/7] Data Loading
# ---------------------------------------------------------------------------

def load_data(
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, dict[str, pd.Series]]:
    """Load HRV and HR data for both patients.

    Returns dict keyed by patient_id, each containing:
      - "hrv": pd.Series of average HRV (ms)
      - "hr":  pd.Series of sleep heart rate (bpm)
      - "hr_lowest": pd.Series of lowest sleep heart rate (bpm)
    """
    result: dict[str, dict[str, pd.Series]] = {}

    for p in patients:
        # Load from oura_sleep_periods (type='long_sleep' filter built into load_patient_data)
        sp = load_patient_data(p, "oura_sleep_periods", columns="day, average_hrv, average_heart_rate, lowest_heart_rate")

        hrv = sp["average_hrv"].dropna() if not sp.empty and "average_hrv" in sp.columns else pd.Series(dtype=float)
        hr = sp["average_heart_rate"].dropna() if not sp.empty and "average_heart_rate" in sp.columns else pd.Series(dtype=float)
        hr_low = sp["lowest_heart_rate"].dropna() if not sp.empty and "lowest_heart_rate" in sp.columns else pd.Series(dtype=float)

        hrv.name = "hrv"
        hr.name = "hr"
        hr_low.name = "hr_lowest"

        result[p.patient_id] = {"hrv": hrv, "hr": hr, "hr_lowest": hr_low}
        logger.info(
            "Loaded %s: %d HRV days, %d HR days",
            p.display_name, len(hrv), len(hr),
        )

    return result


# ---------------------------------------------------------------------------
# [2/7] Timeline Normalization
# ---------------------------------------------------------------------------

def normalize_timelines(
    data: dict[str, dict[str, pd.Series]],
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, dict[str, Any]]:
    """Add days_since_event to each patient's data."""
    patient_map = {p.patient_id: p for p in patients}
    enriched: dict[str, dict[str, Any]] = {}

    for pid, metrics in data.items():
        p = patient_map[pid]
        dse = days_since_event(metrics["hrv"].index, p.event_date) if not metrics["hrv"].empty else pd.Series(dtype=int)
        enriched[pid] = {
            **metrics,
            "days_since_event": dse,
            "event_date": p.event_date,
            "event_label": p.event_label,
        }
    return enriched


# ---------------------------------------------------------------------------
# [3/7] Rolling Metrics
# ---------------------------------------------------------------------------

def compute_rolling(
    data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute rolling averages, CV, and trailing slope for HRV and HR."""
    for pid, metrics in data.items():
        hrv = metrics["hrv"]
        hr = metrics["hr"]

        if not hrv.empty:
            metrics["hrv_7d"] = hrv.rolling(7, min_periods=4).mean()
            metrics["hrv_14d"] = hrv.rolling(14, min_periods=7).mean()
            rolling_std = hrv.rolling(7, min_periods=4).std()
            rolling_mean = hrv.rolling(7, min_periods=4).mean()
            metrics["hrv_cv_7d"] = rolling_std / rolling_mean.replace(0, np.nan)
        else:
            metrics["hrv_7d"] = pd.Series(dtype=float)
            metrics["hrv_14d"] = pd.Series(dtype=float)
            metrics["hrv_cv_7d"] = pd.Series(dtype=float)

        if not hr.empty:
            metrics["hr_7d"] = hr.rolling(7, min_periods=4).mean()
            metrics["hr_14d"] = hr.rolling(14, min_periods=7).mean()
        else:
            metrics["hr_7d"] = pd.Series(dtype=float)
            metrics["hr_14d"] = pd.Series(dtype=float)

    return data


# ---------------------------------------------------------------------------
# [4/7] Normalized Metrics
# ---------------------------------------------------------------------------

def compute_normalized(
    data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Z-score, percent-of-baseline, and population percentile for HRV."""
    for pid, metrics in data.items():
        hrv = metrics["hrv"]

        if hrv.empty or len(hrv) < 3:
            metrics["hrv_zscore"] = pd.Series(dtype=float)
            metrics["hrv_pct_baseline"] = pd.Series(dtype=float)
            metrics["hrv_pop_percentile"] = 0.0
            continue

        # Z-score normalization
        nr = zscore_normalize(hrv, patient_id=pid)
        metrics["hrv_zscore"] = nr.z_scores

        # Percent-of-baseline (first 14 days)
        first14 = hrv.iloc[:14]
        baseline_mean = first14.mean() if len(first14) > 0 else hrv.mean()
        if baseline_mean == 0 or np.isnan(baseline_mean):
            baseline_mean = 1.0
        metrics["hrv_pct_baseline"] = (hrv / baseline_mean) * 100
        metrics["hrv_baseline_mean"] = baseline_mean

        # Population percentile
        metrics["hrv_pop_percentile"] = _population_percentile(float(hrv.mean()))

    return data


# ---------------------------------------------------------------------------
# [5/7] Trends & Comparison
# ---------------------------------------------------------------------------

def compute_trends_and_comparison(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, Any]:
    """Compute per-patient trends and cross-patient comparison metrics."""
    patient_map = {p.patient_id: p for p in patients}
    stats_result: dict[str, Any] = {"patients": {}, "comparison": {}}

    for pid, metrics in data.items():
        p = patient_map[pid]
        hrv = metrics["hrv"]
        hr = metrics["hr"]
        hr_low = metrics["hr_lowest"]

        # Last 30 days for trend
        hrv_last30 = hrv.tail(30) if not hrv.empty else pd.Series(dtype=float)
        hr_last30 = hr.tail(30) if not hr.empty else pd.Series(dtype=float)

        hrv_trend = _compute_linear_trend(hrv_last30)
        hr_trend = _compute_linear_trend(hr_last30)

        dse = metrics.get("days_since_event", pd.Series(dtype=int))
        dse_range = [int(dse.min()), int(dse.max())] if not dse.empty else [0, 0]

        pct_below_esc = (
            float((hrv < ESC_RMSSD_DEFICIENCY).sum() / len(hrv) * 100)
            if not hrv.empty else 0.0
        )

        stats_result["patients"][pid] = {
            "label": p.display_name,
            "event": p.event_label,
            "event_date": str(p.event_date),
            "data_days": int(len(hrv)),
            "days_since_event_range": dse_range,
            "hrv": {
                "mean": float(hrv.mean()) if not hrv.empty else 0.0,
                "median": float(hrv.median()) if not hrv.empty else 0.0,
                "std": float(hrv.std()) if not hrv.empty else 0.0,
                "min": float(hrv.min()) if not hrv.empty else 0.0,
                "max": float(hrv.max()) if not hrv.empty else 0.0,
                "trend_slope_per_week": hrv_trend["slope_per_week"],
                "trend_p_value": hrv_trend["p_value"],
                "trend_direction": hrv_trend["direction"],
                "pct_below_esc_threshold": pct_below_esc,
                "population_percentile": metrics.get("hrv_pop_percentile", 0.0),
            },
            "heart_rate": {
                "mean_sleep_hr": float(hr.mean()) if not hr.empty else 0.0,
                "mean_lowest_hr": float(hr_low.mean()) if not hr_low.empty else 0.0,
                "trend_slope_per_week": hr_trend["slope_per_week"],
                "trend_direction": hr_trend["direction"],
            },
        }

    # Cross-patient comparison
    pids = list(data.keys())
    if len(pids) >= 2:
        h_hrv = data[pids[0]]["hrv"]
        m_hrv = data[pids[1]]["hrv"]
        h_hr = data[pids[0]]["hr"]
        m_hr = data[pids[1]]["hr"]

        h_mean = float(h_hrv.mean()) if not h_hrv.empty else 1.0
        m_mean = float(m_hrv.mean()) if not m_hrv.empty else 1.0
        hrv_ratio = m_mean / h_mean if h_mean > 0 else 0.0

        h_hr_mean = float(h_hr.mean()) if not h_hr.empty else 0.0
        m_hr_mean = float(m_hr.mean()) if not m_hr.empty else 0.0

        # Convergence: compare first vs last 14 days of overlap period
        h_first14 = float(h_hrv.head(14).mean()) if len(h_hrv) >= 14 else float(h_hrv.mean()) if not h_hrv.empty else 0.0
        h_last14 = float(h_hrv.tail(14).mean()) if len(h_hrv) >= 14 else float(h_hrv.mean()) if not h_hrv.empty else 0.0
        m_first14 = float(m_hrv.head(14).mean()) if len(m_hrv) >= 14 else float(m_hrv.mean()) if not m_hrv.empty else 0.0
        m_last14 = float(m_hrv.tail(14).mean()) if len(m_hrv) >= 14 else float(m_hrv.mean()) if not m_hrv.empty else 0.0

        gap_early = abs(m_first14 - h_first14)
        gap_late = abs(m_last14 - h_last14)
        converging = gap_late < gap_early

        stats_result["comparison"] = {
            "hrv_ratio": round(hrv_ratio, 2),
            "hrv_gap_ms": round(m_mean - h_mean, 1),
            "hr_gap_bpm": round(h_hr_mean - m_hr_mean, 1),
            "trajectories_converging": converging,
            "severity_classification": {
                pids[0]: "severe_autonomic_dysfunction" if h_mean < ESC_RMSSD_DEFICIENCY else "moderate_autonomic_impairment",
                pids[1]: "mild_autonomic_impairment" if m_mean > 30 else "moderate_autonomic_impairment",
            },
        }

    return stats_result


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _fig_hrv_trajectory(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Fig 1: HRV Trajectory -- dual panel: raw (log y) + z-score."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=["HRV (RMSSD) -- Raw Values", "HRV (RMSSD) -- Z-Score Normalized"],
    )
    patient_map = {p.patient_id: p for p in patients}

    for pid, metrics in data.items():
        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        hrv = metrics["hrv"]
        hrv_7d = metrics.get("hrv_7d", pd.Series(dtype=float))
        hrv_z = metrics.get("hrv_zscore", pd.Series(dtype=float))

        if hrv.empty:
            continue

        # Top panel: raw HRV
        fig.add_trace(go.Scatter(
            x=hrv.index, y=hrv.values,
            mode="markers", marker=dict(size=3, color=color, opacity=0.3),
            name=f"{p.display_name} (daily)", legendgroup=pid, showlegend=False,
        ), row=1, col=1)

        if not hrv_7d.empty:
            fig.add_trace(go.Scatter(
                x=hrv_7d.index, y=hrv_7d.values,
                mode="lines", line=dict(color=color, width=2.5),
                name=f"{p.display_name} (7d avg)", legendgroup=pid,
            ), row=1, col=1)

        # Bottom panel: z-score
        if not hrv_z.empty:
            fig.add_trace(go.Scatter(
                x=hrv_z.index, y=hrv_z.values,
                mode="lines", line=dict(color=color, width=2),
                name=f"{p.display_name} (z-score)", legendgroup=pid, showlegend=False,
            ), row=2, col=1)

    # Reference lines on top panel
    _add_reference_line(fig, ESC_RMSSD_DEFICIENCY, f"ESC Threshold ({ESC_RMSSD_DEFICIENCY}ms)", ACCENT_RED, row=1, col=1)
    _add_reference_line(fig, POPULATION_RMSSD_MEDIAN, f"Population Median ({POPULATION_RMSSD_MEDIAN}ms)", ACCENT_AMBER, row=1, col=1)

    # Z-score zero reference
    _add_reference_line(fig, 0, "Patient Mean", TEXT_SECONDARY, dash="dot", row=2, col=1)

    # Ruxolitinib event marker on both panels
    rux_ts = pd.Timestamp(TREATMENT_START)
    for r in [1, 2]:
        _add_event_vline(fig, rux_ts, "Rux Start", ACCENT_CYAN, row=r, col=1)

    fig.update_yaxes(type="log", title_text="RMSSD (ms, log scale)", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score (patient-relative)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_layout(
        height=700,
        title=dict(text="HRV Recovery Trajectories", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=60, r=20, t=60, b=50),
        hovermode="x unified",
    )
    return fig


def _fig_hr_trajectory(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Fig 2: Heart Rate Trajectory -- dual panel: raw + z-score."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=["Sleep Heart Rate -- Raw Values", "Sleep Heart Rate -- Z-Score Normalized"],
    )
    patient_map = {p.patient_id: p for p in patients}

    for pid, metrics in data.items():
        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        hr = metrics["hr"]
        hr_7d = metrics.get("hr_7d", pd.Series(dtype=float))

        if hr.empty:
            continue

        # Top panel: raw HR
        fig.add_trace(go.Scatter(
            x=hr.index, y=hr.values,
            mode="markers", marker=dict(size=3, color=color, opacity=0.3),
            name=f"{p.display_name} (daily)", legendgroup=pid, showlegend=False,
        ), row=1, col=1)

        if not hr_7d.empty:
            fig.add_trace(go.Scatter(
                x=hr_7d.index, y=hr_7d.values,
                mode="lines", line=dict(color=color, width=2.5),
                name=f"{p.display_name} (7d avg)", legendgroup=pid,
            ), row=1, col=1)

        # Bottom panel: z-score
        if not hr.empty and len(hr) >= 3:
            nr = zscore_normalize(hr, patient_id=pid)
            fig.add_trace(go.Scatter(
                x=nr.z_scores.index, y=nr.z_scores.values,
                mode="lines", line=dict(color=color, width=2),
                name=f"{p.display_name} (z-score)", legendgroup=pid, showlegend=False,
            ), row=2, col=1)

    _add_reference_line(fig, 0, "Patient Mean", TEXT_SECONDARY, dash="dot", row=2, col=1)

    rux_ts = pd.Timestamp(TREATMENT_START)
    for r in [1, 2]:
        _add_event_vline(fig, rux_ts, "Rux Start", ACCENT_CYAN, row=r, col=1)

    fig.update_yaxes(title_text="Heart Rate (bpm)", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score (patient-relative)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_layout(
        height=700,
        title=dict(text="Heart Rate Comparison", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=60, r=20, t=60, b=50),
        hovermode="x unified",
    )
    return fig


def _fig_pct_baseline(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Fig 3: Percent-of-Baseline HRV."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    for pid, metrics in data.items():
        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        pct = metrics.get("hrv_pct_baseline", pd.Series(dtype=float))

        if pct.empty:
            continue

        # X-axis = ordinal day into observation (0-based)
        x_days = np.arange(len(pct))

        fig.add_trace(go.Scatter(
            x=x_days, y=pct.values,
            mode="markers", marker=dict(size=3, color=color, opacity=0.3),
            name=f"{p.display_name} (daily)", legendgroup=pid, showlegend=False,
        ))

        if len(pct) >= 7:
            rolling = pct.rolling(7, min_periods=4).mean()
            fig.add_trace(go.Scatter(
                x=x_days, y=rolling.values,
                mode="lines", line=dict(color=color, width=2.5),
                name=f"{p.display_name} (7d avg)", legendgroup=pid,
            ))

    _add_reference_line(fig, 100, "Baseline (100%)", TEXT_SECONDARY, dash="dot")

    fig.update_layout(
        height=450,
        title=dict(text="HRV Relative to First 14 Days", font=dict(size=16)),
        xaxis_title="Days Into Observation",
        yaxis_title="% of Baseline HRV",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
    )
    return fig


def _fig_hrv_distribution(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Fig 4: HRV Distribution -- overlapping violins with population band."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    for pid, metrics in data.items():
        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        hrv = metrics["hrv"].dropna()

        if hrv.empty:
            continue

        fig.add_trace(go.Violin(
            y=hrv.values,
            name=p.display_name,
            marker_color=color,
            box_visible=True,
            meanline_visible=True,
            opacity=0.8,
            side="positive" if pid == patients[0].patient_id else "negative",
            scalegroup=pid,
        ))

    # Population reference band
    pop_low = POPULATION_RMSSD_MEAN - POPULATION_RMSSD_SD
    pop_high = POPULATION_RMSSD_MEAN + POPULATION_RMSSD_SD
    fig.add_hrect(
        y0=pop_low, y1=pop_high,
        fillcolor=ACCENT_AMBER, opacity=0.08,
        line_width=0,
        annotation_text="Population +/-1 SD",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color=ACCENT_AMBER,
    )
    _add_reference_line(fig, POPULATION_RMSSD_MEDIAN, f"Pop. Median ({POPULATION_RMSSD_MEDIAN}ms)", ACCENT_AMBER)
    _add_reference_line(fig, ESC_RMSSD_DEFICIENCY, f"ESC Threshold ({ESC_RMSSD_DEFICIENCY}ms)", ACCENT_RED)

    fig.update_layout(
        height=500,
        title=dict(text="HRV Distribution Comparison", font=dict(size=16)),
        yaxis_title="RMSSD (ms)",
        showlegend=True,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def _fig_long_term_context(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Fig 5: Mitchell's full HRV timeline with Henrik's window overlaid."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    # Mitchell = second patient (longer dataset)
    mitch_pid = patients[1].patient_id
    henrik_pid = patients[0].patient_id

    m_hrv = data[mitch_pid]["hrv"]
    h_hrv = data[henrik_pid]["hrv"]

    m_p = patient_map[mitch_pid]
    h_p = patient_map[henrik_pid]

    if not m_hrv.empty:
        # Mitchell: full range scatter
        fig.add_trace(go.Scatter(
            x=m_hrv.index, y=m_hrv.values,
            mode="markers", marker=dict(size=2, color=ACCENT_GREEN, opacity=0.2),
            name=f"{m_p.display_name} (daily)", legendgroup="mitch", showlegend=False,
        ))
        # 30-day rolling
        if len(m_hrv) >= 30:
            rolling = m_hrv.rolling(30, min_periods=15).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values,
                mode="lines", line=dict(color=ACCENT_GREEN, width=2.5),
                name=f"{m_p.display_name} (30d avg)", legendgroup="mitch",
            ))

    if not h_hrv.empty:
        # Henrik: overlay his window
        fig.add_trace(go.Scatter(
            x=h_hrv.index, y=h_hrv.values,
            mode="markers", marker=dict(size=4, color=ACCENT_BLUE, opacity=0.6),
            name=f"{h_p.display_name} (daily)", legendgroup="henrik", showlegend=False,
        ))
        if len(h_hrv) >= 7:
            rolling = h_hrv.rolling(7, min_periods=4).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values,
                mode="lines", line=dict(color=ACCENT_BLUE, width=3),
                name=f"{h_p.display_name} (7d avg)", legendgroup="henrik",
            ))

        # Shade Henrik's data window
        h_start = h_hrv.index.min()
        h_end = h_hrv.index.max()
        fig.add_vrect(
            x0=h_start, x1=h_end,
            fillcolor=ACCENT_BLUE, opacity=0.06,
            line_width=0,
            annotation_text="Henrik's window",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=ACCENT_BLUE,
        )

    _add_reference_line(fig, ESC_RMSSD_DEFICIENCY, f"ESC ({ESC_RMSSD_DEFICIENCY}ms)", ACCENT_RED)
    _add_reference_line(fig, POPULATION_RMSSD_MEDIAN, f"Pop. Median ({POPULATION_RMSSD_MEDIAN}ms)", ACCENT_AMBER)

    fig.update_layout(
        height=450,
        title=dict(text="Long-Term Context: Mitchell's 5-Year HRV with Henrik's Window", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="RMSSD (ms)",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
    )
    return fig


def _fig_autonomic_coupling(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Fig 6: Autonomic Coupling -- HR vs HRV scatter."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    for pid, metrics in data.items():
        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        hrv = metrics["hrv"]
        hr = metrics["hr"]

        if hrv.empty or hr.empty:
            continue

        # Align by shared dates
        shared_idx = hrv.index.intersection(hr.index)
        if len(shared_idx) < 3:
            continue

        hrv_aligned = hrv.loc[shared_idx]
        hr_aligned = hr.loc[shared_idx]

        fig.add_trace(go.Scatter(
            x=hr_aligned.values, y=hrv_aligned.values,
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.6),
            name=p.display_name,
            legendgroup=pid,
            hovertemplate="%{x:.0f} bpm / %{y:.1f} ms<extra>" + p.display_name + "</extra>",
        ))

        # Trend line (linear regression)
        if len(shared_idx) >= 5:
            slope, intercept, r, p_val, se = scipy_stats.linregress(
                hr_aligned.values.astype(float), hrv_aligned.values.astype(float)
            )
            x_range = np.linspace(hr_aligned.min(), hr_aligned.max(), 50)
            fig.add_trace(go.Scatter(
                x=x_range, y=slope * x_range + intercept,
                mode="lines", line=dict(color=color, width=2, dash="dash"),
                name=f"{p.display_name} (r={r:.2f})",
                legendgroup=pid,
            ))

    fig.update_layout(
        height=500,
        title=dict(text="Autonomic Coupling: Sleep HR vs HRV", font=dict(size=16)),
        xaxis_title="Sleep Heart Rate (bpm)",
        yaxis_title="HRV RMSSD (ms)",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# [6/7] HTML Report Assembly
# ---------------------------------------------------------------------------

def build_html(
    data: dict[str, dict[str, Any]],
    stats_result: dict[str, Any],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Build the full HTML report."""
    sections: list[str] = []

    # -- KPI Row --
    h_stats = stats_result["patients"].get(patients[0].patient_id, {})
    m_stats = stats_result["patients"].get(patients[1].patient_id, {})
    comp = stats_result.get("comparison", {})

    h_hrv_mean = h_stats.get("hrv", {}).get("mean", 0)
    m_hrv_mean = m_stats.get("hrv", {}).get("mean", 0)
    h_hr_mean = h_stats.get("heart_rate", {}).get("mean_sleep_hr", 0)
    m_hr_mean = m_stats.get("heart_rate", {}).get("mean_sleep_hr", 0)
    hrv_ratio = comp.get("hrv_ratio", 0)
    h_trend_dir = h_stats.get("hrv", {}).get("trend_direction", "N/A")

    kpi_row = make_kpi_row(
        make_kpi_card(
            "HENRIK MEAN HRV", h_hrv_mean, "ms",
            status="critical",
            detail=f"Below ESC threshold ({ESC_RMSSD_DEFICIENCY}ms)" if h_hrv_mean < ESC_RMSSD_DEFICIENCY else "Above ESC threshold",
            status_label="Severe" if h_hrv_mean < ESC_RMSSD_DEFICIENCY else "Low",
        ),
        make_kpi_card(
            "MITCHELL MEAN HRV", m_hrv_mean, "ms",
            status="info",
            detail=f"Pop. percentile: {m_stats.get('hrv', {}).get('population_percentile', 0):.0f}%",
        ),
        make_kpi_card(
            "HENRIK SLEEP HR", h_hr_mean, "bpm",
            status="warning" if h_hr_mean > 75 else "normal",
            detail="Elevated" if h_hr_mean > 75 else "Acceptable range",
        ),
        make_kpi_card(
            "MITCHELL SLEEP HR", m_hr_mean, "bpm",
            status="normal",
            detail="Athletic range" if m_hr_mean < 55 else "Normal range",
        ),
        make_kpi_card(
            "HRV GAP RATIO", hrv_ratio, "x",
            status="info",
            detail=f"Mitchell/Henrik ({comp.get('hrv_gap_ms', 0):.0f}ms gap)",
        ),
        make_kpi_card(
            "TRAJECTORY", h_trend_dir.title(), "",
            status="normal" if h_trend_dir == "improving" else ("warning" if h_trend_dir == "stable" else "critical"),
            detail=f"Henrik's 30-day HRV trend",
            status_label=h_trend_dir.title(),
        ),
    )
    sections.append(kpi_row)

    # -- Section 1: HRV Trajectory --
    sections.append(section_html_or_placeholder(
        "HRV Recovery Trajectories",
        lambda: make_section(
            "Autonomic Recovery Trajectories",
            _embed(_fig_hrv_trajectory(data, patients)),
            section_id="hrv-trajectory",
        ),
    ))

    # -- Section 2: HR Trajectory --
    sections.append(section_html_or_placeholder(
        "Heart Rate Comparison",
        lambda: make_section(
            "Heart Rate Comparison",
            _embed(_fig_hr_trajectory(data, patients)),
            section_id="hr-trajectory",
        ),
    ))

    # -- Section 3: Percent-of-Baseline --
    sections.append(section_html_or_placeholder(
        "Relative Recovery",
        lambda: make_section(
            "Relative Recovery (% of Baseline)",
            _embed(_fig_pct_baseline(data, patients)),
            section_id="pct-baseline",
        ),
    ))

    # -- Section 4: HRV Distribution --
    sections.append(section_html_or_placeholder(
        "HRV Distribution",
        lambda: make_section(
            "HRV Distribution Comparison",
            _embed(_fig_hrv_distribution(data, patients)),
            section_id="hrv-distribution",
        ),
    ))

    # -- Section 5: Long-Term Context --
    sections.append(section_html_or_placeholder(
        "Long-Term Context",
        lambda: make_section(
            "Long-Term Context",
            _embed(_fig_long_term_context(data, patients)),
            section_id="long-term-context",
        ),
    ))

    # -- Section 6: Autonomic Coupling --
    sections.append(section_html_or_placeholder(
        "Autonomic Coupling",
        lambda: make_section(
            "Autonomic Coupling (HR vs HRV)",
            _embed(_fig_autonomic_coupling(data, patients)),
            section_id="autonomic-coupling",
        ),
    ))

    # -- Section 7: Clinical Context --
    clinical_note = (
        '<p style="color:#9CA3AF;line-height:1.7;">'
        "This report compares two fundamentally different clinical trajectories. "
        "<strong>Henrik</strong> is 2+ years post-allogeneic HSCT with chronic GVHD and "
        "severe autonomic dysfunction (HRV consistently below the ESC 15ms threshold). "
        "<strong>Mitchell</strong> is ~15 months post-stroke (bilateral carotid/vertebral "
        "artery dissection) with mildly impaired but recovering autonomic function. "
        "Direct HRV magnitude comparison is less meaningful than trajectory shape and "
        "relative changes within each patient's own range.</p>"
        '<p style="color:#6B7280;line-height:1.7;margin-top:12px;">'
        "Normalization approaches (z-score, percent-of-baseline) allow meaningful "
        "cross-patient comparison despite the 5x difference in absolute HRV values. "
        "All heart rate values are derived from sleep periods (not readiness scores). "
        "Oura readiness 'resting_heart_rate' is a 0-100 score and is never used as bpm.</p>"
    )
    sections.append(make_section(
        "Clinical Context",
        clinical_note,
        section_id="clinical-context",
    ))

    body = "\n".join(sections)
    return wrap_html(
        title="Autonomic Recovery Trajectories",
        body_content=body,
        report_id="comp_autonomic",
        subtitle="Module 1: Comparative Autonomic Analysis",
        header_meta="Henrik (post-HSCT) vs Mitchell (post-Stroke)",
    )


# ---------------------------------------------------------------------------
# [7/7] JSON Export
# ---------------------------------------------------------------------------

def export_json(stats_result: dict[str, Any]) -> None:
    """Write structured metrics JSON."""
    output = {
        "report": "comparative_autonomic",
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        **stats_result,
    }

    # Sanitize NaN for JSON
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    output = _sanitize(output)
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("JSON metrics written to %s", JSON_OUTPUT)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run comparative autonomic analysis pipeline."""
    logger.info("[1/7] Loading patient data...")
    patients = default_patients()
    raw_data = load_data(patients)

    logger.info("[2/7] Normalizing timelines...")
    data = normalize_timelines(raw_data, patients)

    logger.info("[3/7] Computing rolling metrics...")
    data = compute_rolling(data)

    logger.info("[4/7] Computing normalized metrics...")
    data = compute_normalized(data)

    logger.info("[5/7] Computing trends and comparison...")
    stats_result = compute_trends_and_comparison(data, patients)

    logger.info("[6/7] Generating HTML report...")
    html = build_html(data, stats_result, patients)
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", HTML_OUTPUT)

    logger.info("[7/7] Exporting JSON metrics...")
    export_json(stats_result)

    logger.info("Comparative autonomic analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
