#!/usr/bin/env python3
"""Module 6: Breathing Rate Analysis & BOS Screening.

Compares breathing rate patterns between Henrik (post-HSCT) and Mitchell
(post-stroke). Elevated breathing rate is an early marker for BOS
(Bronchiolitis Obliterans Syndrome) post-HSCT.

Outputs:
  - Interactive HTML dashboard: reports/comparative_breathing_analysis.html
  - JSON metrics:               reports/comparative_breathing_metrics.json

Usage:
    python analysis/analyze_comparative_breathing.py
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
    TREATMENT_START,
    FONT_FAMILY,
)
from _comparative_utils import (
    PatientConfig,
    default_patients,
    load_patient_data,
    zscore_normalize,
    compare_distributions,
    dual_patient_timeseries,
    dual_patient_distribution,
    PATIENT_COLORS,
)
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    disclaimer_banner,
    format_p_value,
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

HTML_OUTPUT = REPORTS_DIR / "comparative_breathing_analysis.html"
JSON_OUTPUT = REPORTS_DIR / "comparative_breathing_metrics.json"

# Clinical thresholds
BREATH_NORMAL_LOW = 12.0   # brpm, lower bound of normal sleep range
BREATH_NORMAL_HIGH = 20.0  # brpm, upper bound of normal sleep range
BREATH_ELEVATED = 18.0     # brpm, BOS screening concern threshold
BOS_TREND_CONCERN = 0.02   # brpm/day, concerning upward trend slope
ANOMALY_SD_THRESHOLD = 2.0  # standard deviations above mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(fig: go.Figure) -> str:
    """Embed a Plotly figure as inline HTML (no JS bundle)."""
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _compute_linear_trend(series: pd.Series) -> dict:
    """Compute linear regression slope (per day) + p-value."""
    clean = series.dropna()
    if len(clean) < 5:
        return {"slope_per_day": np.nan, "p_value": np.nan, "direction": "insufficient data"}
    x = np.arange(len(clean), dtype=float)
    y = clean.values.astype(float)
    slope, intercept, r, p, se = scipy_stats.linregress(x, y)
    if p > 0.10 or np.isnan(p):
        direction = "stable"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"
    return {
        "slope_per_day": float(slope),
        "p_value": float(p),
        "r_value": float(r),
        "intercept": float(intercept),
        "direction": direction,
        "n_points": len(clean),
    }


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


def _bos_risk_status(mean_breath: float, trend_slope: float, pct_elevated: float) -> tuple[str, str]:
    """Classify BOS risk level.

    Returns (status, label) for KPI card.
    """
    if np.isnan(mean_breath):
        return "neutral", "No Data"
    if mean_breath > BREATH_ELEVATED and trend_slope > BOS_TREND_CONCERN:
        return "critical", "High Risk"
    if mean_breath > BREATH_ELEVATED or trend_slope > BOS_TREND_CONCERN:
        return "warning", "Monitor"
    if pct_elevated > 15:
        return "warning", "Watch"
    return "normal", "Low Risk"


# ---------------------------------------------------------------------------
# [1/7] Data Loading
# ---------------------------------------------------------------------------

def load_data(
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, dict[str, pd.Series]]:
    """Load breathing rate, HRV, and HR data for both patients.

    Returns dict keyed by patient_id, each containing:
      - "breath": pd.Series of average_breath (brpm)
      - "hrv":    pd.Series of average_hrv (ms)
      - "hr":     pd.Series of average_heart_rate (bpm)
      - "hr_lowest": pd.Series of lowest_heart_rate (bpm)
      - "duration": pd.Series of total_sleep_duration (sec)
      - "efficiency": pd.Series of efficiency (%)
    """
    result: dict[str, dict[str, pd.Series]] = {}

    for p in patients:
        sp = load_patient_data(
            p, "oura_sleep_periods",
            columns="day, average_breath, average_hrv, average_heart_rate, lowest_heart_rate, total_sleep_duration, efficiency",
        )

        def _col(name: str) -> pd.Series:
            if not sp.empty and name in sp.columns:
                s = sp[name].dropna()
                s.name = name
                return s
            return pd.Series(dtype=float, name=name)

        result[p.patient_id] = {
            "breath": _col("average_breath"),
            "hrv": _col("average_hrv"),
            "hr": _col("average_heart_rate"),
            "hr_lowest": _col("lowest_heart_rate"),
            "duration": _col("total_sleep_duration"),
            "efficiency": _col("efficiency"),
        }
        logger.info(
            "Loaded %s: %d breathing days, %d HRV days",
            p.display_name, len(result[p.patient_id]["breath"]),
            len(result[p.patient_id]["hrv"]),
        )

    return result


# ---------------------------------------------------------------------------
# [2/7] Rolling Metrics
# ---------------------------------------------------------------------------

def compute_rolling(
    data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute 7-day and 30-day rolling averages for breathing rate."""
    for pid, metrics in data.items():
        breath = metrics["breath"]
        if not breath.empty:
            metrics["breath_7d"] = breath.rolling(7, min_periods=4).mean()
            metrics["breath_30d"] = breath.rolling(30, min_periods=15).mean()
        else:
            metrics["breath_7d"] = pd.Series(dtype=float)
            metrics["breath_30d"] = pd.Series(dtype=float)
    return data


# ---------------------------------------------------------------------------
# [3/7] BOS Screening (Henrik)
# ---------------------------------------------------------------------------

def compute_bos_screening(
    data: dict[str, dict[str, Any]],
    henrik_id: str,
) -> dict[str, Any]:
    """Compute BOS risk indicators for Henrik.

    Returns dict with trend analysis, elevated count, risk classification.
    """
    breath = data[henrik_id]["breath"]
    if breath.empty or len(breath) < 5:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "pct_elevated": 0.0,
            "elevated_nights": 0,
            "total_nights": 0,
            "trend": {"slope_per_day": np.nan, "p_value": np.nan, "direction": "insufficient data"},
            "risk_status": "neutral",
            "risk_label": "No Data",
        }

    mean_val = float(breath.mean())
    std_val = float(breath.std())
    pct_elevated = float((breath > BREATH_ELEVATED).sum() / len(breath) * 100)
    trend = _compute_linear_trend(breath)
    risk_status, risk_label = _bos_risk_status(
        mean_val, trend.get("slope_per_day", 0.0), pct_elevated,
    )

    return {
        "mean": mean_val,
        "median": float(breath.median()),
        "std": std_val,
        "min": float(breath.min()),
        "max": float(breath.max()),
        "pct_elevated": pct_elevated,
        "elevated_nights": int((breath > BREATH_ELEVATED).sum()),
        "total_nights": len(breath),
        "trend": trend,
        "risk_status": risk_status,
        "risk_label": risk_label,
    }


# ---------------------------------------------------------------------------
# [4/7] Pre/Post Ruxolitinib Analysis (Henrik)
# ---------------------------------------------------------------------------

def compute_rux_effect(
    data: dict[str, dict[str, Any]],
    henrik_id: str,
) -> dict[str, Any]:
    """Mann-Whitney U for breathing rate pre vs post Ruxolitinib."""
    breath = data[henrik_id]["breath"]
    rux_ts = pd.Timestamp(TREATMENT_START)

    if breath.empty:
        return {
            "pre_mean": np.nan, "post_mean": np.nan,
            "pre_n": 0, "post_n": 0,
            "test": "insufficient_data",
        }

    pre = breath[breath.index < rux_ts]
    post = breath[breath.index >= rux_ts]

    if len(pre) < 3 or len(post) < 3:
        return {
            "pre_mean": float(pre.mean()) if not pre.empty else np.nan,
            "post_mean": float(post.mean()) if not post.empty else np.nan,
            "pre_median": float(pre.median()) if not pre.empty else np.nan,
            "post_median": float(post.median()) if not post.empty else np.nan,
            "pre_n": len(pre),
            "post_n": len(post),
            "test": "insufficient_data",
        }

    comparison = compare_distributions(pre, post)
    return {
        "pre_mean": float(pre.mean()),
        "post_mean": float(post.mean()),
        "pre_median": float(pre.median()),
        "post_median": float(post.median()),
        "pre_std": float(pre.std()),
        "post_std": float(post.std()),
        "pre_n": len(pre),
        "post_n": len(post),
        "delta_mean": float(post.mean() - pre.mean()),
        **comparison,
    }


# ---------------------------------------------------------------------------
# [5/7] Cross-Patient Comparison
# ---------------------------------------------------------------------------

def compute_comparison(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, Any]:
    """Distribution comparison, z-score analysis, coupling metrics."""
    h_id = patients[0].patient_id
    m_id = patients[1].patient_id
    h_breath = data[h_id]["breath"]
    m_breath = data[m_id]["breath"]

    # Distribution comparison
    dist_comp = compare_distributions(h_breath, m_breath)

    # Z-score normalization
    h_norm = zscore_normalize(h_breath, patient_id=h_id) if not h_breath.empty and len(h_breath) >= 3 else None
    m_norm = zscore_normalize(m_breath, patient_id=m_id) if not m_breath.empty and len(m_breath) >= 3 else None

    # Breathing-HRV coupling (Spearman) for each patient
    coupling: dict[str, dict[str, Any]] = {}
    for pid in [h_id, m_id]:
        breath = data[pid]["breath"]
        hrv = data[pid]["hrv"]
        if breath.empty or hrv.empty:
            coupling[pid] = {"spearman_r": np.nan, "p_value": np.nan, "n": 0}
            continue
        # Align by date index
        common = breath.index.intersection(hrv.index)
        if len(common) < 5:
            coupling[pid] = {"spearman_r": np.nan, "p_value": np.nan, "n": len(common)}
            continue
        b_aligned = breath.loc[common].astype(float)
        h_aligned = hrv.loc[common].astype(float)
        r, p = scipy_stats.spearmanr(b_aligned.values, h_aligned.values)
        coupling[pid] = {
            "spearman_r": float(r),
            "p_value": float(p),
            "n": len(common),
            "coupling_intact": abs(r) > 0.2 and p < 0.05,
        }

    return {
        "distribution": dist_comp,
        "henrik_zscore_mean": float(h_norm.z_scores.mean()) if h_norm else np.nan,
        "mitch_zscore_mean": float(m_norm.z_scores.mean()) if m_norm else np.nan,
        "coupling": coupling,
    }


# ---------------------------------------------------------------------------
# [6a/7] Anomaly Detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    data: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Flag nights with breath > 2 SD above personal mean."""
    anomalies: dict[str, list[dict[str, Any]]] = {}

    for pid, metrics in data.items():
        breath = metrics["breath"]
        if breath.empty or len(breath) < 5:
            anomalies[pid] = []
            continue

        mean_val = breath.mean()
        std_val = breath.std()
        if std_val == 0 or np.isnan(std_val):
            anomalies[pid] = []
            continue

        threshold = mean_val + ANOMALY_SD_THRESHOLD * std_val
        elevated = breath[breath > threshold]

        patient_anomalies: list[dict[str, Any]] = []
        for dt, val in elevated.items():
            record: dict[str, Any] = {
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "breath_rate": float(val),
                "z_score": float((val - mean_val) / std_val),
            }
            # Add co-occurring metrics if available
            if dt in metrics["hrv"].index:
                record["hrv"] = float(metrics["hrv"].loc[dt])
            if dt in metrics["hr"].index:
                record["hr"] = float(metrics["hr"].loc[dt])
            patient_anomalies.append(record)

        anomalies[pid] = patient_anomalies

    return anomalies


# ---------------------------------------------------------------------------
# [6b/7] Visualizations
# ---------------------------------------------------------------------------

def _fig_breathing_timeline(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Fig 1: Dual timeline -- both patients' breathing rate with 7d rolling mean."""
    breath_data = {p.patient_id: data[p.patient_id]["breath"] for p in patients}
    fig = dual_patient_timeseries(
        breath_data,
        patients,
        title="Breathing Rate Over Time",
        y_label="Breathing Rate (brpm)",
        show_rolling=7,
        event_lines=True,
    )

    # Add normal range reference lines
    fig.add_shape(
        type="rect",
        x0=0, x1=1, xref="paper",
        y0=BREATH_NORMAL_LOW, y1=BREATH_NORMAL_HIGH,
        fillcolor=ACCENT_GREEN,
        opacity=0.05,
        line_width=0,
        layer="below",
    )
    _add_reference_line(fig, BREATH_NORMAL_LOW, f"Normal Low ({BREATH_NORMAL_LOW} brpm)", ACCENT_GREEN, dash="dot")
    _add_reference_line(fig, BREATH_NORMAL_HIGH, f"Normal High ({BREATH_NORMAL_HIGH} brpm)", ACCENT_GREEN, dash="dot")

    # Ruxolitinib marker
    _add_event_vline(fig, pd.Timestamp(TREATMENT_START), "Rux Start", ACCENT_CYAN)

    fig.update_layout(height=500)
    return fig


def _fig_distribution(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Fig 2: Distribution comparison -- violin plots for both patients."""
    breath_data = {p.patient_id: data[p.patient_id]["breath"] for p in patients}
    fig = dual_patient_distribution(
        breath_data,
        patients,
        title="Breathing Rate Distribution Comparison",
        kind="violin",
    )
    _add_reference_line(fig, BREATH_NORMAL_LOW, f"Normal Low ({BREATH_NORMAL_LOW})", ACCENT_GREEN, dash="dot")
    _add_reference_line(fig, BREATH_NORMAL_HIGH, f"Normal High ({BREATH_NORMAL_HIGH})", ACCENT_GREEN, dash="dot")
    _add_reference_line(fig, BREATH_ELEVATED, f"Elevated ({BREATH_ELEVATED})", ACCENT_AMBER, dash="dash")

    fig.update_layout(
        height=450,
        yaxis_title="Breathing Rate (brpm)",
    )
    return fig


def _fig_rux_effect(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
    rux_stats: dict[str, Any],
) -> go.Figure:
    """Fig 3: Pre/post Rux -- Henrik only, box/violin with stats annotation."""
    h_id = patients[0].patient_id
    breath = data[h_id]["breath"]
    rux_ts = pd.Timestamp(TREATMENT_START)

    pre = breath[breath.index < rux_ts].dropna()
    post = breath[breath.index >= rux_ts].dropna()

    fig = go.Figure()

    if not pre.empty:
        fig.add_trace(go.Violin(
            y=pre.values,
            name=f"Pre-Rux (n={len(pre)})",
            marker_color=ACCENT_AMBER,
            box_visible=True,
            meanline_visible=True,
            opacity=0.8,
            side="negative",
            width=1.5,
        ))
    if not post.empty:
        fig.add_trace(go.Violin(
            y=post.values,
            name=f"Post-Rux (n={len(post)})",
            marker_color=ACCENT_CYAN,
            box_visible=True,
            meanline_visible=True,
            opacity=0.8,
            side="positive",
            width=1.5,
        ))

    # Stats annotation
    p_val = rux_stats.get("p_value", np.nan)
    d_val = rux_stats.get("effect_size", 0.0)
    delta = rux_stats.get("delta_mean", 0.0)
    sig = "Yes" if rux_stats.get("significant", False) else "No"

    annotation_text = (
        f"Mann-Whitney U: {format_p_value(p_val)}<br>"
        f"Cohen's d: {d_val:.2f} ({rux_stats.get('effect_label', 'N/A')})<br>"
        f"Mean delta: {delta:+.2f} brpm<br>"
        f"Significant: {sig}"
    )
    fig.add_annotation(
        x=0.5, y=1.15, xref="paper", yref="paper",
        text=annotation_text,
        showarrow=False,
        font=dict(size=11, color=TEXT_SECONDARY),
        align="center",
        bgcolor="rgba(26,29,39,0.8)",
        bordercolor=TEXT_SECONDARY,
        borderwidth=1,
        borderpad=8,
    )

    _add_reference_line(fig, BREATH_ELEVATED, f"Elevated ({BREATH_ELEVATED})", ACCENT_AMBER, dash="dash")

    fig.update_layout(
        height=500,
        title=dict(text="Ruxolitinib Effect on Breathing Rate (Henrik)", font=dict(size=16)),
        yaxis_title="Breathing Rate (brpm)",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=60, r=20, t=90, b=60),
    )
    return fig


def _fig_breath_hrv_scatter(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
    coupling_stats: dict[str, dict[str, Any]],
) -> go.Figure:
    """Fig 4: Breathing-HRV scatter -- two subplots, one per patient."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[p.display_name for p in patients],
        horizontal_spacing=0.12,
    )

    for idx, p in enumerate(patients, 1):
        pid = p.patient_id
        breath = data[pid]["breath"]
        hrv = data[pid]["hrv"]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        coup = coupling_stats.get(pid, {})

        if breath.empty or hrv.empty:
            continue

        common = breath.index.intersection(hrv.index)
        if len(common) < 3:
            continue

        b_vals = breath.loc[common].values.astype(float)
        h_vals = hrv.loc[common].values.astype(float)

        fig.add_trace(go.Scatter(
            x=b_vals, y=h_vals,
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.5),
            name=p.display_name,
            showlegend=False,
        ), row=1, col=idx)

        # Regression line
        if len(common) >= 5:
            slope, intercept, r, p_val, se = scipy_stats.linregress(b_vals, h_vals)
            x_range = np.linspace(b_vals.min(), b_vals.max(), 50)
            fig.add_trace(go.Scatter(
                x=x_range, y=slope * x_range + intercept,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"r={coup.get('spearman_r', r):.2f}",
                showlegend=True,
            ), row=1, col=idx)

            # Annotation with Spearman stats
            sp_r = coup.get("spearman_r", np.nan)
            sp_p = coup.get("p_value", np.nan)
            fig.add_annotation(
                x=0.5, y=0.95,
                xref=f"x{idx} domain" if idx > 1 else "x domain",
                yref=f"y{idx} domain" if idx > 1 else "y domain",
                text=f"Spearman r={sp_r:.2f}, {format_p_value(sp_p)}",
                showarrow=False,
                font=dict(size=10, color=color),
                bgcolor="rgba(26,29,39,0.7)",
                borderpad=4,
            )

    fig.update_xaxes(title_text="Breathing Rate (brpm)", row=1, col=1)
    fig.update_xaxes(title_text="Breathing Rate (brpm)", row=1, col=2)
    fig.update_yaxes(title_text="HRV RMSSD (ms)", row=1, col=1)
    fig.update_yaxes(title_text="HRV RMSSD (ms)", row=1, col=2)

    fig.update_layout(
        height=450,
        title=dict(text="Respiratory-Autonomic Coupling (Breath Rate vs HRV)", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def _fig_bos_panel(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
    bos_stats: dict[str, Any],
) -> go.Figure:
    """Fig 5: BOS risk panel -- Henrik's trend with thresholds and trend line."""
    h_id = patients[0].patient_id
    breath = data[h_id]["breath"]
    breath_7d = data[h_id].get("breath_7d", pd.Series(dtype=float))
    breath_30d = data[h_id].get("breath_30d", pd.Series(dtype=float))

    fig = go.Figure()

    if not breath.empty:
        # Daily scatter
        fig.add_trace(go.Scatter(
            x=breath.index, y=breath.values,
            mode="markers",
            marker=dict(size=4, color=ACCENT_BLUE, opacity=0.3),
            name="Daily",
            showlegend=True,
        ))

    if not breath_7d.empty:
        fig.add_trace(go.Scatter(
            x=breath_7d.index, y=breath_7d.values,
            mode="lines",
            line=dict(color=ACCENT_BLUE, width=2.5),
            name="7-day avg",
        ))

    if not breath_30d.empty:
        fig.add_trace(go.Scatter(
            x=breath_30d.index, y=breath_30d.values,
            mode="lines",
            line=dict(color=ACCENT_PURPLE, width=2, dash="dot"),
            name="30-day avg",
        ))

    # Linear trend line
    trend = bos_stats.get("trend", {})
    slope = trend.get("slope_per_day", np.nan)
    intercept = trend.get("intercept", np.nan)
    if not np.isnan(slope) and not np.isnan(intercept) and not breath.empty:
        x_num = np.arange(len(breath), dtype=float)
        y_trend = slope * x_num + intercept
        fig.add_trace(go.Scatter(
            x=breath.dropna().index, y=y_trend[:len(breath.dropna())],
            mode="lines",
            line=dict(color=ACCENT_AMBER, width=2, dash="dash"),
            name=f"Trend ({slope:+.4f}/day)",
        ))

    # Threshold lines
    _add_reference_line(fig, BREATH_ELEVATED, f"BOS Concern ({BREATH_ELEVATED} brpm)", ACCENT_RED, dash="dash")
    _add_reference_line(fig, BREATH_NORMAL_HIGH, f"Normal Upper ({BREATH_NORMAL_HIGH} brpm)", ACCENT_GREEN, dash="dot")
    _add_reference_line(fig, BREATH_NORMAL_LOW, f"Normal Lower ({BREATH_NORMAL_LOW} brpm)", ACCENT_GREEN, dash="dot")

    # Ruxolitinib marker
    _add_event_vline(fig, pd.Timestamp(TREATMENT_START), "Rux Start", ACCENT_CYAN)

    # Slope annotation
    p_text = format_p_value(trend.get("p_value"))
    risk_label = bos_stats.get("risk_label", "N/A")
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=(
            f"Linear trend: {slope:+.4f} brpm/day<br>"
            f"{p_text}, n={trend.get('n_points', 0)}<br>"
            f"Risk: {risk_label}"
        ),
        showarrow=False,
        font=dict(size=11, color=TEXT_SECONDARY),
        align="left",
        bgcolor="rgba(26,29,39,0.8)",
        bordercolor=TEXT_SECONDARY,
        borderwidth=1,
        borderpad=8,
        xanchor="left",
        yanchor="top",
    )

    fig.update_layout(
        height=500,
        title=dict(text="BOS Screening: Henrik Breathing Rate Trend", font=dict(size=16)),
        yaxis_title="Breathing Rate (brpm)",
        xaxis_title="Date",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=60, r=20, t=60, b=60),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# [6c/7] HTML Report Assembly
# ---------------------------------------------------------------------------

def build_html(
    data: dict[str, dict[str, Any]],
    bos_stats: dict[str, Any],
    rux_stats: dict[str, Any],
    comparison: dict[str, Any],
    anomalies: dict[str, list[dict[str, Any]]],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Build the full HTML report."""
    sections: list[str] = []

    h_id = patients[0].patient_id
    m_id = patients[1].patient_id
    h_breath = data[h_id]["breath"]
    m_breath = data[m_id]["breath"]

    h_mean = float(h_breath.mean()) if not h_breath.empty else 0.0
    m_mean = float(m_breath.mean()) if not m_breath.empty else 0.0
    h_trend_dir = bos_stats.get("trend", {}).get("direction", "N/A")
    risk_status = bos_stats.get("risk_status", "neutral")
    risk_label = bos_stats.get("risk_label", "N/A")

    # Determine KPI statuses
    h_status = "normal" if BREATH_NORMAL_LOW <= h_mean <= BREATH_NORMAL_HIGH else (
        "warning" if h_mean > BREATH_NORMAL_HIGH else "info"
    )
    m_status = "normal" if BREATH_NORMAL_LOW <= m_mean <= BREATH_NORMAL_HIGH else (
        "warning" if m_mean > BREATH_NORMAL_HIGH else "info"
    )
    trend_status = "normal" if h_trend_dir == "decreasing" else (
        "warning" if h_trend_dir == "increasing" else "info"
    )

    kpi_row = make_kpi_row(
        make_kpi_card(
            "HENRIK MEAN BREATH", h_mean, "brpm",
            status=h_status,
            detail=f"Range: {BREATH_NORMAL_LOW}-{BREATH_NORMAL_HIGH} brpm normal",
            explainer="Average nighttime breathing rate",
        ),
        make_kpi_card(
            "MITCHELL MEAN BREATH", m_mean, "brpm",
            status=m_status,
            detail="Reference (no respiratory comorbidity)",
            explainer="Average nighttime breathing rate",
        ),
        make_kpi_card(
            "HENRIK TREND", h_trend_dir.title(), "",
            status=trend_status,
            detail=f"Slope: {bos_stats.get('trend', {}).get('slope_per_day', 0):.4f}/day",
            status_label=h_trend_dir.title(),
        ),
        make_kpi_card(
            "BOS RISK", risk_label, "",
            status=risk_status,
            detail=f"{bos_stats.get('pct_elevated', 0):.1f}% nights above {BREATH_ELEVATED} brpm",
            explainer="Bronchiolitis Obliterans Syndrome screening",
            status_label=risk_label,
        ),
    )
    sections.append(kpi_row)

    # -- Section 1: Breathing Rate Trends --
    sections.append(section_html_or_placeholder(
        "Breathing Rate Trends",
        lambda: make_section(
            "Breathing Rate Trends",
            _embed(_fig_breathing_timeline(data, patients)),
            section_id="breathing-trends",
        ),
    ))

    # -- Section 2: Distribution Comparison --
    sections.append(section_html_or_placeholder(
        "Distribution Comparison",
        lambda: make_section(
            "Distribution Comparison",
            _build_distribution_section(data, patients, comparison),
            section_id="distribution",
        ),
    ))

    # -- Section 3: Ruxolitinib Effect --
    sections.append(section_html_or_placeholder(
        "Ruxolitinib Effect",
        lambda: make_section(
            "Ruxolitinib Effect on Breathing Rate",
            _embed(_fig_rux_effect(data, patients, rux_stats)),
            section_id="rux-effect",
        ),
    ))

    # -- Section 4: Respiratory-Autonomic Coupling --
    coupling_stats = comparison.get("coupling", {})
    sections.append(section_html_or_placeholder(
        "Respiratory-Autonomic Coupling",
        lambda: make_section(
            "Respiratory-Autonomic Coupling",
            _build_coupling_section(data, patients, coupling_stats),
            section_id="coupling",
        ),
    ))

    # -- Section 5: BOS Screening --
    sections.append(section_html_or_placeholder(
        "BOS Screening",
        lambda: make_section(
            "BOS Screening Panel",
            _embed(_fig_bos_panel(data, patients, bos_stats)),
            section_id="bos-screening",
        ),
    ))

    # -- Section 6: Clinical Interpretation --
    sections.append(make_section(
        "Clinical Interpretation",
        _build_clinical_interpretation(bos_stats, rux_stats, comparison, anomalies, patients),
        section_id="clinical-interpretation",
    ))

    # -- Section 7: Methodology --
    sections.append(make_section(
        "Methodology",
        _build_methodology(),
        section_id="methodology",
    ))

    body = "\n".join(sections)
    return wrap_html(
        title="Breathing Rate Analysis & BOS Screening",
        body_content=body,
        report_id="comp_breathing",
        subtitle="Module 6: Comparative Breathing Analysis",
        header_meta="Henrik (post-HSCT) vs Mitchell (post-Stroke)",
    )


def _build_distribution_section(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
    comparison: dict[str, Any],
) -> str:
    """Distribution chart plus comparison stats table."""
    chart_html = _embed(_fig_distribution(data, patients))

    dist = comparison.get("distribution", {})
    p_val = dist.get("p_value", np.nan)
    d_val = dist.get("effect_size", 0.0)

    stats_html = (
        f'<div style="margin-top:16px;padding:12px;background:rgba(26,29,39,0.5);border-radius:8px;">'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;">'
        f'<strong>Mann-Whitney U:</strong> {format_p_value(p_val)} &middot; '
        f'<strong>Cohen\'s d:</strong> {d_val:.2f} ({dist.get("effect_label", "N/A")}) &middot; '
        f'<strong>95% CI for mean difference:</strong> '
        f'({dist.get("ci_95", (np.nan, np.nan))[0]:.2f}, {dist.get("ci_95", (np.nan, np.nan))[1]:.2f}) brpm'
        f'</p></div>'
    )

    return chart_html + stats_html


def _build_coupling_section(
    data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
    coupling_stats: dict[str, dict[str, Any]],
) -> str:
    """Scatter plot plus coupling interpretation."""
    chart_html = _embed(_fig_breath_hrv_scatter(data, patients, coupling_stats))

    h_id = patients[0].patient_id
    m_id = patients[1].patient_id
    h_coup = coupling_stats.get(h_id, {})
    m_coup = coupling_stats.get(m_id, {})

    h_intact = h_coup.get("coupling_intact", False)
    m_intact = m_coup.get("coupling_intact", False)

    interp = (
        f'<div style="margin-top:16px;padding:12px;background:rgba(26,29,39,0.5);border-radius:8px;">'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;">'
        f'<strong>Respiratory sinus arrhythmia (RSA):</strong> In healthy individuals, '
        f'higher HRV correlates with lower breathing rate through vagal modulation. '
        f'Disrupted coupling may indicate autonomic dysfunction.</p>'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;margin-top:8px;">'
        f'<strong>Henrik:</strong> r={h_coup.get("spearman_r", 0):.2f}, '
        f'{format_p_value(h_coup.get("p_value"))}, n={h_coup.get("n", 0)} '
        f'-- {"Coupling intact" if h_intact else "Coupling disrupted/weak"}<br>'
        f'<strong>Mitchell:</strong> r={m_coup.get("spearman_r", 0):.2f}, '
        f'{format_p_value(m_coup.get("p_value"))}, n={m_coup.get("n", 0)} '
        f'-- {"Coupling intact" if m_intact else "Coupling disrupted/weak"}'
        f'</p></div>'
    )

    return chart_html + interp


def _build_clinical_interpretation(
    bos_stats: dict[str, Any],
    rux_stats: dict[str, Any],
    comparison: dict[str, Any],
    anomalies: dict[str, list[dict[str, Any]]],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Clinical interpretation narrative."""
    h_id = patients[0].patient_id
    risk_label = bos_stats.get("risk_label", "N/A")
    slope = bos_stats.get("trend", {}).get("slope_per_day", 0)
    pct_elevated = bos_stats.get("pct_elevated", 0)
    h_anomalies = anomalies.get(h_id, [])

    rux_sig = rux_stats.get("significant", False)
    rux_delta = rux_stats.get("delta_mean", 0)
    rux_label = rux_stats.get("effect_label", "N/A")

    coupling = comparison.get("coupling", {}).get(h_id, {})
    coupling_intact = coupling.get("coupling_intact", False)

    # Anomaly table
    anomaly_html = ""
    if h_anomalies:
        rows = "".join(
            f'<tr><td style="padding:6px 12px;">{a["date"]}</td>'
            f'<td style="padding:6px 12px;">{a["breath_rate"]:.1f}</td>'
            f'<td style="padding:6px 12px;">{a["z_score"]:.1f}</td>'
            f'<td style="padding:6px 12px;">{a.get("hrv", "N/A")}</td>'
            f'<td style="padding:6px 12px;">{a.get("hr", "N/A")}</td></tr>'
            for a in h_anomalies[:15]
        )
        anomaly_html = (
            f'<h3 style="color:{TEXT_PRIMARY};margin-top:20px;">Anomalous Nights (Henrik)</h3>'
            f'<p style="color:{TEXT_SECONDARY};">Nights with breathing rate &gt;2 SD above personal mean '
            f'({len(h_anomalies)} detected):</p>'
            f'<table style="width:100%;border-collapse:collapse;margin-top:8px;">'
            f'<thead><tr style="border-bottom:1px solid {TEXT_SECONDARY};">'
            f'<th style="padding:6px 12px;text-align:left;color:{TEXT_SECONDARY};">Date</th>'
            f'<th style="padding:6px 12px;text-align:left;color:{TEXT_SECONDARY};">Breath (brpm)</th>'
            f'<th style="padding:6px 12px;text-align:left;color:{TEXT_SECONDARY};">Z-Score</th>'
            f'<th style="padding:6px 12px;text-align:left;color:{TEXT_SECONDARY};">HRV (ms)</th>'
            f'<th style="padding:6px 12px;text-align:left;color:{TEXT_SECONDARY};">HR (bpm)</th>'
            f'</tr></thead><tbody style="color:{TEXT_PRIMARY};">{rows}</tbody></table>'
        )

    return (
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;">'
        f'<strong>BOS Screening:</strong> Henrik\'s breathing rate trend is classified as '
        f'<strong>{risk_label}</strong>. The linear trend slope is {slope:+.4f} brpm/day '
        f'({pct_elevated:.1f}% of nights above the {BREATH_ELEVATED} brpm elevated threshold). '
        f'Post-HSCT patients require ongoing monitoring for BOS, especially with chronic GVHD.</p>'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;margin-top:12px;">'
        f'<strong>Ruxolitinib Effect:</strong> '
        f'{"Statistically significant" if rux_sig else "No statistically significant"} '
        f'difference in breathing rate pre vs post Ruxolitinib '
        f'(delta: {rux_delta:+.2f} brpm, effect: {rux_label}). '
        f'Ruxolitinib, a JAK inhibitor, reduces inflammatory cytokines and may '
        f'modulate respiratory drive indirectly through GVHD suppression.</p>'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;margin-top:12px;">'
        f'<strong>Autonomic Coupling:</strong> '
        f'{"Respiratory-autonomic coupling appears intact" if coupling_intact else "Respiratory-autonomic coupling is disrupted or weak"} '
        f'in Henrik (Spearman r={coupling.get("spearman_r", 0):.2f}). '
        f'Disrupted breath-HRV coupling in HSCT patients may reflect '
        f'autonomic neuropathy or chronic inflammatory burden affecting vagal tone.</p>'
        f'{anomaly_html}'
        f'<p style="color:#6B7280;line-height:1.7;margin-top:16px;">'
        f'<em>This analysis uses nighttime breathing rate from Oura Ring sleep periods '
        f'(long sleep only). Clinical BOS diagnosis requires pulmonary function tests '
        f'(FEV1, DLCO). Wearable breathing rate is a screening adjunct, not diagnostic.</em></p>'
    )


def _build_methodology() -> str:
    """Methodology section."""
    return (
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;">'
        f'<strong>Data Source:</strong> Oura Ring sleep periods (type=long_sleep), '
        f'average_breath column. Both patients\' data loaded from separate SQLite databases.</p>'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;margin-top:8px;">'
        f'<strong>BOS Screening:</strong> Linear regression on daily breathing rate. '
        f'Thresholds: elevated &gt;{BREATH_ELEVATED} brpm, concerning trend &gt;{BOS_TREND_CONCERN} brpm/day. '
        f'Normal sleep breathing range: {BREATH_NORMAL_LOW}-{BREATH_NORMAL_HIGH} brpm.</p>'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;margin-top:8px;">'
        f'<strong>Ruxolitinib Comparison:</strong> Mann-Whitney U test (non-parametric, '
        f'appropriate for non-normal distributions). Cohen\'s d for effect size, '
        f'bootstrap 95% CI for mean difference (10,000 iterations).</p>'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;margin-top:8px;">'
        f'<strong>Cross-Patient Comparison:</strong> Mann-Whitney U with Cohen\'s d. '
        f'Z-score normalization relative to each patient\'s own mean/SD for fair comparison.</p>'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;margin-top:8px;">'
        f'<strong>Coupling Analysis:</strong> Spearman rank correlation between breathing rate '
        f'and HRV (RMSSD) on overlapping dates. Respiratory sinus arrhythmia predicts '
        f'negative correlation (higher HRV, lower breath rate) when vagal modulation is intact.</p>'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;margin-top:8px;">'
        f'<strong>Anomaly Detection:</strong> Nights with breathing rate &gt;{ANOMALY_SD_THRESHOLD} SD '
        f'above personal mean are flagged. Co-occurring HRV and HR values provide clinical context.</p>'
    )


# ---------------------------------------------------------------------------
# [7/7] JSON Export
# ---------------------------------------------------------------------------

def export_json(
    bos_stats: dict[str, Any],
    rux_stats: dict[str, Any],
    comparison: dict[str, Any],
    anomalies: dict[str, list[dict[str, Any]]],
    patients: tuple[PatientConfig, PatientConfig],
    data: dict[str, dict[str, Any]],
) -> None:
    """Write structured metrics JSON."""
    h_id = patients[0].patient_id
    m_id = patients[1].patient_id
    h_breath = data[h_id]["breath"]
    m_breath = data[m_id]["breath"]

    output: dict[str, Any] = {
        "report": "comparative_breathing",
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "patients": {
            h_id: {
                "label": patients[0].display_name,
                "breath_rate": {
                    "mean": float(h_breath.mean()) if not h_breath.empty else None,
                    "median": float(h_breath.median()) if not h_breath.empty else None,
                    "std": float(h_breath.std()) if not h_breath.empty else None,
                    "min": float(h_breath.min()) if not h_breath.empty else None,
                    "max": float(h_breath.max()) if not h_breath.empty else None,
                    "n_days": len(h_breath),
                },
            },
            m_id: {
                "label": patients[1].display_name,
                "breath_rate": {
                    "mean": float(m_breath.mean()) if not m_breath.empty else None,
                    "median": float(m_breath.median()) if not m_breath.empty else None,
                    "std": float(m_breath.std()) if not m_breath.empty else None,
                    "min": float(m_breath.min()) if not m_breath.empty else None,
                    "max": float(m_breath.max()) if not m_breath.empty else None,
                    "n_days": len(m_breath),
                },
            },
        },
        "bos_screening": bos_stats,
        "ruxolitinib_effect": rux_stats,
        "comparison": comparison,
        "anomalies": {
            h_id: anomalies.get(h_id, []),
            m_id: anomalies.get(m_id, []),
        },
    }

    # Sanitize NaN/Inf for JSON
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
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
    """Run comparative breathing analysis pipeline."""
    logger.info("[1/7] Loading patient data...")
    patients = default_patients()
    if patients[1] is None:
        print("Skipping: mitch.db not found (second patient data not available)")
        return 0
    raw_data = load_data(patients)

    logger.info("[2/7] Computing rolling metrics...")
    data = compute_rolling(raw_data)

    logger.info("[3/7] BOS screening (Henrik)...")
    bos_stats = compute_bos_screening(data, patients[0].patient_id)

    logger.info("[4/7] Pre/post Ruxolitinib analysis...")
    rux_stats = compute_rux_effect(data, patients[0].patient_id)

    logger.info("[5/7] Cross-patient comparison...")
    comparison = compute_comparison(data, patients)

    logger.info("[5b/7] Anomaly detection...")
    anomalies = detect_anomalies(data)

    logger.info("[6/7] Generating HTML report...")
    html = build_html(data, bos_stats, rux_stats, comparison, anomalies, patients)
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", HTML_OUTPUT)

    logger.info("[7/7] Exporting JSON metrics...")
    export_json(bos_stats, rux_stats, comparison, anomalies, patients, data)

    logger.info("Comparative breathing analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
