#!/usr/bin/env python3
"""Module 1: Autonomic Recovery Trajectories.

Compares HRV and resting HR recovery patterns between Patient 1 (post-HSCT)
and Patient 2 (post-stroke), normalized to days-since-their-major-event.

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

from profiles import PROFILES
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
    TEXT_TERTIARY,
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


def _classify_severity(
    hrv_series: pd.Series,
    patient_id: str,
    recent_window: int = 30,
) -> dict[str, Any]:
    """Classify autonomic severity using recent data and trajectory.

    Uses the last ``recent_window`` days of HRV data to determine current
    severity, then checks year-over-year trajectory to adjust the
    classification when values are declining.

    Classification tiers (based on recent-window RMSSD mean):
      - severe_autonomic_dysfunction:   RMSSD < 15 ms
                                        OR (RMSSD < 25 ms AND declining)
      - moderate_autonomic_dysfunction: RMSSD 15-25 ms
                                        OR (RMSSD 25-40 ms AND declining >10%/yr)
      - mild_autonomic_impairment:      RMSSD 25-40 ms AND stable
      - normal_range:                   RMSSD > 40 ms AND stable or improving

    For patients aged > 55, RMSSD 20-40 ms is considered age-appropriate normal
    and thresholds are adjusted accordingly.
    """
    if hrv_series.empty:
        return {
            "classification": "insufficient_data",
            "recent_mean_ms": None,
            "full_history_mean_ms": None,
            "recent_window_days": recent_window,
            "trajectory": "insufficient data",
            "yoy_change_pct": None,
            "age_adjusted": False,
            "age_adjustment_note": None,
        }

    full_mean = float(hrv_series.mean())

    # Calendar-based recent window (last N days by date, not tail N rows)
    clean = hrv_series.dropna()
    last_date = clean.index.max()
    cutoff = last_date - pd.Timedelta(days=recent_window)
    recent = clean.loc[clean.index > cutoff]
    if recent.empty:
        recent = clean
    recent_mean = float(recent.mean())

    # Year-over-year change: compare earliest 90-day window to latest 90-day window
    yoy_change_pct = None
    if len(hrv_series) >= 180:
        early_window = hrv_series.head(90).mean()
        late_window = hrv_series.tail(90).mean()
        if early_window > 0:
            yoy_change_pct = float((late_window - early_window) / early_window * 100)

    # Recent-window trend (30-day linear regression)
    recent_trend = _compute_linear_trend(recent)
    trajectory = recent_trend["direction"]

    declining = trajectory == "declining"
    declining_significant = (
        yoy_change_pct is not None and yoy_change_pct < -10
    )

    # Age adjustment for patients > 55
    patient_age = PROFILES.get(patient_id, {}).get("age")
    age_adjusted = False
    age_note = None
    if patient_age is not None and patient_age > 55:
        age_adjusted = True
        age_note = (
            f"Patient age {patient_age}: RMSSD 20-40 ms is within "
            f"age-expected range for adults >55. Population norms decline "
            f"~3-5 ms/decade after age 30."
        )

    # Classification logic
    if age_adjusted:
        # Age-adjusted thresholds for >55
        if recent_mean < 15:
            classification = "severe_autonomic_dysfunction"
        elif recent_mean < 20 or (recent_mean < 25 and declining):
            classification = "moderate_autonomic_dysfunction"
        elif recent_mean < 30 and declining_significant:
            classification = "moderate_autonomic_dysfunction"
        elif recent_mean < 40 and declining_significant:
            classification = "mild_autonomic_impairment"
        else:
            classification = "normal_range"
    else:
        # Standard thresholds
        if recent_mean < 15 or (recent_mean < 25 and declining):
            classification = "severe_autonomic_dysfunction"
        elif (15 <= recent_mean < 25) or (25 <= recent_mean < 40 and declining_significant):
            classification = "moderate_autonomic_dysfunction"
        elif 25 <= recent_mean < 40:
            classification = "mild_autonomic_impairment"
        elif recent_mean >= 40 and not declining and not declining_significant:
            classification = "normal_range"
        else:
            # RMSSD >= 40 but declining (recent trend or >10% year-over-year)
            classification = "mild_autonomic_impairment"

    return {
        "classification": classification,
        "recent_mean_ms": round(recent_mean, 1),
        "full_history_mean_ms": round(full_mean, 1),
        "recent_window_days": int(len(recent)),
        "trajectory": trajectory,
        "yoy_change_pct": round(yoy_change_pct, 1) if yoy_change_pct is not None else None,
        "age_adjusted": age_adjusted,
        "age_adjustment_note": age_note,
    }


# ---------------------------------------------------------------------------
# [1/7] Data Loading
# ---------------------------------------------------------------------------

def load_data(
    patients: list[PatientConfig],
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
    patients: list[PatientConfig],
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
    patients: list[PatientConfig],
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

        dse = metrics.get("days_since_event", pd.Series(dtype=float))
        dse_valid = dse.dropna()
        dse_range = (
            [int(dse_valid.min()), int(dse_valid.max())]
            if not dse_valid.empty
            else [0, 0]
        )

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

    # Severity classification for all patients (recent-window + trajectory)
    severity_classifications: dict[str, dict[str, Any]] = {}
    for pid, metrics in data.items():
        severity_classifications[pid] = _classify_severity(metrics["hrv"], pid)

    stats_result["severity_classification"] = severity_classifications

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
        }

    return stats_result


# ---------------------------------------------------------------------------
# [5b/7] Recent Window & Trend Assessment
# ---------------------------------------------------------------------------

def compute_recent_window(
    data: dict[str, dict[str, Any]],
    patients: list[PatientConfig],
    window_days: int = 30,
) -> dict[str, Any]:
    """Compute recent-window stats and full-period linear trends per patient.

    Returns a dict keyed by patient_id, each containing:
      - full_mean / recent_mean for HRV, lowest HR, avg HR
      - pct_change (recent vs full)
      - declining flag (recent mean >10% below full-history mean)
      - full-period linear trend (slope, direction)
    """
    patient_map = {p.patient_id: p for p in patients}
    result: dict[str, Any] = {}

    for pid, metrics in data.items():
        p = patient_map[pid]
        hrv = metrics["hrv"]
        hr = metrics["hr"]
        hr_low = metrics["hr_lowest"]

        entry: dict[str, Any] = {
            "label": p.display_name,
            "window_days": window_days,
        }

        for metric_name, series in [("hrv", hrv), ("lowest_hr", hr_low), ("avg_hr", hr)]:
            clean = series.dropna()
            if clean.empty:
                entry[metric_name] = {
                    "full_mean": None, "recent_mean": None,
                    "pct_change": None, "declining": False,
                    "recent_start": None, "recent_end": None,
                }
                continue

            full_mean = float(clean.mean())

            # Most recent N calendar days (not tail N rows)
            last_date = clean.index.max()
            cutoff = last_date - pd.Timedelta(days=window_days)
            recent = clean.loc[clean.index > cutoff]
            recent_mean = float(recent.mean()) if not recent.empty else full_mean

            pct_change = ((recent_mean - full_mean) / full_mean * 100) if full_mean != 0 else 0.0
            declining = pct_change < -10.0

            entry[metric_name] = {
                "full_mean": round(full_mean, 1),
                "recent_mean": round(recent_mean, 1),
                "pct_change": round(pct_change, 1),
                "declining": declining,
                "recent_start": str(recent.index.min().date()) if not recent.empty else None,
                "recent_end": str(recent.index.max().date()) if not recent.empty else None,
                "recent_n": int(len(recent)),
            }

        # Full-period linear trend for HRV
        hrv_clean = hrv.dropna()
        if len(hrv_clean) >= 10:
            x = np.arange(len(hrv_clean), dtype=float)
            y = hrv_clean.values.astype(float)
            slope, intercept, r, p_val, se = scipy_stats.linregress(x, y)
            slope_per_week = slope * 7
            total_days = (hrv_clean.index.max() - hrv_clean.index.min()).days
            predicted_start = intercept
            predicted_end = intercept + slope * len(hrv_clean)
            entry["full_period_trend"] = {
                "slope_per_week": round(float(slope_per_week), 3),
                "p_value": float(p_val),
                "r_squared": round(float(r ** 2), 3),
                "direction": _trend_direction(slope, p_val),
                "predicted_start": round(float(predicted_start), 1),
                "predicted_end": round(float(predicted_end), 1),
                "total_days": total_days,
            }
        else:
            entry["full_period_trend"] = {
                "slope_per_week": None,
                "p_value": None,
                "r_squared": None,
                "direction": "insufficient data",
                "predicted_start": None,
                "predicted_end": None,
                "total_days": 0,
            }

        result[pid] = entry

    return result


def _build_recent_window_html(
    recent_window: dict[str, Any],
    patients: list[PatientConfig],
) -> str:
    """Build HTML section for the recent 30-day window comparison."""
    patient_map = {p.patient_id: p for p in patients}

    # Table header
    rows = []
    for i, p in enumerate(patients):
        pid = p.patient_id
        rw = recent_window.get(pid)
        if rw is None:
            continue
        label = f"P{i + 1}"
        hrv = rw.get("hrv", {})
        hr_low = rw.get("lowest_hr", {})
        avg_hr = rw.get("avg_hr", {})

        def _fmt(val: float | None) -> str:
            return f"{val:.1f}" if val is not None else "N/A"

        def _pct_badge(pct: float | None, declining: bool) -> str:
            if pct is None:
                return '<span style="color:#6B7280;">N/A</span>'
            color = ACCENT_RED if declining else (ACCENT_AMBER if pct < -5 else ACCENT_GREEN)
            arrow = "&#9660;" if pct < 0 else "&#9650;" if pct > 0 else "&#8212;"
            return f'<span style="color:{color};font-weight:600;">{arrow} {pct:+.1f}%</span>'

        def _trend_badge(trend: dict) -> str:
            direction = trend.get("direction", "N/A")
            slope = trend.get("slope_per_week")
            colors = {"improving": ACCENT_GREEN, "declining": ACCENT_RED, "stable": ACCENT_AMBER}
            color = colors.get(direction, TEXT_SECONDARY)
            slope_str = f" ({slope:+.2f} ms/wk)" if slope is not None else ""
            return f'<span style="color:{color};font-weight:600;">{direction.upper()}{slope_str}</span>'

        trend = rw.get("full_period_trend", {})
        pred_start = trend.get("predicted_start")
        pred_end = trend.get("predicted_end")
        trend_note = ""
        if pred_start is not None and pred_end is not None:
            trend_note = f"<br><span style='color:#6B7280;font-size:0.85em;'>Regression: {pred_start:.0f} &rarr; {pred_end:.0f} ms over {trend.get('total_days', 0)} days</span>"

        rows.append(f"""
        <tr>
          <td style="font-weight:600;color:{PATIENT_COLORS.get(pid, ACCENT_PURPLE)};">{label}</td>
          <td>{_fmt(hrv.get('full_mean'))}</td>
          <td>{_fmt(hrv.get('recent_mean'))} {_pct_badge(hrv.get('pct_change'), hrv.get('declining', False))}</td>
          <td>{_fmt(hr_low.get('full_mean'))}</td>
          <td>{_fmt(hr_low.get('recent_mean'))} {_pct_badge(hr_low.get('pct_change'), hr_low.get('declining', False))}</td>
          <td>{_fmt(avg_hr.get('full_mean'))}</td>
          <td>{_fmt(avg_hr.get('recent_mean'))} {_pct_badge(avg_hr.get('pct_change'), avg_hr.get('declining', False))}</td>
          <td>{_trend_badge(trend)}{trend_note}</td>
        </tr>""")

    # Flags for declining patients
    flags = []
    for i, p in enumerate(patients):
        pid = p.patient_id
        rw = recent_window.get(pid)
        if rw is None:
            continue
        label = f"P{i + 1}"
        for metric_label, key in [("HRV", "hrv"), ("Lowest HR", "lowest_hr"), ("Avg HR", "avg_hr")]:
            m = rw.get(key, {})
            if m.get("declining"):
                pct = m.get("pct_change", 0)
                flags.append(
                    f'<div style="color:{ACCENT_RED};padding:6px 12px;background:rgba(239,68,68,0.08);'
                    f'border-radius:6px;margin-top:6px;font-size:0.9em;">'
                    f'&#9888; {label} {metric_label}: recent 30-day mean is {abs(pct):.1f}% below full-history mean '
                    f'({m.get("full_mean")} &rarr; {m.get("recent_mean")})'
                    f'</div>'
                )

    flags_html = "\n".join(flags) if flags else (
        f'<div style="color:{ACCENT_GREEN};padding:6px 12px;font-size:0.9em;">'
        'No patients flagged for &gt;10% decline in recent window.</div>'
    )

    html = f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:0.9em;margin:12px 0;">
      <thead>
        <tr style="border-bottom:1px solid {TEXT_TERTIARY};color:{TEXT_SECONDARY};text-align:left;">
          <th style="padding:8px 6px;"></th>
          <th style="padding:8px 6px;">HRV Full</th>
          <th style="padding:8px 6px;">HRV 30d</th>
          <th style="padding:8px 6px;">Low HR Full</th>
          <th style="padding:8px 6px;">Low HR 30d</th>
          <th style="padding:8px 6px;">Avg HR Full</th>
          <th style="padding:8px 6px;">Avg HR 30d</th>
          <th style="padding:8px 6px;">Full-Period Trend</th>
        </tr>
      </thead>
      <tbody style="color:{TEXT_PRIMARY};">
        {"".join(rows)}
      </tbody>
    </table>
    </div>
    <div style="margin-top:8px;">
      <strong style="color:{TEXT_SECONDARY};font-size:0.9em;">Decline Flags (recent 30d mean &gt;10% below full-history mean):</strong>
      {flags_html}
    </div>
    <p style="color:{TEXT_TERTIARY};font-size:0.8em;margin-top:10px;">
      Full-period trend uses linear regression across each patient's entire data range.
      Declining = recent 30-day mean is more than 10% below full-history mean, indicating
      the full-history average overstates current health status.
    </p>
    """
    return html


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _fig_hrv_trajectory(
    data: dict[str, dict[str, Any]],
    patients: list[PatientConfig],
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
    patients: list[PatientConfig],
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
    patients: list[PatientConfig],
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
    patients: list[PatientConfig],
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
            opacity=0.7,
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
    patients: list[PatientConfig],
) -> go.Figure:
    """Fig 5: All patients' HRV timeline overlaid, longest dataset as background."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    # Sort patients by data length (longest first) for background layering
    sorted_pids = sorted(data.keys(), key=lambda pid: len(data[pid].get("hrv", pd.Series(dtype=float))), reverse=True)

    for i, pid in enumerate(sorted_pids):
        p = patient_map.get(pid)
        if p is None:
            continue
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        hrv = data[pid]["hrv"]
        if hrv.empty:
            continue

        # Daily scatter
        fig.add_trace(go.Scatter(
            x=hrv.index, y=hrv.values,
            mode="markers",
            marker=dict(size=2 if i == 0 else 4, color=color, opacity=0.2 if i == 0 else 0.6),
            name=f"{p.display_name} (daily)", legendgroup=pid, showlegend=False,
        ))
        # Rolling average
        roll_window = 30 if i == 0 else 7
        min_per = roll_window // 2
        if len(hrv) >= roll_window:
            rolling = hrv.rolling(roll_window, min_periods=min_per).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values,
                mode="lines", line=dict(color=color, width=2.5),
                name=f"{p.display_name} ({roll_window}d avg)", legendgroup=pid,
            ))

    _add_reference_line(fig, ESC_RMSSD_DEFICIENCY, f"ESC ({ESC_RMSSD_DEFICIENCY}ms)", ACCENT_RED)
    _add_reference_line(fig, POPULATION_RMSSD_MEDIAN, f"Pop. Median ({POPULATION_RMSSD_MEDIAN}ms)", ACCENT_AMBER)

    fig.update_layout(
        height=450,
        title=dict(text="Long-Term HRV Context: All Patients", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="RMSSD (ms)",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
    )
    return fig


def _fig_autonomic_coupling(
    data: dict[str, dict[str, Any]],
    patients: list[PatientConfig],
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
    patients: list[PatientConfig],
    recent_window: dict[str, Any] | None = None,
) -> str:
    """Build the full HTML report."""
    sections: list[str] = []

    # -- KPI Row -- one card per patient for HRV and HR
    comp = stats_result.get("comparison", {})
    kpi_cards: list[str] = []
    for i, p in enumerate(patients):
        pid = p.patient_id
        p_stats = stats_result["patients"].get(pid, {})
        hrv_mean = p_stats.get("hrv", {}).get("mean", 0)
        hr_mean = p_stats.get("heart_rate", {}).get("mean_sleep_hr", 0)
        label = f"P{i + 1}"
        kpi_cards.append(make_kpi_card(
            f"{label} MEAN HRV", hrv_mean, "ms",
            status="critical" if hrv_mean < ESC_RMSSD_DEFICIENCY else "info",
            detail=f"Below ESC threshold ({ESC_RMSSD_DEFICIENCY}ms)" if hrv_mean < ESC_RMSSD_DEFICIENCY else f"Pop. percentile: {p_stats.get('hrv', {}).get('population_percentile', 0):.0f}%",
        ))
        kpi_cards.append(make_kpi_card(
            f"{label} SLEEP HR", hr_mean, "bpm",
            status="warning" if hr_mean > 75 else "normal",
            detail="Elevated" if hr_mean > 75 else "Normal range",
        ))
    # Henrik-specific trajectory card
    h_stats = stats_result["patients"].get("henrik", {})
    h_trend_dir = h_stats.get("hrv", {}).get("trend_direction", "N/A")
    kpi_cards.append(make_kpi_card(
        "TRAJECTORY", h_trend_dir.title(), "",
        status="normal" if h_trend_dir == "improving" else ("warning" if h_trend_dir == "stable" else "critical"),
        detail="P1's 30-day HRV trend",
        status_label=h_trend_dir.title(),
    ))
    sections.append(make_kpi_row(*kpi_cards))

    # -- Recent Window Comparison --
    if recent_window:
        sections.append(section_html_or_placeholder(
            "Recent 30-Day Window",
            lambda: make_section(
                "Recent 30-Day Window vs Full History",
                _build_recent_window_html(recent_window, patients),
                section_id="recent-window",
            ),
        ))

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
        "<strong>Patient 1</strong> is 2+ years post-allogeneic HSCT with chronic GVHD and "
        "severe autonomic dysfunction (HRV consistently below the ESC 15ms threshold). "
        "<strong>Patient 2</strong> is ~15 months post-stroke (bilateral carotid/vertebral "
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
        header_meta=" vs ".join(p.display_name for p in patients),
    )


# ---------------------------------------------------------------------------
# [7/7] JSON Export
# ---------------------------------------------------------------------------

def export_json(
    stats_result: dict[str, Any],
    recent_window: dict[str, Any] | None = None,
) -> None:
    """Write structured metrics JSON."""
    output = {
        "report": "comparative_autonomic",
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        **stats_result,
    }
    if recent_window:
        output["recent_window"] = recent_window

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
    if len(patients) < 2:
        print("Skipping: need at least 2 patient databases for comparative analysis")
        return 0
    patient_map = {p.patient_id: p for p in patients}
    raw_data = load_data(patients)

    logger.info("[2/7] Normalizing timelines...")
    data = normalize_timelines(raw_data, patients)

    logger.info("[3/7] Computing rolling metrics...")
    data = compute_rolling(data)

    logger.info("[4/7] Computing normalized metrics...")
    data = compute_normalized(data)

    logger.info("[5/7] Computing trends and comparison...")
    stats_result = compute_trends_and_comparison(data, patients)

    for pid, sev in stats_result.get("severity_classification", {}).items():
        logger.info(
            "  %s severity: %s (recent %.1f ms, full-history %.1f ms, trajectory=%s%s)",
            pid,
            sev["classification"],
            sev.get("recent_mean_ms") or 0,
            sev.get("full_history_mean_ms") or 0,
            sev.get("trajectory", "N/A"),
            f", age-adjusted" if sev.get("age_adjusted") else "",
        )

    logger.info("[5b/7] Computing recent window comparison...")
    recent_window = compute_recent_window(data, patients)
    for pid, rw in recent_window.items():
        hrv_info = rw.get("hrv", {})
        trend_info = rw.get("full_period_trend", {})
        logger.info(
            "  %s: HRV full=%.1f, recent-30d=%.1f (%+.1f%%), trend=%s",
            rw.get("label", pid),
            hrv_info.get("full_mean") or 0,
            hrv_info.get("recent_mean") or 0,
            hrv_info.get("pct_change") or 0,
            trend_info.get("direction", "N/A"),
        )

    logger.info("[6/7] Generating HTML report...")
    html = build_html(data, stats_result, patients, recent_window=recent_window)
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", HTML_OUTPUT)

    logger.info("[7/7] Exporting JSON metrics...")
    export_json(stats_result, recent_window=recent_window)

    logger.info("Comparative autonomic analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
