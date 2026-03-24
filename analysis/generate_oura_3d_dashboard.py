#!/usr/bin/env python3
"""
generate_oura_3d_dashboard.py — Unified CMO Dashboard

The front door. One HTML file shows everything: CausalImpact p-values,
UKF state estimates, disease-state models, anomaly detection, composite
biomarkers, and raw biosignal visualizations. The goal is fast external
review of an exploratory N=1 wearable dataset, not clinical proof.

Architecture:
  Tab 1 (Overview)     — Narrative + KPI cards + Hero ITS + Forest plot
  Tab 2 (Disease)      — rSLDS states + Anomaly timeline + Biomarkers + SpO2
  Tab 3 (Biosignals)   — Raw biosignal timeline + HR Terrain + Phase Space
                          + Sleep heatmap + Circadian radar

Data sources:
  - SQLite database (raw Oura ring data)
  - 10 JSON files from analysis modules (p-values, states, biomarkers)

Output: reports/oura_3d_dashboard.html
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH, REPORTS_DIR,
    TREATMENT_START_STR, KNOWN_EVENT_DATE,
    DATA_START,
    PATIENT_LABEL, PATIENT_TIMEZONE,
    FONT_FAMILY, PLOTLY_CDN_URL,
    C_CRITICAL, C_ACCENT,
    C_PRE_TX, C_POST_TX,
)

from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section, format_p_value,
    BG_PRIMARY, BG_SURFACE, BG_ELEVATED,
    BORDER_SUBTLE, BORDER_DEFAULT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_PURPLE, ACCENT_CYAN,
    STATUS_COLORS,
)
from _bos_risk import BOS_COMPONENT_LABELS, bos_status, format_bos_label
pio.templates.default = "clinical_dark"

OUTPUT_FILE = REPORTS_DIR / "oura_3d_dashboard.html"

# ---------------------------------------------------------------------------
# Clinical events
# ---------------------------------------------------------------------------
RUXOLITINIB_START = TREATMENT_START_STR
ACUTE_EVENT = KNOWN_EVENT_DATE

# ---------------------------------------------------------------------------
# Dark theme color palette (colorblind-safe)
# ---------------------------------------------------------------------------
C_PRE = "#4589FF"        # Blue — pre-ruxolitinib
C_POST = "#F59E0B"       # Amber — post-ruxolitinib (design system ACCENT_AMBER)
C_INTERVENTION = "#FFFFFF"  # White dashed — March 16 line
C_CI_BAND = "rgba(255,255,255,0.1)"  # Subtle — confidence intervals
C_ALERT = "#CC79A7"      # Magenta — anomaly alerts
C_COUNTERFACTUAL = "#999999"  # Grey — predicted counterfactual
C_STATES = ["#10B981", "#F59E0B", "#EF4444", "#3B82F6"]  # Remission, Pre-flare, Flare, Recovery
C_STATE_NAMES = ["Remission", "Pre-flare", "Active Flare", "Recovery"]

# Existing colors (for 3D panels) — from config
COLOR_PRE_RUX = C_PRE_TX
COLOR_POST_RUX = C_POST_TX
COLOR_EVENT = C_ACCENT
COLOR_SURFACE_SCALE = "Plasma"

# Sleep phase colors
PHASE_COLORS = {1: "#6366F1", 2: "#3B82F6", 3: "#10B981", 4: "#EF4444"}
PHASE_NAMES = {1: "Deep", 2: "Light", 3: "REM", 4: "Awake"}

# Note: DARK_LAYOUT removed — the "clinical_dark" Plotly template
# handles paper_bgcolor, plot_bgcolor, font, gridcolor automatically.


def _add_vline(
    fig: go.Figure,
    x: str,
    line_dash: str = "dash",
    line_color: str = "white",
    line_width: float = 2,
    annotation_text: str | None = None,
    annotation_position: str = "top",
    annotation_font: dict | None = None,
    row: int | str | None = None,
    col: int | str | None = None,
) -> None:
    """Add vertical line + optional annotation without Plotly string-axis bug.

    Plotly's add_vline with annotation_text fails when the x-axis uses string
    values (dates as strings) because it tries to compute mean() of strings.
    This helper uses add_shape + add_annotation directly instead.
    """
    shape_kwargs: dict[str, Any] = dict(
        type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
        line=dict(dash=line_dash, color=line_color, width=line_width),
    )
    if row is not None and col is not None:
        fig.add_shape(**shape_kwargs, row=row, col=col)
    else:
        fig.add_shape(**shape_kwargs)

    if annotation_text:
        y_pos = 1.02 if annotation_position == "top" else -0.05
        a_font = annotation_font or dict(color=TEXT_PRIMARY, size=11)
        fig.add_annotation(
            x=x, y=y_pos, yref="paper",
            text=annotation_text, showarrow=False,
            font=a_font, bgcolor=f"rgba(15,17,23,0.7)",
            borderpad=3,
        )


# Alias for backwards-compat within this file — canonical is _theme.format_p_value
_format_p_value = format_p_value


# ===================================================================
# DATABASE CONNECTION
# ===================================================================

def get_db_connection() -> sqlite3.Connection:
    """Open read-only connection to biometrics database."""
    if not DATABASE_PATH.exists():
        print(f"ERROR: Database not found at {DATABASE_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _apply_data_start(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Restrict frame to analysis window start date from config."""
    if col not in df.columns:
        return df
    return df[df[col] >= DATA_START].reset_index(drop=True)


# ===================================================================
# DATA LOADING — RAW DATABASE
# ===================================================================

def load_heart_rate(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all heart rate readings with parsed timestamps."""
    df = pd.read_sql_query(
        "SELECT timestamp, bpm FROM oura_heart_rate ORDER BY timestamp", conn,
    )
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_local"] = df["ts"].dt.tz_convert(PATIENT_TIMEZONE)
    df["date"] = df["ts_local"].dt.date
    df["hour"] = df["ts_local"].dt.hour
    df["minute"] = df["ts_local"].dt.minute
    df["time_decimal"] = df["hour"] + df["minute"] / 60.0
    return _apply_data_start(df, "date")


def load_hrv(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load HRV readings."""
    df = pd.read_sql_query(
        "SELECT timestamp, rmssd FROM oura_hrv WHERE rmssd IS NOT NULL ORDER BY timestamp",
        conn,
    )
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_local"] = df["ts"].dt.tz_convert(PATIENT_TIMEZONE)
    df["date"] = df["ts_local"].dt.date
    df["hour"] = df["ts_local"].dt.hour
    return _apply_data_start(df, "date")


def load_sleep_periods(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load sleep periods (main sleep only)."""
    df = pd.read_sql_query(
        """SELECT period_id, day, type, average_hrv, average_heart_rate,
                  average_breath, total_sleep_duration, rem_sleep_duration,
                  deep_sleep_duration, light_sleep_duration, awake_time,
                  efficiency, lowest_heart_rate, bedtime_start, bedtime_end,
                  time_in_bed, restless_periods, latency
           FROM oura_sleep_periods
           WHERE type IN ('long_sleep', 'sleep')
           ORDER BY day""",
        conn,
    )
    df["day_date"] = pd.to_datetime(df["day"]).dt.date
    for col in ["total_sleep_duration", "rem_sleep_duration",
                "deep_sleep_duration", "light_sleep_duration",
                "awake_time", "time_in_bed"]:
        if col in df.columns:
            df[f"{col}_h"] = df[col] / 3600.0
    return _apply_data_start(df, "day_date")


def load_sleep_epochs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load sleep epochs joined with period dates."""
    df = pd.read_sql_query(
        """SELECT se.period_id, se.epoch_index, se.phase, sp.day
           FROM oura_sleep_epochs se
           JOIN oura_sleep_periods sp ON se.period_id = sp.period_id
           WHERE sp.type IN ('long_sleep', 'sleep')
           ORDER BY sp.day, se.epoch_index""",
        conn,
    )
    df["day_date"] = pd.to_datetime(df["day"]).dt.date
    return _apply_data_start(df, "day_date")


def load_spo2(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load SpO2 data."""
    df = pd.read_sql_query(
        "SELECT date, spo2_average FROM oura_spo2 WHERE spo2_average > 0 ORDER BY date",
        conn,
    )
    df["day_date"] = pd.to_datetime(df["date"]).dt.date
    return _apply_data_start(df, "day_date")


def load_readiness(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load readiness data for temperature deviation."""
    df = pd.read_sql_query(
        "SELECT date, temperature_deviation, score FROM oura_readiness ORDER BY date",
        conn,
    )
    df["day_date"] = pd.to_datetime(df["date"]).dt.date
    return _apply_data_start(df, "day_date")


def load_activity(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load activity data."""
    df = pd.read_sql_query(
        "SELECT date, steps, active_calories FROM oura_activity ORDER BY date",
        conn,
    )
    df["day_date"] = pd.to_datetime(df["date"]).dt.date
    return _apply_data_start(df, "day_date")


# ===================================================================
# DATA LOADING — ANALYSIS MODULE JSON OUTPUTS
# ===================================================================

def load_analysis_outputs(reports_dir: Path) -> dict:
    """Load all analysis module JSON outputs with graceful degradation."""
    outputs: dict[str, dict] = {}
    json_files = {
        "digital_twin": "digital_twin_metrics.json",
        "causal": "causal_inference_metrics.json",
        "causal_ts": "causal_timeseries.json",
        "gvhd": "gvhd_prediction_metrics.json",
        "anomaly": "anomaly_detection_metrics.json",
        "biomarkers": "composite_biomarkers.json",
        "hrv": "advanced_hrv_metrics.json",
        "sleep": "advanced_sleep_metrics.json",
        "spo2": "spo2_bos_metrics.json",
        "foundation": "foundation_model_metrics.json",
        "full_analysis": "oura_full_analysis.json",
    }
    for key, filename in json_files.items():
        path = reports_dir / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                outputs[key] = json.load(f)
            print(f"  Loaded {filename}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            outputs[key] = {}
            print(f"  WARNING: {filename} not available ({e.__class__.__name__})")
    return outputs


# ===================================================================
# NARRATIVE SUMMARY
# ===================================================================

def build_narrative_summary(outputs: dict, summary: dict) -> str:
    """One-sentence clinical answer + supporting numbers."""
    causal = outputs.get("causal", {})
    dt = outputs.get("digital_twin", {})
    streams = causal.get("causal_impact", {}).get("streams", {})

    # Temperature deviation — strongest signal
    temp = streams.get("temperature_deviation", {})
    temp_p = temp.get("p_value")
    temp_prob = temp.get("probability_of_effect")
    temp_q = temp.get("q_value_bh")
    temp_sig_fdr = bool(temp.get("significant_fdr", False))
    temp_low_conf = bool(temp.get("low_confidence", False))

    # Lowest HR
    lhr = streams.get("lowest_heart_rate", {})
    lhr_p = lhr.get("p_value")

    # Drug response from digital twin
    drug = dt.get("drug_response", {}).get("response_stats", {})
    cardiac_p = drug.get("Cardiac Reserve", {}).get("mann_whitney_p")
    sleep_p = drug.get("Sleep Quality", {}).get("mann_whitney_p")

    # Build sentence
    post_days = summary.get("post_days", 0)
    temp_p_label = _format_p_value(temp_p)
    temp_q_label = f"{temp_q:.3f}" if temp_q is not None else "N/A"
    temp_prob_text = f", {temp_prob:.0f}% posterior probability" if temp_prob is not None else ""

    if temp_p is not None and temp_p < 0.05:
        headline = (
            f"Temperature deviation is the strongest exploratory unadjusted p-value signal after "
            f"ruxolitinib ({temp_p_label}, q={temp_q_label}{temp_prob_text}). "
        )
        if not temp_sig_fdr or temp_low_conf:
            qualifiers = []
            if not temp_sig_fdr and temp_q is not None:
                qualifiers.append("does not survive FDR correction")
            if temp_low_conf:
                qualifiers.append("is flagged low confidence")
            headline += "It " + " and ".join(qualifiers) + ". "
        headline += (
            f"The post-intervention window spans {post_days} days and remains confounded by "
            "HEV diagnosed on 2026-03-18. "
        )
    elif temp_p is not None:
        headline = (
            f"Temperature deviation shows the lowest unadjusted p-value observed "
            f"({temp_p_label}, q={temp_q_label}{temp_prob_text}) in a confounded "
            f"{post_days}-day post-intervention window. "
        )
    else:
        headline = (
            f"Analysis modules detected multiple biosignal shifts after ruxolitinib initiation, "
            f"but the {post_days}-day post-intervention window remains too short for strong causal claims. "
        )

    # Supporting stats
    support_parts = []
    if cardiac_p is not None:
        support_parts.append(f"Cardiac Reserve shift {_format_p_value(cardiac_p)}")
    if sleep_p is not None:
        support_parts.append(f"Sleep Quality shift {_format_p_value(sleep_p)}")
    if lhr_p is not None:
        support_parts.append(f"Lowest HR {_format_p_value(lhr_p)}")

    n_streams = len(streams)
    raw_sig_streams = sum(1 for s in streams.values() if isinstance(s, dict) and s.get("p_value", 1) < 0.05)
    fdr_sig_streams = sum(1 for s in streams.values() if isinstance(s, dict) and s.get("significant_fdr", False))

    support = (
        f"{raw_sig_streams}/{n_streams} CausalImpact streams reach raw p<0.05; "
        f"{fdr_sig_streams}/{n_streams} survive FDR correction."
    )
    if support_parts:
        support += " Separate UKF comparisons: " + ", ".join(support_parts) + "."

    return headline + support


# ===================================================================
# KPI CARDS WITH SVG SPARKLINES
# ===================================================================

def _svg_sparkline(values: list[float], width: int = 80, height: int = 24,
                   color: str = ACCENT_BLUE) -> str:
    """Generate an inline SVG sparkline polyline."""
    if not values or len(values) < 2:
        return ""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) < 2:
        return ""
    vmin, vmax = min(vals), max(vals)
    vrange = vmax - vmin if vmax != vmin else 1.0
    points = []
    for i, v in enumerate(vals):
        x = i / (len(vals) - 1) * width
        y = height - ((v - vmin) / vrange) * (height - 2) - 1
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" style="vertical-align:middle;margin-left:8px">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def build_kpi_cards(outputs: dict, summary: dict) -> str:
    """Build 8 KPI cards as HTML with inline SVG sparklines."""
    causal = outputs.get("causal", {})
    dt = outputs.get("digital_twin", {})
    spo2 = outputs.get("spo2", {})
    anomaly = outputs.get("anomaly", {})
    biomarkers = outputs.get("biomarkers", {})
    causal_ts = outputs.get("causal_ts", {})
    full_analysis = outputs.get("full_analysis", {})

    streams = causal.get("causal_impact", {}).get("streams", {})
    drug = dt.get("drug_response", {}).get("response_stats", {})

    def _as_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    # Helper
    def _card(label: str, value: str, unit: str, detail: str, status: str,
              sparkline_svg: str = "") -> str:
        status_colors = {
            "good": ACCENT_GREEN, "normal": ACCENT_GREEN, "warning": ACCENT_AMBER,
            "critical": ACCENT_RED, "neutral": TEXT_TERTIARY,
        }
        sc = status_colors.get(status, TEXT_SECONDARY)
        return f"""
        <div class="kpi-card">
          <div class="kpi-status" style="background:{sc}"></div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span>{sparkline_svg}</div>
          <div class="kpi-detail">{detail}</div>
        </div>"""

    cards = []

    # 1. Temperature
    temp = streams.get("temperature_deviation", {})
    temp_p = temp.get("p_value")
    temp_ts = causal_ts.get("streams", {}).get("temperature_deviation", {})
    temp_vals = temp_ts.get("actual", [])[-14:] if temp_ts else []
    p_str = _format_p_value(temp_p)
    status = "good" if temp_p and temp_p < 0.05 else ("warning" if temp_p and temp_p < 0.1 else "neutral")
    cards.append(_card("Temperature", p_str, "",
                       f"Effect: {temp.get('avg_effect', 0):.2f}\u00b0C" if temp else "No data",
                       status, _svg_sparkline(temp_vals, color=C_POST)))

    # 2. HRV (RMSSD)
    hrv_mean = _as_float(full_analysis.get("rmssd_mean"))
    if hrv_mean is None:
        hrv_mean = _as_float(summary.get("mean_hrv"))
    hrv_str = f"{hrv_mean:.1f}" if hrv_mean else "N/A"
    cards.append(_card("HRV (RMSSD)", hrv_str, "ms",
                       f"vs healthy 42-49 ms | Kleiger 1987 / Bigger 1992",
                       "critical",
                       _svg_sparkline([], color=ACCENT_PURPLE)))

    # 3. Resting HR
    hr_mean = summary.get("mean_hr")
    hr_str = f"{hr_mean:.0f}" if hr_mean else "N/A"
    lhr = streams.get("lowest_heart_rate", {})
    lhr_p = lhr.get("p_value")
    hr_detail = f"Lowest HR {_format_p_value(lhr_p)}" if lhr_p is not None else "IST threshold: 90 bpm"
    cards.append(_card("Resting HR", hr_str, "bpm", hr_detail,
                       "critical" if hr_mean and hr_mean > 90 else "warning"))

    # 4. SpO2 — use summary (same source as JSON output) to avoid mismatch
    spo2_mean = _as_float(full_analysis.get("spo2_mean"))
    if spo2_mean is None:
        spo2_mean = _as_float(summary.get("mean_spo2"))
    bos = spo2.get("bos_risk", {})
    spo2_str = f"{spo2_mean:.1f}" if spo2_mean else "N/A"
    bos_level = bos.get("risk_level", "N/A")
    bos_detail = format_bos_label(bos)
    cards.append(_card("SpO2", spo2_str, "%",
                       f"BOS risk: {bos_detail}",
                       bos_status(bos_level)))

    # 5. Sleep Efficiency
    eff = summary.get("mean_efficiency")
    eff_str = f"{eff:.0f}" if eff else "N/A"
    sleep_out = outputs.get("sleep", {})
    sleep_rux_p = sleep_out.get("ruxolitinib_comparison", {}).get("efficiency", {}).get("mann_whitney_p")
    eff_detail = f"Rux {_format_p_value(sleep_rux_p)}" if sleep_rux_p is not None else "Below 85% threshold"
    cards.append(_card("Sleep Efficiency", eff_str, "%", eff_detail,
                       "warning" if eff and eff < 85 else "good"))

    # 6. Autonomic Tone (UKF)
    ukf_final = dt.get("ukf", {}).get("state_means_final", {})
    at = ukf_final.get("Autonomic Tone")
    at_str = f"{at:.2f}" if at is not None else "N/A"
    at_p = drug.get("Autonomic Tone", {}).get("mann_whitney_p")
    at_dir = drug.get("Autonomic Tone", {}).get("direction", "")
    cards.append(_card("Autonomic Tone", at_str, "SD",
                       f"UKF state | {at_dir} | {_format_p_value(at_p)}" if at_p is not None else "UKF latent state",
                       "good" if at_dir == "improved" else "warning"))

    # 7. GVHD Score
    gvhd_bm = biomarkers.get("biomarkers", {}).get("gvhd_score", {})
    gvhd_val = gvhd_bm.get("latest_value")
    gvhd_str = f"{gvhd_val:.0f}" if gvhd_val is not None else "N/A"
    cards.append(_card("GVHD Score", gvhd_str, "/100",
                       f"7d avg: {gvhd_bm.get('latest_7d_avg', 0):.0f}" if gvhd_bm else "Composite biomarker",
                       "warning" if gvhd_val and gvhd_val > 50 else "good"))

    # 8. Anomaly Days
    ens = anomaly.get("ensemble", {})
    n_anom = ens.get("n_anomaly_days", 0)
    cards.append(_card("Anomaly Days", str(n_anom), f"/{summary.get('total_nights', '?')}",
                       f"Feb 9 detected ({n_anom} anomaly days, N=1)",
                       "warning" if n_anom > 5 else "good"))

    return "\n".join(cards)


# ===================================================================
# TAB 1: HERO ITS CHART (Interrupted Time Series with counterfactual)
# ===================================================================

def build_hero_its_chart(outputs: dict, readiness_df: pd.DataFrame) -> go.Figure:
    """Interrupted Time Series: actual vs counterfactual for temperature deviation."""
    print("  Building Hero ITS Chart...")

    causal_ts = outputs.get("causal_ts", {})
    streams = causal_ts.get("streams", {})
    temp_stream = streams.get("temperature_deviation", {})

    fig = go.Figure()

    if temp_stream and temp_stream.get("dates"):
        dates = temp_stream["dates"]
        actual = temp_stream["actual"]
        predicted = temp_stream["predicted"]
        pred_lower = temp_stream["pred_lower"]
        pred_upper = temp_stream["pred_upper"]
        intervention_idx = temp_stream.get("intervention_idx", 0)
        p_val = temp_stream.get("p_value", None)

        # Background split: subtle pre/post shading
        if intervention_idx > 0 and intervention_idx < len(dates):
            fig.add_vrect(
                x0=dates[0], x1=RUXOLITINIB_START,
                fillcolor="rgba(69,137,255,0.04)", line_width=0,
            )
            fig.add_vrect(
                x0=RUXOLITINIB_START, x1=dates[-1],
                fillcolor="rgba(245,158,11,0.04)", line_width=0,
            )

        # Pre-intervention actual
        fig.add_trace(go.Scatter(
            x=dates[:intervention_idx], y=actual[:intervention_idx],
            mode="lines+markers",
            line=dict(color=C_PRE, width=2.5),
            marker=dict(size=4, color=C_PRE),
            name="Actual (pre-ruxolitinib)",
            hovertemplate="<b>%{x|%b %d}</b><br>Temp dev: %{y:.3f} \u00b0C<extra></extra>",
        ))

        # Post-intervention actual
        fig.add_trace(go.Scatter(
            x=dates[intervention_idx:], y=actual[intervention_idx:],
            mode="lines+markers",
            line=dict(color=C_POST, width=3),
            marker=dict(size=6, color=C_POST),
            name="Actual (post-ruxolitinib)",
            hovertemplate="<b>%{x|%b %d}</b><br>Temp dev: %{y:.3f} \u00b0C<extra></extra>",
        ))

        # Counterfactual (post-intervention only)
        fig.add_trace(go.Scatter(
            x=dates[intervention_idx:], y=predicted[intervention_idx:],
            mode="lines",
            line=dict(color=C_COUNTERFACTUAL, width=2, dash="dash"),
            name="Counterfactual (predicted without drug)",
            hovertemplate="<b>%{x|%b %d}</b><br>Predicted: %{y:.3f} \u00b0C<extra></extra>",
        ))

        # CI band (post-intervention)
        fig.add_trace(go.Scatter(
            x=dates[intervention_idx:] + dates[intervention_idx:][::-1],
            y=pred_upper[intervention_idx:] + pred_lower[intervention_idx:][::-1],
            fill="toself",
            fillcolor="rgba(153,153,153,0.12)",
            line=dict(width=0),
            name="95% CI",
            hoverinfo="skip",
        ))

        # Intervention line — dramatic
        _add_vline(fig, x=RUXOLITINIB_START, line_dash="dash",
                   line_color=C_INTERVENTION, line_width=2.5,
                   annotation_text="<b>Ruxolitinib start</b>",
                   annotation_position="top",
                   annotation_font=dict(color="#FFFFFF", size=12))

        # Acute event line
        _add_vline(fig, x=ACUTE_EVENT, line_dash="dot",
                   line_color=C_ALERT, line_width=1.5,
                   annotation_text="Acute event",
                   annotation_position="bottom",
                   annotation_font=dict(color=C_ALERT, size=10))

        # P-value badge
        if p_val is not None:
            fig.add_annotation(
                x=dates[-1], y=max(actual[-3:]) if actual[-3:] else 0,
                text=f"<b>{_format_p_value(p_val)}</b>",
                showarrow=True, arrowhead=2, arrowcolor=C_POST,
                font=dict(size=14, color=C_POST),
                bgcolor="rgba(15,17,23,0.9)", bordercolor=C_POST, borderwidth=1,
                borderpad=6,
            )

    else:
        # Fallback: raw temperature from database
        if not readiness_df.empty:
            temp_sorted = readiness_df.sort_values("day_date")
            fig.add_trace(go.Bar(
                x=[str(d) for d in temp_sorted["day_date"]],
                y=temp_sorted["temperature_deviation"],
                marker=dict(
                    color=[C_POST if str(d) >= RUXOLITINIB_START else C_PRE
                           for d in temp_sorted["day_date"]],
                    opacity=0.7,
                ),
                name="Temperature deviation",
                hovertemplate="%{x}<br>Temp: %{y:.2f}\u00b0C<extra></extra>",
            ))
            fig.add_vline(x=RUXOLITINIB_START, line_dash="dash",
                          line_color=C_INTERVENTION, line_width=2)

    fig.update_layout(
        margin=dict(l=60, r=40, t=50, b=50),
        xaxis=dict(
            title="Date",
            tickformat="%d %b",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.15)", spikethickness=1,
            spikedash="dot",
        ),
        yaxis=dict(
            title="Temperature Deviation (\u00b0C)",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.15)", spikethickness=1,
            spikedash="dot",
            gridcolor=BORDER_SUBTLE,
        ),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
                    font=dict(size=11)),
        hovermode="x unified",
    )

    return fig


# ===================================================================
# TAB 1: DUAL FOREST PLOT (CausalImpact + UKF)
# ===================================================================

def build_forest_plot(outputs: dict) -> go.Figure:
    """Dual forest plot: CausalImpact effect sizes + UKF state shifts."""
    print("  Building Forest Plot...")

    causal = outputs.get("causal", {})
    dt = outputs.get("digital_twin", {})
    streams = causal.get("causal_impact", {}).get("streams", {})
    drug = dt.get("drug_response", {}).get("response_stats", {})

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["CausalImpact: Relative Effect (%)", "UKF: Latent State Shifts (SD)"],
        horizontal_spacing=0.12,
    )

    # Style subplot title annotations for dark theme
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=13, color=TEXT_PRIMARY))

    # --- Left: CausalImpact ---
    # Near-zero baselines (e.g. temperature_deviation ≈ 0.08°C) produce
    # extreme relative_effect_pct values (-287%) that blow out the x-axis.
    # Fix: cap display at ±DISPLAY_CAP%, use arrow markers for capped values,
    # show true values in hover text.  Remove CI whiskers — the JSON's
    # ci_lower/ci_upper are posterior intervals on the counterfactual, not
    # confidence intervals on the relative effect.
    DISPLAY_CAP = 80  # ±80% keeps axis readable for clinical audience

    if streams:
        sorted_streams = sorted(
            [(k, v) for k, v in streams.items() if isinstance(v, dict) and "p_value" in v],
            key=lambda x: x[1]["p_value"],
        )

        labels = []
        effects = []
        colors = []
        p_texts = []
        markers = []
        hover_effects = []
        hover_p_labels = []

        for key, s in sorted_streams:
            label = s.get("label", key).replace(" (", "\n(")
            if len(label) > 30:
                label = label[:28] + "..."
            labels.append(label)

            p = s.get("p_value", 1)
            sig_fdr = s.get("significant_fdr", False)
            if sig_fdr:
                colors.append(ACCENT_GREEN)
            elif p < 0.05:
                colors.append(C_POST)
            elif p < 0.1:
                colors.append(ACCENT_BLUE)
            else:
                colors.append(TEXT_TERTIARY)
            hover_p_labels.append(_format_p_value(p, decimals=4))

            rel_eff = s.get("relative_effect_pct")
            # When upstream sets relative_effect_pct to null (near-zero baseline),
            # compute it from available absolute values so the chart still shows data.
            if rel_eff is None or not np.isfinite(rel_eff):
                avg_eff = s.get("avg_effect")
                avg_cf = s.get("avg_counterfactual_post")
                if avg_eff is not None and avg_cf is not None and abs(avg_cf) > 1e-6:
                    rel_eff = avg_eff / avg_cf * 100
                else:
                    effects.append(0.0)
                    hover_effects.append("n/a (counterfactual near zero)")
                    p_texts.append(f"{_format_p_value(p)} | rel n/a")
                    markers.append("circle-open")
                    continue

            # Cap extreme relative effects for display
            capped = abs(rel_eff) > DISPLAY_CAP
            display_eff = max(-DISPLAY_CAP, min(DISPLAY_CAP, rel_eff))
            effects.append(display_eff)
            hover_effects.append(f"{rel_eff:+.1f}%")

            # Show true effect in label for capped metrics
            if capped:
                p_texts.append(f"{_format_p_value(p)} ({rel_eff:+.0f}%)")
            else:
                p_texts.append(_format_p_value(p))

            # Directional arrow for capped values, diamond for normal
            if capped and rel_eff < 0:
                markers.append("triangle-left")
            elif capped and rel_eff > 0:
                markers.append("triangle-right")
            else:
                markers.append("diamond")

        fig.add_trace(go.Scatter(
            y=labels, x=effects,
            mode="markers+text",
            marker=dict(
                color=colors,
                size=[14 if m.startswith("triangle") else 10 for m in markers],
                symbol=markers,
                line=dict(width=1, color=TEXT_PRIMARY),
            ),
            customdata=list(zip(hover_effects, hover_p_labels)),
            text=p_texts,
            textposition="middle right",
            textfont=dict(size=10, color=TEXT_SECONDARY),
            name="CausalImpact",
            showlegend=False,
            hovertemplate="%{y}<br>Relative effect: %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        ), row=1, col=1)

        # Vertical no-effect line - prominent
        fig.add_vline(x=0, line_dash="solid", line_color="rgba(255,255,255,0.4)",
                      line_width=1.5, row=1, col=1)

        fig.update_xaxes(title_text="Relative effect (%)", row=1, col=1,
                         range=[-DISPLAY_CAP - 15, DISPLAY_CAP + 40],
                         gridcolor=BORDER_SUBTLE)
    else:
        fig.add_annotation(
            text="No CausalImpact data", x=0.25, y=0.5,
            xref="paper", yref="paper", showarrow=False,
            font=dict(size=14, color=TEXT_TERTIARY),
        )

    # --- Right: UKF State Shifts ---
    if drug:
        # Sort by absolute shift size (largest effect first)
        sorted_drug = sorted(drug.items(), key=lambda x: abs(x[1].get("shift_sd", 0)), reverse=True)

        state_labels = []
        shifts = []
        shift_colors = []
        p_labels = []

        for state, stats in sorted_drug:
            state_labels.append(state)
            sd = stats.get("shift_sd", 0)
            shifts.append(sd)
            p = stats.get("mann_whitney_p")
            direction = stats.get("direction", "stable")
            if direction == "improved":
                shift_colors.append(ACCENT_GREEN)
            elif direction == "worsened":
                shift_colors.append(ACCENT_RED)
            else:
                shift_colors.append(TEXT_TERTIARY)
            if p is not None:
                sig = " *" if p < 0.05 else ""
                p_labels.append(f"{_format_p_value(p, decimals=4)}{sig}")
            else:
                p_labels.append("")

        fig.add_trace(go.Scatter(
            y=state_labels, x=shifts,
            mode="markers+text",
            marker=dict(
                color=shift_colors, size=12, symbol="diamond",
                line=dict(width=1, color=TEXT_PRIMARY),
            ),
            text=p_labels,
            textposition="middle right",
            textfont=dict(size=10, color=TEXT_SECONDARY),
            name="UKF Shifts",
            showlegend=False,
            hovertemplate="%{y}<br>Shift: %{x:.2f} SD<br>%{text}<extra></extra>",
        ), row=1, col=2)

        fig.add_vline(x=0, line_dash="solid", line_color="rgba(255,255,255,0.4)",
                      line_width=1.5, row=1, col=2)

        fig.update_xaxes(title_text="Shift (SD units)", row=1, col=2,
                         gridcolor=BORDER_SUBTLE)
    else:
        fig.add_annotation(
            text="No UKF drug response data", x=0.75, y=0.5,
            xref="paper", yref="paper", showarrow=False,
            font=dict(size=14, color=TEXT_TERTIARY),
        )

    fig.update_layout(
        margin=dict(l=160, r=60, t=100, b=50),
        height=max(350, 40 * max(len(streams), len(drug), 5) + 120),
    )

    return fig


# ===================================================================
# TAB 2: rSLDS DISEASE STATE TIMELINE
# ===================================================================

def build_disease_states(outputs: dict) -> go.Figure:
    """Stacked area chart of rSLDS state probabilities over time."""
    print("  Building Disease State Timeline...")

    gvhd = outputs.get("gvhd", {})
    rslds = gvhd.get("rslds", {})
    daily_states = rslds.get("daily_states", {})

    fig = go.Figure()

    if daily_states and daily_states.get("dates"):
        dates = daily_states["dates"]
        probs = np.array(daily_states["state_probabilities"])
        state_names = daily_states.get("state_names", C_STATE_NAMES)

        # Stacked area traces (bottom to top)
        for i in range(min(probs.shape[1], 4)):
            fig.add_trace(go.Scatter(
                x=dates, y=probs[:, i],
                mode="lines",
                line=dict(width=0.5, color=C_STATES[i % len(C_STATES)]),
                fillcolor=f"rgba({int(C_STATES[i % len(C_STATES)][1:3], 16)},{int(C_STATES[i % len(C_STATES)][3:5], 16)},{int(C_STATES[i % len(C_STATES)][5:7], 16)},0.5)",
                fill="tonexty" if i > 0 else "tozeroy",
                name=state_names[i] if i < len(state_names) else f"State {i}",
                stackgroup="states",
                hovertemplate=f"<b>%{{x|%b %d}}</b><br>{state_names[i] if i < len(state_names) else f'State {i}'}: %{{y:.2f}}<extra></extra>",
            ))

        # Event lines
        _add_vline(fig, x=RUXOLITINIB_START, line_dash="dash",
                   line_color=C_INTERVENTION, line_width=2,
                   annotation_text="Ruxolitinib", annotation_position="top",
                   annotation_font=dict(color=TEXT_PRIMARY, size=10))
        _add_vline(fig, x=ACUTE_EVENT, line_dash="dot",
                   line_color=C_ALERT, line_width=1.5,
                   annotation_text="Acute event", annotation_position="top",
                   annotation_font=dict(color=C_ALERT, size=10))
    else:
        # Fallback: composite GVHD scores
        composite = gvhd.get("composite", {})
        daily_scores = composite.get("daily_scores", {})
        if daily_scores:
            # daily_scores can be dict {date: score} or list [{date, score}]
            if isinstance(daily_scores, dict):
                dates = list(daily_scores.keys())
                values = list(daily_scores.values())
            else:
                dates = [s["date"] for s in daily_scores]
                values = [s["score"] for s in daily_scores]
            fig.add_trace(go.Scatter(
                x=dates, y=values, mode="lines+markers",
                line=dict(color=C_ALERT, width=2),
                marker=dict(size=4),
                name="Composite GVHD Score",
            ))

    fig.update_layout(
        margin=dict(l=60, r=40, t=50, b=50),
        yaxis=dict(range=[0, 1], title="State Probability", gridcolor=BORDER_SUBTLE),
        xaxis=dict(
            title="Date", tickformat="%d %b",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.15)", spikethickness=1, spikedash="dot",
        ),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    return fig


# ===================================================================
# TAB 2: ANOMALY DETECTION TIMELINE
# ===================================================================

def build_anomaly_timeline(outputs: dict) -> go.Figure:
    """Bar chart of daily ensemble anomaly scores."""
    print("  Building Anomaly Timeline...")

    anomaly = outputs.get("anomaly", {})
    ensemble = anomaly.get("ensemble", {})
    daily_scores = ensemble.get("daily_scores", [])
    threshold = ensemble.get("threshold_90pct", 0.686)

    fig = go.Figure()

    if daily_scores:
        dates = [s["date"] if isinstance(s.get("date"), str) else str(s.get("date", "")) for s in daily_scores]
        scores = [s.get("ensemble_score", 0) for s in daily_scores]

        colors = [C_ALERT if sc > threshold else "rgba(69,137,255,0.4)" for sc in scores]
        border_colors = [C_ALERT if sc > threshold else "rgba(69,137,255,0.6)" for sc in scores]

        fig.add_trace(go.Bar(
            x=dates, y=scores,
            marker=dict(
                color=colors, opacity=0.9,
                line=dict(color=border_colors, width=0.5),
            ),
            name="Anomaly Score",
            hovertemplate="<b>%{x|%b %d}</b><br>Score: %{y:.3f}<extra></extra>",
        ))

        # Threshold line
        fig.add_hline(y=threshold, line_dash="dash", line_color=C_ALERT,
                      line_width=1.5,
                      annotation_text=f"90th pctile: {threshold:.3f}",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color=C_ALERT))

        # Event lines
        fig.add_vline(x=RUXOLITINIB_START, line_dash="dash",
                      line_color=C_INTERVENTION, line_width=1.5)
        fig.add_vline(x=ACUTE_EVENT, line_dash="dot",
                      line_color=C_ALERT, line_width=1.5)

        # Annotate Feb 9
        if ACUTE_EVENT in dates:
            idx = dates.index(ACUTE_EVENT)
            fig.add_annotation(
                x=ACUTE_EVENT, y=scores[idx],
                text=f"Feb 9: {scores[idx]:.3f}",
                showarrow=True, arrowhead=2, arrowcolor=C_ALERT,
                font=dict(size=10, color=TEXT_PRIMARY),
                bgcolor="rgba(15,17,23,0.9)", bordercolor=C_ALERT,
            )

    fig.update_layout(
        margin=dict(l=60, r=40, t=50, b=50),
        xaxis=dict(
            title="Date", tickformat="%d %b",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.15)", spikethickness=1, spikedash="dot",
        ),
        yaxis=dict(title="Ensemble Anomaly Score", gridcolor=BORDER_SUBTLE),
        height=280,
    )

    return fig


# ===================================================================
# TAB 2: COMPOSITE BIOMARKER TRENDS (3x2 small multiples)
# ===================================================================

def build_biomarker_trends(outputs: dict) -> go.Figure:
    """3x2 grid of composite biomarker trends."""
    print("  Building Biomarker Trends...")

    biomarkers = outputs.get("biomarkers", {}).get("biomarkers", {})

    names_order = ["adsi", "gvhd_score", "recovery_index",
                   "cv_risk", "allostatic_load", "pharma_response"]
    display_names = ["ADSI", "GVHD Score", "Recovery Index",
                     "CV Risk", "Allostatic Load", "Pharma Response"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[dn for dn in display_names],
        vertical_spacing=0.1, horizontal_spacing=0.1,
    )

    rux_date = RUXOLITINIB_START

    for i, (key, name) in enumerate(zip(names_order, display_names)):
        row = i // 2 + 1
        col = i % 2 + 1

        bm = biomarkers.get(key, {})
        daily = bm.get("daily", [])

        if daily:
            dates = [d["date"] for d in daily]
            values = [d["value"] for d in daily]

            # Pre/post colors
            pre_dates = [d for d in dates if d < rux_date]
            pre_vals = [v for d, v in zip(dates, values) if d < rux_date]
            post_dates = [d for d in dates if d >= rux_date]
            post_vals = [v for d, v in zip(dates, values) if d >= rux_date]

            # Pre line — sparkline aesthetic
            fig.add_trace(go.Scatter(
                x=pre_dates, y=pre_vals,
                mode="lines", line=dict(color=C_PRE, width=1.5),
                showlegend=False,
                hovertemplate=f"<b>%{{x|%b %d}}</b><br>{name}: %{{y:.1f}}<extra></extra>",
            ), row=row, col=col)

            # Post line
            if post_dates:
                fig.add_trace(go.Scatter(
                    x=post_dates, y=post_vals,
                    mode="lines+markers", line=dict(color=C_POST, width=2),
                    marker=dict(size=3, color=C_POST),
                    showlegend=False,
                    hovertemplate=f"<b>%{{x|%b %d}}</b><br>{name}: %{{y:.1f}}<extra></extra>",
                ), row=row, col=col)

            # Rux line
            fig.add_vline(x=rux_date, line_dash="dash",
                          line_color="rgba(255,255,255,0.3)", line_width=1,
                          row=row, col=col)

            # Pre/post means
            rux_info = bm.get("ruxolitinib", {})
            pre_mean = rux_info.get("pre_mean")
            post_mean = rux_info.get("post_mean") if rux_info.get("post_n", 0) > 0 else None

            if pre_mean is not None:
                fig.add_hline(y=pre_mean, line_dash="dot",
                              line_color="rgba(69,137,255,0.4)", line_width=1,
                              row=row, col=col)
            if post_mean is not None:
                fig.add_hline(y=post_mean, line_dash="dot",
                              line_color="rgba(230,159,0,0.4)", line_width=1,
                              row=row, col=col)

    # Sparkline aesthetic: minimal axis clutter
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=12, color=TEXT_PRIMARY, family=FONT_FAMILY))
    fig.update_xaxes(tickformat="%d %b", tickfont=dict(size=9, color=TEXT_TERTIARY),
                     showgrid=True, gridcolor=BORDER_SUBTLE)
    fig.update_yaxes(tickfont=dict(size=9, color=TEXT_TERTIARY),
                     showgrid=True, gridcolor=BORDER_SUBTLE)
    fig.update_layout(
        margin=dict(l=50, r=30, t=120, b=40),
        height=650,
        showlegend=False,
    )

    return fig


# ===================================================================
# TAB 2: SpO2 / BOS RISK PANEL
# ===================================================================

def build_spo2_panel(outputs: dict, spo2_df: pd.DataFrame) -> go.Figure:
    """SpO2 trend with BOS risk assessment overlay."""
    print("  Building SpO2/BOS Panel...")

    spo2_data = outputs.get("spo2", {})
    trend = spo2_data.get("trend", {})
    bos = spo2_data.get("bos_risk", {})

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["SpO2 Trend", "BOS Risk Components"],
        column_widths=[0.65, 0.35],
    )

    # Left: SpO2 time series
    if not spo2_df.empty:
        dates = [str(d) for d in spo2_df["day_date"]]
        values = spo2_df["spo2_average"].values

        colors = [C_POST if d >= RUXOLITINIB_START else C_PRE for d in dates]

        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode="markers+lines",
            marker=dict(size=6, color=colors, opacity=0.85,
                        line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
            line=dict(color=C_PRE, width=1.5),
            name="SpO2",
            hovertemplate="<b>%{x|%b %d}</b><br>SpO2: %{y:.1f} %<extra></extra>",
        ), row=1, col=1)

        # 95% threshold
        fig.add_hline(y=95, line_dash="dash", line_color="rgba(231,76,60,0.5)",
                      line_width=1, row=1, col=1,
                      annotation_text="95% threshold",
                      annotation_font=dict(size=9, color="rgba(231,76,60,0.7)"))

        # Trend line
        slope = trend.get("slope_pct_per_day", 0)
        p_trend = trend.get("p_value", 1)
        if dates and len(dates) > 1:
            fig.add_annotation(
                x=dates[-1], y=min(values) - 0.2,
                text=f"Slope: {slope:.4f}%/day (p={p_trend:.3f})",
                showarrow=False,
                font=dict(size=10, color=TEXT_SECONDARY),
                row=1, col=1,
            )

        fig.add_vline(x=RUXOLITINIB_START, line_dash="dash",
                      line_color="rgba(255,255,255,0.3)", line_width=1, row=1, col=1)

    # Right: BOS risk components
    components = bos.get("component_scores", {})
    if components:
        comp_names = [BOS_COMPONENT_LABELS.get(name, name) for name in components.keys()]
        comp_values = list(components.values())

        fig.add_trace(go.Bar(
            y=comp_names, x=comp_values,
            orientation="h",
            marker=dict(
                color=[C_ALERT if v > 30 else (C_POST if v > 10 else C_PRE)
                       for v in comp_values],
                opacity=0.8,
            ),
            text=[f"{v:.0f}" for v in comp_values],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_PRIMARY),
            name="BOS Components",
            showlegend=False,
            hovertemplate="%{y}: %{x:.1f}/100<extra></extra>",
        ), row=1, col=2)

        # Overall score annotation
        overall = bos.get("composite_score", 0)
        level = bos.get("risk_level", "N/A")
        fig.add_annotation(
            x=max(comp_values) * 0.6 if comp_values else 50,
            y=len(comp_names) - 0.5,
            text=f"<b>Overall: {overall:.0f}/100 ({level})</b>",
            showarrow=False,
            font=dict(size=12, color=C_ALERT if str(level).upper() in {"MODERATE", "ELEVATED", "HIGH"} else TEXT_PRIMARY),
            row=1, col=2,
        )

    # Style subplot titles for dark theme
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=13, color=TEXT_PRIMARY))

    fig.update_xaxes(tickformat="%d %b", row=1, col=1, gridcolor=BORDER_SUBTLE,
                     showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor="rgba(255,255,255,0.15)", spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=BORDER_SUBTLE, row=1, col=1)
    fig.update_xaxes(gridcolor=BORDER_SUBTLE, row=1, col=2)
    fig.update_yaxes(gridcolor=BORDER_SUBTLE, row=1, col=2)

    fig.update_layout(
        margin=dict(l=60, r=40, t=100, b=50),
        height=320,
        showlegend=False,
    )

    return fig


# ===================================================================
# TAB 3: HEART RATE TERRAIN MAP (3D Surface) — preserved
# ===================================================================

def build_hr_terrain(hr_df: pd.DataFrame, sleep_df: pd.DataFrame) -> go.Figure:
    """3D surface: HR across nights. X=time, Y=date, Z=HR."""
    print("  Building HR Terrain Map...")

    nights = []
    for _, row in sleep_df.iterrows():
        if pd.isna(row.get("bedtime_start")) or pd.isna(row.get("bedtime_end")):
            continue
        try:
            bt_start = pd.to_datetime(row["bedtime_start"], utc=True)
            bt_end = pd.to_datetime(row["bedtime_end"], utc=True)
        except Exception:
            continue
        nights.append({
            "day": row["day"], "day_date": row["day_date"],
            "start": bt_start, "end": bt_end,
            "duration_h": (bt_end - bt_start).total_seconds() / 3600,
        })

    if not nights:
        return go.Figure()

    n_bins = 120
    bin_minutes = 5
    max_hours = n_bins * bin_minutes / 60

    dates = [n["day"] for n in nights]
    z_matrix = np.full((len(nights), n_bins), np.nan)

    for i, night in enumerate(nights):
        mask = (hr_df["ts"] >= night["start"]) & (hr_df["ts"] <= night["end"])
        night_hr = hr_df.loc[mask].copy()
        if night_hr.empty:
            continue
        night_hr["minutes_from_start"] = (
            (night_hr["ts"] - night["start"]).dt.total_seconds() / 60
        )
        night_hr["bin"] = (night_hr["minutes_from_start"] / bin_minutes).astype(int)
        night_hr = night_hr[night_hr["bin"] < n_bins]
        binned = night_hr.groupby("bin")["bpm"].mean()
        for b, val in binned.items():
            z_matrix[i, int(b)] = val

    # Interpolate gaps
    for i in range(len(nights)):
        row = z_matrix[i, :]
        valid = ~np.isnan(row)
        if valid.sum() >= 2:
            x_valid = np.where(valid)[0]
            y_valid = row[valid]
            x_all = np.arange(n_bins)
            interp_mask = (x_all >= x_valid[0]) & (x_all <= x_valid[-1])
            z_matrix[i, interp_mask] = np.interp(x_all[interp_mask], x_valid, y_valid)

    x_hours = np.arange(n_bins) * bin_minutes / 60

    # Refined clinical colorscale: deep blue -> teal -> amber -> red
    hr_colorscale = [
        [0.0, "#0d1b2a"],
        [0.2, "#1b3a5c"],
        [0.4, "#2a7f8e"],
        [0.6, "#48b97c"],
        [0.75, "#f0c840"],
        [0.9, "#e8712a"],
        [1.0, "#d42020"],
    ]

    fig = go.Figure(data=[go.Surface(
        z=z_matrix, x=x_hours, y=list(range(len(dates))),
        colorscale=hr_colorscale,
        colorbar=dict(title=dict(text="HR (bpm)", font=dict(size=13, color=TEXT_PRIMARY)),
                      thickness=15, len=0.6, tickfont=dict(color=TEXT_PRIMARY, size=11),
                      outlinecolor=BORDER_SUBTLE, outlinewidth=1),
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "Hours from bedtime: %{x:.1f}h<br>"
            "HR: %{z:.0f} bpm<extra></extra>"
        ),
        customdata=np.array([[d] * n_bins for d in dates]),
        lighting=dict(ambient=0.65, diffuse=0.5, specular=0.15, roughness=0.6, fresnel=0.1),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="rgba(255,255,255,0.3)", project_z=True),
            x=dict(show=True, color="rgba(255,255,255,0.04)", width=1),
            y=dict(show=True, color="rgba(255,255,255,0.04)", width=1),
        ),
        opacity=0.95,
    )])

    # Event markers
    for idx_val, label, color in [
        (RUXOLITINIB_START, "Ruxolitinib start", C_POST),
        (ACUTE_EVENT, "Acute decompensation", C_ALERT),
    ]:
        ev_idx = None
        for j, d in enumerate(dates):
            if d == idx_val:
                ev_idx = j
                break
        if ev_idx is not None:
            fig.add_trace(go.Scatter3d(
                x=[0, max_hours], y=[ev_idx, ev_idx], z=[110, 110],
                mode="lines+text",
                line=dict(color=color, width=6),
                text=[label, ""],
                textposition="top center",
                textfont=dict(size=11, color=color),
                showlegend=False, hoverinfo="text",
                hovertext=f"{label} {idx_val}",
            ))

    tick_vals = list(range(0, len(dates), 7))
    tick_texts = [dates[i] for i in tick_vals if i < len(dates)]

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title=dict(text="Hours from bedtime", font=dict(size=14, color=TEXT_PRIMARY)),
                       range=[0, 10],
                       backgroundcolor=BG_PRIMARY, gridcolor="rgba(255,255,255,0.06)",
                       color=TEXT_PRIMARY, tickfont=dict(size=11)),
            yaxis=dict(title=dict(text="Night", font=dict(size=14, color=TEXT_PRIMARY)),
                       tickvals=tick_vals, ticktext=tick_texts,
                       backgroundcolor=BG_PRIMARY, gridcolor="rgba(255,255,255,0.06)",
                       color=TEXT_PRIMARY, tickfont=dict(size=10)),
            zaxis=dict(title=dict(text="HR (bpm)", font=dict(size=14, color=TEXT_PRIMARY)),
                       range=[60, 110],
                       backgroundcolor=BG_PRIMARY, gridcolor="rgba(255,255,255,0.06)",
                       color=TEXT_PRIMARY, tickfont=dict(size=11)),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0), up=dict(x=0, y=0, z=1)),
            aspectratio=dict(x=1.5, y=2, z=0.7),
            bgcolor=BG_PRIMARY,
        ),
        height=700,
    )

    return fig


# ===================================================================
# TAB 3: PHASE SPACE (3D Scatter) — preserved
# ===================================================================

def build_phase_space(sleep_df: pd.DataFrame, spo2_df: pd.DataFrame) -> go.Figure:
    """3D scatter: (HR, HRV, SpO2) per night. Color: pre/post ruxolitinib."""
    print("  Building Phase Space...")

    merged = sleep_df.copy()
    spo2_map = dict(zip(spo2_df["day_date"], spo2_df["spo2_average"]))
    merged["spo2"] = merged["day_date"].map(spo2_map)
    plot_df = merged.dropna(subset=["average_heart_rate", "average_hrv"]).copy()

    if plot_df.empty:
        return go.Figure()

    rux_date = datetime.strptime(RUXOLITINIB_START, "%Y-%m-%d").date()
    event_date = datetime.strptime(ACUTE_EVENT, "%Y-%m-%d").date()
    plot_df["phase"] = plot_df["day_date"].apply(
        lambda d: "Post-Ruxolitinib" if d >= rux_date else "Pre-Ruxolitinib"
    )
    plot_df["is_event"] = plot_df["day_date"] == event_date

    eff = plot_df["efficiency"].fillna(70)
    plot_df["marker_size"] = 5 + (eff - eff.min()) / max((eff.max() - eff.min()), 1) * 15
    spo2_mean = plot_df["spo2"].mean()
    plot_df["spo2_plot"] = plot_df["spo2"].fillna(spo2_mean)

    fig = go.Figure()

    # Time-based coloring within each phase for temporal progression
    # Parse base hex to RGB for creating per-point RGBA colors
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    for phase_name, base_color in [("Pre-Ruxolitinib", C_PRE), ("Post-Ruxolitinib", C_POST)]:
        mask = plot_df["phase"] == phase_name
        sub = plot_df[mask].sort_values("day_date")
        if sub.empty:
            continue

        # Color intensity by temporal order within phase (via RGBA per point)
        r, g, b = _hex_to_rgb(base_color)
        n_pts = len(sub)
        point_colors = [
            f"rgba({r},{g},{b},{0.4 + 0.55 * (i / max(n_pts - 1, 1))})"
            for i in range(n_pts)
        ]

        fig.add_trace(go.Scatter3d(
            x=sub["average_heart_rate"], y=sub["average_hrv"], z=sub["spo2_plot"],
            mode="markers",
            marker=dict(
                size=sub["marker_size"].values, color=point_colors,
                line=dict(width=0.8, color="rgba(255,255,255,0.5)"),
            ),
            text=[
                f"<b>{row['day']}</b><br>HR: {row['average_heart_rate']:.0f} bpm<br>"
                f"HRV: {row['average_hrv']:.1f} ms<br>"
                f"SpO2: {row['spo2']:.1f}%<br>Eff: {row['efficiency']}%"
                if pd.notna(row.get("spo2")) else
                f"<b>{row['day']}</b><br>HR: {row['average_heart_rate']:.0f} bpm<br>"
                f"HRV: {row['average_hrv']:.1f} ms<br>SpO2: N/A<br>Eff: {row['efficiency']}%"
                for _, row in sub.iterrows()
            ],
            hoverinfo="text", name=phase_name,
        ))

    # Trajectory line connecting sequential points
    plot_df_sorted = plot_df.sort_values("day_date")
    fig.add_trace(go.Scatter3d(
        x=plot_df_sorted["average_heart_rate"],
        y=plot_df_sorted["average_hrv"],
        z=plot_df_sorted["spo2_plot"],
        mode="lines", line=dict(color="rgba(255,255,255,0.15)", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))

    # Acute event marker
    event_row = plot_df[plot_df["is_event"]]
    if not event_row.empty:
        er = event_row.iloc[0]
        fig.add_trace(go.Scatter3d(
            x=[er["average_heart_rate"]], y=[er["average_hrv"]], z=[er["spo2_plot"]],
            mode="markers+text",
            marker=dict(size=14, color=C_ALERT, symbol="diamond",
                        line=dict(width=2, color="white")),
            text=[f"{ACUTE_EVENT}"], textposition="top center",
            textfont=dict(size=11, color=C_ALERT),
            name="Acute Event", hoverinfo="text",
            hovertext=f"Acute decompensation {ACUTE_EVENT}",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title=dict(text="Mean HR (bpm)", font=dict(size=14, color=TEXT_PRIMARY)),
                       backgroundcolor=BG_PRIMARY,
                       gridcolor="rgba(255,255,255,0.06)", color=TEXT_PRIMARY,
                       tickfont=dict(size=11)),
            yaxis=dict(title=dict(text="Mean HRV (ms)", font=dict(size=14, color=TEXT_PRIMARY)),
                       backgroundcolor=BG_PRIMARY,
                       gridcolor="rgba(255,255,255,0.06)", color=TEXT_PRIMARY,
                       tickfont=dict(size=11)),
            zaxis=dict(title=dict(text="SpO2 (%)", font=dict(size=14, color=TEXT_PRIMARY)),
                       backgroundcolor=BG_PRIMARY,
                       gridcolor="rgba(255,255,255,0.06)", color=TEXT_PRIMARY,
                       tickfont=dict(size=11)),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectratio=dict(x=1, y=1, z=0.7),
            bgcolor=BG_PRIMARY,
        ),
        height=700,
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02,
                    bgcolor="rgba(26,29,39,0.8)", font=dict(size=12)),
    )

    return fig


# ===================================================================
# TAB 3: SLEEP ARCHITECTURE HEATMAP — preserved
# ===================================================================

def build_sleep_heatmap(epochs_df: pd.DataFrame, sleep_df: pd.DataFrame) -> go.Figure:
    """Heatmap of sleep phases across all nights."""
    print("  Building Sleep Architecture Heatmap...")

    if epochs_df.empty:
        return go.Figure()

    dates = sorted(epochs_df["day_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    max_epochs = epochs_df.groupby("day")["epoch_index"].max().max() + 1
    max_epochs = min(int(max_epochs), 140)

    z_matrix = np.full((len(dates), max_epochs), np.nan)
    for _, row in epochs_df.iterrows():
        date_idx = date_to_idx.get(row["day_date"])
        epoch_idx = int(row["epoch_index"])
        if date_idx is not None and epoch_idx < max_epochs:
            z_matrix[date_idx, epoch_idx] = row["phase"]

    # Refined sleep phase colorscale — high contrast on dark background
    colorscale = [
        [0.0, "#1A1D27"], [0.125, "#7C3AED"], [0.25, "#7C3AED"],   # Deep — vivid purple
        [0.25, "#3B82F6"], [0.5, "#3B82F6"],                        # Light — blue
        [0.5, "#10B981"], [0.75, "#10B981"],                        # REM — emerald
        [0.75, "#EF4444"], [1.0, "#EF4444"],                        # Awake — red
    ]

    date_labels = [str(d) for d in dates]
    x_hours = np.arange(max_epochs) * 5 / 60

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix, x=x_hours, y=date_labels,
        colorscale=colorscale, zmin=0.5, zmax=4.5,
        colorbar=dict(
            title=dict(text="Sleep Phase", font=dict(size=13, color=TEXT_PRIMARY)),
            tickvals=[1, 2, 3, 4], ticktext=["Deep", "Light", "REM", "Awake"],
            thickness=15, len=0.6, tickfont=dict(color=TEXT_PRIMARY, size=11),
            outlinecolor=BORDER_SUBTLE, outlinewidth=1,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>Time: %{x:.1f}h<br>Phase: %{customdata}<extra></extra>"
        ),
        customdata=np.vectorize(
            lambda x: PHASE_NAMES.get(int(x), "?") if not np.isnan(x) else ""
        )(z_matrix),
        xgap=0, ygap=1,
    ))

    # Event annotations
    for date_str, label, color in [
        (RUXOLITINIB_START, "Ruxolitinib start", C_POST),
        (ACUTE_EVENT, f"Acute event {ACUTE_EVENT}", C_ALERT),
    ]:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
        if target in date_to_idx:
            fig.add_annotation(
                x=-0.3, y=str(target), text=f"  {label}",
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor=color,
                font=dict(size=10, color=color), xanchor="right", ax=-60, ay=0,
            )

    fig.update_layout(
        margin=dict(l=100, r=40, t=50, b=60),
        xaxis=dict(title="Hours from bedtime", dtick=1, gridcolor=BORDER_SUBTLE,
                   tickfont=dict(size=11, color=TEXT_SECONDARY)),
        yaxis=dict(title="Night", autorange="reversed", dtick=7,
                   tickfont=dict(size=10, color=TEXT_SECONDARY)),
        height=700,
    )

    return fig


# ===================================================================
# TAB 3: CIRCADIAN RHYTHM RADAR — BUG FIXED
# ===================================================================

def build_circadian_radar(hr_df: pd.DataFrame, outputs: dict) -> go.Figure:
    """Polar plot of average HR per hour of night, pre vs post ruxolitinib."""
    print("  Building Circadian Rhythm Radar...")

    rux_date = datetime.strptime(RUXOLITINIB_START, "%Y-%m-%d").date()

    night_mask = (hr_df["hour"] >= 22) | (hr_df["hour"] < 8)
    night_hr = hr_df[night_mask].copy()

    if night_hr.empty:
        return go.Figure()

    def hour_to_night_index(h: int) -> int:
        return h - 22 if h >= 22 else h + 2

    night_hr["night_hour"] = night_hr["hour"].apply(hour_to_night_index)
    night_hr["is_post_rux"] = night_hr["date"] >= rux_date

    pre = night_hr[~night_hr["is_post_rux"]].groupby("night_hour")["bpm"].mean()
    post = night_hr[night_hr["is_post_rux"]].groupby("night_hour")["bpm"].mean()

    # BUG FIX: count readings, not sum of bpm values
    n_pre_readings = len(night_hr[~night_hr["is_post_rux"]])
    n_post_readings = len(night_hr[night_hr["is_post_rux"]])

    hour_labels = ["22:00", "23:00", "00:00", "01:00", "02:00",
                   "03:00", "04:00", "05:00", "06:00", "07:00"]

    fig = go.Figure()

    # Pre-ruxolitinib
    pre_r = [pre.get(i, np.nan) for i in range(10)]
    pre_r_closed = pre_r + [pre_r[0]]
    theta_labels_closed = hour_labels + [hour_labels[0]]

    fig.add_trace(go.Scatterpolar(
        r=pre_r_closed, theta=theta_labels_closed,
        mode="lines+markers",
        line=dict(color=C_PRE, width=2.5),
        marker=dict(size=6, color=C_PRE, line=dict(width=1, color="rgba(255,255,255,0.4)")),
        fill="toself", fillcolor="rgba(69,137,255,0.12)",
        name=f"Pre-Ruxolitinib (n={n_pre_readings:,} readings)",
        hovertemplate="<b>%{theta}</b><br>HR: %{r:.1f} bpm<extra>Pre-Rux</extra>",
    ))

    # Post-ruxolitinib
    if not post.empty:
        post_r = [post.get(i, np.nan) for i in range(10)]
        post_r_closed = post_r + [post_r[0]]

        fig.add_trace(go.Scatterpolar(
            r=post_r_closed, theta=theta_labels_closed,
            mode="lines+markers",
            line=dict(color=C_POST, width=2.5),
            marker=dict(size=6, color=C_POST, line=dict(width=1, color="rgba(255,255,255,0.4)")),
            fill="toself", fillcolor="rgba(230,159,0,0.12)",
            name=f"Post-Ruxolitinib (n={n_post_readings:,} readings)",
            hovertemplate="<b>%{theta}</b><br>HR: %{r:.1f} bpm<extra>Post-Rux</extra>",
        ))

    # SD bands for pre
    pre_std = night_hr[~night_hr["is_post_rux"]].groupby("night_hour")["bpm"].std()
    pre_upper = [(pre.get(i, 0) + pre_std.get(i, 0)) for i in range(10)]
    pre_lower = [(pre.get(i, 0) - pre_std.get(i, 0)) for i in range(10)]

    fig.add_trace(go.Scatterpolar(
        r=pre_upper + [pre_upper[0]], theta=theta_labels_closed,
        mode="lines", line=dict(color="rgba(69,137,255,0.3)", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=pre_lower + [pre_lower[0]], theta=theta_labels_closed,
        mode="lines", line=dict(color="rgba(69,137,255,0.3)", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # Add cosinor stats from HRV analysis if available
    hrv_out = outputs.get("hrv", {})
    cosinor = hrv_out.get("cosinor", {})
    if cosinor:
        mesor = cosinor.get("mesor_bpm")
        amplitude = cosinor.get("amplitude_bpm")
        acrophase = cosinor.get("acrophase_time")
        if mesor and amplitude:
            fig.add_annotation(
                text=f"Cosinor: MESOR={mesor:.1f} bpm, Amp={amplitude:.1f}, Acro={acrophase or 'N/A'}",
                xref="paper", yref="paper", x=0.5, y=-0.15,
                showarrow=False, font=dict(size=10, color=TEXT_SECONDARY),
            )

    all_vals = [v for v in pre_r + (list(post.values) if not post.empty else []) if not np.isnan(v)]
    r_min = min(all_vals) - 5 if all_vals else 60
    r_max = max(all_vals) + 5 if all_vals else 110

    fig.update_layout(
        margin=dict(l=60, r=60, t=50, b=60),
        polar=dict(
            radialaxis=dict(visible=True, range=[r_min, r_max],
                            ticksuffix=" bpm", tickfont=dict(size=10, color=TEXT_SECONDARY),
                            gridcolor="rgba(255,255,255,0.08)",
                            linecolor=BORDER_SUBTLE),
            angularaxis=dict(tickfont=dict(size=13, color=TEXT_PRIMARY, family=FONT_FAMILY),
                             direction="clockwise", rotation=90,
                             gridcolor="rgba(255,255,255,0.08)",
                             linecolor=BORDER_SUBTLE),
            bgcolor=BG_PRIMARY,
        ),
        height=550,
        legend=dict(yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
                    orientation="h", font=dict(size=11)),
    )

    return fig


# ===================================================================
# TAB 3: BIOSIGNAL TIMELINE (5-row subplots) — preserved with UKF overlay
# ===================================================================

def build_biosignal_timeline(
    sleep_df: pd.DataFrame, hrv_df: pd.DataFrame,
    spo2_df: pd.DataFrame, readiness_df: pd.DataFrame,
    outputs: dict,
) -> go.Figure:
    """5-row subplot timeline with March 16 annotations and UKF overlay."""
    print("  Building Biosignal Timeline...")

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=(
            "Nightly Average Heart Rate",
            "Nightly Average HRV (RMSSD)",
            "SpO2 Average",
            "Temperature Deviation",
            "Sleep Efficiency",
        ),
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],
    )

    all_dates = sorted(sleep_df["day_date"].unique())
    if not all_dates:
        return fig
    date_min = str(min(all_dates))
    date_max = str(max(all_dates))

    def add_annotations(fig: go.Figure, row: int) -> None:
        fig.add_vrect(x0=date_min, x1=RUXOLITINIB_START,
                      fillcolor="rgba(69,137,255,0.03)", line_width=0, row=row, col=1)
        fig.add_vrect(x0=RUXOLITINIB_START, x1=date_max,
                      fillcolor="rgba(230,159,0,0.03)", line_width=0, row=row, col=1)
        fig.add_vline(x=RUXOLITINIB_START, line_dash="dash",
                      line_color="rgba(255,255,255,0.5)", line_width=1.5, row=row, col=1)
        fig.add_vline(x=ACUTE_EVENT, line_dash="dot",
                      line_color=C_ALERT, line_width=1.5, row=row, col=1)

    # ---- Row 1: Heart Rate ----
    sleep_sorted = sleep_df.sort_values("day_date")
    dates_hr = [str(d) for d in sleep_sorted["day_date"]]
    hr_rolling = sleep_sorted["average_heart_rate"].rolling(7, min_periods=1, center=True).mean()

    fig.add_trace(go.Scatter(
        x=dates_hr, y=sleep_sorted["average_heart_rate"],
        mode="markers", marker=dict(size=4, color=C_PRE, opacity=0.5),
        name="Nightly HR", showlegend=False,
        hovertemplate="<b>%{x|%b %d}</b><br>HR: %{y:.0f} bpm<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_hr, y=hr_rolling,
        mode="lines", line=dict(color=C_PRE, width=2.5),
        name="7d avg", showlegend=False,
        hovertemplate="<b>%{x|%b %d}</b><br>7d avg: %{y:.0f} bpm<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_hr, y=sleep_sorted["lowest_heart_rate"],
        mode="markers", marker=dict(size=3, color=ACCENT_GREEN, opacity=0.4, symbol="diamond"),
        name="Lowest HR", showlegend=False,
        hovertemplate="<b>%{x|%b %d}</b><br>Lowest: %{y} bpm<extra></extra>",
    ), row=1, col=1)
    add_annotations(fig, 1)

    # ---- Row 2: HRV ----
    fig.add_trace(go.Scatter(
        x=dates_hr, y=sleep_sorted["average_hrv"],
        mode="markers+lines",
        marker=dict(size=4, color=ACCENT_PURPLE, opacity=0.6),
        line=dict(color=ACCENT_PURPLE, width=1.5),
        name="Nightly HRV", showlegend=False,
        hovertemplate="<b>%{x|%b %d}</b><br>HRV: %{y:.1f} ms<extra></extra>",
    ), row=2, col=1)
    hrv_rolling = sleep_sorted["average_hrv"].rolling(7, min_periods=1, center=True).mean()
    fig.add_trace(go.Scatter(
        x=dates_hr, y=hrv_rolling,
        mode="lines", line=dict(color=ACCENT_PURPLE, width=2.5),
        name="HRV 7d avg", showlegend=False,
    ), row=2, col=1)
    add_annotations(fig, 2)

    # ---- Row 3: SpO2 ----
    if not spo2_df.empty:
        fig.add_trace(go.Scatter(
            x=[str(d) for d in spo2_df["day_date"]], y=spo2_df["spo2_average"],
            mode="markers+lines",
            marker=dict(size=5, color=ACCENT_CYAN, opacity=0.7),
            line=dict(color=ACCENT_CYAN, width=1.5),
            name="SpO2", showlegend=False,
            hovertemplate="<b>%{x|%b %d}</b><br>SpO2: %{y:.1f} %<extra></extra>",
        ), row=3, col=1)
        fig.add_hline(y=95, line_dash="dash", line_color="rgba(231,76,60,0.5)",
                      row=3, col=1, annotation_text="95%",
                      annotation_position="bottom right",
                      annotation_font=dict(size=9, color="rgba(231,76,60,0.7)"))
    add_annotations(fig, 3)

    # ---- Row 4: Temperature Deviation ----
    if not readiness_df.empty:
        temp_sorted = readiness_df.sort_values("day_date")
        fig.add_trace(go.Bar(
            x=[str(d) for d in temp_sorted["day_date"]],
            y=temp_sorted["temperature_deviation"],
            marker=dict(
                color=temp_sorted["temperature_deviation"].apply(
                    lambda v: "#EF4444" if v > 0.5 else ("#3B82F6" if v < -0.5 else "#6B7280")
                ),
                opacity=0.7,
            ),
            name="Temp deviation", showlegend=False,
            hovertemplate="<b>%{x|%b %d}</b><br>Temp: %{y:.2f} \u00b0C<extra></extra>",
        ), row=4, col=1)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=0.5, row=4, col=1)
    add_annotations(fig, 4)

    # ---- Row 5: Sleep Efficiency ----
    fig.add_trace(go.Scatter(
        x=dates_hr, y=sleep_sorted["efficiency"],
        mode="markers+lines",
        marker=dict(size=4, color="#10B981", opacity=0.6),
        line=dict(color="#10B981", width=1.5),
        name="Efficiency", showlegend=False,
        hovertemplate="<b>%{x|%b %d}</b><br>Eff: %{y:.0f} %<extra></extra>",
    ), row=5, col=1)
    eff_rolling = sleep_sorted["efficiency"].rolling(7, min_periods=1, center=True).mean()
    fig.add_trace(go.Scatter(
        x=dates_hr, y=eff_rolling,
        mode="lines", line=dict(color="#10B981", width=2.5),
        name="Eff 7d avg", showlegend=False,
    ), row=5, col=1)
    fig.add_hline(y=85, line_dash="dash", line_color="rgba(16,185,129,0.5)",
                  row=5, col=1, annotation_text="85%",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="rgba(16,185,129,0.7)"))
    add_annotations(fig, 5)

    # Y-axis labels — sparkline aesthetic: minimal text
    for row_i, label in enumerate(["bpm", "ms", "%", "\u00b0C", "%"], 1):
        fig.update_yaxes(title_text=label, row=row_i, col=1, color=TEXT_PRIMARY,
                         gridcolor="rgba(255,255,255,0.05)",
                         tickfont=dict(size=9, color=TEXT_TERTIARY))
    # Add crosshair spikes to all x-axes
    for row_i in range(1, 6):
        fig.update_xaxes(
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.12)", spikethickness=1, spikedash="dot",
            tickformat="%d %b", tickfont=dict(size=9, color=TEXT_TERTIARY),
            row=row_i, col=1,
        )
    fig.update_xaxes(title_text="Date", row=5, col=1, color=TEXT_PRIMARY)

    # Legend annotation
    fig.add_annotation(
        text=(f'<span style="color:{C_PRE}">--- Ruxolitinib start ({RUXOLITINIB_START})</span>'
              f'  <span style="color:{C_ALERT}">... Acute event ({ACUTE_EVENT})</span>'),
        xref="paper", yref="paper", x=0.5, y=1.02,
        showarrow=False, font=dict(size=11, color=TEXT_PRIMARY),
    )

    # Style subplot titles — sparkline aesthetic
    for ann in fig.layout.annotations:
        if ann.text and ann.text != fig.layout.annotations[-1].text:  # skip legend annotation
            ann.update(font=dict(size=11, color=TEXT_SECONDARY, family=FONT_FAMILY))

    fig.update_layout(
        margin=dict(l=60, r=40, t=120, b=40),
        height=1000,
        showlegend=False,
    )

    return fig


# ===================================================================
# SUMMARY STATISTICS
# ===================================================================

def compute_summary(
    sleep_df: pd.DataFrame, hrv_df: pd.DataFrame, hr_df: pd.DataFrame,
    spo2_df: pd.DataFrame, readiness_df: pd.DataFrame, epochs_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute summary statistics for the dashboard."""
    rux_date = datetime.strptime(RUXOLITINIB_START, "%Y-%m-%d").date()

    observed_days: set = set()
    for frame, column in (
        (sleep_df, "day_date"),
        (spo2_df, "day_date"),
        (readiness_df, "day_date"),
        (epochs_df, "day_date"),
    ):
        if column in frame.columns:
            observed_days.update(d for d in frame[column].dropna().tolist())
    if "date" in hrv_df.columns:
        observed_days.update(d for d in hrv_df["date"].dropna().tolist())
    elif "ts_local" in hrv_df.columns:
        observed_days.update(d.date() for d in hrv_df["ts_local"].dropna().tolist())
    elif "ts" in hrv_df.columns:
        observed_days.update(d.date() for d in hrv_df["ts"].dropna().tolist())

    if "date" in hr_df.columns:
        observed_days.update(d for d in hr_df["date"].dropna().tolist())
    elif "ts_local" in hr_df.columns:
        observed_days.update(d.date() for d in hr_df["ts_local"].dropna().tolist())
    elif "ts" in hr_df.columns:
        observed_days.update(d.date() for d in hr_df["ts"].dropna().tolist())

    observed_days = {d for d in observed_days if d is not None and d >= DATA_START}
    observed_start = min(observed_days) if observed_days else sleep_df["day_date"].min()
    if observed_start and observed_start < DATA_START:
        observed_start = DATA_START
    observed_end = max(observed_days) if observed_days else sleep_df["day_date"].max()

    pre_sleep = sleep_df[sleep_df["day_date"] < rux_date]
    post_sleep = sleep_df[sleep_df["day_date"] >= rux_date]

    total_epochs = len(epochs_df)
    phase_pcts = {}
    if total_epochs > 0:
        for phase_val, phase_name in PHASE_NAMES.items():
            count = (epochs_df["phase"] == phase_val).sum()
            phase_pcts[phase_name] = count / total_epochs * 100

    n_unique_days = 0
    pre_days = 0
    post_days = 0
    if observed_start is not None and observed_end is not None:
        # Keep window-day counts consistent with other reports (inclusive date span),
        # even if one or more individual days have missing measurements.
        n_unique_days = (observed_end - observed_start).days + 1

        pre_end = min(observed_end, rux_date - timedelta(days=1))
        if observed_start <= pre_end:
            pre_days = (pre_end - observed_start).days + 1

        post_start = max(observed_start, rux_date)
        if post_start <= observed_end:
            post_days = (observed_end - post_start).days + 1

    return {
        "data_start": str(observed_start),
        "data_end": str(observed_end),
        "date_range": f"{observed_start} to {observed_end}",
        "n_unique_days": n_unique_days,
        "total_nights": len(sleep_df),
        "total_hr_readings": len(hr_df),
        "total_hrv_readings": len(hrv_df),
        "total_epochs": total_epochs,
        "mean_hr": sleep_df["average_heart_rate"].mean(),
        "mean_hrv": sleep_df["average_hrv"].mean(),
        "mean_efficiency": sleep_df["efficiency"].mean(),
        "mean_spo2": spo2_df["spo2_average"].mean() if not spo2_df.empty else None,
        "mean_temp_dev": readiness_df["temperature_deviation"].mean() if not readiness_df.empty else None,
        "pre_days": pre_days,
        "post_days": post_days,
        "pre_sleep_periods": len(pre_sleep),
        "post_sleep_periods": len(post_sleep),
        "pre_mean_hr": pre_sleep["average_heart_rate"].mean() if not pre_sleep.empty else None,
        "post_mean_hr": post_sleep["average_heart_rate"].mean() if not post_sleep.empty else None,
        "pre_mean_hrv": pre_sleep["average_hrv"].mean() if not pre_sleep.empty else None,
        "post_mean_hrv": post_sleep["average_hrv"].mean() if not post_sleep.empty else None,
        "phase_pcts": phase_pcts,
        "mean_total_sleep_h": sleep_df["total_sleep_duration_h"].mean(),
        "mean_rem_h": sleep_df["rem_sleep_duration_h"].mean(),
        "mean_deep_h": sleep_df["deep_sleep_duration_h"].mean(),
    }


# ===================================================================
# HTML ASSEMBLY — DARK THEME WITH TABS AND LAZY LOADING
# ===================================================================

def build_html(
    figures: dict[str, go.Figure],
    kpi_html: str,
    narrative: str,
    summary: dict[str, Any],
    outputs: dict,
) -> str:
    """Assemble all figures into a unified dark-theme dashboard with 3 tabs."""
    print("  Assembling HTML...")

    # Convert figures to JSON for lazy loading
    chart_json: dict[str, str] = {}
    for key, fig in figures.items():
        try:
            chart_json[key] = pio.to_json(fig)
        except Exception as e:
            print(f"    WARNING: Could not serialize {key}: {e}")
            chart_json[key] = "{}"

    charts_js = json.dumps(chart_json, ensure_ascii=False)

    # Data completeness
    n_json = sum(1 for k, v in outputs.items() if v)
    total_json = len(outputs)

    # Format helpers
    def fmt(val, suffix="", decimals=1):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{decimals}f}{suffix}"
        return f"{val}{suffix}"

    phase_html = ""
    for name, pct in summary.get("phase_pcts", {}).items():
        color = {"Deep": "#6366F1", "Light": "#3B82F6", "REM": "#10B981", "Awake": "#EF4444"}.get(name, "#6B7280")
        phase_html += f'<span style="color:{color};font-weight:600">{name}: {pct:.1f}%</span> '

    # --- Extract key stats for hero ---
    causal = outputs.get("causal", {})
    streams_data = causal.get("causal_impact", {}).get("streams", {})
    temp_ci = streams_data.get("temperature_deviation", {})
    temp_p = temp_ci.get("p_value")
    temp_q = temp_ci.get("q_value_bh")
    fdr_sig_count = sum(1 for s in streams_data.values() if isinstance(s, dict) and s.get("significant_fdr", False))
    total_streams = len(streams_data)

    dt_data = outputs.get("digital_twin", {})
    drug_resp = dt_data.get("drug_response", {}).get("response_stats", {})
    n_modules = sum(1 for k, v in outputs.items() if v and k != "causal_ts")
    total_readings = summary['total_hr_readings'] + summary['total_hrv_readings']

    # p-value display
    p_display = _format_p_value(temp_p)
    q_display = f"{temp_q:.3f}" if temp_q is not None else "N/A"

    # --- Build body content ---
    body_parts: list[str] = []

    # ===================== HERO SECTION =====================
    body_parts.append(f"""
    <div class="dash-hero odt-reveal">
      <div class="dash-hero-glow"></div>
      <div class="dash-hero-glow dash-hero-glow-2"></div>
      <div class="dash-hero-content">
        <div class="dash-hero-badge">
          <span class="dash-hero-pulse"></span>
          POST-HSCT BIOMETRIC MONITORING
        </div>
        <h1 class="dash-hero-title">
          Consumer wearable suggests a physiological shift after JAK inhibitor start<br>
          <span class="dash-hero-highlight">exploratory N=1 monitoring in a confounded window</span>
        </h1>
        <div class="dash-hero-stats">
          <div class="dash-hero-stat-card dash-hero-stat-primary">
            <div class="dash-hero-stat-value">{p_display}</div>
            <div class="dash-hero-stat-label">Lowest raw<br>p-value</div>
          </div>
          <div class="dash-hero-stat-card">
            <div class="dash-hero-stat-value">{q_display}</div>
            <div class="dash-hero-stat-label">FDR-adjusted<br>q-value</div>
          </div>
          <div class="dash-hero-stat-card">
            <div class="dash-hero-stat-value">{fdr_sig_count}/{total_streams}</div>
            <div class="dash-hero-stat-label">Streams surviving<br>FDR correction</div>
          </div>
          <div class="dash-hero-stat-card">
            <div class="dash-hero-stat-value">{total_readings:,}</div>
            <div class="dash-hero-stat-label">Biometric readings<br>analyzed</div>
          </div>
        </div>
        <div class="dash-hero-meta">
          Oura Ring Gen 4 &middot; {summary['n_unique_days']} days &middot;
          {n_modules}/{sum(1 for k in outputs if k != "causal_ts")} analysis modules &middot;
          Ruxolitinib 10 mg BID from {RUXOLITINIB_START}
        </div>
      </div>
    </div>""")

    # ===================== COMPARISON STRIP =====================
    body_parts.append(f"""
    <div class="dash-comparison odt-reveal">
      <div class="dash-comparison-card">
        <div class="dash-comparison-icon" style="background:rgba(59,130,246,0.15);color:{ACCENT_BLUE}">PRE</div>
        <div class="dash-comparison-body">
          <div class="dash-comparison-title">Pre-Ruxolitinib</div>
          <div class="dash-comparison-detail">
            {summary['pre_days']} days &middot; {summary['pre_sleep_periods']} sleep periods
          </div>
          <div class="dash-comparison-metrics">
            <span>HR <strong>{fmt(summary['pre_mean_hr'])}</strong> bpm</span>
            <span>HRV <strong>{fmt(summary['pre_mean_hrv'])}</strong> ms</span>
          </div>
        </div>
      </div>
      <div class="dash-comparison-divider">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M5 12h14m-4-4l4 4-4 4" stroke="{TEXT_TERTIARY}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
      <div class="dash-comparison-card">
        <div class="dash-comparison-icon" style="background:rgba(245,158,11,0.15);color:{ACCENT_AMBER}">POST</div>
        <div class="dash-comparison-body">
          <div class="dash-comparison-title">Post-Ruxolitinib</div>
          <div class="dash-comparison-detail">
            {summary['post_days']} days &middot; {summary['post_sleep_periods']} sleep periods
          </div>
          <div class="dash-comparison-metrics">
            <span>HR <strong>{fmt(summary['post_mean_hr'])}</strong> bpm</span>
            <span>HRV <strong>{fmt(summary['post_mean_hrv'])}</strong> ms</span>
          </div>
        </div>
      </div>
    </div>""")

    # ===================== NARRATIVE =====================
    body_parts.append(f"""
    <div class="dash-narrative odt-reveal">
      <div class="dash-narrative-icon">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M10 2L2 7l8 5 8-5-8-5zM2 13l8 5 8-5" stroke="{ACCENT_BLUE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
      <div class="dash-narrative-text">
        <strong>Key Finding</strong><br>{narrative}
      </div>
    </div>""")

    # ===================== KPI CARDS =====================
    body_parts.append(f"""
    <div class="dash-kpi-grid odt-reveal">
      {kpi_html}
    </div>""")

    # ===================== TAB NAVIGATION =====================
    body_parts.append(f"""
    <div class="dash-tab-wrapper">
      <div class="dash-tab-nav">
        <button class="dash-tab-btn active" data-tab="overview">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 012.5 1h3A1.5 1.5 0 017 2.5v3A1.5 1.5 0 015.5 7h-3A1.5 1.5 0 011 5.5v-3zm8 0A1.5 1.5 0 0110.5 1h3A1.5 1.5 0 0115 2.5v3A1.5 1.5 0 0113.5 7h-3A1.5 1.5 0 019 5.5v-3zm-8 8A1.5 1.5 0 012.5 9h3A1.5 1.5 0 017 10.5v3A1.5 1.5 0 015.5 15h-3A1.5 1.5 0 011 13.5v-3zm8 0A1.5 1.5 0 0110.5 9h3a1.5 1.5 0 011.5 1.5v3a1.5 1.5 0 01-1.5 1.5h-3A1.5 1.5 0 019 13.5v-3z"/></svg>
          Overview
        </button>
        <button class="dash-tab-btn" data-tab="disease">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a1 1 0 011 1v5.268l4.562-2.634a1 1 0 111 1.732L10 9.134l2.562 4.768a1 1 0 01-1.732 1L8 10.134l-2.83 4.768a1 1 0 01-1.732-1L6 9.134 1.438 6.366a1 1 0 111-1.732L7 7.268V2a1 1 0 011-1z"/></svg>
          Disease Monitoring
        </button>
        <button class="dash-tab-btn" data-tab="biosignals">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M0 8a.5.5 0 01.5-.5h2.022A4.5 4.5 0 018 2.5a4.5 4.5 0 015.478 5H15.5a.5.5 0 010 1h-2.022A4.5 4.5 0 018 13.5a4.5 4.5 0 01-5.478-5H.5A.5.5 0 010 8z"/></svg>
          Advanced Biosignals
        </button>
      </div>
    </div>""")

    # ===================== TAB 1: OVERVIEW =====================
    body_parts.append(f"""
    <section id="tab-overview" class="dash-tab-section active">
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Interrupted Time Series: Temperature Response</h3>
          <span class="odt-badge odt-badge-green">CausalImpact BSTS</span>
        </div>
        <div id="chart-hero_its" class="chart-box" data-chart="hero_its">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Treatment Effect Sizes</h3>
          <span class="odt-badge odt-badge-blue">CausalImpact + UKF</span>
        </div>
        <div id="chart-forest_plot" class="chart-box" data-chart="forest_plot">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
    </section>""")

    # ===================== TAB 2: DISEASE MONITORING =====================
    body_parts.append(f"""
    <section id="tab-disease" class="dash-tab-section">
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Disease State Trajectory</h3>
          <span class="odt-badge odt-badge-amber">rSLDS 4-State Model</span>
        </div>
        <div id="chart-disease_states" class="chart-box" data-chart="disease_states">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Anomaly Detection</h3>
          <span class="odt-badge odt-badge-red">5-Algorithm Ensemble</span>
        </div>
        <div id="chart-anomaly_timeline" class="chart-box" data-chart="anomaly_timeline">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Composite Biomarker Trends</h3>
          <span class="odt-badge odt-badge-blue">6 Biomarkers</span>
        </div>
        <div id="chart-biomarker_trends" class="chart-box" data-chart="biomarker_trends">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>SpO2 & BOS Risk</h3>
          <span class="odt-badge odt-badge-amber">Pulmonary Screening</span>
        </div>
        <div id="chart-spo2_panel" class="chart-box" data-chart="spo2_panel">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
    </section>""")

    # ===================== TAB 3: ADVANCED BIOSIGNALS =====================
    body_parts.append(f"""
    <section id="tab-biosignals" class="dash-tab-section">
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Combined Biosignal Timeline</h3>
          <span class="odt-badge odt-badge-blue">5-Channel View</span>
        </div>
        <div id="chart-biosignal_timeline" class="chart-box" data-chart="biosignal_timeline">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Heart Rate Terrain Map</h3>
          <span class="odt-badge odt-badge-green">3D Surface</span>
        </div>
        <div id="chart-hr_terrain" class="chart-box" data-chart="hr_terrain">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Phase Space Reconstruction</h3>
          <span class="odt-badge odt-badge-blue">Nonlinear Dynamics</span>
        </div>
        <div id="chart-phase_space" class="chart-box" data-chart="phase_space">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Sleep Architecture Heatmap</h3>
          <span class="odt-badge odt-badge-blue">Epoch-Level</span>
        </div>
        <div id="chart-sleep_heatmap" class="chart-box" data-chart="sleep_heatmap">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
      <div class="dash-chart-container">
        <div class="dash-chart-header">
          <h3>Circadian Rhythm Analysis</h3>
          <span class="odt-badge odt-badge-amber">Cosinor Model</span>
        </div>
        <div id="chart-circadian_radar" class="chart-box" data-chart="circadian_radar">
          <div class="dash-chart-skeleton"><div class="odt-skeleton" style="height:100%"></div></div>
        </div>
      </div>
    </section>""")

    # ===================== METHODOLOGY (COLLAPSIBLE) =====================
    body_parts.append(f"""
    <details class="dash-methodology odt-reveal">
      <summary class="dash-methodology-toggle">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 1v14M1 8h14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        Methodology & Data Provenance
      </summary>
      <div class="dash-methodology-content">
        <div class="dash-methodology-grid">
          <div class="dash-methodology-item">
            <div class="dash-methodology-label">Data Source</div>
            <div>Oura Ring Gen 4 (continuous PPG, accelerometry, thermometry)</div>
          </div>
          <div class="dash-methodology-item">
            <div class="dash-methodology-label">Volume</div>
            <div>{summary['total_hr_readings']:,} HR readings &middot; {summary['total_hrv_readings']:,} HRV samples &middot; {summary['total_epochs']:,} sleep epochs</div>
          </div>
          <div class="dash-methodology-item">
            <div class="dash-methodology-label">Analysis Pipeline</div>
            <div>CausalImpact (BSTS) &middot; Unscented Kalman Filter (5 latent states) &middot; rSLDS (4 disease states) &middot; 5-algorithm anomaly ensemble &middot; 6 composite biomarkers &middot; Chronos-2 / ARIMA forecasting</div>
          </div>
          <div class="dash-methodology-item">
            <div class="dash-methodology-label">Sleep Architecture</div>
            <div>{phase_html}</div>
          </div>
        </div>
        <div class="dash-methodology-disclaimer">
          For research and clinical discussion only. Not a diagnostic tool. All p-values are exploratory.
        </div>
      </div>
    </details>""")

    body_content = "\n".join(body_parts)

    # --- Extra CSS for dashboard-specific components ---
    extra_css = f"""
/* Hide standard header — custom hero replaces it */
.odt-header {{ display: none; }}

/* ============================================================
   HERO SECTION
   ============================================================ */
.dash-hero {{
  position: relative;
  padding: 56px 40px 46px;
  margin: -20px -40px 28px;
  overflow: hidden;
  background: linear-gradient(135deg, #0d1018 0%, #141a2e 50%, #0d1018 100%);
  border-bottom: 1px solid rgba(59,130,246,0.08);
}}
.dash-hero-glow {{
  position: absolute;
  width: 600px; height: 600px;
  border-radius: 50%;
  filter: blur(120px);
  opacity: 0.05;
  background: radial-gradient(circle, {ACCENT_BLUE} 0%, transparent 70%);
  top: -200px; left: -100px;
  pointer-events: none;
  animation: pulseGlow 15s ease-in-out infinite;
}}
.dash-hero-glow-2 {{
  background: radial-gradient(circle, {ACCENT_PURPLE} 0%, transparent 70%);
  top: -100px; right: -200px; left: auto;
  opacity: 0.04;
  animation-delay: 7s;
}}
.dash-hero-content {{
  position: relative; z-index: 1;
  max-width: 1100px; margin: 0 auto;
  text-align: center;
}}
.dash-hero-badge {{
  display: inline-flex;
  align-items: center; gap: 8px;
  padding: 6px 16px;
  border-radius: 20px;
  background: rgba(59,130,246,0.1);
  border: 1px solid rgba(59,130,246,0.25);
  font-size: 0.6875rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: {ACCENT_BLUE};
  text-transform: uppercase;
  margin-bottom: 20px;
}}
.dash-hero-pulse {{
  width: 8px; height: 8px;
  border-radius: 50%;
  background: {ACCENT_GREEN};
  display: inline-block;
  animation: pulseGlow 2s ease-in-out infinite;
  box-shadow: 0 0 8px rgba(16,185,129,0.6);
}}
.dash-hero-title {{
  font-size: clamp(2.2rem, 4vw, 3.2rem);
  font-weight: 800;
  line-height: 1.05;
  color: {TEXT_PRIMARY};
  margin-bottom: 28px;
  letter-spacing: -0.03em;
  max-width: 980px;
  margin-left: auto;
  margin-right: auto;
}}
.dash-hero-highlight {{
  background: linear-gradient(135deg, {ACCENT_BLUE} 0%, {ACCENT_CYAN} 50%, {ACCENT_PURPLE} 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.dash-hero-stats {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  max-width: 900px;
  margin: 0 auto 24px;
}}
.dash-hero-stat-card {{
  background: var(--bg-surface);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 20px 16px;
  transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}}
.dash-hero-stat-card:hover {{
  background: var(--bg-elevated);
  border-color: rgba(255,255,255,0.1);
  transform: translateY(-1px);
  box-shadow: 0 6px 24px rgba(0,0,0,0.25);
}}
.dash-hero-stat-primary {{
  background: rgba(59,130,246,0.08);
  border-color: rgba(59,130,246,0.2);
}}
.dash-hero-stat-primary:hover {{
  border-color: rgba(59,130,246,0.4);
  box-shadow: 0 8px 32px rgba(59,130,246,0.15);
}}
.dash-hero-stat-value {{
  font-size: 1.9rem;
  font-weight: 800;
  color: {TEXT_PRIMARY};
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}}
.dash-hero-stat-primary .dash-hero-stat-value {{
  color: {ACCENT_BLUE};
}}
.dash-hero-stat-label {{
  font-size: 0.75rem;
  color: {TEXT_SECONDARY};
  margin-top: 6px;
  line-height: 1.4;
  letter-spacing: 0.01em;
}}
.dash-hero-meta {{
  font-size: 0.8125rem;
  color: {TEXT_TERTIARY};
  letter-spacing: 0.02em;
}}

/* ============================================================
   COMPARISON STRIP (Pre/Post)
   ============================================================ */
.dash-comparison {{
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
  padding: 0 8px;
}}
.dash-comparison-card {{
  flex: 1;
  display: flex;
  align-items: center;
  gap: 14px;
  background: var(--bg-surface);
  border: 1px solid {BORDER_SUBTLE};
  border-radius: 12px;
  padding: 18px 22px;
  transition: all 0.25s;
}}
.dash-comparison-card:hover {{
  background: var(--bg-elevated);
  border-color: {BORDER_DEFAULT};
}}
.dash-comparison-icon {{
  width: 44px; height: 44px;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 0.6875rem;
  letter-spacing: 0.08em;
  flex-shrink: 0;
}}
.dash-comparison-body {{ flex: 1; min-width: 0; }}
.dash-comparison-title {{
  font-size: 0.9375rem; font-weight: 700; color: {TEXT_PRIMARY};
  margin-bottom: 2px;
}}
.dash-comparison-detail {{
  font-size: 0.75rem; color: {TEXT_TERTIARY};
  margin-bottom: 6px;
}}
.dash-comparison-metrics {{
  display: flex; gap: 16px; font-size: 0.8125rem; color: {TEXT_SECONDARY};
  flex-wrap: wrap;
}}
.dash-comparison-metrics strong {{
  color: {TEXT_PRIMARY}; font-weight: 700;
}}
.dash-comparison-divider {{
  flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  width: 40px; height: 40px;
  background: var(--bg-elevated);
  border-radius: 50%;
  border: 1px solid {BORDER_SUBTLE};
}}

/* ============================================================
   NARRATIVE
   ============================================================ */
.dash-narrative {{
  display: flex; gap: 14px;
  align-items: flex-start;
  padding: 20px 22px;
  background: rgba(59,130,246,0.04);
  border: 1px solid rgba(59,130,246,0.12);
  border-radius: 12px;
  margin-bottom: 24px;
  font-size: 0.875rem;
  color: {TEXT_SECONDARY};
  line-height: 1.6;
}}
.dash-narrative-icon {{
  flex-shrink: 0;
  width: 36px; height: 36px;
  background: rgba(59,130,246,0.1);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  margin-top: 2px;
}}
.dash-narrative-text strong {{
  color: {TEXT_PRIMARY};
  font-weight: 700;
}}

/* ============================================================
   KPI CARDS (dashboard-specific with sparklines)
   ============================================================ */
.dash-kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 28px;
}}
.kpi-card {{
  background: var(--bg-surface);
  border-radius: 12px;
  padding: 22px 24px;
  border: 1px solid {BORDER_SUBTLE};
  position: relative;
  overflow: hidden;
  transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
}}
.kpi-card:hover {{
  background: var(--bg-elevated);
  border-color: {BORDER_DEFAULT};
  transform: translateY(-1px);
  box-shadow: 0 6px 24px rgba(0,0,0,0.25);
}}
.kpi-status {{
  position: absolute; top: 0; right: 0;
  width: 4px; height: 100%;
  border-radius: 0 12px 12px 0;
}}
.kpi-label {{
  font-size: 0.6875rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: {TEXT_TERTIARY};
  font-weight: 600;
  margin-bottom: 8px;
}}
.kpi-value {{
  font-size: 1.85rem;
  font-weight: 800;
  color: {TEXT_PRIMARY};
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
  display: flex;
  align-items: baseline;
  gap: 6px;
}}
.kpi-unit {{
  font-size: 0.75rem;
  color: {TEXT_TERTIARY};
  font-weight: 500;
  letter-spacing: 0;
}}
.kpi-detail {{
  font-size: 0.6875rem;
  color: {TEXT_SECONDARY};
  margin-top: 8px;
  line-height: 1.4;
}}

/* ============================================================
   TAB NAVIGATION (Pill style)
   ============================================================ */
.dash-tab-wrapper {{
  margin-bottom: 24px;
  display: flex;
  justify-content: center;
}}
.dash-tab-nav {{
  display: inline-flex;
  gap: 4px;
  background: var(--bg-surface);
  border: 1px solid {BORDER_SUBTLE};
  border-radius: 12px;
  padding: 4px;
}}
.dash-tab-btn {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 22px;
  font-size: 0.875rem;
  font-weight: 600;
  color: {TEXT_TERTIARY};
  background: transparent;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
  font-family: inherit;
  white-space: nowrap;
}}
.dash-tab-btn svg {{
  opacity: 0.5;
  transition: opacity 0.25s;
}}
.dash-tab-btn:hover {{
  color: {TEXT_SECONDARY};
  background: var(--bg-elevated);
}}
.dash-tab-btn:hover svg {{ opacity: 0.7; }}
.dash-tab-btn.active {{
  color: {TEXT_PRIMARY};
  background: rgba(59,130,246,0.12);
  box-shadow: 0 2px 8px rgba(59,130,246,0.15);
}}
.dash-tab-btn.active svg {{ opacity: 1; color: {ACCENT_BLUE}; }}

/* ============================================================
   TAB SECTIONS & CHART CONTAINERS
   ============================================================ */
.dash-tab-section {{
  display: none;
  animation: fadeInUp 0.4s ease-out;
}}
.dash-tab-section.active {{ display: block; }}

.dash-chart-container {{
  margin-bottom: 20px;
  background: var(--bg-surface);
  border: 1px solid {BORDER_SUBTLE};
  border-radius: 12px;
  overflow: hidden;
  transition: border-color 0.25s;
  box-shadow: 0 18px 36px rgba(0,0,0,0.18);
}}
.dash-chart-container:hover {{
  border-color: {BORDER_DEFAULT};
}}
.dash-chart-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 24px 14px;
  border-bottom: 1px solid var(--border-subtle);
  gap: 12px;
  flex-wrap: wrap;
}}
.dash-chart-header h3 {{
  font-size: 1rem;
  font-weight: 700;
  color: {TEXT_PRIMARY};
  margin: 0;
  letter-spacing: -0.01em;
}}
.chart-box {{
  display: block;
  min-height: 420px;
  position: relative;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.01), transparent 24%),
    linear-gradient(180deg, rgba(15,17,23,0.22), rgba(15,17,23,0.06));
}}

/* Skeleton loading shimmer */
.dash-chart-skeleton {{
  position: absolute;
  inset: 0;
  padding: 20px;
}}

/* ============================================================
   METHODOLOGY (Collapsible)
   ============================================================ */
.dash-methodology {{
  margin-top: 32px;
  border: 1px solid {BORDER_SUBTLE};
  border-radius: 12px;
  overflow: hidden;
  font-size: 0.75rem;
  color: {TEXT_SECONDARY};
}}
.dash-methodology-toggle {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 22px;
  cursor: pointer;
  color: {TEXT_SECONDARY};
  font-weight: 600;
  font-size: 0.8125rem;
  background: var(--bg-surface);
  transition: all 0.2s;
  list-style: none;
}}
.dash-methodology-toggle::-webkit-details-marker {{ display: none; }}
.dash-methodology-toggle:hover {{
  color: {TEXT_PRIMARY};
  background: var(--bg-elevated);
}}
.dash-methodology-toggle svg {{
  transition: transform 0.25s;
}}
.dash-methodology[open] .dash-methodology-toggle svg {{
  transform: rotate(45deg);
}}
.dash-methodology-content {{
  padding: 0 22px 20px;
  animation: fadeIn 0.3s ease-out;
}}
.dash-methodology-grid {{
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 16px;
}}
.dash-methodology-item {{
  padding: 12px 16px;
  background: var(--bg-elevated);
  border-radius: 12px;
  border: 1px solid var(--border-subtle);
}}
.dash-methodology-label {{
  font-size: 0.625rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: {TEXT_TERTIARY};
  font-weight: 700;
  margin-bottom: 6px;
}}
.dash-methodology-disclaimer {{
  padding: 12px 16px;
  background: rgba(239,68,68,0.05);
  border: 1px solid rgba(239,68,68,0.1);
  border-radius: 12px;
  font-size: 0.6875rem;
  color: {TEXT_TERTIARY};
  font-style: italic;
}}

/* ============================================================
   RESPONSIVE
   ============================================================ */
@media (max-width: 1024px) {{
  .dash-hero-stats {{ grid-template-columns: repeat(2, 1fr); }}
  .dash-kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .dash-methodology-grid {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 768px) {{
  .dash-hero {{ padding: 32px 20px; margin: -20px -20px 20px; }}
  .dash-hero-title {{ font-size: 1.75rem; }}
  .dash-hero-stats {{ grid-template-columns: 1fr 1fr; gap: 10px; }}
  .dash-hero-stat-card {{ padding: 14px 12px; }}
  .dash-hero-stat-value {{ font-size: 1.375rem; }}
  .dash-kpi-grid {{ grid-template-columns: 1fr 1fr; gap: 10px; }}
  .dash-comparison {{ flex-direction: column; }}
  .dash-comparison-divider {{ transform: rotate(90deg); }}
  .dash-tab-nav {{ flex-wrap: wrap; justify-content: center; }}
  .dash-tab-btn {{ padding: 8px 14px; font-size: 0.75rem; }}
  .dash-chart-header {{ padding: 14px 16px 12px; }}
}}
@media (max-width: 480px) {{
  .dash-hero-stats {{ grid-template-columns: 1fr; }}
  .dash-kpi-grid {{ grid-template-columns: 1fr; }}
}}
"""

    # --- Extra JS: chart data + tab switching + lazy loading + scroll reveal ---
    extra_js = f"""
// Chart data (JSON strings, parsed on demand)
const chartData = {charts_js};

// ---- Scroll Reveal Observer ----
const revealObserver = new IntersectionObserver((entries) => {{
  entries.forEach(entry => {{
    if (entry.isIntersecting) {{
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }}
  }});
}}, {{ threshold: 0.08, rootMargin: '0px 0px -40px 0px' }});
document.querySelectorAll('.odt-reveal').forEach(el => revealObserver.observe(el));

// ---- Tab Switching ----
document.querySelectorAll('.dash-tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.dash-tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.dash-tab-section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    const tabId = 'tab-' + btn.dataset.tab;
    const tabEl = document.getElementById(tabId);
    tabEl.classList.add('active');
    // Render or resize charts in the newly active tab
    tabEl.querySelectorAll('.chart-box').forEach(el => {{
      if (el.dataset.rendered === 'true') {{
        Plotly.Plots.resize(el);
      }} else {{
        renderChart(el);
      }}
    }});
    tabEl.querySelectorAll('.odt-reveal:not(.visible)').forEach(el => revealObserver.observe(el));
    // Smooth scroll to tabs
    document.querySelector('.dash-tab-wrapper').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }});
}});

// ---- Lazy Chart Rendering ----
const dashObserver = new IntersectionObserver((entries) => {{
  entries.forEach(entry => {{
    if (entry.isIntersecting) {{
      const el = entry.target;
      const key = el.dataset.chart;
      if (chartData[key] && el.dataset.rendered !== 'true') {{
        try {{
          const d = JSON.parse(chartData[key]);
          el.innerHTML = '';
          Plotly.newPlot(el.id, d.data, d.layout, {{
            responsive: true,
            displayModeBar: true,
            scrollZoom: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false,
          }}).then((graphDiv) => {{
            window.__odtEnhancePlotly?.(graphDiv);
            Plotly.Plots.resize(graphDiv);
            window.requestAnimationFrame(() => Plotly.Plots.resize(graphDiv));
            el.style.opacity = '1';
          }});
          el.dataset.rendered = 'true';
        }} catch (e) {{
          el.innerHTML = '<div style="padding:20px;color:{ACCENT_RED};text-align:center;font-size:0.8125rem">Chart data unavailable</div>';
          console.error('Chart render error:', key, e);
        }}
      }}
    }}
  }});
}}, {{ rootMargin: '500px' }});

// Observe all chart containers
document.querySelectorAll('.chart-box').forEach(el => dashObserver.observe(el));

// Eagerly render charts in the active tab (don't wait for IntersectionObserver)
function renderChart(el) {{
  const key = el.dataset.chart;
  if (chartData[key] && el.dataset.rendered !== 'true') {{
    try {{
      const d = JSON.parse(chartData[key]);
      el.innerHTML = '';
      Plotly.newPlot(el.id, d.data, d.layout, {{
        responsive: true,
        displayModeBar: true,
        scrollZoom: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
      }}).then((graphDiv) => {{
        window.__odtEnhancePlotly?.(graphDiv);
        Plotly.Plots.resize(graphDiv);
      }});
      el.dataset.rendered = 'true';
    }} catch (e) {{
      console.error('Chart render error:', key, e);
    }}
  }}
}}
// Render active tab charts immediately
document.querySelectorAll('.dash-tab-section.active .chart-box').forEach(renderChart);
"""

    subtitle = (
        f"Oura Gen 4: {summary['date_range']} "
        f"({summary['n_unique_days']} unique days, "
        f"{summary['pre_sleep_periods'] + summary['post_sleep_periods']} sleep periods) "
        f"&middot; {n_json}/{total_json} analysis modules loaded "
        f"&middot; Ruxolitinib 10 mg BID from {RUXOLITINIB_START}"
    )

    return wrap_html(
        title="Oura Digital Twin Dashboard",
        body_content=body_content,
        report_id="3d_dashboard",
        subtitle=subtitle,
        extra_css=extra_css,
        extra_js=extra_js,
        data_end=summary["data_end"],
        post_days=summary["post_days"],
    )


# ===================================================================
# MAIN
# ===================================================================

def main() -> int:
    print("=" * 60)
    print("Oura Digital Twin — Unified CMO Dashboard")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load analysis module outputs
    print("\nLoading analysis module JSON outputs...")
    outputs = load_analysis_outputs(REPORTS_DIR)

    # Load raw data from database
    print("\nLoading data from database...")
    conn = get_db_connection()
    try:
        hr_df = load_heart_rate(conn)
        print(f"  Heart rate: {len(hr_df):,} readings")

        hrv_df = load_hrv(conn)
        print(f"  HRV: {len(hrv_df):,} readings")

        sleep_df = load_sleep_periods(conn)
        print(f"  Sleep periods: {len(sleep_df)} nights")

        epochs_df = load_sleep_epochs(conn)
        print(f"  Sleep epochs: {len(epochs_df):,}")

        spo2_df = load_spo2(conn)
        print(f"  SpO2: {len(spo2_df)} readings")

        readiness_df = load_readiness(conn)
        print(f"  Readiness: {len(readiness_df)} days")

        activity_df = load_activity(conn)
        print(f"  Activity: {len(activity_df)} days")
    finally:
        conn.close()

    # Compute summary
    print("\nComputing summary statistics...")
    summary = compute_summary(sleep_df, hrv_df, hr_df, spo2_df, readiness_df, epochs_df)
    full_analysis = outputs.get("full_analysis", {})
    for summary_key, full_key in (("mean_hrv", "rmssd_mean"), ("mean_spo2", "spo2_mean")):
        raw_value = full_analysis.get(full_key)
        try:
            if raw_value is not None:
                summary[summary_key] = float(raw_value)
        except (TypeError, ValueError):
            pass

    # Build narrative and KPI cards
    print("\nBuilding narrative and KPI cards...")
    narrative = build_narrative_summary(outputs, summary)
    kpi_html = build_kpi_cards(outputs, summary)

    # Build all visualizations
    print("\nBuilding visualizations...")
    figures = {}

    # Tab 1: Overview
    figures["hero_its"] = build_hero_its_chart(outputs, readiness_df)
    figures["forest_plot"] = build_forest_plot(outputs)

    # Tab 2: Disease Monitoring
    figures["disease_states"] = build_disease_states(outputs)
    figures["anomaly_timeline"] = build_anomaly_timeline(outputs)
    figures["biomarker_trends"] = build_biomarker_trends(outputs)
    figures["spo2_panel"] = build_spo2_panel(outputs, spo2_df)

    # Tab 3: Advanced Biosignals
    figures["biosignal_timeline"] = build_biosignal_timeline(
        sleep_df, hrv_df, spo2_df, readiness_df, outputs)
    figures["hr_terrain"] = build_hr_terrain(hr_df, sleep_df)
    figures["phase_space"] = build_phase_space(sleep_df, spo2_df)
    figures["sleep_heatmap"] = build_sleep_heatmap(epochs_df, sleep_df)
    figures["circadian_radar"] = build_circadian_radar(hr_df, outputs)

    # Assemble HTML
    html_content = build_html(figures, kpi_html, narrative, summary, outputs)

    # Write output
    print(f"\nWriting dashboard to {OUTPUT_FILE}...")
    OUTPUT_FILE.write_text(html_content, encoding="utf-8")

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  Output size: {size_mb:.1f} MB")

    # Write JSON metrics
    json_path = REPORTS_DIR / "oura_3d_dashboard_metrics.json"
    bos_payload = outputs.get("spo2", {}).get("bos_risk", {})
    bos_score = None
    bos_level = None
    bos_recommendation = None
    if isinstance(bos_payload, dict):
        raw_score = bos_payload.get("composite_score")
        try:
            if raw_score is not None:
                bos_score = float(raw_score)
        except (TypeError, ValueError):
            bos_score = None
        raw_level = bos_payload.get("risk_level")
        if raw_level is not None:
            bos_level = str(raw_level).upper()
        raw_recommendation = bos_payload.get("recommendation")
        if raw_recommendation is not None:
            bos_recommendation = str(raw_recommendation)

    json_metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_range": {
            "start": summary.get("data_start"),
            "end": summary.get("data_end"),
        },
        "data_points": {
            "unique_days": summary.get("n_unique_days", 0),
            "heart_rate_readings": summary.get("total_hr_readings", 0),
            "hrv_readings": summary.get("total_hrv_readings", 0),
            "sleep_nights": summary.get("total_nights", 0),
            "sleep_epochs": summary.get("total_epochs", 0),
        },
        "phase_distribution": summary.get("phase_pcts", {}),
        "summary_stats": {
            "mean_hr_bpm": round(summary["mean_hr"], 1) if summary.get("mean_hr") else None,
            "mean_hrv_ms": round(summary["mean_hrv"], 1) if summary.get("mean_hrv") else None,
            "mean_spo2_pct": round(summary["mean_spo2"], 1) if summary.get("mean_spo2") else None,
            "mean_sleep_efficiency_pct": round(summary["mean_efficiency"], 1) if summary.get("mean_efficiency") else None,
            "mean_temp_deviation_c": round(summary["mean_temp_dev"], 2) if summary.get("mean_temp_dev") else None,
            "mean_total_sleep_h": round(summary["mean_total_sleep_h"], 1) if summary.get("mean_total_sleep_h") else None,
        },
        "bos_risk": {
            "score_100": round(bos_score, 1) if bos_score is not None else None,
            "level": bos_level,
            "recommendation": bos_recommendation,
        },
        "pre_vs_post_treatment": {
            "pre_days": summary.get("pre_days", 0),
            "post_days": summary.get("post_days", 0),
            "pre_mean_hr": round(summary["pre_mean_hr"], 1) if summary.get("pre_mean_hr") else None,
            "post_mean_hr": round(summary["post_mean_hr"], 1) if summary.get("post_mean_hr") else None,
            "pre_mean_hrv": round(summary["pre_mean_hrv"], 1) if summary.get("pre_mean_hrv") else None,
            "post_mean_hrv": round(summary["post_mean_hrv"], 1) if summary.get("post_mean_hrv") else None,
        },
        "analysis_modules_loaded": {k: bool(v) for k, v in outputs.items()},
        "n_figures": len(figures),
        "html_size_mb": round(size_mb, 1),
    }
    json_path.write_text(json.dumps(json_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON metrics: {json_path}")

    print(f"\nDone! Open in browser:")
    print(f"  {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
