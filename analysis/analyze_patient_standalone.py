#!/usr/bin/env python3
"""Standalone health dashboard for non-Henrik patients.

Usage:
    python analysis/analyze_patient_standalone.py --profile mitch
    python analysis/analyze_patient_standalone.py --profile wenche
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Project imports (no Henrik-specific clinical dates)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from profiles import PROFILES
from analysis._hardening import safe_connect, safe_read_sql
from analysis._theme import (
    make_kpi_card, make_kpi_row, make_section, get_base_css,
    get_plotly_enhancer_js,
    BG_PRIMARY, TEXT_SECONDARY,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_CYAN,
    C_HRV, C_SLEEP, C_ACTIVITY,
)

pio.templates.default = "clinical_dark"

REPORTS_DIR = _PROJECT_ROOT / "reports"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_INTER_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2'
    '?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
)


def standalone_wrap_html(
    title: str,
    body_content: str,
    profile_label: str,
    condition: str,
) -> str:
    """Assemble a standalone HTML page with the project's dark clinical theme."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="{BG_PRIMARY}">
<meta name="robots" content="noindex, nofollow">
<title>{title} — Oura Digital Twin</title>
{_INTER_FONT_LINK}
<script src="{PLOTLY_CDN_URL}"></script>
{get_base_css()}
</head>
<body>

<div class="odt-context-strip">
<span class="odt-ctx-item">Oura Ring sensor data — not clinical measurements</span>
<span class="odt-ctx-dot"></span>
<span class="odt-ctx-item">Single-patient standalone dashboard</span>
</div>

<div class="odt-header">
  <h1>{title}</h1>
  <div class="subtitle">{profile_label} &mdash; {condition}</div>
  <div class="metadata">Generated {generated}</div>
</div>

<div class="odt-container">
{body_content}
</div>

<div class="odt-footer">
  <div>All metrics derived from Oura Ring consumer wearable data. Not clinical-grade.</div>
  <div>Single-patient overview. Not validated for clinical decision-making.</div>
  <div>Open source under MIT License &middot; &copy; 2026
  <a href="https://theeducationalequalityinstitute.org">The Educational Equality Institute</a></div>
</div>
{get_plotly_enhancer_js()}
</body>
</html>"""



def load_sleep_periods(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = (
        "SELECT day, average_hrv, average_heart_rate, average_breath, "
        "total_sleep_duration, rem_sleep_duration, deep_sleep_duration, "
        "light_sleep_duration, awake_time, efficiency, lowest_heart_rate "
        "FROM oura_sleep_periods WHERE type = 'long_sleep' ORDER BY day"
    )
    df = safe_read_sql(sql, conn, label="sleep_periods")
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.dropna(subset=["day"]).set_index("day").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_hrv(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = "SELECT timestamp, rmssd FROM oura_hrv ORDER BY timestamp"
    df = safe_read_sql(sql, conn, label="hrv")
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["rmssd"].mean().dropna()
    daily.index = pd.to_datetime(daily.index)
    daily.name = "rmssd"
    return daily.to_frame()


def load_activity(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = "SELECT date, steps, active_calories, daily_movement FROM oura_activity ORDER BY date"
    df = safe_read_sql(sql, conn, label="activity")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_heart_rate_daily(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = "SELECT timestamp, bpm FROM oura_heart_rate ORDER BY timestamp"
    df = safe_read_sql(sql, conn, label="heart_rate")
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["bpm"].agg(["mean", "min"]).dropna()
    daily.columns = ["avg_hr", "min_hr"]
    daily.index = pd.to_datetime(daily.index)
    return daily


def _chart_html(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _timeseries_with_rolling(
    series: pd.Series, title: str, y_label: str, color: str,
    rolling_window: int = 7,
) -> go.Figure:
    fig = go.Figure()
    clean = series.dropna()
    if clean.empty:
        fig.add_annotation(text="No data available", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig.add_trace(go.Scatter(
        x=clean.index, y=clean.values,
        mode="markers",
        marker=dict(size=3, color=color, opacity=0.3),
        name="Daily",
        showlegend=False,
    ))

    if len(clean) >= rolling_window:
        rolling = clean.rolling(rolling_window, min_periods=max(1, rolling_window // 2)).mean()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values,
            mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{rolling_window}-day avg",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title=y_label,
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
        height=400,
    )
    return fig


def _histogram(series: pd.Series, title: str, x_label: str, color: str) -> go.Figure:
    fig = go.Figure()
    clean = series.dropna()
    if clean.empty:
        fig.add_annotation(text="No data available", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig.add_trace(go.Histogram(
        x=clean.values,
        nbinsx=40,
        marker_color=color,
        opacity=0.8,
        name=x_label,
    ))
    fig.add_vline(x=clean.median(), line_dash="dash",
                  line_color=TEXT_SECONDARY, opacity=0.7,
                  annotation_text=f"Median: {clean.median():.1f}")

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=x_label,
        yaxis_title="Count",
        margin=dict(l=60, r=20, t=50, b=60),
        height=350,
    )
    return fig


def _dual_timeseries(
    s1: pd.Series, s2: pd.Series, title: str, y_label: str,
    c1: str, c2: str, n1: str = "Series 1", n2: str = "Series 2",
    rolling_window: int = 7,
) -> go.Figure:
    fig = go.Figure()

    for s, color, name in [(s1, c1, n1), (s2, c2, n2)]:
        clean = s.dropna()
        if clean.empty:
            continue
        fig.add_trace(go.Scatter(
            x=clean.index, y=clean.values,
            mode="markers", marker=dict(size=3, color=color, opacity=0.25),
            name=f"{name} (daily)", showlegend=False,
        ))
        if len(clean) >= rolling_window:
            rolling = clean.rolling(rolling_window, min_periods=max(1, rolling_window // 2)).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values,
                mode="lines", line=dict(color=color, width=2.5),
                name=f"{name} ({rolling_window}d avg)",
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title=y_label, xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
        height=400,
    )
    return fig


def build_sleep_stages_chart(sleep: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    total = sleep["total_sleep_duration"]
    mask = total > 0
    if mask.sum() == 0:
        fig.add_annotation(text="No sleep stage data", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    deep_pct = (sleep["deep_sleep_duration"][mask] / total[mask] * 100)
    rem_pct = (sleep["rem_sleep_duration"][mask] / total[mask] * 100)
    light_pct = (sleep["light_sleep_duration"][mask] / total[mask] * 100)

    window = 7
    for pct, name, color in [
        (deep_pct, "Deep", ACCENT_PURPLE),
        (rem_pct, "REM", ACCENT_CYAN),
        (light_pct, "Light", ACCENT_BLUE),
    ]:
        smooth = pct.rolling(window, min_periods=max(1, window // 2)).mean()
        fig.add_trace(go.Scatter(
            x=smooth.index, y=smooth.values,
            mode="lines", name=name,
            line=dict(color=color, width=2),
            stackgroup="stages",
        ))

    fig.update_layout(
        title=dict(text="Sleep Stage Distribution (7-day avg %)", font=dict(size=16)),
        yaxis_title="% of Total Sleep",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
        height=400,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone patient health dashboard")
    parser.add_argument(
        "--profile", required=True,
        help="Profile key from profiles.py (e.g. mitch, wenche)",
    )
    args = parser.parse_args()

    profile_key = args.profile.lower()
    if profile_key not in PROFILES:
        print(f"ERROR: Unknown profile '{profile_key}'. "
              f"Available: {', '.join(PROFILES.keys())}", file=sys.stderr)
        sys.exit(1)

    if profile_key == "henrik":
        print("ERROR: Use the main analysis pipeline for Henrik (P1). "
              "This script is for non-Henrik patients.", file=sys.stderr)
        sys.exit(1)

    profile = PROFILES[profile_key]
    db_path = Path(profile["database"])
    label = profile["label"]
    condition = profile["condition"]
    age = profile.get("age", "N/A")
    ring_gen = profile.get("ring_gen", "?")

    print(f"[standalone] Generating dashboard for {profile_key} ({label})")
    print(f"  DB: {db_path}")

    conn = safe_connect(db_path, read_only=True)

    sleep = load_sleep_periods(conn)
    hrv = load_hrv(conn)
    activity = load_activity(conn)
    hr_daily = load_heart_rate_daily(conn)
    conn.close()

    data_days = max(len(sleep), len(hrv), len(activity), len(hr_daily))
    print(f"  Data days: sleep={len(sleep)}, hrv={len(hrv)}, "
          f"activity={len(activity)}, hr={len(hr_daily)}")

    mean_rmssd = sleep["average_hrv"].mean() if not sleep.empty and "average_hrv" in sleep.columns else np.nan
    mean_rhr = sleep["lowest_heart_rate"].mean() if not sleep.empty else np.nan
    mean_sleep_hrs = (sleep["total_sleep_duration"].mean() / 3600) if not sleep.empty else np.nan
    mean_efficiency = sleep["efficiency"].mean() if not sleep.empty else np.nan
    mean_steps = activity["steps"].mean() if not activity.empty else np.nan

    def _status(val, thresholds, invert=False):
        """Return status string. thresholds=(warn, crit) for lower-is-worse."""
        if np.isnan(val):
            return "neutral"
        if invert:  # higher is worse (e.g. RHR)
            return "critical" if val > thresholds[1] else "warning" if val > thresholds[0] else "good"
        return "critical" if val < thresholds[0] else "warning" if val < thresholds[1] else "good"

    hrv_status = _status(mean_rmssd, (20, 30))
    rhr_status = _status(mean_rhr, (75, 85), invert=True)
    sleep_status = _status(mean_sleep_hrs, (6, 7))
    eff_status = _status(mean_efficiency, (80, 85))

    body = ""

    body += make_kpi_row(
        make_kpi_card("MEAN HRV (RMSSD)", mean_rmssd, "ms",
                      status=hrv_status, decimals=1,
                      detail="Sleep-period average"),
        make_kpi_card("RESTING HR", mean_rhr, "bpm",
                      status=rhr_status, decimals=0,
                      detail="Lowest during sleep"),
        make_kpi_card("SLEEP DURATION", mean_sleep_hrs, "hrs",
                      status=sleep_status, decimals=1,
                      detail="Long-sleep average"),
        make_kpi_card("SLEEP EFFICIENCY", mean_efficiency, "%",
                      status=eff_status, decimals=0,
                      detail="Time asleep / time in bed"),
        make_kpi_card("DATA DAYS", data_days, "",
                      status="info", decimals=0,
                      detail=f"Oura Gen {ring_gen} &middot; Age {age}"),
    )

    hrv_content = ""
    if not sleep.empty and "average_hrv" in sleep.columns:
        hrv_series = sleep["average_hrv"].dropna()
        fig_hrv_ts = _timeseries_with_rolling(
            hrv_series, "Daily RMSSD (from sleep periods)", "RMSSD (ms)", C_HRV,
        )
        hrv_content += _chart_html(fig_hrv_ts)

        fig_hrv_hist = _histogram(hrv_series, "RMSSD Distribution", "RMSSD (ms)", C_HRV)
        hrv_content += _chart_html(fig_hrv_hist)
    elif not hrv.empty:
        hrv_series = hrv["rmssd"].dropna()
        fig_hrv_ts = _timeseries_with_rolling(
            hrv_series, "Daily RMSSD (aggregated from 5-min readings)", "RMSSD (ms)", C_HRV,
        )
        hrv_content += _chart_html(fig_hrv_ts)

        fig_hrv_hist = _histogram(hrv_series, "RMSSD Distribution", "RMSSD (ms)", C_HRV)
        hrv_content += _chart_html(fig_hrv_hist)
    else:
        hrv_content = "<p>No HRV data available.</p>"

    body += make_section("Heart Rate Variability", hrv_content, section_id="hrv")

    hr_content = ""
    if not sleep.empty and "lowest_heart_rate" in sleep.columns and "average_heart_rate" in sleep.columns:
        lowest_hr = sleep["lowest_heart_rate"].dropna()
        avg_hr = sleep["average_heart_rate"].dropna()
        fig_hr = _dual_timeseries(
            lowest_hr, avg_hr,
            "Heart Rate During Sleep", "BPM",
            ACCENT_GREEN, ACCENT_BLUE,
            "Lowest HR", "Average HR",
        )
        hr_content += _chart_html(fig_hr)
    elif not hr_daily.empty:
        fig_hr = _dual_timeseries(
            hr_daily["min_hr"], hr_daily["avg_hr"],
            "Daily Heart Rate", "BPM",
            ACCENT_GREEN, ACCENT_BLUE,
            "Min HR", "Average HR",
        )
        hr_content += _chart_html(fig_hr)
    else:
        hr_content = "<p>No heart rate data available.</p>"

    body += make_section("Heart Rate", hr_content, section_id="heart-rate")

    sleep_content = ""
    if not sleep.empty:
        # Duration
        dur_hrs = (sleep["total_sleep_duration"] / 3600).dropna()
        fig_dur = _timeseries_with_rolling(
            dur_hrs, "Sleep Duration", "Hours", C_SLEEP,
        )
        sleep_content += _chart_html(fig_dur)

        # Efficiency
        eff = sleep["efficiency"].dropna()
        fig_eff = _timeseries_with_rolling(
            eff, "Sleep Efficiency", "%", ACCENT_CYAN,
        )
        sleep_content += _chart_html(fig_eff)

        # Stage breakdown
        stage_cols = ["deep_sleep_duration", "rem_sleep_duration", "light_sleep_duration"]
        if all(c in sleep.columns for c in stage_cols):
            fig_stages = build_sleep_stages_chart(sleep)
            sleep_content += _chart_html(fig_stages)
    else:
        sleep_content = "<p>No sleep data available.</p>"

    body += make_section("Sleep", sleep_content, section_id="sleep")

    activity_content = ""
    if not activity.empty and "steps" in activity.columns:
        steps = activity["steps"].dropna()
        fig_steps = _timeseries_with_rolling(
            steps, "Daily Steps", "Steps", C_ACTIVITY,
        )
        activity_content += _chart_html(fig_steps)
    else:
        activity_content = "<p>No activity data available.</p>"

    body += make_section("Activity", activity_content, section_id="activity")

    html = standalone_wrap_html(
        title=f"{label} Dashboard",
        body_content=body,
        profile_label=label,
        condition=condition,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / f"{profile_key}_standalone_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  HTML: {html_path}")

    metrics = {
        "profile": profile_key,
        "label": label,
        "condition": condition,
        "age": age,
        "ring_gen": ring_gen,
        "generated": datetime.now().isoformat(),
        "data_days": int(data_days),
        "kpis": {
            "mean_rmssd_ms": round(float(mean_rmssd), 1) if not np.isnan(mean_rmssd) else None,
            "mean_resting_hr_bpm": round(float(mean_rhr), 0) if not np.isnan(mean_rhr) else None,
            "mean_sleep_duration_hrs": round(float(mean_sleep_hrs), 2) if not np.isnan(mean_sleep_hrs) else None,
            "mean_sleep_efficiency_pct": round(float(mean_efficiency), 0) if not np.isnan(mean_efficiency) else None,
            "mean_daily_steps": round(float(mean_steps), 0) if not np.isnan(mean_steps) else None,
        },
    }
    json_path = REPORTS_DIR / f"{profile_key}_standalone_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  JSON: {json_path}")
    print("[standalone] Done.")


if __name__ == "__main__":
    main()
