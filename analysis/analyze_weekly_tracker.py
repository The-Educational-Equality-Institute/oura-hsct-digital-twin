#!/usr/bin/env python3
"""
Weekly Trend Tracker — This Week vs Last Week

Compares the last 7 days against the prior 7 days across 12 key health
metrics. Designed for doctor visits: clear KPI cards, sparklines, traffic
lights, and an auto-generated doctor summary.

Outputs:
  - Interactive HTML: reports/weekly_tracker.html
  - Structured JSON:  reports/weekly_tracker.json

Usage:
    python analysis/analyze_weekly_tracker.py
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path resolution & config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATABASE_PATH, REPORTS_DIR, TREATMENT_START, PATIENT_LABEL,
    TREATMENT_START_STR, FONT_FAMILY,
)
from _hardening import safe_connect, safe_read_sql, section_html_or_placeholder
from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    COLORWAY, STATUS_COLORS, BG_PRIMARY, BG_SURFACE, BG_ELEVATED,
    BORDER_SUBTLE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    ACCENT_PURPLE, ACCENT_CYAN, ACCENT_ORANGE,
    C_HR, C_HRV, C_SLEEP, C_TEMP,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "weekly_tracker.html"
JSON_OUTPUT = REPORTS_DIR / "weekly_tracker.json"

TODAY = date.today()
THIS_WEEK_END = TODAY
THIS_WEEK_START = TODAY - timedelta(days=6)
LAST_WEEK_END = THIS_WEEK_START - timedelta(days=1)
LAST_WEEK_START = LAST_WEEK_END - timedelta(days=6)
WINDOW_START = LAST_WEEK_START  # 14 days total


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricDef:
    key: str
    display_name: str
    table: str
    column: str
    unit: str
    higher_is_better: Optional[bool]  # None = closer to 0 is better
    decimals: int = 1
    transform: Optional[str] = None  # "sec_to_hours" or "abs"
    color: str = ACCENT_BLUE
    normal_range: tuple[float, float] = (0.0, 999.0)
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None


METRICS = [
    MetricDef(
        key="hrv_avg", display_name="HRV (RMSSD)",
        table="oura_sleep_periods", column="average_hrv",
        unit="ms", higher_is_better=True, color=C_HRV,
        normal_range=(15.0, 80.0), critical_low=10.0,
    ),
    MetricDef(
        key="lowest_hr", display_name="Lowest HR",
        table="oura_sleep_periods", column="lowest_heart_rate",
        unit="bpm", higher_is_better=False, color=C_HR,
        normal_range=(40.0, 70.0), critical_high=85.0,
    ),
    MetricDef(
        key="avg_hr", display_name="Average HR",
        table="oura_sleep_periods", column="average_heart_rate",
        unit="bpm", higher_is_better=False, color=C_HR,
        normal_range=(50.0, 75.0), critical_high=90.0,
    ),
    MetricDef(
        key="sleep_duration", display_name="Sleep Duration",
        table="oura_sleep_periods", column="total_sleep_duration",
        unit="h", higher_is_better=True, transform="sec_to_hours",
        color=C_SLEEP, normal_range=(6.0, 9.0), critical_low=4.0,
    ),
    MetricDef(
        key="deep_sleep", display_name="Deep Sleep",
        table="oura_sleep_periods", column="deep_sleep_duration",
        unit="h", higher_is_better=True, transform="sec_to_hours",
        color=C_SLEEP, normal_range=(0.8, 2.5), critical_low=0.3,
    ),
    MetricDef(
        key="rem_sleep", display_name="REM Sleep",
        table="oura_sleep_periods", column="rem_sleep_duration",
        unit="h", higher_is_better=True, transform="sec_to_hours",
        color=C_SLEEP, normal_range=(1.0, 2.5), critical_low=0.3,
    ),
    MetricDef(
        key="efficiency", display_name="Sleep Efficiency",
        table="oura_sleep_periods", column="efficiency",
        unit="%", higher_is_better=True, decimals=0,
        color=C_SLEEP, normal_range=(80.0, 100.0), critical_low=70.0,
    ),
    MetricDef(
        key="readiness", display_name="Readiness Score",
        table="oura_readiness", column="score",
        unit="pts", higher_is_better=True, decimals=0,
        color=ACCENT_BLUE, normal_range=(60.0, 100.0), critical_low=40.0,
    ),
    MetricDef(
        key="recovery", display_name="Recovery Index",
        table="oura_readiness", column="recovery_index",
        unit="pts", higher_is_better=True, decimals=0,
        color=ACCENT_GREEN, normal_range=(40.0, 100.0), critical_low=20.0,
    ),
    MetricDef(
        key="steps", display_name="Steps",
        table="oura_activity", column="steps",
        unit="steps", higher_is_better=True, decimals=0,
        color=ACCENT_CYAN, normal_range=(3000.0, 15000.0), critical_low=1000.0,
    ),
    MetricDef(
        key="temp_dev", display_name="Temp Deviation",
        table="oura_readiness", column="temperature_deviation",
        unit="\u00b0C", higher_is_better=None, transform="abs",
        decimals=2, color=C_TEMP,
        normal_range=(0.0, 0.5), critical_high=1.5,
    ),
    MetricDef(
        key="breath_rate", display_name="Breath Rate",
        table="oura_sleep_periods", column="average_breath",
        unit="brpm", higher_is_better=False,
        color=ACCENT_PURPLE, normal_range=(12.0, 20.0), critical_high=25.0,
    ),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_14day_data(conn) -> dict[str, pd.DataFrame]:
    """Load last 14 days from each required table."""
    start_str = str(WINDOW_START)
    end_str = str(THIS_WEEK_END)

    tables: dict[str, pd.DataFrame] = {}

    # oura_sleep_periods (type='long_sleep', date column = 'day')
    sql_sp = (
        f"SELECT day, average_hrv, lowest_heart_rate, average_heart_rate, "
        f"total_sleep_duration, deep_sleep_duration, rem_sleep_duration, "
        f"efficiency, average_breath "
        f"FROM oura_sleep_periods "
        f"WHERE type = 'long_sleep' AND day BETWEEN '{start_str}' AND '{end_str}' "
        f"ORDER BY day"
    )
    df_sp = safe_read_sql(sql_sp, conn, label="sleep_periods_14d")
    if not df_sp.empty:
        df_sp["day"] = pd.to_datetime(df_sp["day"], errors="coerce", utc=True)
        df_sp = df_sp.dropna(subset=["day"])
        df_sp = df_sp.set_index("day").sort_index()
        df_sp = df_sp[~df_sp.index.duplicated(keep="last")]
    tables["oura_sleep_periods"] = df_sp

    # oura_readiness
    sql_rd = (
        f"SELECT date, score, recovery_index, temperature_deviation "
        f"FROM oura_readiness "
        f"WHERE date BETWEEN '{start_str}' AND '{end_str}' "
        f"ORDER BY date"
    )
    df_rd = safe_read_sql(sql_rd, conn, label="readiness_14d")
    if not df_rd.empty:
        df_rd["date"] = pd.to_datetime(df_rd["date"], errors="coerce", utc=True)
        df_rd = df_rd.dropna(subset=["date"])
        df_rd = df_rd.set_index("date").sort_index()
        df_rd = df_rd[~df_rd.index.duplicated(keep="last")]
    tables["oura_readiness"] = df_rd

    # oura_activity
    sql_ac = (
        f"SELECT date, steps "
        f"FROM oura_activity "
        f"WHERE date BETWEEN '{start_str}' AND '{end_str}' "
        f"ORDER BY date"
    )
    df_ac = safe_read_sql(sql_ac, conn, label="activity_14d")
    if not df_ac.empty:
        df_ac["date"] = pd.to_datetime(df_ac["date"], errors="coerce", utc=True)
        df_ac = df_ac.dropna(subset=["date"])
        df_ac = df_ac.set_index("date").sort_index()
        df_ac = df_ac[~df_ac.index.duplicated(keep="last")]
    tables["oura_activity"] = df_ac

    return tables


def extract_series(tables: dict[str, pd.DataFrame], m: MetricDef) -> pd.Series:
    """Pull a single metric series from the loaded tables, applying transforms."""
    df = tables.get(m.table, pd.DataFrame())
    if df.empty or m.column not in df.columns:
        return pd.Series(dtype=float, name=m.key)
    s = df[m.column].dropna().astype(float)
    s.name = m.key
    if m.transform == "sec_to_hours":
        s = s / 3600.0
    return s


# ---------------------------------------------------------------------------
# Weekly comparison
# ---------------------------------------------------------------------------

@dataclass
class WeeklyResult:
    key: str
    display_name: str
    unit: str
    this_week_mean: Optional[float]
    last_week_mean: Optional[float]
    delta: Optional[float]
    pct_change: Optional[float]
    direction: str  # "improving", "stable", "declining"
    status: str  # "good", "warning", "critical"
    n_this: int
    n_last: int
    series_14d: pd.Series
    higher_is_better: Optional[bool]
    decimals: int


def compute_weekly(m: MetricDef, series: pd.Series) -> WeeklyResult:
    """Compute this week vs last week for a single metric."""
    tw_start = pd.Timestamp(THIS_WEEK_START, tz="UTC")
    tw_end = pd.Timestamp(THIS_WEEK_END, tz="UTC") + pd.Timedelta(days=1)
    lw_start = pd.Timestamp(LAST_WEEK_START, tz="UTC")
    lw_end = pd.Timestamp(LAST_WEEK_END, tz="UTC") + pd.Timedelta(days=1)

    this_week = series[(series.index >= tw_start) & (series.index < tw_end)]
    last_week = series[(series.index >= lw_start) & (series.index < lw_end)]

    # For temp deviation, use absolute values for comparison
    if m.higher_is_better is None and m.transform == "abs":
        this_week = this_week.abs()
        last_week = last_week.abs()

    tw_mean = float(this_week.mean()) if len(this_week) > 0 else None
    lw_mean = float(last_week.mean()) if len(last_week) > 0 else None

    delta = None
    pct_change = None
    direction = "stable"
    status = "good"

    if tw_mean is not None and lw_mean is not None:
        delta = tw_mean - lw_mean

        if lw_mean != 0:
            pct_change = (delta / abs(lw_mean)) * 100.0
        else:
            pct_change = 0.0

        # Direction based on clinical meaning
        direction = _classify_direction(delta, pct_change, m.higher_is_better)

        # Traffic light status
        status = _classify_status(tw_mean, delta, pct_change, m)
    elif tw_mean is not None:
        status = _value_status(tw_mean, m)
    else:
        status = "warning"
        direction = "no_data"

    return WeeklyResult(
        key=m.key,
        display_name=m.display_name,
        unit=m.unit,
        this_week_mean=tw_mean,
        last_week_mean=lw_mean,
        delta=delta,
        pct_change=pct_change,
        direction=direction,
        status=status,
        n_this=len(this_week),
        n_last=len(last_week),
        series_14d=series,
        higher_is_better=m.higher_is_better,
        decimals=m.decimals,
    )


def _classify_direction(
    delta: float, pct_change: float, higher_is_better: Optional[bool],
) -> str:
    """Determine if the change is improving, stable, or declining."""
    abs_pct = abs(pct_change)
    if abs_pct < 3.0:
        return "stable"

    if higher_is_better is None:
        # Closer to 0 is better (temp deviation) -- increasing abs = declining
        return "declining" if delta > 0 else "improving"

    if higher_is_better:
        return "improving" if delta > 0 else "declining"
    else:
        return "declining" if delta > 0 else "improving"


def _value_status(value: float, m: MetricDef) -> str:
    """Traffic light from absolute value alone."""
    if m.critical_low is not None and value < m.critical_low:
        return "critical"
    if m.critical_high is not None and value > m.critical_high:
        return "critical"
    low, high = m.normal_range
    if low <= value <= high:
        return "good"
    return "warning"


def _classify_status(
    tw_mean: float, delta: float, pct_change: float, m: MetricDef,
) -> str:
    """Assign green/amber/red traffic light."""
    # Critical absolute values always override
    if m.critical_low is not None and tw_mean < m.critical_low:
        return "critical"
    if m.critical_high is not None and tw_mean > m.critical_high:
        return "critical"

    low, high = m.normal_range
    in_range = low <= tw_mean <= high
    abs_pct = abs(pct_change)

    # Significant decline (>10% worse)
    direction = _classify_direction(delta, pct_change, m.higher_is_better)
    if direction == "declining" and abs_pct > 10.0:
        return "critical"

    if direction == "declining" and abs_pct > 5.0:
        return "warning"

    if not in_range and direction == "declining":
        return "warning"

    if in_range and direction in ("improving", "stable"):
        return "good"

    if in_range:
        return "good"

    # Out of range but improving
    if direction == "improving":
        return "warning"

    return "warning"


# ---------------------------------------------------------------------------
# Doctor summary generation
# ---------------------------------------------------------------------------

def generate_doctor_summary(
    results: list[WeeklyResult],
    days_on_rux: int,
) -> list[str]:
    """Auto-generate 3-5 actionable bullet points for the doctor."""
    bullets: list[str] = []

    # Sort by absolute pct_change descending to find biggest movers
    ranked = sorted(
        [r for r in results if r.pct_change is not None],
        key=lambda r: abs(r.pct_change or 0),
        reverse=True,
    )

    # Critical items first
    critical = [r for r in results if r.status == "critical"]
    for r in critical[:2]:
        arrow = _direction_word(r.direction)
        val_str = _fmt_value(r.this_week_mean, r.decimals)
        bullets.append(
            f"ATTENTION: {r.display_name} is {arrow} at {val_str} {r.unit} "
            f"({_fmt_pct(r.pct_change)} week-over-week)"
        )

    # Biggest improvers
    improving = [r for r in ranked if r.direction == "improving" and r.status != "critical"]
    for r in improving[:2]:
        val_str = _fmt_value(r.this_week_mean, r.decimals)
        bullets.append(
            f"{r.display_name} improved {_fmt_pct(r.pct_change)} to {val_str} {r.unit}"
        )

    # Biggest decliners (not already in critical)
    critical_keys = {r.key for r in critical}
    declining = [
        r for r in ranked
        if r.direction == "declining" and r.key not in critical_keys
    ]
    for r in declining[:1]:
        val_str = _fmt_value(r.this_week_mean, r.decimals)
        bullets.append(
            f"{r.display_name} declined {_fmt_pct(r.pct_change)} to {val_str} {r.unit} -- monitor"
        )

    # Ruxolitinib context
    if days_on_rux > 0:
        bullets.append(
            f"Day {days_on_rux} on ruxolitinib (started {TREATMENT_START_STR})"
        )

    # Stable summary if few changes
    stable = [r for r in results if r.direction == "stable"]
    if len(stable) >= 8:
        bullets.append(
            f"{len(stable)} of 12 metrics stable week-over-week"
        )

    return bullets[:5]


def _direction_word(direction: str) -> str:
    if direction == "improving":
        return "improving"
    if direction == "declining":
        return "declining"
    return "stable"


def _fmt_value(val: Optional[float], decimals: int) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_pct(pct: Optional[float]) -> str:
    if pct is None:
        return "N/A"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


# ---------------------------------------------------------------------------
# Sparkline charts
# ---------------------------------------------------------------------------

def make_sparkline(series: pd.Series, m: MetricDef) -> str:
    """Build a tiny 14-day Plotly sparkline for embedding."""
    if series.empty:
        return "<div style='height:50px;color:#6B7280;font-size:11px;'>No data</div>"

    plot_series = series.copy()
    if m.transform == "sec_to_hours":
        plot_series = plot_series / 3600.0
    if m.transform == "abs":
        plot_series = plot_series.abs()

    fig = go.Figure()

    # Shaded region for "this week"
    tw_start = pd.Timestamp(THIS_WEEK_START, tz="UTC")
    tw_end = pd.Timestamp(THIS_WEEK_END, tz="UTC") + pd.Timedelta(days=1)
    fig.add_shape(
        type="rect",
        x0=tw_start, x1=tw_end,
        y0=0, y1=1, yref="paper",
        fillcolor="rgba(59,130,246,0.08)",
        line=dict(width=0),
        layer="below",
    )

    fig.add_trace(go.Scatter(
        x=plot_series.index,
        y=plot_series.values,
        mode="lines+markers",
        line=dict(color=m.color, width=1.5),
        marker=dict(size=3, color=m.color),
        hovertemplate="%{x|%b %d}: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        height=55, width=180,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


# ---------------------------------------------------------------------------
# Ruxolitinib section
# ---------------------------------------------------------------------------

def build_rux_section(
    results: list[WeeklyResult],
    conn,
    days_on_rux: int,
) -> str:
    """Ruxolitinib tracking: days since start + improvement vs pre-rux baseline."""
    if days_on_rux <= 0:
        return (
            '<p style="color:#9CA3AF;">Ruxolitinib has not started yet '
            f'(scheduled: {TREATMENT_START_STR})</p>'
        )

    # Load pre-rux baseline (14 days before treatment start)
    baseline_start = str(TREATMENT_START - timedelta(days=14))
    baseline_end = str(TREATMENT_START - timedelta(days=1))

    baseline_metrics: dict[str, float] = {}
    for m in METRICS:
        date_col = "day" if m.table == "oura_sleep_periods" else "date"
        filter_clause = " AND type = 'long_sleep'" if m.table == "oura_sleep_periods" else ""
        sql = (
            f"SELECT {m.column} FROM {m.table} "
            f"WHERE {date_col} BETWEEN '{baseline_start}' AND '{baseline_end}'"
            f"{filter_clause}"
        )
        df = safe_read_sql(sql, conn, label=f"rux_baseline_{m.key}")
        if not df.empty and m.column in df.columns:
            vals = df[m.column].dropna().astype(float)
            if m.transform == "sec_to_hours":
                vals = vals / 3600.0
            if m.transform == "abs":
                vals = vals.abs()
            if len(vals) > 0:
                baseline_metrics[m.key] = float(vals.mean())

    rows = []
    for r in results:
        if r.this_week_mean is None or r.key not in baseline_metrics:
            continue
        baseline_val = baseline_metrics[r.key]
        if baseline_val == 0:
            continue
        cum_delta = r.this_week_mean - baseline_val
        cum_pct = (cum_delta / abs(baseline_val)) * 100.0

        # Direction arrow
        if r.higher_is_better is True:
            arrow = "\u2191" if cum_delta > 0 else "\u2193"
            cls = "good" if cum_delta > 0 else "warning"
        elif r.higher_is_better is False:
            arrow = "\u2193" if cum_delta < 0 else "\u2191"
            cls = "good" if cum_delta < 0 else "warning"
        else:
            arrow = "\u2193" if abs(r.this_week_mean) < abs(baseline_val) else "\u2191"
            cls = "good" if abs(r.this_week_mean) < abs(baseline_val) else "warning"

        color = ACCENT_GREEN if cls == "good" else ACCENT_AMBER
        rows.append(
            f"<tr>"
            f"<td>{r.display_name}</td>"
            f"<td>{_fmt_value(baseline_val, r.decimals)} {r.unit}</td>"
            f"<td>{_fmt_value(r.this_week_mean, r.decimals)} {r.unit}</td>"
            f"<td style='color:{color}'>{arrow} {_fmt_pct(cum_pct)}</td>"
            f"</tr>"
        )

    table_html = (
        f"<p><b>Day {days_on_rux}</b> on ruxolitinib "
        f"(started {TREATMENT_START_STR})</p>"
        f"<table>"
        f"<tr><th>Metric</th><th>Pre-Rux Baseline</th>"
        f"<th>This Week</th><th>Change</th></tr>"
        f"{''.join(rows)}"
        f"</table>"
    )
    return table_html


# ---------------------------------------------------------------------------
# HTML construction
# ---------------------------------------------------------------------------

EXTRA_CSS = """
.weekly-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.weekly-card {
    background: #1A1D27;
    border: 1px solid #2D3348;
    border-radius: 10px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    position: relative;
    overflow: hidden;
}
.weekly-card .status-bar {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    border-radius: 10px 0 0 10px;
}
.weekly-card .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-left: 8px;
}
.weekly-card .metric-name {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9CA3AF;
}
.weekly-card .traffic-light {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
}
.weekly-card .metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #E8E8ED;
    padding-left: 8px;
    line-height: 1.1;
}
.weekly-card .metric-unit {
    font-size: 13px;
    font-weight: 400;
    color: #6B7280;
    margin-left: 3px;
}
.weekly-card .change-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-left: 8px;
    font-size: 12px;
}
.weekly-card .change-badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 11px;
}
.weekly-card .sparkline-box {
    margin-top: 2px;
    min-height: 55px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.weekly-card .last-week-label {
    font-size: 11px;
    color: #6B7280;
    padding-left: 8px;
}
.doctor-summary {
    background: #242837;
    border: 1px solid #2D3348;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 16px 0;
}
.doctor-summary h3 {
    margin: 0 0 12px 0;
    font-size: 15px;
    color: #E8E8ED;
}
.doctor-summary ul {
    margin: 0;
    padding-left: 20px;
}
.doctor-summary li {
    margin-bottom: 8px;
    font-size: 14px;
    line-height: 1.5;
    color: #E8E8ED;
}
.doctor-summary li.attention {
    color: #FCA5A5;
}
"""


def _direction_arrow_html(result: WeeklyResult) -> str:
    """Return colored arrow + pct change badge."""
    if result.pct_change is None or result.direction == "no_data":
        return '<span style="color:#6B7280;">--</span>'

    pct_str = _fmt_pct(result.pct_change)

    if result.direction == "improving":
        color = ACCENT_GREEN
        bg = "rgba(16,185,129,0.15)"
        arrow = "\u2191" if (result.higher_is_better is True or result.higher_is_better is None) else "\u2193"
        if result.higher_is_better is False:
            arrow = "\u2193"
    elif result.direction == "declining":
        color = ACCENT_RED
        bg = "rgba(239,68,68,0.15)"
        arrow = "\u2193" if (result.higher_is_better is True or result.higher_is_better is None) else "\u2191"
        if result.higher_is_better is False:
            arrow = "\u2191"
    else:
        color = TEXT_SECONDARY
        bg = "rgba(156,163,175,0.10)"
        arrow = "\u2194"

    return (
        f'<span class="change-badge" '
        f'style="color:{color};background:{bg}">'
        f'{arrow} {pct_str}</span>'
    )


def _traffic_light_color(status: str) -> str:
    return {
        "good": ACCENT_GREEN,
        "warning": ACCENT_AMBER,
        "critical": ACCENT_RED,
    }.get(status, TEXT_TERTIARY)


def build_metric_card(result: WeeklyResult, sparkline_html: str) -> str:
    """Build a single metric card with value, change, sparkline."""
    tl_color = _traffic_light_color(result.status)
    status_color = STATUS_COLORS.get(
        "normal" if result.status == "good" else result.status, "transparent"
    )
    val_str = _fmt_value(result.this_week_mean, result.decimals)
    lw_str = (
        f"Last week: {_fmt_value(result.last_week_mean, result.decimals)} {result.unit}"
        if result.last_week_mean is not None else "Last week: N/A"
    )

    arrow_html = _direction_arrow_html(result)

    return (
        f'<div class="weekly-card">'
        f'<div class="status-bar" style="background:{status_color}"></div>'
        f'<div class="card-header">'
        f'<span class="metric-name">{result.display_name}</span>'
        f'<span class="traffic-light" style="background:{tl_color}" '
        f'title="{result.status}"></span>'
        f'</div>'
        f'<div class="metric-value">{val_str}'
        f'<span class="metric-unit">{result.unit}</span></div>'
        f'<div class="change-row">{arrow_html}'
        f'<span class="last-week-label">{lw_str}</span></div>'
        f'<div class="sparkline-box">{sparkline_html}</div>'
        f'</div>'
    )


def build_doctor_summary_html(bullets: list[str]) -> str:
    """Format the doctor summary as styled HTML."""
    items = []
    for b in bullets:
        cls = ' class="attention"' if b.startswith("ATTENTION") else ""
        items.append(f"<li{cls}>{b}</li>")

    return (
        f'<div class="doctor-summary">'
        f'<h3>Doctor Summary &mdash; Week of {THIS_WEEK_START} to {THIS_WEEK_END}</h3>'
        f'<ul>{"".join(items)}</ul>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def build_json(
    results: list[WeeklyResult],
    bullets: list[str],
    days_on_rux: int,
) -> dict:
    """Build structured JSON output."""
    metrics_json = {}
    for r in results:
        metrics_json[r.key] = {
            "display_name": r.display_name,
            "this_week": round(r.this_week_mean, r.decimals) if r.this_week_mean is not None else None,
            "last_week": round(r.last_week_mean, r.decimals) if r.last_week_mean is not None else None,
            "delta": round(r.delta, r.decimals + 1) if r.delta is not None else None,
            "pct_change": round(r.pct_change, 1) if r.pct_change is not None else None,
            "direction": r.direction,
            "status": r.status,
            "unit": r.unit,
            "n_this_week": r.n_this,
            "n_last_week": r.n_last,
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "week_ending": str(THIS_WEEK_END),
        "this_week_range": f"{THIS_WEEK_START} to {THIS_WEEK_END}",
        "last_week_range": f"{LAST_WEEK_START} to {LAST_WEEK_END}",
        "days_on_ruxolitinib": days_on_rux,
        "metrics": metrics_json,
        "doctor_summary": bullets,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Load data, compute weekly comparisons, generate HTML + JSON."""
    print("Weekly Trend Tracker — This Week vs Last Week")
    print("=" * 60)
    print(f"  This week:  {THIS_WEEK_START} to {THIS_WEEK_END}")
    print(f"  Last week:  {LAST_WEEK_START} to {LAST_WEEK_END}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Days on ruxolitinib
    days_on_rux = max(0, (TODAY - TREATMENT_START).days)
    print(f"  Days on rux: {days_on_rux}")

    # 1. Load data
    print("\n[1/5] Loading 14-day data window...")
    conn = safe_connect(DATABASE_PATH, read_only=True)
    try:
        tables = load_14day_data(conn)
        for tname, tdf in tables.items():
            print(f"  -> {tname}: {len(tdf)} rows")

        # 2. Compute weekly results
        print("\n[2/5] Computing weekly comparisons...")
        results: list[WeeklyResult] = []
        for m in METRICS:
            series = extract_series(tables, m)
            wr = compute_weekly(m, series)
            results.append(wr)
            arrow = {"improving": "+", "declining": "-", "stable": "=", "no_data": "?"}
            print(
                f"  {m.display_name:20s}  "
                f"this={_fmt_value(wr.this_week_mean, m.decimals):>8s}  "
                f"last={_fmt_value(wr.last_week_mean, m.decimals):>8s}  "
                f"{_fmt_pct(wr.pct_change):>8s}  "
                f"[{arrow.get(wr.direction, '?')}] {wr.status}"
            )

        # 3. Doctor summary
        print("\n[3/5] Generating doctor summary...")
        bullets = generate_doctor_summary(results, days_on_rux)
        for b in bullets:
            print(f"  - {b}")

        # 4. Build HTML
        print("\n[4/5] Building HTML report...")

        # KPI overview row (top-level summary)
        n_good = sum(1 for r in results if r.status == "good")
        n_warn = sum(1 for r in results if r.status == "warning")
        n_crit = sum(1 for r in results if r.status == "critical")
        n_improving = sum(1 for r in results if r.direction == "improving")
        n_declining = sum(1 for r in results if r.direction == "declining")

        overview_cards = make_kpi_row(
            make_kpi_card("GREEN", n_good, status="good", decimals=0,
                          detail="Metrics in normal range", status_label="On Track"),
            make_kpi_card("AMBER", n_warn, status="warning", decimals=0,
                          detail="Borderline or declining", status_label="Watch"),
            make_kpi_card("RED", n_crit, status="critical", decimals=0,
                          detail="Needs attention", status_label="Alert"),
            make_kpi_card("IMPROVING", n_improving, status="good", decimals=0,
                          detail="Week-over-week"),
            make_kpi_card("DECLINING", n_declining,
                          status="critical" if n_declining > 4 else "warning" if n_declining > 2 else "info",
                          decimals=0, detail="Week-over-week"),
        )

        body = overview_cards

        # Doctor summary
        body += build_doctor_summary_html(bullets)

        # Metric cards grid
        cards_html = ""
        for i, r in enumerate(results):
            m = METRICS[i]
            sparkline = section_html_or_placeholder(
                f"sparkline_{m.key}",
                make_sparkline,
                r.series_14d, m,
            )
            cards_html += build_metric_card(r, sparkline)

        body += make_section(
            "Metric Details",
            f'<div class="weekly-grid">{cards_html}</div>',
            section_id="metrics",
        )

        # Ruxolitinib section
        rux_html = section_html_or_placeholder(
            "Ruxolitinib Progress",
            build_rux_section,
            results, conn, days_on_rux,
        )
        body += make_section("Ruxolitinib Progress", rux_html, section_id="ruxolitinib")

        html_content = wrap_html(
            title="Weekly Tracker",
            body_content=body,
            report_id="weekly",
            subtitle=f"{THIS_WEEK_START} to {THIS_WEEK_END}",
            header_meta="Patient 1 \u2014 Weekly Tracker",
            extra_css=EXTRA_CSS,
        )
        HTML_OUTPUT.write_text(html_content, encoding="utf-8")
        print(f"  -> HTML: {HTML_OUTPUT}")
    finally:
        conn.close()

    # 5. Export JSON
    print("\n[5/5] Exporting JSON...")
    json_data = build_json(results, bullets, days_on_rux)
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> JSON: {JSON_OUTPUT}")

    # Final summary
    print("\n" + "=" * 60)
    print("WEEKLY TRACKER COMPLETE")
    print(f"  Green: {n_good}  Amber: {n_warn}  Red: {n_crit}")
    print(f"  Improving: {n_improving}  Declining: {n_declining}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
