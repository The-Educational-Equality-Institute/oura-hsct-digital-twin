"""Shared utilities for comparative analysis modules.

Provides patient configuration, data loading, normalization, alignment,
statistical comparison, and shared Plotly helpers for cross-patient reports.

All 5 comparative analysis scripts import from this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis._hardening import safe_connect, safe_read_sql
from profiles import PROFILES

try:
    from analysis._theme import (
        ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_CYAN,
        ACCENT_AMBER, ACCENT_RED, ACCENT_ORANGE, ACCENT_PINK,
        BG_PRIMARY, BG_SURFACE, BG_ELEVATED,
        TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
        BORDER_SUBTLE,
    )
except ImportError:
    ACCENT_BLUE = "#3B82F6"
    ACCENT_GREEN = "#10B981"
    ACCENT_PURPLE = "#8B5CF6"
    ACCENT_CYAN = "#06B6D4"
    ACCENT_AMBER = "#F59E0B"
    ACCENT_RED = "#EF4444"
    ACCENT_ORANGE = "#F97316"
    ACCENT_PINK = "#EC4899"
    BG_PRIMARY = "#0F1117"
    BG_SURFACE = "#1A1D27"
    BG_ELEVATED = "#242837"
    TEXT_PRIMARY = "#E8E8ED"
    TEXT_SECONDARY = "#9CA3AF"
    TEXT_TERTIARY = "#6B7280"
    BORDER_SUBTLE = "#2D3348"


# ---------------------------------------------------------------------------
# 1. Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatientConfig:
    patient_id: str
    display_name: str
    db_path: Path
    event_date: date
    event_label: str
    color: str


@dataclass
class NormalizedResult:
    patient_id: str
    raw: pd.Series
    z_scores: pd.Series
    percentiles: pd.Series
    baseline_mean: float
    baseline_std: float
    n_observations: int


# Metrics safe for cross-patient comparison
# (name, table, column, display_name, unit, higher_is_better)
COMPARABLE_METRICS = [
    ("hrv_avg", "oura_sleep_periods", "average_hrv", "HRV (RMSSD)", "ms", True),
    ("rhr", "oura_sleep_periods", "lowest_heart_rate", "Resting Heart Rate", "bpm", False),
    ("sleep_duration", "oura_sleep_periods", "total_sleep_duration", "Sleep Duration", "sec", True),
    ("deep_sleep", "oura_sleep_periods", "deep_sleep_duration", "Deep Sleep", "sec", True),
    ("rem_sleep", "oura_sleep_periods", "rem_sleep_duration", "REM Sleep", "sec", True),
    ("efficiency", "oura_sleep_periods", "efficiency", "Sleep Efficiency", "%", True),
    ("steps", "oura_activity", "steps", "Daily Steps", "steps", True),
    ("active_cal", "oura_activity", "active_calories", "Active Calories", "kcal", True),
    ("sleep_score", "oura_sleep", "score", "Sleep Score", "pts", True),
    ("readiness_score", "oura_readiness", "score", "Readiness Score", "pts", True),
    ("breath_avg", "oura_sleep_periods", "average_breath", "Breath Rate", "brpm", False),
    ("temp_delta", "oura_sleep", "temperature_delta", "Temp Deviation", "°C", None),
]


def default_patients() -> tuple[PatientConfig, PatientConfig]:
    """Build PatientConfig for Henrik and Mitch from profiles.py."""
    h = PROFILES["henrik"]
    m = PROFILES["mitch"]
    henrik = PatientConfig(
        patient_id="henrik",
        display_name=f"Henrik (post-{h['major_event_label']})",
        db_path=Path(h["database"]),
        event_date=h["major_event_date"],
        event_label=h["major_event_label"],
        color=ACCENT_BLUE,
    )
    mitch = PatientConfig(
        patient_id="mitch",
        display_name=f"Mitchell (post-{m['major_event_label']})",
        db_path=Path(m["database"]),
        event_date=m["major_event_date"],
        event_label=m["major_event_label"],
        color=ACCENT_GREEN,
    )
    return henrik, mitch


# ---------------------------------------------------------------------------
# 2. Data Loading
# ---------------------------------------------------------------------------

def load_patient_data(
    patient: PatientConfig,
    table: str,
    columns: str = "*",
    date_range: Optional[tuple[str, str]] = None,
) -> pd.DataFrame:
    """Load data from a patient's database, indexed by date."""
    conn = safe_connect(patient.db_path, read_only=True)
    try:
        date_col = "day" if table == "oura_sleep_periods" else "date"
        where = ""
        if date_range:
            where = f" WHERE {date_col} BETWEEN '{date_range[0]}' AND '{date_range[1]}'"

        # For sleep_periods, filter to long_sleep only
        if table == "oura_sleep_periods":
            if where:
                where += " AND type = 'long_sleep'"
            else:
                where = " WHERE type = 'long_sleep'"

        sql = f"SELECT {columns} FROM {table}{where} ORDER BY {date_col}"
        df = safe_read_sql(sql, conn, label=f"{patient.patient_id}/{table}")

        if df.empty:
            return df

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col).sort_index()
        # Remove duplicate dates, keep last
        df = df[~df.index.duplicated(keep="last")]
        return df
    finally:
        conn.close()


def load_both_patients(
    patients: Optional[tuple[PatientConfig, PatientConfig]] = None,
    table: str = "oura_sleep",
    columns: str = "*",
) -> dict[str, pd.DataFrame]:
    """Load from both patient databases."""
    if patients is None:
        patients = default_patients()
    return {
        p.patient_id: load_patient_data(p, table, columns)
        for p in patients
    }


def load_metric(
    metric_name: str,
    table: str,
    column: str,
    patients: Optional[tuple[PatientConfig, PatientConfig]] = None,
) -> dict[str, pd.Series]:
    """Load a single metric for both patients as Series."""
    if patients is None:
        patients = default_patients()
    result = {}
    date_col = "day" if table == "oura_sleep_periods" else "date"
    cols = f"{date_col}, {column}"
    for p in patients:
        df = load_patient_data(p, table, columns=cols)
        if not df.empty and column in df.columns:
            s = df[column].dropna()
            s.name = metric_name
            result[p.patient_id] = s
        else:
            result[p.patient_id] = pd.Series(dtype=float, name=metric_name)
    return result


# ---------------------------------------------------------------------------
# 3. Z-Score Normalization
# ---------------------------------------------------------------------------

def zscore_normalize(
    series: pd.Series,
    patient_id: str = "",
    baseline_period: Optional[tuple[str, str]] = None,
) -> NormalizedResult:
    """Z-score normalize relative to patient's own mean/std."""
    clean = series.dropna()
    if baseline_period:
        mask = (clean.index >= baseline_period[0]) & (clean.index <= baseline_period[1])
        baseline = clean[mask]
    else:
        baseline = clean

    mean = baseline.mean() if len(baseline) > 0 else 0.0
    std = baseline.std() if len(baseline) > 1 else 1.0
    if std == 0 or np.isnan(std):
        std = 1.0

    z = (series - mean) / std
    pct = series.rank(pct=True) * 100

    return NormalizedResult(
        patient_id=patient_id,
        raw=series,
        z_scores=z,
        percentiles=pct,
        baseline_mean=float(mean),
        baseline_std=float(std),
        n_observations=int(clean.count()),
    )


def zscore_both(
    data: dict[str, pd.Series],
    baseline_periods: Optional[dict[str, tuple[str, str]]] = None,
) -> dict[str, NormalizedResult]:
    """Apply z-score normalization to both patients."""
    result = {}
    for pid, series in data.items():
        bp = baseline_periods.get(pid) if baseline_periods else None
        result[pid] = zscore_normalize(series, patient_id=pid, baseline_period=bp)
    return result


# ---------------------------------------------------------------------------
# 4. Percentile-of-Self
# ---------------------------------------------------------------------------

def percentile_of_self(series: pd.Series) -> pd.Series:
    """Rank each day within the patient's own distribution (0-100)."""
    return series.rank(pct=True) * 100


def percentile_both(data: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Apply percentile-of-self to both patients."""
    return {pid: percentile_of_self(s) for pid, s in data.items()}


# ---------------------------------------------------------------------------
# 5. Overlap & Alignment
# ---------------------------------------------------------------------------

def find_date_overlap(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Find intersection of dates where both patients have data."""
    indices = [df.index for df in data.values() if not df.empty]
    if len(indices) < 2:
        return indices[0] if indices else pd.DatetimeIndex([])
    overlap = indices[0]
    for idx in indices[1:]:
        overlap = overlap.intersection(idx)
    return overlap


def align_to_overlap(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Filter both patients to overlapping dates only."""
    overlap = find_date_overlap(data)
    return {pid: df.loc[df.index.intersection(overlap)] for pid, df in data.items()}


def days_since_event(dates: pd.DatetimeIndex, event_date: date) -> pd.Series:
    """Convert date index to integer days-since-event."""
    event_dt = pd.Timestamp(event_date)
    return pd.Series(
        (dates - event_dt).days,
        index=dates,
        name="days_since_event",
    )


def align_by_event(
    data: dict[str, pd.Series],
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, pd.Series]:
    """Re-index both patients by days-since-event."""
    patient_map = {p.patient_id: p for p in patients}
    result = {}
    for pid, series in data.items():
        p = patient_map[pid]
        dse = days_since_event(series.index, p.event_date)
        aligned = series.copy()
        aligned.index = dse.values
        aligned = aligned.sort_index()
        result[pid] = aligned
    return result


# ---------------------------------------------------------------------------
# 6. Statistical Comparison
# ---------------------------------------------------------------------------

def effect_size_cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Compute Cohen's d with pooled standard deviation."""
    a_clean = a.dropna()
    b_clean = b.dropna()
    na, nb = len(a_clean), len(b_clean)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * a_clean.std() ** 2 + (nb - 1) * b_clean.std() ** 2)
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((a_clean.mean() - b_clean.mean()) / pooled_std)


def _effect_label(d: float) -> str:
    """Classify Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(
    a: pd.Series,
    b: pd.Series,
    func=None,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for difference in means (or custom func)."""
    a_clean = a.dropna().values
    b_clean = b.dropna().values
    if len(a_clean) < 2 or len(b_clean) < 2:
        return (np.nan, np.nan)

    if func is None:
        func = lambda x, y: np.mean(x) - np.mean(y)

    rng = np.random.default_rng(42)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        a_sample = rng.choice(a_clean, size=len(a_clean), replace=True)
        b_sample = rng.choice(b_clean, size=len(b_clean), replace=True)
        diffs[i] = func(a_sample, b_sample)

    alpha = (1 - ci) / 2
    return (float(np.percentile(diffs, alpha * 100)),
            float(np.percentile(diffs, (1 - alpha) * 100)))


def compare_distributions(
    a: pd.Series,
    b: pd.Series,
    test: str = "auto",
) -> dict:
    """Compare two distributions using Mann-Whitney U (default)."""
    a_clean = a.dropna()
    b_clean = b.dropna()

    if len(a_clean) < 3 or len(b_clean) < 3:
        return {
            "test_name": "insufficient_data",
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "effect_size": 0.0,
            "effect_label": "insufficient data",
            "ci_95": (np.nan, np.nan),
        }

    stat, p = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
    d = effect_size_cohens_d(a_clean, b_clean)
    ci = bootstrap_ci(a_clean, b_clean)

    return {
        "test_name": "Mann-Whitney U",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < 0.05,
        "effect_size": float(d),
        "effect_label": _effect_label(d),
        "ci_95": ci,
    }


# ---------------------------------------------------------------------------
# 7. Shared Plot Helpers
# ---------------------------------------------------------------------------

# Patient colors for consistent styling
PATIENT_COLORS = {
    "henrik": ACCENT_BLUE,
    "mitch": ACCENT_GREEN,
}


def _add_event_line(fig: go.Figure, patient: PatientConfig, y_range=None):
    """Add vertical event line using shape + annotation (never add_vline with annotation_text)."""
    event_ts = pd.Timestamp(patient.event_date)
    color = PATIENT_COLORS.get(patient.patient_id, ACCENT_PURPLE)

    fig.add_shape(
        type="line",
        x0=event_ts, x1=event_ts,
        y0=0, y1=1, yref="paper",
        line=dict(color=color, width=1.5, dash="dash"),
        opacity=0.6,
    )
    fig.add_annotation(
        x=event_ts, y=1.02, yref="paper",
        text=f"{patient.event_label}",
        showarrow=False,
        font=dict(size=10, color=color),
    )


def dual_patient_timeseries(
    data: dict[str, pd.Series],
    patients: tuple[PatientConfig, PatientConfig],
    title: str = "",
    y_label: str = "",
    show_rolling: int = 7,
    normalize: Optional[str] = None,
    event_lines: bool = True,
) -> go.Figure:
    """Plot both patients overlaid on a time axis."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    for pid, series in data.items():
        if series.empty:
            continue
        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)

        plot_data = series
        if normalize == "zscore":
            nr = zscore_normalize(series, patient_id=pid)
            plot_data = nr.z_scores
        elif normalize == "percentile":
            plot_data = percentile_of_self(series)

        # Raw scatter
        fig.add_trace(go.Scatter(
            x=plot_data.index, y=plot_data.values,
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.3),
            name=f"{p.display_name} (daily)",
            legendgroup=pid,
            showlegend=False,
        ))

        # Rolling mean
        if show_rolling and len(plot_data) >= show_rolling:
            rolling = plot_data.rolling(show_rolling, min_periods=max(1, show_rolling // 2)).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{p.display_name} ({show_rolling}d avg)",
                legendgroup=pid,
            ))

    # Event lines
    if event_lines:
        for p in patients:
            if p.patient_id in data and not data[p.patient_id].empty:
                _add_event_line(fig, p)

    y_title = y_label
    if normalize == "zscore":
        y_title = f"{y_label} (z-score)" if y_label else "z-score"
    elif normalize == "percentile":
        y_title = f"{y_label} (percentile)" if y_label else "percentile"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title=y_title,
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


def dual_patient_distribution(
    data: dict[str, pd.Series],
    patients: tuple[PatientConfig, PatientConfig],
    title: str = "",
    kind: str = "violin",
) -> go.Figure:
    """Side-by-side violin/box/histogram for both patients."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    for pid, series in data.items():
        if series.empty:
            continue
        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        clean = series.dropna()

        if kind == "violin":
            fig.add_trace(go.Violin(
                y=clean.values,
                name=p.display_name,
                marker_color=color,
                box_visible=True,
                meanline_visible=True,
                opacity=0.8,
            ))
        elif kind == "box":
            fig.add_trace(go.Box(
                y=clean.values,
                name=p.display_name,
                marker_color=color,
                boxmean="sd",
            ))
        else:  # histogram
            fig.add_trace(go.Histogram(
                x=clean.values,
                name=p.display_name,
                marker_color=color,
                opacity=0.7,
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=True,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def event_aligned_comparison(
    data: dict[str, pd.Series],
    patients: tuple[PatientConfig, PatientConfig],
    title: str = "",
    window: tuple[int, int] = (-30, 365),
    y_label: str = "",
    show_rolling: int = 7,
) -> go.Figure:
    """X-axis = days-since-event, both patients overlaid."""
    fig = go.Figure()
    aligned = align_by_event(data, patients)
    patient_map = {p.patient_id: p for p in patients}

    for pid, series in aligned.items():
        if series.empty:
            continue
        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)

        # Filter to window
        mask = (series.index >= window[0]) & (series.index <= window[1])
        windowed = series[mask]

        if windowed.empty:
            continue

        # Scatter
        fig.add_trace(go.Scatter(
            x=windowed.index, y=windowed.values,
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.3),
            name=f"{p.display_name} (daily)",
            legendgroup=pid,
            showlegend=False,
        ))

        # Rolling mean
        if show_rolling and len(windowed) >= show_rolling:
            rolling = windowed.sort_index().rolling(show_rolling, min_periods=max(1, show_rolling // 2)).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{p.display_name} ({show_rolling}d avg)",
                legendgroup=pid,
            ))

    # Event day vertical line at x=0
    fig.add_shape(
        type="line", x0=0, x1=0, y0=0, y1=1, yref="paper",
        line=dict(color=TEXT_SECONDARY, width=1.5, dash="dash"), opacity=0.6,
    )
    fig.add_annotation(
        x=0, y=1.02, yref="paper",
        text="Event Day", showarrow=False,
        font=dict(size=10, color=TEXT_SECONDARY),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Days Since Event",
        yaxis_title=y_label,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig
