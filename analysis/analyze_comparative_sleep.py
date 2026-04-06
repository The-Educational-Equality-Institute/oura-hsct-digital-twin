#!/usr/bin/env python3
"""Module 3: Sleep Architecture as Health Signal.

Compares deep/REM/light/awake percentages, efficiency, and timing between
Henrik (post-HSCT) and Mitchell (post-stroke) as markers of recovery quality.

Outputs:
  - Interactive HTML dashboard: reports/comparative_sleep_analysis.html
  - JSON metrics:               reports/comparative_sleep_metrics.json

Usage:
    python analysis/analyze_comparative_sleep.py
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

from config import REPORTS_DIR, FONT_FAMILY, TREATMENT_START
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
    BG_PRIMARY,
    BG_SURFACE,
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

HTML_OUTPUT = REPORTS_DIR / "comparative_sleep_analysis.html"
JSON_OUTPUT = REPORTS_DIR / "comparative_sleep_metrics.json"

# ---------------------------------------------------------------------------
# Population norms
# ---------------------------------------------------------------------------
NORMS = {
    "general": {
        "deep_pct": (13.0, 23.0),
        "rem_pct": (20.0, 25.0),
        "efficiency": (85.0, 100.0),
        "total_hours": (7.0, 9.0),
    },
    "post_hsct": {
        "deep_pct": (8.0, 15.0),
        "rem_pct": (10.0, 18.0),
        "efficiency": (70.0, 82.0),
    },
    "post_stroke": {
        "deep_pct": (10.0, 18.0),
        "rem_pct": (12.0, 20.0),
        "efficiency": (72.0, 85.0),
    },
}

# Stage display colors
STAGE_COLORS = {
    "deep": ACCENT_PURPLE,
    "rem": ACCENT_CYAN,
    "light": ACCENT_AMBER,
    "awake": ACCENT_RED,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    fig.add_shape(
        type="line",
        x0=x_val, x1=x_val,
        y0=0, y1=1, yref="paper",
        line=dict(color=color, width=1.5, dash="dash"),
        opacity=0.5,
        row=row, col=col,
    )
    fig.add_annotation(
        x=x_val, y=1.02, yref="paper",
        text=label,
        showarrow=False,
        font=dict(size=9, color=color),
        row=row, col=col,
    )


def _bedtime_to_hour(ts_str: str) -> float | None:
    """Convert ISO 8601 bedtime string to decimal hour, handling midnight crossing.

    23:30 -> 23.5, 00:30 -> 24.5, 01:15 -> 25.25
    """
    if not ts_str or pd.isna(ts_str):
        return None
    try:
        dt = pd.Timestamp(ts_str)
        hour = dt.hour + dt.minute / 60
        # Treat hours before 12 noon as next-day (past midnight)
        if hour < 12:
            hour += 24
        return hour
    except Exception:
        return None


def _hour_to_display(h: float) -> str:
    """Convert decimal hour (possibly > 24) to HH:MM display string."""
    h_mod = h % 24
    hh = int(h_mod)
    mm = int((h_mod - hh) * 60)
    return f"{hh:02d}:{mm:02d}"


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = 0
    less = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                more += 1
            elif xi < yi:
                less += 1
    return (more - less) / (n_x * n_y)


def _cliffs_delta_label(d: float) -> str:
    """Classify Cliff's delta magnitude."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def _compute_zscore(value: float, norm_range: tuple[float, float]) -> float:
    """Compute z-score relative to a norm range (midpoint, half-width as SD proxy)."""
    mid = (norm_range[0] + norm_range[1]) / 2
    sd = (norm_range[1] - norm_range[0]) / 2
    if sd == 0:
        return 0.0
    return (value - mid) / sd


# ---------------------------------------------------------------------------
# [1/7] Data Loading
# ---------------------------------------------------------------------------

def load_sleep_data(
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, pd.DataFrame]:
    """Load sleep architecture data for both patients from oura_sleep_periods.

    Returns dict keyed by patient_id, each a DataFrame with columns:
      deep_sec, rem_sec, light_sec, awake_sec, total_sec, efficiency,
      bedtime_start, bedtime_end, deep_pct, rem_pct, light_pct, awake_pct,
      total_hours
    """
    result: dict[str, pd.DataFrame] = {}

    for p in patients:
        cols = (
            "day, deep_sleep_duration, rem_sleep_duration, "
            "light_sleep_duration, awake_time, total_sleep_duration, "
            "efficiency, bedtime_start, bedtime_end"
        )
        df = load_patient_data(p, "oura_sleep_periods", columns=cols)

        if df.empty:
            result[p.patient_id] = pd.DataFrame()
            logger.warning("No sleep_periods data for %s", p.display_name)
            continue

        # Rename for clarity
        df = df.rename(columns={
            "deep_sleep_duration": "deep_sec",
            "rem_sleep_duration": "rem_sec",
            "light_sleep_duration": "light_sec",
            "awake_time": "awake_sec",
            "total_sleep_duration": "total_sec",
        })

        # Drop rows with missing duration data
        duration_cols = ["deep_sec", "rem_sec", "light_sec", "awake_sec", "total_sec"]
        for c in duration_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df.dropna(subset=["total_sec"])
        df = df[df["total_sec"] > 0]

        if df.empty:
            result[p.patient_id] = pd.DataFrame()
            continue

        # Compute architecture percentages: stage / (total_sleep + awake) * 100
        denominator = df["total_sec"] + df["awake_sec"].fillna(0)
        denominator = denominator.replace(0, np.nan)
        df["deep_pct"] = (df["deep_sec"] / denominator * 100).round(2)
        df["rem_pct"] = (df["rem_sec"] / denominator * 100).round(2)
        df["light_pct"] = (df["light_sec"] / denominator * 100).round(2)
        df["awake_pct"] = (df["awake_sec"] / denominator * 100).round(2)
        df["total_hours"] = (df["total_sec"] / 3600).round(2)

        # Compute bedtime/wake hours
        if "bedtime_start" in df.columns:
            df["bedtime_hour"] = df["bedtime_start"].apply(_bedtime_to_hour)
        if "bedtime_end" in df.columns:
            df["wake_hour"] = df["bedtime_end"].apply(
                lambda x: pd.Timestamp(x).hour + pd.Timestamp(x).minute / 60
                if pd.notna(x) else None
            )
        # Sleep midpoint
        if "bedtime_hour" in df.columns and "wake_hour" in df.columns:
            bt = df["bedtime_hour"]
            wk = df["wake_hour"]
            # Wake hour needs midnight adjustment if bedtime > 24
            adjusted_wake = wk.copy()
            mask = bt > 24
            adjusted_wake[mask & (wk < 12)] = wk[mask & (wk < 12)] + 24
            # For non-midnight-crossing cases
            adjusted_wake[~mask & (wk < bt)] = wk[~mask & (wk < bt)] + 24
            df["midpoint_hour"] = ((bt + adjusted_wake) / 2).round(2)

        # Weekday flag (0=Mon ... 6=Sun)
        df["weekday"] = df.index.dayofweek
        df["is_weekend"] = df["weekday"].isin([4, 5])  # Fri/Sat nights

        result[p.patient_id] = df
        logger.info(
            "Loaded %s: %d nights, avg %.1f hrs, efficiency %.0f%%",
            p.display_name, len(df),
            df["total_hours"].mean(),
            df["efficiency"].mean() if "efficiency" in df.columns else 0,
        )

    return result


def load_readiness_sleep_balance(
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, pd.Series]:
    """Load sleep_balance score from oura_readiness."""
    result: dict[str, pd.Series] = {}
    for p in patients:
        df = load_patient_data(p, "oura_readiness", columns="date, sleep_balance")
        if not df.empty and "sleep_balance" in df.columns:
            s = df["sleep_balance"].dropna()
            s.name = "sleep_balance"
            result[p.patient_id] = s
        else:
            result[p.patient_id] = pd.Series(dtype=float, name="sleep_balance")
    return result


# ---------------------------------------------------------------------------
# [2/7] Architecture Statistics
# ---------------------------------------------------------------------------

def compute_architecture_stats(
    data: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Compute per-patient architecture summary stats."""
    result: dict[str, dict[str, Any]] = {}

    for pid, df in data.items():
        if df.empty:
            result[pid] = {}
            continue

        stages = {}
        for stage in ["deep_pct", "rem_pct", "light_pct", "awake_pct"]:
            s = df[stage].dropna()
            stages[stage] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
                "min": float(s.min()),
                "max": float(s.max()),
            }

        # Total hours
        th = df["total_hours"].dropna()
        stages["total_hours"] = {
            "mean": float(th.mean()),
            "median": float(th.median()),
            "std": float(th.std()),
        }

        # Weekday vs weekend
        wd_mask = ~df["is_weekend"]
        we_mask = df["is_weekend"]
        weekday_weekend = {}
        for stage in ["deep_pct", "rem_pct", "total_hours", "efficiency"]:
            col = df[stage].dropna() if stage in df.columns else pd.Series(dtype=float)
            wd = col[wd_mask.reindex(col.index, fill_value=False)]
            we = col[we_mask.reindex(col.index, fill_value=False)]
            weekday_weekend[stage] = {
                "weekday_mean": float(wd.mean()) if len(wd) > 0 else None,
                "weekend_mean": float(we.mean()) if len(we) > 0 else None,
            }

        # Temporal evolution: first half vs second half
        n = len(df)
        mid = n // 2
        first_half = df.iloc[:mid]
        second_half = df.iloc[mid:]
        temporal = {}
        for stage in ["deep_pct", "rem_pct", "efficiency"]:
            if stage in df.columns:
                fh = first_half[stage].dropna()
                sh = second_half[stage].dropna()
                temporal[stage] = {
                    "first_half_mean": float(fh.mean()) if len(fh) > 0 else None,
                    "second_half_mean": float(sh.mean()) if len(sh) > 0 else None,
                }

        result[pid] = {
            "stages": stages,
            "weekday_weekend": weekday_weekend,
            "temporal": temporal,
            "n_nights": n,
        }

    return result


# ---------------------------------------------------------------------------
# [3/7] Efficiency Analysis
# ---------------------------------------------------------------------------

def compute_efficiency_stats(
    data: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Compute efficiency trends, distribution stats, and classification."""
    result: dict[str, dict[str, Any]] = {}

    for pid, df in data.items():
        if df.empty or "efficiency" not in df.columns:
            result[pid] = {}
            continue

        eff = df["efficiency"].dropna()
        n = len(eff)
        if n < 3:
            result[pid] = {"mean": float(eff.mean()), "n": n}
            continue

        # Distribution stats
        pct_below_75 = float((eff < 75).sum() / n * 100)
        pct_below_85 = float((eff < 85).sum() / n * 100)
        pct_above_90 = float((eff >= 90).sum() / n * 100)

        # Linear regression trend
        x = np.arange(n, dtype=float)
        y = eff.values.astype(float)
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
        slope_per_week = slope * 7

        # Spearman correlation with time
        rho, rho_p = scipy_stats.spearmanr(x, y)

        # Skewness
        skew = float(scipy_stats.skew(y))

        result[pid] = {
            "mean": float(eff.mean()),
            "median": float(eff.median()),
            "std": float(eff.std()),
            "pct_below_75": pct_below_75,
            "pct_below_85": pct_below_85,
            "pct_above_90": pct_above_90,
            "trend_slope_per_week": float(slope_per_week),
            "trend_r_squared": float(r_value ** 2),
            "trend_p_value": float(p_value),
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "skewness": skew,
            "n": n,
            "regression": {
                "slope": float(slope),
                "intercept": float(intercept),
            },
        }

    return result


# ---------------------------------------------------------------------------
# [4/7] Timing Analysis
# ---------------------------------------------------------------------------

def compute_timing_stats(
    data: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Compute bedtime/wake consistency and social jet lag metrics."""
    result: dict[str, dict[str, Any]] = {}

    for pid, df in data.items():
        if df.empty:
            result[pid] = {}
            continue

        bt = df["bedtime_hour"].dropna() if "bedtime_hour" in df.columns else pd.Series(dtype=float)
        wk = df["wake_hour"].dropna() if "wake_hour" in df.columns else pd.Series(dtype=float)
        mp = df["midpoint_hour"].dropna() if "midpoint_hour" in df.columns else pd.Series(dtype=float)

        bt_mean = float(bt.mean()) if len(bt) > 0 else None
        bt_sd_min = float(bt.std() * 60) if len(bt) > 1 else None
        wk_mean = float(wk.mean()) if len(wk) > 0 else None
        wk_sd_min = float(wk.std() * 60) if len(wk) > 1 else None

        # Social jet lag: weekday vs weekend midpoint difference
        wd_mask = ~df["is_weekend"]
        we_mask = df["is_weekend"]
        sjl_min = None
        if "midpoint_hour" in df.columns:
            wd_mp = df.loc[wd_mask, "midpoint_hour"].dropna()
            we_mp = df.loc[we_mask, "midpoint_hour"].dropna()
            if len(wd_mp) > 0 and len(we_mp) > 0:
                sjl_min = float(abs(we_mp.mean() - wd_mp.mean()) * 60)

        # Circadian regularity: CV of sleep midpoint
        mp_cv = None
        if len(mp) > 1:
            mp_mean = mp.mean()
            if mp_mean != 0:
                mp_cv = float(mp.std() / mp_mean * 100)

        result[pid] = {
            "bedtime_mean_hour": bt_mean,
            "bedtime_sd_min": bt_sd_min,
            "wake_mean_hour": wk_mean,
            "wake_sd_min": wk_sd_min,
            "social_jetlag_min": sjl_min,
            "midpoint_cv_pct": mp_cv,
        }

    return result


# ---------------------------------------------------------------------------
# [5/7] Inter-Patient Comparison
# ---------------------------------------------------------------------------

def compute_inter_patient_comparison(
    data: dict[str, pd.DataFrame],
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, dict[str, Any]]:
    """Mann-Whitney U, Cohen's d, Cliff's delta, bootstrap CI for key metrics."""
    pids = [p.patient_id for p in patients]
    if len(pids) < 2 or pids[0] not in data or pids[1] not in data:
        return {}

    df_h = data[pids[0]]
    df_m = data[pids[1]]

    if df_h.empty or df_m.empty:
        return {}

    metrics_to_compare = [
        ("deep_pct", "Deep Sleep %"),
        ("rem_pct", "REM Sleep %"),
        ("light_pct", "Light Sleep %"),
        ("awake_pct", "Awake %"),
        ("efficiency", "Efficiency"),
        ("total_hours", "Total Hours"),
    ]

    # Add timing if available
    if "bedtime_hour" in df_h.columns and "bedtime_hour" in df_m.columns:
        metrics_to_compare.append(("bedtime_hour", "Bedtime Hour"))

    comparison: dict[str, dict[str, Any]] = {}
    for col, label in metrics_to_compare:
        h_vals = df_h[col].dropna() if col in df_h.columns else pd.Series(dtype=float)
        m_vals = df_m[col].dropna() if col in df_m.columns else pd.Series(dtype=float)

        if len(h_vals) < 3 or len(m_vals) < 3:
            comparison[col] = {"label": label, "insufficient_data": True}
            continue

        # Mann-Whitney U
        stat, p = scipy_stats.mannwhitneyu(h_vals, m_vals, alternative="two-sided")

        # Cohen's d
        d = effect_size_cohens_d(h_vals, m_vals)

        # Cliff's delta
        cd = _cliffs_delta(h_vals.values, m_vals.values)

        # Bootstrap 95% CI for median difference
        ci = bootstrap_ci(
            h_vals, m_vals,
            func=lambda x, y: np.median(x) - np.median(y),
            n_bootstrap=5000,
        )

        comparison[col] = {
            "label": label,
            "henrik_mean": float(h_vals.mean()),
            "mitchell_mean": float(m_vals.mean()),
            "mann_whitney_U": float(stat),
            "p_value": float(p),
            "significant": p < 0.05,
            "cohens_d": float(d),
            "cohens_d_label": _cohens_d_label(d),
            "cliffs_delta": float(cd),
            "cliffs_delta_label": _cliffs_delta_label(cd),
            "bootstrap_ci_95": (float(ci[0]), float(ci[1])),
        }

    return comparison


def _cohens_d_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# [5b/7] Benchmark Comparison
# ---------------------------------------------------------------------------

def compute_benchmarks(
    arch_stats: dict[str, dict[str, Any]],
    eff_stats: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, dict[str, Any]]:
    """Z-scores relative to population norms, post-HSCT norms, post-stroke norms."""
    result: dict[str, dict[str, Any]] = {}

    norm_map = {
        "henrik": "post_hsct",
        "mitch": "post_stroke",
    }

    for p in patients:
        pid = p.patient_id
        a = arch_stats.get(pid, {})
        e = eff_stats.get(pid, {})
        stages = a.get("stages", {})

        deep_mean = stages.get("deep_pct", {}).get("mean", 0)
        rem_mean = stages.get("rem_pct", {}).get("mean", 0)
        eff_mean = e.get("mean", 0)
        total_mean = stages.get("total_hours", {}).get("mean", 0)

        benchmarks: dict[str, Any] = {}

        # General population z-scores
        benchmarks["general"] = {
            "deep_pct_z": _compute_zscore(deep_mean, NORMS["general"]["deep_pct"]),
            "rem_pct_z": _compute_zscore(rem_mean, NORMS["general"]["rem_pct"]),
            "efficiency_z": _compute_zscore(eff_mean, NORMS["general"]["efficiency"]),
            "total_hours_z": _compute_zscore(total_mean, NORMS["general"]["total_hours"]),
        }

        # Condition-specific norms
        cond_key = norm_map.get(pid, "general")
        cond_norms = NORMS.get(cond_key, {})
        cond_z: dict[str, float] = {}
        if "deep_pct" in cond_norms:
            cond_z["deep_pct_z"] = _compute_zscore(deep_mean, cond_norms["deep_pct"])
        if "rem_pct" in cond_norms:
            cond_z["rem_pct_z"] = _compute_zscore(rem_mean, cond_norms["rem_pct"])
        if "efficiency" in cond_norms:
            cond_z["efficiency_z"] = _compute_zscore(eff_mean, cond_norms["efficiency"])
        benchmarks[cond_key] = cond_z

        result[pid] = benchmarks

    return result


# ---------------------------------------------------------------------------
# [5c/7] Recovery Quality Indicators
# ---------------------------------------------------------------------------

def compute_recovery_indicators(
    data: dict[str, pd.DataFrame],
    eff_stats: dict[str, dict[str, Any]],
    sleep_balance: dict[str, pd.Series],
) -> dict[str, dict[str, Any]]:
    """REM rebound, deep adequacy, sleep debt, efficiency trend, architecture stability."""
    result: dict[str, dict[str, Any]] = {}

    for pid, df in data.items():
        if df.empty:
            result[pid] = {}
            continue

        indicators: dict[str, Any] = {}

        # REM rebound: nights where REM% > mean+1SD following low-REM nights
        rem = df["rem_pct"].dropna()
        if len(rem) > 3:
            rem_mean = rem.mean()
            rem_sd = rem.std()
            low_rem_mask = rem < (rem_mean - rem_sd)
            rebound_count = 0
            total_low = 0
            for i in range(len(rem) - 1):
                if low_rem_mask.iloc[i]:
                    total_low += 1
                    if rem.iloc[i + 1] > (rem_mean + rem_sd):
                        rebound_count += 1
            indicators["rem_rebound_rate"] = (
                float(rebound_count / total_low * 100) if total_low > 0 else 0.0
            )
            indicators["rem_rebound_events"] = rebound_count
        else:
            indicators["rem_rebound_rate"] = 0.0
            indicators["rem_rebound_events"] = 0

        # Deep sleep adequacy: % of nights meeting 13% minimum
        deep = df["deep_pct"].dropna()
        if len(deep) > 0:
            indicators["deep_adequacy_pct"] = float(
                (deep >= 13.0).sum() / len(deep) * 100
            )
        else:
            indicators["deep_adequacy_pct"] = 0.0

        # Sleep debt: cumulative deviation from 7h target
        th = df["total_hours"].dropna()
        if len(th) > 0:
            daily_debt = th - 7.0
            indicators["cumulative_debt_hours"] = float(daily_debt.sum())
            indicators["avg_daily_debt_min"] = float(daily_debt.mean() * 60)
        else:
            indicators["cumulative_debt_hours"] = 0.0
            indicators["avg_daily_debt_min"] = 0.0

        # Efficiency trend (from eff_stats)
        e = eff_stats.get(pid, {})
        indicators["efficiency_trend_rho"] = e.get("spearman_rho", None)
        indicators["efficiency_trend_p"] = e.get("spearman_p", None)
        indicators["efficiency_trend_slope_per_week"] = e.get("trend_slope_per_week", None)

        # Architecture stability: CV per stage %
        stability = {}
        for stage in ["deep_pct", "rem_pct", "light_pct", "awake_pct"]:
            s = df[stage].dropna()
            if len(s) > 1 and s.mean() != 0:
                stability[stage] = float(s.std() / s.mean() * 100)
            else:
                stability[stage] = None
        indicators["architecture_cv"] = stability

        # Sleep balance score
        sb = sleep_balance.get(pid, pd.Series(dtype=float))
        if not sb.empty:
            indicators["sleep_balance_mean"] = float(sb.mean())
            indicators["sleep_balance_latest"] = float(sb.iloc[-1])
        else:
            indicators["sleep_balance_mean"] = None
            indicators["sleep_balance_latest"] = None

        result[pid] = indicators

    return result


# ---------------------------------------------------------------------------
# Visualization 1: Stacked Area
# ---------------------------------------------------------------------------

def _fig_stacked_area(
    data: dict[str, pd.DataFrame],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Two-panel stacked area of nightly architecture percentages."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[p.display_name for p in patients],
    )

    stages = [
        ("deep_pct", "Deep", STAGE_COLORS["deep"]),
        ("rem_pct", "REM", STAGE_COLORS["rem"]),
        ("light_pct", "Light", STAGE_COLORS["light"]),
        ("awake_pct", "Awake", STAGE_COLORS["awake"]),
    ]

    for row_idx, p in enumerate(patients, 1):
        df = data.get(p.patient_id, pd.DataFrame())
        if df.empty:
            continue

        for col_name, label, color in stages:
            vals = df[col_name].fillna(0)
            # 7-day rolling for smoother area
            if len(vals) >= 7:
                smoothed = vals.rolling(7, min_periods=3).mean()
            else:
                smoothed = vals

            fig.add_trace(go.Scatter(
                x=smoothed.index,
                y=smoothed.values,
                mode="lines",
                name=label,
                stackgroup=f"stack_{row_idx}",
                fillcolor=color.replace(")", ",0.6)").replace("rgb", "rgba") if "rgb" in color else color,
                line=dict(width=0.5, color=color),
                legendgroup=label,
                showlegend=(row_idx == 1),
            ), row=row_idx, col=1)

        # Norm band for deep sleep (general: 13-23%)
        fig.add_hrect(
            y0=13, y1=23,
            fillcolor=STAGE_COLORS["deep"], opacity=0.05,
            line_width=0,
            row=row_idx, col=1,
        )

        # Rux start line for Henrik
        if p.patient_id == "henrik":
            _add_event_vline(fig, pd.Timestamp(TREATMENT_START), "Rux Start", ACCENT_CYAN, row=row_idx, col=1)

    fig.update_yaxes(title_text="% of Total", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="% of Total", range=[0, 100], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_layout(
        height=700,
        title=dict(text="Sleep Architecture Over Time (7-day Rolling)", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=60, r=20, t=60, b=50),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Visualization 2: Architecture Distributions (2x2 violin grid)
# ---------------------------------------------------------------------------

def _fig_arch_distributions(
    data: dict[str, pd.DataFrame],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """2x2 grid of overlapping violin plots per sleep stage."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Deep Sleep %", "REM Sleep %", "Light Sleep %", "Awake %"],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    stages = [
        ("deep_pct", 1, 1, NORMS["general"]["deep_pct"]),
        ("rem_pct", 1, 2, NORMS["general"]["rem_pct"]),
        ("light_pct", 2, 1, None),
        ("awake_pct", 2, 2, None),
    ]

    for stage_col, row, col, norm_range in stages:
        for p in patients:
            pid = p.patient_id
            df = data.get(pid, pd.DataFrame())
            if df.empty or stage_col not in df.columns:
                continue

            vals = df[stage_col].dropna()
            color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)

            fig.add_trace(go.Violin(
                y=vals.values,
                name=p.display_name,
                marker_color=color,
                box_visible=True,
                meanline_visible=True,
                opacity=0.7,
                legendgroup=pid,
                showlegend=(row == 1 and col == 1),
                scalegroup=f"{stage_col}",
            ), row=row, col=col)

        # Norm range band
        if norm_range is not None:
            fig.add_hrect(
                y0=norm_range[0], y1=norm_range[1],
                fillcolor=ACCENT_AMBER, opacity=0.08,
                line_width=0,
                row=row, col=col,
            )

    fig.update_layout(
        height=600,
        title=dict(text="Sleep Architecture Distributions", font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Visualization 3: Efficiency Comparison
# ---------------------------------------------------------------------------

def _fig_efficiency(
    data: dict[str, pd.DataFrame],
    patients: tuple[PatientConfig, PatientConfig],
    eff_stats: dict[str, dict[str, Any]],
) -> go.Figure:
    """Dual line chart (nightly + rolling) + overlapping KDE."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15,
        subplot_titles=["Nightly Efficiency + 7-Day Rolling", "Efficiency Distribution"],
        row_heights=[0.6, 0.4],
    )

    for p in patients:
        pid = p.patient_id
        df = data.get(pid, pd.DataFrame())
        if df.empty or "efficiency" not in df.columns:
            continue

        eff = df["efficiency"].dropna()
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)

        # Top: nightly scatter + rolling
        fig.add_trace(go.Scatter(
            x=eff.index, y=eff.values,
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.3),
            name=f"{p.display_name} (nightly)",
            legendgroup=pid,
            showlegend=False,
        ), row=1, col=1)

        if len(eff) >= 7:
            rolling = eff.rolling(7, min_periods=4).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{p.display_name} (7d avg)",
                legendgroup=pid,
            ), row=1, col=1)

        # Bottom: KDE
        if len(eff) >= 5:
            kde_x = np.linspace(
                max(40, eff.min() - 5),
                min(100, eff.max() + 5),
                200,
            )
            kde = scipy_stats.gaussian_kde(eff.values)
            kde_y = kde(kde_x)
            fig.add_trace(go.Scatter(
                x=kde_x, y=kde_y,
                mode="lines",
                fill="tozeroy",
                line=dict(color=color, width=2),
                fillcolor=color.replace(")", ",0.2)").replace("rgb", "rgba") if "rgb" in color else color,
                name=f"{p.display_name}",
                legendgroup=pid,
                showlegend=False,
                opacity=0.7,
            ), row=2, col=1)

    # Reference lines at 75% and 85%
    _add_reference_line(fig, 75, "Poor (75%)", ACCENT_RED, row=1, col=1)
    _add_reference_line(fig, 85, "Recommended (85%)", ACCENT_GREEN, row=1, col=1)

    # Rux start for Henrik
    _add_event_vline(fig, pd.Timestamp(TREATMENT_START), "Rux Start", ACCENT_CYAN, row=1, col=1)

    fig.update_yaxes(title_text="Efficiency %", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Efficiency %", row=2, col=1)

    fig.update_layout(
        height=650,
        title=dict(text="Sleep Efficiency Comparison", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=60, r=20, t=60, b=50),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Visualization 4: Timing Analysis
# ---------------------------------------------------------------------------

def _fig_timing(
    data: dict[str, pd.DataFrame],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Scatter of bedtime hour over time + box plots of midpoint variability."""
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=["Bedtime Over Time", "Sleep Midpoint Variability"],
        horizontal_spacing=0.1,
    )

    for p in patients:
        pid = p.patient_id
        df = data.get(pid, pd.DataFrame())
        if df.empty:
            continue

        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)

        # Left: bedtime scatter over time
        if "bedtime_hour" in df.columns:
            bt = df["bedtime_hour"].dropna()
            fig.add_trace(go.Scatter(
                x=bt.index, y=bt.values,
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.5),
                name=p.display_name,
                legendgroup=pid,
            ), row=1, col=1)

            # Rolling average
            if len(bt) >= 7:
                bt_roll = bt.rolling(7, min_periods=4).mean()
                fig.add_trace(go.Scatter(
                    x=bt_roll.index, y=bt_roll.values,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{p.display_name} (7d avg)",
                    legendgroup=pid,
                    showlegend=False,
                ), row=1, col=1)

        # Right: box plot of midpoint variability
        if "midpoint_hour" in df.columns:
            mp = df["midpoint_hour"].dropna()
            fig.add_trace(go.Box(
                y=mp.values,
                name=p.display_name,
                marker_color=color,
                boxmean="sd",
                legendgroup=pid,
                showlegend=False,
            ), row=1, col=2)

    # Y-axis formatting for bedtime hours
    fig.update_yaxes(
        title_text="Hour (24h, >24 = past midnight)",
        row=1, col=1,
        tickvals=[22, 23, 24, 25, 26, 27],
        ticktext=["22:00", "23:00", "00:00", "01:00", "02:00", "03:00"],
    )
    fig.update_yaxes(title_text="Midpoint Hour", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=1, col=1)

    fig.update_layout(
        height=450,
        title=dict(text="Sleep Timing Analysis", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Visualization 5: Benchmark Radar
# ---------------------------------------------------------------------------

def _fig_benchmark_radar(
    arch_stats: dict[str, dict[str, Any]],
    eff_stats: dict[str, dict[str, Any]],
    sleep_balance: dict[str, pd.Series],
    timing_stats: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """6-axis radar: deep%, REM%, efficiency, total hours, regularity, sleep balance."""
    categories = [
        "Deep Sleep %", "REM Sleep %", "Efficiency",
        "Total Hours", "Regularity", "Sleep Balance",
    ]

    # Population norm targets (outer ring)
    norm_vals = [
        (NORMS["general"]["deep_pct"][0] + NORMS["general"]["deep_pct"][1]) / 2,  # 18
        (NORMS["general"]["rem_pct"][0] + NORMS["general"]["rem_pct"][1]) / 2,    # 22.5
        (NORMS["general"]["efficiency"][0] + NORMS["general"]["efficiency"][1]) / 2,  # 92.5
        (NORMS["general"]["total_hours"][0] + NORMS["general"]["total_hours"][1]) / 2,  # 8
        90,   # Regularity target (low CV)
        80,   # Sleep balance target
    ]

    # Normalize all to 0-100 scale relative to norms
    def _normalize_to_scale(val: float, norm_target: float) -> float:
        if norm_target == 0:
            return 0
        return min(120, max(0, val / norm_target * 100))

    fig = go.Figure()

    # Norm ring
    norm_scaled = [100] * len(categories)  # The norm ring is the 100% ring
    fig.add_trace(go.Scatterpolar(
        r=norm_scaled + [norm_scaled[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(245,158,11,0.08)",
        line=dict(color=ACCENT_AMBER, width=1.5, dash="dash"),
        name="Population Norms",
        opacity=0.6,
    ))

    for p in patients:
        pid = p.patient_id
        a = arch_stats.get(pid, {})
        e = eff_stats.get(pid, {})
        t = timing_stats.get(pid, {})
        sb = sleep_balance.get(pid, pd.Series(dtype=float))
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)

        stages = a.get("stages", {})
        deep_val = stages.get("deep_pct", {}).get("mean", 0)
        rem_val = stages.get("rem_pct", {}).get("mean", 0)
        eff_val = e.get("mean", 0)
        total_val = stages.get("total_hours", {}).get("mean", 0)

        # Regularity: invert CV of midpoint (lower CV = better regularity)
        mp_cv = t.get("midpoint_cv_pct", 5)
        if mp_cv is not None and mp_cv > 0:
            regularity = max(0, 100 - mp_cv * 10)
        else:
            regularity = 50

        sb_val = float(sb.mean()) if not sb.empty else 50

        raw_vals = [deep_val, rem_val, eff_val, total_val, regularity, sb_val]
        scaled = [_normalize_to_scale(v, n) for v, n in zip(raw_vals, norm_vals)]

        fig.add_trace(go.Scatterpolar(
            r=scaled + [scaled[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(color)},0.15)",
            line=dict(color=color, width=2),
            name=p.display_name,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 120],
                tickvals=[25, 50, 75, 100],
                ticktext=["25%", "50%", "75%", "100%"],
            ),
        ),
        height=500,
        title=dict(text="Sleep Health Benchmark Radar", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB string for rgba()."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return f"{int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"
    return "128,128,128"


# ---------------------------------------------------------------------------
# Visualization 6: Recovery Trajectory
# ---------------------------------------------------------------------------

def _fig_recovery_trajectory(
    data: dict[str, pd.DataFrame],
    eff_stats: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Efficiency trend with linear fit, annotated slope/R^2/Spearman."""
    fig = go.Figure()

    for p in patients:
        pid = p.patient_id
        df = data.get(pid, pd.DataFrame())
        if df.empty or "efficiency" not in df.columns:
            continue

        eff = df["efficiency"].dropna()
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        e = eff_stats.get(pid, {})

        # Scatter
        fig.add_trace(go.Scatter(
            x=eff.index, y=eff.values,
            mode="markers",
            marker=dict(size=4, color=color, opacity=0.4),
            name=f"{p.display_name} (nightly)",
            legendgroup=pid,
            showlegend=False,
        ))

        # 7-day rolling
        if len(eff) >= 7:
            rolling = eff.rolling(7, min_periods=4).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{p.display_name} (7d avg)",
                legendgroup=pid,
            ))

        # Linear fit
        reg = e.get("regression", {})
        slope = reg.get("slope", None)
        intercept = reg.get("intercept", None)
        if slope is not None and intercept is not None and len(eff) >= 5:
            x_num = np.arange(len(eff))
            fit_y = intercept + slope * x_num
            fig.add_trace(go.Scatter(
                x=eff.index, y=fit_y,
                mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                name=f"{p.display_name} (trend)",
                legendgroup=pid,
                showlegend=False,
            ))

            # Annotation with slope, R^2, Spearman
            r2 = e.get("trend_r_squared", 0)
            rho = e.get("spearman_rho", 0)
            slope_wk = e.get("trend_slope_per_week", 0)
            annotation_text = (
                f"{p.display_name}<br>"
                f"Slope: {slope_wk:+.2f}%/wk, R\u00b2={r2:.3f}<br>"
                f"Spearman \u03c1={rho:.3f}"
            )
            # Place annotation at last data point
            fig.add_annotation(
                x=eff.index[-1],
                y=fit_y[-1],
                text=annotation_text,
                showarrow=True,
                arrowhead=2,
                ax=40 if pid == "henrik" else -40,
                ay=-40,
                font=dict(size=10, color=color),
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
                bgcolor=BG_SURFACE,
                opacity=0.9,
            )

    # Henrik: Rux start
    _add_event_vline(fig, pd.Timestamp(TREATMENT_START), "Rux Start", ACCENT_CYAN)

    # Reference lines
    _add_reference_line(fig, 75, "Poor (75%)", ACCENT_RED)
    _add_reference_line(fig, 85, "Recommended (85%)", ACCENT_GREEN)

    fig.update_layout(
        height=500,
        title=dict(text="Recovery Trajectory: Efficiency Trend", font=dict(size=16)),
        yaxis_title="Sleep Efficiency %",
        xaxis_title="Date",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=60, r=20, t=60, b=60),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# HTML Assembly
# ---------------------------------------------------------------------------

def _build_stat_comparison_table(comparison: dict[str, dict[str, Any]]) -> str:
    """Build HTML table of all statistical comparisons."""
    if not comparison:
        return "<p>Insufficient data for statistical comparison.</p>"

    rows_html = ""
    for metric_key, info in comparison.items():
        if info.get("insufficient_data"):
            continue
        label = info.get("label", metric_key)
        h_mean = info.get("henrik_mean", 0)
        m_mean = info.get("mitchell_mean", 0)
        p_val = info.get("p_value", np.nan)
        sig = info.get("significant", False)
        cd = info.get("cohens_d", 0)
        cd_label = info.get("cohens_d_label", "")
        cliff = info.get("cliffs_delta", 0)
        cliff_label = info.get("cliffs_delta_label", "")
        ci = info.get("bootstrap_ci_95", (np.nan, np.nan))

        sig_indicator = "***" if sig and p_val < 0.001 else ("**" if sig and p_val < 0.01 else ("*" if sig else ""))
        sig_style = f'style="color:{ACCENT_GREEN};font-weight:600"' if sig else ""

        rows_html += f"""
        <tr>
            <td>{label}</td>
            <td>{h_mean:.1f}</td>
            <td>{m_mean:.1f}</td>
            <td {sig_style}>{format_p_value(p_val)} {sig_indicator}</td>
            <td>{cd:+.2f} ({cd_label})</td>
            <td>{cliff:+.2f} ({cliff_label})</td>
            <td>[{ci[0]:+.1f}, {ci[1]:+.1f}]</td>
        </tr>"""

    return f"""
    <div style="overflow-x:auto">
    <table class="odt-table" style="width:100%;font-size:0.85rem">
    <thead>
        <tr>
            <th>Metric</th>
            <th>Henrik</th>
            <th>Mitchell</th>
            <th>p-value</th>
            <th>Cohen's d</th>
            <th>Cliff's \u0394</th>
            <th>95% CI (median diff)</th>
        </tr>
    </thead>
    <tbody>{rows_html}
    </tbody>
    </table>
    </div>
    <p style="font-size:0.75rem;color:{TEXT_TERTIARY};margin-top:8px">
    * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001 &middot;
    CI = bootstrap 95% confidence interval for median difference (Henrik - Mitchell)
    </p>"""


def _build_benchmark_table(
    benchmarks: dict[str, dict[str, Any]],
    arch_stats: dict[str, dict[str, Any]],
    eff_stats: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Build HTML table comparing patients against population norms."""
    rows_html = ""

    metrics = [
        ("deep_pct", "Deep Sleep %", NORMS["general"]["deep_pct"]),
        ("rem_pct", "REM Sleep %", NORMS["general"]["rem_pct"]),
        ("efficiency", "Efficiency %", NORMS["general"]["efficiency"]),
        ("total_hours", "Total Hours", NORMS["general"]["total_hours"]),
    ]

    for metric_key, label, gen_norm in metrics:
        row = f"<td>{label}</td>"
        row += f"<td>{gen_norm[0]:.0f} - {gen_norm[1]:.0f}</td>"

        for p in patients:
            pid = p.patient_id
            a = arch_stats.get(pid, {})
            e = eff_stats.get(pid, {})

            if metric_key == "efficiency":
                val = e.get("mean", 0)
            else:
                val = a.get("stages", {}).get(metric_key, {}).get("mean", 0)

            z = benchmarks.get(pid, {}).get("general", {}).get(f"{metric_key}_z", 0)
            z_color = ACCENT_GREEN if abs(z) < 1 else (ACCENT_AMBER if abs(z) < 2 else ACCENT_RED)
            row += f'<td>{val:.1f} <span style="color:{z_color};font-size:0.75rem">(z={z:+.1f})</span></td>'

        rows_html += f"<tr>{row}</tr>"

    return f"""
    <div style="overflow-x:auto">
    <table class="odt-table" style="width:100%;font-size:0.85rem">
    <thead>
        <tr>
            <th>Metric</th>
            <th>General Norms</th>
            <th>{patients[0].display_name}</th>
            <th>{patients[1].display_name}</th>
        </tr>
    </thead>
    <tbody>{rows_html}</tbody>
    </table>
    </div>"""


def _build_clinical_interpretation(
    arch_stats: dict[str, dict[str, Any]],
    eff_stats: dict[str, dict[str, Any]],
    timing_stats: dict[str, dict[str, Any]],
    recovery: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Auto-generated clinical interpretation bullet points."""
    bullets: list[str] = []

    # Per-patient observations
    for p in patients:
        pid = p.patient_id
        a = arch_stats.get(pid, {})
        e = eff_stats.get(pid, {})
        t = timing_stats.get(pid, {})
        r = recovery.get(pid, {})
        stages = a.get("stages", {})

        deep = stages.get("deep_pct", {}).get("mean", 0)
        rem = stages.get("rem_pct", {}).get("mean", 0)
        eff_mean = e.get("mean", 0)
        total = stages.get("total_hours", {}).get("mean", 0)
        debt = r.get("avg_daily_debt_min", 0)

        # Architecture observations
        if deep < NORMS["general"]["deep_pct"][0]:
            bullets.append(
                f"<strong>{p.display_name}</strong>: Deep sleep ({deep:.1f}%) is below "
                f"the general population norm ({NORMS['general']['deep_pct'][0]}-{NORMS['general']['deep_pct'][1]}%), "
                f"consistent with post-treatment sleep disruption."
            )
        if rem < NORMS["general"]["rem_pct"][0]:
            bullets.append(
                f"<strong>{p.display_name}</strong>: REM sleep ({rem:.1f}%) is below population norms "
                f"({NORMS['general']['rem_pct'][0]}-{NORMS['general']['rem_pct'][1]}%), "
                f"suggesting possible autonomic interference with dream-stage cycling."
            )

        # Efficiency observations
        if eff_mean < 75:
            bullets.append(
                f"<strong>{p.display_name}</strong>: Sleep efficiency ({eff_mean:.0f}%) is below the 75% threshold, "
                f"indicating clinically poor sleep quality."
            )
        elif eff_mean < 85:
            bullets.append(
                f"<strong>{p.display_name}</strong>: Sleep efficiency ({eff_mean:.0f}%) is below "
                f"the recommended 85% threshold."
            )

        # Sleep debt
        if debt < -30:
            bullets.append(
                f"<strong>{p.display_name}</strong>: Average daily sleep debt of "
                f"{abs(debt):.0f} minutes below the 7-hour target, accumulating chronic sleep restriction."
            )

        # Efficiency trend
        slope = e.get("trend_slope_per_week", 0)
        trend_p = e.get("trend_p_value", 1)
        if trend_p < 0.05 and slope is not None:
            direction = "improving" if slope > 0 else "declining"
            bullets.append(
                f"<strong>{p.display_name}</strong>: Efficiency trend is statistically {direction} "
                f"({slope:+.2f}%/week, p={trend_p:.3f})."
            )

        # Timing
        bt_sd = t.get("bedtime_sd_min", 0)
        if bt_sd is not None and bt_sd > 60:
            bullets.append(
                f"<strong>{p.display_name}</strong>: High bedtime variability "
                f"(SD={bt_sd:.0f} min) suggests inconsistent sleep schedule, "
                f"which may impair circadian entrainment."
            )

        sjl = t.get("social_jetlag_min", 0)
        if sjl is not None and sjl > 60:
            bullets.append(
                f"<strong>{p.display_name}</strong>: Social jet lag estimate of "
                f"{sjl:.0f} minutes suggests significant weekday/weekend misalignment."
            )

    # Cross-patient comparison highlights
    deep_comp = comparison.get("deep_pct", {})
    if deep_comp.get("significant"):
        bullets.append(
            f"Deep sleep percentage differs significantly between patients "
            f"(p={deep_comp.get('p_value', 0):.3f}, "
            f"Cohen's d={deep_comp.get('cohens_d', 0):+.2f}, "
            f"{deep_comp.get('cohens_d_label', '')} effect)."
        )

    eff_comp = comparison.get("efficiency", {})
    if eff_comp.get("significant"):
        bullets.append(
            f"Sleep efficiency differs significantly "
            f"(p={eff_comp.get('p_value', 0):.3f}, "
            f"Cohen's d={eff_comp.get('cohens_d', 0):+.2f}, "
            f"{eff_comp.get('cohens_d_label', '')} effect)."
        )

    if not bullets:
        bullets.append("Data is insufficient to draw detailed clinical interpretations.")

    items = "\n".join(f"<li>{b}</li>" for b in bullets)
    return f'<ul style="line-height:1.8;font-size:0.9rem">{items}</ul>'


def _build_methodology_section() -> str:
    """Methodology and limitations section."""
    return f"""
    <div style="font-size:0.85rem;color:{TEXT_SECONDARY};line-height:1.7">
    <h3>Data Source</h3>
    <p>Sleep architecture data from Oura Ring wearable sensors (oura_sleep_periods table, type='long_sleep').
    Durations are recorded in seconds by the Oura API and converted to hours/percentages for analysis.</p>

    <h3>Architecture Percentages</h3>
    <p>Computed as <code>stage_duration / (total_sleep_duration + awake_time) &times; 100</code>,
    ensuring all stages sum to 100% of time in bed.</p>

    <h3>Statistical Tests</h3>
    <ul>
        <li><strong>Mann-Whitney U</strong>: Non-parametric test for distribution differences (does not assume normality)</li>
        <li><strong>Cohen's d</strong>: Standardized mean difference (pooled SD); |d| &lt; 0.2 = negligible, &lt; 0.5 = small, &lt; 0.8 = medium, else large</li>
        <li><strong>Cliff's delta</strong>: Non-parametric effect size based on rank ordering</li>
        <li><strong>Bootstrap CI</strong>: 5,000-iteration bootstrap for median difference confidence interval</li>
        <li><strong>Linear regression</strong>: Ordinary least squares for efficiency trend</li>
        <li><strong>Spearman rank correlation</strong>: Monotonic trend detection for recovery trajectory</li>
    </ul>

    <h3>Bedtime Handling</h3>
    <p>Bedtime hours past midnight are encoded as 24+ (e.g., 00:30 = 24.5) to avoid
    discontinuities in variability and midpoint calculations.</p>

    <h3>Population Norms</h3>
    <ul>
        <li><strong>General (age 30-39)</strong>: Deep 13-23%, REM 20-25%, Efficiency &ge;85%, Total 7-9h
            (Ohayon et al. 2004, Hirshkowitz et al. 2015)</li>
        <li><strong>Post-HSCT</strong>: Deep 8-15%, REM 10-18%, Efficiency 70-82%
            (Jim et al. 2014, Rischer et al. 2020)</li>
        <li><strong>Post-stroke</strong>: Deep 10-18%, REM 12-20%, Efficiency 72-85%
            (Leppavuori et al. 2002, Duss et al. 2018)</li>
    </ul>

    <h3>Limitations</h3>
    <ul>
        <li>N=2 case study: findings are descriptive, not generalizable</li>
        <li>Oura Ring is a consumer wearable, not a polysomnograph; sleep staging has known accuracy limitations</li>
        <li>Different observation windows and data density between patients</li>
        <li>No control for confounders (medications, environment, activity levels)</li>
        <li>Population norms are age-adjusted approximations, not individual-level standards</li>
    </ul>
    </div>"""


def build_html(
    data: dict[str, pd.DataFrame],
    arch_stats: dict[str, dict[str, Any]],
    eff_stats: dict[str, dict[str, Any]],
    timing_stats: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
    benchmarks: dict[str, dict[str, Any]],
    recovery: dict[str, dict[str, Any]],
    sleep_balance: dict[str, pd.Series],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Build the full HTML report."""
    sections: list[str] = []

    # -- KPI Row (Executive Summary) --
    h_arch = arch_stats.get(patients[0].patient_id, {}).get("stages", {})
    m_arch = arch_stats.get(patients[1].patient_id, {}).get("stages", {})
    h_eff = eff_stats.get(patients[0].patient_id, {})
    m_eff = eff_stats.get(patients[1].patient_id, {})
    h_rec = recovery.get(patients[0].patient_id, {})
    m_rec = recovery.get(patients[1].patient_id, {})

    h_total = h_arch.get("total_hours", {}).get("mean", 0)
    m_total = m_arch.get("total_hours", {}).get("mean", 0)
    h_eff_mean = h_eff.get("mean", 0)
    m_eff_mean = m_eff.get("mean", 0)
    h_deep = h_arch.get("deep_pct", {}).get("mean", 0)
    m_deep = m_arch.get("deep_pct", {}).get("mean", 0)

    kpi_row = make_kpi_row(
        make_kpi_card(
            "HENRIK AVG SLEEP", h_total, "hrs",
            status="critical" if h_total < 6 else ("warning" if h_total < 7 else "normal"),
            detail=f"Target: 7-9 hrs",
            status_label="Short" if h_total < 6 else ("Below target" if h_total < 7 else "Adequate"),
        ),
        make_kpi_card(
            "MITCHELL AVG SLEEP", m_total, "hrs",
            status="warning" if m_total < 7 else "normal",
            detail=f"Target: 7-9 hrs",
            status_label="Below target" if m_total < 7 else "Adequate",
        ),
        make_kpi_card(
            "HENRIK EFFICIENCY", h_eff_mean, "%",
            status="critical" if h_eff_mean < 75 else ("warning" if h_eff_mean < 85 else "normal"),
            detail=f"{h_eff.get('pct_below_75', 0):.0f}% nights below 75%",
        ),
        make_kpi_card(
            "MITCHELL EFFICIENCY", m_eff_mean, "%",
            status="critical" if m_eff_mean < 75 else ("warning" if m_eff_mean < 85 else "normal"),
            detail=f"{m_eff.get('pct_below_75', 0):.0f}% nights below 75%",
        ),
        make_kpi_card(
            "HENRIK DEEP SLEEP", h_deep, "%",
            status="warning" if h_deep < 13 else "normal",
            detail=f"Norm: 13-23%",
        ),
        make_kpi_card(
            "MITCHELL DEEP SLEEP", m_deep, "%",
            status="warning" if m_deep < 13 else "normal",
            detail=f"Norm: 13-23%",
        ),
    )
    sections.append(kpi_row)

    # -- Disclaimer --
    sections.append(disclaimer_banner())

    # -- Section 1: Sleep Architecture --
    sections.append(section_html_or_placeholder(
        "Sleep Architecture",
        lambda: make_section(
            "Sleep Architecture Over Time",
            _embed(_fig_stacked_area(data, patients))
            + "<br>"
            + _embed(_fig_arch_distributions(data, patients)),
            section_id="architecture",
        ),
    ))

    # -- Section 2: Sleep Efficiency --
    sections.append(section_html_or_placeholder(
        "Sleep Efficiency",
        lambda: make_section(
            "Sleep Efficiency Trends",
            _embed(_fig_efficiency(data, patients, eff_stats)),
            section_id="efficiency",
        ),
    ))

    # -- Section 3: Sleep Timing --
    sections.append(section_html_or_placeholder(
        "Sleep Timing",
        lambda: make_section(
            "Sleep Timing Analysis",
            _embed(_fig_timing(data, patients)),
            section_id="timing",
        ),
    ))

    # -- Section 4: Benchmark Comparison --
    sections.append(section_html_or_placeholder(
        "Benchmark Comparison",
        lambda: make_section(
            "Benchmark Comparison",
            _embed(_fig_benchmark_radar(arch_stats, eff_stats, sleep_balance, timing_stats, patients))
            + _build_benchmark_table(benchmarks, arch_stats, eff_stats, patients),
            section_id="benchmarks",
        ),
    ))

    # -- Section 5: Recovery Trajectory --
    sections.append(section_html_or_placeholder(
        "Recovery Trajectory",
        lambda: make_section(
            "Recovery Trajectory",
            _embed(_fig_recovery_trajectory(data, eff_stats, patients)),
            section_id="recovery",
        ),
    ))

    # -- Section 6: Statistical Comparison Table --
    sections.append(section_html_or_placeholder(
        "Statistical Comparison",
        lambda: make_section(
            "Statistical Comparison",
            _build_stat_comparison_table(comparison),
            section_id="stats",
        ),
    ))

    # -- Section 7: Clinical Interpretation --
    sections.append(section_html_or_placeholder(
        "Clinical Interpretation",
        lambda: make_section(
            "Clinical Interpretation",
            _build_clinical_interpretation(
                arch_stats, eff_stats, timing_stats,
                recovery, comparison, patients,
            ),
            section_id="interpretation",
        ),
    ))

    # -- Section 8: Methodology --
    sections.append(section_html_or_placeholder(
        "Methodology",
        lambda: make_section(
            "Methodology & Limitations",
            _build_methodology_section(),
            section_id="methodology",
        ),
    ))

    body = "\n".join(sections)

    return wrap_html(
        title="Sleep Architecture as Health Signal",
        body_content=body,
        report_id="comp_sleep",
        subtitle="Module 3: Comparative Sleep Analysis",
        header_meta="Henrik (post-HSCT) vs Mitchell (post-Stroke)",
    )


# ---------------------------------------------------------------------------
# JSON Export
# ---------------------------------------------------------------------------

def export_json(
    arch_stats: dict[str, dict[str, Any]],
    eff_stats: dict[str, dict[str, Any]],
    timing_stats: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
    benchmarks: dict[str, dict[str, Any]],
    recovery: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> None:
    """Write structured metrics JSON."""

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return [_sanitize(v) for v in obj.tolist()]
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    output = {
        "meta": {
            "report": "comparative_sleep",
            "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "patients": {
                p.patient_id: {
                    "display_name": p.display_name,
                    "event_date": str(p.event_date),
                    "event_label": p.event_label,
                }
                for p in patients
            },
        },
        "architecture": arch_stats,
        "efficiency": eff_stats,
        "timing": timing_stats,
        "comparison": comparison,
        "benchmarks": benchmarks,
        "recovery_indicators": recovery,
    }

    output = _sanitize(output)

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("JSON metrics written to %s", JSON_OUTPUT)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    """Run comparative sleep analysis pipeline."""
    logger.info("[1/7] Loading patient data...")
    patients = default_patients()
    if patients[1] is None:
        print("Skipping: mitch.db not found (second patient data not available)")
        return 0
    data = load_sleep_data(patients)
    sleep_balance = load_readiness_sleep_balance(patients)

    logger.info("[2/7] Computing architecture statistics...")
    arch_stats = compute_architecture_stats(data)

    logger.info("[3/7] Computing efficiency statistics...")
    eff_stats = compute_efficiency_stats(data)

    logger.info("[4/7] Computing timing statistics...")
    timing_stats = compute_timing_stats(data)

    logger.info("[5/7] Computing inter-patient comparison...")
    comparison = compute_inter_patient_comparison(data, patients)

    benchmarks = compute_benchmarks(arch_stats, eff_stats, patients)
    recovery = compute_recovery_indicators(data, eff_stats, sleep_balance)

    logger.info("[6/7] Generating HTML report...")
    html = build_html(
        data, arch_stats, eff_stats, timing_stats,
        comparison, benchmarks, recovery, sleep_balance, patients,
    )
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", HTML_OUTPUT)

    logger.info("[7/7] Exporting JSON metrics...")
    export_json(
        arch_stats, eff_stats, timing_stats,
        comparison, benchmarks, recovery, patients,
    )

    logger.info("Comparative sleep analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
