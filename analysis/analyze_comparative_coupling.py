#!/usr/bin/env python3
"""Module 4: Activity-Recovery Coupling.

Analyze whether more activity on day N predicts better or worse sleep/HRV
on day N+1, and whether this relationship differs between Patient 1 (post-HSCT)
and Patient 2 (post-stroke).

Outputs:
  - Interactive HTML dashboard: reports/comparative_activity_recovery_coupling.html
  - JSON metrics:               reports/comparative_activity_recovery_coupling.json

Usage:
    python analysis/analyze_comparative_coupling.py
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

from config import REPORTS_DIR, FONT_FAMILY
from _comparative_utils import (
    PatientConfig,
    default_patients,
    load_patient_data,
    PATIENT_COLORS,
)
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    disclaimer_banner,
    format_p_value,
    BG_PRIMARY,
    BG_SURFACE,
    BG_ELEVATED,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_PURPLE,
    ACCENT_CYAN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    BORDER_SUBTLE,
)
from _hardening import safe_connect, safe_read_sql, section_html_or_placeholder

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "comparative_activity_recovery_coupling.html"
JSON_OUTPUT = REPORTS_DIR / "comparative_activity_recovery_coupling.json"


# ---------------------------------------------------------------------------
# Activity & Recovery metric definitions
# ---------------------------------------------------------------------------

ACTIVITY_METRICS = [
    ("steps", "Steps"),
    ("active_calories", "Active Calories"),
    ("medium_activity_time", "Medium Activity"),
    ("high_activity_time", "High Activity"),
    ("score", "Activity Score"),
    ("composite_activity", "Composite Activity"),
]

RECOVERY_METRICS = [
    ("hrv_average", "HRV (RMSSD)"),
    ("total_sleep_duration", "Sleep Duration"),
    ("efficiency", "Sleep Efficiency"),
    ("deep_sleep_duration", "Deep Sleep"),
    ("rem_sleep_duration", "REM Sleep"),
    ("average_heart_rate", "Avg Heart Rate"),
    ("lowest_heart_rate", "Lowest Heart Rate"),
    ("sleep_score", "Sleep Score"),
    ("readiness_score", "Readiness Score"),
    ("recovery_index", "Recovery Index"),
]

# Key pairs for focused analyses
KEY_PAIRS = [
    ("steps", "hrv_average"),
    ("steps", "readiness_score"),
    ("composite_activity", "hrv_average"),
    ("composite_activity", "readiness_score"),
    ("steps", "total_sleep_duration"),
    ("active_calories", "hrv_average"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(fig: go.Figure) -> str:
    """Embed a Plotly figure as inline HTML (no JS bundle)."""
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float, returning default for NaN/None."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _fisher_z(r: float) -> float:
    """Fisher r-to-z transformation."""
    r = max(-0.999, min(0.999, r))
    return 0.5 * np.log((1 + r) / (1 - r))


def _fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z-transform confidence interval for Spearman r."""
    if n < 4:
        return (np.nan, np.nan)
    z = _fisher_z(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
    z_lo = z - z_crit * se
    z_hi = z + z_crit * se
    r_lo = np.tanh(z_lo)
    r_hi = np.tanh(z_hi)
    return (float(r_lo), float(r_hi))


def _fisher_r_to_z_test(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Fisher r-to-z test comparing two independent Spearman correlations.

    Returns (z_statistic, p_value).
    """
    if n1 < 4 or n2 < 4:
        return (np.nan, np.nan)
    z1 = _fisher_z(r1)
    z2 = _fisher_z(r2)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p = 2 * scipy_stats.norm.sf(abs(z_stat))
    return (float(z_stat), float(p))


def _spearman_safe(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Compute Spearman r with NaN protection. Returns (r, p, n)."""
    mask = x.notna() & y.notna()
    x_c, y_c = x[mask], y[mask]
    n = len(x_c)
    if n < 5:
        return (np.nan, np.nan, n)
    r, p = scipy_stats.spearmanr(x_c, y_c)
    return (float(r), float(p), n)


def _ols_safe(x: pd.Series, y: pd.Series) -> dict:
    """Simple OLS regression with NaN protection."""
    mask = x.notna() & y.notna()
    x_c, y_c = x[mask].values.astype(float), y[mask].values.astype(float)
    n = len(x_c)
    if n < 5:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan,
                "p_value": np.nan, "std_beta": np.nan, "n": n}
    slope, intercept, r, p, se = scipy_stats.linregress(x_c, y_c)
    x_std = np.std(x_c, ddof=1)
    y_std = np.std(y_c, ddof=1)
    std_beta = slope * x_std / y_std if y_std > 0 else np.nan
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r ** 2),
        "p_value": float(p),
        "std_beta": float(std_beta),
        "n": n,
    }


def _adaptive_bins(n: int) -> int:
    """Choose number of quantile bins based on sample size."""
    if n >= 40:
        return 4
    if n >= 20:
        return 3
    return 2


# ---------------------------------------------------------------------------
# [1/7] Data Loading
# ---------------------------------------------------------------------------

def load_data(
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, pd.DataFrame]:
    """Load and merge activity + recovery data for both patients.

    Returns a dict keyed by patient_id, each containing a single DataFrame
    with date index and all activity + recovery columns.
    """
    result: dict[str, pd.DataFrame] = {}

    for p in patients:
        frames = {}

        # --- Activity data ---
        act = load_patient_data(
            p, "oura_activity",
            columns="date, steps, active_calories, medium_activity_time, high_activity_time, score",
        )
        if not act.empty:
            act = act.rename(columns={"score": "activity_score"})
            frames["activity"] = act

        # --- Sleep periods (long_sleep) for HR/HRV ---
        sp = load_patient_data(
            p, "oura_sleep_periods",
            columns="day, average_hrv, total_sleep_duration, efficiency, "
                    "deep_sleep_duration, rem_sleep_duration, "
                    "average_heart_rate, lowest_heart_rate",
        )
        if not sp.empty:
            sp = sp.rename(columns={"average_hrv": "hrv_average"})
            # Convert durations from seconds to hours
            for dur_col in ["total_sleep_duration", "deep_sleep_duration", "rem_sleep_duration"]:
                if dur_col in sp.columns:
                    sp[dur_col] = sp[dur_col] / 3600.0
            frames["sleep_periods"] = sp

        # --- Sleep scores ---
        slp = load_patient_data(
            p, "oura_sleep",
            columns="date, score",
        )
        if not slp.empty:
            slp = slp.rename(columns={"score": "sleep_score"})
            frames["sleep"] = slp

        # --- Readiness scores ---
        rdy = load_patient_data(
            p, "oura_readiness",
            columns="date, score",
        )
        if not rdy.empty:
            rdy = rdy.rename(columns={"score": "readiness_score"})
            frames["readiness"] = rdy

        # Merge all on date index
        if not frames:
            result[p.patient_id] = pd.DataFrame()
            continue

        merged = None
        for key, df in frames.items():
            if merged is None:
                merged = df
            else:
                merged = merged.join(df, how="outer", rsuffix=f"_{key}")

        # Ensure activity_score column exists (renamed from score)
        if "score" in merged.columns and "activity_score" not in merged.columns:
            merged = merged.rename(columns={"score": "activity_score"})

        # Compute composite activity: z-scored mean of steps + active_cal + medium_time
        composite_cols = ["steps", "active_calories", "medium_activity_time"]
        available_comp = [c for c in composite_cols if c in merged.columns]
        if len(available_comp) >= 2:
            z_parts = []
            for c in available_comp:
                s = merged[c].astype(float)
                mean_val = s.mean()
                std_val = s.std()
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1.0
                z_parts.append((s - mean_val) / std_val)
            merged["composite_activity"] = pd.concat(z_parts, axis=1).mean(axis=1)
        else:
            merged["composite_activity"] = np.nan

        # Compute recovery index: z-scored mean of HRV + sleep_score + readiness_score
        ri_cols = ["hrv_average", "sleep_score", "readiness_score"]
        available_ri = [c for c in ri_cols if c in merged.columns]
        if len(available_ri) >= 2:
            z_parts = []
            for c in available_ri:
                s = merged[c].astype(float)
                mean_val = s.mean()
                std_val = s.std()
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1.0
                z_parts.append((s - mean_val) / std_val)
            merged["recovery_index"] = pd.concat(z_parts, axis=1).mean(axis=1)
        else:
            merged["recovery_index"] = np.nan

        result[p.patient_id] = merged
        logger.info(
            "Loaded %s: %d days, columns=%s",
            p.display_name, len(merged), list(merged.columns),
        )

    return result


# ---------------------------------------------------------------------------
# [2/7] Build Lagged Pairs
# ---------------------------------------------------------------------------

def build_lagged_pairs(
    data: dict[str, pd.DataFrame],
    lag: int = 1,
) -> dict[str, pd.DataFrame]:
    """Build day-N activity -> day-N+lag recovery pairs.

    Returns dict keyed by patient_id, each a DataFrame where every activity
    column is from day N and every recovery column is from day N+lag.
    """
    act_cols = [c for c, _ in ACTIVITY_METRICS]
    rec_cols = [c for c, _ in RECOVERY_METRICS]
    result: dict[str, pd.DataFrame] = {}

    for pid, df in data.items():
        if df.empty:
            result[pid] = pd.DataFrame()
            continue

        avail_act = [c for c in act_cols if c in df.columns]
        avail_rec = [c for c in rec_cols if c in df.columns]

        if not avail_act or not avail_rec:
            result[pid] = pd.DataFrame()
            continue

        # Activity from day N
        act_df = df[avail_act].copy()
        # Recovery from day N+lag
        rec_df = df[avail_rec].shift(-lag).copy()

        paired = pd.concat([act_df, rec_df], axis=1)
        # Drop rows where ALL activity or ALL recovery are NaN
        paired = paired.dropna(subset=avail_act, how="all")
        paired = paired.dropna(subset=avail_rec, how="all")
        result[pid] = paired

    return result


# ---------------------------------------------------------------------------
# [3/7] Lag Correlation Analysis
# ---------------------------------------------------------------------------

def compute_lag_correlations(
    data: dict[str, pd.DataFrame],
    lags: list[int] | None = None,
) -> dict[str, list[dict]]:
    """Compute Spearman correlations at multiple lags.

    Returns dict keyed by patient_id, each a list of dicts with fields:
    activity, recovery, lag, r, p, n, ci_lo, ci_hi.
    """
    if lags is None:
        lags = [0, 1, 2]

    act_cols = [c for c, _ in ACTIVITY_METRICS]
    rec_cols = [c for c, _ in RECOVERY_METRICS]
    result: dict[str, list[dict]] = {}

    for pid, df in data.items():
        if df.empty:
            result[pid] = []
            continue

        records: list[dict] = []
        avail_act = [c for c in act_cols if c in df.columns]
        avail_rec = [c for c in rec_cols if c in df.columns]

        for lag in lags:
            for act in avail_act:
                for rec in avail_rec:
                    x = df[act].astype(float)
                    y = df[rec].shift(-lag).astype(float)
                    r, p, n = _spearman_safe(x, y)
                    ci_lo, ci_hi = _fisher_z_ci(r, n) if not np.isnan(r) else (np.nan, np.nan)
                    records.append({
                        "activity": act,
                        "recovery": rec,
                        "lag": lag,
                        "r": r,
                        "p": p,
                        "n": n,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    })
        result[pid] = records

    return result


# ---------------------------------------------------------------------------
# [4/7] Cross-Correlation Functions
# ---------------------------------------------------------------------------

def compute_cross_correlations(
    data: dict[str, pd.DataFrame],
    pairs: list[tuple[str, str]] | None = None,
    max_lag: int = 3,
) -> dict[str, dict[str, list[dict]]]:
    """Compute cross-correlations at lags -max_lag to +max_lag.

    Returns {patient_id: {pair_key: [{lag, r, p, n, sig}, ...]}}.
    """
    if pairs is None:
        pairs = [
            ("steps", "hrv_average"),
            ("steps", "readiness_score"),
            ("composite_activity", "hrv_average"),
        ]

    result: dict[str, dict[str, list[dict]]] = {}
    for pid, df in data.items():
        if df.empty:
            result[pid] = {}
            continue

        pair_results: dict[str, list[dict]] = {}
        for act, rec in pairs:
            if act not in df.columns or rec not in df.columns:
                continue
            key = f"{act}_vs_{rec}"
            lag_records: list[dict] = []

            for lag in range(-max_lag, max_lag + 1):
                x = df[act].astype(float)
                y = df[rec].shift(-lag).astype(float)
                r, p, n = _spearman_safe(x, y)
                lag_records.append({
                    "lag": lag,
                    "r": _safe_float(r),
                    "p": _safe_float(p, 1.0),
                    "n": n,
                    "sig": (not np.isnan(p)) and p < 0.10,
                })
            pair_results[key] = lag_records
        result[pid] = pair_results

    return result


# ---------------------------------------------------------------------------
# [5/7] Dose-Response Analysis
# ---------------------------------------------------------------------------

def compute_dose_response(
    data: dict[str, pd.DataFrame],
    activity_col: str = "steps",
    recovery_col: str = "hrv_average",
) -> dict[str, dict]:
    """Bin activity into adaptive quantiles and compare next-day recovery.

    Returns per-patient dict with bin_labels, bin_means, bin_cis, kruskal_h, kruskal_p.
    """
    result: dict[str, dict] = {}

    for pid, df in data.items():
        if df.empty or activity_col not in df.columns or recovery_col not in df.columns:
            result[pid] = {"bins": [], "kruskal_h": np.nan, "kruskal_p": np.nan, "n": 0}
            continue

        x = df[activity_col].astype(float)
        y_next = df[recovery_col].shift(-1).astype(float)
        mask = x.notna() & y_next.notna()
        x_c, y_c = x[mask], y_next[mask]
        n = len(x_c)

        if n < 6:
            result[pid] = {"bins": [], "kruskal_h": np.nan, "kruskal_p": np.nan, "n": n}
            continue

        n_bins = _adaptive_bins(n)
        try:
            x_c_binned = pd.qcut(x_c, q=n_bins, duplicates="drop")
        except ValueError:
            result[pid] = {"bins": [], "kruskal_h": np.nan, "kruskal_p": np.nan, "n": n}
            continue

        groups = []
        bin_info = []
        for label, group_idx in x_c_binned.groupby(x_c_binned).groups.items():
            recovery_vals = y_c.loc[group_idx].dropna()
            if len(recovery_vals) < 2:
                continue
            groups.append(recovery_vals.values)
            mean_val = float(recovery_vals.mean())
            # Bootstrap CI
            rng = np.random.default_rng(42)
            boot_means = np.array([
                rng.choice(recovery_vals.values, size=len(recovery_vals), replace=True).mean()
                for _ in range(2000)
            ])
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))
            bin_info.append({
                "label": str(label),
                "n": len(recovery_vals),
                "mean": mean_val,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })

        # Kruskal-Wallis test
        kruskal_h, kruskal_p = np.nan, np.nan
        if len(groups) >= 2:
            try:
                kruskal_h, kruskal_p = scipy_stats.kruskal(*groups)
                kruskal_h = float(kruskal_h)
                kruskal_p = float(kruskal_p)
            except Exception:
                pass

        result[pid] = {
            "bins": bin_info,
            "kruskal_h": kruskal_h,
            "kruskal_p": kruskal_p,
            "n": n,
            "n_bins": len(bin_info),
        }

    return result


# ---------------------------------------------------------------------------
# [6/7] Regression, Correlation Matrix, Clinical Assessment
# ---------------------------------------------------------------------------

def compute_regression(
    data: dict[str, pd.DataFrame],
    pairs: list[tuple[str, str]] | None = None,
) -> dict[str, dict[str, dict]]:
    """OLS regression: recovery[N+1] ~ activity[N] for key pairs."""
    if pairs is None:
        pairs = KEY_PAIRS

    result: dict[str, dict[str, dict]] = {}
    for pid, df in data.items():
        if df.empty:
            result[pid] = {}
            continue
        pair_results: dict[str, dict] = {}
        for act, rec in pairs:
            if act not in df.columns or rec not in df.columns:
                continue
            key = f"{act}_vs_{rec}"
            x = df[act].astype(float)
            y_next = df[rec].shift(-1).astype(float)
            pair_results[key] = _ols_safe(x, y_next)
        result[pid] = pair_results
    return result


def compute_correlation_matrix(
    data: dict[str, pd.DataFrame],
    lag: int = 1,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Full activity x recovery correlation matrix at specified lag.

    Returns {patient_id: {"r": DataFrame, "p": DataFrame}}.
    """
    act_cols = [c for c, _ in ACTIVITY_METRICS]
    rec_cols = [c for c, _ in RECOVERY_METRICS]
    result: dict[str, dict[str, pd.DataFrame]] = {}

    for pid, df in data.items():
        if df.empty:
            result[pid] = {"r": pd.DataFrame(), "p": pd.DataFrame()}
            continue

        avail_act = [c for c in act_cols if c in df.columns]
        avail_rec = [c for c in rec_cols if c in df.columns]

        r_matrix = pd.DataFrame(index=avail_act, columns=avail_rec, dtype=float)
        p_matrix = pd.DataFrame(index=avail_act, columns=avail_rec, dtype=float)

        for act in avail_act:
            for rec in avail_rec:
                x = df[act].astype(float)
                y = df[rec].shift(-lag).astype(float)
                r, p, n = _spearman_safe(x, y)
                r_matrix.loc[act, rec] = r
                p_matrix.loc[act, rec] = p

        result[pid] = {"r": r_matrix, "p": p_matrix}
    return result


def assess_coupling(
    lag_corrs: dict[str, list[dict]],
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, dict]:
    """Clinical assessment of coupling per patient.

    Returns per-patient: coupling_direction, strength, primary_r, primary_p,
    recovery_capacity_rating.
    """
    result: dict[str, dict] = {}

    for p in patients:
        pid = p.patient_id
        records = lag_corrs.get(pid, [])
        # Focus on steps->HRV at lag 1 as primary indicator
        primary = [
            rec for rec in records
            if rec["activity"] == "steps" and rec["recovery"] == "hrv_average" and rec["lag"] == 1
        ]
        if not primary:
            result[pid] = {
                "coupling_direction": "insufficient_data",
                "strength": "unknown",
                "primary_r": np.nan,
                "primary_p": np.nan,
                "recovery_capacity_rating": "unknown",
            }
            continue

        r = primary[0]["r"]
        p_val = primary[0]["p"]
        n = primary[0]["n"]

        if np.isnan(r) or np.isnan(p_val):
            direction = "insufficient_data"
            strength = "unknown"
        elif r > 0.15 and p_val < 0.10:
            direction = "positive"
            strength = "strong" if abs(r) > 0.4 else "moderate" if abs(r) > 0.25 else "weak"
        elif r < -0.15 and p_val < 0.10:
            direction = "negative"
            strength = "strong" if abs(r) > 0.4 else "moderate" if abs(r) > 0.25 else "weak"
        else:
            direction = "absent"
            strength = "negligible"

        # Recovery capacity rating based on coupling direction and HRV context
        if direction == "positive":
            rating = "adaptive"
        elif direction == "negative":
            rating = "maladaptive"
        elif direction == "absent" and n < 30:
            rating = "underpowered"
        else:
            rating = "decoupled"

        result[pid] = {
            "coupling_direction": direction,
            "strength": strength,
            "primary_r": _safe_float(r),
            "primary_p": _safe_float(p_val, 1.0),
            "primary_n": n,
            "recovery_capacity_rating": rating,
        }

    return result


def compare_patients(
    coupling: dict[str, dict],
    lag_corrs: dict[str, list[dict]],
    dose_response: dict[str, dict],
    patients: tuple[PatientConfig, PatientConfig],
) -> dict:
    """Cross-patient comparison using Fisher r-to-z test."""
    pids = [p.patient_id for p in patients]
    if len(pids) < 2:
        return {"fisher_z": np.nan, "fisher_p": np.nan, "interpretation": "insufficient patients"}

    h, m = pids[0], pids[1]
    h_coup = coupling.get(h, {})
    m_coup = coupling.get(m, {})

    r1 = _safe_float(h_coup.get("primary_r"))
    n1 = h_coup.get("primary_n", 0)
    r2 = _safe_float(m_coup.get("primary_r"))
    n2 = m_coup.get("primary_n", 0)

    z_stat, fisher_p = _fisher_r_to_z_test(r1, n1, r2, n2)

    # Dose-response slope comparison
    h_dr = dose_response.get(h, {})
    m_dr = dose_response.get(m, {})
    h_bins = h_dr.get("bins", [])
    m_bins = m_dr.get("bins", [])

    h_slope = np.nan
    m_slope = np.nan
    if len(h_bins) >= 2:
        h_slope = (h_bins[-1]["mean"] - h_bins[0]["mean"]) / len(h_bins)
    if len(m_bins) >= 2:
        m_slope = (m_bins[-1]["mean"] - m_bins[0]["mean"]) / len(m_bins)

    # Interpretation
    h_dir = h_coup.get("coupling_direction", "unknown")
    m_dir = m_coup.get("coupling_direction", "unknown")

    if h_dir == m_dir:
        interp = f"Both patients show {h_dir} activity-recovery coupling."
    else:
        interp = (
            f"Divergent coupling: Patient 1 shows {h_dir} coupling while "
            f"Patient 2 shows {m_dir} coupling, suggesting different "
            f"physiological recovery capacity."
        )

    if not np.isnan(fisher_p) and fisher_p < 0.10:
        interp += " This difference is statistically notable (p < 0.10)."

    return {
        "fisher_z": _safe_float(z_stat),
        "fisher_p": _safe_float(fisher_p, 1.0),
        "henrik_r": _safe_float(r1),
        "henrik_n": n1,
        "mitch_r": _safe_float(r2),
        "mitch_n": n2,
        "dose_response_slope_henrik": _safe_float(h_slope),
        "dose_response_slope_mitch": _safe_float(m_slope),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _fig_dual_scatter(
    data: dict[str, pd.DataFrame],
    patients: tuple[PatientConfig, PatientConfig],
    act_col: str = "steps",
    rec_col: str = "hrv_average",
    act_label: str = "Steps (day N)",
    rec_label: str = "HRV (day N+1)",
) -> go.Figure:
    """Fig 1: Dual scatter -- activity[N] vs recovery[N+1] for both patients."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[p.display_name for p in patients],
        horizontal_spacing=0.10,
    )

    for i, p in enumerate(patients, 1):
        pid = p.patient_id
        df = data.get(pid, pd.DataFrame())
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)

        if df.empty or act_col not in df.columns or rec_col not in df.columns:
            continue

        x = df[act_col].astype(float)
        y = df[rec_col].shift(-1).astype(float)
        mask = x.notna() & y.notna()
        x_c, y_c = x[mask], y[mask]

        if len(x_c) < 3:
            continue

        # Scatter
        fig.add_trace(go.Scatter(
            x=x_c.values, y=y_c.values,
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.5),
            name=p.display_name,
            showlegend=True,
        ), row=1, col=i)

        # OLS trend line
        ols = _ols_safe(x_c, y_c)
        if not np.isnan(ols["slope"]):
            x_range = np.linspace(float(x_c.min()), float(x_c.max()), 50)
            y_pred = ols["slope"] * x_range + ols["intercept"]
            fig.add_trace(go.Scatter(
                x=x_range, y=y_pred,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"OLS (R\u00b2={ols['r_squared']:.3f}, p={ols['p_value']:.3f})",
                showlegend=True,
            ), row=1, col=i)

    fig.update_xaxes(title_text=act_label, row=1, col=1)
    fig.update_xaxes(title_text=act_label, row=1, col=2)
    fig.update_yaxes(title_text=rec_label, row=1, col=1)
    fig.update_yaxes(title_text=rec_label, row=1, col=2)

    fig.update_layout(
        height=450,
        title=dict(text=f"Activity-Recovery Coupling: {act_label} vs {rec_label}", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=70, b=80),
    )
    return fig


def _fig_cross_correlation(
    xcorr: dict[str, dict[str, list[dict]]],
    patients: tuple[PatientConfig, PatientConfig],
    pair_key: str = "steps_vs_hrv_average",
    title_suffix: str = "Steps vs HRV",
) -> go.Figure:
    """Fig 2: Cross-correlation bars at lags -3 to +3."""
    fig = go.Figure()

    for p in patients:
        pid = p.patient_id
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        pair_data = xcorr.get(pid, {}).get(pair_key, [])

        if not pair_data:
            continue

        lags = [d["lag"] for d in pair_data]
        rs = [d["r"] for d in pair_data]
        sigs = [d["sig"] for d in pair_data]

        # Star text for significant lags
        texts = ["*" if s else "" for s in sigs]

        fig.add_trace(go.Bar(
            x=lags, y=rs,
            name=p.display_name,
            marker_color=color,
            opacity=0.8,
            text=texts,
            textposition="outside",
            textfont=dict(size=14, color=ACCENT_AMBER),
        ))

    # Zero reference line
    fig.add_shape(
        type="line",
        x0=-3.5, x1=3.5,
        y0=0, y1=0,
        line=dict(color=TEXT_SECONDARY, width=1, dash="dash"),
    )

    fig.update_layout(
        barmode="group",
        height=400,
        title=dict(text=f"Cross-Correlation: {title_suffix}", font=dict(size=16)),
        xaxis_title="Lag (days, positive = activity leads recovery)",
        yaxis_title="Spearman r",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=60, b=80),
    )
    fig.add_annotation(
        x=1, y=1.02, xref="paper", yref="paper",
        text="* p < 0.10",
        showarrow=False,
        font=dict(size=10, color=ACCENT_AMBER),
    )
    return fig


def _fig_dose_response(
    dose_response: dict[str, dict],
    patients: tuple[PatientConfig, PatientConfig],
    title: str = "Dose-Response: Steps vs Next-Day HRV",
) -> go.Figure:
    """Fig 3: Grouped bar chart with error bars for dose-response."""
    fig = go.Figure()

    for p in patients:
        pid = p.patient_id
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        dr = dose_response.get(pid, {})
        bins = dr.get("bins", [])

        if not bins:
            continue

        labels = [f"Q{i+1}" for i in range(len(bins))]
        means = [b["mean"] for b in bins]
        err_lo = [b["mean"] - b["ci_lo"] for b in bins]
        err_hi = [b["ci_hi"] - b["mean"] for b in bins]

        fig.add_trace(go.Bar(
            x=labels, y=means,
            name=p.display_name,
            marker_color=color,
            opacity=0.8,
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_hi,
                arrayminus=err_lo,
                color=TEXT_SECONDARY,
            ),
        ))

    fig.update_layout(
        barmode="group",
        height=400,
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Activity Quantile (Low to High)",
        yaxis_title="Mean Next-Day HRV (ms)",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=60, b=80),
    )

    return fig


def _fig_correlation_heatmap(
    corr_matrices: dict[str, dict[str, pd.DataFrame]],
    patient: PatientConfig,
) -> go.Figure:
    """Fig 4/5: Correlation heatmap (activity rows x recovery cols)."""
    pid = patient.patient_id
    mat = corr_matrices.get(pid, {})
    r_df = mat.get("r", pd.DataFrame())
    p_df = mat.get("p", pd.DataFrame())

    if r_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=f"Correlation Matrix: {patient.display_name} (no data)"),
            height=300,
        )
        return fig

    # Display labels
    act_labels = {c: lbl for c, lbl in ACTIVITY_METRICS}
    rec_labels = {c: lbl for c, lbl in RECOVERY_METRICS}
    y_labels = [act_labels.get(c, c) for c in r_df.index]
    x_labels = [rec_labels.get(c, c) for c in r_df.columns]

    # Annotate with r value and significance star
    r_vals = r_df.values.astype(float)
    p_vals = p_df.values.astype(float)
    annotations = []
    for i in range(r_vals.shape[0]):
        for j in range(r_vals.shape[1]):
            r_val = r_vals[i, j]
            p_val = p_vals[i, j]
            if np.isnan(r_val):
                text = ""
            else:
                star = "*" if (not np.isnan(p_val) and p_val < 0.10) else ""
                text = f"{r_val:.2f}{star}"
            annotations.append(text)

    annotation_text = np.array(annotations).reshape(r_vals.shape)

    fig = go.Figure(data=go.Heatmap(
        z=r_vals,
        x=x_labels,
        y=y_labels,
        text=annotation_text,
        texttemplate="%{text}",
        colorscale="RdBu",
        zmid=0,
        zmin=-0.6,
        zmax=0.6,
        colorbar=dict(title="Spearman r", len=0.8),
    ))

    fig.update_layout(
        height=350,
        title=dict(text=f"Lag-1 Correlation Matrix: {patient.display_name}", font=dict(size=16)),
        xaxis=dict(tickangle=-45, side="bottom"),
        margin=dict(l=120, r=20, t=60, b=100),
    )
    return fig


def _fig_lagged_heatmap(
    data: dict[str, pd.DataFrame],
    patients: tuple[PatientConfig, PatientConfig],
    recovery_cols: list[str] | None = None,
    max_lag: int = 3,
) -> go.Figure:
    """Fig 6: Side-by-side lagged heatmaps (lags on y, recovery on x)."""
    if recovery_cols is None:
        recovery_cols = ["hrv_average", "total_sleep_duration", "efficiency",
                         "sleep_score", "readiness_score"]

    rec_labels = {c: lbl for c, lbl in RECOVERY_METRICS}
    lags = list(range(0, max_lag + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[p.display_name for p in patients],
        horizontal_spacing=0.12,
    )

    for idx, p in enumerate(patients, 1):
        pid = p.patient_id
        df = data.get(pid, pd.DataFrame())

        if df.empty or "steps" not in df.columns:
            continue

        avail_rec = [c for c in recovery_cols if c in df.columns]
        x_labels = [rec_labels.get(c, c) for c in avail_rec]

        z_matrix = np.full((len(lags), len(avail_rec)), np.nan)
        for i, lag in enumerate(lags):
            for j, rec in enumerate(avail_rec):
                x = df["steps"].astype(float)
                y = df[rec].shift(-lag).astype(float)
                r, p_val, n = _spearman_safe(x, y)
                z_matrix[i, j] = r

        # Annotation text
        annot = []
        for i in range(z_matrix.shape[0]):
            row_text = []
            for j in range(z_matrix.shape[1]):
                v = z_matrix[i, j]
                row_text.append(f"{v:.2f}" if not np.isnan(v) else "")
            annot.append(row_text)

        fig.add_trace(go.Heatmap(
            z=z_matrix,
            x=x_labels,
            y=[f"Lag {l}" for l in lags],
            text=annot,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            zmin=-0.5,
            zmax=0.5,
            colorbar=dict(title="r", len=0.8) if idx == 2 else dict(title="r", len=0.8, x=-0.05),
            showscale=(idx == 2),
        ), row=1, col=idx)

    fig.update_layout(
        height=350,
        title=dict(text="Steps vs Recovery Metrics Across Lags", font=dict(size=16)),
        margin=dict(l=80, r=20, t=70, b=100),
    )
    for col in [1, 2]:
        fig.update_xaxes(tickangle=-45, row=1, col=col)
    return fig


# ---------------------------------------------------------------------------
# HTML Assembly
# ---------------------------------------------------------------------------

def build_html(
    data: dict[str, pd.DataFrame],
    lag_corrs: dict[str, list[dict]],
    xcorr: dict[str, dict[str, list[dict]]],
    dose_response: dict[str, dict],
    regression: dict[str, dict[str, dict]],
    corr_matrices: dict[str, dict[str, pd.DataFrame]],
    coupling: dict[str, dict],
    comparison: dict,
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Assemble the full HTML report."""
    sections: list[str] = []
    h, m = patients

    # --- KPI Row ---
    h_coup = coupling.get(h.patient_id, {})
    m_coup = coupling.get(m.patient_id, {})

    h_r = _safe_float(h_coup.get("primary_r"))
    m_r = _safe_float(m_coup.get("primary_r"))
    fisher_p = _safe_float(comparison.get("fisher_p", 1.0), 1.0)
    h_rating = h_coup.get("recovery_capacity_rating", "unknown")
    m_rating = m_coup.get("recovery_capacity_rating", "unknown")

    # Status for KPI cards
    def _coupling_status(direction: str) -> str:
        if direction == "positive":
            return "good"
        if direction == "negative":
            return "critical"
        return "warning"

    kpi_row = make_kpi_row(
        make_kpi_card(
            "P1 COUPLING r",
            h_r,
            detail=f"Steps vs HRV (lag 1, n={h_coup.get('primary_n', 0)})",
            status=_coupling_status(h_coup.get("coupling_direction", "absent")),
            status_label=h_coup.get("coupling_direction", "unknown").title(),
            decimals=3,
        ),
        make_kpi_card(
            "P2 COUPLING r",
            m_r,
            detail=f"Steps vs HRV (lag 1, n={m_coup.get('primary_n', 0)})",
            status=_coupling_status(m_coup.get("coupling_direction", "absent")),
            status_label=m_coup.get("coupling_direction", "unknown").title(),
            decimals=3,
        ),
        make_kpi_card(
            "DIFFERENCE (FISHER z)",
            fisher_p,
            detail="p-value for coupling difference",
            status="info" if fisher_p < 0.10 else "neutral",
            status_label="Notable" if fisher_p < 0.10 else "Not significant",
            decimals=3,
        ),
        make_kpi_card(
            "P1 RECOVERY",
            h_rating.replace("_", " ").title(),
            detail=f"{h_coup.get('strength', 'unknown')} {h_coup.get('coupling_direction', '')}",
            status=_coupling_status(h_coup.get("coupling_direction", "absent")),
        ),
        make_kpi_card(
            "P2 RECOVERY",
            m_rating.replace("_", " ").title(),
            detail=f"{m_coup.get('strength', 'unknown')} {m_coup.get('coupling_direction', '')}",
            status=_coupling_status(m_coup.get("coupling_direction", "absent")),
        ),
    )
    sections.append(kpi_row)

    # --- Executive Summary ---
    interp = comparison.get("interpretation", "Analysis complete.")
    h_dir = h_coup.get("coupling_direction", "unknown")
    m_dir = m_coup.get("coupling_direction", "unknown")

    summary_html = (
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;">'
        f"This report examines whether physical activity on day N predicts "
        f"sleep quality and autonomic recovery on day N+1 for both patients. "
        f"<strong>Patient 1</strong> (post-HSCT, ~2700 steps/day, HRV ~9ms) shows "
        f"<strong>{h_dir}</strong> coupling (r={h_r:.3f}). "
        f"<strong>Patient 2</strong> (post-stroke, ~10000 steps/day, HRV ~43ms) shows "
        f"<strong>{m_dir}</strong> coupling (r={m_r:.3f}).</p>"
        f'<p style="color:{TEXT_SECONDARY};line-height:1.7;margin-top:8px;">{interp}</p>'
    )
    sections.append(make_section(
        "Executive Summary",
        summary_html,
        section_id="executive-summary",
    ))

    # --- Scatter Plots ---
    sections.append(section_html_or_placeholder(
        "Scatter Plots",
        lambda: make_section(
            "Activity vs Next-Day Recovery",
            _embed(_fig_dual_scatter(data, patients, "steps", "hrv_average",
                                     "Steps (day N)", "HRV (day N+1)"))
            + _embed(_fig_dual_scatter(data, patients, "steps", "readiness_score",
                                       "Steps (day N)", "Readiness Score (day N+1)")),
            section_id="scatter-plots",
        ),
    ))

    # --- Cross-Correlation ---
    sections.append(section_html_or_placeholder(
        "Cross-Correlation",
        lambda: make_section(
            "Cross-Correlation Functions",
            _embed(_fig_cross_correlation(xcorr, patients, "steps_vs_hrv_average", "Steps vs HRV"))
            + _embed(_fig_cross_correlation(xcorr, patients, "steps_vs_readiness_score", "Steps vs Readiness"))
            + f'<p style="color:{TEXT_TERTIARY};margin-top:12px;">'
            f"Positive lags = activity leads recovery (causal direction). "
            f"Negative lags = recovery leads activity (reverse check). "
            f"* indicates p &lt; 0.10.</p>",
            section_id="cross-correlation",
        ),
    ))

    # --- Dose-Response ---
    sections.append(section_html_or_placeholder(
        "Dose-Response",
        lambda: _build_dose_response_section(dose_response, patients),
    ))

    # --- Correlation Heatmaps ---
    sections.append(section_html_or_placeholder(
        "Correlation Heatmaps",
        lambda: make_section(
            "Full Correlation Matrices (Lag 1)",
            _embed(_fig_correlation_heatmap(corr_matrices, h))
            + _embed(_fig_correlation_heatmap(corr_matrices, m))
            + f'<p style="color:{TEXT_TERTIARY};margin-top:12px;">'
            f"Each cell shows Spearman r for activity[N] vs recovery[N+1]. "
            f"* indicates p &lt; 0.10. Colorscale: blue=positive, red=negative.</p>",
            section_id="heatmaps",
        ),
    ))

    # --- Lagged Heatmap ---
    sections.append(section_html_or_placeholder(
        "Lagged Heatmap",
        lambda: make_section(
            "Steps vs Recovery Across Lags",
            _embed(_fig_lagged_heatmap(data, patients)),
            section_id="lagged-heatmap",
        ),
    ))

    # --- Clinical Interpretation ---
    sections.append(section_html_or_placeholder(
        "Clinical Interpretation",
        lambda: _build_clinical_section(coupling, comparison, patients),
    ))

    body = "\n".join(sections)
    return wrap_html(
        title="Activity-Recovery Coupling",
        body_content=body,
        report_id="comp_coupling",
        subtitle="Module 4: Comparative Activity-Recovery Analysis",
        header_meta="Patient 1 (post-HSCT) vs Patient 2 (post-Stroke)",
    )


def _build_dose_response_section(
    dose_response: dict[str, dict],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Build the dose-response section with stats table."""
    fig_html = _embed(_fig_dose_response(dose_response, patients))

    # Stats table
    rows = ""
    for p in patients:
        dr = dose_response.get(p.patient_id, {})
        n = dr.get("n", 0)
        n_bins = dr.get("n_bins", 0)
        kh = dr.get("kruskal_h", np.nan)
        kp = dr.get("kruskal_p", np.nan)
        kh_str = f"{kh:.2f}" if not np.isnan(kh) else "N/A"
        kp_str = format_p_value(kp) if not np.isnan(kp) else "N/A"
        sig = "Yes" if (not np.isnan(kp) and kp < 0.10) else "No"
        rows += (
            f"<tr><td>{p.display_name}</td><td>{n}</td><td>{n_bins}</td>"
            f"<td>{kh_str}</td><td>{kp_str}</td><td>{sig}</td></tr>"
        )

    table_html = (
        f'<table style="width:100%;border-collapse:collapse;margin-top:16px;">'
        f'<thead><tr style="border-bottom:1px solid {BORDER_SUBTLE};">'
        f'<th style="text-align:left;padding:8px;color:{TEXT_SECONDARY};">Patient</th>'
        f'<th style="text-align:center;padding:8px;color:{TEXT_SECONDARY};">N pairs</th>'
        f'<th style="text-align:center;padding:8px;color:{TEXT_SECONDARY};">Bins</th>'
        f'<th style="text-align:center;padding:8px;color:{TEXT_SECONDARY};">Kruskal H</th>'
        f'<th style="text-align:center;padding:8px;color:{TEXT_SECONDARY};">p-value</th>'
        f'<th style="text-align:center;padding:8px;color:{TEXT_SECONDARY};">Sig (p&lt;0.10)</th>'
        f'</tr></thead><tbody style="color:{TEXT_PRIMARY};">{rows}</tbody></table>'
    )

    return make_section(
        "Dose-Response: Activity Bins vs Next-Day HRV",
        fig_html + table_html,
        section_id="dose-response",
    )


def _build_clinical_section(
    coupling: dict[str, dict],
    comparison: dict,
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Build the clinical interpretation section."""
    h, m = patients
    h_c = coupling.get(h.patient_id, {})
    m_c = coupling.get(m.patient_id, {})

    per_patient_html = ""
    for p, c in [(h, h_c), (m, m_c)]:
        direction = c.get("coupling_direction", "unknown")
        strength = c.get("strength", "unknown")
        rating = c.get("recovery_capacity_rating", "unknown").replace("_", " ").title()
        r_val = _safe_float(c.get("primary_r"))
        p_val = _safe_float(c.get("primary_p", 1.0), 1.0)
        n = c.get("primary_n", 0)

        if direction == "positive":
            clinical_meaning = (
                "More activity is associated with better next-day recovery. "
                "This suggests functional physiological adaptation to physical stress."
            )
        elif direction == "negative":
            clinical_meaning = (
                "More activity is associated with worse next-day recovery. "
                "This may indicate overexertion relative to recovery capacity."
            )
        elif direction == "absent":
            clinical_meaning = (
                "Activity and next-day recovery are effectively uncoupled. "
                "Other factors may dominate recovery dynamics."
            )
        else:
            clinical_meaning = "Insufficient data to assess coupling."

        per_patient_html += (
            f'<div style="padding:16px;background:{BG_ELEVATED};border-radius:8px;'
            f'margin-bottom:12px;border-left:3px solid {PATIENT_COLORS.get(p.patient_id, ACCENT_PURPLE)};">'
            f'<h3 style="margin:0 0 8px 0;color:{TEXT_PRIMARY};">{p.display_name}</h3>'
            f'<p style="color:{TEXT_SECONDARY};line-height:1.6;">'
            f"<strong>Coupling:</strong> {direction} ({strength}) &mdash; "
            f"r={r_val:.3f}, p={format_p_value(p_val)}, n={n}<br>"
            f"<strong>Recovery capacity:</strong> {rating}<br>"
            f"{clinical_meaning}</p></div>"
        )

    # Comparative narrative
    interp = comparison.get("interpretation", "")
    fisher_p = _safe_float(comparison.get("fisher_p", 1.0), 1.0)

    comparative_html = (
        f'<div style="padding:16px;background:{BG_SURFACE};border-radius:8px;margin-top:16px;">'
        f'<h3 style="margin:0 0 8px 0;color:{TEXT_PRIMARY};">Cross-Patient Comparison</h3>'
        f'<p style="color:{TEXT_SECONDARY};line-height:1.6;">{interp}</p>'
        f'<p style="color:{TEXT_TERTIARY};line-height:1.6;margin-top:8px;">'
        f"Fisher r-to-z test p-value: {format_p_value(fisher_p)}</p></div>"
    )

    # Caveats
    h_n = coupling.get(h.patient_id, {}).get("primary_n", 0)
    caveats_html = (
        f'<div style="padding:16px;background:{BG_SURFACE};border-radius:8px;margin-top:16px;'
        f'border:1px solid {BORDER_SUBTLE};">'
        f'<h3 style="margin:0 0 8px 0;color:{ACCENT_AMBER};">Caveats</h3>'
        f'<ul style="color:{TEXT_SECONDARY};line-height:1.8;padding-left:20px;">'
        f"<li>Patient 1 has only ~{h_n} usable lag-1 pairs. With small N, "
        f"many true correlations will not reach statistical significance. "
        f"Effect sizes (r values) are more informative than p-values.</li>"
        f"<li>Relaxed significance threshold (p &lt; 0.10) used throughout "
        f"to balance Type I and Type II error with limited data.</li>"
        f"<li>Observational design: correlations do not establish causation. "
        f"Confounders (illness severity, medication, stress) are not controlled.</li>"
        f"<li>P1's extremely low activity level (~2700 steps, ~6 min active) "
        f"provides limited variance to detect dose-response relationships.</li>"
        f"<li>Oura readiness 'resting_heart_rate' is a 0-100 score, NOT bpm. "
        f"Heart rate values come from sleep_periods only.</li>"
        f"</ul></div>"
    )

    return make_section(
        "Clinical Interpretation",
        per_patient_html + comparative_html + caveats_html,
        section_id="clinical-interpretation",
    )


# ---------------------------------------------------------------------------
# JSON Export
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    """Recursively sanitize NaN/Inf for JSON serialization."""
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
    if isinstance(obj, pd.DataFrame):
        return _sanitize(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def export_json(
    lag_corrs: dict[str, list[dict]],
    xcorr: dict[str, dict[str, list[dict]]],
    dose_response: dict[str, dict],
    regression: dict[str, dict[str, dict]],
    corr_matrices: dict[str, dict[str, pd.DataFrame]],
    coupling: dict[str, dict],
    comparison: dict,
) -> None:
    """Write structured metrics JSON."""
    # Extract lag-1 correlations only for concise output
    lag1_corrs: dict[str, list[dict]] = {}
    for pid, records in lag_corrs.items():
        lag1_corrs[pid] = [r for r in records if r["lag"] == 1]

    # Convert correlation matrices to serializable form
    corr_json: dict[str, dict] = {}
    for pid, mat in corr_matrices.items():
        corr_json[pid] = {
            "r": mat["r"].to_dict() if not mat["r"].empty else {},
            "p": mat["p"].to_dict() if not mat["p"].empty else {},
        }

    output = {
        "report": "comparative_activity_recovery_coupling",
        "report_id": "comp_coupling",
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "lag1_correlations": lag1_corrs,
        "cross_correlations": xcorr,
        "dose_response": dose_response,
        "regression": regression,
        "correlation_matrices": corr_json,
        "coupling_assessment": coupling,
        "cross_patient_comparison": comparison,
    }

    output = _sanitize(output)
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("JSON metrics written to %s", JSON_OUTPUT)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run comparative activity-recovery coupling analysis pipeline."""
    logger.info("[1/7] Loading patient data...")
    patients = default_patients()
    if patients[1] is None:
        print("Skipping: mitch.db not found (second patient data not available)")
        return 0
    data = load_data(patients)

    for pid, df in data.items():
        if df.empty:
            logger.warning("No data for patient %s", pid)

    logger.info("[2/7] Computing lag correlations (lags 0, 1, 2)...")
    lag_corrs = compute_lag_correlations(data, lags=[0, 1, 2])

    logger.info("[3/7] Computing cross-correlation functions...")
    xcorr = compute_cross_correlations(data)

    logger.info("[4/7] Computing dose-response analysis...")
    dose_response_result = compute_dose_response(data, "steps", "hrv_average")

    logger.info("[5/7] Computing regression and correlation matrices...")
    regression = compute_regression(data)
    corr_matrices = compute_correlation_matrix(data, lag=1)

    logger.info("[6/7] Assessing coupling and cross-patient comparison...")
    coupling = assess_coupling(lag_corrs, patients)
    comparison = compare_patients(coupling, lag_corrs, dose_response_result, patients)

    logger.info("[7/7] Generating HTML report...")
    html = build_html(
        data, lag_corrs, xcorr, dose_response_result, regression,
        corr_matrices, coupling, comparison, patients,
    )
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", HTML_OUTPUT)

    export_json(
        lag_corrs, xcorr, dose_response_result, regression,
        corr_matrices, coupling, comparison,
    )

    logger.info("Comparative activity-recovery coupling analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
