#!/usr/bin/env python3
"""
Ruxolitinib Dose-Response Forecast

Forward projection of HRV and HR recovery under ruxolitinib treatment.
Fits linear and exponential models to post-treatment data, bootstraps
confidence intervals, and estimates when clinical thresholds will be crossed.

Outputs:
  - Interactive HTML: reports/rux_forecast.html
  - Structured JSON:  reports/rux_forecast.json

Usage:
    python analysis/analyze_rux_forecast.py
"""
from __future__ import annotations

import json
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
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Path resolution & config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATABASE_PATH,
    REPORTS_DIR,
    TREATMENT_START,
    KNOWN_EVENT_DATE,
    DATA_START,
    ESC_RMSSD_DEFICIENCY,
    POPULATION_RMSSD_MEDIAN,
    POPULATION_RMSSD_MEAN,
    FONT_FAMILY,
    NORM_RMSSD_P25,
    HSCT_RMSSD_RANGE,
)
from _hardening import safe_connect, safe_read_sql, section_html_or_placeholder
from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    COLORWAY, BG_PRIMARY, BG_SURFACE, BG_ELEVATED,
    BORDER_SUBTLE, BORDER_DEFAULT, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    ACCENT_PURPLE, ACCENT_CYAN, ACCENT_ORANGE,
    C_PRE_TX, C_POST_TX, C_RUX_LINE, C_FORECAST, C_COUNTERFACTUAL,
    C_HRV, C_HR,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "rux_forecast.html"
JSON_OUTPUT = REPORTS_DIR / "rux_forecast.json"

TODAY = date.today()

# ---------------------------------------------------------------------------
# Clinical thresholds
# ---------------------------------------------------------------------------
HRV_TARGETS = {
    "esc_15ms": {"value": 15, "label": "ESC Threshold (15 ms)", "color": ACCENT_RED},
    "hsct_25ms": {"value": 25, "label": "HSCT Range Low (25 ms)", "color": ACCENT_AMBER},
    "pop_p25_36ms": {"value": 36, "label": "Population 25th pct (36 ms)", "color": ACCENT_GREEN},
}

HR_TARGETS = {
    "normal_70": {"value": 70, "label": "Normal (<70 bpm)", "color": ACCENT_GREEN},
    "good_65": {"value": 65, "label": "Good (<65 bpm)", "color": ACCENT_CYAN},
    "excellent_60": {"value": 60, "label": "Excellent (<60 bpm)", "color": ACCENT_PURPLE},
}

EFFICIENCY_TARGETS = {
    "healthy_85": {"value": 85, "label": "Healthy (>85%)", "color": ACCENT_GREEN},
}

BOOTSTRAP_N = 1000
FORECAST_DAYS = 365  # max projection horizon

# Phase boundaries
ACUTE_DATE = KNOWN_EVENT_DATE  # Feb 9
RUX_DATE = TREATMENT_START     # Mar 16


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sleep_data(conn) -> pd.DataFrame:
    """Load long_sleep periods with HRV, HR, and efficiency."""
    sql = """
        SELECT day, average_hrv, average_heart_rate, lowest_heart_rate, efficiency
        FROM oura_sleep_periods
        WHERE type = 'long_sleep'
        ORDER BY day
    """
    df = safe_read_sql(sql, conn, label="sleep_periods", required=True)
    if df.empty:
        return df

    df["day"] = pd.to_datetime(df["day"], errors="coerce", utc=True)
    df = df.dropna(subset=["day"])
    df["day_date"] = df["day"].dt.date
    return df


# ---------------------------------------------------------------------------
# Phase splitting
# ---------------------------------------------------------------------------

def split_phases(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split data into pre-acute, post-acute-pre-rux, and post-rux."""
    pre = df[df["day_date"] < ACUTE_DATE].copy()
    mid = df[(df["day_date"] >= ACUTE_DATE) & (df["day_date"] < RUX_DATE)].copy()
    post = df[df["day_date"] >= RUX_DATE].copy()
    return {"pre_acute": pre, "post_acute_pre_rux": mid, "post_rux": post}


# ---------------------------------------------------------------------------
# Regression & forecasting
# ---------------------------------------------------------------------------

def linear_regression(days: np.ndarray, values: np.ndarray) -> dict:
    """Fit OLS linear regression, return slope/intercept/r2."""
    if len(days) < 3:
        return {"slope": 0.0, "intercept": float(np.nanmean(values)),
                "r2": 0.0, "n": len(days)}
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(days, values)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n": int(len(days)),
    }


def exp_recovery(t: np.ndarray, a: float, tau: float, baseline: float) -> np.ndarray:
    """Exponential recovery: y = a * (1 - exp(-t/tau)) + baseline."""
    return a * (1.0 - np.exp(-t / tau)) + baseline


def fit_exponential(days: np.ndarray, values: np.ndarray) -> dict | None:
    """Try fitting exponential recovery curve. Returns None if it fails."""
    if len(days) < 5:
        return None
    try:
        y_min, y_max = float(np.nanmin(values)), float(np.nanmax(values))
        y_range = max(y_max - y_min, 1.0)
        p0 = [y_range * 2, 30.0, y_min]
        bounds = ([0, 1, -200], [500, 1000, 200])
        popt, pcov = curve_fit(exp_recovery, days, values, p0=p0, bounds=bounds,
                               maxfev=5000)
        y_pred = exp_recovery(days, *popt)
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n = len(values)
        k_lin = 2
        k_exp = 3
        aic_exp = n * np.log(ss_res / n + 1e-15) + 2 * k_exp
        bic_exp = n * np.log(ss_res / n + 1e-15) + k_exp * np.log(n)
        return {
            "a": float(popt[0]),
            "tau": float(popt[1]),
            "baseline": float(popt[2]),
            "r2": float(r2),
            "aic": float(aic_exp),
            "bic": float(bic_exp),
            "n": n,
        }
    except (RuntimeError, ValueError, TypeError):
        return None


def compute_aic_bic_linear(days: np.ndarray, values: np.ndarray, reg: dict) -> dict:
    """Compute AIC/BIC for linear model."""
    n = len(values)
    y_pred = reg["slope"] * days + reg["intercept"]
    ss_res = np.sum((values - y_pred) ** 2)
    k = 2
    aic = n * np.log(ss_res / n + 1e-15) + 2 * k
    bic = n * np.log(ss_res / n + 1e-15) + k * np.log(n)
    return {"aic": float(aic), "bic": float(bic)}


def crossing_date(
    slope: float, intercept: float, target: float, ref_date: date, direction: str,
) -> date | None:
    """Compute the date when a linear trend crosses a target value.

    direction: 'above' (metric needs to rise above target) or 'below' (needs to drop below).
    """
    if slope == 0:
        return None
    days_to_target = (target - intercept) / slope
    if days_to_target < 0:
        return None
    # For HR (direction='below'): slope must be negative
    if direction == "below" and slope >= 0:
        return None
    # For HRV (direction='above'): slope must be positive
    if direction == "above" and slope <= 0:
        return None
    cross = ref_date + timedelta(days=int(days_to_target))
    max_date = ref_date + timedelta(days=FORECAST_DAYS)
    if cross > max_date:
        return None
    return cross


def crossing_date_exp(
    params: dict, target: float, ref_date: date, direction: str,
) -> date | None:
    """Compute crossing date for exponential model."""
    a, tau, baseline = params["a"], params["tau"], params["baseline"]
    # y = a * (1 - exp(-t/tau)) + baseline = target
    # => 1 - exp(-t/tau) = (target - baseline) / a
    ratio = (target - baseline) / a if a != 0 else None
    if ratio is None or ratio <= 0 or ratio >= 1:
        return None
    t = -tau * np.log(1 - ratio)
    if t < 0:
        return None
    cross = ref_date + timedelta(days=int(t))
    max_date = ref_date + timedelta(days=FORECAST_DAYS)
    if cross > max_date:
        return None
    # Check direction
    if direction == "above" and a <= 0:
        return None
    if direction == "below" and a >= 0:
        return None
    return cross


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_slopes(
    days: np.ndarray, values: np.ndarray, n_iter: int = BOOTSTRAP_N,
) -> np.ndarray:
    """Bootstrap the linear regression slope."""
    rng = np.random.default_rng(42)
    slopes = np.empty(n_iter)
    n = len(days)
    for i in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        s, _, _, _, _ = scipy_stats.linregress(days[idx], values[idx])
        slopes[i] = s
    return slopes


def bootstrap_crossing_dates(
    slopes: np.ndarray, intercept: float, target: float,
    ref_date: date, direction: str,
) -> dict:
    """Get percentile crossing dates from bootstrapped slopes."""
    dates = []
    for s in slopes:
        d = crossing_date(s, intercept, target, ref_date, direction)
        if d is not None:
            dates.append(d)
    if not dates:
        return {"p50": None, "p75": None, "p95": None, "n_valid": 0,
                "n_total": len(slopes)}
    dates_sorted = sorted(dates)
    n = len(dates_sorted)
    return {
        "p50": dates_sorted[int(n * 0.50)],
        "p75": dates_sorted[int(min(n * 0.75, n - 1))],
        "p95": dates_sorted[int(min(n * 0.95, n - 1))],
        "n_valid": n,
        "n_total": len(slopes),
    }


# ---------------------------------------------------------------------------
# Per-metric forecast computation
# ---------------------------------------------------------------------------

def forecast_metric(
    df_post: pd.DataFrame, col: str, targets: dict, direction: str,
    metric_name: str,
) -> dict:
    """Run full forecast pipeline for one metric."""
    vals = df_post[[col]].dropna()
    if vals.empty or len(vals) < 3:
        return {
            "metric": metric_name,
            "status": "insufficient_data",
            "n_points": len(vals),
        }

    # Days since rux start
    day_nums = np.array([(d - RUX_DATE).days for d in df_post.loc[vals.index, "day_date"]])
    values = vals[col].values.astype(float)

    # Current stats
    current_mean = float(np.nanmean(values[-7:])) if len(values) >= 7 else float(np.nanmean(values))
    current_last = float(values[-1])

    # Linear regression
    reg = linear_regression(day_nums, values)
    slope_per_day = reg["slope"]
    slope_per_week = slope_per_day * 7

    # Check direction — if no improvement, report it
    improving = (direction == "above" and slope_per_day > 0) or \
                (direction == "below" and slope_per_day < 0)

    # AIC/BIC for linear
    lin_ic = compute_aic_bic_linear(day_nums, values, reg)

    # Exponential fit
    exp_fit = fit_exponential(day_nums, values)

    # Determine better model
    best_model = "linear"
    if exp_fit is not None:
        if exp_fit["aic"] < lin_ic["aic"]:
            best_model = "exponential"

    # Bootstrap slopes
    slopes = bootstrap_slopes(day_nums, values)
    slope_ci = {
        "p5": float(np.percentile(slopes, 5)),
        "p25": float(np.percentile(slopes, 25)),
        "p50": float(np.percentile(slopes, 50)),
        "p75": float(np.percentile(slopes, 75)),
        "p95": float(np.percentile(slopes, 95)),
    }

    # Three scenarios
    scenarios = {
        "optimistic": slope_ci["p75"] if direction == "above" else slope_ci["p25"],
        "expected": slope_ci["p50"],
        "conservative": slope_ci["p25"] if direction == "above" else slope_ci["p75"],
    }

    # Forecast targets
    target_results = {}
    for tkey, tinfo in targets.items():
        # Already past target?
        already_met = (direction == "above" and current_mean >= tinfo["value"]) or \
                      (direction == "below" and current_mean <= tinfo["value"])

        if already_met:
            target_results[tkey] = {
                "value": tinfo["value"],
                "label": tinfo["label"],
                "status": "already_met",
                "expected_date": None,
                "ci_95": [None, None],
                "scenarios": {},
            }
            continue

        if not improving:
            target_results[tkey] = {
                "value": tinfo["value"],
                "label": tinfo["label"],
                "status": "insufficient_improvement",
                "expected_date": None,
                "ci_95": [None, None],
                "scenarios": {},
            }
            continue

        # Linear crossing (expected)
        expected_date = crossing_date(
            slope_per_day, reg["intercept"], tinfo["value"], RUX_DATE, direction,
        )

        # Exponential crossing
        exp_date = None
        if exp_fit is not None:
            exp_date = crossing_date_exp(exp_fit, tinfo["value"], RUX_DATE, direction)

        # Bootstrap CI
        boot_dates = bootstrap_crossing_dates(
            slopes, reg["intercept"], tinfo["value"], RUX_DATE, direction,
        )

        # Scenario dates
        scen_dates = {}
        for sname, sslope in scenarios.items():
            scen_dates[sname] = crossing_date(
                sslope, reg["intercept"], tinfo["value"], RUX_DATE, direction,
            )

        # Use best model date
        best_date = exp_date if best_model == "exponential" and exp_date else expected_date

        target_results[tkey] = {
            "value": tinfo["value"],
            "label": tinfo["label"],
            "status": "projected",
            "expected_date": str(best_date) if best_date else None,
            "linear_date": str(expected_date) if expected_date else None,
            "exp_date": str(exp_date) if exp_date else None,
            "ci_95": [
                str(boot_dates["p50"]) if boot_dates["p50"] else None,
                str(boot_dates["p95"]) if boot_dates["p95"] else None,
            ],
            "scenarios": {k: str(v) if v else None for k, v in scen_dates.items()},
            "bootstrap_valid_pct": (
                round(100 * boot_dates["n_valid"] / boot_dates["n_total"], 1)
                if boot_dates["n_total"] > 0 else 0
            ),
        }

    return {
        "metric": metric_name,
        "status": "forecast_available" if improving else "no_improvement",
        "n_points": len(values),
        "current_mean_7d": round(current_mean, 2),
        "current_last": round(current_last, 2),
        "slope_per_day": round(slope_per_day, 4),
        "slope_per_week": round(slope_per_week, 3),
        "r2": round(reg["r2"], 4),
        "best_model": best_model,
        "linear": {
            "slope": round(reg["slope"], 4),
            "intercept": round(reg["intercept"], 2),
            "r2": round(reg["r2"], 4),
            "aic": round(lin_ic["aic"], 2),
            "bic": round(lin_ic["bic"], 2),
        },
        "exponential": {
            "a": round(exp_fit["a"], 2),
            "tau": round(exp_fit["tau"], 2),
            "baseline": round(exp_fit["baseline"], 2),
            "r2": round(exp_fit["r2"], 4),
            "aic": round(exp_fit["aic"], 2),
            "bic": round(exp_fit["bic"], 2),
        } if exp_fit else None,
        "slope_ci": {k: round(v, 5) for k, v in slope_ci.items()},
        "targets": target_results,
        "day_nums": day_nums.tolist(),
        "values": values.tolist(),
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _date_range_for_forecast(ref_date: date, max_days: int = 180) -> list[date]:
    """Generate a date range for forecast projection."""
    return [ref_date + timedelta(days=d) for d in range(max_days)]


def build_trajectory_chart(
    df_all: pd.DataFrame, phases: dict, fc_hrv: dict, fc_hr: dict,
) -> str:
    """Build HRV and HR trajectory chart with forecast bands."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("HRV (RMSSD) Trajectory & Forecast", "Heart Rate Trajectory & Forecast"),
    )

    # --- HRV subplot (row 1) ---
    for phase_name, phase_color, phase_label in [
        ("pre_acute", C_PRE_TX, "Pre-Acute"),
        ("post_acute_pre_rux", ACCENT_AMBER, "Post-Acute"),
        ("post_rux", ACCENT_BLUE, "Post-Rux"),
    ]:
        pdf = phases[phase_name]
        if pdf.empty:
            continue
        hrv_vals = pdf.dropna(subset=["average_hrv"])
        if hrv_vals.empty:
            continue
        fig.add_trace(go.Scatter(
            x=hrv_vals["day"], y=hrv_vals["average_hrv"],
            mode="markers",
            marker=dict(color=phase_color, size=6, opacity=0.8),
            name=f"HRV {phase_label}",
            legendgroup=f"hrv_{phase_name}",
            hovertemplate="%{x|%b %d}: %{y:.1f} ms<extra></extra>",
        ), row=1, col=1)

    # HRV regression + forecast
    if fc_hrv.get("status") == "forecast_available":
        _add_forecast_line(fig, fc_hrv, row=1, metric_label="HRV",
                           color=C_FORECAST)
        # Target lines
        for tkey, tinfo in HRV_TARGETS.items():
            fig.add_shape(
                type="line",
                x0=df_all["day"].min(), x1=pd.Timestamp(TODAY + timedelta(days=120), tz="UTC"),
                y0=tinfo["value"], y1=tinfo["value"],
                line=dict(color=tinfo["color"], width=1.5, dash="dot"),
                row=1, col=1,
            )
            fig.add_annotation(
                x=pd.Timestamp(TODAY + timedelta(days=100), tz="UTC"),
                y=tinfo["value"] + 1,
                text=tinfo["label"],
                showarrow=False,
                font=dict(size=10, color=tinfo["color"]),
                row=1, col=1,
            )

    # --- HR subplot (row 2) ---
    for phase_name, phase_color, phase_label in [
        ("pre_acute", C_PRE_TX, "Pre-Acute"),
        ("post_acute_pre_rux", ACCENT_AMBER, "Post-Acute"),
        ("post_rux", ACCENT_BLUE, "Post-Rux"),
    ]:
        pdf = phases[phase_name]
        if pdf.empty:
            continue
        hr_vals = pdf.dropna(subset=["lowest_heart_rate"])
        if hr_vals.empty:
            continue
        fig.add_trace(go.Scatter(
            x=hr_vals["day"], y=hr_vals["lowest_heart_rate"],
            mode="markers",
            marker=dict(color=phase_color, size=6, opacity=0.8),
            name=f"Lowest HR {phase_label}",
            legendgroup=f"hr_{phase_name}",
            showlegend=False,
            hovertemplate="%{x|%b %d}: %{y:.0f} bpm<extra></extra>",
        ), row=2, col=1)

    # HR regression + forecast
    if fc_hr.get("status") == "forecast_available":
        _add_forecast_line(fig, fc_hr, row=2, metric_label="HR",
                           color=C_FORECAST)
        for tkey, tinfo in HR_TARGETS.items():
            fig.add_shape(
                type="line",
                x0=df_all["day"].min(), x1=pd.Timestamp(TODAY + timedelta(days=120), tz="UTC"),
                y0=tinfo["value"], y1=tinfo["value"],
                line=dict(color=tinfo["color"], width=1.5, dash="dot"),
                row=2, col=1,
            )
            fig.add_annotation(
                x=pd.Timestamp(TODAY + timedelta(days=100), tz="UTC"),
                y=tinfo["value"] - 1,
                text=tinfo["label"],
                showarrow=False,
                font=dict(size=10, color=tinfo["color"]),
                row=2, col=1,
            )

    # Rux start vertical line (both subplots)
    # In make_subplots, first y-axis is "y", second is "y2"
    yref_map = {1: "y domain", 2: "y2 domain"}
    for row in [1, 2]:
        fig.add_shape(
            type="line",
            x0=pd.Timestamp(RUX_DATE, tz="UTC"),
            x1=pd.Timestamp(RUX_DATE, tz="UTC"),
            y0=0, y1=1, yref=yref_map[row],
            line=dict(color=C_RUX_LINE, width=2, dash="dash"),
            row=row, col=1,
        )
        fig.add_annotation(
            x=pd.Timestamp(RUX_DATE, tz="UTC"), y=1, yref=yref_map[row],
            text="Rux Start", showarrow=False,
            font=dict(size=10, color=C_RUX_LINE),
            yshift=12, row=row, col=1,
        )

    fig.update_yaxes(title_text="RMSSD (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Lowest HR (bpm)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def _add_forecast_line(fig, fc: dict, row: int, metric_label: str, color: str):
    """Add regression line + CI band to a subplot."""
    if "day_nums" not in fc or "values" not in fc:
        return
    day_nums = np.array(fc["day_nums"])
    slope = fc["linear"]["slope"]
    intercept = fc["linear"]["intercept"]

    # Actual regression line over observed data
    x_obs = [pd.Timestamp(RUX_DATE + timedelta(days=int(d)), tz="UTC") for d in day_nums]
    y_obs = slope * day_nums + intercept

    # Future projection
    future_days = np.arange(0, 150)
    x_fut = [pd.Timestamp(RUX_DATE + timedelta(days=int(d)), tz="UTC") for d in future_days]
    y_fut = slope * future_days + intercept

    # CI band from bootstrap
    slope_lo = fc["slope_ci"]["p5"]
    slope_hi = fc["slope_ci"]["p95"]
    y_lo = slope_lo * future_days + intercept
    y_hi = slope_hi * future_days + intercept

    # CI band (shaded)
    fig.add_trace(go.Scatter(
        x=x_fut + x_fut[::-1],
        y=list(y_hi) + list(y_lo[::-1]),
        fill="toself",
        fillcolor=f"rgba(6,182,212,0.12)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ), row=row, col=1)

    # Regression line
    fig.add_trace(go.Scatter(
        x=x_fut, y=list(y_fut),
        mode="lines",
        line=dict(color=color, width=2, dash="dash"),
        name=f"{metric_label} Forecast",
        hovertemplate="%{x|%b %d}: %{y:.1f}<extra>Forecast</extra>",
    ), row=row, col=1)


def build_milestone_chart(fc_hrv: dict, fc_hr: dict) -> str:
    """Build horizontal bar chart of milestone ETAs."""
    milestones = []

    for metric_name, fc, targets in [
        ("HRV", fc_hrv, HRV_TARGETS),
        ("Lowest HR", fc_hr, HR_TARGETS),
    ]:
        if fc.get("status") not in ("forecast_available", "no_improvement"):
            continue
        for tkey, tinfo in targets.items():
            tr = fc.get("targets", {}).get(tkey, {})
            status = tr.get("status", "unknown")
            label = f"{metric_name}: {tinfo['label']}"

            if status == "already_met":
                milestones.append({
                    "label": label, "days": 0, "color": ACCENT_GREEN,
                    "status": "Met", "ci_lo": 0, "ci_hi": 0,
                })
            elif status == "projected":
                exp_date_str = tr.get("expected_date")
                if exp_date_str:
                    exp_date = date.fromisoformat(exp_date_str)
                    days_until = (exp_date - TODAY).days
                    ci_lo_str = tr.get("ci_95", [None, None])[0]
                    ci_hi_str = tr.get("ci_95", [None, None])[1]
                    ci_lo = (date.fromisoformat(ci_lo_str) - TODAY).days if ci_lo_str else days_until
                    ci_hi = (date.fromisoformat(ci_hi_str) - TODAY).days if ci_hi_str else days_until
                    milestones.append({
                        "label": label, "days": max(0, days_until),
                        "color": tinfo["color"],
                        "status": f"~{exp_date_str}",
                        "ci_lo": max(0, ci_lo), "ci_hi": max(0, ci_hi),
                    })
            elif status == "insufficient_improvement":
                milestones.append({
                    "label": label, "days": FORECAST_DAYS,
                    "color": TEXT_TERTIARY,
                    "status": "No improvement trend",
                    "ci_lo": FORECAST_DAYS, "ci_hi": FORECAST_DAYS,
                })

    if not milestones:
        return "<p style='color:#9CA3AF;'>No milestones to display.</p>"

    fig = go.Figure()

    labels = [m["label"] for m in milestones]
    days = [m["days"] for m in milestones]
    colors = [m["color"] for m in milestones]
    texts = [m["status"] for m in milestones]
    errors_minus = [max(0, m["days"] - m["ci_lo"]) for m in milestones]
    errors_plus = [max(0, m["ci_hi"] - m["days"]) for m in milestones]

    fig.add_trace(go.Bar(
        y=labels,
        x=days,
        orientation="h",
        marker=dict(color=colors, opacity=0.8),
        text=texts,
        textposition="outside",
        textfont=dict(size=11),
        error_x=dict(
            type="data",
            symmetric=False,
            array=errors_plus,
            arrayminus=errors_minus,
            color=TEXT_SECONDARY,
            thickness=1.5,
            width=5,
        ),
        hovertemplate="%{y}: %{x} days from now<extra></extra>",
    ))

    fig.update_layout(
        title="Milestone Timeline (Days from Today)",
        height=max(250, 55 * len(milestones)),
        xaxis_title="Days from Today",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=250, r=80, t=50, b=40),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_model_comparison_chart(
    df_post: pd.DataFrame, fc_hrv: dict,
) -> str:
    """Overlay linear vs exponential fit for HRV."""
    if fc_hrv.get("status") != "forecast_available":
        return "<p style='color:#9CA3AF;'>Insufficient data for model comparison.</p>"

    hrv = df_post.dropna(subset=["average_hrv"])
    if hrv.empty:
        return "<p style='color:#9CA3AF;'>No HRV data.</p>"

    day_nums = np.array([(d - RUX_DATE).days for d in hrv["day_date"]])
    values = hrv["average_hrv"].values.astype(float)

    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(
        x=hrv["day"], y=values,
        mode="markers",
        marker=dict(color=C_HRV, size=8),
        name="Observed HRV",
    ))

    # Linear fit (extend into future)
    future_days = np.arange(0, 120)
    future_dates = [pd.Timestamp(RUX_DATE + timedelta(days=int(d)), tz="UTC")
                    for d in future_days]
    lin_slope = fc_hrv["linear"]["slope"]
    lin_int = fc_hrv["linear"]["intercept"]
    lin_y = lin_slope * future_days + lin_int
    lin_r2 = fc_hrv["linear"]["r2"]
    lin_aic = fc_hrv["linear"]["aic"]

    fig.add_trace(go.Scatter(
        x=future_dates, y=lin_y,
        mode="lines",
        line=dict(color=ACCENT_BLUE, width=2, dash="dash"),
        name=f"Linear (R²={lin_r2:.3f}, AIC={lin_aic:.1f})",
    ))

    # Exponential fit
    exp_params = fc_hrv.get("exponential")
    if exp_params:
        exp_y = exp_recovery(future_days, exp_params["a"], exp_params["tau"],
                             exp_params["baseline"])
        exp_r2 = exp_params["r2"]
        exp_aic = exp_params["aic"]
        fig.add_trace(go.Scatter(
            x=future_dates, y=list(exp_y),
            mode="lines",
            line=dict(color=ACCENT_ORANGE, width=2),
            name=f"Exponential (R²={exp_r2:.3f}, AIC={exp_aic:.1f})",
        ))

    # Target lines
    for tkey, tinfo in HRV_TARGETS.items():
        fig.add_shape(
            type="line",
            x0=hrv["day"].min(), x1=future_dates[-1],
            y0=tinfo["value"], y1=tinfo["value"],
            line=dict(color=tinfo["color"], width=1, dash="dot"),
        )

    best = fc_hrv["best_model"]
    fig.update_layout(
        title=f"HRV Recovery Model Comparison (best: {best})",
        yaxis_title="RMSSD (ms)",
        xaxis_title="Date",
        height=420,
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_phase_chart(phases: dict) -> str:
    """Bar chart of mean values across three phases."""
    metrics_cfg = [
        ("average_hrv", "HRV (ms)", C_HRV),
        ("lowest_heart_rate", "Lowest HR (bpm)", C_HR),
        ("efficiency", "Efficiency (%)", ACCENT_BLUE),
    ]
    phase_labels = ["Pre-Acute", "Post-Acute / Pre-Rux", "Post-Rux"]
    phase_keys = ["pre_acute", "post_acute_pre_rux", "post_rux"]
    phase_colors = [C_PRE_TX, ACCENT_AMBER, ACCENT_BLUE]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m[1] for m in metrics_cfg],
        horizontal_spacing=0.08,
    )

    for col_idx, (col, label, _) in enumerate(metrics_cfg, 1):
        means = []
        sds = []
        for pk in phase_keys:
            pdf = phases[pk]
            vals = pdf[col].dropna()
            means.append(float(vals.mean()) if not vals.empty else 0)
            sds.append(float(vals.std()) if len(vals) > 1 else 0)

        fig.add_trace(go.Bar(
            x=phase_labels,
            y=means,
            marker=dict(color=phase_colors, opacity=0.85),
            error_y=dict(type="data", array=sds, visible=True,
                         color=TEXT_SECONDARY, thickness=1),
            text=[f"{m:.1f}" for m in means],
            textposition="outside",
            textfont=dict(size=11),
            showlegend=False,
            hovertemplate="%{x}: %{y:.1f}<extra></extra>",
        ), row=1, col=col_idx)

    fig.update_layout(
        title="Three-Phase Recovery Context (Mean +/- SD)",
        height=380,
        margin=dict(t=70, b=40),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


# ---------------------------------------------------------------------------
# HTML section builders
# ---------------------------------------------------------------------------

def build_kpi_section(fc_hrv: dict, fc_hr: dict, days_on_rux: int) -> str:
    """Top KPI row: current metrics + first target ETA."""
    hrv_val = fc_hrv.get("current_mean_7d", "N/A")
    hr_val = fc_hr.get("current_mean_7d", "N/A")

    # HRV status
    hrv_status = "critical"
    hrv_status_label = "Deficient"
    if isinstance(hrv_val, (int, float)):
        if hrv_val >= ESC_RMSSD_DEFICIENCY:
            hrv_status = "warning"
            hrv_status_label = "Low"
        if hrv_val >= NORM_RMSSD_P25:
            hrv_status = "good"
            hrv_status_label = "Normal"

    # Find first HRV target ETA
    first_eta = "N/A"
    if fc_hrv.get("targets"):
        for tkey in ["esc_15ms", "hsct_25ms", "pop_p25_36ms"]:
            tr = fc_hrv["targets"].get(tkey, {})
            if tr.get("status") == "already_met":
                first_eta = "Already met"
                break
            if tr.get("status") == "projected" and tr.get("expected_date"):
                first_eta = tr["expected_date"]
                break
            if tr.get("status") == "insufficient_improvement":
                first_eta = "No trend"
                break

    return make_kpi_row(
        make_kpi_card(
            "HRV (7-DAY)",
            hrv_val,
            unit="ms",
            status=hrv_status,
            status_label=hrv_status_label,
            detail=f"Target: {ESC_RMSSD_DEFICIENCY} ms ESC threshold",
            explainer="Average RMSSD over past 7 nights",
        ),
        make_kpi_card(
            "LOWEST HR (7-DAY)",
            hr_val,
            unit="bpm",
            status="warning" if isinstance(hr_val, (int, float)) and hr_val > 70 else "good",
            status_label="Elevated" if isinstance(hr_val, (int, float)) and hr_val > 70 else "Normal",
            detail="Target: <70 bpm",
            explainer="Mean lowest heart rate over past 7 nights",
        ),
        make_kpi_card(
            "DAYS ON RUX",
            days_on_rux,
            unit="days",
            status="info",
            decimals=0,
            detail=f"Since {TREATMENT_START}",
            explainer="Days since ruxolitinib initiation",
        ),
        make_kpi_card(
            "FIRST TARGET ETA",
            first_eta,
            status="info" if first_eta not in ("N/A", "No trend") else "warning",
            status_label="Projected" if first_eta not in ("N/A", "No trend", "Already met") else "",
            detail="HRV reaching ESC 15 ms threshold",
            explainer="Expected date based on current trajectory",
        ),
    )


def build_forecast_summary(fc_hrv: dict, fc_hr: dict, days_on_rux: int) -> str:
    """Doctor-ready forecast summary card."""
    lines = []
    lines.append(f"<strong>Treatment duration:</strong> {days_on_rux} days on ruxolitinib")
    lines.append("")

    # HRV section
    lines.append("<strong>HRV (RMSSD) Forecast:</strong>")
    if fc_hrv.get("status") == "forecast_available":
        lines.append(
            f"&bull; Current 7-day mean: {fc_hrv['current_mean_7d']:.1f} ms "
            f"(slope: {fc_hrv['slope_per_week']:+.2f} ms/week, "
            f"R&sup2;={fc_hrv['r2']:.3f})"
        )
        lines.append(f"&bull; Best model: {fc_hrv['best_model']}")
        for tkey in ["esc_15ms", "hsct_25ms", "pop_p25_36ms"]:
            tr = fc_hrv.get("targets", {}).get(tkey, {})
            if tr.get("status") == "already_met":
                lines.append(f"&bull; {tr['label']}: <span style='color:{ACCENT_GREEN};'>Already met</span>")
            elif tr.get("status") == "projected":
                ci = tr.get("ci_95", [None, None])
                ci_str = ""
                if ci[0] and ci[1]:
                    ci_str = f" (50-95% CI: {ci[0]} to {ci[1]})"
                lines.append(
                    f"&bull; {tr['label']}: <span style='color:{ACCENT_CYAN};'>"
                    f"{tr.get('expected_date', 'N/A')}</span>{ci_str}"
                )
            elif tr.get("status") == "insufficient_improvement":
                lines.append(
                    f"&bull; {tr['label']}: <span style='color:{ACCENT_RED};'>"
                    f"Insufficient improvement for projection</span>"
                )
    elif fc_hrv.get("status") == "no_improvement":
        lines.append(
            f"&bull; Current 7-day mean: {fc_hrv.get('current_mean_7d', 'N/A')} ms"
        )
        lines.append(
            f"&bull; <span style='color:{ACCENT_RED};'>No positive trend detected "
            f"(slope: {fc_hrv.get('slope_per_week', 0):+.2f} ms/week)</span>"
        )
    else:
        lines.append("&bull; Insufficient data for forecast")

    lines.append("")

    # HR section
    lines.append("<strong>Heart Rate Forecast:</strong>")
    if fc_hr.get("status") == "forecast_available":
        lines.append(
            f"&bull; Current 7-day mean lowest HR: {fc_hr['current_mean_7d']:.1f} bpm "
            f"(slope: {fc_hr['slope_per_week']:+.2f} bpm/week)"
        )
        for tkey in ["normal_70", "good_65", "excellent_60"]:
            tr = fc_hr.get("targets", {}).get(tkey, {})
            if tr.get("status") == "already_met":
                lines.append(f"&bull; {tr['label']}: <span style='color:{ACCENT_GREEN};'>Already met</span>")
            elif tr.get("status") == "projected":
                lines.append(
                    f"&bull; {tr['label']}: <span style='color:{ACCENT_CYAN};'>"
                    f"{tr.get('expected_date', 'N/A')}</span>"
                )
            elif tr.get("status") == "insufficient_improvement":
                lines.append(
                    f"&bull; {tr['label']}: <span style='color:{ACCENT_RED};'>"
                    f"Insufficient improvement</span>"
                )
    elif fc_hr.get("status") == "no_improvement":
        lines.append(
            f"&bull; Current 7-day mean: {fc_hr.get('current_mean_7d', 'N/A')} bpm"
        )
        lines.append(
            f"&bull; <span style='color:{ACCENT_AMBER};'>No decreasing trend detected</span>"
        )
    else:
        lines.append("&bull; Insufficient data for forecast")

    content = "<br>\n".join(lines)
    return (
        f'<div style="padding:16px 20px;background:{BG_ELEVATED};border-radius:10px;'
        f'border:1px solid {BORDER_SUBTLE};line-height:1.7;font-size:14px;'
        f'color:{TEXT_PRIMARY};">{content}</div>'
    )


def build_methodology_section(fc_hrv: dict, fc_hr: dict) -> str:
    """Methodology and caveats."""
    n_hrv = fc_hrv.get("n_points", 0)
    n_hr = fc_hr.get("n_points", 0)
    return (
        f'<div style="font-size:13px;color:{TEXT_SECONDARY};line-height:1.7;">'
        f'<p><strong>Data:</strong> {n_hrv} post-ruxolitinib HRV observations, '
        f'{n_hr} post-ruxolitinib HR observations from Oura Ring Gen 4 '
        f'(consumer wearable, not clinical-grade).</p>'
        f'<p><strong>Linear model:</strong> OLS regression on day number since '
        f'treatment start ({TREATMENT_START}). Bootstrap: {BOOTSTRAP_N} resamples '
        f'for confidence intervals.</p>'
        f'<p><strong>Exponential model:</strong> y = a &middot; (1 - e<sup>-t/&tau;</sup>) + baseline, '
        f'fit via scipy.optimize.curve_fit with bounded parameters. '
        f'Model selection by AIC (lower = better).</p>'
        f'<p><strong>Three scenarios:</strong> Optimistic (75th pct slope), '
        f'Expected (50th pct), Conservative (25th pct).</p>'
        f'<p style="color:{ACCENT_AMBER};"><strong>Limitations (small N):</strong> '
        f'With only ~{max(n_hrv, n_hr)} data points, forecasts carry wide confidence '
        f'intervals and may shift substantially as more data accumulates. '
        f'Physiological recovery is non-linear and depends on factors not captured '
        f'in wearable data (infection status, medication changes, immune reconstitution). '
        f'These projections are exploratory, not clinical recommendations.</p>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------

def build_json(
    fc_hrv: dict, fc_hr: dict, fc_eff: dict,
    phases: dict, days_on_rux: int,
) -> dict:
    """Build structured JSON output."""
    # Remove bulky arrays from forecasts
    def _clean_fc(fc: dict) -> dict:
        out = {k: v for k, v in fc.items() if k not in ("day_nums", "values")}
        return out

    phase_summary = {}
    for pname, pdf in phases.items():
        phase_summary[pname] = {
            "n": len(pdf),
            "hrv_mean": round(float(pdf["average_hrv"].mean()), 2) if not pdf["average_hrv"].dropna().empty else None,
            "hr_mean": round(float(pdf["lowest_heart_rate"].mean()), 2) if not pdf["lowest_heart_rate"].dropna().empty else None,
            "efficiency_mean": round(float(pdf["efficiency"].mean()), 2) if not pdf["efficiency"].dropna().empty else None,
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "days_on_ruxolitinib": days_on_rux,
        "forecasts": {
            "hrv_average": _clean_fc(fc_hrv),
            "lowest_heart_rate": _clean_fc(fc_hr),
            "efficiency": _clean_fc(fc_eff),
        },
        "phase_summary": phase_summary,
    }


# ---------------------------------------------------------------------------
# Extra CSS
# ---------------------------------------------------------------------------

EXTRA_CSS = """
.forecast-summary-card {
    padding: 16px 20px;
    border-radius: 10px;
    line-height: 1.7;
    font-size: 14px;
}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Load data, compute forecasts, generate HTML + JSON."""
    print("Ruxolitinib Dose-Response Forecast")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    days_on_rux = max(0, (TODAY - TREATMENT_START).days)
    print(f"  Days on ruxolitinib: {days_on_rux}")
    print(f"  Treatment start: {TREATMENT_START}")
    print(f"  Acute event: {ACUTE_DATE}")

    # 1. Load data
    print("\n[1/6] Loading sleep data...")
    conn = safe_connect(DATABASE_PATH, read_only=True)
    try:
        df = load_sleep_data(conn)
        print(f"  -> {len(df)} long_sleep periods")

        if df.empty:
            print("ERROR: No sleep data found.", file=sys.stderr)
            return 1

        # 2. Phase split
        print("\n[2/6] Splitting into three phases...")
        phases = split_phases(df)
        for pname, pdf in phases.items():
            print(f"  -> {pname}: {len(pdf)} nights")

        df_post = phases["post_rux"]
        if len(df_post) < 3:
            print("ERROR: Fewer than 3 post-Rux data points.", file=sys.stderr)
            return 1

        # 3. Compute forecasts
        print("\n[3/6] Computing HRV forecast...")
        fc_hrv = forecast_metric(
            df_post, "average_hrv", HRV_TARGETS, "above", "HRV (RMSSD)",
        )
        print(f"  Status: {fc_hrv.get('status')}")
        if fc_hrv.get("slope_per_week"):
            print(f"  Slope: {fc_hrv['slope_per_week']:+.3f} ms/week")
            print(f"  Best model: {fc_hrv.get('best_model')}")

        print("\n[4/6] Computing HR forecast...")
        fc_hr = forecast_metric(
            df_post, "lowest_heart_rate", HR_TARGETS, "below", "Lowest Heart Rate",
        )
        print(f"  Status: {fc_hr.get('status')}")
        if fc_hr.get("slope_per_week"):
            print(f"  Slope: {fc_hr['slope_per_week']:+.3f} bpm/week")

        print("\n  Computing efficiency forecast...")
        fc_eff = forecast_metric(
            df_post, "efficiency", EFFICIENCY_TARGETS, "above", "Sleep Efficiency",
        )
        print(f"  Efficiency status: {fc_eff.get('status')}")

        # 5. Build HTML
        print("\n[5/6] Building HTML report...")

        # KPI section
        kpi_html = section_html_or_placeholder(
            "KPI Row",
            build_kpi_section, fc_hrv, fc_hr, days_on_rux,
        )

        # Forecast summary
        summary_html = section_html_or_placeholder(
            "Forecast Summary",
            build_forecast_summary, fc_hrv, fc_hr, days_on_rux,
        )

        # Trajectory chart
        trajectory_html = section_html_or_placeholder(
            "Trajectory Chart",
            build_trajectory_chart, df, phases, fc_hrv, fc_hr,
        )

        # Milestone timeline
        milestone_html = section_html_or_placeholder(
            "Milestone Timeline",
            build_milestone_chart, fc_hrv, fc_hr,
        )

        # Model comparison
        model_html = section_html_or_placeholder(
            "Model Comparison",
            build_model_comparison_chart, df_post, fc_hrv,
        )

        # Phase chart
        phase_html = section_html_or_placeholder(
            "Phase Chart",
            build_phase_chart, phases,
        )

        # Methodology
        method_html = section_html_or_placeholder(
            "Methodology",
            build_methodology_section, fc_hrv, fc_hr,
        )

        body = kpi_html
        body += make_section(
            "Forecast Summary",
            summary_html,
            section_id="forecast-summary",
        )
        body += make_section(
            "HRV & HR Trajectories with Forecast",
            trajectory_html,
            section_id="trajectories",
        )
        body += make_section(
            "Milestone Timeline",
            milestone_html,
            section_id="milestones",
        )
        body += make_section(
            "Model Comparison: Linear vs Exponential (HRV)",
            model_html,
            section_id="model-comparison",
        )
        body += make_section(
            "Three-Phase Recovery Context",
            phase_html,
            section_id="phase-context",
        )
        body += make_section(
            "Methodology & Caveats",
            method_html,
            section_id="methodology",
        )

        html_content = wrap_html(
            title="Ruxolitinib Forecast",
            body_content=body,
            report_id="forecast",
            header_meta="Henrik \u2014 Ruxolitinib Forecast",
            extra_css=EXTRA_CSS,
        )
        HTML_OUTPUT.write_text(html_content, encoding="utf-8")
        print(f"  -> HTML: {HTML_OUTPUT}")

    finally:
        conn.close()

    # 6. Export JSON
    print("\n[6/6] Exporting JSON...")
    json_data = build_json(fc_hrv, fc_hr, fc_eff, phases, days_on_rux)
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> JSON: {JSON_OUTPUT}")

    # Summary
    print("\n" + "=" * 60)
    print("FORECAST COMPLETE")
    if fc_hrv.get("status") == "forecast_available":
        first_t = fc_hrv.get("targets", {}).get("esc_15ms", {})
        if first_t.get("expected_date"):
            print(f"  HRV -> 15ms ESC: ~{first_t['expected_date']}")
    if fc_hr.get("status") == "forecast_available":
        first_h = fc_hr.get("targets", {}).get("normal_70", {})
        if first_h.get("expected_date"):
            print(f"  Lowest HR -> 70bpm: ~{first_h['expected_date']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
