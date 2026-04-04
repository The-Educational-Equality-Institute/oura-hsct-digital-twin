#!/usr/bin/env python3
"""
Causal Inference Engine for Oura Ring Biometric Data

Four complementary causal analysis methods explore whether Oura biometrics
shifted after ruxolitinib in a post-HSCT patient. See config.py for details.

Methods:
  1. Google CausalImpact (Bayesian Structural Time Series)
  2. Granger Causality Network (tigramite PCMCI+)
  3. Transfer Entropy (binning-based estimation)
  4. Intervention Response Decomposition (mediation analysis)

Output:
  - Interactive HTML report: reports/causal_inference_report.html
  - JSON metrics: reports/causal_inference_metrics.json

Usage:
    python analysis/analyze_oura_causal.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
import traceback
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

# pandas>=3 removed DataFrame.applymap; pycausalimpact still calls it.
if not hasattr(pd.DataFrame, "applymap") and hasattr(pd.DataFrame, "map"):
    pd.DataFrame.applymap = pd.DataFrame.map  # type: ignore[attr-defined]

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from statsmodels.tools.sm_exceptions import ValueWarning
    warnings.filterwarnings("ignore", category=ValueWarning)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Import guards for optional dependencies
# ---------------------------------------------------------------------------
try:
    from causalimpact import CausalImpact
    CAUSALIMPACT_AVAILABLE = True
except ImportError:
    CAUSALIMPACT_AVAILABLE = False
    CausalImpact = None  # type: ignore[assignment,misc]

try:
    from statsmodels.stats.multitest import multipletests
    MULTIPLETESTS_AVAILABLE = True
except ImportError:
    MULTIPLETESTS_AVAILABLE = False
    multipletests = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardening constants
# ---------------------------------------------------------------------------
MIN_PRE_DAYS = 14      # Minimum pre-intervention data points
MIN_POST_DAYS = 3      # Minimum post-intervention data points
MAX_NAN_FRACTION = 0.30  # Maximum fraction of NaN values allowed
CI_WIDTH_WARN_RATIO = 2.0  # Flag low confidence when CI width > 200% of point estimate
PLACEBO_MIN_PRE = 7    # Minimum pre-days for placebo tests
PLACEBO_MIN_POST = 5   # Minimum post-days for placebo tests
SAFE_LOG_MIN = 1e-300  # Floor for log operations
SAFE_DIV_EPS = 1e-15   # Epsilon for safe division

# ---------------------------------------------------------------------------
# Path resolution & patient config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH,
    REPORTS_DIR,
    TRANSPLANT_DATE,
    TREATMENT_START,
    PATIENT_AGE,
    PATIENT_LABEL,
    DATA_START,
    ESC_RMSSD_DEFICIENCY,
    NOCTURNAL_HR_ELEVATED,
    POPULATION_RMSSD_MEDIAN,
)

from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section, format_p_value,
    COLORWAY, STATUS_COLORS, BG_PRIMARY, BG_SURFACE, BG_ELEVATED,
    BORDER_SUBTLE, BORDER_DEFAULT, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    C_PRE_TX, C_POST_TX, C_RUX_LINE, C_EFFECT, C_COUNTERFACTUAL,
)
pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "causal_inference_report.html"
JSON_OUTPUT = REPORTS_DIR / "causal_inference_metrics.json"
TS_JSON_OUTPUT = REPORTS_DIR / "causal_timeseries.json"

# Color palette — uses dark theme from _theme
COLOR_PRE = C_PRE_TX
COLOR_POST = C_POST_TX
COLOR_RUX_LINE = C_RUX_LINE
COLOR_COUNTERFACTUAL = C_COUNTERFACTUAL
COLOR_CI_BAND = "rgba(147, 197, 253, 0.12)"
COLOR_EFFECT = C_EFFECT

# Plotly layout defaults — template handles most styling; t=60 since
# make_section already provides an h2 header (no redundant Plotly title)
LAYOUT_DEFAULTS = dict(
    margin=dict(l=70, r=30, t=60, b=40),
)

# Clinical reference values — imported from config.py

# PCMCI parameters
PCMCI_TAU_MAX = 7
PCMCI_ALPHA = 0.05

# Transfer entropy parameters
TE_N_BINS = 5
TE_HISTORY = 3

# Bootstrap parameters
BOOTSTRAP_N = 2000
BOOTSTRAP_CI = 95


# ---------------------------------------------------------------------------
# Numerical safety helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division guarded against zero / near-zero denominators."""
    if abs(denominator) < SAFE_DIV_EPS:
        return default
    return numerator / denominator


def _safe_pct(numerator: float, denominator: float) -> float:
    """Percentage guarded against zero denominators."""
    return _safe_div(numerator, denominator, default=0.0) * 100


def _relative_effect_is_meaningful(
    avg_actual: float,
    avg_counterfactual: float,
    avg_effect: float,
) -> bool:
    """Suppress relative effects when the counterfactual is near zero or flips sign."""
    values = (avg_actual, avg_counterfactual, avg_effect)
    if not all(np.isfinite(v) for v in values):
        return False
    if abs(avg_counterfactual) < SAFE_DIV_EPS:
        return False
    if avg_actual * avg_counterfactual <= 0:
        return False
    if abs(avg_effect) >= abs(avg_counterfactual):
        return False
    return True


def _format_relative_effect_html(value: float | None) -> str:
    """Render relative effects conservatively in HTML tables."""
    if value is None or not np.isfinite(value):
        return "&mdash;"
    return f"{value:+.1f}%"


def _make_ci_failure_result(
    meta: dict[str, Any],
    reason: str,
    *,
    n_pre: int = 0,
    n_post: int = 0,
) -> dict[str, Any]:
    """Return a standardized failure result dict for a CausalImpact metric.

    The report will show a warning badge instead of crashing.
    """
    return {
        "label": meta["label"],
        "unit": meta["unit"],
        "higher_is_better": meta["higher_is_better"],
        "n_pre": n_pre,
        "n_post": n_post,
        "avg_actual_post": 0.0,
        "avg_counterfactual_post": 0.0,
        "avg_effect": 0.0,
        "relative_effect_pct": None,
        "p_value": 1.0,
        "probability_of_effect": 0.0,
        "ci_lower": None,
        "ci_upper": None,
        "summary": None,
        "favorable": False,
        "significant": False,
        "error": reason,
        "low_confidence": True,
        "ts_dates": [],
        "ts_actual": [],
        "ts_predicted": [],
        "ts_pred_lower": [],
        "ts_pred_upper": [],
    }


def _check_convergence(
    actual_post: np.ndarray,
    pred_post: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
    point_estimate: float,
) -> tuple[bool, str]:
    """Check MCMC convergence quality.

    Returns (low_confidence: bool, reason: str).
    Flags low confidence when the 95% CI width exceeds CI_WIDTH_WARN_RATIO
    times the absolute point estimate.
    """
    if len(pred_lower) == 0 or len(pred_upper) == 0:
        return True, "No posterior intervals available"

    ci_width = float(np.nanmean(pred_upper - pred_lower))
    abs_estimate = abs(point_estimate)

    if abs_estimate < SAFE_DIV_EPS:
        # Point estimate is ~zero; wide CI is expected
        return False, ""

    ratio = ci_width / abs_estimate
    if ratio > CI_WIDTH_WARN_RATIO:
        reason = (
            f"CI width ({ci_width:.2f}) is {ratio:.1f}x the point estimate "
            f"({point_estimate:.2f}) - posterior may not have converged"
        )
        return True, reason

    return False, ""


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_data() -> dict[str, pd.DataFrame]:
    """Load all Oura tables into DataFrames."""
    print("[DATA] Loading biometric data from database...")

    if not DATABASE_PATH.exists():
        print(f"ERROR: Database not found at {DATABASE_PATH}. Run: python api/import_oura.py --days 90", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)

    # HRV epochs (5-min intervals)
    hrv = pd.read_sql_query(
        "SELECT timestamp, rmssd FROM oura_hrv ORDER BY timestamp", conn
    )
    hrv["date"] = pd.to_datetime(hrv["timestamp"], utc=True).dt.date.astype(str)
    hrv["rmssd"] = pd.to_numeric(hrv["rmssd"], errors="coerce")

    # Heart rate (continuous)
    hr = pd.read_sql_query(
        "SELECT timestamp, bpm FROM oura_heart_rate ORDER BY timestamp", conn
    )
    hr["date"] = pd.to_datetime(hr["timestamp"], utc=True).dt.date.astype(str)
    hr["bpm"] = pd.to_numeric(hr["bpm"], errors="coerce")

    # Sleep periods (per-night aggregates)
    sleep_periods = pd.read_sql_query(
        """SELECT day as date, average_hrv, average_heart_rate, average_breath,
                  total_sleep_duration, rem_sleep_duration, deep_sleep_duration,
                  light_sleep_duration, awake_time, efficiency, lowest_heart_rate
           FROM oura_sleep_periods
           WHERE type = 'long_sleep'
           ORDER BY day""",
        conn,
    )
    for col in sleep_periods.columns:
        if col != "date":
            sleep_periods[col] = pd.to_numeric(sleep_periods[col], errors="coerce")

    # SpO2
    spo2 = pd.read_sql_query(
        "SELECT date, spo2_average FROM oura_spo2 WHERE spo2_average > 0 ORDER BY date",
        conn,
    )
    spo2["spo2_average"] = pd.to_numeric(spo2["spo2_average"], errors="coerce")

    # Readiness (NOTE: most fields are CONTRIBUTOR SCORES 0-100, NOT physiological
    # values, EXCEPT temperature_deviation which IS a real temperature offset)
    readiness = pd.read_sql_query(
        """SELECT date, score as readiness_score, temperature_deviation,
                  recovery_index, resting_heart_rate as rhr_score
           FROM oura_readiness ORDER BY date""",
        conn,
    )
    for col in readiness.columns:
        if col != "date":
            readiness[col] = pd.to_numeric(readiness[col], errors="coerce")

    # Activity
    activity = pd.read_sql_query(
        "SELECT date, score as activity_score, active_calories, steps, daily_movement FROM oura_activity ORDER BY date",
        conn,
    )
    for col in activity.columns:
        if col != "date":
            activity[col] = pd.to_numeric(activity[col], errors="coerce")

    conn.close()

    data = {
        "hrv": hrv,
        "hr": hr,
        "sleep_periods": sleep_periods,
        "spo2": spo2,
        "readiness": readiness,
        "activity": activity,
    }

    for name, df in data.items():
        print(f"  {name}: {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")

    return data


def build_daily_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build unified daily feature matrix from all biometric sources.

    Returns DataFrame with string 'date' column and numeric feature columns.
    """
    print("[DATA] Building daily feature matrix...")

    # Aggregate HRV by day
    hrv_daily = (
        data["hrv"]
        .groupby("date")
        .agg(mean_rmssd=("rmssd", "mean"),
             max_rmssd=("rmssd", "max"),
             std_rmssd=("rmssd", "std"),
             median_rmssd=("rmssd", "median"))
        .reset_index()
    )

    # Aggregate HR by day
    hr_daily = (
        data["hr"]
        .groupby("date")
        .agg(mean_hr=("bpm", "mean"),
             std_hr=("bpm", "std"),
             min_hr=("bpm", "min"))
        .reset_index()
    )

    # Sleep periods - already per-night
    sp = data["sleep_periods"].copy()
    total_sec = sp["total_sleep_duration"].replace(0, np.nan)
    sp["sleep_efficiency"] = sp["efficiency"]
    sp["deep_pct"] = sp["deep_sleep_duration"] / total_sec * 100
    sp["rem_pct"] = sp["rem_sleep_duration"] / total_sec * 100
    sp["total_hours"] = sp["total_sleep_duration"] / 3600
    sp_features = sp[["date", "sleep_efficiency", "deep_pct", "rem_pct",
                       "total_hours", "average_heart_rate", "lowest_heart_rate",
                       "average_breath", "rem_sleep_duration",
                       "deep_sleep_duration"]].copy()

    # SpO2
    spo2 = data["spo2"][["date", "spo2_average"]].copy()

    # Readiness
    readiness = data["readiness"][["date", "readiness_score", "temperature_deviation",
                                    "recovery_index"]].copy()

    # Activity
    activity = data["activity"][["date", "steps", "active_calories"]].copy()

    # Merge all on date
    all_dates = sorted(set(
        hrv_daily["date"].tolist()
        + hr_daily["date"].tolist()
        + sp_features["date"].tolist()
    ))
    daily = pd.DataFrame({"date": all_dates})

    daily = daily.merge(hrv_daily, on="date", how="left")
    daily = daily.merge(hr_daily, on="date", how="left")
    daily = daily.merge(sp_features, on="date", how="left")
    daily = daily.merge(spo2, on="date", how="left")
    daily = daily.merge(readiness, on="date", how="left")
    daily = daily.merge(activity, on="date", how="left")

    daily = daily.sort_values("date").reset_index(drop=True)
    data_start_str = str(DATA_START)
    daily = daily[daily["date"] >= data_start_str].reset_index(drop=True)

    # Add period labels
    rux_str = str(TREATMENT_START)
    daily["period"] = daily["date"].apply(
        lambda d: "post" if d >= rux_str else "pre"
    )

    n_pre = (daily["period"] == "pre").sum()
    n_post = (daily["period"] == "post").sum()
    print(f"  Daily matrix: {len(daily)} days x {len(daily.columns)} features")
    print(f"  Date range: {daily['date'].iloc[0]} to {daily['date'].iloc[-1]}")
    print(f"  Pre-ruxolitinib: {n_pre} days | Post-ruxolitinib: {n_post} days")

    return daily


def _add_rux_line(fig: go.Figure, row: int = 1, col: int = 1) -> None:
    """Add a vertical line marking ruxolitinib start with glow effect."""
    rux_dt = pd.Timestamp(TREATMENT_START)
    yref = "y domain" if row == 1 else f"y{row} domain"

    # Glow layers: outer to inner, decreasing width + increasing opacity
    glow_layers = [
        ("rgba(59, 130, 246, 0.06)", 12),
        ("rgba(59, 130, 246, 0.10)", 8),
        ("rgba(59, 130, 246, 0.18)", 5),
        ("rgba(59, 130, 246, 0.35)", 3),
    ]
    for glow_color, glow_width in glow_layers:
        fig.add_shape(
            type="line",
            x0=rux_dt, x1=rux_dt,
            y0=0, y1=1, yref=yref,
            line=dict(color=glow_color, width=glow_width),
            row=row, col=col,
        )

    # Core line — solid, bright
    fig.add_shape(
        type="line",
        x0=rux_dt, x1=rux_dt,
        y0=0, y1=1, yref=yref,
        line=dict(color=ACCENT_BLUE, width=2),
        row=row, col=col,
    )
    fig.add_annotation(
        x=rux_dt, y=1, yref=yref,
        text="<b>Ruxolitinib start</b>",
        showarrow=True, arrowhead=2,
        ax=60, ay=-24,
        font=dict(color="#FFFFFF", size=11),
        arrowcolor=ACCENT_BLUE,
        bgcolor="rgba(59, 130, 246, 0.15)",
        bordercolor=ACCENT_BLUE,
        borderwidth=1,
        borderpad=4,
        row=row, col=col,
    )


# ===========================================================================
# SECTION 1: CAUSALIMPACT (Bayesian Structural Time Series)
# ===========================================================================

def run_causal_impact(daily: pd.DataFrame) -> dict[str, Any]:
    """Run Google CausalImpact on each biometric stream.

    Uses Bayesian structural time series to estimate counterfactual
    (what would have happened without ruxolitinib) and compute causal effect.
    """
    print("\n" + "=" * 70)
    print("[1/4] CAUSALIMPACT (Bayesian Structural Time Series)")
    print("=" * 70)
    t0 = time.perf_counter()

    results: dict[str, Any] = {
        "method": "CausalImpact (BSTS)",
        "streams": {},
        "runtime_s": 0,
    }

    if not CAUSALIMPACT_AVAILABLE:
        msg = ("CausalImpact package not installed. "
               "Install with: pip install pycausalimpact")
        print(f"  WARNING: {msg}")
        results["error"] = msg
        results["runtime_s"] = round(time.perf_counter() - t0, 2)
        return results

    # Define streams to analyze — all 11 metrics
    streams = {
        "rem_sleep_duration": {"label": "REM sleep duration (s)", "unit": "s",
                               "higher_is_better": True},
        "rem_pct": {"label": "REM sleep fraction (%)", "unit": "%",
                    "higher_is_better": True},
        "total_hours": {"label": "Total sleep (hours)", "unit": "h",
                        "higher_is_better": True},
        "deep_sleep_duration": {"label": "Deep sleep duration (s)", "unit": "s",
                                "higher_is_better": True},
        "mean_rmssd": {"label": "HRV mean RMSSD (ms)", "unit": "ms",
                       "higher_is_better": True},
        "max_rmssd": {"label": "HRV max RMSSD (ms)", "unit": "ms",
                      "higher_is_better": True},
        "lowest_heart_rate": {"label": "Lowest heart rate (bpm)", "unit": "bpm",
                              "higher_is_better": False},
        "mean_hr": {"label": "Average heart rate (bpm)", "unit": "bpm",
                    "higher_is_better": False},
        "average_breath": {"label": "Respiratory rate (br/min)", "unit": "br/min",
                           "higher_is_better": False},
        "spo2_average": {"label": "SpO2 (%)", "unit": "%",
                         "higher_is_better": True},
        "temperature_deviation": {"label": "Temperature deviation (°C)", "unit": "°C",
                                  "higher_is_better": False},
    }

    rux_str = str(TREATMENT_START)

    for stream_name, meta in streams.items():
        print(f"\n  Processing: {meta['label']}...")

        # Get valid data for this stream
        df = daily[["date", stream_name]].copy()

        # --- Guard: NaN fraction ---
        total_rows = len(df)
        nan_count = df[stream_name].isna().sum()
        nan_frac = _safe_div(nan_count, total_rows, default=1.0)
        if nan_frac > MAX_NAN_FRACTION:
            reason = f"Too many NaN values: {nan_count}/{total_rows} ({nan_frac:.0%} > {MAX_NAN_FRACTION:.0%} threshold)"
            print(f"    Skipping - {reason}")
            results["streams"][stream_name] = _make_ci_failure_result(meta, reason)
            continue

        df = df.dropna(subset=[stream_name])
        if len(df) < 10:
            reason = f"Only {len(df)} data points (need >= 10)"
            print(f"    Skipping - {reason}")
            results["streams"][stream_name] = _make_ci_failure_result(meta, reason)
            continue

        # --- Guard: pre/post minimum counts ---
        n_pre = (df["date"] < rux_str).sum()
        n_post = (df["date"] >= rux_str).sum()
        if n_pre < MIN_PRE_DAYS:
            reason = f"Only {n_pre} pre-period points (need >= {MIN_PRE_DAYS})"
            print(f"    Skipping - {reason}")
            results["streams"][stream_name] = _make_ci_failure_result(
                meta, reason, n_pre=n_pre, n_post=n_post)
            continue
        if n_post < MIN_POST_DAYS:
            reason = f"Only {n_post} post-period points (need >= {MIN_POST_DAYS})"
            print(f"    Skipping - {reason}")
            results["streams"][stream_name] = _make_ci_failure_result(
                meta, reason, n_pre=n_pre, n_post=n_post)
            continue

        # Build time series with DatetimeIndex and reindex to continuous daily range
        # (CausalImpact requires post_start to exist in the index)
        ts = df.set_index(pd.to_datetime(df["date"]))[[stream_name]].copy()
        ts.index.name = None

        # Reindex to continuous daily range so CausalImpact can find exact dates
        full_range = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
        ts = ts.reindex(full_range)
        # Interpolate gaps (max 3 consecutive days) for BSTS model
        ts[stream_name] = ts[stream_name].interpolate(method="linear", limit=3)
        # Drop any remaining NaN at edges
        ts = ts.dropna()

        if len(ts) < 10:
            print(f"    Skipping after reindex - only {len(ts)} points")
            results["streams"][stream_name] = {"error": f"Only {len(ts)} points after reindex"}
            continue

        # Define pre/post periods
        pre_start = str(ts.index.min().date())
        pre_end = str((TREATMENT_START - timedelta(days=1)))
        post_start = str(TREATMENT_START)
        post_end = str(ts.index.max().date())

        # Verify post_start exists in index after reindex
        if pd.Timestamp(post_start) not in ts.index:
            # Find the nearest available post date
            post_dates = ts.index[ts.index >= pd.Timestamp(post_start)]
            if len(post_dates) == 0:
                print("    Skipping - no post data in reindexed series")
                results["streams"][stream_name] = {"error": "No post-period data after reindex"}
                continue
            post_start = str(post_dates[0].date())
            print(f"    Adjusted post_start to {post_start}")

        print(f"    Pre: {pre_start} to {pre_end} ({n_pre} days)")
        print(f"    Post: {post_start} to {post_end} ({n_post} days)")

        try:
            ci = CausalImpact(
                ts, [pre_start, pre_end], [post_start, post_end],
                niter=5000, nseasons=[{"period": 7}],
            )

            # Extract results
            inferences = ci.inferences
            try:
                summary_text = ci.summary()
            except Exception:
                summary_text = "(summary generation failed)"
            try:
                report_text = ci.summary("report")
            except Exception:
                report_text = "(report generation failed)"

            # Compute effect metrics - use post_start (may have been adjusted)
            post_ts = pd.Timestamp(post_start)
            post_mask = np.array(ts.index >= post_ts)  # ensure plain ndarray
            actual_post = ts.loc[post_mask, stream_name].values

            # Inferences may use integer or datetime index; align by position
            if inferences is not None and len(inferences) == len(ts):
                inf_post = inferences.iloc[post_mask]
                pred_post = inf_post["preds"].values
                pred_lower = inf_post["preds_lower"].values
                pred_upper = inf_post["preds_upper"].values
            elif inferences is not None:
                # Try datetime alignment
                try:
                    inf_post = inferences.loc[post_ts:]
                    pred_post = inf_post["preds"].values
                    pred_lower = inf_post["preds_lower"].values
                    pred_upper = inf_post["preds_upper"].values
                except (KeyError, TypeError):
                    pred_post = np.array([])
                    pred_lower = np.array([])
                    pred_upper = np.array([])
            else:
                pred_post = np.array([])
                pred_lower = np.array([])
                pred_upper = np.array([])

            if len(actual_post) > 0 and len(pred_post) > 0:
                avg_effect = float(np.nanmean(actual_post - pred_post))
                avg_actual = float(np.nanmean(actual_post))
                avg_predicted = float(np.nanmean(pred_post))
                rel_effect = _safe_pct(avg_effect, avg_predicted)
                rel_effect_meaningful = _relative_effect_is_meaningful(
                    avg_actual, avg_predicted, avg_effect
                )
            else:
                avg_effect = 0.0
                avg_actual = 0.0
                avg_predicted = 0.0
                rel_effect = 0.0
                rel_effect_meaningful = False

            p_value = getattr(ci, "p_value", None)
            if p_value is None or not np.isfinite(p_value):
                p_value = 1.0

            # --- Convergence diagnostics ---
            low_confidence, convergence_note = _check_convergence(
                actual_post, pred_post, pred_lower, pred_upper, avg_effect,
            )
            if low_confidence:
                print(f"    WARNING: Low confidence - {convergence_note}")

            ci_lower_val = (
                round(float(np.nanmean(actual_post - pred_upper)), 3)
                if len(pred_upper) > 0 else None
            )
            ci_upper_val = (
                round(float(np.nanmean(actual_post - pred_lower)), 3)
                if len(pred_lower) > 0 else None
            )

            stream_result = {
                "label": meta["label"],
                "unit": meta["unit"],
                "higher_is_better": meta["higher_is_better"],
                "n_pre": int(n_pre),
                "n_post": int(n_post),
                "avg_actual_post": round(avg_actual, 3),
                "avg_counterfactual_post": round(avg_predicted, 3),
                "avg_effect": round(avg_effect, 3),
                "relative_effect_pct": round(rel_effect, 2) if rel_effect_meaningful else None,
                "p_value": round(float(p_value), 6),
                "probability_of_effect": round((1 - float(p_value)) * 100, 2),
                "ci_lower": ci_lower_val,
                "ci_upper": ci_upper_val,
                "summary": summary_text,
                "favorable": (avg_effect > 0) == meta["higher_is_better"],
                "low_confidence": low_confidence,
                "convergence_note": convergence_note if low_confidence else "",
                # Store time series for plotting
                # Inferences may have integer or datetime index; use positional
                "ts_dates": [str(d.date()) for d in ts.index],
                "ts_actual": ts[stream_name].tolist(),
                "ts_predicted": (inferences["preds"].values.tolist()[:len(ts)]
                                 if inferences is not None else []),
                "ts_pred_lower": (inferences["preds_lower"].values.tolist()[:len(ts)]
                                  if inferences is not None else []),
                "ts_pred_upper": (inferences["preds_upper"].values.tolist()[:len(ts)]
                                  if inferences is not None else []),
            }

            direction = "favorable" if stream_result["favorable"] else "unfavorable"
            conf_tag = " [LOW CONFIDENCE]" if low_confidence else ""
            rel_effect_str = f"{rel_effect:+.1f}%" if rel_effect_meaningful else "n/a"
            print(f"    Effect: {avg_effect:+.2f} {meta['unit']} ({rel_effect_str}), p={p_value:.4f} [{direction}]{conf_tag}")

            results["streams"][stream_name] = stream_result

        except Exception as e:
            error_msg = f"CausalImpact failed: {type(e).__name__}: {e}"
            print(f"    ERROR: {error_msg}")
            traceback.print_exc()
            results["streams"][stream_name] = _make_ci_failure_result(
                meta, error_msg, n_pre=n_pre, n_post=n_post)

    # ------------------------------------------------------------------
    # Benjamini-Hochberg FDR correction across all metrics
    # ------------------------------------------------------------------
    valid_keys = [k for k, v in results["streams"].items()
                  if isinstance(v, dict) and "error" not in v and "p_value" in v]

    # Guard: filter out NaN/1.0-only p-values before FDR
    usable_keys = [k for k in valid_keys
                   if np.isfinite(results["streams"][k]["p_value"])
                   and results["streams"][k]["p_value"] < 1.0]

    if usable_keys and MULTIPLETESTS_AVAILABLE:
        raw_pvals = np.array([results["streams"][k]["p_value"] for k in usable_keys])
        try:
            reject, qvals, _, _ = multipletests(raw_pvals, method="fdr_bh", alpha=0.05)
            for i, k in enumerate(usable_keys):
                results["streams"][k]["q_value_bh"] = round(float(qvals[i]), 6)
                results["streams"][k]["significant_fdr"] = bool(reject[i])

            n_sig_fdr = int(np.sum(reject))
            print(f"\n  FDR correction (Benjamini-Hochberg): {n_sig_fdr}/{len(usable_keys)} metrics significant at q < 0.05")
        except Exception as fdr_err:
            print(f"\n  WARNING: FDR correction failed: {fdr_err}")
            for k in usable_keys:
                results["streams"][k]["q_value_bh"] = results["streams"][k]["p_value"]
                results["streams"][k]["significant_fdr"] = results["streams"][k]["p_value"] < 0.05
    elif not MULTIPLETESTS_AVAILABLE:
        print("\n  WARNING: statsmodels not available; skipping FDR correction")
        for k in valid_keys:
            results["streams"][k]["q_value_bh"] = results["streams"][k]["p_value"]
            results["streams"][k]["significant_fdr"] = results["streams"][k]["p_value"] < 0.05
    elif not usable_keys:
        print("\n  WARNING: All p-values are NaN or 1.0; skipping FDR correction")
        for k in valid_keys:
            results["streams"][k]["q_value_bh"] = 1.0
            results["streams"][k]["significant_fdr"] = False
        for k in valid_keys:
            s = results["streams"][k]
            tag = "SIG" if s["significant_fdr"] else "ns"
            print(f"    {s['label']}: p={s['p_value']:.4f} -> q={s['q_value_bh']:.4f} [{tag}]")

    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    print(f"\n  CausalImpact complete in {results['runtime_s']}s")
    return results


def run_placebo_tests(daily: pd.DataFrame, ci_results: dict[str, Any]) -> dict[str, Any]:
    """Run CausalImpact on 3 random placebo dates before March 16 to validate
    that the real intervention date is special.

    Selects the top 3 metrics (by q-value / p-value) and runs CausalImpact
    with placebo intervention dates. If the model is well-specified, placebo
    dates should NOT show significance.
    """
    print("\n" + "=" * 70)
    print("[PLACEBO] PLACEBO TESTS (CausalImpact falsification)")
    print("=" * 70)
    t0 = time.perf_counter()

    results: dict[str, Any] = {
        "method": "Placebo falsification tests",
        "placebo_dates": [],
        "metrics_tested": [],
        "tests": [],
        "runtime_s": 0,
    }

    if not CAUSALIMPACT_AVAILABLE:
        msg = "CausalImpact package not installed; skipping placebo tests"
        print(f"  WARNING: {msg}")
        results["error"] = msg
        results["runtime_s"] = round(time.perf_counter() - t0, 2)
        return results

    # Pick top 3 metrics by significance (lowest q_value or p_value)
    streams = ci_results.get("streams", {})
    valid = [(k, v) for k, v in streams.items()
             if isinstance(v, dict) and "error" not in v and "p_value" in v]
    valid.sort(key=lambda x: x[1].get("q_value_bh", x[1].get("p_value", 1)))
    top3 = valid[:3]

    if not top3:
        print("  No valid metrics to test.")
        results["error"] = "No valid CI metrics for placebo testing"
        return results

    metric_names = [k for k, _ in top3]
    results["metrics_tested"] = [
        {"key": k, "label": v["label"], "q_value": v.get("q_value_bh", v["p_value"])}
        for k, v in top3
    ]
    print(f"  Testing top 3 metrics: {[v['label'] for _, v in top3]}")

    # Choose 3 placebo dates in the pre-intervention period
    rux_str = str(TREATMENT_START)
    pre_dates = sorted([d for d in daily["date"].unique() if d < rux_str])

    if len(pre_dates) < 21:
        print(f"  Not enough pre-period data ({len(pre_dates)} days) for placebo tests (need 21+)")
        results["error"] = f"Insufficient pre-period ({len(pre_dates)} days)"
        return results

    # Pick 3 dates at roughly 25%, 50%, 75% of the pre-period (at least 7 days from
    # start and 7 days before end to ensure enough data on both sides)
    usable = pre_dates[7:-7]
    if len(usable) < 3:
        print(f"  Not enough usable dates for placebo tests ({len(usable)})")
        results["error"] = f"Insufficient usable pre-dates ({len(usable)})"
        return results

    n = len(usable)
    placebo_indices = [n // 4, n // 2, 3 * n // 4]
    placebo_dates = [usable[i] for i in placebo_indices]
    results["placebo_dates"] = placebo_dates
    print(f"  Placebo dates: {placebo_dates}")

    for placebo_date in placebo_dates:
        for metric_key in metric_names:
            meta = streams[metric_key]
            print(f"\n  Placebo {placebo_date} | {meta['label']}...")

            df = daily[["date", metric_key]].dropna(subset=[metric_key]).copy()
            ts = df.set_index(pd.to_datetime(df["date"]))[[metric_key]].copy()
            ts.index.name = None

            full_range = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
            ts = ts.reindex(full_range)
            ts[metric_key] = ts[metric_key].interpolate(method="linear", limit=3)
            ts = ts.dropna()

            # For placebo: pre = start to placebo-1, post = placebo to real_rux-1
            pre_start_str = str(ts.index.min().date())
            pre_end_str = str((pd.Timestamp(placebo_date) - timedelta(days=1)).date())
            post_start_str = placebo_date
            # End the placebo post-period one day before the real intervention
            post_end_str = str((TREATMENT_START - timedelta(days=1)))

            # Verify dates exist in index
            if pd.Timestamp(post_start_str) not in ts.index:
                post_candidates = ts.index[ts.index >= pd.Timestamp(post_start_str)]
                if len(post_candidates) == 0:
                    results["tests"].append({
                        "placebo_date": placebo_date,
                        "metric": metric_key,
                        "error": "No post data",
                    })
                    continue
                post_start_str = str(post_candidates[0].date())

            if pd.Timestamp(post_end_str) not in ts.index:
                pre_candidates = ts.index[ts.index <= pd.Timestamp(post_end_str)]
                if len(pre_candidates) == 0:
                    results["tests"].append({
                        "placebo_date": placebo_date,
                        "metric": metric_key,
                        "error": "No pre-end data",
                    })
                    continue
                post_end_str = str(pre_candidates[-1].date())

            n_pre_pl = (ts.index < pd.Timestamp(post_start_str)).sum()
            n_post_pl = ((ts.index >= pd.Timestamp(post_start_str)) &
                         (ts.index <= pd.Timestamp(post_end_str))).sum()

            if n_pre_pl < PLACEBO_MIN_PRE or n_post_pl < PLACEBO_MIN_POST:
                reason = (f"Insufficient data (pre={n_pre_pl} need>={PLACEBO_MIN_PRE}, "
                          f"post={n_post_pl} need>={PLACEBO_MIN_POST})")
                print(f"    Skipping - {reason}")
                results["tests"].append({
                    "placebo_date": placebo_date,
                    "metric": metric_key,
                    "label": meta["label"],
                    "error": reason,
                })
                continue

            try:
                # Trim series to end at post_end (exclude real post-intervention data)
                ts_trimmed = ts.loc[:pd.Timestamp(post_end_str)].copy()

                ci = CausalImpact(
                    ts_trimmed,
                    [pre_start_str, pre_end_str],
                    [post_start_str, post_end_str],
                    niter=5000, nseasons=[{"period": 7}],
                )

                p_value = getattr(ci, "p_value", None)
                if p_value is None or not np.isfinite(p_value):
                    p_value = 1.0

                # Compute effect
                post_mask = ts_trimmed.index >= pd.Timestamp(post_start_str)
                actual_post = ts_trimmed.loc[post_mask, metric_key].values
                inferences = ci.inferences
                if inferences is not None and len(inferences) == len(ts_trimmed):
                    pred_post = inferences.iloc[np.array(post_mask)]["preds"].values
                else:
                    pred_post = np.array([])

                avg_effect = float(np.nanmean(actual_post - pred_post)) if len(pred_post) > 0 else 0.0
                # Guard against NaN effect
                if not np.isfinite(avg_effect):
                    avg_effect = 0.0

                test_result = {
                    "placebo_date": placebo_date,
                    "metric": metric_key,
                    "label": meta["label"],
                    "n_pre": int(n_pre_pl),
                    "n_post": int(n_post_pl),
                    "p_value": round(float(p_value), 6),
                    "avg_effect": round(avg_effect, 3),
                    "significant": float(p_value) < 0.05,
                }
                sig_str = "SIG" if test_result["significant"] else "ns"
                print(f"    p={p_value:.4f} [{sig_str}], effect={avg_effect:+.3f}")
                results["tests"].append(test_result)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                print(f"    ERROR: {error_msg}")
                results["tests"].append({
                    "placebo_date": placebo_date,
                    "metric": metric_key,
                    "label": meta["label"],
                    "error": error_msg,
                })

    # Summary
    valid_tests = [t for t in results["tests"] if "error" not in t]
    n_sig_placebo = sum(1 for t in valid_tests if t.get("significant", False))
    results["n_total_tests"] = len(valid_tests)
    results["n_significant_placebo"] = n_sig_placebo
    results["placebo_validation"] = "PASS" if n_sig_placebo == 0 else "PARTIAL" if n_sig_placebo <= 1 else "FAIL"

    print(f"\n  Placebo summary: {n_sig_placebo}/{len(valid_tests)} tests significant")
    print(f"  Validation: {results['placebo_validation']}")

    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    return results


def plot_causal_impact(ci_results: dict[str, Any]) -> list[go.Figure]:
    """Generate Plotly figures for CausalImpact results.

    Each metric gets its own figure (not crammed into one massive subplot)
    for clarity and scrollability. Left panel: actual vs counterfactual.
    Right panel: point effects.
    """
    figures = []

    streams = ci_results.get("streams", {})
    # Only plot streams that have time series data (skip failures with empty ts_dates)
    valid_streams = {k: v for k, v in streams.items()
                     if "error" not in v and v.get("ts_dates")}

    if not valid_streams:
        return figures

    rux_dt = pd.Timestamp(TREATMENT_START)

    for stream_name, s in valid_streams.items():
        dates = pd.to_datetime(s["ts_dates"])
        actual = s["ts_actual"]
        predicted = s["ts_predicted"]
        pred_lower = s["ts_pred_lower"]
        pred_upper = s["ts_pred_upper"]
        label = s["label"]
        unit = s["unit"]

        fig = make_subplots(
            rows=1, cols=2,
            shared_xaxes=False,
            horizontal_spacing=0.08,
            subplot_titles=[
                f"{label} -- Actual vs. counterfactual",
                f"{label} -- Causal effect",
            ],
            column_widths=[0.6, 0.4],
        )

        # --- Mask for pre/post coloring ---
        pre_mask = dates < rux_dt
        post_mask = dates >= rux_dt

        # --- Left column: actual vs counterfactual ---

        # Confidence band (drawn first = behind everything)
        if pred_lower and pred_upper:
            fig.add_trace(
                go.Scatter(
                    x=list(dates) + list(dates)[::-1],
                    y=list(pred_upper) + list(pred_lower)[::-1],
                    fill="toself",
                    fillcolor="rgba(147, 197, 253, 0.08)",
                    line=dict(width=0),
                    name="95% CI",
                    legendgroup="ci",
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )

        # Counterfactual line (dashed, muted)
        if predicted:
            fig.add_trace(
                go.Scatter(
                    x=dates, y=predicted,
                    mode="lines",
                    name="Counterfactual",
                    line=dict(color=COLOR_COUNTERFACTUAL, width=1.5, dash="dash"),
                    legendgroup="counterfactual",
                    hovertemplate=(
                        f"<b>%{{x|%b %d}}</b><br>"
                        f"{label}<br>"
                        f"Counterfactual: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        # Actual: pre-period (muted)
        pre_dates = dates[pre_mask]
        pre_actual = [actual[i] for i in range(len(actual)) if pre_mask[i]]
        if len(pre_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pre_dates, y=pre_actual,
                    mode="lines",
                    name="Actual (pre)",
                    line=dict(color=C_PRE_TX, width=1.5),
                    legendgroup="actual_pre",
                    hovertemplate=(
                        f"<b>%{{x|%b %d}}</b><br>"
                        f"{label}<br>"
                        f"Actual: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        # Actual: post-period (vibrant + markers)
        post_dates = dates[post_mask]
        post_actual = [actual[i] for i in range(len(actual)) if post_mask[i]]
        if len(post_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=post_dates, y=post_actual,
                    mode="lines+markers",
                    name="Actual (post)",
                    line=dict(color="#FFFFFF", width=2.5),
                    marker=dict(size=5, color=ACCENT_BLUE,
                                line=dict(width=1.5, color="#FFFFFF")),
                    legendgroup="actual_post",
                    hovertemplate=(
                        f"<b>%{{x|%b %d}}</b><br>"
                        f"{label}<br>"
                        f"Actual: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        # --- Right column: point effects ---
        if predicted:
            effects = [a - p for a, p in zip(actual, predicted)]
            hib = s["higher_is_better"]
            bar_colors = [
                COLOR_EFFECT if (e > 0 and hib) or (e < 0 and not hib)
                else ACCENT_RED if (e < 0 and hib) or (e > 0 and not hib)
                else TEXT_TERTIARY
                for e in effects
            ]

            fig.add_trace(
                go.Bar(
                    x=dates, y=effects,
                    marker=dict(
                        color=bar_colors,
                        line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
                    ),
                    name="Point effect",
                    legendgroup="effect",
                    hovertemplate=(
                        f"<b>%{{x|%b %d}}</b><br>"
                        f"{label}<br>"
                        f"Effect: %{{y:+.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=1, col=2,
            )
            # Zero line
            fig.add_shape(
                type="line",
                x0=dates.min(), x1=dates.max(),
                y0=0, y1=0,
                line=dict(color=TEXT_TERTIARY, width=1, dash="dot"),
                row=1, col=2,
            )

            # Mean effect annotation on bars
            post_effects = [effects[i] for i in range(len(effects)) if post_mask[i]]
            if post_effects:
                mean_eff = float(np.nanmean(post_effects))
                fig.add_annotation(
                    x=dates[post_mask].mean(),
                    y=mean_eff,
                    text=f"Mean: {mean_eff:+.1f}",
                    showarrow=True, arrowhead=0,
                    ax=0, ay=-30,
                    font=dict(color=ACCENT_BLUE, size=11, family="Inter"),
                    bgcolor="rgba(59, 130, 246, 0.12)",
                    bordercolor=ACCENT_BLUE,
                    borderwidth=1,
                    borderpad=3,
                    row=1, col=2,
                )

        # Add ruxolitinib lines (glow effect)
        _add_rux_line(fig, row=1, col=1)
        _add_rux_line(fig, row=1, col=2)

        # Axis formatting
        fig.update_yaxes(title_text=unit, row=1, col=1)
        fig.update_yaxes(title_text=f"Delta ({unit})", row=1, col=2)
        fig.update_xaxes(
            tickformat="%b %d",
            spikemode="across", spikethickness=1,
            spikecolor=BORDER_DEFAULT, spikedash="dot",
            row=1, col=1,
        )
        fig.update_xaxes(
            tickformat="%b %d",
            spikemode="across", spikethickness=1,
            spikecolor=BORDER_DEFAULT, spikedash="dot",
            row=1, col=2,
        )

        # Significance badge in title
        p_val = s.get("p_value", 1.0)
        q_val = s.get("q_value_bh", p_val)
        favorable = s.get("favorable", False)

        fig.update_layout(
            title="",
            height=340,
            hovermode="x unified",
            **{**LAYOUT_DEFAULTS, "margin": {**LAYOUT_DEFAULTS["margin"], "t": 100}},
        )

        figures.append(fig)

    return figures


# ===========================================================================
# SECTION 2: GRANGER CAUSALITY NETWORK (tigramite PCMCI+)
# ===========================================================================

def run_pcmci(daily: pd.DataFrame) -> dict[str, Any]:
    """Run PCMCI+ Granger causality analysis on multivariate biometric streams.

    Tests for lagged causal relationships between biometric variables using
    partial correlation (ParCorr) independence test, controlling for confounders.
    """
    print("\n" + "=" * 70)
    print("[2/4] GRANGER CAUSALITY NETWORK (tigramite PCMCI+)")
    print("=" * 70)
    t0 = time.perf_counter()

    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        print("  tigramite not installed - skipping PCMCI+")
        print("  Install: pip install tigramite")
        return {"method": "PCMCI+ (tigramite)", "skipped": True, "reason": "tigramite not installed", "figures": []}

    results: dict[str, Any] = {
        "method": "PCMCI+ (tigramite)",
        "full_period": {},
        "pre_period": {},
        "runtime_s": 0,
    }

    # Variables for analysis — all 11 metrics
    var_cols = ["rem_sleep_duration", "rem_pct", "total_hours",
                "deep_sleep_duration", "mean_rmssd", "max_rmssd",
                "lowest_heart_rate", "mean_hr", "average_breath",
                "spo2_average", "temperature_deviation"]
    var_labels = ["REMdur", "REMpct", "TotalSleep", "DeepDur",
                  "RMSSD", "RMSSDmax", "LowestHR", "AvgHR",
                  "RespRate", "SpO2", "TempDev"]

    # Build clean matrix
    df = daily[["date", "period"] + var_cols].copy()
    df[var_cols] = df[var_cols].interpolate(method="linear", limit=3)
    df = df.dropna(subset=var_cols)

    if len(df) < 15:
        print(f"  ERROR: Only {len(df)} complete rows, need at least 15")
        results["error"] = f"Insufficient data ({len(df)} rows)"
        return results

    print(f"  Using {len(df)} days with {len(var_cols)} variables")
    print(f"  Variables: {var_labels}")

    def _run_pcmci_on_data(
        data_array: np.ndarray,
        label: str,
    ) -> dict[str, Any]:
        """Run PCMCI+ on a numpy array and return structured results."""
        n_obs = data_array.shape[0]
        tau_max = min(PCMCI_TAU_MAX, n_obs // 4)  # Adapt to data length

        print(f"\n  Running PCMCI+ on {label} ({n_obs} observations, tau_max={tau_max})...")

        dataframe = pp.DataFrame(
            data_array,
            var_names=var_labels,
        )
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=parcorr,
            verbosity=0,
        )

        pcmci_results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=PCMCI_ALPHA)

        val_matrix = pcmci_results["val_matrix"]
        p_matrix = pcmci_results["p_matrix"]

        # Extract significant links
        sig_links = []
        n_vars = len(var_labels)
        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(tau_max + 1):
                    if i == j and tau == 0:
                        continue  # skip contemporaneous self-links
                    if p_matrix[i, j, tau] < PCMCI_ALPHA:
                        sig_links.append({
                            "source": var_labels[i],
                            "target": var_labels[j],
                            "lag": int(tau),
                            "val": round(float(val_matrix[i, j, tau]), 4),
                            "p_value": round(float(p_matrix[i, j, tau]), 6),
                        })

        # Sort by absolute correlation strength
        sig_links.sort(key=lambda x: abs(x["val"]), reverse=True)

        print(f"    Found {len(sig_links)} significant causal links (alpha={PCMCI_ALPHA})")
        for link in sig_links[:5]:
            direction = "+" if link["val"] > 0 else "-"
            print(f"      {link['source']} -> {link['target']} (lag={link['lag']}, "
                  f"r={link['val']:{direction}.3f}, p={link['p_value']:.4f})")

        return {
            "n_observations": int(n_obs),
            "tau_max": int(tau_max),
            "n_significant_links": len(sig_links),
            "significant_links": sig_links,
            "val_matrix": val_matrix.tolist(),
            "p_matrix": p_matrix.tolist(),
            "var_labels": var_labels,
        }

    # Full period
    full_data = df[var_cols].values
    results["full_period"] = _run_pcmci_on_data(full_data, "full period")

    # Pre-ruxolitinib period only
    pre_df = df[df["period"] == "pre"]
    if len(pre_df) >= 15:
        pre_data = pre_df[var_cols].values
        results["pre_period"] = _run_pcmci_on_data(pre_data, "pre-ruxolitinib")
    else:
        print(f"  Skipping pre-period analysis: only {len(pre_df)} observations")
        results["pre_period"] = {"error": f"Insufficient pre-period data ({len(pre_df)} rows)"}

    # Compare networks: links unique to full vs pre
    full_links_set = set()
    for link in results["full_period"].get("significant_links", []):
        full_links_set.add((link["source"], link["target"], link["lag"]))

    pre_links_set = set()
    for link in results.get("pre_period", {}).get("significant_links", []):
        pre_links_set.add((link["source"], link["target"], link["lag"]))

    new_links = full_links_set - pre_links_set
    lost_links = pre_links_set - full_links_set

    results["network_comparison"] = {
        "new_links_in_full": [
            {"source": s, "target": t, "lag": l} for s, t, l in new_links
        ],
        "lost_links_from_pre": [
            {"source": s, "target": t, "lag": l} for s, t, l in lost_links
        ],
        "n_new": len(new_links),
        "n_lost": len(lost_links),
    }

    if new_links:
        print(f"\n  New causal links in full period (not in pre-period): {len(new_links)}")
        for s, t, l in new_links:
            print(f"    NEW: {s} -> {t} (lag={l})")

    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    print(f"\n  PCMCI+ complete in {results['runtime_s']}s")
    return results


def plot_pcmci(pcmci_results: dict[str, Any]) -> list[go.Figure]:
    """Generate Plotly causal graph visualization for PCMCI+ results.

    Nodes have subtle glow halos based on connection count.
    Edge thickness scales with correlation strength.
    A radial gradient background centers the network.
    """
    figures = []

    for period_key, period_label in [("full_period", "Full period"),
                                      ("pre_period", "Pre-ruxolitinib")]:
        period_data = pcmci_results.get(period_key, {})
        if "error" in period_data or not period_data:
            continue

        sig_links = period_data.get("significant_links", [])
        var_labels = period_data.get("var_labels", [])

        if not var_labels:
            continue

        n_vars = len(var_labels)

        # Arrange nodes in a circle
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
        # Start from top (pi/2) and go clockwise
        angles = np.pi / 2 - angles
        radius = 1.0
        node_x = radius * np.cos(angles)
        node_y = radius * np.sin(angles)

        fig = go.Figure()

        # Count in/out edges per node for sizing and glow
        in_count = {v: 0 for v in var_labels}
        out_count = {v: 0 for v in var_labels}
        for link in sig_links:
            in_count[link["target"]] += 1
            out_count[link["source"]] += 1

        total_count = {v: in_count[v] + out_count[v] for v in var_labels}
        max_count = max(total_count.values()) if total_count else 1

        # --- Radial gradient background (concentric rings) ---
        ring_radii = [0.3, 0.6, 0.9, 1.2]
        ring_opacities = [0.06, 0.04, 0.03, 0.02]
        for r_radius, r_opacity in zip(ring_radii, ring_opacities):
            theta_ring = np.linspace(0, 2 * np.pi, 60)
            fig.add_trace(go.Scatter(
                x=(r_radius * np.cos(theta_ring)).tolist(),
                y=(r_radius * np.sin(theta_ring)).tolist(),
                mode="lines",
                line=dict(color=f"rgba(59, 130, 246, {r_opacity})", width=1),
                hoverinfo="skip",
                showlegend=False,
            ))

        # --- Node glow halos (drawn before edges, before nodes) ---
        node_colors = [COLORWAY[i % len(COLORWAY)] for i in range(n_vars)]
        for idx, v in enumerate(var_labels):
            importance = total_count[v] / max(max_count, 1)
            if importance > 0:
                # Two glow layers per node
                for glow_mult, glow_alpha in [(3.0, 0.08), (2.0, 0.15)]:
                    glow_size = (22 + 10 * total_count[v]) * glow_mult
                    c = node_colors[idx]
                    # Extract hex -> rgba
                    fig.add_trace(go.Scatter(
                        x=[float(node_x[idx])], y=[float(node_y[idx])],
                        mode="markers",
                        marker=dict(
                            size=glow_size,
                            color=c,
                            opacity=glow_alpha * importance,
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    ))

        # --- Draw edges (causal links) ---
        for link in sig_links:
            src_idx = var_labels.index(link["source"])
            tgt_idx = var_labels.index(link["target"])

            # Color by sign: positive = blue/cyan, negative = red/pink
            if link["val"] > 0:
                edge_color = ACCENT_BLUE
                edge_glow = "rgba(59, 130, 246, 0.15)"
            else:
                edge_color = ACCENT_RED
                edge_glow = "rgba(239, 68, 68, 0.15)"

            edge_width = max(1.5, min(7, abs(link["val"]) * 12))

            # Offset for multiple lags
            offset = link["lag"] * 0.03

            x0, y0 = float(node_x[src_idx]) + offset, float(node_y[src_idx]) + offset
            x1, y1 = float(node_x[tgt_idx]), float(node_y[tgt_idx])

            # Edge glow
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color=edge_glow, width=edge_width + 4),
                hoverinfo="skip",
                showlegend=False,
            ))

            # Edge core
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                hoverinfo="text",
                hovertext=(
                    f"<b>{link['source']} -> {link['target']}</b><br>"
                    f"Lag: {link['lag']} day{'s' if link['lag'] != 1 else ''}<br>"
                    f"r = {link['val']:+.3f}<br>"
                    f"p = {link['p_value']:.4f}"
                ),
                showlegend=False,
            ))

            # Arrowhead (triangle marker near target)
            mid_x = 0.82 * x1 + 0.18 * x0
            mid_y = 0.82 * y1 + 0.18 * y0
            arrow_angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)) - 90)
            fig.add_trace(go.Scatter(
                x=[mid_x], y=[mid_y],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=max(8, edge_width + 3),
                    color=edge_color,
                    angle=arrow_angle,
                ),
                hoverinfo="skip",
                showlegend=False,
            ))

        # --- Draw nodes ---
        node_sizes = [22 + 10 * total_count[v] for v in var_labels]

        fig.add_trace(go.Scatter(
            x=node_x.tolist(),
            y=node_y.tolist(),
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2.5, color=BG_PRIMARY),
            ),
            text=var_labels,
            textposition="top center",
            textfont=dict(size=11, color=TEXT_PRIMARY, family="Inter"),
            hoverinfo="text",
            hovertext=[
                f"<b>{v}</b><br>"
                f"Incoming links: {in_count[v]}<br>"
                f"Outgoing links: {out_count[v]}<br>"
                f"Total connections: {total_count[v]}"
                for v in var_labels
            ],
            showlegend=False,
        ))

        n_links = len(sig_links)

        # Legend traces for edge color semantics
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=ACCENT_BLUE, width=3),
            name="Positive correlation",
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=ACCENT_RED, width=3),
            name="Negative correlation",
        ))

        fig.update_layout(
            title="",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showline=False, range=[-1.7, 1.7]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showline=False, range=[-1.7, 1.7], scaleanchor="x"),
            height=650,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.08,
                        xanchor="center", x=0.5),
            annotations=[dict(
                text=f"<b>{period_label}</b> -- {n_links} significant links (alpha={PCMCI_ALPHA})",
                x=0.5, y=1.06, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=13, color=TEXT_SECONDARY),
            )],
            **LAYOUT_DEFAULTS,
        )

        figures.append(fig)

    return figures


# ===========================================================================
# SECTION 3: TRANSFER ENTROPY
# ===========================================================================

def _discretize(series: np.ndarray, n_bins: int = TE_N_BINS) -> np.ndarray:
    """Discretize a continuous series into n_bins equal-frequency bins."""
    # Use quantile-based binning
    try:
        bins = np.quantile(series[~np.isnan(series)],
                          np.linspace(0, 1, n_bins + 1))
        # Make bins unique
        bins = np.unique(bins)
        if len(bins) < 2:
            return np.zeros(len(series), dtype=int)
        digitized = np.digitize(series, bins[1:-1])
        return digitized
    except Exception:
        return np.zeros(len(series), dtype=int)


def _transfer_entropy(source: np.ndarray, target: np.ndarray,
                      history: int = TE_HISTORY) -> float:
    """Compute transfer entropy from source to target.

    TE(X -> Y) = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k})

    Uses a histogram-based estimator (robust for small N).
    """
    n = len(source)
    if n <= history + 1:
        return 0.0

    # Discretize
    src_d = _discretize(source)
    tgt_d = _discretize(target)

    # Build joint histories
    # Y past: (y_{t-1}, ..., y_{t-k})
    # X past: (x_{t-1}, ..., x_{t-k})
    # Y present: y_t

    y_present = tgt_d[history:]
    y_past = np.column_stack([tgt_d[history - h - 1: n - h - 1] for h in range(history)])
    x_past = np.column_stack([src_d[history - h - 1: n - h - 1] for h in range(history)])

    # Convert to tuple keys
    def _to_keys(arr: np.ndarray) -> list[tuple]:
        if arr.ndim == 1:
            return [(int(v),) for v in arr]
        return [tuple(int(x) for x in row) for row in arr]

    y_pres_keys = _to_keys(y_present)
    y_past_keys = _to_keys(y_past)
    x_past_keys = _to_keys(x_past)

    # Joint keys
    yx_past_keys = [yp + xp for yp, xp in zip(y_past_keys, x_past_keys)]

    from collections import Counter

    def _entropy(keys_list: list) -> float:
        counts = Counter(keys_list)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum(
            (c / total) * np.log2(max(c / total, SAFE_LOG_MIN))
            for c in counts.values() if c > 0
        )

    # Joint entropies for conditional entropy calculation
    # H(Y_t | Y_past) = H(Y_t, Y_past) - H(Y_past)
    yt_ypast_keys = [(yp,) + ypa for yp, ypa in zip(y_pres_keys, y_past_keys)]
    h_yt_ypast = _entropy(yt_ypast_keys) - _entropy(y_past_keys)

    # H(Y_t | Y_past, X_past) = H(Y_t, Y_past, X_past) - H(Y_past, X_past)
    yt_yxpast_keys = [(yp,) + yxp for yp, yxp in zip(y_pres_keys, yx_past_keys)]
    h_yt_yxpast = _entropy(yt_yxpast_keys) - _entropy(yx_past_keys)

    te = h_yt_ypast - h_yt_yxpast
    return max(0.0, te)  # TE should be non-negative (bias can cause small negatives)


def run_transfer_entropy(daily: pd.DataFrame) -> dict[str, Any]:
    """Compute transfer entropy between all pairs of biometric streams.

    Compares TE matrices for pre vs full period to detect changes in
    information flow after ruxolitinib start.
    """
    print("\n" + "=" * 70)
    print("[3/4] TRANSFER ENTROPY (binning-based)")
    print("=" * 70)
    t0 = time.perf_counter()

    results: dict[str, Any] = {
        "method": "Transfer Entropy (histogram)",
        "full_period": {},
        "pre_period": {},
        "runtime_s": 0,
    }

    var_cols = ["rem_sleep_duration", "rem_pct", "total_hours",
                "deep_sleep_duration", "mean_rmssd", "max_rmssd",
                "lowest_heart_rate", "mean_hr", "average_breath",
                "spo2_average", "temperature_deviation"]
    var_labels = ["REMdur", "REMpct", "TotalSleep", "DeepDur",
                  "RMSSD", "RMSSDmax", "LowestHR", "AvgHR",
                  "RespRate", "SpO2", "TempDev"]

    df = daily[["date", "period"] + var_cols].copy()
    df[var_cols] = df[var_cols].interpolate(method="linear", limit=3)
    df = df.dropna(subset=var_cols)

    if len(df) < 10:
        print(f"  ERROR: Only {len(df)} complete rows")
        results["error"] = f"Insufficient data ({len(df)} rows)"
        return results

    def _compute_te_matrix(data: np.ndarray, label: str) -> dict[str, Any]:
        """Compute pairwise transfer entropy matrix."""
        n_vars = data.shape[1]
        te_matrix = np.zeros((n_vars, n_vars))

        print(f"\n  Computing TE matrix for {label} ({data.shape[0]} observations)...")

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                te_matrix[i, j] = _transfer_entropy(data[:, i], data[:, j])

        # Compute net TE (asymmetry)
        net_te = te_matrix - te_matrix.T

        # Top directed links
        top_links = []
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and te_matrix[i, j] > 0.01:
                    top_links.append({
                        "source": var_labels[i],
                        "target": var_labels[j],
                        "te": round(float(te_matrix[i, j]), 4),
                        "net_te": round(float(net_te[i, j]), 4),
                    })

        top_links.sort(key=lambda x: x["te"], reverse=True)

        print("    Top TE links:")
        for link in top_links[:5]:
            net = f"(net: {link['net_te']:+.4f})" if link["net_te"] != 0 else ""
            print(f"      {link['source']} -> {link['target']}: {link['te']:.4f} {net}")

        return {
            "te_matrix": te_matrix.tolist(),
            "net_te_matrix": net_te.tolist(),
            "top_links": top_links,
            "n_observations": int(data.shape[0]),
            "var_labels": var_labels,
        }

    # Full period
    full_data = df[var_cols].values
    results["full_period"] = _compute_te_matrix(full_data, "full period")

    # Pre-ruxolitinib
    pre_df = df[df["period"] == "pre"]
    if len(pre_df) >= 10:
        pre_data = pre_df[var_cols].values
        results["pre_period"] = _compute_te_matrix(pre_data, "pre-ruxolitinib")

        # Compute TE difference matrix
        full_te = np.array(results["full_period"]["te_matrix"])
        pre_te = np.array(results["pre_period"]["te_matrix"])
        diff_te = full_te - pre_te

        results["te_difference"] = {
            "diff_matrix": diff_te.tolist(),
            "var_labels": var_labels,
            "max_increase": {
                "value": round(float(np.max(diff_te)), 4),
                "pair": None,
            },
            "max_decrease": {
                "value": round(float(np.min(diff_te)), 4),
                "pair": None,
            },
        }

        # Find max increase/decrease pair
        max_idx = np.unravel_index(np.argmax(diff_te), diff_te.shape)
        min_idx = np.unravel_index(np.argmin(diff_te), diff_te.shape)
        results["te_difference"]["max_increase"]["pair"] = (
            f"{var_labels[max_idx[0]]} -> {var_labels[max_idx[1]]}"
        )
        results["te_difference"]["max_decrease"]["pair"] = (
            f"{var_labels[min_idx[0]]} -> {var_labels[min_idx[1]]}"
        )
    else:
        results["pre_period"] = {"error": f"Insufficient pre-period data ({len(pre_df)} rows)"}

    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    print(f"\n  Transfer Entropy complete in {results['runtime_s']}s")
    return results


def plot_transfer_entropy(te_results: dict[str, Any]) -> list[go.Figure]:
    """Generate heatmap visualizations for transfer entropy matrices.

    Uses refined dark-theme colorscales with adaptive annotation text color
    for readability on both light and dark heatmap cells.
    """
    figures = []

    matrices_to_plot = []

    for period_key, title in [("full_period", "Full period"),
                               ("pre_period", "Pre-ruxolitinib")]:
        period = te_results.get(period_key, {})
        if "error" not in period and "te_matrix" in period:
            matrices_to_plot.append((period["te_matrix"], period["var_labels"], title))

    # Add difference matrix if available
    diff = te_results.get("te_difference")
    if diff and "diff_matrix" in diff:
        matrices_to_plot.append((diff["diff_matrix"], diff["var_labels"],
                                "Change (Full - Pre)"))

    if not matrices_to_plot:
        return figures

    # Dark-theme-friendly colorscale for absolute TE (deep navy -> cyan -> bright white)
    te_colorscale = [
        [0.0, BG_SURFACE],
        [0.15, "#1A2744"],
        [0.30, "#1E3A5F"],
        [0.50, "#2563EB"],
        [0.70, "#3B82F6"],
        [0.85, "#60A5FA"],
        [1.0, "#DBEAFE"],
    ]

    # Diverging colorscale for difference (red -> dark bg -> blue)
    te_diverging = [
        [0.0, "#DC2626"],
        [0.2, "#F87171"],
        [0.4, "#FCA5A5"],
        [0.5, BG_SURFACE],
        [0.6, "#93C5FD"],
        [0.8, "#3B82F6"],
        [1.0, "#1D4ED8"],
    ]

    n_plots = len(matrices_to_plot)
    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=[t for _, _, t in matrices_to_plot],
        horizontal_spacing=0.1,
    )

    for col, (matrix, labels, title) in enumerate(matrices_to_plot, start=1):
        mat = np.array(matrix)

        # Use different colorscales for difference vs absolute
        if "Change" in title:
            colorscale = te_diverging
            abs_max = float(np.max(np.abs(mat))) if np.max(np.abs(mat)) > 0 else 0.1
            zmin = -abs_max
            zmax = abs_max
        else:
            colorscale = te_colorscale
            zmin = 0
            zmax = float(np.max(mat)) if np.max(mat) > 0 else 1

        # Adaptive text color: dark text on light cells, light text on dark cells
        z_mid = (zmin + zmax) / 2
        text_colors = [
            [
                (
                    ""
                    if i == j
                    else (
                        "#1A1D27" if mat[i][j] > z_mid + (zmax - z_mid) * 0.5
                        else TEXT_PRIMARY
                    )
                )
                for j in range(len(labels))
            ]
            for i in range(len(labels))
        ]

        # Format text annotations
        text = [[f"{mat[i][j]:.3f}" if i != j else ""
                 for j in range(len(labels))]
                for i in range(len(labels))]

        fig.add_trace(
            go.Heatmap(
                z=mat.tolist(),
                x=labels,
                y=labels,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=9),
                colorscale=colorscale,
                zmin=zmin, zmax=zmax,
                showscale=True,
                colorbar=dict(
                    title=dict(text="TE (bits)", font=dict(size=11, color=TEXT_SECONDARY)),
                    len=0.8,
                    thickness=12,
                    tickfont=dict(size=10, color=TEXT_SECONDARY),
                    outlinewidth=0,
                ),
                xgap=2, ygap=2,
                hovertemplate=(
                    "<b>%{y} -> %{x}</b><br>"
                    "Transfer Entropy: %{z:.4f} bits<extra></extra>"
                ),
            ),
            row=1, col=col,
        )

        fig.update_xaxes(
            title_text="Target",
            tickangle=45,
            tickfont=dict(size=9),
            row=1, col=col,
        )
        fig.update_yaxes(
            title_text="Source",
            tickfont=dict(size=9),
            row=1, col=col,
        )

    fig.update_layout(
        title="",
        height=550,
        **{**LAYOUT_DEFAULTS, "margin": {**LAYOUT_DEFAULTS["margin"], "t": 100}},
    )

    figures.append(fig)
    return figures


# ===========================================================================
# SECTION 4: INTERVENTION RESPONSE DECOMPOSITION (Mediation Analysis)
# ===========================================================================

def run_mediation_analysis(daily: pd.DataFrame) -> dict[str, Any]:
    """Decompose total ruxolitinib effect into causal pathways using
    linear mediation analysis with bootstrap confidence intervals.

    Pathways:
      1. Direct cardiac: ruxolitinib -> HR change
      2. Autonomic: ruxolitinib -> HRV change
      3. Sleep-mediated: ruxolitinib -> sleep quality -> recovery
      4. Inflammatory: ruxolitinib -> temperature -> downstream effects
    """
    print("\n" + "=" * 70)
    print("[4/4] INTERVENTION RESPONSE DECOMPOSITION (Mediation Analysis)")
    print("=" * 70)
    t0 = time.perf_counter()

    results: dict[str, Any] = {
        "method": "Linear Mediation Analysis with Bootstrap CIs",
        "pathways": {},
        "total_effect": {},
        "runtime_s": 0,
    }

    # Binary treatment variable
    rux_str = str(TREATMENT_START)
    df = daily.copy()
    df["treatment"] = (df["date"] >= rux_str).astype(float)

    # Outcome: readiness score (composite recovery measure)
    outcome_col = "readiness_score"
    mediator_configs = {
        "direct_cardiac": {
            "label": "Direct cardiac",
            "description": "Ruxolitinib -> HR change -> Readiness",
            "mediator": "mean_hr",
            "expected_direction": "decrease",
        },
        "autonomic": {
            "label": "Autonomic",
            "description": "Ruxolitinib -> HRV change -> Readiness",
            "mediator": "mean_rmssd",
            "expected_direction": "increase",
        },
        "sleep_mediated": {
            "label": "Sleep-mediated",
            "description": "Ruxolitinib -> Sleep efficiency -> Readiness",
            "mediator": "sleep_efficiency",
            "expected_direction": "increase",
        },
        "inflammatory": {
            "label": "Inflammatory",
            "description": "Ruxolitinib -> Temperature deviation -> Readiness",
            "mediator": "temperature_deviation",
            "expected_direction": "decrease",
        },
    }

    # Check we have outcome data
    df_clean = df.dropna(subset=[outcome_col])
    if len(df_clean) < 10:
        print(f"  ERROR: Only {len(df_clean)} rows with {outcome_col}")
        results["error"] = f"Insufficient {outcome_col} data"
        return results

    # Total effect: treatment -> outcome
    pre_outcome = df_clean.loc[df_clean["treatment"] == 0, outcome_col]
    post_outcome = df_clean.loc[df_clean["treatment"] == 1, outcome_col]

    if len(pre_outcome) > 0 and len(post_outcome) > 0:
        total_effect = float(post_outcome.mean() - pre_outcome.mean())
        # Welch's t-test
        t_stat, t_pval = scipy_stats.ttest_ind(post_outcome, pre_outcome, equal_var=False)
    else:
        total_effect = 0.0
        t_stat, t_pval = 0.0, 1.0

    results["total_effect"] = {
        "effect": round(total_effect, 3),
        "pre_mean": round(float(pre_outcome.mean()), 3) if len(pre_outcome) > 0 else None,
        "post_mean": round(float(post_outcome.mean()), 3) if len(post_outcome) > 0 else None,
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(t_pval), 6),
        "n_pre": int(len(pre_outcome)),
        "n_post": int(len(post_outcome)),
    }

    print(f"\n  Total effect (Treatment -> {outcome_col}):")
    print(f"    Pre mean: {results['total_effect']['pre_mean']}")
    print(f"    Post mean: {results['total_effect']['post_mean']}")
    print(f"    Effect: {total_effect:+.3f} (t={t_stat:.2f}, p={t_pval:.4f})")

    # Mediation analysis for each pathway
    for pathway_key, config in mediator_configs.items():
        mediator_col = config["mediator"]
        print(f"\n  Pathway: {config['label']} ({config['description']})")

        # Need treatment, mediator, and outcome
        pathway_df = df[[outcome_col, mediator_col, "treatment"]].dropna()

        if len(pathway_df) < 10:
            print(f"    Skipping - only {len(pathway_df)} complete rows")
            results["pathways"][pathway_key] = {
                "label": config["label"],
                "error": f"Insufficient data ({len(pathway_df)} rows)",
            }
            continue

        treatment = pathway_df["treatment"].values
        mediator = pathway_df[mediator_col].values
        outcome = pathway_df[outcome_col].values

        # Standardize for comparability
        from sklearn.preprocessing import StandardScaler
        scaler_m = StandardScaler()
        scaler_o = StandardScaler()
        mediator_std = scaler_m.fit_transform(mediator.reshape(-1, 1)).ravel()
        outcome_std = scaler_o.fit_transform(outcome.reshape(-1, 1)).ravel()

        def _mediation_estimates(treat: np.ndarray, med: np.ndarray,
                                 out: np.ndarray) -> dict[str, float]:
            """Compute mediation effect sizes using Baron-Kenny approach."""
            from numpy.linalg import lstsq

            n = len(treat)

            # Path a: treatment -> mediator
            X_a = np.column_stack([np.ones(n), treat])
            coef_a, _, _, _ = lstsq(X_a, med, rcond=None)
            a = coef_a[1]  # treatment coefficient

            # Path b + c': mediator + treatment -> outcome
            X_bc = np.column_stack([np.ones(n), med, treat])
            coef_bc, _, _, _ = lstsq(X_bc, out, rcond=None)
            b = coef_bc[1]  # mediator coefficient
            c_prime = coef_bc[2]  # direct effect of treatment

            # Total effect: treatment -> outcome
            X_c = np.column_stack([np.ones(n), treat])
            coef_c, _, _, _ = lstsq(X_c, out, rcond=None)
            c = coef_c[1]

            # Indirect (mediated) effect = a * b
            indirect = a * b

            # Proportion mediated (safe against zero total effect)
            prop_mediated = _safe_div(indirect, c, default=0.0)

            return {
                "a_path": float(a),
                "b_path": float(b),
                "c_total": float(c),
                "c_prime_direct": float(c_prime),
                "indirect_ab": float(indirect),
                "proportion_mediated": float(prop_mediated),
            }

        # Point estimates
        point_est = _mediation_estimates(treatment, mediator_std, outcome_std)

        # Bootstrap confidence intervals
        np.random.seed(42)
        bootstrap_indirect = []
        bootstrap_a = []
        bootstrap_b = []
        n_obs = len(treatment)

        for _ in range(BOOTSTRAP_N):
            idx = np.random.choice(n_obs, size=n_obs, replace=True)
            try:
                boot_est = _mediation_estimates(
                    treatment[idx], mediator_std[idx], outcome_std[idx]
                )
                bootstrap_indirect.append(boot_est["indirect_ab"])
                bootstrap_a.append(boot_est["a_path"])
                bootstrap_b.append(boot_est["b_path"])
            except Exception:
                continue

        bootstrap_indirect = np.array(bootstrap_indirect)
        bootstrap_a = np.array(bootstrap_a)
        bootstrap_b = np.array(bootstrap_b)

        alpha = (100 - BOOTSTRAP_CI) / 2

        if len(bootstrap_indirect) > 100:
            indirect_ci = (
                float(np.percentile(bootstrap_indirect, alpha)),
                float(np.percentile(bootstrap_indirect, 100 - alpha)),
            )
            # Sobel-like p-value: proportion of bootstrap samples crossing zero
            p_indirect = float(min(
                min(
                    (bootstrap_indirect <= 0).mean(),
                    (bootstrap_indirect >= 0).mean(),
                ) * 2,
                1.0,
            ))
        else:
            indirect_ci = (0, 0)
            p_indirect = 1.0

        # Pre/post mediator change
        pre_med = float(pathway_df.loc[pathway_df["treatment"] == 0, mediator_col].mean())
        post_med = float(pathway_df.loc[pathway_df["treatment"] == 1, mediator_col].mean())

        pathway_result = {
            "label": config["label"],
            "description": config["description"],
            "mediator": mediator_col,
            "expected_direction": config["expected_direction"],
            "n_observations": int(len(pathway_df)),
            "mediator_pre_mean": round(pre_med, 3),
            "mediator_post_mean": round(post_med, 3),
            "mediator_change": round(post_med - pre_med, 3),
            "a_path": round(point_est["a_path"], 4),
            "b_path": round(point_est["b_path"], 4),
            "total_effect": round(point_est["c_total"], 4),
            "direct_effect": round(point_est["c_prime_direct"], 4),
            "indirect_effect": round(point_est["indirect_ab"], 4),
            "indirect_ci_lower": round(indirect_ci[0], 4),
            "indirect_ci_upper": round(indirect_ci[1], 4),
            "proportion_mediated": round(point_est["proportion_mediated"] * 100, 1),
            "p_indirect": round(p_indirect, 6),
            "n_bootstrap": len(bootstrap_indirect),
            "bootstrap_indirect_distribution": bootstrap_indirect.tolist(),
        }

        # Determine if effect is in expected direction
        med_change = post_med - pre_med
        if config["expected_direction"] == "increase":
            pathway_result["favorable"] = med_change > 0
        else:
            pathway_result["favorable"] = med_change < 0

        direction = "favorable" if pathway_result["favorable"] else "unfavorable"
        sig = "significant" if p_indirect < 0.05 else "not significant"

        print(f"    a-path (T->M): {point_est['a_path']:+.4f}")
        print(f"    b-path (M->Y): {point_est['b_path']:+.4f}")
        print(f"    Indirect (a*b): {point_est['indirect_ab']:+.4f} "
              f"[{indirect_ci[0]:+.4f}, {indirect_ci[1]:+.4f}] p={p_indirect:.4f}")
        print(f"    Mediator: {pre_med:.2f} -> {post_med:.2f} ({med_change:+.2f}) [{direction}]")
        print(f"    Proportion mediated: {point_est['proportion_mediated']:.1%} [{sig}]")

        results["pathways"][pathway_key] = pathway_result

    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    print(f"\n  Mediation Analysis complete in {results['runtime_s']}s")
    return results


def plot_mediation(mediation_results: dict[str, Any]) -> list[go.Figure]:
    """Generate Plotly visualizations for mediation analysis results.

    Figure 1: Pathway effect sizes as horizontal bars with value labels and CIs.
    Figure 2: Bootstrap distributions with point estimates and CI shading.
    """
    figures = []

    pathways = mediation_results.get("pathways", {})
    valid_pathways = {k: v for k, v in pathways.items() if "error" not in v}

    if not valid_pathways:
        return figures

    # Figure 1: Pathway effect sizes with CIs
    labels = []
    effects = []
    ci_lower = []
    ci_upper = []
    colors = []
    sig_markers = []

    for key, p in valid_pathways.items():
        labels.append(p["label"])
        effects.append(p["indirect_effect"])
        ci_lower.append(p["indirect_ci_lower"])
        ci_upper.append(p["indirect_ci_upper"])
        colors.append(COLOR_EFFECT if p.get("favorable") else ACCENT_RED)
        sig_markers.append(p.get("p_indirect", 1) < 0.05)

    fig1 = go.Figure()

    fig1.add_trace(go.Bar(
        y=labels,
        x=effects,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=1, color="rgba(255,255,255,0.2)"),
        ),
        error_x=dict(
            type="data",
            symmetric=False,
            array=[u - e for u, e in zip(ci_upper, effects)],
            arrayminus=[e - l for l, e in zip(ci_lower, effects)],
            color="rgba(255,255,255,0.5)",
            thickness=2,
            width=4,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Indirect effect: %{x:.4f}<br>"
            "<extra></extra>"
        ),
        text=[f"{e:+.4f}" for e in effects],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRIMARY),
    ))

    # Zero reference line
    fig1.add_shape(
        type="line",
        x0=0, x1=0,
        y0=-0.5, y1=len(labels) - 0.5,
        line=dict(color=TEXT_TERTIARY, width=1, dash="dot"),
    )

    # Significance markers next to bars
    for i, (eff, sig) in enumerate(zip(effects, sig_markers)):
        if sig:
            x_pos = max(ci_upper[i], eff) + abs(eff) * 0.3 if eff > 0 else min(ci_lower[i], eff) - abs(eff) * 0.3
            fig1.add_annotation(
                x=x_pos, y=i,
                text="*",
                showarrow=False,
                font=dict(size=18, color=ACCENT_GREEN),
            )

    fig1.update_layout(
        title="",
        xaxis_title="Standardized indirect effect (a x b)",
        height=max(300, 80 * len(labels) + 100),
        **LAYOUT_DEFAULTS,
    )

    figures.append(fig1)

    # Figure 2: Bootstrap distributions
    n_valid = len(valid_pathways)
    if n_valid > 0:
        fig2 = make_subplots(
            rows=1, cols=n_valid,
            subplot_titles=[p["label"] for p in valid_pathways.values()],
            horizontal_spacing=0.08,
        )

        for i, (key, p) in enumerate(valid_pathways.items(), start=1):
            dist = p.get("bootstrap_indirect_distribution", [])
            if not dist:
                continue

            yref = f"y{i} domain" if i > 1 else "y domain"

            # Histogram with edge definition
            fig2.add_trace(
                go.Histogram(
                    x=dist,
                    nbinsx=50,
                    marker=dict(
                        color=colors[i - 1],
                        line=dict(width=0.5, color="rgba(255,255,255,0.1)"),
                    ),
                    opacity=0.75,
                    name=p["label"],
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{x:.4f}</b><br>"
                        "Count: %{y}<extra></extra>"
                    ),
                ),
                row=1, col=i,
            )

            # 95% CI shading
            ci_lo = p["indirect_ci_lower"]
            ci_hi = p["indirect_ci_upper"]
            fig2.add_vrect(
                x0=ci_lo, x1=ci_hi,
                fillcolor="rgba(59, 130, 246, 0.08)",
                line=dict(width=0),
                row=1, col=i,
            )

            # CI boundary lines
            for ci_bound in [ci_lo, ci_hi]:
                fig2.add_shape(
                    type="line",
                    x0=ci_bound, x1=ci_bound,
                    y0=0, y1=1, yref=yref,
                    line=dict(color="rgba(59, 130, 246, 0.4)", width=1, dash="dot"),
                    row=1, col=i,
                )

            # Zero line (null hypothesis)
            fig2.add_shape(
                type="line",
                x0=0, x1=0,
                y0=0, y1=1, yref=yref,
                line=dict(color=TEXT_TERTIARY, width=1.5, dash="dash"),
                row=1, col=i,
            )

            # Point estimate line (prominent)
            point_est = p["indirect_effect"]
            fig2.add_shape(
                type="line",
                x0=point_est, x1=point_est,
                y0=0, y1=1, yref=yref,
                line=dict(color="#FFFFFF", width=2),
                row=1, col=i,
            )

            # Point estimate annotation
            fig2.add_annotation(
                x=point_est, y=1, yref=yref,
                text=f"{point_est:+.4f}",
                showarrow=True,
                arrowhead=0,
                ax=0, ay=-20,
                font=dict(size=10, color="#FFFFFF"),
                bgcolor="rgba(59, 130, 246, 0.2)",
                bordercolor=ACCENT_BLUE,
                borderwidth=1,
                borderpad=2,
                row=1, col=i,
            )

            fig2.update_xaxes(title_text="Indirect effect", row=1, col=i)

        fig2.update_layout(
            title="",
            height=380,
            **{**LAYOUT_DEFAULTS, "margin": {**LAYOUT_DEFAULTS["margin"], "t": 100}},
        )

        figures.append(fig2)

    return figures


# ===========================================================================
# HTML REPORT GENERATION
# ===========================================================================

def generate_html_report(
    daily: pd.DataFrame,
    all_results: dict[str, dict],
    all_figures: dict[str, list[go.Figure]],
) -> str:
    """Generate self-contained interactive HTML report with dark theme."""
    print("\n[REPORT] Generating interactive HTML report...")

    # Convert all figures to JSON for embedding via script tags
    fig_divs: list[tuple[str, str]] = []
    fig_scripts: list[str] = []

    fig_counter = 0
    for section_key, figs in all_figures.items():
        for fig in figs:
            fig_id = f"fig_{fig_counter}"
            fig_divs.append((section_key, fig_id))
            fig_scripts.append(
                f"var {fig_id}_data = {fig.to_json()};\n"
                f"var {fig_id}_el = document.getElementById('{fig_id}');\n"
                f"if ({fig_id}_el) {{\n"
                f"    {fig_id}_el.innerHTML = '';\n"
                f"    Plotly.newPlot('{fig_id}', {fig_id}_data.data, {fig_id}_data.layout, "
                f"{{responsive: true}}).then((graphDiv) => {{\n"
                f"        window.__odtEnhancePlotly?.(graphDiv);\n"
                f"        Plotly.Plots.resize(graphDiv);\n"
                f"    }});\n"
                f"}}"
            )
            fig_counter += 1

    # Build section HTML
    ci_html = _build_ci_summary(all_results.get("causal_impact", {}))
    stat_power_html = _build_statistical_power_section(daily, all_results)
    placebo_html = _build_placebo_summary(all_results.get("placebo_tests", {}))
    pcmci_html = _build_pcmci_summary(all_results.get("pcmci", {}))
    te_html = _build_te_summary(all_results.get("transfer_entropy", {}))
    mediation_html = _build_mediation_summary(all_results.get("mediation", {}))

    # Organize figure divs by section
    section_figs: dict[str, list[str]] = {}
    for section_key, fig_id in fig_divs:
        section_figs.setdefault(section_key, []).append(fig_id)

    def _figs_for(section: str) -> str:
        ids = section_figs.get(section, [])
        return "\n".join(
            f'<div style="margin:16px 0"><div id="{fid}"></div></div>'
            for fid in ids
        )

    n_pre = (daily["period"] == "pre").sum()
    n_post = (daily["period"] == "post").sum()

    # Runtime summary
    runtimes = {}
    for key, result in all_results.items():
        if isinstance(result, dict):
            runtimes[key] = result.get("runtime_s", 0)
    total_runtime = sum(runtimes.values())

    # --- Build KPI row ---
    ci_streams = all_results.get("causal_impact", {}).get("streams", {})
    usable_streams = [
        s for s in ci_streams.values()
        if isinstance(s, dict) and "error" not in s
    ]
    n_sig_raw = sum(1 for s in usable_streams if s.get("p_value", 1) < 0.05)
    n_sig_fdr = sum(1 for s in usable_streams if s.get("significant_fdr", False))
    strongest_stream = min(
        usable_streams,
        key=lambda s: s.get("p_value", 1.0),
        default=None,
    )
    strongest_p = strongest_stream.get("p_value", 1.0) if strongest_stream else 1.0
    strongest_q = strongest_stream.get("q_value_bh", strongest_p) if strongest_stream else 1.0
    strongest_label = strongest_stream.get("label", "N/A") if strongest_stream else "N/A"

    post_status = "warning" if n_post < 14 else "normal"
    post_label = "Insufficient" if n_post < 14 else ""

    fdr_status = "normal" if n_sig_fdr > 0 else "warning"
    fdr_label = "" if n_sig_fdr > 0 else "None"

    lowest_p_status = "normal" if strongest_q < 0.05 else "warning"
    lowest_p_label = "Significant" if strongest_q < 0.05 else "Not significant"

    kpi_row = make_kpi_row(
        make_kpi_card("Pre-intervention", n_pre, "days", status="info", decimals=0),
        make_kpi_card("Post-intervention", n_post, "days", status=post_status, decimals=0, status_label=post_label),
        make_kpi_card("Raw p<0.05", f"{n_sig_raw}/{len(ci_streams)}", "", status="normal" if n_sig_raw > 0 else "info"),
        make_kpi_card("FDR-significant", f"{n_sig_fdr}/{len(ci_streams)}", "", status=fdr_status, status_label=fdr_label),
        make_kpi_card(
            "Lowest raw p",
            format_p_value(strongest_p),
            "",
            status=lowest_p_status,
            detail=f"{strongest_label} | q={strongest_q:.4f}",
            status_label=lowest_p_label,
        ),
        make_kpi_card("Methods used", "4", "", status="info", detail="CI + PCMCI+ + TE + Mediation"),
    )

    # --- Build body ---
    body_parts = []

    # Intro
    body_parts.append(f"""
    <div class="odt-narrative">
        Four complementary causal analysis methods explore whether Oura biometrics
        shifted after ruxolitinib
        (10 mg BID, started {TREATMENT_START}) on Oura Ring biometrics.
        <strong>Data period:</strong> {daily['date'].iloc[0]} to {daily['date'].iloc[-1]}
        ({len(daily)} days).
    </div>""")

    # Warning if short post period
    if n_post < 14:
        body_parts.append(f"""
    <div class="causal-warning">
        <strong>Warning:</strong> Post-intervention period is very short ({n_post} days).
        Results should be interpreted with caution. Minimum 14-21 days of post-intervention data recommended.
    </div>""")

    body_parts.append(kpi_row)

    # TOC
    body_parts.append("""
    <div class="causal-toc">
        <strong>Table of contents:</strong>
        <ol>
            <li><a href="#ci">CausalImpact - Bayesian Structural Time Series</a></li>
            <li><a href="#statpower">Statistical Power &amp; Interpretation</a></li>
            <li><a href="#placebo">Placebo tests (falsification)</a></li>
            <li><a href="#pcmci">Granger Causality Network (PCMCI+)</a></li>
            <li><a href="#te">Transfer Entropy</a></li>
            <li><a href="#mediation">Intervention Response Decomposition</a></li>
            <li><a href="#clinical">Clinical Interpretation</a></li>
        </ol>
    </div>""")

    # Section 1: CausalImpact
    ci_method = (
        '<div class="causal-method-note">'
        '<strong>Method:</strong> Bayesian Structural Time Series (BSTS) models pre-intervention dynamics '
        'and generates a counterfactual prediction for the post-period. The difference between actual and '
        'counterfactual estimates the causal effect, with full posterior uncertainty. '
        'MCMC: 5,000 iterations, weekly seasonal component (nseasons=7). '
        'Benjamini-Hochberg FDR correction for multiple testing.</div>'
    )
    body_parts.append(make_section(
        "1. CausalImpact - Bayesian Structural Time Series Analysis",
        ci_method + ci_html + _figs_for("causal_impact"),
        section_id="ci",
    ))

    # Section 1a: Statistical Power
    sp_method = (
        '<div class="causal-method-note">'
        '<strong>Purpose:</strong> All 11 metrics sorted by statistical significance, '
        'with Benjamini-Hochberg corrected q-values for multiple testing.</div>'
    )
    body_parts.append(make_section(
        "1a. Statistical Power &amp; Interpretation",
        sp_method + stat_power_html,
        section_id="statpower",
    ))

    # Section 1b: Placebo tests
    placebo_method = (
        '<div class="causal-method-note">'
        '<strong>Method:</strong> CausalImpact is run with 3 random placebo dates in the pre-period '
        'on the 3 most significant metrics. Placebo dates should NOT show significant effects.</div>'
    )
    body_parts.append(make_section(
        "1b. Placebo tests (intervention date falsification)",
        placebo_method + placebo_html,
        section_id="placebo",
    ))

    # Section 2: PCMCI+
    pcmci_method = (
        f'<div class="causal-method-note">'
        f'<strong>Method:</strong> PCMCI+ (tigramite) tests for time-lagged causal relationships '
        f'between biometric variables using partial correlation. '
        f'Tau_max = {PCMCI_TAU_MAX} days.</div>'
    )
    body_parts.append(make_section(
        "2. Granger Causality Network (PCMCI+)",
        pcmci_method + pcmci_html + _figs_for("pcmci"),
        section_id="pcmci",
    ))

    # Section 3: Transfer Entropy
    te_method = (
        '<div class="causal-method-note">'
        '<strong>Method:</strong> Transfer entropy quantifies directional information flow '
        'between biometric streams. Comparison of TE matrices for pre- and full period '
        'reveals changes in information coupling after ruxolitinib start.</div>'
    )
    body_parts.append(make_section(
        "3. Transfer Entropy",
        te_method + te_html + _figs_for("transfer_entropy"),
        section_id="te",
    ))

    # Section 4: Mediation Analysis
    med_method = (
        f'<div class="causal-method-note">'
        f'<strong>Method:</strong> Linear mediation analysis (Baron-Kenny) decomposes total '
        f'ruxolitinib effect into four mediating pathways. '
        f'Bootstrap ({BOOTSTRAP_N} iterations) for confidence intervals.</div>'
    )
    body_parts.append(make_section(
        "4. Intervention Response Decomposition",
        med_method + mediation_html + _figs_for("mediation"),
        section_id="mediation",
    ))

    # Section 5: Clinical interpretation
    body_parts.append(make_section(
        "5. Clinical Interpretation",
        _build_clinical_interpretation(daily, all_results),
        section_id="clinical",
    ))

    body = "\n".join(body_parts)

    # Extra CSS for causal-specific classes
    extra_css = f"""
/* Causal report-specific */
.causal-method-note {{
  padding: 12px 16px;
  background: rgba(59,130,246,0.08);
  border-left: 3px solid {ACCENT_BLUE};
  border-radius: 0 8px 8px 0;
  font-size: 0.9375rem;
  color: {TEXT_SECONDARY};
  margin-bottom: 16px;
}}
.causal-warning {{
  padding: 12px 16px;
  background: rgba(239,68,68,0.08);
  border-left: 3px solid {ACCENT_RED};
  border-radius: 0 8px 8px 0;
  margin-bottom: 16px;
  font-size: 0.9375rem;
  color: {TEXT_PRIMARY};
}}
.causal-clinical {{
  padding: 16px 20px;
  background: rgba(245,158,11,0.08);
  border-left: 3px solid {ACCENT_AMBER};
  border-radius: 0 8px 8px 0;
  margin: 16px 0;
  font-size: 0.9375rem;
  color: {TEXT_PRIMARY};
}}
.causal-clinical h3 {{
  color: {ACCENT_AMBER};
  margin-top: 16px;
  margin-bottom: 8px;
}}
.causal-clinical ul, .causal-clinical ol {{
  margin: 8px 0 8px 20px;
  color: {TEXT_SECONDARY};
}}
.causal-clinical li {{
  margin-bottom: 6px;
}}
.causal-toc {{
  background: {BG_ELEVATED};
  padding: 16px 24px;
  border-radius: 10px;
  margin: 20px 0;
  border: 1px solid {BORDER_SUBTLE};
}}
.causal-toc a {{
  color: {ACCENT_BLUE};
  text-decoration: none;
}}
.causal-toc a:hover {{ text-decoration: underline; }}
.causal-toc ol {{
  margin: 8px 0 0 20px;
  color: {TEXT_SECONDARY};
}}
.badge {{
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 0.85em;
  font-weight: 600;
}}
.badge-sig {{ background: rgba(16,185,129,0.15); color: {ACCENT_GREEN}; }}
.badge-ns {{ background: rgba(107,114,128,0.2); color: {TEXT_SECONDARY}; }}
.badge-warn {{ background: rgba(245,158,11,0.15); color: {ACCENT_AMBER}; }}
.badge-fail {{ background: rgba(239,68,68,0.15); color: {ACCENT_RED}; }}
.favorable {{ color: {ACCENT_GREEN}; }}
.unfavorable {{ color: {ACCENT_RED}; }}
.neutral {{ color: {TEXT_TERTIARY}; }}
.causal-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin: 16px 0;
}}
.causal-stat {{
  text-align: center;
  padding: 16px;
  background: {BG_ELEVATED};
  border-radius: 10px;
  border: 1px solid {BORDER_SUBTLE};
}}
.causal-stat .value {{ font-size: 1.5rem; font-weight: 700; color: {TEXT_PRIMARY}; }}
.causal-stat .label {{ font-size: 0.8125rem; color: {TEXT_SECONDARY}; margin-top: 4px; }}
.causal-data-badge {{
  background: rgba(59,130,246,0.1);
  border: 2px solid {ACCENT_BLUE};
  border-radius: 10px;
  padding: 16px 24px;
  margin: 16px 0;
  text-align: center;
}}
.causal-data-badge .value {{
  font-size: 1.5rem;
  font-weight: 700;
  color: {ACCENT_BLUE};
}}
.causal-data-badge .detail {{
  font-size: 0.875rem;
  color: {TEXT_SECONDARY};
  margin-top: 4px;
}}
"""

    # Figure rendering JS
    extra_js = "\n".join(fig_scripts)

    subtitle = (
        f"Data period: {daily['date'].iloc[0]} to {daily['date'].iloc[-1]} "
        f"({len(daily)} days) | Total runtime: {total_runtime:.1f}s"
    )

    return wrap_html(
        title="Causal Inference: Ruxolitinib",
        body_content=body,
        report_id="causal",
        subtitle=subtitle,
        extra_css=extra_css,
        extra_js=extra_js,
        data_end=daily["date"].iloc[-1],
        post_days=int((daily["date"] >= str(TREATMENT_START)).sum()),
    )


def _build_ci_summary(ci_results: dict[str, Any]) -> str:
    """Build CausalImpact summary section."""
    if "error" in ci_results:
        return (f'<div class="causal-warning"><p><span class="badge badge-fail">FAILED</span> '
                f'{ci_results["error"]}</p></div>')
    streams = ci_results.get("streams", {})
    if not streams:
        return '<p>No CausalImpact results available.</p>'

    rows = []
    for key, s in streams.items():
        if "error" in s and s.get("p_value", 1.0) == 1.0 and s.get("avg_effect", 0) == 0:
            # Complete failure - show warning row
            label = s.get("label", key)
            error_msg = s.get("error", "Unknown error")
            rows.append(
                f'<tr><td>{label}</td>'
                f'<td colspan="7">'
                f'<span class="badge-fail">FAILED</span> '
                f'<em>{error_msg}</em></td></tr>'
            )
            continue

        fav_class = "favorable" if s.get("favorable") else "unfavorable"

        # Use FDR-corrected significance if available
        q_val = s.get("q_value_bh")
        sig_fdr = s.get("significant_fdr", False)
        if q_val is not None:
            sig_badge = ('<span class="badge badge-sig">Sig (FDR)</span>'
                         if sig_fdr
                         else '<span class="badge badge-ns">NS (FDR)</span>')
            q_str = f'{q_val:.4f}'
        else:
            sig_badge = ('<span class="badge badge-sig">Significant</span>'
                         if s.get("p_value", 1) < 0.05
                         else '<span class="badge badge-ns">Not significant</span>')
            q_str = '-'

        # Low confidence warning badge
        low_conf_badge = ""
        if s.get("low_confidence"):
            low_conf_badge = (' <span class="badge-warn" '
                              f'title="{s.get("convergence_note", "")}">LOW CONFIDENCE</span>')

        ci_str = ""
        if s.get("ci_lower") is not None and s.get("ci_upper") is not None:
            ci_str = f'[{s["ci_lower"]:+.2f}, {s["ci_upper"]:+.2f}]'

        rows.append(
            f'<tr>'
            f'<td><strong>{s["label"]}</strong>{low_conf_badge}</td>'
            f'<td>{s.get("avg_actual_post", 0):.2f}</td>'
            f'<td>{s.get("avg_counterfactual_post", 0):.2f}</td>'
            f'<td class="{fav_class}">{s.get("avg_effect", 0):+.2f} {s["unit"]}</td>'
            f'<td>{_format_relative_effect_html(s.get("relative_effect_pct"))}</td>'
            f'<td>{ci_str}</td>'
            f'<td>{s.get("p_value", 1):.4f}</td>'
            f'<td>{q_str} {sig_badge}</td>'
            f'</tr>'
        )

    return f"""
        <div class="causal-method-note">
            <strong>FDR correction:</strong> The Benjamini-Hochberg method is applied to control
            for multiple testing (11 simultaneous tests). q-values (adjusted p-values) below 0.05
            indicate statistical significance after FDR correction.
        </div>
        <table>
            <thead><tr>
                <th>Stream</th><th>Actual (post)</th><th>Counterfactual</th>
                <th>Causal effect</th><th>Relative</th><th>95% CI</th><th>p-value</th><th>q-value (BH)</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>"""


def _build_placebo_summary(placebo_results: dict[str, Any]) -> str:
    """Build placebo test summary section."""
    if not placebo_results or "error" in placebo_results:
        err = placebo_results.get("error", "No placebo results") if placebo_results else "Not run"
        return f'<p><em>{err}</em></p>'

    tests = placebo_results.get("tests", [])
    placebo_dates = placebo_results.get("placebo_dates", [])
    metrics_tested = placebo_results.get("metrics_tested", [])
    validation = placebo_results.get("placebo_validation", "UNKNOWN")
    n_sig = placebo_results.get("n_significant_placebo", 0)
    n_total = placebo_results.get("n_total_tests", 0)

    # Validation badge
    if validation == "PASS":
        val_badge = '<span class="badge badge-sig">PASSED</span>'
        val_class = "favorable"
    elif validation == "PARTIAL":
        val_badge = '<span class="badge-warn">PARTIAL</span>'
        val_class = "neutral"
    else:
        val_badge = '<span class="badge badge-ns">NOT PASSED</span>'
        val_class = "unfavorable"

    html_parts = []

    # Summary stats
    html_parts.append(f"""
    <div class="causal-grid">
        <div class="causal-stat">
            <div class="value {val_class}">{validation}</div>
            <div class="label">Validation result {val_badge}</div>
        </div>
        <div class="causal-stat">
            <div class="value">{n_sig}/{n_total}</div>
            <div class="label">Significant placebo tests</div>
        </div>
        <div class="causal-stat">
            <div class="value">{len(placebo_dates)}</div>
            <div class="label">Placebo dates tested</div>
        </div>
    </div>""")

    # Metrics tested
    if metrics_tested:
        metric_list = ", ".join(f'{m["label"]} (q={m["q_value"]:.4f})' for m in metrics_tested)
        html_parts.append(f'<p><strong>Metrics tested:</strong> {metric_list}</p>')

    html_parts.append(f'<p><strong>Placebo dates:</strong> {", ".join(placebo_dates)}</p>')

    # Results table
    valid_tests = [t for t in tests if "error" not in t]
    error_tests = [t for t in tests if "error" in t]

    if valid_tests:
        rows = []
        for t in valid_tests:
            sig_class = "unfavorable" if t.get("significant") else "favorable"
            sig_label = "Sig (false alarm)" if t.get("significant") else "NS (expected)"
            rows.append(
                f'<tr>'
                f'<td>{t["placebo_date"]}</td>'
                f'<td>{t.get("label", t["metric"])}</td>'
                f'<td>{t.get("n_pre", "-")}</td>'
                f'<td>{t.get("n_post", "-")}</td>'
                f'<td>{t.get("avg_effect", 0):+.3f}</td>'
                f'<td>{t.get("p_value", 1):.4f}</td>'
                f'<td class="{sig_class}">{sig_label}</td>'
                f'</tr>'
            )

        html_parts.append(f"""
        <table>
            <thead><tr>
                <th>Placebo date</th><th>Metric</th><th>N pre</th><th>N post</th>
                <th>Effect</th><th>p-value</th><th>Result</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>""")

    if error_tests:
        err_items = ", ".join(f'{t.get("label", t.get("metric", "?"))} @ {t.get("placebo_date", "?")} ({t["error"]})'
                              for t in error_tests)
        html_parts.append(f'<p><em>Errors in {len(error_tests)} tests: {err_items}</em></p>')

    # Interpretation
    if validation == "PASS":
        html_parts.append("""
        <div class="causal-clinical">
            <strong>Interpretation:</strong> No placebo dates produced significant results.
            This strengthens the conclusion that changes observed after March 16 are linked to
            ruxolitinib initiation and not random fluctuations in the time series.
        </div>""")
    elif validation == "PARTIAL":
        html_parts.append("""
        <div class="causal-warning">
            <strong>Interpretation:</strong> One placebo date produced a significant result.
            This may indicate natural variability in the time series. CausalImpact results
            should be interpreted with extra caution.
        </div>""")
    else:
        html_parts.append("""
        <div class="causal-warning">
            <strong>Interpretation:</strong> Multiple placebo dates produced significant results.
            The CausalImpact model may have specification issues, or there is too much
            variability in the pre-period to draw causal conclusions with this method.
        </div>""")

    return "".join(html_parts)


def _build_pcmci_summary(pcmci_results: dict[str, Any]) -> str:
    """Build PCMCI+ summary section."""
    if "error" in pcmci_results:
        return f'<p>Error: {pcmci_results["error"]}</p>'

    html_parts = []

    for period_key, label in [("full_period", "Full period"), ("pre_period", "Pre-ruxolitinib")]:
        period = pcmci_results.get(period_key, {})
        if "error" in period:
            html_parts.append(f'<p><em>{label}: {period["error"]}</em></p>')
            continue

        links = period.get("significant_links", [])
        n = period.get("n_observations", 0)

        if not links:
            html_parts.append(f'<p><strong>{label}</strong> ({n} days): No significant causal links found.</p>')
            continue

        rows = []
        for link in links[:10]:
            rows.append(
                f'<tr>'
                f'<td>{link["source"]}</td>'
                f'<td>{link["target"]}</td>'
                f'<td>{link["lag"]} days</td>'
                f'<td>{link["val"]:+.3f}</td>'
                f'<td>{link["p_value"]:.4f}</td>'
                f'</tr>'
            )

        html_parts.append(f"""
        <h3>{label} ({n} days, {len(links)} significant links)</h3>
        <table>
            <thead><tr><th>Source</th><th>Target</th><th>Lag</th><th>Correlation</th><th>p-value</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>""")

    # Network comparison
    comp = pcmci_results.get("network_comparison", {})
    if comp:
        new = comp.get("new_links_in_full", [])
        lost = comp.get("lost_links_from_pre", [])
        if new or lost:
            html_parts.append("<h3>Network change after ruxolitinib</h3>")
            if new:
                items = ", ".join(f"{l['source']}->{l['target']}(lag={l['lag']})" for l in new)
                html_parts.append(f'<p class="favorable"><strong>New links:</strong> {items}</p>')
            if lost:
                items = ", ".join(f"{l['source']}->{l['target']}(lag={l['lag']})" for l in lost)
                html_parts.append(f'<p class="unfavorable"><strong>Lost links:</strong> {items}</p>')

    return "".join(html_parts)


def _build_te_summary(te_results: dict[str, Any]) -> str:
    """Build Transfer Entropy summary section."""
    if "error" in te_results:
        return f'<p>Error: {te_results["error"]}</p>'

    html_parts = []

    for period_key, label in [("full_period", "Full period"), ("pre_period", "Pre-ruxolitinib")]:
        period = te_results.get(period_key, {})
        if "error" in period:
            html_parts.append(f'<p><em>{label}: {period["error"]}</em></p>')
            continue

        links = period.get("top_links", [])
        n = period.get("n_observations", 0)

        if not links:
            html_parts.append(f'<p><strong>{label}</strong> ({n} days): No significant TE links.</p>')
            continue

        rows = []
        for link in links[:8]:
            net_str = f'{link["net_te"]:+.4f}' if link["net_te"] != 0 else "-"
            rows.append(
                f'<tr>'
                f'<td>{link["source"]}</td>'
                f'<td>{link["target"]}</td>'
                f'<td>{link["te"]:.4f}</td>'
                f'<td>{net_str}</td>'
                f'</tr>'
            )

        html_parts.append(f"""
        <h3>{label} ({n} days)</h3>
        <table>
            <thead><tr><th>Source</th><th>Target</th><th>TE (bits)</th><th>Net TE</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>""")

    # TE difference
    diff = te_results.get("te_difference")
    if diff:
        max_inc = diff.get("max_increase", {})
        max_dec = diff.get("max_decrease", {})
        html_parts.append(f"""
        <h3>Change in information flow</h3>
        <p><strong>Largest increase:</strong> {max_inc.get('pair', 'N/A')} ({max_inc.get('value', 0):+.4f} bits)</p>
        <p><strong>Largest decrease:</strong> {max_dec.get('pair', 'N/A')} ({max_dec.get('value', 0):+.4f} bits)</p>
        """)

    return "".join(html_parts)


def _build_mediation_summary(mediation_results: dict[str, Any]) -> str:
    """Build mediation analysis summary section."""
    if "error" in mediation_results:
        return f'<p>Error: {mediation_results["error"]}</p>'

    total = mediation_results.get("total_effect", {})
    pathways = mediation_results.get("pathways", {})

    html_parts = []

    # Total effect summary
    if total:
        sig_badge = ('<span class="badge badge-sig">Significant</span>'
                     if total.get("p_value", 1) < 0.05
                     else '<span class="badge badge-ns">Not significant</span>')
        html_parts.append(f"""
        <div class="causal-grid">
            <div class="causal-stat">
                <div class="value">{total.get('effect', 0):+.1f}</div>
                <div class="label">Total effect (readiness score)</div>
            </div>
            <div class="causal-stat">
                <div class="value">{total.get('pre_mean', 0):.1f} -> {total.get('post_mean', 0):.1f}</div>
                <div class="label">Pre -> Post average</div>
            </div>
            <div class="causal-stat">
                <div class="value">{format_p_value(total.get('p_value', 1.0))}</div>
                <div class="label">Raw p-value {sig_badge}</div>
            </div>
        </div>""")

    # Pathway table
    if pathways:
        rows = []
        for key, p in pathways.items():
            if "error" in p:
                rows.append(
                    f'<tr><td>{p.get("label", key)}</td>'
                    f'<td colspan="7"><em>{p["error"]}</em></td></tr>'
                )
                continue

            fav_class = "favorable" if p.get("favorable") else "unfavorable"
            sig_badge = ('<span class="badge badge-sig">Sig</span>'
                         if p.get("p_indirect", 1) < 0.05
                         else '<span class="badge badge-ns">NS</span>')

            rows.append(
                f'<tr>'
                f'<td><strong>{p["label"]}</strong><br>'
                f'<small>{p["description"]}</small></td>'
                f'<td>{p["mediator_pre_mean"]:.2f} -> {p["mediator_post_mean"]:.2f}</td>'
                f'<td>{p["a_path"]:+.3f}</td>'
                f'<td>{p["b_path"]:+.3f}</td>'
                f'<td class="{fav_class}">{p["indirect_effect"]:+.4f}<br>'
                f'<small>[{p["indirect_ci_lower"]:+.4f}, {p["indirect_ci_upper"]:+.4f}]</small></td>'
                f'<td>{p["proportion_mediated"]:.1f}%</td>'
                f'<td>{p["p_indirect"]:.4f} {sig_badge}</td>'
                f'</tr>'
            )

        html_parts.append(f"""
        <table>
            <thead><tr>
                <th>Pathway</th><th>Mediator (pre->post)</th><th>a (T->M)</th>
                <th>b (M->Y)</th><th>Indirect effect [95% CI]</th>
                <th>% mediated</th><th>p-value</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>""")

    return "".join(html_parts)


def _build_statistical_power_section(
    daily: pd.DataFrame,
    all_results: dict[str, dict],
) -> str:
    """Build the Statistical Power & Interpretation section.

    Shows all 11 metrics sorted by p-value with BH-corrected q-values,
    interpretation text, placebo test summary, and data collection status.
    """
    ci_results = all_results.get("causal_impact", {})
    placebo_results = all_results.get("placebo_tests", {})
    streams = ci_results.get("streams", {})

    if not streams:
        return '<p>No CausalImpact results available for statistical power analysis.</p>'

    # --- Calculate post-intervention days from data ---
    rux_str = str(TREATMENT_START)
    post_dates = daily.loc[daily["date"] >= rux_str, "date"]
    n_post_days = len(post_dates)
    last_data_date = daily["date"].iloc[-1] if len(daily) > 0 else rux_str

    # --- Build sorted metrics list ---
    metric_rows = []
    for key, s in streams.items():
        if not isinstance(s, dict):
            continue
        # Skip complete failures with no useful data
        if "error" in s and s.get("p_value", 1.0) == 1.0 and s.get("avg_effect", 0) == 0:
            continue

        p_val = s.get("p_value", 1.0)
        q_val = s.get("q_value_bh", p_val)
        effect = s.get("avg_effect", 0.0)
        rel_effect = s.get("relative_effect_pct")
        higher_better = s.get("higher_is_better", True)

        # Direction
        if effect > 0:
            direction_arrow = "&#8593;"  # up arrow
            direction_text = "increased"
        elif effect < 0:
            direction_arrow = "&#8595;"  # down arrow
            direction_text = "decreased"
        else:
            direction_arrow = "&#8212;"  # em dash
            direction_text = "no change"

        # Significance status
        if q_val < 0.05:
            sig_status = "FDR significant"
            sig_class = "favorable"
        elif p_val < 0.05:
            sig_status = "Raw p&lt;0.05 only"
            sig_class = "neutral"
        elif p_val < 0.10:
            sig_status = "Near-significant"
            sig_class = "neutral"
        else:
            sig_status = "Not significant"
            sig_class = ""

        metric_rows.append({
            "label": s.get("label", key),
            "key": key,
            "direction_arrow": direction_arrow,
            "direction_text": direction_text,
            "effect": effect,
            "unit": s.get("unit", ""),
            "rel_effect": rel_effect,
            "p_value": p_val,
            "q_value": q_val,
            "low_confidence": s.get("low_confidence", False),
            "sig_status": sig_status,
            "sig_class": sig_class,
        })

    # Sort by p-value ascending
    metric_rows.sort(key=lambda x: x["p_value"])

    # --- Find the strongest signal for lead text ---
    strongest = metric_rows[0] if metric_rows else None

    # --- Find specific metrics for interpretation text ---
    def _find_p(metric_key: str) -> str:
        s = streams.get(metric_key, {})
        p = s.get("p_value", 1.0) if isinstance(s, dict) else 1.0
        return format_p_value(p)

    p_lowest_hr = _find_p("lowest_heart_rate")
    p_mean_rmssd = _find_p("mean_rmssd")
    p_rem_dur = _find_p("rem_sleep_duration")

    # --- Data collection status badge ---
    badge_html = f"""
    <div class="causal-data-badge">
        <div class="value">Day {n_post_days} of ongoing monitoring</div>
        <div class="detail">Data through {last_data_date}. Next milestone at Day 14.</div>
    </div>"""

    # --- Lead text ---
    lead_html = ""
    if strongest:
        strongest_q_sentence = (
            "remains significant after Benjamini-Hochberg correction"
            if strongest["q_value"] < 0.05 else
            "does not remain significant after Benjamini-Hochberg correction"
        )
        lead_html = f"""
    <p style="font-size: 1.05em; line-height: 1.7; color: {TEXT_PRIMARY};">
        <strong>{strongest['label']}</strong> is the strongest hypothesis-generating raw p-value signal
        ({format_p_value(strongest['p_value'])}, q={strongest['q_value']:.3f}) but {strongest_q_sentence}.
    </p>"""

    tests = placebo_results.get("tests", []) if isinstance(placebo_results, dict) else []
    valid_tests = [t for t in tests if isinstance(t, dict) and "error" not in t]
    n_sig_placebo = sum(1 for t in valid_tests if t.get("significant", False))

    # --- Complete metrics table ---
    table_rows = []
    for i, m in enumerate(metric_rows):
        bg = BG_ELEVATED if i % 2 == 0 else BG_SURFACE
        table_rows.append(
            f'<tr style="background:{bg};">'
            f'<td><strong>{m["label"]}</strong></td>'
            f'<td style="text-align:center;">{m["direction_arrow"]} {m["direction_text"]}</td>'
            f'<td style="text-align:right;">{m["effect"]:+.2f} {m["unit"]}</td>'
            f'<td style="text-align:right;">{_format_relative_effect_html(m["rel_effect"])}</td>'
            f'<td style="text-align:right;">{m["p_value"]:.4f}</td>'
            f'<td style="text-align:right;">{m["q_value"]:.4f}</td>'
            f'<td class="{m["sig_class"]}">{m["sig_status"]}</td>'
            f'</tr>'
        )

    metrics_table = f"""
    <table>
        <thead><tr>
            <th>Metric</th><th style="text-align:center;">Direction</th>
            <th style="text-align:right;">Absolute effect</th>
            <th style="text-align:right;">Relative effect</th>
            <th style="text-align:right;">Raw p-value</th>
            <th style="text-align:right;">BH q-value</th>
            <th>Significance</th>
        </tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
    </table>"""

    # --- Interpretation text ---
    placebo_clause = (
        f"{n_sig_placebo}/{len(valid_tests)} placebo tests also reached p&lt;0.05"
        if valid_tests else
        "placebo validation was not available in this run"
    )
    strongest_label = strongest["label"] if strongest else "the leading metric"
    interpretation_html = f"""
    <div class="causal-method-note" style="margin-top: 20px;">
        <strong>Interpretation:</strong>
        With {n_post_days} post-intervention days, <strong>{strongest_label}</strong> is the strongest
        raw p-value signal, but no stream remains significant after BH correction. {placebo_clause},
        so the current result should be treated as hypothesis-generating rather than confirmed.
        Autonomic metrics (HRV, lowest HR, REM) still trend in the expected direction and may
        stabilize with additional follow-up. A 14-day post-intervention window is expected to
        clarify borderline metrics (lowest HR {p_lowest_hr}, HRV {p_mean_rmssd},
        REM {p_rem_dur}).
    </div>"""

    # --- Placebo test summary ---
    placebo_html = ""
    if valid_tests:
        placebo_rows = []
        for t in valid_tests:
            sig_text = "Yes" if t.get("significant") else "No"
            sig_style = f'color: {ACCENT_RED}; font-weight: 600;' if t.get("significant") else f'color: {ACCENT_GREEN};'
            placebo_rows.append(
                f'<tr>'
                f'<td>{t.get("placebo_date", "")}</td>'
                f'<td>{t.get("label", t.get("metric", ""))}</td>'
                f'<td style="text-align:right;">{t.get("p_value", 1):.4f}</td>'
                f'<td style="{sig_style}">{sig_text}</td>'
                f'</tr>'
            )

        placebo_html = f"""
    <h3 style="margin-top: 24px; color: {TEXT_PRIMARY};">Placebo test summary</h3>
    <table>
        <thead><tr>
            <th>Placebo date</th><th>Metric</th>
            <th style="text-align:right;">p-value</th><th>Significant?</th>
        </tr></thead>
        <tbody>{''.join(placebo_rows)}</tbody>
    </table>
    <p style="margin-top: 8px;">
        <strong>{n_sig_placebo}/{len(valid_tests)}</strong> placebo tests reached significance.
        This {'supports' if n_sig_placebo == 0 else 'tempers'} the March 16 signal and keeps the
        current result in the hypothesis-generating category.
    </p>"""

    return f"""
    {badge_html}
    {lead_html}
    {metrics_table}
    {interpretation_html}
    {placebo_html}
    """


def _build_clinical_interpretation(
    daily: pd.DataFrame,
    all_results: dict[str, dict],
) -> str:
    """Build clinical interpretation combining all four methods."""
    n_pre = (daily["period"] == "pre").sum()
    n_post = (daily["period"] == "post").sum()

    # Count significant findings across methods (use FDR-corrected if available)
    sig_ci_raw = 0
    sig_ci_fdr = 0
    ci_streams = all_results.get("causal_impact", {}).get("streams", {})
    for s in ci_streams.values():
        if isinstance(s, dict) and s.get("p_value", 1) < 0.05:
            sig_ci_raw += 1
        if isinstance(s, dict) and s.get("significant_fdr", False):
            sig_ci_fdr += 1

    # Find strongest signal for executive summary
    strongest_label = "N/A"
    strongest_p = 1.0
    strongest_q = 1.0
    for s in ci_streams.values():
        if isinstance(s, dict) and "error" not in s:
            p = s.get("p_value", 1.0)
            if p < strongest_p:
                strongest_p = p
                strongest_q = s.get("q_value_bh", p)
                strongest_label = s.get("label", "?")

    # Count metrics trending in expected direction
    n_favorable = sum(
        1 for s in ci_streams.values()
        if isinstance(s, dict) and s.get("favorable", False)
    )
    n_total_streams = sum(
        1 for s in ci_streams.values()
        if isinstance(s, dict) and "error" not in s
    )

    pcmci_full = all_results.get("pcmci", {}).get("full_period", {})
    n_causal_links = pcmci_full.get("n_significant_links", 0) if isinstance(pcmci_full, dict) else 0

    n_pathways_sig = 0
    pathways = all_results.get("mediation", {}).get("pathways", {})
    for p in pathways.values():
        if isinstance(p, dict) and p.get("p_indirect", 1) < 0.05:
            n_pathways_sig += 1

    # Placebo validation
    placebo = all_results.get("placebo_tests", {})
    placebo_validation = placebo.get("placebo_validation", "N/A")
    n_sig_placebo = placebo.get("n_significant_placebo", "?")
    n_total_placebo = placebo.get("n_total_tests", "?")

    # Post-intervention days
    rux_str = str(TREATMENT_START)
    n_post_data = (daily["date"] >= rux_str).sum()

    return f"""
    <div class="causal-clinical">
        <h3>Executive summary</h3>
        <div class="causal-grid" style="margin-bottom: 16px;">
            <div class="causal-stat">
                <div class="value favorable">{strongest_label}</div>
                <div class="label">Strongest signal: p={strongest_p:.3f}</div>
            </div>
            <div class="causal-stat">
                <div class="value">{n_favorable}/{n_total_streams}</div>
                <div class="label">Metrics trending in expected direction</div>
            </div>
            <div class="causal-stat">
                <div class="value">{n_post_data}</div>
                <div class="label">Post-intervention days (Day 14 target)</div>
            </div>
        </div>
        <ul>
            <li><strong>{strongest_label}:</strong> strongest hypothesis-generating raw p-value signal
            (p={strongest_p:.4f}, q={strongest_q:.4f}). It {"survives" if strongest_q < 0.05 else "does not survive"}
            FDR correction, so confirmation still depends on more post-treatment follow-up.</li>
            <li><strong>CausalImpact:</strong> {sig_ci_raw} of {len(ci_streams)} biometric streams show
            significant causal change (p &lt; 0.05). After Benjamini-Hochberg FDR correction:
            {sig_ci_fdr} of {len(ci_streams)} remain significant (q &lt; 0.05).</li>
            <li><strong>Placebo validation:</strong> {placebo_validation} - {n_sig_placebo}/{n_total_placebo}
            placebo tests reached significance. This {"provides supportive descriptive evidence" if placebo_validation == "PASS" else "keeps the result vulnerable to false positives"}
            rather than establishing a confirmed intervention effect.</li>
            <li><strong>PCMCI+:</strong> {n_causal_links} significant time-lagged causal links
            identified in the biometric network</li>
            <li><strong>Mediation analysis:</strong> {n_pathways_sig} of {len(pathways)} mediating pathways
            show significant indirect effect</li>
        </ul>

        <h3>Limitations</h3>
        <ul>
            <li><strong>Short post-period ({n_post} days):</strong> All results are preliminary.
            Minimum 14-21 days of post-intervention data recommended for robust causal inference.</li>
            <li><strong>Confounders:</strong> Linear methods cannot capture non-linear interactions.
            Seasonal variation, activity level, and other medications are not controlled for.</li>
            <li><strong>Wearable data:</strong> Oura Ring is not a medical device.
            Measurements have inherent noise that can affect causal estimates.</li>
            <li><strong>Single patient:</strong> N=1 study without control group. Causality cannot be
            definitively established, but Bayesian posterior probability of effect provides a strength measure.</li>
            <li><strong>HEV diagnosis:</strong> HEV was diagnosed 2026-03-18 (2 days after
            ruxolitinib start). Hepatitis may confound biometric changes.</li>
        </ul>

        <h3>Recommendations</h3>
        <ol>
            <li>Repeat analysis after 2-3 weeks of ruxolitinib treatment for robust causal inference</li>
            <li>Add HEV-related biomarkers (ALT, bilirubin) as time-varying covariates</li>
            <li>Consider synthetic control method when longer time series are available</li>
            <li>Combine with clinical endpoints (GVHD scoring, ferritin) for multimodal analysis</li>
        </ol>
    </div>"""


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """Run all causal inference methods and generate report."""
    print("=" * 70)
    print("  CAUSAL INFERENCE ENGINE - Oura Ring Biometric Streams")
    print(f"  {PATIENT_LABEL} | Ruxolitinib 10mg BID from {TREATMENT_START}")
    print("=" * 70)
    t_total = time.perf_counter()

    # Load data
    data = load_data()
    daily = build_daily_matrix(data)

    all_results: dict[str, dict] = {}
    all_figures: dict[str, list[go.Figure]] = {}

    # ---------------------------------------------------------------
    # Section 1: CausalImpact
    # ---------------------------------------------------------------
    try:
        all_results["causal_impact"] = run_causal_impact(daily)
        all_figures["causal_impact"] = plot_causal_impact(all_results["causal_impact"])
        print(f"  [OK] CausalImpact: {len(all_figures.get('causal_impact', []))} figures")
    except Exception as e:
        print(f"  [ERROR] CausalImpact failed: {e}")
        traceback.print_exc()
        all_results["causal_impact"] = {"error": str(e), "method": "CausalImpact"}
        all_figures["causal_impact"] = []

    # ---------------------------------------------------------------
    # Section 1b: Placebo tests (requires CausalImpact results)
    # ---------------------------------------------------------------
    try:
        ci_res = all_results.get("causal_impact", {})
        if "error" not in ci_res:
            all_results["placebo_tests"] = run_placebo_tests(daily, ci_res)
            print(f"  [OK] Placebo tests: {all_results['placebo_tests'].get('placebo_validation', 'N/A')}")
        else:
            all_results["placebo_tests"] = {"error": "CausalImpact failed, skipping placebo"}
    except Exception as e:
        print(f"  [ERROR] Placebo tests failed: {e}")
        traceback.print_exc()
        all_results["placebo_tests"] = {"error": str(e), "method": "Placebo tests"}

    # ---------------------------------------------------------------
    # Section 2: PCMCI+
    # ---------------------------------------------------------------
    try:
        all_results["pcmci"] = run_pcmci(daily)
        all_figures["pcmci"] = plot_pcmci(all_results["pcmci"])
        print(f"  [OK] PCMCI+: {len(all_figures.get('pcmci', []))} figures")
    except Exception as e:
        print(f"  [ERROR] PCMCI+ failed: {e}")
        traceback.print_exc()
        all_results["pcmci"] = {"error": str(e), "method": "PCMCI+"}
        all_figures["pcmci"] = []

    # ---------------------------------------------------------------
    # Section 3: Transfer Entropy
    # ---------------------------------------------------------------
    try:
        all_results["transfer_entropy"] = run_transfer_entropy(daily)
        all_figures["transfer_entropy"] = plot_transfer_entropy(all_results["transfer_entropy"])
        print(f"  [OK] Transfer Entropy: {len(all_figures.get('transfer_entropy', []))} figures")
    except Exception as e:
        print(f"  [ERROR] Transfer Entropy failed: {e}")
        traceback.print_exc()
        all_results["transfer_entropy"] = {"error": str(e), "method": "Transfer Entropy"}
        all_figures["transfer_entropy"] = []

    # ---------------------------------------------------------------
    # Section 4: Mediation Analysis
    # ---------------------------------------------------------------
    try:
        all_results["mediation"] = run_mediation_analysis(daily)
        all_figures["mediation"] = plot_mediation(all_results["mediation"])
        print(f"  [OK] Mediation: {len(all_figures.get('mediation', []))} figures")
    except Exception as e:
        print(f"  [ERROR] Mediation Analysis failed: {e}")
        traceback.print_exc()
        all_results["mediation"] = {"error": str(e), "method": "Mediation Analysis"}
        all_figures["mediation"] = []

    # ---------------------------------------------------------------
    # Generate HTML report
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[REPORT] Building final HTML report...")
    print("=" * 70)

    html_content = generate_html_report(daily, all_results, all_figures)
    HTML_OUTPUT.write_text(html_content, encoding="utf-8")
    print(f"  HTML report saved: {HTML_OUTPUT}")

    # ---------------------------------------------------------------
    # Generate JSON metrics
    # ---------------------------------------------------------------
    def _sanitize_for_json(obj: Any) -> Any:
        """Make object JSON-serializable."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (date, datetime)):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize_for_json(v) for v in obj]
        return obj

    # Build metrics (exclude large arrays for manageable JSON)
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "generated": generated_at,
        "generated_at": generated_at,
        "data_range": {
            "start": daily["date"].iloc[0],
            "end": daily["date"].iloc[-1],
            "n_days": len(daily),
            "n_pre": int((daily["period"] == "pre").sum()),
            "n_post": int((daily["period"] == "post").sum()),
        },
        "ruxolitinib_start": str(TREATMENT_START),
    }

    for method_key, result in all_results.items():
        if isinstance(result, dict):
            # Strip large arrays (time series, bootstrap distributions)
            stripped = {}
            for k, v in result.items():
                if k in ("ts_dates", "ts_actual", "ts_predicted", "ts_pred_lower",
                         "ts_pred_upper", "bootstrap_indirect_distribution",
                         "val_matrix", "p_matrix"):
                    continue
                if isinstance(v, dict):
                    inner = {}
                    for ik, iv in v.items():
                        if ik in ("ts_dates", "ts_actual", "ts_predicted",
                                  "ts_pred_lower", "ts_pred_upper",
                                  "bootstrap_indirect_distribution",
                                  "val_matrix", "p_matrix"):
                            continue
                        # Also strip nested large arrays
                        if isinstance(iv, dict):
                            inner[ik] = {
                                iik: iiv for iik, iiv in iv.items()
                                if iik not in ("ts_dates", "ts_actual", "ts_predicted",
                                               "ts_pred_lower", "ts_pred_upper",
                                               "bootstrap_indirect_distribution",
                                               "val_matrix", "p_matrix",
                                               "te_matrix", "net_te_matrix",
                                               "diff_matrix")
                            }
                        else:
                            inner[ik] = iv
                    stripped[k] = inner
                else:
                    stripped[k] = v
            metrics[method_key] = stripped

    metrics = _sanitize_for_json(metrics)

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON metrics saved: {JSON_OUTPUT}")

    # ---------------------------------------------------------------
    # Causal time series JSON (consumed by 3D dashboard)
    # ---------------------------------------------------------------
    ci_results = all_results.get("causal_impact", {})
    ci_streams = ci_results.get("streams", {})
    ts_data: dict[str, Any] = {
        "generated": generated_at,
        "generated_at": generated_at,
        "streams": {},
    }
    for stream_key, s in ci_streams.items():
        if not isinstance(s, dict) or "error" in s:
            continue
        ts_data["streams"][stream_key] = {
            "label": s.get("label", stream_key),
            "p_value": s.get("p_value", 1.0),
            "relative_effect_pct": s.get("relative_effect_pct"),
            "intervention_idx": s.get("n_pre", 0),
            "dates": s.get("ts_dates", []),
            "actual": s.get("ts_actual", []),
            "predicted": s.get("ts_predicted", []),
            "pred_lower": s.get("ts_pred_lower", []),
            "pred_upper": s.get("ts_pred_upper", []),
        }
    ts_data = _sanitize_for_json(ts_data)
    with open(TS_JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(ts_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Time series JSON saved: {TS_JSON_OUTPUT}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print("  CAUSAL INFERENCE COMPLETE")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  HTML report: {HTML_OUTPUT}")
    print(f"  JSON metrics: {JSON_OUTPUT}")

    # Quick summary of findings
    for method_key, result in all_results.items():
        if isinstance(result, dict) and "error" not in result:
            status = "OK"
            runtime = result.get("runtime_s", 0)
        else:
            status = f"FAILED: {result.get('error', 'unknown')}" if isinstance(result, dict) else "FAILED"
            runtime = 0
        print(f"  {method_key}: {status} ({runtime:.1f}s)")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
