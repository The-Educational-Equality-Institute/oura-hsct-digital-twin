#!/usr/bin/env python3
"""
Sequential CausalImpact Analysis - Isolate Jakavi vs. Beta-Blocker Effects

N-of-1 study with two interventions introduced at different times:
  - Ruxolitinib (Jakavi): started 2026-03-16
  - Beta-blocker: added 2026-04-08 on top of Jakavi

A single pooled CausalImpact run confounds the two drugs. This script runs
two separate analyses to disentangle their individual causal effects:

  Run A - Jakavi effect (isolated):
    Pre:  DATA_START to 2026-03-15  (no treatment)
    Post: 2026-03-16 to 2026-04-07  (Jakavi-only window, before BB)

  Run B - Marginal beta-blocker effect:
    Pre:  2026-03-16 to 2026-04-07  (Jakavi-only as new baseline)
    Post: 2026-04-08 to latest data  (Jakavi + BB)

Metrics analyzed: mean_rmssd, lowest_heart_rate, average_heart_rate,
                  sleep_efficiency

Output:
  - reports/sequential_causal_impact.html
  - reports/sequential_causal_impact_metrics.json

Usage:
    python analysis/analyze_sequential_ci.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# pandas>=3 removed DataFrame.applymap; pycausalimpact still calls it.
if not hasattr(pd.DataFrame, "applymap") and hasattr(pd.DataFrame, "map"):
    pd.DataFrame.applymap = pd.DataFrame.map  # type: ignore[attr-defined]

# pandas>=3 removed positional integer indexing on Series with named index;
# pycausalimpact internally does mu[0], sig[0] which triggers KeyError.
_orig_series_getitem = pd.Series.__getitem__
def _patched_series_getitem(self, key):  # type: ignore[override]
    try:
        return _orig_series_getitem(self, key)
    except KeyError:
        if isinstance(key, int):
            return self.iloc[key]
        raise
pd.Series.__getitem__ = _patched_series_getitem  # type: ignore[assignment]

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from statsmodels.tools.sm_exceptions import ValueWarning
    warnings.filterwarnings("ignore", category=ValueWarning)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Import guards for CausalImpact
# ---------------------------------------------------------------------------
CAUSALIMPACT_AVAILABLE = False
CausalImpact = None

try:
    from causalimpact import CausalImpact as _CI
    CausalImpact = _CI
    CAUSALIMPACT_AVAILABLE = True
except ImportError:
    pass

if not CAUSALIMPACT_AVAILABLE:
    try:
        from tfcausalimpact import CausalImpact as _TFCI
        CausalImpact = _TFCI
        CAUSALIMPACT_AVAILABLE = True
    except ImportError:
        pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution & patient config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH,
    REPORTS_DIR,
    TREATMENT_START,
    BETA_BLOCKER_START,
    DATA_START,
    PATIENT_LABEL,
)

from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    format_p_value,
    disclaimer_banner,
    COLORWAY,
    STATUS_COLORS,
    BG_PRIMARY,
    BG_SURFACE,
    BG_ELEVATED,
    BORDER_SUBTLE,
    BORDER_DEFAULT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_PURPLE,
    ACCENT_CYAN,
    C_PRE_TX,
    C_POST_TX,
    C_RUX_LINE,
    C_EFFECT,
    C_COUNTERFACTUAL,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "sequential_causal_impact.html"
JSON_OUTPUT = REPORTS_DIR / "sequential_causal_impact_metrics.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_PRE_DAYS = 7
MIN_POST_DAYS = 3
MAX_NAN_FRACTION = 0.40
CI_WIDTH_WARN_RATIO = 2.0
SAFE_DIV_EPS = 1e-15
NITER = 5000

# Metric definitions - the four metrics to analyze
METRIC_DEFS = {
    "mean_rmssd": {
        "label": "HRV Mean RMSSD",
        "unit": "ms",
        "higher_is_better": True,
        "description": "Nocturnal heart rate variability (parasympathetic tone)",
    },
    "lowest_heart_rate": {
        "label": "Lowest Heart Rate",
        "unit": "bpm",
        "higher_is_better": False,
        "description": "Minimum sleeping heart rate (cardiovascular recovery)",
    },
    "average_heart_rate": {
        "label": "Average Heart Rate",
        "unit": "bpm",
        "higher_is_better": False,
        "description": "Mean sleeping heart rate (autonomic load indicator)",
    },
    "sleep_efficiency": {
        "label": "Sleep Efficiency",
        "unit": "%",
        "higher_is_better": True,
        "description": "Percentage of time in bed spent asleep",
    },
}

# Run definitions
RUN_A_LABEL = "Run A: Jakavi Effect (Isolated)"
RUN_B_LABEL = "Run B: Marginal Beta-Blocker Effect"

# Colors for the two runs
COLOR_RUN_A = ACCENT_BLUE
COLOR_RUN_B = ACCENT_PURPLE
COLOR_COUNTERFACTUAL = C_COUNTERFACTUAL
COLOR_CI_BAND_A = "rgba(59, 130, 246, 0.12)"
COLOR_CI_BAND_B = "rgba(139, 92, 246, 0.12)"

LAYOUT_DEFAULTS = dict(
    margin=dict(l=70, r=30, t=60, b=40),
)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < SAFE_DIV_EPS:
        return default
    return num / den


def _safe_pct(num: float, den: float) -> float:
    return _safe_div(num, den, default=0.0) * 100


def _favorable(effect: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return effect > 0
    return effect < 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_metrics() -> pd.DataFrame:
    """Load and aggregate daily metrics from the Oura database."""
    print("[DATA] Loading biometric data...")

    if not DATABASE_PATH.exists():
        print(f"ERROR: Database not found at {DATABASE_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)

    # HRV: aggregate 5-min epochs to daily mean_rmssd
    hrv = pd.read_sql_query(
        "SELECT timestamp, rmssd FROM oura_hrv ORDER BY timestamp", conn
    )
    hrv["date"] = pd.to_datetime(hrv["timestamp"], utc=True).dt.date.astype(str)
    hrv["rmssd"] = pd.to_numeric(hrv["rmssd"], errors="coerce")
    hrv_daily = (
        hrv.groupby("date")
        .agg(mean_rmssd=("rmssd", "mean"))
        .reset_index()
    )

    # Sleep periods: lowest_heart_rate, average_heart_rate, efficiency.
    # A date can carry several long_sleep periods (a split night); keep the
    # longest so every date appears exactly once, matching the convention in
    # analyze_oura_causal.py. Without this the daily matrix carries duplicate
    # dates and reindexing onto a continuous date range raises
    # "cannot reindex on an axis with duplicate labels".
    sleep = pd.read_sql_query(
        """SELECT day as date, lowest_heart_rate, average_heart_rate, efficiency,
                  total_sleep_duration
           FROM oura_sleep_periods
           WHERE type = 'long_sleep'
           ORDER BY day""",
        conn,
    )
    for col in ["lowest_heart_rate", "average_heart_rate", "efficiency",
                "total_sleep_duration"]:
        sleep[col] = pd.to_numeric(sleep[col], errors="coerce")
    sleep = (
        sleep.sort_values(["date", "total_sleep_duration"], ascending=[True, False])
        .drop_duplicates(subset="date", keep="first")
        .drop(columns=["total_sleep_duration"])
        .reset_index(drop=True)
    )
    sleep = sleep.rename(columns={"efficiency": "sleep_efficiency"})

    conn.close()

    # Merge on date
    daily = hrv_daily.merge(sleep, on="date", how="outer")
    daily = daily.sort_values("date").reset_index(drop=True)

    # Filter to analysis window
    daily = daily[daily["date"] >= str(DATA_START)].copy()

    print(f"  Daily matrix: {len(daily)} rows, {daily['date'].min()} to {daily['date'].max()}")
    for col in METRIC_DEFS:
        valid = daily[col].notna().sum()
        print(f"    {col}: {valid} valid days")

    return daily


# ---------------------------------------------------------------------------
# CausalImpact runner for a single metric + single run
# ---------------------------------------------------------------------------

def run_single_ci(
    daily: pd.DataFrame,
    metric: str,
    meta: dict,
    pre_start_date: date,
    pre_end_date: date,
    post_start_date: date,
    post_end_date: date,
    run_label: str,
) -> dict[str, Any]:
    """Run CausalImpact for one metric on one pre/post period definition.

    Returns a result dict with effect estimates, p-value, time series, etc.
    """
    result: dict[str, Any] = {
        "label": meta["label"],
        "unit": meta["unit"],
        "higher_is_better": meta["higher_is_better"],
        "run": run_label,
        "pre_start": str(pre_start_date),
        "pre_end": str(pre_end_date),
        "post_start": str(post_start_date),
        "post_end": str(post_end_date),
    }

    if not CAUSALIMPACT_AVAILABLE:
        result["error"] = "CausalImpact not installed. Install: pip install pycausalimpact"
        result["significant"] = False
        result["favorable"] = False
        return result

    # Subset data to the full window
    df = daily[["date", metric]].copy()
    df = df[(df["date"] >= str(pre_start_date)) & (df["date"] <= str(post_end_date))]

    # NaN guard
    total = len(df)
    nans = df[metric].isna().sum()
    nan_frac = _safe_div(nans, total, default=1.0)
    if nan_frac > MAX_NAN_FRACTION:
        result["error"] = f"Too many NaN: {nans}/{total} ({nan_frac:.0%})"
        result["significant"] = False
        result["favorable"] = False
        return result

    df = df.dropna(subset=[metric])
    if len(df) < 8:
        result["error"] = f"Only {len(df)} data points (need >= 8)"
        result["significant"] = False
        result["favorable"] = False
        return result

    # Pre/post counts
    pre_str = str(pre_end_date)
    n_pre = (df["date"] <= pre_str).sum()
    n_post = (df["date"] > pre_str).sum()
    result["n_pre"] = n_pre
    result["n_post"] = n_post

    if n_pre < MIN_PRE_DAYS:
        result["error"] = f"Only {n_pre} pre-period days (need >= {MIN_PRE_DAYS})"
        result["significant"] = False
        result["favorable"] = False
        return result
    if n_post < MIN_POST_DAYS:
        result["error"] = f"Only {n_post} post-period days (need >= {MIN_POST_DAYS})"
        result["significant"] = False
        result["favorable"] = False
        return result

    # Build continuous time series with DatetimeIndex
    ts = df.set_index(pd.to_datetime(df["date"]))[[metric]].copy()
    ts.index.name = None
    full_range = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
    ts = ts.reindex(full_range)
    ts[metric] = ts[metric].interpolate(method="linear", limit=3)
    ts = ts.dropna()

    if len(ts) < 8:
        result["error"] = f"Only {len(ts)} points after reindex"
        result["significant"] = False
        result["favorable"] = False
        return result

    # Define period strings
    ci_pre_start = str(ts.index.min().date())
    ci_pre_end = str(pre_end_date)
    ci_post_start = str(post_start_date)
    ci_post_end = str(ts.index.max().date())

    # Ensure post_start is in the index
    if pd.Timestamp(ci_post_start) not in ts.index:
        post_dates = ts.index[ts.index >= pd.Timestamp(ci_post_start)]
        if len(post_dates) == 0:
            result["error"] = "No post-period data after reindex"
            result["significant"] = False
            result["favorable"] = False
            return result
        ci_post_start = str(post_dates[0].date())

    print(f"      Pre: {ci_pre_start} to {ci_pre_end} ({n_pre} days)")
    print(f"      Post: {ci_post_start} to {ci_post_end} ({n_post} days)")

    try:
        ci = CausalImpact(
            ts, [ci_pre_start, ci_pre_end], [ci_post_start, ci_post_end],
            niter=NITER, nseasons=[{"period": 7}],
        )

        inferences = ci.inferences
        post_ts = pd.Timestamp(ci_post_start)
        post_mask = np.array(ts.index >= post_ts)
        actual_post = ts.loc[post_mask, metric].values

        # Extract predictions aligned to the time series
        if inferences is not None and len(inferences) == len(ts):
            inf_post = inferences.iloc[post_mask]
            pred_post = inf_post["preds"].values
            pred_lower = inf_post["preds_lower"].values
            pred_upper = inf_post["preds_upper"].values
        elif inferences is not None:
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

        # Compute effect statistics
        if len(actual_post) > 0 and len(pred_post) > 0:
            avg_actual = float(np.nanmean(actual_post))
            avg_predicted = float(np.nanmean(pred_post))
            avg_effect = float(np.nanmean(actual_post - pred_post))
            rel_effect = _safe_pct(avg_effect, avg_predicted)
        else:
            avg_actual = 0.0
            avg_predicted = 0.0
            avg_effect = 0.0
            rel_effect = 0.0

        p_value = getattr(ci, "p_value", None)
        if p_value is None or not np.isfinite(p_value):
            p_value = 1.0

        # Credible interval on the average effect
        if len(pred_lower) > 0 and len(pred_upper) > 0:
            ci_lower_effect = float(np.nanmean(actual_post - pred_upper))
            ci_upper_effect = float(np.nanmean(actual_post - pred_lower))
        else:
            ci_lower_effect = None
            ci_upper_effect = None

        # Convergence check
        low_confidence = False
        if ci_lower_effect is not None and ci_upper_effect is not None:
            ci_width = abs(ci_upper_effect - ci_lower_effect)
            if abs(avg_effect) > SAFE_DIV_EPS and ci_width > CI_WIDTH_WARN_RATIO * abs(avg_effect):
                low_confidence = True

        significant = p_value < 0.05
        is_favorable = _favorable(avg_effect, meta["higher_is_better"])

        # Full time series for plotting
        all_dates = [str(d.date()) for d in ts.index]
        all_actual = ts[metric].tolist()

        # Full predictions (pre + post)
        if inferences is not None and len(inferences) == len(ts):
            all_predicted = inferences["preds"].tolist()
            all_pred_lower = inferences["preds_lower"].tolist()
            all_pred_upper = inferences["preds_upper"].tolist()
        else:
            all_predicted = [None] * len(ts)
            all_pred_lower = [None] * len(ts)
            all_pred_upper = [None] * len(ts)

        result.update({
            "avg_actual_post": round(avg_actual, 2),
            "avg_counterfactual_post": round(avg_predicted, 2),
            "avg_effect": round(avg_effect, 2),
            "relative_effect_pct": round(rel_effect, 1),
            "p_value": round(p_value, 4),
            "probability_of_effect": round(1.0 - p_value, 4),
            "ci_lower": round(ci_lower_effect, 2) if ci_lower_effect is not None else None,
            "ci_upper": round(ci_upper_effect, 2) if ci_upper_effect is not None else None,
            "significant": significant,
            "favorable": is_favorable,
            "low_confidence": low_confidence,
            "ts_dates": all_dates,
            "ts_actual": all_actual,
            "ts_predicted": all_predicted,
            "ts_pred_lower": all_pred_lower,
            "ts_pred_upper": all_pred_upper,
        })

        direction = "+" if avg_effect > 0 else ""
        sig_marker = "*" if significant else ""
        print(f"      Effect: {direction}{avg_effect:.2f} {meta['unit']} "
              f"({rel_effect:+.1f}%), {format_p_value(p_value)}{sig_marker}")

    except Exception as exc:
        result["error"] = f"CausalImpact failed: {type(exc).__name__}: {exc}"
        result["significant"] = False
        result["favorable"] = False
        print(f"      ERROR: {result['error']}")

    return result


# ---------------------------------------------------------------------------
# Main analysis runner
# ---------------------------------------------------------------------------

def run_sequential_analysis(daily: pd.DataFrame) -> dict[str, Any]:
    """Run both sequential CausalImpact analyses for all metrics."""
    print("\n" + "=" * 72)
    print("SEQUENTIAL CAUSALIMPACT ANALYSIS")
    print("=" * 72)
    t0 = time.perf_counter()

    # Determine actual data end date
    data_end = pd.to_datetime(daily["date"].max()).date()

    # Period definitions
    # Run A: Isolate Jakavi
    run_a_pre_start = DATA_START
    run_a_pre_end = TREATMENT_START - timedelta(days=1)  # 2026-03-15
    run_a_post_start = TREATMENT_START                    # 2026-03-16
    run_a_post_end = BETA_BLOCKER_START - timedelta(days=1)  # 2026-04-07

    # Run B: Marginal beta-blocker
    run_b_pre_start = TREATMENT_START                     # 2026-03-16
    run_b_pre_end = BETA_BLOCKER_START - timedelta(days=1)  # 2026-04-07
    run_b_post_start = BETA_BLOCKER_START                 # 2026-04-08
    run_b_post_end = data_end                             # latest data

    results: dict[str, Any] = {
        "method": "Sequential CausalImpact (BSTS)",
        "description": "Two separate CausalImpact runs to isolate each drug's causal effect",
        "run_a": {
            "label": RUN_A_LABEL,
            "pre_period": f"{run_a_pre_start} to {run_a_pre_end}",
            "post_period": f"{run_a_post_start} to {run_a_post_end}",
            "streams": {},
        },
        "run_b": {
            "label": RUN_B_LABEL,
            "pre_period": f"{run_b_pre_start} to {run_b_pre_end}",
            "post_period": f"{run_b_post_start} to {run_b_post_end}",
            "streams": {},
        },
        "data_end": str(data_end),
        "runtime_s": 0,
    }

    for metric, meta in METRIC_DEFS.items():
        # Run A
        print(f"\n  [{RUN_A_LABEL}] {meta['label']}...")
        res_a = run_single_ci(
            daily, metric, meta,
            run_a_pre_start, run_a_pre_end, run_a_post_start, run_a_post_end,
            "A",
        )
        results["run_a"]["streams"][metric] = res_a

        # Run B
        print(f"  [{RUN_B_LABEL}] {meta['label']}...")
        res_b = run_single_ci(
            daily, metric, meta,
            run_b_pre_start, run_b_pre_end, run_b_post_start, run_b_post_end,
            "B",
        )
        results["run_b"]["streams"][metric] = res_b

    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    print(f"\n  Total runtime: {results['runtime_s']:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

def _build_ci_chart(
    result: dict,
    meta: dict,
    run_label: str,
    intervention_date: date,
    color_main: str,
    color_band: str,
) -> go.Figure | None:
    """Build a time series chart for one CausalImpact run."""
    if "error" in result:
        return None
    if "ts_dates" not in result or not result["ts_dates"]:
        return None

    dates = result["ts_dates"]
    actual = result["ts_actual"]
    predicted = result["ts_predicted"]
    pred_lower = result["ts_pred_lower"]
    pred_upper = result["ts_pred_upper"]

    fig = go.Figure()

    # Credible interval band
    valid_lower = [v for v in pred_lower if v is not None]
    valid_upper = [v for v in pred_upper if v is not None]
    if valid_lower and valid_upper:
        band_dates = [d for d, v in zip(dates, pred_lower) if v is not None]
        band_lower = [v for v in pred_lower if v is not None]
        band_upper = [v for v in pred_upper if v is not None]
        fig.add_trace(go.Scatter(
            x=band_dates + band_dates[::-1],
            y=band_upper + band_lower[::-1],
            fill="toself",
            fillcolor=color_band,
            line=dict(width=0),
            name="95% Credible Interval",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Counterfactual prediction
    valid_pred = [(d, v) for d, v in zip(dates, predicted) if v is not None]
    if valid_pred:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in valid_pred],
            y=[p[1] for p in valid_pred],
            mode="lines",
            name="Counterfactual",
            line=dict(color=COLOR_COUNTERFACTUAL, width=2, dash="dash"),
        ))

    # Actual observed data
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode="lines+markers",
        name="Observed",
        line=dict(color=color_main, width=2.5),
        marker=dict(size=4),
    ))

    # Intervention line
    fig.add_vline(
        x=str(intervention_date),
        line_dash="dot",
        line_color=ACCENT_AMBER,
        line_width=2,
    )
    ann_text = run_label.split(":")[1].strip() if ":" in run_label else run_label
    fig.add_annotation(
        x=str(intervention_date), y=1, yref="paper",
        text=ann_text, showarrow=False,
        font=dict(color=ACCENT_AMBER, size=11),
        yshift=10,
    )

    effect_str = f"{result.get('avg_effect', 0):+.2f} {meta['unit']}"
    p_str = format_p_value(result.get("p_value"))
    title_suffix = f" | Effect: {effect_str}, {p_str}"

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"{meta['label']}{title_suffix}", font=dict(size=14)),
        xaxis_title="Date",
        yaxis_title=f"{meta['label']} ({meta['unit']})",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def _build_comparison_chart(
    results: dict,
    metric: str,
    meta: dict,
) -> go.Figure:
    """Build a combined chart showing both runs overlaid on the full timeline."""
    fig = go.Figure()

    res_a = results["run_a"]["streams"].get(metric, {})
    res_b = results["run_b"]["streams"].get(metric, {})

    # Collect all actual data from both runs
    all_dates = []
    all_actual = []

    if "ts_dates" in res_a and res_a["ts_dates"]:
        all_dates.extend(res_a["ts_dates"])
        all_actual.extend(res_a["ts_actual"])
    if "ts_dates" in res_b and res_b["ts_dates"]:
        # Only add dates not already in Run A
        existing = set(all_dates)
        for d, v in zip(res_b["ts_dates"], res_b["ts_actual"]):
            if d not in existing:
                all_dates.append(d)
                all_actual.append(v)

    if all_dates:
        # Sort by date
        paired = sorted(zip(all_dates, all_actual))
        all_dates = [p[0] for p in paired]
        all_actual = [p[1] for p in paired]

        fig.add_trace(go.Scatter(
            x=all_dates,
            y=all_actual,
            mode="lines+markers",
            name="Observed",
            line=dict(color=TEXT_PRIMARY, width=2),
            marker=dict(size=3),
        ))

    # Run A counterfactual
    if "ts_predicted" in res_a:
        valid = [(d, v) for d, v in zip(res_a.get("ts_dates", []), res_a.get("ts_predicted", []))
                 if v is not None]
        if valid:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in valid],
                y=[p[1] for p in valid],
                mode="lines",
                name="Run A Counterfactual",
                line=dict(color=COLOR_RUN_A, width=2, dash="dash"),
            ))

    # Run B counterfactual
    if "ts_predicted" in res_b:
        valid = [(d, v) for d, v in zip(res_b.get("ts_dates", []), res_b.get("ts_predicted", []))
                 if v is not None]
        if valid:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in valid],
                y=[p[1] for p in valid],
                mode="lines",
                name="Run B Counterfactual",
                line=dict(color=COLOR_RUN_B, width=2, dash="dash"),
            ))

    # Intervention lines
    fig.add_vline(
        x=str(TREATMENT_START), line_dash="dot", line_color=COLOR_RUN_A, line_width=2,
    )
    fig.add_annotation(
        x=str(TREATMENT_START), y=1, yref="paper",
        text="Jakavi Start", showarrow=False,
        font=dict(color=COLOR_RUN_A, size=11),
        yshift=10,
    )
    fig.add_vline(
        x=str(BETA_BLOCKER_START), line_dash="dot", line_color=COLOR_RUN_B, line_width=2,
    )
    fig.add_annotation(
        x=str(BETA_BLOCKER_START), y=1, yref="paper",
        text="BB Start", showarrow=False,
        font=dict(color=COLOR_RUN_B, size=11),
        yshift=10,
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"{meta['label']} - Full Timeline", font=dict(size=14)),
        xaxis_title="Date",
        yaxis_title=f"{meta['label']} ({meta['unit']})",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _effect_status(result: dict, meta: dict) -> str:
    """Determine KPI status string from result."""
    if "error" in result:
        return "warning"
    if not result.get("significant", False):
        return "info"
    if result.get("favorable", False):
        return "good"
    return "critical"


def _effect_detail(result: dict, meta: dict) -> str:
    """Build detail string for KPI card."""
    if "error" in result:
        return result["error"]
    parts = []
    p = result.get("p_value")
    if p is not None:
        parts.append(format_p_value(p))
    sig = result.get("significant", False)
    parts.append("Significant" if sig else "Not significant")
    fav = result.get("favorable", False)
    if sig:
        parts.append("Favorable" if fav else "Unfavorable")
    if result.get("low_confidence"):
        parts.append("Low confidence")
    return " | ".join(parts)


def _results_table_html(run_results: dict, run_label: str) -> str:
    """Build an HTML summary table for one run's results."""
    rows = []
    for metric, meta in METRIC_DEFS.items():
        res = run_results.get(metric, {})
        if "error" in res:
            rows.append(
                f'<tr><td>{meta["label"]}</td>'
                f'<td colspan="6" style="color:{ACCENT_AMBER};">{res["error"]}</td></tr>'
            )
            continue

        effect = res.get("avg_effect", 0)
        rel = res.get("relative_effect_pct", 0)
        p = res.get("p_value", 1.0)
        ci_lo = res.get("ci_lower")
        ci_hi = res.get("ci_upper")
        sig = res.get("significant", False)
        fav = res.get("favorable", False)

        # Direction indicator
        if sig and fav:
            badge = f'<span style="color:{ACCENT_GREEN};">Favorable</span>'
        elif sig and not fav:
            badge = f'<span style="color:{ACCENT_RED};">Unfavorable</span>'
        else:
            badge = f'<span style="color:{TEXT_TERTIARY};">Not significant</span>'

        ci_str = f"[{ci_lo:+.2f}, {ci_hi:+.2f}]" if ci_lo is not None and ci_hi is not None else "N/A"

        rows.append(
            f'<tr>'
            f'<td>{meta["label"]}</td>'
            f'<td style="text-align:right;">{effect:+.2f} {meta["unit"]}</td>'
            f'<td style="text-align:right;">{rel:+.1f}%</td>'
            f'<td style="text-align:right;">{ci_str}</td>'
            f'<td style="text-align:right;">{format_p_value(p)}</td>'
            f'<td style="text-align:center;">{badge}</td>'
            f'</tr>'
        )

    return f"""
    <table style="width:100%;border-collapse:collapse;margin:1em 0;">
    <thead>
    <tr style="border-bottom:2px solid {BORDER_DEFAULT};">
        <th style="text-align:left;padding:8px 12px;">Metric</th>
        <th style="text-align:right;padding:8px 12px;">Absolute Effect</th>
        <th style="text-align:right;padding:8px 12px;">Relative Effect</th>
        <th style="text-align:right;padding:8px 12px;">95% CI</th>
        <th style="text-align:right;padding:8px 12px;">Tail-area Prob</th>
        <th style="text-align:center;padding:8px 12px;">Verdict</th>
    </tr>
    </thead>
    <tbody>
    {"".join(rows)}
    </tbody>
    </table>
    """


def build_html_report(results: dict, daily: pd.DataFrame) -> str:
    """Assemble the full HTML report."""
    print("\n[REPORT] Building HTML...")

    data_end = pd.to_datetime(daily["date"].max()).date()
    body_parts: list[str] = []

    # --- Methodology section ---
    methodology = f"""
    <div style="color:{TEXT_SECONDARY};line-height:1.7;margin-bottom:1em;">
    <p>A single pooled CausalImpact analysis confounds the effects of Jakavi
    (ruxolitinib) and the beta-blocker because both interventions fall within
    the post-period. This report runs <strong>two separate Bayesian Structural
    Time Series (BSTS)</strong> analyses to disentangle their individual contributions:</p>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5em;margin:1em 0;">
      <div style="background:{BG_ELEVATED};padding:1em 1.2em;border-radius:8px;border-left:3px solid {COLOR_RUN_A};">
        <strong style="color:{COLOR_RUN_A};">Run A: Jakavi Effect (Isolated)</strong><br>
        <span style="color:{TEXT_TERTIARY};">Pre-period:</span> {DATA_START} to {TREATMENT_START - timedelta(days=1)}<br>
        <span style="color:{TEXT_TERTIARY};">Post-period:</span> {TREATMENT_START} to {BETA_BLOCKER_START - timedelta(days=1)}<br>
        <em>Estimates what Jakavi alone does, before the beta-blocker is introduced.</em>
      </div>
      <div style="background:{BG_ELEVATED};padding:1em 1.2em;border-radius:8px;border-left:3px solid {COLOR_RUN_B};">
        <strong style="color:{COLOR_RUN_B};">Run B: Marginal Beta-Blocker Effect</strong><br>
        <span style="color:{TEXT_TERTIARY};">Pre-period:</span> {TREATMENT_START} to {BETA_BLOCKER_START - timedelta(days=1)}<br>
        <span style="color:{TEXT_TERTIARY};">Post-period:</span> {BETA_BLOCKER_START} to {data_end}<br>
        <em>Uses the Jakavi-only window as the new baseline, then estimates what
        the beta-blocker adds on top.</em>
      </div>
    </div>

    <p style="color:{TEXT_TERTIARY};font-size:0.9em;">
    Each run uses {NITER} MCMC iterations with weekly seasonality.
    Posterior tail-area probability &lt; 0.05 indicates statistical significance.
    Credible intervals are 95% highest posterior density.</p>
    </div>
    """
    body_parts.append(make_section("Methodology: Sequential Causal Decomposition", methodology, section_id="methodology"))

    # --- KPI summary cards ---
    run_a_streams = results["run_a"]["streams"]
    run_b_streams = results["run_b"]["streams"]

    # Count significants
    a_sig = sum(1 for r in run_a_streams.values() if r.get("significant"))
    b_sig = sum(1 for r in run_b_streams.values() if r.get("significant"))
    a_fav = sum(1 for r in run_a_streams.values() if r.get("significant") and r.get("favorable"))
    b_fav = sum(1 for r in run_b_streams.values() if r.get("significant") and r.get("favorable"))

    kpi_cards = make_kpi_row(
        make_kpi_card(
            "JAKAVI SIGNIFICANT",
            f"{a_sig}/{len(METRIC_DEFS)}",
            status="good" if a_sig > 0 else "info",
            detail=f"{a_fav} favorable" if a_sig > 0 else "No significant effects detected",
        ),
        make_kpi_card(
            "BB SIGNIFICANT",
            f"{b_sig}/{len(METRIC_DEFS)}",
            status="good" if b_sig > 0 else "info",
            detail=f"{b_fav} favorable" if b_sig > 0 else "No significant effects detected",
        ),
        make_kpi_card(
            "RUNTIME",
            results.get("runtime_s", 0),
            unit="s",
            decimals=1,
            status="neutral",
            detail=f"2 runs x {len(METRIC_DEFS)} metrics",
        ),
        make_kpi_card(
            "DATA RANGE",
            f"{(data_end - DATA_START).days}",
            unit="days",
            status="neutral",
            detail=f"{DATA_START} to {data_end}",
        ),
    )
    body_parts.append(kpi_cards)

    # --- Run A: Jakavi section ---
    run_a_table = _results_table_html(run_a_streams, RUN_A_LABEL)
    run_a_charts = ""
    chart_data_dict: dict[str, str] = {}

    for metric, meta in METRIC_DEFS.items():
        res = run_a_streams.get(metric, {})
        fig = _build_ci_chart(res, meta, RUN_A_LABEL, TREATMENT_START, COLOR_RUN_A, COLOR_CI_BAND_A)
        if fig is not None:
            chart_key = f"run_a_{metric}"
            chart_data_dict[chart_key] = fig.to_json()
            run_a_charts += (
                f'<div id="chart-{chart_key}" class="chart-box" '
                f'data-chart="{chart_key}" style="height:350px;">Loading...</div>'
            )

    body_parts.append(make_section(
        f"{RUN_A_LABEL}",
        f"""<p style="color:{TEXT_SECONDARY};margin-bottom:0.5em;">
        Jakavi started {TREATMENT_START}. Pre-period: {DATA_START} to {TREATMENT_START - timedelta(days=1)}
        ({(TREATMENT_START - DATA_START).days} days).
        Post-period: {TREATMENT_START} to {BETA_BLOCKER_START - timedelta(days=1)}
        ({(BETA_BLOCKER_START - TREATMENT_START).days} days, before beta-blocker).
        </p>{run_a_table}{run_a_charts}""",
        section_id="run-a",
    ))

    # --- Run B: Beta-blocker section ---
    run_b_table = _results_table_html(run_b_streams, RUN_B_LABEL)
    run_b_charts = ""

    for metric, meta in METRIC_DEFS.items():
        res = run_b_streams.get(metric, {})
        fig = _build_ci_chart(res, meta, RUN_B_LABEL, BETA_BLOCKER_START, COLOR_RUN_B, COLOR_CI_BAND_B)
        if fig is not None:
            chart_key = f"run_b_{metric}"
            chart_data_dict[chart_key] = fig.to_json()
            run_b_charts += (
                f'<div id="chart-{chart_key}" class="chart-box" '
                f'data-chart="{chart_key}" style="height:350px;">Loading...</div>'
            )

    bb_post_days = (data_end - BETA_BLOCKER_START).days + 1
    body_parts.append(make_section(
        f"{RUN_B_LABEL}",
        f"""<p style="color:{TEXT_SECONDARY};margin-bottom:0.5em;">
        Beta-blocker started {BETA_BLOCKER_START}. Pre-period (Jakavi-only baseline):
        {TREATMENT_START} to {BETA_BLOCKER_START - timedelta(days=1)}
        ({(BETA_BLOCKER_START - TREATMENT_START).days} days).
        Post-period: {BETA_BLOCKER_START} to {data_end} ({bb_post_days} days).
        </p>{run_b_table}{run_b_charts}""",
        section_id="run-b",
    ))

    # --- Combined timeline charts ---
    combined_charts = ""
    for metric, meta in METRIC_DEFS.items():
        fig = _build_comparison_chart(results, metric, meta)
        chart_key = f"combined_{metric}"
        chart_data_dict[chart_key] = fig.to_json()
        combined_charts += (
            f'<div id="chart-{chart_key}" class="chart-box" '
            f'data-chart="{chart_key}" style="height:380px;">Loading...</div>'
        )

    body_parts.append(make_section(
        "Combined Timeline View",
        f"""<p style="color:{TEXT_SECONDARY};margin-bottom:0.5em;">
        Full timeline with both intervention points and counterfactual predictions
        overlaid. <span style="color:{COLOR_RUN_A};">Blue dashed</span> = Jakavi counterfactual,
        <span style="color:{COLOR_RUN_B};">purple dashed</span> = BB counterfactual.
        </p>{combined_charts}""",
        section_id="combined",
    ))

    # --- Interpretation section ---
    interp_rows = []
    for metric, meta in METRIC_DEFS.items():
        res_a = run_a_streams.get(metric, {})
        res_b = run_b_streams.get(metric, {})

        a_str = _interpret_single(res_a, meta, "Jakavi")
        b_str = _interpret_single(res_b, meta, "Beta-blocker")

        interp_rows.append(
            f'<tr>'
            f'<td style="font-weight:600;padding:8px 12px;">{meta["label"]}</td>'
            f'<td style="padding:8px 12px;">{a_str}</td>'
            f'<td style="padding:8px 12px;">{b_str}</td>'
            f'</tr>'
        )

    interp_table = f"""
    <table style="width:100%;border-collapse:collapse;margin:1em 0;">
    <thead>
    <tr style="border-bottom:2px solid {BORDER_DEFAULT};">
        <th style="text-align:left;padding:8px 12px;">Metric</th>
        <th style="text-align:left;padding:8px 12px;">Jakavi Contribution</th>
        <th style="text-align:left;padding:8px 12px;">Beta-Blocker Contribution</th>
    </tr>
    </thead>
    <tbody>
    {"".join(interp_rows)}
    </tbody>
    </table>
    """

    body_parts.append(make_section(
        "Clinical Interpretation",
        f"""<p style="color:{TEXT_SECONDARY};margin-bottom:0.5em;">
        Summary of each drug's isolated contribution to observed biometric changes.
        This sequential design helps attribute changes to the correct intervention,
        though N-of-1 limitations still apply.
        </p>{interp_table}
        <p style="color:{TEXT_TERTIARY};font-size:0.85em;margin-top:1em;">
        <strong>Caution:</strong> Run B has a shorter pre-period (Jakavi-only window)
        which may reduce statistical power. Effects flagged as non-significant may
        become significant with more post-BB data accumulation.
        The HEV co-infection is a confounder that cannot be disentangled in this design.
        </p>""",
        section_id="interpretation",
    ))

    body_html = "\n".join(body_parts)

    return wrap_html(
        title="Sequential CausalImpact Analysis",
        body_content=body_html,
        report_id="causal",
        subtitle="Isolating Jakavi vs. Beta-Blocker Causal Effects",
        chart_data=chart_data_dict,
        data_end=data_end,
    )


def _interpret_single(result: dict, meta: dict, drug_name: str) -> str:
    """Generate a plain-English interpretation for one run/metric."""
    if "error" in result:
        return f'<span style="color:{ACCENT_AMBER};">Could not estimate ({result["error"]})</span>'

    effect = result.get("avg_effect", 0)
    rel = result.get("relative_effect_pct", 0)
    sig = result.get("significant", False)
    fav = result.get("favorable", False)
    p = result.get("p_value", 1.0)

    direction = "increased" if effect > 0 else "decreased"
    abs_effect = abs(effect)

    if not sig:
        return (
            f'<span style="color:{TEXT_TERTIARY};">No significant effect detected. '
            f'{drug_name} {direction} {meta["label"].lower()} by '
            f'{abs_effect:.1f} {meta["unit"]} ({rel:+.1f}%), '
            f'but {format_p_value(p)} (not significant).</span>'
        )

    color = ACCENT_GREEN if fav else ACCENT_RED
    verdict = "favorable" if fav else "unfavorable"
    return (
        f'<span style="color:{color};">{drug_name} significantly '
        f'{direction} {meta["label"].lower()} by '
        f'{abs_effect:.1f} {meta["unit"]} ({rel:+.1f}%), '
        f'{format_p_value(p)} - <strong>{verdict}</strong>.</span>'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    print("=" * 72)
    print("SEQUENTIAL CAUSALIMPACT: Jakavi vs. Beta-Blocker Decomposition")
    print("=" * 72)

    if not CAUSALIMPACT_AVAILABLE:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "dependency_unavailable",
            "missing": "pycausalimpact or tfcausalimpact",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        JSON_OUTPUT.write_text(json.dumps(payload, indent=2))
        body = make_section(
            "Optional dependency unavailable",
            (
                "<p>Sequential CausalImpact requires <code>pycausalimpact</code> "
                "or <code>tfcausalimpact</code>, which is part of the optional "
                "full-stack environment. Core report regeneration continues "
                "without this analysis.</p>"
            ),
            section_id="dependency",
        )
        html = wrap_html(
            title="Sequential CausalImpact Analysis",
            body_content=body,
            report_id="sequential_ci",
        )
        HTML_OUTPUT.write_text(html)
        print("\nWARN: CausalImpact package not available, wrote dependency placeholder")
        print(f"[OUTPUT] JSON metrics: {JSON_OUTPUT}")
        print(f"[OUTPUT] HTML report: {HTML_OUTPUT}")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    daily = load_daily_metrics()
    results = run_sequential_analysis(daily)

    # Save JSON metrics
    json_safe = json.loads(json.dumps(results, default=str))
    JSON_OUTPUT.write_text(json.dumps(json_safe, indent=2, default=str))
    print(f"\n[OUTPUT] JSON metrics: {JSON_OUTPUT}")

    # Build and save HTML report
    html = build_html_report(results, daily)
    HTML_OUTPUT.write_text(html)
    print(f"[OUTPUT] HTML report: {HTML_OUTPUT}")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for run_key, run_label in [("run_a", RUN_A_LABEL), ("run_b", RUN_B_LABEL)]:
        print(f"\n  {run_label}:")
        for metric, meta in METRIC_DEFS.items():
            res = results[run_key]["streams"].get(metric, {})
            if "error" in res:
                print(f"    {meta['label']}: ERROR - {res['error']}")
            else:
                effect = res.get("avg_effect", 0)
                rel = res.get("relative_effect_pct", 0)
                p = res.get("p_value", 1.0)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                fav = "favorable" if res.get("favorable") else "unfavorable"
                print(f"    {meta['label']}: {effect:+.2f} {meta['unit']} "
                      f"({rel:+.1f}%), {format_p_value(p)}{sig} [{fav}]")

    print(f"\n  Runtime: {results['runtime_s']:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
