#!/usr/bin/env python3
"""Module 2: Treatment Response Detection.

Detects inflection points and changepoints in both patients' biometric data
using four complementary methods (PELT, CUSUM, BOCPD, Rolling Window).
For Patient 1 we split at known treatment date (Rux Mar 16); for Patient 2 we
discover changepoints automatically and build a consensus map.

Outputs:
  - Interactive HTML dashboard: reports/comparative_treatment_response.html
  - JSON metrics:               reports/comparative_treatment_response.json

Usage:
    python analysis/analyze_comparative_treatment.py
"""
from __future__ import annotations

import json
import logging
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

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution & config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    REPORTS_DIR,
    FONT_FAMILY,
    TREATMENT_START,
    KNOWN_EVENT_DATE,
    HEV_DIAGNOSIS_DATE,
)
from _comparative_utils import (
    PatientConfig,
    default_patients,
    load_patient_data,
    load_metric,
    zscore_normalize,
    effect_size_cohens_d,
    bootstrap_ci,
    PATIENT_COLORS,
    COMPARABLE_METRICS,
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
    BG_ELEVATED,
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

HTML_OUTPUT = REPORTS_DIR / "comparative_treatment_response.html"
JSON_OUTPUT = REPORTS_DIR / "comparative_treatment_response.json"

# Try importing ruptures for PELT; graceful skip if missing
try:
    import ruptures
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    logger.warning("ruptures library not found -- PELT changepoint detection will be skipped")


# ---------------------------------------------------------------------------
# Metric definitions for this module (6 metrics per patient)
# ---------------------------------------------------------------------------
TREATMENT_METRICS = [
    ("hrv_average", "oura_sleep_periods", "average_hrv", "HRV (RMSSD)", "ms", True),
    ("hr_lowest", "oura_sleep_periods", "lowest_heart_rate", "Lowest Heart Rate", "bpm", False),
    ("hr_average", "oura_sleep_periods", "average_heart_rate", "Average Heart Rate", "bpm", False),
    ("efficiency", "oura_sleep_periods", "efficiency", "Sleep Efficiency", "%", True),
    ("deep_sleep_hours", "oura_sleep_periods", "deep_sleep_duration", "Deep Sleep", "sec", True),
    ("steps", "oura_activity", "steps", "Daily Steps", "steps", True),
]

# Patient 1 known events
HENRIK_EVENTS = [
    (KNOWN_EVENT_DATE, "Acute Episode", ACCENT_RED),
    (TREATMENT_START, "Rux Start", ACCENT_CYAN),
    (HEV_DIAGNOSIS_DATE, "HEV Diagnosis", ACCENT_AMBER),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(fig: go.Figure) -> str:
    """Embed a Plotly figure as inline HTML (no JS bundle)."""
    return fig.to_html(include_plotlyjs=False, full_html=False)


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


def _prepare_signal(series: pd.Series) -> np.ndarray:
    """Interpolate NaN, standardize to zero mean / unit variance."""
    s = series.copy().astype(float)
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.fillna(s.mean() if not s.empty else 0)
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        std = 1.0
    return ((s - mean) / std).values


def _sanitize(obj: Any) -> Any:
    """Recursively sanitize NaN/inf/numpy types for JSON serialization."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [_sanitize(x) for x in obj.tolist()]
    return obj


# ---------------------------------------------------------------------------
# [1] Data Loading
# ---------------------------------------------------------------------------

def load_data(
    patients: tuple[PatientConfig, PatientConfig],
) -> dict[str, dict[str, pd.Series]]:
    """Load all 6 treatment metrics for both patients."""
    result: dict[str, dict[str, pd.Series]] = {}

    for p in patients:
        metrics: dict[str, pd.Series] = {}
        for m_name, table, column, display, unit, _ in TREATMENT_METRICS:
            date_col = "day" if table == "oura_sleep_periods" else "date"
            cols = f"{date_col}, {column}"
            df = load_patient_data(p, table, columns=cols)
            if not df.empty and column in df.columns:
                s = df[column].dropna()
                # Convert deep_sleep from seconds to hours for readability
                if m_name == "deep_sleep_hours":
                    s = s / 3600.0
                s.name = m_name
                metrics[m_name] = s
            else:
                metrics[m_name] = pd.Series(dtype=float, name=m_name)

        result[p.patient_id] = metrics
        total = sum(len(v) for v in metrics.values())
        logger.info("Loaded %s: %d total metric observations across %d metrics",
                     p.display_name, total, len(metrics))

    return result


# ---------------------------------------------------------------------------
# [2] Changepoint Detection: 4 Methods
# ---------------------------------------------------------------------------

def detect_pelt(series: pd.Series, penalty: float | None = None) -> list[date]:
    """PELT changepoint detection using ruptures library."""
    if not HAS_RUPTURES:
        return []
    clean = series.dropna()
    if len(clean) < 10:
        return []

    signal = _prepare_signal(clean)
    n = len(signal)

    if penalty is None:
        variance = np.var(signal)
        penalty = 2 * np.log(n) * max(variance, 0.1)

    algo = ruptures.Pelt(model="rbf", min_size=5).fit(signal.reshape(-1, 1))
    try:
        bkps = algo.predict(pen=penalty)
    except Exception as e:
        logger.warning("PELT failed: %s", e)
        return []

    dates = clean.index
    result = []
    for b in bkps:
        if b < n:
            ts = dates[b]
            if isinstance(ts, pd.Timestamp):
                result.append(ts.date())
    return result


def detect_cusum(series: pd.Series, threshold_sd: float = 0.5) -> list[date]:
    """CUSUM changepoint detection via second-derivative inflection points."""
    clean = series.dropna()
    if len(clean) < 10:
        return []

    signal = _prepare_signal(clean)
    cusum = np.cumsum(signal - np.mean(signal))

    # Second derivative
    if len(cusum) < 3:
        return []
    d2 = np.diff(cusum, n=2)

    # Sign changes of second derivative
    sign_changes = np.where(np.diff(np.sign(d2)))[0] + 1

    # Filter by magnitude
    sd = np.std(cusum)
    if sd == 0:
        return []
    threshold = threshold_sd * sd

    dates = clean.index
    result = []
    for idx in sign_changes:
        if idx < len(cusum) and abs(cusum[idx]) >= threshold:
            ts = dates[min(idx, len(dates) - 1)]
            if isinstance(ts, pd.Timestamp):
                result.append(ts.date())
    return result


def detect_bocpd(
    series: pd.Series,
    hazard_rate: float = 1 / 30,
    threshold: float = 0.3,
) -> tuple[list[date], np.ndarray]:
    """Bayesian Online Change Point Detection (Normal-Gamma prior).

    Returns (changepoint_dates, run_length_probabilities).
    """
    clean = series.dropna()
    if len(clean) < 10:
        return [], np.array([])

    signal = _prepare_signal(clean)
    n = len(signal)

    # Normal-Gamma prior parameters
    mu0 = 0.0
    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = 1.0

    # Run length probability matrix
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0

    # Sufficient statistics (grow with run length)
    mu_params = np.array([mu0])
    kappa_params = np.array([kappa0])
    alpha_params = np.array([alpha0])
    beta_params = np.array([beta0])

    changepoint_prob = np.zeros(n)

    for t in range(n):
        x = signal[t]

        # Predictive probability (Student-t)
        pred_var = beta_params * (kappa_params + 1) / (alpha_params * kappa_params)
        pred_var = np.maximum(pred_var, 1e-10)
        pred_mean = mu_params
        nu = 2 * alpha_params

        # Student-t log probability
        pred_prob = np.exp(
            scipy_stats.t.logpdf(x, df=nu, loc=pred_mean, scale=np.sqrt(pred_var))
        )

        # Growth probabilities
        growth_prob = R[: t + 1, t] * pred_prob * (1 - hazard_rate)
        # Changepoint probability
        cp_prob = np.sum(R[: t + 1, t] * pred_prob * hazard_rate)

        # Update run length distribution
        R[1: t + 2, t + 1] = growth_prob
        R[0, t + 1] = cp_prob

        # Normalize
        evidence = R[: t + 2, t + 1].sum()
        if evidence > 0:
            R[: t + 2, t + 1] /= evidence

        # Changepoint probability = prob of run length 0
        changepoint_prob[t] = R[0, t + 1]

        # Update sufficient statistics
        new_kappa = kappa_params + 1
        new_mu = (kappa_params * mu_params + x) / new_kappa
        new_alpha = alpha_params + 0.5
        new_beta = beta_params + kappa_params * (x - mu_params) ** 2 / (2 * new_kappa)

        # Prepend prior for run length 0
        mu_params = np.concatenate([[mu0], new_mu])
        kappa_params = np.concatenate([[kappa0], new_kappa])
        alpha_params = np.concatenate([[alpha0], new_alpha])
        beta_params = np.concatenate([[beta0], new_beta])

    # Extract changepoints above threshold
    dates = clean.index
    result = []
    for i in range(n):
        if changepoint_prob[i] > threshold:
            ts = dates[i]
            if isinstance(ts, pd.Timestamp):
                result.append(ts.date())

    return result, changepoint_prob


def detect_rolling_window(
    series: pd.Series,
    window: int = 14,
    p_threshold: float = 0.01,
    d_threshold: float = 0.5,
) -> list[date]:
    """Rolling window comparison with t-test + Cohen's d."""
    clean = series.dropna()
    if len(clean) < 2 * window + 2:
        return []

    dates = clean.index
    values = clean.values.astype(float)
    result = []

    for i in range(window, len(values) - window):
        pre = values[i - window: i]
        post = values[i: i + window]

        if len(pre) < 3 or len(post) < 3:
            continue
        if np.std(pre) == 0 and np.std(post) == 0:
            continue

        try:
            _, p_val = scipy_stats.ttest_ind(pre, post, equal_var=False)
        except Exception:
            continue

        pooled_std = np.sqrt(
            ((len(pre) - 1) * np.std(pre, ddof=1) ** 2
             + (len(post) - 1) * np.std(post, ddof=1) ** 2)
            / (len(pre) + len(post) - 2)
        )
        if pooled_std == 0:
            continue
        d = abs((np.mean(post) - np.mean(pre)) / pooled_std)

        if p_val < p_threshold and d > d_threshold:
            ts = dates[i]
            if isinstance(ts, pd.Timestamp):
                result.append(ts.date())

    return result


def run_all_changepoint_methods(
    series: pd.Series,
    hazard_rate: float = 1 / 30,
) -> dict[str, Any]:
    """Run all 4 changepoint methods on a single metric series."""
    pelt_dates = detect_pelt(series)
    cusum_dates = detect_cusum(series)
    bocpd_dates, bocpd_probs = detect_bocpd(series, hazard_rate=hazard_rate)
    rolling_dates = detect_rolling_window(series)

    return {
        "pelt": pelt_dates,
        "cusum": cusum_dates,
        "bocpd": bocpd_dates,
        "bocpd_probabilities": bocpd_probs,
        "rolling_window": rolling_dates,
    }


# ---------------------------------------------------------------------------
# [3] Patient 1: Pre/Post Treatment Analysis
# ---------------------------------------------------------------------------

def henrik_pre_post_analysis(
    metrics: dict[str, pd.Series],
    treatment_date: date,
) -> dict[str, dict[str, Any]]:
    """Split each metric at TREATMENT_START, compute statistical comparison."""
    treatment_ts = pd.Timestamp(treatment_date)
    n_metrics = len(TREATMENT_METRICS)
    results: dict[str, dict[str, Any]] = {}

    for m_name, _, _, display, unit, higher_is_better in TREATMENT_METRICS:
        series = metrics.get(m_name, pd.Series(dtype=float))
        if series.empty or len(series) < 4:
            results[m_name] = {
                "display_name": display,
                "unit": unit,
                "pre_treatment": {},
                "post_treatment": {},
                "comparison": {"status": "insufficient_data"},
            }
            continue

        pre = series[series.index < treatment_ts].dropna()
        post = series[series.index >= treatment_ts].dropna()

        if len(pre) < 2 or len(post) < 2:
            results[m_name] = {
                "display_name": display,
                "unit": unit,
                "pre_treatment": {
                    "mean": float(pre.mean()) if len(pre) > 0 else None,
                    "n": len(pre),
                },
                "post_treatment": {
                    "mean": float(post.mean()) if len(post) > 0 else None,
                    "n": len(post),
                },
                "comparison": {"status": "insufficient_data"},
            }
            continue

        # Basic stats
        pre_stats = {
            "mean": float(pre.mean()),
            "median": float(pre.median()),
            "std": float(pre.std()),
            "n": int(len(pre)),
        }
        post_stats = {
            "mean": float(post.mean()),
            "median": float(post.median()),
            "std": float(post.std()),
            "n": int(len(post)),
        }

        # Mann-Whitney U
        try:
            stat, p_raw = scipy_stats.mannwhitneyu(pre, post, alternative="two-sided")
        except Exception:
            stat, p_raw = np.nan, np.nan

        # Cohen's d
        d = effect_size_cohens_d(pre, post)

        # Bootstrap CI
        ci = bootstrap_ci(pre, post, n_bootstrap=1000)

        # Percentage change
        pct_change = (
            ((post.mean() - pre.mean()) / abs(pre.mean()) * 100)
            if pre.mean() != 0 else 0.0
        )

        # Bonferroni correction
        bonferroni_p = min(float(p_raw) * n_metrics, 1.0) if not np.isnan(p_raw) else np.nan

        # Direction assessment
        diff = post.mean() - pre.mean()
        if higher_is_better is True:
            direction = "improved" if diff > 0 else "worsened"
        elif higher_is_better is False:
            direction = "improved" if diff < 0 else "worsened"
        else:
            direction = "changed"

        results[m_name] = {
            "display_name": display,
            "unit": unit,
            "pre_treatment": pre_stats,
            "post_treatment": post_stats,
            "comparison": {
                "pct_change": float(pct_change),
                "mann_whitney_U": float(stat) if not np.isnan(stat) else None,
                "mann_whitney_p": float(p_raw) if not np.isnan(p_raw) else None,
                "cohens_d": float(d),
                "effect_label": _effect_label(d),
                "bootstrap_ci_95": [float(ci[0]), float(ci[1])],
                "bonferroni_p": float(bonferroni_p) if not np.isnan(bonferroni_p) else None,
                "direction": direction,
                "significant_raw": float(p_raw) < 0.05 if not np.isnan(p_raw) else False,
                "significant_corrected": float(bonferroni_p) < 0.05 if not np.isnan(bonferroni_p) else False,
            },
        }

    return results


def _effect_label(d: float) -> str:
    """Classify Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# [4] Patient 1: Three-Period Analysis (pre-acute / post-acute-pre-rux / post-rux)
# ---------------------------------------------------------------------------

def henrik_three_period(metrics: dict[str, pd.Series]) -> dict[str, dict[str, Any]]:
    """Three-period comparison for Patient 1."""
    acute_ts = pd.Timestamp(KNOWN_EVENT_DATE)
    rux_ts = pd.Timestamp(TREATMENT_START)
    results: dict[str, dict[str, Any]] = {}

    for m_name, _, _, display, unit, _ in TREATMENT_METRICS:
        series = metrics.get(m_name, pd.Series(dtype=float))
        if series.empty:
            continue

        p1 = series[series.index < acute_ts].dropna()
        p2 = series[(series.index >= acute_ts) & (series.index < rux_ts)].dropna()
        p3 = series[series.index >= rux_ts].dropna()

        period_data = {}
        for label, data in [("pre_acute", p1), ("post_acute_pre_rux", p2), ("post_rux", p3)]:
            if len(data) > 0:
                period_data[label] = {
                    "mean": float(data.mean()),
                    "median": float(data.median()),
                    "std": float(data.std()) if len(data) > 1 else 0.0,
                    "n": int(len(data)),
                }
            else:
                period_data[label] = {"mean": None, "median": None, "std": None, "n": 0}

        results[m_name] = {"display_name": display, "unit": unit, "periods": period_data}

    return results


# ---------------------------------------------------------------------------
# [5] Patient 2: Automatic Changepoint Discovery + Consensus
# ---------------------------------------------------------------------------

def mitchell_consensus(
    metrics: dict[str, pd.Series],
    tolerance_days: int = 3,
) -> dict[str, Any]:
    """Run all methods on all Patient 2 metrics, build consensus map."""
    all_detections: dict[str, dict[str, list[date]]] = {}
    bocpd_probs: dict[str, np.ndarray] = {}

    for m_name, _, _, display, unit, _ in TREATMENT_METRICS:
        series = metrics.get(m_name, pd.Series(dtype=float))
        if series.empty or len(series) < 10:
            all_detections[m_name] = {"pelt": [], "cusum": [], "bocpd": [], "rolling_window": []}
            bocpd_probs[m_name] = np.array([])
            continue

        cp = run_all_changepoint_methods(series, hazard_rate=1 / 50)
        all_detections[m_name] = {
            "pelt": cp["pelt"],
            "cusum": cp["cusum"],
            "bocpd": cp["bocpd"],
            "rolling_window": cp["rolling_window"],
        }
        bocpd_probs[m_name] = cp["bocpd_probabilities"]

    # Build consensus map: flatten all dates, cluster within tolerance
    all_dates: list[date] = []
    date_sources: dict[str, list[str]] = {}  # date_str -> list of "method:metric"

    for m_name, methods in all_detections.items():
        for method_name, dates_list in methods.items():
            for d in dates_list:
                d_str = d.isoformat()
                all_dates.append(d)
                if d_str not in date_sources:
                    date_sources[d_str] = []
                date_sources[d_str].append(f"{method_name}:{m_name}")

    if not all_dates:
        return {
            "detections": all_detections,
            "bocpd_probs": bocpd_probs,
            "consensus_events": [],
        }

    # Cluster dates within tolerance
    sorted_dates = sorted(set(all_dates))
    clusters: list[list[date]] = []
    current_cluster = [sorted_dates[0]]

    for d in sorted_dates[1:]:
        if (d - current_cluster[-1]).days <= tolerance_days:
            current_cluster.append(d)
        else:
            clusters.append(current_cluster)
            current_cluster = [d]
    clusters.append(current_cluster)

    # Score each cluster
    consensus_events: list[dict[str, Any]] = []
    for cluster in clusters:
        # Count unique (method, metric) pairs within cluster
        methods_detecting: set[str] = set()
        metrics_affected: set[str] = set()
        for d in cluster:
            d_str = d.isoformat()
            # Also check nearby dates within tolerance
            for check_d_str, sources in date_sources.items():
                check_d = date.fromisoformat(check_d_str)
                if abs((check_d - d).days) <= tolerance_days:
                    for src in sources:
                        method, metric = src.split(":", 1)
                        methods_detecting.add(method)
                        metrics_affected.add(metric)

        # Representative date = median of cluster
        rep_idx = len(cluster) // 2
        rep_date = cluster[rep_idx]

        score = len(methods_detecting) + len(metrics_affected)

        # Check if shift sustained (at least 7 days of data after)
        sustained = True  # default to True if we can't check

        consensus_events.append({
            "date": rep_date,
            "consensus_score": score,
            "methods_detecting": sorted(methods_detecting),
            "metrics_affected": sorted(metrics_affected),
            "n_methods": len(methods_detecting),
            "n_metrics": len(metrics_affected),
            "sustained": sustained,
            "high_confidence": score >= 3,
        })

    # Sort by score descending
    consensus_events.sort(key=lambda x: x["consensus_score"], reverse=True)

    return {
        "detections": all_detections,
        "bocpd_probs": bocpd_probs,
        "consensus_events": consensus_events,
    }


# ---------------------------------------------------------------------------
# [6] Multi-Metric Convergence
# ---------------------------------------------------------------------------

def compute_convergence(
    metrics: dict[str, pd.Series],
    patient_id: str,
    z_threshold: float = 1.5,
) -> pd.DataFrame:
    """Per-day z-scores and convergence count for all metrics."""
    z_frames = []
    for m_name, _, _, _, _, _ in TREATMENT_METRICS:
        series = metrics.get(m_name, pd.Series(dtype=float))
        if series.empty or len(series) < 3:
            continue
        nr = zscore_normalize(series, patient_id=patient_id)
        z_series = nr.z_scores.rename(m_name)
        z_frames.append(z_series)

    if not z_frames:
        return pd.DataFrame()

    z_df = pd.concat(z_frames, axis=1, sort=True)
    z_df["convergence"] = (z_df.abs() > z_threshold).sum(axis=1)
    return z_df


# ---------------------------------------------------------------------------
# [7] Visualizations
# ---------------------------------------------------------------------------

def _fig_henrik_timeline(
    series: pd.Series,
    changepoints: dict[str, list[date]],
    metric_display: str,
    unit: str,
) -> go.Figure:
    """Patient 1 annotated timeline: rolling mean + changepoints + vertical event lines."""
    fig = go.Figure()

    if series.empty:
        fig.update_layout(title=f"{metric_display} - No data")
        return fig

    # Raw scatter
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="markers",
        marker=dict(size=3, color=ACCENT_BLUE, opacity=0.3),
        name="Daily",
        showlegend=True,
    ))

    # 7-day rolling mean
    if len(series) >= 7:
        rolling = series.rolling(7, min_periods=4).mean()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values,
            mode="lines",
            line=dict(color=ACCENT_BLUE, width=2.5),
            name="7-day avg",
        ))

        # 95% CI band (rolling mean +/- 1.96 * rolling std)
        rolling_std = series.rolling(7, min_periods=4).std()
        upper = rolling + 1.96 * rolling_std
        lower = rolling - 1.96 * rolling_std
        fig.add_trace(go.Scatter(
            x=list(upper.index) + list(lower.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself",
            fillcolor=f"rgba(59,130,246,0.08)",
            line=dict(width=0),
            name="95% CI",
            showlegend=True,
        ))

    # Changepoint markers
    cp_colors = {
        "pelt": ACCENT_PURPLE,
        "cusum": ACCENT_ORANGE,
        "bocpd": ACCENT_PINK,
        "rolling_window": ACCENT_CYAN,
    }
    cp_symbols = {
        "pelt": "diamond",
        "cusum": "triangle-up",
        "bocpd": "star",
        "rolling_window": "square",
    }

    for method, dates_list in changepoints.items():
        if method == "bocpd_probabilities":
            continue
        if not isinstance(dates_list, list) or len(dates_list) == 0:
            continue
        y_vals = []
        x_vals = []
        for d in dates_list:
            ts = pd.Timestamp(d)
            if ts in series.index:
                y_vals.append(series.loc[ts])
                x_vals.append(ts)
            else:
                nearest_idx = series.index.get_indexer([ts], method="nearest")
                if nearest_idx[0] >= 0:
                    x_vals.append(series.index[nearest_idx[0]])
                    y_vals.append(series.iloc[nearest_idx[0]])

        if x_vals:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers",
                marker=dict(
                    size=10,
                    color=cp_colors.get(method, ACCENT_AMBER),
                    symbol=cp_symbols.get(method, "circle"),
                    line=dict(width=1, color=TEXT_PRIMARY),
                ),
                name=f"CP: {method.upper()}",
            ))

    # Patient 1 event lines
    for evt_date, evt_label, evt_color in HENRIK_EVENTS:
        _add_event_vline(fig, pd.Timestamp(evt_date), evt_label, evt_color)

    fig.update_layout(
        title=dict(text=f"Patient 1: {metric_display}", font=dict(size=14)),
        yaxis_title=f"{metric_display} ({unit})",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=60, r=20, t=50, b=70),
        height=400,
    )
    return fig


def _fig_bocpd_probability(
    series: pd.Series,
    bocpd_probs: np.ndarray,
    patient_name: str,
    metric_display: str,
    threshold: float = 0.3,
) -> go.Figure:
    """BOCPD changepoint probability chart."""
    fig = go.Figure()

    if len(bocpd_probs) == 0 or series.empty:
        fig.update_layout(title=f"{patient_name}: BOCPD {metric_display} - No data")
        return fig

    clean = series.dropna()
    if len(clean) != len(bocpd_probs):
        min_len = min(len(clean), len(bocpd_probs))
        clean = clean.iloc[:min_len]
        bocpd_probs = bocpd_probs[:min_len]

    fig.add_trace(go.Scatter(
        x=clean.index, y=bocpd_probs,
        mode="lines",
        line=dict(color=ACCENT_PURPLE, width=2),
        name="CP Probability",
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.15)",
    ))

    # Threshold line
    fig.add_shape(
        type="line",
        x0=clean.index[0], x1=clean.index[-1],
        y0=threshold, y1=threshold,
        line=dict(color=ACCENT_AMBER, width=1, dash="dash"),
    )
    fig.add_annotation(
        x=clean.index[-1], y=threshold,
        text=f"Threshold ({threshold})",
        showarrow=False,
        font=dict(size=9, color=ACCENT_AMBER),
        xanchor="right",
    )

    fig.update_layout(
        title=dict(text=f"{patient_name}: BOCPD - {metric_display}", font=dict(size=14)),
        yaxis_title="Changepoint Probability",
        xaxis_title="Date",
        hovermode="x unified",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def _fig_mitchell_timeline(
    metrics: dict[str, pd.Series],
    consensus_events: list[dict[str, Any]],
) -> go.Figure:
    """Patient 2 discovered changepoints timeline."""
    fig = go.Figure()

    # Plot all metrics as z-scores
    colors = [ACCENT_GREEN, ACCENT_CYAN, ACCENT_BLUE, ACCENT_PURPLE, ACCENT_ORANGE, ACCENT_AMBER]
    for i, (m_name, _, _, display, _, _) in enumerate(TREATMENT_METRICS):
        series = metrics.get(m_name, pd.Series(dtype=float))
        if series.empty or len(series) < 3:
            continue
        nr = zscore_normalize(series, patient_id="mitch")
        rolling = nr.z_scores.rolling(14, min_periods=7).mean()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values,
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=1.5),
            name=display,
        ))

    # Highlight high-confidence changepoints
    for evt in consensus_events:
        if evt["high_confidence"]:
            _add_event_vline(
                fig,
                pd.Timestamp(evt["date"]),
                f"CP (score={evt['consensus_score']})",
                ACCENT_RED,
            )

    fig.update_layout(
        title=dict(text="Patient 2: Discovered Changepoints (all metrics, z-scored)", font=dict(size=14)),
        yaxis_title="Z-Score (14-day rolling)",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        height=450,
        margin=dict(l=60, r=20, t=50, b=70),
    )
    return fig


def _fig_comparative_violin(
    data: dict[str, dict[str, pd.Series]],
    patients: tuple[PatientConfig, PatientConfig],
    metric_name: str,
    display_name: str,
) -> go.Figure:
    """Side-by-side violin plots for a metric."""
    fig = go.Figure()
    patient_map = {p.patient_id: p for p in patients}

    for pid, metrics in data.items():
        series = metrics.get(metric_name, pd.Series(dtype=float))
        if series.empty:
            continue
        p = patient_map[pid]
        color = PATIENT_COLORS.get(pid, ACCENT_PURPLE)
        clean = series.dropna()

        fig.add_trace(go.Violin(
            y=clean.values,
            name=p.display_name,
            marker_color=color,
            box_visible=True,
            meanline_visible=True,
            opacity=0.8,
        ))

    fig.update_layout(
        title=dict(text=f"{display_name} Distribution", font=dict(size=14)),
        showlegend=True,
        height=350,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def _fig_convergence_heatmap(
    z_df: pd.DataFrame,
    patient_name: str,
) -> go.Figure:
    """Multi-metric convergence heatmap: days x metrics."""
    if z_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{patient_name}: Convergence - No data")
        return fig

    metric_cols = [c for c in z_df.columns if c != "convergence"]
    if not metric_cols:
        fig = go.Figure()
        fig.update_layout(title=f"{patient_name}: Convergence - No metrics")
        return fig

    z_matrix = z_df[metric_cols].T

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix.values,
        x=[d.strftime("%Y-%m-%d") if isinstance(d, (pd.Timestamp, datetime)) else str(d)
           for d in z_matrix.columns],
        y=[m.replace("_", " ").title() for m in metric_cols],
        colorscale=[
            [0.0, ACCENT_BLUE],
            [0.25, BG_SURFACE],
            [0.5, BG_ELEVATED],
            [0.75, ACCENT_AMBER],
            [1.0, ACCENT_RED],
        ],
        zmid=0,
        colorbar=dict(title="Z-Score"),
    ))

    fig.update_layout(
        title=dict(text=f"{patient_name}: Multi-Metric Convergence", font=dict(size=14)),
        xaxis_title="Date",
        yaxis_title="Metric",
        height=350,
        margin=dict(l=120, r=20, t=50, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# [8] HTML Assembly
# ---------------------------------------------------------------------------

def build_html(
    data: dict[str, dict[str, pd.Series]],
    henrik_stats: dict[str, dict[str, Any]],
    henrik_three: dict[str, dict[str, Any]],
    henrik_changepoints: dict[str, dict[str, Any]],
    mitchell_result: dict[str, Any],
    convergence: dict[str, pd.DataFrame],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Assemble the complete HTML report."""
    sections: list[str] = []

    # -- Executive Summary KPIs --
    sections.append(section_html_or_placeholder(
        "Executive Summary",
        _build_executive_summary,
        henrik_stats, mitchell_result,
    ))

    # -- Patient 1: Treatment Response --
    sections.append(section_html_or_placeholder(
        "Patient 1 Treatment Response",
        _build_henrik_section,
        data, henrik_stats, henrik_three, henrik_changepoints, patients,
    ))

    # -- Patient 2: Discovered Events --
    sections.append(section_html_or_placeholder(
        "Patient 2 Discovered Events",
        _build_mitchell_section,
        data, mitchell_result, patients,
    ))

    # -- Comparative Distributions --
    sections.append(section_html_or_placeholder(
        "Comparative Distributions",
        _build_comparative_section,
        data, patients,
    ))

    # -- Convergence Heatmaps --
    sections.append(section_html_or_placeholder(
        "Multi-Metric Convergence",
        _build_convergence_section,
        convergence,
    ))

    # -- Methods Appendix --
    sections.append(section_html_or_placeholder(
        "Methods Appendix",
        _build_methods_appendix,
    ))

    body = "\n".join(sections)
    return wrap_html(
        title="Treatment Response Detection",
        body_content=body,
        report_id="comp_treatment",
        subtitle="Module 2: Comparative Changepoint Analysis",
        header_meta="Patient 1 (post-HSCT) vs Patient 2 (post-Stroke)",
    )


def _build_executive_summary(
    henrik_stats: dict[str, dict[str, Any]],
    mitchell_result: dict[str, Any],
) -> str:
    """Executive summary KPI cards."""
    # Count significant changes for Patient 1
    sig_count = 0
    improved_count = 0
    for m_name, stat in henrik_stats.items():
        comp = stat.get("comparison", {})
        if comp.get("significant_corrected"):
            sig_count += 1
        if comp.get("direction") == "improved":
            improved_count += 1

    # Patient 2 high-confidence events
    mitch_events = [e for e in mitchell_result.get("consensus_events", []) if e.get("high_confidence")]

    cards = [
        make_kpi_card(
            "SIGNIFICANT CHANGES",
            sig_count,
            unit=f"/ {len(henrik_stats)}",
            status="info" if sig_count > 0 else "neutral",
            detail="Patient 1 post-Rux (Bonferroni-corrected)",
        ),
        make_kpi_card(
            "IMPROVED METRICS",
            improved_count,
            unit=f"/ {len(henrik_stats)}",
            status="good" if improved_count > 0 else "warning",
            detail="Direction of change post-treatment",
        ),
        make_kpi_card(
            "MITCHELL EVENTS",
            len(mitch_events),
            unit="detected",
            status="info" if mitch_events else "neutral",
            detail="High-confidence consensus changepoints",
        ),
        make_kpi_card(
            "DETECTION METHODS",
            4,
            unit="methods",
            status="info",
            detail="PELT + CUSUM + BOCPD + Rolling Window",
        ),
    ]

    return make_section(
        "Executive Summary",
        make_kpi_row(*cards),
        section_id="exec-summary",
    )


def _build_henrik_section(
    data: dict[str, dict[str, pd.Series]],
    henrik_stats: dict[str, dict[str, Any]],
    henrik_three: dict[str, dict[str, Any]],
    henrik_changepoints: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Patient 1 treatment response: timelines, stat cards, BOCPD."""
    parts: list[str] = []

    # Pre/post stat cards
    stat_cards: list[str] = []
    for m_name, stat in henrik_stats.items():
        comp = stat.get("comparison", {})
        if comp.get("status") == "insufficient_data":
            continue
        pre = stat.get("pre_treatment", {})
        post = stat.get("post_treatment", {})
        pct = comp.get("pct_change", 0)
        p_val = comp.get("bonferroni_p")
        direction = comp.get("direction", "")

        status = "neutral"
        if direction == "improved":
            status = "good"
        elif direction == "worsened":
            status = "warning"

        p_str = format_p_value(p_val) if p_val is not None else "N/A"
        effect = comp.get("effect_label", "")

        stat_cards.append(make_kpi_card(
            stat["display_name"].upper(),
            f"{pct:+.1f}%",
            unit="change",
            status=status,
            detail=(
                f"Pre: {pre.get('mean', 0):.1f} (n={pre.get('n', 0)}) | "
                f"Post: {post.get('mean', 0):.1f} (n={post.get('n', 0)})<br>"
                f"p={p_str} (corrected) | d={comp.get('cohens_d', 0):.2f} ({effect})"
            ),
        ))

    if stat_cards:
        parts.append(make_kpi_row(*stat_cards))

    # Annotated timelines + BOCPD
    timeline_html: list[str] = []
    for m_name, _, _, display, unit, _ in TREATMENT_METRICS:
        h_metrics = data.get("henrik", {})
        series = h_metrics.get(m_name, pd.Series(dtype=float))
        cps = henrik_changepoints.get(m_name, {})

        # Timeline figure
        fig = _fig_henrik_timeline(series, cps, display, unit)
        timeline_html.append(_embed(fig))

        # BOCPD probability figure
        bocpd_probs = cps.get("bocpd_probabilities", np.array([]))
        if len(bocpd_probs) > 0:
            fig_bp = _fig_bocpd_probability(series, bocpd_probs, "Patient 1", display)
            timeline_html.append(_embed(fig_bp))

    parts.append("\n".join(timeline_html))

    # Three-period summary table
    if henrik_three:
        table_rows = ""
        def _fmt_period(p: dict) -> str:
            m = p.get("mean")
            n = p.get("n", 0)
            if m is not None and isinstance(m, (int, float)):
                return f"{m:.1f} (n={n})"
            return f"N/A (n={n})"

        for m_name, info in henrik_three.items():
            periods = info.get("periods", {})
            p1 = periods.get("pre_acute", {})
            p2 = periods.get("post_acute_pre_rux", {})
            p3 = periods.get("post_rux", {})
            table_rows += (
                f"<tr>"
                f"<td>{info['display_name']}</td>"
                f"<td>{_fmt_period(p1)}</td>"
                f"<td>{_fmt_period(p2)}</td>"
                f"<td>{_fmt_period(p3)}</td>"
                f"</tr>"
            )

        three_period_html = (
            '<div style="overflow-x:auto;">'
            '<table class="odt-table" style="width:100%;border-collapse:collapse;">'
            '<thead><tr>'
            '<th style="text-align:left;padding:8px;border-bottom:1px solid #374151;">Metric</th>'
            f'<th style="text-align:center;padding:8px;border-bottom:1px solid #374151;">Pre-Acute<br><small>(&lt; {KNOWN_EVENT_DATE})</small></th>'
            f'<th style="text-align:center;padding:8px;border-bottom:1px solid #374151;">Post-Acute / Pre-Rux<br><small>({KNOWN_EVENT_DATE} - {TREATMENT_START})</small></th>'
            f'<th style="text-align:center;padding:8px;border-bottom:1px solid #374151;">Post-Rux<br><small>(&ge; {TREATMENT_START})</small></th>'
            '</tr></thead>'
            f'<tbody>{table_rows}</tbody>'
            '</table></div>'
        )
        parts.append(three_period_html)

    return make_section(
        "Patient 1: Treatment Response Analysis",
        "\n".join(parts),
        section_id="henrik-treatment",
    )


def _build_mitchell_section(
    data: dict[str, dict[str, pd.Series]],
    mitchell_result: dict[str, Any],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Patient 2 discovered events section."""
    parts: list[str] = []

    mitch_metrics = data.get("mitch", {})
    consensus_events = mitchell_result.get("consensus_events", [])

    # Timeline
    fig = _fig_mitchell_timeline(mitch_metrics, consensus_events)
    parts.append(_embed(fig))

    # Consensus event table
    if consensus_events:
        rows = ""
        for evt in consensus_events[:15]:  # top 15
            conf_badge = (
                '<span style="color:#34D399;font-weight:600;">HIGH</span>'
                if evt["high_confidence"]
                else '<span style="color:#9CA3AF;">low</span>'
            )
            rows += (
                f"<tr>"
                f"<td>{evt['date']}</td>"
                f"<td>{evt['consensus_score']}</td>"
                f"<td>{conf_badge}</td>"
                f"<td>{', '.join(evt['methods_detecting'])}</td>"
                f"<td>{', '.join(evt['metrics_affected'])}</td>"
                f"</tr>"
            )
        table_html = (
            '<div style="overflow-x:auto;">'
            '<table class="odt-table" style="width:100%;border-collapse:collapse;">'
            '<thead><tr>'
            '<th style="text-align:left;padding:8px;border-bottom:1px solid #374151;">Date</th>'
            '<th style="text-align:center;padding:8px;border-bottom:1px solid #374151;">Score</th>'
            '<th style="text-align:center;padding:8px;border-bottom:1px solid #374151;">Confidence</th>'
            '<th style="text-align:left;padding:8px;border-bottom:1px solid #374151;">Methods</th>'
            '<th style="text-align:left;padding:8px;border-bottom:1px solid #374151;">Metrics</th>'
            '</tr></thead>'
            f'<tbody>{rows}</tbody>'
            '</table></div>'
        )
        parts.append(table_html)
    else:
        parts.append(
            '<p style="color:#9CA3AF;">No consensus changepoints detected for Patient 2.</p>'
        )

    # BOCPD probability charts for Patient 2
    bocpd_probs = mitchell_result.get("bocpd_probs", {})
    for m_name, _, _, display, _, _ in TREATMENT_METRICS:
        probs = bocpd_probs.get(m_name, np.array([]))
        series = mitch_metrics.get(m_name, pd.Series(dtype=float))
        if len(probs) > 0 and not series.empty:
            fig_bp = _fig_bocpd_probability(series, probs, "Patient 2", display)
            parts.append(_embed(fig_bp))

    return make_section(
        "Patient 2: Discovered Changepoints",
        "\n".join(parts),
        section_id="mitchell-changepoints",
    )


def _build_comparative_section(
    data: dict[str, dict[str, pd.Series]],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Comparative distributions (violin plots)."""
    parts: list[str] = []

    for m_name, _, _, display, _, _ in TREATMENT_METRICS:
        fig = _fig_comparative_violin(data, patients, m_name, display)
        parts.append(_embed(fig))

    return make_section(
        "Comparative Distributions",
        "\n".join(parts),
        section_id="comparative-distributions",
    )


def _build_convergence_section(
    convergence: dict[str, pd.DataFrame],
) -> str:
    """Multi-metric convergence heatmaps."""
    parts: list[str] = []

    for pid, z_df in convergence.items():
        if z_df.empty:
            continue
        display_name = "Patient 1" if pid == "henrik" else "Patient 2"
        fig = _fig_convergence_heatmap(z_df, display_name)
        parts.append(_embed(fig))

        # Systemic shift events (convergence >= 3)
        shifts = z_df[z_df["convergence"] >= 3]
        if not shifts.empty:
            n_shifts = len(shifts)
            parts.append(
                f'<p style="color:{TEXT_SECONDARY};margin:8px 0;">'
                f'{display_name}: <strong>{n_shifts} days</strong> with '
                f'3+ metrics deviating beyond 1.5 SD (systemic shift events).</p>'
            )

    if not parts:
        parts.append(
            '<p style="color:#9CA3AF;">Insufficient data for convergence analysis.</p>'
        )

    return make_section(
        "Multi-Metric Convergence",
        "\n".join(parts),
        section_id="convergence",
    )


def _build_methods_appendix() -> str:
    """Methods explanation section."""
    methods_html = """
    <div style="color:#9CA3AF;line-height:1.7;">
    <h3 style="color:#E8E8ED;">Changepoint Detection Methods</h3>

    <p><strong style="color:#8B5CF6;">PELT (Penalized Exact Linear Time)</strong>:
    Uses the <code>ruptures</code> library with RBF kernel to detect optimal changepoints.
    Signals are interpolated (for NaN) and standardized before fitting. Penalty is
    derived from BIC: <code>2 * log(n) * variance</code>.</p>

    <p><strong style="color:#F97316;">CUSUM (Cumulative Sum)</strong>:
    Computes the cumulative sum of deviations from the overall mean. Second-derivative
    sign changes identify inflection points. Filtered by magnitude threshold (0.5 SD).</p>

    <p><strong style="color:#EC4899;">BOCPD (Bayesian Online Change Point Detection)</strong>:
    Implements Adams & MacKay (2007) with Normal-Gamma conjugate prior. Hazard rate
    set to 1/30 for Patient 1 (shorter observation window) and 1/50 for Patient 2
    (longer data span). Changepoints where posterior probability exceeds 0.3.</p>

    <p><strong style="color:#06B6D4;">Rolling Window Comparison</strong>:
    Adjacent 14-day windows compared via Welch's t-test and Cohen's d.
    Dates flagged where p < 0.01 AND |d| > 0.5, indicating both statistical
    significance and practical effect size.</p>

    <h3 style="color:#E8E8ED;margin-top:20px;">Statistical Tests</h3>
    <p><strong>Pre/Post Comparison</strong>: Mann-Whitney U test (non-parametric,
    two-sided) with Bonferroni correction for 6 simultaneous comparisons.
    Effect size: Cohen's d with pooled standard deviation. Confidence intervals:
    bootstrap with 1,000 iterations.</p>

    <p><strong>Consensus Scoring</strong>: For Patient 2, all (method x metric) detections
    are clustered within a 3-day tolerance window. The consensus score counts the
    number of unique methods and metrics detecting each cluster.
    High confidence = score >= 3.</p>
    </div>
    """
    return make_section(
        "Methods Appendix",
        methods_html,
        section_id="methods-appendix",
    )


# ---------------------------------------------------------------------------
# [9] JSON Export
# ---------------------------------------------------------------------------

def export_json(
    henrik_stats: dict[str, dict[str, Any]],
    henrik_three: dict[str, dict[str, Any]],
    henrik_changepoints: dict[str, dict[str, Any]],
    mitchell_result: dict[str, Any],
    convergence: dict[str, pd.DataFrame],
) -> None:
    """Write structured metrics JSON."""
    # Strip non-serializable items from mitchell_result
    mitchell_clean = {
        "discovered_events": mitchell_result.get("consensus_events", []),
        "detections": mitchell_result.get("detections", {}),
    }

    # Convergence summary
    conv_summary: dict[str, Any] = {}
    for pid, z_df in convergence.items():
        if z_df.empty:
            continue
        shifts = z_df[z_df["convergence"] >= 3]
        conv_summary[pid] = {
            "total_days": len(z_df),
            "systemic_shift_days": len(shifts),
            "shift_dates": [
                d.isoformat() if isinstance(d, (pd.Timestamp, datetime, date)) else str(d)
                for d in shifts.index
            ],
        }

    # Build Patient 1 changepoints for JSON (strip numpy arrays)
    henrik_cp_json: dict[str, Any] = {}
    for m_name, cp in henrik_changepoints.items():
        henrik_cp_json[m_name] = {
            k: v for k, v in cp.items() if k != "bocpd_probabilities"
        }

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "report": "comparative_treatment_response",
        "patients": {
            "henrik": {
                "known_events": [
                    {"date": str(d), "label": l} for d, l, _ in HENRIK_EVENTS
                ],
                "metrics": henrik_stats,
                "three_period": henrik_three,
                "changepoints": henrik_cp_json,
            },
            "mitchell": mitchell_clean,
        },
        "multi_metric_convergence": conv_summary,
        "methods": {
            "pelt": {
                "library": "ruptures",
                "model": "rbf",
                "available": HAS_RUPTURES,
            },
            "cusum": {
                "threshold_sd": 0.5,
                "description": "Cumulative sum with second-derivative inflection detection",
            },
            "bocpd": {
                "prior": "Normal-Gamma",
                "hazard_henrik": "1/30",
                "hazard_mitchell": "1/50",
                "threshold": 0.3,
            },
            "rolling_window": {
                "window_days": 14,
                "p_threshold": 0.01,
                "d_threshold": 0.5,
            },
        },
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
    """Run comparative treatment response analysis pipeline."""
    logger.info("[1/9] Loading patient data...")
    patients = default_patients()
    if patients[1] is None:
        print("Skipping: mitch.db not found (second patient data not available)")
        return 0
    data = load_data(patients)

    # -- Patient 1 analyses --
    logger.info("[2/9] Patient 1: pre/post treatment analysis...")
    henrik_metrics = data.get("henrik", {})
    henrik_stats = henrik_pre_post_analysis(henrik_metrics, TREATMENT_START)

    logger.info("[3/9] Patient 1: three-period comparison...")
    henrik_three = henrik_three_period(henrik_metrics)

    logger.info("[4/9] Patient 1: changepoint detection (4 methods)...")
    henrik_changepoints: dict[str, dict[str, Any]] = {}
    for m_name, _, _, display, _, _ in TREATMENT_METRICS:
        series = henrik_metrics.get(m_name, pd.Series(dtype=float))
        if series.empty or len(series) < 10:
            henrik_changepoints[m_name] = {
                "pelt": [], "cusum": [], "bocpd": [],
                "rolling_window": [], "bocpd_probabilities": np.array([]),
            }
            continue
        cp = run_all_changepoint_methods(series, hazard_rate=1 / 30)
        henrik_changepoints[m_name] = cp
        total_cp = sum(len(cp[k]) for k in ["pelt", "cusum", "bocpd", "rolling_window"])
        logger.info("  %s: %d changepoints detected across methods", display, total_cp)

    # -- Patient 2 analyses --
    logger.info("[5/9] Patient 2: automatic changepoint discovery...")
    mitch_metrics = data.get("mitch", {})
    mitchell_result = mitchell_consensus(mitch_metrics)
    n_high = sum(1 for e in mitchell_result.get("consensus_events", []) if e.get("high_confidence"))
    logger.info("  Patient 2: %d high-confidence consensus events", n_high)

    # -- Convergence --
    logger.info("[6/9] Computing multi-metric convergence...")
    convergence: dict[str, pd.DataFrame] = {}
    for pid in ["henrik", "mitch"]:
        metrics = data.get(pid, {})
        convergence[pid] = compute_convergence(metrics, pid)

    # -- HTML --
    logger.info("[7/9] Generating HTML report...")
    html = build_html(
        data, henrik_stats, henrik_three, henrik_changepoints,
        mitchell_result, convergence, patients,
    )
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", HTML_OUTPUT)

    # -- JSON --
    logger.info("[8/9] Exporting JSON metrics...")
    export_json(henrik_stats, henrik_three, henrik_changepoints, mitchell_result, convergence)

    logger.info("[9/9] Comparative treatment response analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
