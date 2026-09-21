#!/usr/bin/env python3
"""
Tau-U Effect Size Analysis for Single-Case Experimental Design (SCED)

Computes Tau-U (Parker, Vannest, Davis & Sauber, 2011) for phase comparisons
in a post-HSCT patient monitoring context:
  - Phase A (pre-treatment): before 2026-03-16
  - Phase B (Jakavi-only): 2026-03-16 to 2026-04-07
  - Phase C (Jakavi + BB): 2026-04-08 to present

Tau-U combines nonoverlap between phases with baseline trend correction.
Complementary NAP (Nonoverlap of All Pairs) is also computed.

Metrics: mean_rmssd, lowest_heart_rate, average_heart_rate, sleep_efficiency

Output:
  - HTML report: reports/tau_u_effects.html
  - JSON metrics: reports/tau_u_metrics.json

Usage:
    python analysis/analyze_tau_u.py
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution & config imports
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
    disclaimer_banner,
    format_p_value,
    BG_PRIMARY,
    BG_SURFACE,
    BG_ELEVATED,
    BORDER_SUBTLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_PURPLE,
    ACCENT_CYAN,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
HTML_OUTPUT = REPORTS_DIR / "tau_u_effects.html"
JSON_OUTPUT = REPORTS_DIR / "tau_u_metrics.json"

# ---------------------------------------------------------------------------
# Phase boundaries
# ---------------------------------------------------------------------------
PHASE_A_END = TREATMENT_START  # exclusive: Phase A is [DATA_START, TREATMENT_START)
PHASE_B_START = TREATMENT_START
PHASE_B_END = BETA_BLOCKER_START  # exclusive: Phase B is [TREATMENT_START, BETA_BLOCKER_START)
PHASE_C_START = BETA_BLOCKER_START

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
METRICS = {
    "mean_rmssd": {
        "label": "Mean RMSSD",
        "unit": "ms",
        "higher_is_better": True,
        "description": "Nightly mean root mean square of successive RR differences (vagal tone marker)",
    },
    "lowest_heart_rate": {
        "label": "Lowest Heart Rate",
        "unit": "bpm",
        "higher_is_better": False,
        "description": "Lowest heart rate during sleep (lower = better cardiac recovery)",
    },
    "average_heart_rate": {
        "label": "Average Heart Rate",
        "unit": "bpm",
        "higher_is_better": False,
        "description": "Average heart rate during sleep (lower = reduced sympathetic drive)",
    },
    "sleep_efficiency": {
        "label": "Sleep Efficiency",
        "unit": "%",
        "higher_is_better": True,
        "description": "Percentage of time in bed spent sleeping",
    },
}

# Phase comparison definitions
COMPARISONS = {
    "A_vs_B": {
        "label": "A vs B (Jakavi Effect)",
        "description": "Pre-treatment vs Jakavi-only phase",
        "phase_a": "A",
        "phase_b": "B",
    },
    "B_vs_C": {
        "label": "B vs C (BB Marginal Effect)",
        "description": "Jakavi-only vs Jakavi + beta-blocker",
        "phase_a": "B",
        "phase_b": "C",
    },
    "A_vs_BC": {
        "label": "A vs B+C (Combined Post)",
        "description": "Pre-treatment vs all post-treatment",
        "phase_a": "A",
        "phase_b": "BC",
    },
}

# Effect size interpretation thresholds (Parker et al. 2011)
EFFECT_THRESHOLDS = [
    (0.80, "very large"),
    (0.60, "large"),
    (0.20, "moderate"),
    (0.00, "weak"),
]


def classify_effect(tau: float) -> str:
    """Classify absolute Tau-U value into effect size label."""
    abs_tau = abs(tau)
    for threshold, label in EFFECT_THRESHOLDS:
        if abs_tau >= threshold:
            return label
    return "weak"


def effect_status(tau: float, p_value: float, higher_is_better: bool) -> str:
    """Map effect size to theme status color."""
    if p_value > 0.05:
        return "neutral"
    favorable = (tau > 0) == higher_is_better
    abs_tau = abs(tau)
    if abs_tau >= 0.60 and favorable:
        return "good"
    if abs_tau >= 0.20 and favorable:
        return "info"
    if abs_tau >= 0.60 and not favorable:
        return "critical"
    if abs_tau >= 0.20 and not favorable:
        return "warning"
    return "neutral"


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_daily_metrics() -> pd.DataFrame:
    """Load and aggregate daily metrics from the Oura database.

    Returns a DataFrame with columns: date, mean_rmssd, lowest_heart_rate,
    average_heart_rate, sleep_efficiency.
    """
    print("[DATA] Loading biometric data from database...")

    if not DATABASE_PATH.exists():
        print(
            f"ERROR: Database not found at {DATABASE_PATH}. "
            "Run: python api/import_oura.py --days 90",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)

    # HRV epochs -> daily mean RMSSD
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

    # Sleep periods (long_sleep only)
    # A date can carry several long_sleep periods (a split night); keep the
    # longest so every date appears exactly once, matching the convention in
    # analyze_oura_causal.py. Two such dates exist in the current data
    # (2026-05-19, 2026-06-01) and without this they are counted twice.
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
    daily = daily[daily["date"] >= str(DATA_START)].reset_index(drop=True)

    print(f"  Daily metrics: {len(daily)} days, {daily['date'].min()} to {daily['date'].max()}")
    for col in METRICS:
        valid = daily[col].notna().sum()
        print(f"    {col}: {valid} valid observations")

    return daily


def assign_phases(daily: pd.DataFrame) -> pd.DataFrame:
    """Assign phase labels (A, B, C) based on date boundaries."""
    df = daily.copy()

    def _phase(d: str) -> str:
        dt = date.fromisoformat(d)
        if dt < PHASE_A_END:
            return "A"
        if dt < PHASE_B_END:
            return "B"
        return "C"

    df["phase"] = df["date"].apply(_phase)

    for phase in ["A", "B", "C"]:
        n = (df["phase"] == phase).sum()
        print(f"  Phase {phase}: {n} days")

    return df


# ===========================================================================
# TAU-U COMPUTATION (Parker et al. 2011)
# ===========================================================================

def kendall_s(values: np.ndarray) -> int:
    """Compute Kendall's S statistic within a single series.

    S = sum over all i<j of sign(x_j - x_i)
    """
    n = len(values)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = values[j] - values[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    return s


def kendall_tau_within(values: np.ndarray) -> tuple[float, float]:
    """Compute Kendall's tau and p-value for trend within a phase.

    Returns (tau, p_value). Uses the exact formula:
    tau = S / (n*(n-1)/2), var(S) = n*(n-1)*(2n+5)/18
    """
    n = len(values)
    if n < 3:
        return 0.0, 1.0

    s = kendall_s(values)
    n_pairs = n * (n - 1) // 2
    tau = s / n_pairs if n_pairs > 0 else 0.0

    # Variance under null (no ties adjustment for simplicity)
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if var_s <= 0:
        return tau, 1.0

    z = s / math.sqrt(var_s)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))

    return tau, p


def between_phase_s(phase_a_vals: np.ndarray, phase_b_vals: np.ndarray) -> int:
    """Compute Kendall's S between two phases.

    For each pair (a_i, b_j): +1 if b_j > a_i, -1 if b_j < a_i, 0 if tied.
    """
    s = 0
    for a_val in phase_a_vals:
        for b_val in phase_b_vals:
            diff = b_val - a_val
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    return s


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_tau_u(
    phase_a_vals: np.ndarray,
    phase_b_vals: np.ndarray,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Compute Tau-U with optional baseline trend correction.

    For metrics where decrease is favorable (higher_is_better=False),
    we negate the values so that improvement always maps to positive Tau.

    Steps:
    1. Compute uncorrected Tau-U = S_AB / (n_a * n_b)
    2. Test baseline trend (Kendall's tau within Phase A)
    3. If trend is significant (p < 0.05), compute corrected Tau-U
    4. Compute Z, p-value, 95% CI
    """
    # Direction adjustment: flip so positive Tau = improvement
    if not higher_is_better:
        a_vals = -phase_a_vals
        b_vals = -phase_b_vals
    else:
        a_vals = phase_a_vals
        b_vals = phase_b_vals

    n_a = len(a_vals)
    n_b = len(b_vals)

    if n_a < 3 or n_b < 3:
        return _insufficient_data_result(n_a, n_b)

    # Step 1: Uncorrected Tau-U
    s_ab = between_phase_s(a_vals, b_vals)
    n_pairs_ab = n_a * n_b
    tau_uncorrected = s_ab / n_pairs_ab

    # Step 2: Baseline trend test
    baseline_tau, baseline_p = kendall_tau_within(a_vals)
    baseline_trend_significant = baseline_p < 0.05

    # Step 3: Corrected Tau-U (if baseline trend is significant)
    if baseline_trend_significant:
        s_a = kendall_s(a_vals)
        s_corrected = s_ab - s_a
        n_pairs_corrected = n_pairs_ab + n_a * (n_a - 1) // 2
        tau = s_corrected / n_pairs_corrected if n_pairs_corrected > 0 else 0.0
        s_for_z = s_corrected
        corrected = True
    else:
        tau = tau_uncorrected
        s_for_z = s_ab
        corrected = False

    # Step 4: Variance, Z-score, p-value
    # Variance of S under null: n_a * n_b * (n_a + n_b + 1) / 3
    var_s = n_a * n_b * (n_a + n_b + 1) / 3
    se_s = math.sqrt(var_s) if var_s > 0 else 1.0
    z = s_for_z / se_s
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))

    # 95% CI for Tau
    var_tau = var_s / (n_pairs_ab ** 2)
    se_tau = math.sqrt(var_tau)
    ci_lower = tau - 1.96 * se_tau
    ci_upper = tau + 1.96 * se_tau

    # Clamp to [-1, 1]
    tau = max(-1.0, min(1.0, tau))
    ci_lower = max(-1.0, min(1.0, ci_lower))
    ci_upper = max(-1.0, min(1.0, ci_upper))

    return {
        "tau": tau,
        "tau_uncorrected": tau_uncorrected,
        "s_ab": int(s_ab),
        "n_a": n_a,
        "n_b": n_b,
        "n_pairs": n_pairs_ab,
        "z": z,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "baseline_tau": baseline_tau,
        "baseline_p": baseline_p,
        "baseline_trend_significant": baseline_trend_significant,
        "corrected": corrected,
        "effect_size": classify_effect(tau),
        "significant": p_value < 0.05,
    }


def _insufficient_data_result(n_a: int, n_b: int) -> dict[str, Any]:
    """Return a placeholder result when phase data is insufficient."""
    return {
        "tau": float("nan"),
        "tau_uncorrected": float("nan"),
        "s_ab": 0,
        "n_a": n_a,
        "n_b": n_b,
        "n_pairs": n_a * n_b,
        "z": float("nan"),
        "p_value": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "baseline_tau": float("nan"),
        "baseline_p": float("nan"),
        "baseline_trend_significant": False,
        "corrected": False,
        "effect_size": "insufficient data",
        "significant": False,
        "error": f"Insufficient data: n_a={n_a}, n_b={n_b} (need >= 3 each)",
    }


# ===========================================================================
# NAP (Nonoverlap of All Pairs) - Mann-Whitney U based
# ===========================================================================

def compute_nap(
    phase_a_vals: np.ndarray,
    phase_b_vals: np.ndarray,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Compute NAP (Nonoverlap of All Pairs).

    NAP = U / (n_a * n_b) where U counts pairs where b_j > a_i
    (plus 0.5 * ties). Equivalent to the common language effect size.
    """
    if not higher_is_better:
        a_vals = -phase_a_vals
        b_vals = -phase_b_vals
    else:
        a_vals = phase_a_vals
        b_vals = phase_b_vals

    n_a = len(a_vals)
    n_b = len(b_vals)

    if n_a < 1 or n_b < 1:
        return {"nap": float("nan"), "n_a": n_a, "n_b": n_b}

    u = 0.0
    for a_val in a_vals:
        for b_val in b_vals:
            if b_val > a_val:
                u += 1.0
            elif b_val == a_val:
                u += 0.5

    n_pairs = n_a * n_b
    nap = u / n_pairs if n_pairs > 0 else 0.0

    return {"nap": nap, "u": u, "n_a": n_a, "n_b": n_b, "n_pairs": n_pairs}


# ===========================================================================
# ANALYSIS ORCHESTRATION
# ===========================================================================

def get_phase_values(
    df: pd.DataFrame, metric: str, comparison: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract phase A and phase B values for a given comparison."""
    phase_a_label = comparison["phase_a"]
    phase_b_label = comparison["phase_b"]

    if phase_a_label == "A":
        mask_a = df["phase"] == "A"
    elif phase_a_label == "B":
        mask_a = df["phase"] == "B"
    else:
        mask_a = df["phase"] == "A"

    if phase_b_label == "B":
        mask_b = df["phase"] == "B"
    elif phase_b_label == "C":
        mask_b = df["phase"] == "C"
    elif phase_b_label == "BC":
        mask_b = df["phase"].isin(["B", "C"])
    else:
        mask_b = df["phase"] == "B"

    a_vals = df.loc[mask_a, metric].dropna().values.astype(float)
    b_vals = df.loc[mask_b, metric].dropna().values.astype(float)

    return a_vals, b_vals


def run_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Run Tau-U and NAP for all metric x comparison combinations."""
    print("\n[ANALYSIS] Computing Tau-U effect sizes...")

    results: dict[str, Any] = {
        "generated": datetime.now().isoformat(),
        "phases": {
            "A": {"label": "Pre-treatment", "end": str(PHASE_A_END)},
            "B": {"label": "Jakavi-only", "start": str(PHASE_B_START), "end": str(PHASE_B_END)},
            "C": {"label": "Jakavi + BB", "start": str(PHASE_C_START)},
        },
        "comparisons": {},
    }

    for comp_key, comp_def in COMPARISONS.items():
        results["comparisons"][comp_key] = {
            "label": comp_def["label"],
            "description": comp_def["description"],
            "metrics": {},
        }

        for metric_key, metric_def in METRICS.items():
            a_vals, b_vals = get_phase_values(df, metric_key, comp_def)

            # Phase descriptive stats
            a_mean = float(np.nanmean(a_vals)) if len(a_vals) > 0 else float("nan")
            b_mean = float(np.nanmean(b_vals)) if len(b_vals) > 0 else float("nan")
            a_sd = float(np.nanstd(a_vals, ddof=1)) if len(a_vals) > 1 else float("nan")
            b_sd = float(np.nanstd(b_vals, ddof=1)) if len(b_vals) > 1 else float("nan")

            # Tau-U
            tau_u = compute_tau_u(a_vals, b_vals, metric_def["higher_is_better"])

            # NAP
            nap = compute_nap(a_vals, b_vals, metric_def["higher_is_better"])

            result_entry = {
                "label": metric_def["label"],
                "unit": metric_def["unit"],
                "higher_is_better": metric_def["higher_is_better"],
                "phase_a_mean": _safe_float(a_mean),
                "phase_a_sd": _safe_float(a_sd),
                "phase_b_mean": _safe_float(b_mean),
                "phase_b_sd": _safe_float(b_sd),
                "n_a": tau_u["n_a"],
                "n_b": tau_u["n_b"],
                "tau_u": tau_u,
                "nap": nap,
            }

            results["comparisons"][comp_key]["metrics"][metric_key] = result_entry

            # Console output
            tau_val = tau_u["tau"]
            p_val = tau_u["p_value"]
            sig = "*" if tau_u["significant"] else ""
            corr = " (corrected)" if tau_u["corrected"] else ""
            tau_str = f"{tau_val:+.3f}" if math.isfinite(tau_val) else "N/A"
            p_str = format_p_value(p_val) if math.isfinite(p_val) else "N/A"
            print(
                f"  {comp_def['label']:30s} | {metric_def['label']:20s} | "
                f"Tau={tau_str}{sig}{corr} | {p_str} | {tau_u['effect_size']}"
            )

    return results


def _safe_float(v: float) -> float | None:
    """Convert float to JSON-safe value (None for NaN/inf)."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ===========================================================================
# HTML REPORT GENERATION
# ===========================================================================

def _tau_badge(tau: float, p_value: float) -> str:
    """Small colored badge showing Tau value and significance."""
    if not math.isfinite(tau):
        return '<span style="color:#6B7280">N/A</span>'

    abs_tau = abs(tau)
    if p_value > 0.05:
        color = TEXT_TERTIARY
    elif abs_tau >= 0.60:
        color = ACCENT_GREEN if tau > 0 else ACCENT_RED
    elif abs_tau >= 0.20:
        color = ACCENT_CYAN if tau > 0 else ACCENT_AMBER
    else:
        color = TEXT_SECONDARY

    sig = " *" if p_value < 0.05 else ""
    return f'<span style="color:{color};font-weight:600">{tau:+.3f}{sig}</span>'


def _nap_str(nap_val: float) -> str:
    """Format NAP value."""
    if not math.isfinite(nap_val):
        return "N/A"
    return f"{nap_val:.1%}"


def _direction_arrow(tau: float, higher_is_better: bool) -> str:
    """Arrow indicating direction of change relative to favorable direction."""
    if not math.isfinite(tau) or abs(tau) < 0.01:
        return "&mdash;"
    favorable = (tau > 0) == higher_is_better
    if tau > 0:
        arrow = "&#x25B2;"  # up triangle
        color = ACCENT_GREEN if favorable else ACCENT_RED
    else:
        arrow = "&#x25BC;"  # down triangle
        color = ACCENT_GREEN if favorable else ACCENT_RED
    return f'<span style="color:{color}">{arrow}</span>'


def build_kpi_section(results: dict[str, Any]) -> str:
    """Build KPI cards for the most important comparisons (A vs B+C)."""
    combined = results["comparisons"]["A_vs_BC"]["metrics"]

    cards = []
    for metric_key, metric_def in METRICS.items():
        entry = combined[metric_key]
        tau_u = entry["tau_u"]
        tau = tau_u["tau"]
        p_val = tau_u["p_value"]

        if not math.isfinite(tau):
            cards.append(make_kpi_card(
                metric_def["label"],
                "N/A",
                status="neutral",
                detail="Insufficient data",
            ))
            continue

        status = effect_status(tau, p_val, metric_def["higher_is_better"])
        sig_str = format_p_value(p_val)
        effect_label = tau_u["effect_size"].capitalize()
        corr_note = " (trend-corrected)" if tau_u["corrected"] else ""

        cards.append(make_kpi_card(
            f"TAU-U: {metric_def['label']}",
            tau,
            decimals=3,
            status=status,
            detail=f"{effect_label} effect | {sig_str}{corr_note}",
            status_label=effect_label,
            explainer=f"A vs B+C | NAP={_nap_str(entry['nap']['nap'])}",
        ))

    return make_kpi_row(*cards)


def build_results_table(results: dict[str, Any]) -> str:
    """Build the full results table with all comparisons and metrics."""
    rows = []

    for comp_key, comp_data in results["comparisons"].items():
        for metric_key, entry in comp_data["metrics"].items():
            tau_u = entry["tau_u"]
            nap = entry["nap"]
            tau = tau_u["tau"]
            p_val = tau_u["p_value"]

            # Phase means
            a_mean = entry["phase_a_mean"]
            b_mean = entry["phase_b_mean"]
            a_str = f"{a_mean:.1f}" if a_mean is not None else "N/A"
            b_str = f"{b_mean:.1f}" if b_mean is not None else "N/A"

            # Direction
            direction = _direction_arrow(tau, entry["higher_is_better"])

            # CI
            ci_str = "N/A"
            if math.isfinite(tau_u.get("ci_lower", float("nan"))) and math.isfinite(tau_u.get("ci_upper", float("nan"))):
                ci_str = f"[{tau_u['ci_lower']:+.3f}, {tau_u['ci_upper']:+.3f}]"

            # Baseline info
            bl_str = ""
            if tau_u.get("corrected"):
                bl_str = f'<span style="color:{ACCENT_AMBER};font-size:0.75rem" title="Baseline trend tau={tau_u["baseline_tau"]:.3f}, {format_p_value(tau_u["baseline_p"])}">&#9888; corrected</span>'

            rows.append(
                f"<tr>"
                f'<td style="white-space:nowrap">{comp_data["label"]}</td>'
                f"<td>{entry['label']}</td>"
                f"<td>{entry['n_a']}</td>"
                f"<td>{entry['n_b']}</td>"
                f'<td style="text-align:right">{a_str}</td>'
                f'<td style="text-align:right">{b_str}</td>'
                f"<td>{direction}</td>"
                f"<td>{_tau_badge(tau, p_val)}</td>"
                f'<td style="font-size:0.85rem">{ci_str}</td>'
                f"<td>{format_p_value(p_val)}</td>"
                f"<td>{tau_u['effect_size']}</td>"
                f"<td>{_nap_str(nap['nap'])}</td>"
                f"<td>{bl_str}</td>"
                f"</tr>"
            )

    table = f"""
    <div style="overflow-x:auto">
    <table class="tau-table">
      <thead>
        <tr>
          <th>Comparison</th>
          <th>Metric</th>
          <th>n<sub>A</sub></th>
          <th>n<sub>B</sub></th>
          <th>Mean<sub>A</sub></th>
          <th>Mean<sub>B</sub></th>
          <th>Dir</th>
          <th>Tau-U</th>
          <th>95% CI</th>
          <th>p-value</th>
          <th>Effect</th>
          <th>NAP</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    </div>
    """
    return table


def build_interpretation_notes() -> str:
    """Methodology and interpretation guide."""
    return f"""
    <div class="tau-notes">
      <h3>Methodology</h3>
      <p><b>Tau-U</b> (Parker, Vannest, Davis &amp; Sauber, 2011) is the recommended
      effect size for single-case experimental designs (SCED). It quantifies the
      degree of nonoverlap between phases while optionally correcting for baseline trend.</p>

      <h3>Interpretation Scale</h3>
      <div class="tau-scale">
        <div class="tau-scale-item">
          <span class="tau-scale-bar" style="background:{TEXT_TERTIARY};width:20%"></span>
          <span>|Tau| &lt; 0.20 = <b>Weak</b></span>
        </div>
        <div class="tau-scale-item">
          <span class="tau-scale-bar" style="background:{ACCENT_CYAN};width:40%"></span>
          <span>0.20 &ndash; 0.60 = <b>Moderate</b></span>
        </div>
        <div class="tau-scale-item">
          <span class="tau-scale-bar" style="background:{ACCENT_BLUE};width:60%"></span>
          <span>0.60 &ndash; 0.80 = <b>Large</b></span>
        </div>
        <div class="tau-scale-item">
          <span class="tau-scale-bar" style="background:{ACCENT_GREEN};width:80%"></span>
          <span>&gt; 0.80 = <b>Very large</b></span>
        </div>
      </div>

      <h3>Key Details</h3>
      <ul>
        <li><b>Positive Tau</b> = improvement (direction-adjusted per metric)</li>
        <li><b>Baseline correction:</b> If Phase A shows a significant trend
        (Kendall tau, p&lt;0.05), Tau-U is corrected by subtracting the baseline S statistic.
        A &#9888; symbol marks corrected values.</li>
        <li><b>NAP</b> (Nonoverlap of All Pairs) is a complementary measure:
        probability that a random Phase B observation exceeds a random Phase A observation
        (after direction adjustment). NAP &gt; 0.93 = large, 0.66&ndash;0.92 = medium,
        &lt; 0.66 = weak (Parker &amp; Vannest, 2009).</li>
        <li>* = statistically significant at p &lt; 0.05</li>
      </ul>

      <h3>Phases</h3>
      <ul>
        <li><b>Phase A</b> (Pre-treatment): {DATA_START.strftime("%b %d")} &ndash; {(PHASE_A_END - timedelta(days=1)).strftime("%b %d, %Y")}</li>
        <li><b>Phase B</b> (Jakavi only): {PHASE_B_START.strftime("%b %d")} &ndash; {(PHASE_B_END - timedelta(days=1)).strftime("%b %d, %Y")}</li>
        <li><b>Phase C</b> (Jakavi + Beta-blocker): {PHASE_C_START.strftime("%b %d, %Y")} &ndash; present</li>
      </ul>
    </div>
    """


def build_phase_summary(df: pd.DataFrame) -> str:
    """Build a descriptive statistics summary per phase."""
    rows = []
    for phase in ["A", "B", "C"]:
        phase_data = df[df["phase"] == phase]
        n = len(phase_data)
        for metric_key, metric_def in METRICS.items():
            vals = phase_data[metric_key].dropna()
            if len(vals) == 0:
                rows.append(
                    f"<tr><td>{phase}</td><td>{metric_def['label']}</td>"
                    f"<td>0</td><td colspan='4'>No data</td></tr>"
                )
                continue
            rows.append(
                f"<tr>"
                f"<td>{phase}</td>"
                f"<td>{metric_def['label']}</td>"
                f"<td>{len(vals)}</td>"
                f"<td>{vals.mean():.1f}</td>"
                f"<td>{vals.std(ddof=1):.1f}</td>"
                f"<td>{vals.min():.1f}</td>"
                f"<td>{vals.max():.1f}</td>"
                f"</tr>"
            )

    return f"""
    <div style="overflow-x:auto">
    <table class="tau-table">
      <thead>
        <tr>
          <th>Phase</th>
          <th>Metric</th>
          <th>n</th>
          <th>Mean</th>
          <th>SD</th>
          <th>Min</th>
          <th>Max</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    </div>
    """


EXTRA_CSS = f"""
.tau-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
  margin: 1rem 0;
}}
.tau-table th {{
  background: {BG_ELEVATED};
  color: {TEXT_PRIMARY};
  padding: 0.6rem 0.75rem;
  text-align: left;
  border-bottom: 2px solid {BORDER_SUBTLE};
  font-weight: 600;
  white-space: nowrap;
}}
.tau-table td {{
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid {BORDER_SUBTLE};
  color: {TEXT_PRIMARY};
}}
.tau-table tr:hover {{
  background: rgba(59, 130, 246, 0.05);
}}
.tau-notes {{
  color: {TEXT_SECONDARY};
  font-size: 0.875rem;
  line-height: 1.6;
}}
.tau-notes h3 {{
  color: {TEXT_PRIMARY};
  font-size: 1rem;
  margin: 1.2rem 0 0.4rem;
}}
.tau-notes ul {{
  padding-left: 1.5rem;
}}
.tau-notes li {{
  margin-bottom: 0.3rem;
}}
.tau-scale {{
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  margin: 0.5rem 0;
}}
.tau-scale-item {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
}}
.tau-scale-bar {{
  height: 8px;
  border-radius: 4px;
  min-width: 40px;
}}
"""


def generate_report(df: pd.DataFrame, results: dict[str, Any]) -> str:
    """Assemble the complete HTML report."""
    body = ""

    # KPI cards for combined A vs B+C
    body += make_section(
        "Effect Size Summary (A vs B+C)",
        build_kpi_section(results),
        section_id="summary",
    )

    # Full results table
    body += make_section(
        "Tau-U Results: All Comparisons",
        build_results_table(results),
        section_id="results",
    )

    # Phase descriptive statistics
    body += make_section(
        "Phase Descriptive Statistics",
        build_phase_summary(df),
        section_id="descriptive",
    )

    # Interpretation notes
    body += make_section(
        "Methodology & Interpretation",
        build_interpretation_notes(),
        section_id="methodology",
    )

    latest_date = df["date"].max()

    html = wrap_html(
        "Tau-U Effect Size Analysis",
        body,
        report_id="tau_u",
        subtitle="Single-Case Experimental Design (SCED) Phase Comparisons",
        extra_css=EXTRA_CSS,
        data_end=latest_date,
    )
    return html


# ===========================================================================
# JSON EXPORT
# ===========================================================================

def make_json_safe(obj: Any) -> Any:
    """Recursively convert NaN/inf to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    return obj


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """Run the full Tau-U analysis pipeline."""
    print("=" * 70)
    print("Tau-U Effect Size Analysis (Parker et al. 2011)")
    print("=" * 70)

    # Load data
    daily = load_daily_metrics()
    daily = assign_phases(daily)

    # Run analysis
    results = run_analysis(daily)

    # Generate HTML report
    print("\n[REPORT] Generating HTML report...")
    html = generate_report(daily, results)
    HTML_OUTPUT.write_text(html, encoding="utf-8")
    print(f"  Saved: {HTML_OUTPUT}")

    # Export JSON
    print("[REPORT] Exporting JSON metrics...")
    json_data = make_json_safe(results)
    JSON_OUTPUT.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"  Saved: {JSON_OUTPUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()
