#!/usr/bin/env python3
"""
Placebo / Falsification Tests for Oura HSCT Digital Twin

Runs Mann-Whitney U tests (and optionally CausalImpact) at 20 random
"fake intervention" dates within the pre-treatment period.  If the model
finds significant effects at these placebo dates more often than the
nominal 5% rate, the real treatment-effect estimates are less credible.

Design:
  - Pre-treatment window: DATA_START to TREATMENT_START - 1 day
  - 20 placebo dates drawn with np.random.default_rng(42), each >= 14 days
    from start, end, and the real intervention date
  - At each placebo date the pre-treatment data is split into "pre" vs "post"
    and Mann-Whitney U is computed for each metric
  - False positive rate = fraction of placebo dates with p < 0.05 per metric
  - Well-calibrated: ~5%; liberal: significantly above 5%; conservative: ~0%

Output:
  - reports/placebo_calibration.html   (dark-theme HTML report)
  - reports/placebo_calibration_metrics.json

Usage:
    python analysis/analyze_placebo_tests.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Path setup and config imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH,
    DATA_START,
    REPORTS_DIR,
    TREATMENT_START,
)

from _theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_ELEVATED,
    BG_PRIMARY,
    BG_SURFACE,
    BORDER_DEFAULT,
    BORDER_SUBTLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    format_p_value,
    make_kpi_card,
    make_kpi_row,
    make_section,
    wrap_html,
)

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "clinical_dark"

# ---------------------------------------------------------------------------
# Optional CausalImpact
# ---------------------------------------------------------------------------
# pandas >= 3 shim required by pycausalimpact
if not hasattr(pd.DataFrame, "applymap") and hasattr(pd.DataFrame, "map"):
    pd.DataFrame.applymap = pd.DataFrame.map  # type: ignore[attr-defined]

try:
    from causalimpact import CausalImpact

    CAUSALIMPACT_AVAILABLE = True
except ImportError:
    CAUSALIMPACT_AVAILABLE = False
    CausalImpact = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HTML_OUTPUT = REPORTS_DIR / "placebo_calibration.html"
JSON_OUTPUT = REPORTS_DIR / "placebo_calibration_metrics.json"

N_PLACEBO = 20
MIN_BUFFER_DAYS = 14
ALPHA = 0.05
RNG_SEED = 42

METRICS = [
    ("mean_rmssd", "HRV (RMSSD)", "ms"),
    ("lowest_heart_rate", "Lowest HR", "bpm"),
    ("average_heart_rate", "Average HR", "bpm"),
    ("sleep_efficiency", "Sleep Efficiency", "%"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_data() -> pd.DataFrame:
    """Load and build a daily feature matrix for the pre-treatment period."""
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

    # Sleep periods -> efficiency, heart rates
    sp = pd.read_sql_query(
        """SELECT day as date, efficiency, average_heart_rate, lowest_heart_rate
           FROM oura_sleep_periods
           WHERE type = 'long_sleep'
           ORDER BY day""",
        conn,
    )
    for col in sp.columns:
        if col != "date":
            sp[col] = pd.to_numeric(sp[col], errors="coerce")
    sp = sp.rename(columns={"efficiency": "sleep_efficiency"})

    conn.close()

    # Merge on date
    daily = hrv_daily.merge(sp, on="date", how="outer").sort_values("date").reset_index(drop=True)

    # Filter to pre-treatment period only
    start_str = str(DATA_START)
    end_str = str(TREATMENT_START - timedelta(days=1))
    daily = daily[(daily["date"] >= start_str) & (daily["date"] <= end_str)].copy()
    daily = daily.reset_index(drop=True)

    print(f"  Pre-treatment daily matrix: {len(daily)} rows, {daily['date'].min()} to {daily['date'].max()}")
    return daily


# ---------------------------------------------------------------------------
# Placebo date generation
# ---------------------------------------------------------------------------

def generate_placebo_dates(n: int, rng_seed: int) -> list[date]:
    """Generate n random placebo dates within the pre-treatment window.

    Each date is at least MIN_BUFFER_DAYS from:
      - DATA_START
      - TREATMENT_START
      - the real intervention date (TREATMENT_START)
    This ensures enough data on both sides of the split.
    """
    rng = np.random.default_rng(rng_seed)

    pre_end = TREATMENT_START - timedelta(days=1)
    earliest = DATA_START + timedelta(days=MIN_BUFFER_DAYS)
    latest = pre_end - timedelta(days=MIN_BUFFER_DAYS)

    if earliest >= latest:
        print(
            f"ERROR: Pre-treatment window too short for placebo tests. "
            f"Need at least {2 * MIN_BUFFER_DAYS + 1} days, "
            f"have {(pre_end - DATA_START).days}.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_candidate_days = (latest - earliest).days + 1
    candidate_ordinals = np.arange(earliest.toordinal(), latest.toordinal() + 1)

    # Draw without replacement
    chosen = rng.choice(candidate_ordinals, size=min(n, total_candidate_days), replace=False)
    chosen.sort()

    placebo_dates = [date.fromordinal(int(o)) for o in chosen]

    print(f"[PLACEBO] Generated {len(placebo_dates)} placebo dates:")
    for d in placebo_dates:
        days_from_start = (d - DATA_START).days
        days_to_treatment = (TREATMENT_START - d).days
        print(f"  {d}  (day {days_from_start} from start, {days_to_treatment} before treatment)")

    return placebo_dates


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_mann_whitney(
    daily: pd.DataFrame,
    placebo_date: date,
    metric_col: str,
) -> dict:
    """Run Mann-Whitney U test splitting data at placebo_date.

    Returns dict with u_stat, p_value, n_pre, n_post, pre_median, post_median.
    """
    split_str = str(placebo_date)
    pre = daily.loc[daily["date"] < split_str, metric_col].dropna()
    post = daily.loc[daily["date"] >= split_str, metric_col].dropna()

    result = {
        "n_pre": len(pre),
        "n_post": len(post),
        "pre_median": float(pre.median()) if len(pre) > 0 else None,
        "post_median": float(post.median()) if len(post) > 0 else None,
    }

    if len(pre) < 3 or len(post) < 3:
        result["u_stat"] = None
        result["p_value"] = None
        result["error"] = f"Insufficient data: n_pre={len(pre)}, n_post={len(post)}"
        return result

    u_stat, p_value = scipy_stats.mannwhitneyu(pre, post, alternative="two-sided")
    result["u_stat"] = float(u_stat)
    result["p_value"] = float(p_value)
    return result


def run_causal_impact_placebo(
    daily: pd.DataFrame,
    placebo_date: date,
    metric_col: str,
) -> dict | None:
    """Run CausalImpact at a placebo split if the library is available.

    Returns dict with p_value and summary, or None if unavailable/failed.
    """
    if not CAUSALIMPACT_AVAILABLE:
        return None

    split_str = str(placebo_date)
    ts = daily[["date", metric_col]].dropna().copy()
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.set_index("date").sort_index()

    pre_start = ts.index.min()
    pre_end = pd.Timestamp(placebo_date) - pd.Timedelta(days=1)
    post_start = pd.Timestamp(placebo_date)
    post_end = ts.index.max()

    if pre_end <= pre_start or post_end < post_start:
        return None

    n_pre = len(ts[pre_start:pre_end])
    n_post = len(ts[post_start:post_end])
    if n_pre < 5 or n_post < 3:
        return None

    try:
        ci = CausalImpact(
            ts,
            [pre_start, pre_end],
            [post_start, post_end],
            prior_level_sd=None,
        )
        p_val = ci.p_value
        return {
            "p_value": float(p_val) if p_val is not None and np.isfinite(p_val) else None,
            "abs_effect": float(ci.summary_data.loc["average", "abs_effect"])
            if hasattr(ci, "summary_data") else None,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_placebo_analysis(daily: pd.DataFrame, placebo_dates: list[date]) -> dict:
    """Run all placebo tests and compute false positive rates.

    Returns a structured results dict.
    """
    metric_cols = [m[0] for m in METRICS]
    results_by_date = []

    for i, pdate in enumerate(placebo_dates):
        print(f"  Placebo {i + 1}/{len(placebo_dates)}: {pdate}")
        row = {"placebo_date": str(pdate), "metrics": {}}

        for col in metric_cols:
            mw = run_mann_whitney(daily, pdate, col)
            ci_result = run_causal_impact_placebo(daily, pdate, col)

            entry = {"mann_whitney": mw}
            if ci_result is not None:
                entry["causal_impact"] = ci_result

            row["metrics"][col] = entry

        results_by_date.append(row)

    # Compute false positive rates
    fpr = {}
    for col, label, unit in METRICS:
        mw_significant = 0
        mw_total = 0
        ci_significant = 0
        ci_total = 0

        for row in results_by_date:
            mw = row["metrics"][col]["mann_whitney"]
            if mw.get("p_value") is not None:
                mw_total += 1
                if mw["p_value"] < ALPHA:
                    mw_significant += 1

            ci = row["metrics"][col].get("causal_impact")
            if ci is not None and ci.get("p_value") is not None:
                ci_total += 1
                if ci["p_value"] < ALPHA:
                    ci_significant += 1

        fpr[col] = {
            "label": label,
            "unit": unit,
            "mann_whitney": {
                "significant": mw_significant,
                "total": mw_total,
                "fpr": mw_significant / mw_total if mw_total > 0 else None,
            },
        }
        if ci_total > 0:
            fpr[col]["causal_impact"] = {
                "significant": ci_significant,
                "total": ci_total,
                "fpr": ci_significant / ci_total,
            }

    return {
        "n_placebo": len(placebo_dates),
        "alpha": ALPHA,
        "expected_fpr": ALPHA,
        "pre_treatment_start": str(DATA_START),
        "pre_treatment_end": str(TREATMENT_START - timedelta(days=1)),
        "causal_impact_available": CAUSALIMPACT_AVAILABLE,
        "results_by_date": results_by_date,
        "false_positive_rates": fpr,
    }


# ---------------------------------------------------------------------------
# Calibration assessment
# ---------------------------------------------------------------------------

def assess_calibration(fpr: float | None, n_tests: int) -> tuple[str, str]:
    """Classify calibration and return (assessment, status).

    Uses a binomial test: if fpr is significantly above 5%, "liberal".
    """
    if fpr is None or n_tests == 0:
        return "Insufficient data", "neutral"

    n_sig = round(fpr * n_tests)

    # Binomial test: is the observed false positive count significantly
    # above what we would expect under H0: true FPR = 0.05?
    binom_result = scipy_stats.binomtest(n_sig, n_tests, ALPHA, alternative="greater")
    binom_p = binom_result.pvalue

    if fpr <= 0.01:
        return "Conservative (FPR near 0%)", "info"
    elif fpr <= ALPHA + 0.02:
        return "Well-calibrated", "good"
    elif binom_p < 0.05:
        return f"Liberal (FPR={fpr:.0%}, binom p={binom_p:.3f})", "warning"
    else:
        return f"Acceptable (FPR={fpr:.0%}, not significantly above 5%)", "good"


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def build_p_value_cell(p: float | None) -> str:
    """Format a p-value as an HTML table cell with color coding."""
    if p is None:
        return '<td style="color:#6B7280;">N/A</td>'
    color = ACCENT_RED if p < ALPHA else ACCENT_GREEN
    bold = ' font-weight:600;' if p < ALPHA else ''
    return f'<td style="color:{color};{bold}">{format_p_value(p)}</td>'


def generate_html_report(analysis: dict) -> str:
    """Build the full HTML report."""
    fpr = analysis["false_positive_rates"]
    results = analysis["results_by_date"]
    has_ci = analysis["causal_impact_available"]

    # --- KPI cards: false positive rate per metric ---
    kpi_cards = []
    for col, label, unit in METRICS:
        mw_fpr = fpr[col]["mann_whitney"]["fpr"]
        mw_n = fpr[col]["mann_whitney"]["total"]
        mw_sig = fpr[col]["mann_whitney"]["significant"]

        if mw_fpr is not None:
            assessment, status = assess_calibration(mw_fpr, mw_n)
            kpi_cards.append(make_kpi_card(
                label=f"{label} FPR",
                value=f"{mw_fpr:.0%}",
                unit=f"({mw_sig}/{mw_n})",
                status=status,
                detail=assessment,
                explainer=f"Mann-Whitney false positive rate at alpha={ALPHA}",
            ))
        else:
            kpi_cards.append(make_kpi_card(
                label=f"{label} FPR",
                value="N/A",
                status="neutral",
                detail="Insufficient data for test",
            ))

    body = make_kpi_row(*kpi_cards)

    # --- Summary assessment ---
    all_fprs = [
        fpr[col]["mann_whitney"]["fpr"]
        for col, _, _ in METRICS
        if fpr[col]["mann_whitney"]["fpr"] is not None
    ]
    if all_fprs:
        mean_fpr = np.mean(all_fprs)
        if mean_fpr <= ALPHA + 0.02:
            overall = (
                f"Mean false positive rate across metrics: <b>{mean_fpr:.1%}</b>. "
                f"This is consistent with the nominal {ALPHA:.0%} level, "
                "indicating the Mann-Whitney tests are well-calibrated for this data."
            )
        elif mean_fpr <= 0.15:
            overall = (
                f"Mean false positive rate across metrics: <b>{mean_fpr:.1%}</b>. "
                f"This is somewhat above the nominal {ALPHA:.0%} level. "
                "Real treatment effects should be interpreted with moderate caution."
            )
        else:
            overall = (
                f"Mean false positive rate across metrics: <b>{mean_fpr:.1%}</b>. "
                f"This is substantially above the nominal {ALPHA:.0%} level. "
                "The model is overconfident. Real treatment-effect p-values "
                "should be interpreted very cautiously, and stricter significance "
                "thresholds (e.g., p < 0.01) may be appropriate."
            )
    else:
        overall = "Could not compute false positive rates due to insufficient data."

    summary_html = f'<div style="padding:16px 20px;line-height:1.7;color:{TEXT_PRIMARY};">{overall}</div>'
    body += make_section("Calibration Summary", summary_html, section_id="summary")

    # --- Detailed table of all placebo dates ---
    table_header = "<tr><th>Placebo Date</th><th>Day #</th>"
    for _, label, _ in METRICS:
        table_header += f'<th>{label} p</th>'
        if has_ci:
            table_header += f'<th>{label} CI p</th>'
    table_header += "</tr>"

    table_rows = ""
    for row in results:
        pdate = row["placebo_date"]
        day_num = (date.fromisoformat(pdate) - DATA_START).days
        table_rows += f"<tr><td>{pdate}</td><td>{day_num}</td>"
        for col, _, _ in METRICS:
            mw_p = row["metrics"][col]["mann_whitney"].get("p_value")
            table_rows += build_p_value_cell(mw_p)
            if has_ci:
                ci_data = row["metrics"][col].get("causal_impact")
                ci_p = ci_data.get("p_value") if ci_data else None
                table_rows += build_p_value_cell(ci_p)
        table_rows += "</tr>"

    table_css = f"""
    .placebo-table {{
        width: 100%; border-collapse: collapse; font-size: 0.85rem;
    }}
    .placebo-table th {{
        background: {BG_ELEVATED}; color: {TEXT_SECONDARY};
        padding: 10px 12px; text-align: left; font-weight: 600;
        border-bottom: 2px solid {BORDER_DEFAULT};
        white-space: nowrap;
    }}
    .placebo-table td {{
        padding: 8px 12px; border-bottom: 1px solid {BORDER_SUBTLE};
        color: {TEXT_PRIMARY}; font-variant-numeric: tabular-nums;
    }}
    .placebo-table tr:hover td {{ background: {BG_ELEVATED}; }}
    """

    table_html = (
        f'<div style="overflow-x:auto;">'
        f'<table class="placebo-table">{table_header}{table_rows}</table></div>'
    )
    body += make_section(
        "Placebo Test Results (All Dates)",
        table_html,
        section_id="detail_table",
    )

    # --- FPR summary table ---
    fpr_header = "<tr><th>Metric</th><th>Significant</th><th>Total</th><th>FPR</th><th>Assessment</th>"
    if has_ci:
        fpr_header += "<th>CI Sig</th><th>CI Total</th><th>CI FPR</th>"
    fpr_header += "</tr>"

    fpr_rows = ""
    for col, label, unit in METRICS:
        mw = fpr[col]["mann_whitney"]
        assessment, status = assess_calibration(mw["fpr"], mw["total"])
        status_color = {
            "good": ACCENT_GREEN,
            "warning": ACCENT_AMBER,
            "info": ACCENT_BLUE,
            "neutral": TEXT_TERTIARY,
        }.get(status, TEXT_SECONDARY)

        fpr_str = f'{mw["fpr"]:.0%}' if mw["fpr"] is not None else "N/A"

        fpr_rows += (
            f'<tr><td><b>{label}</b></td>'
            f'<td>{mw["significant"]}</td>'
            f'<td>{mw["total"]}</td>'
            f'<td style="font-weight:600;">{fpr_str}</td>'
            f'<td style="color:{status_color};">{assessment}</td>'
        )
        if has_ci:
            ci_data = fpr[col].get("causal_impact", {})
            ci_sig = ci_data.get("significant", 0)
            ci_total = ci_data.get("total", 0)
            ci_fpr = ci_data.get("fpr")
            ci_fpr_str = f"{ci_fpr:.0%}" if ci_fpr is not None else "N/A"
            fpr_rows += f"<td>{ci_sig}</td><td>{ci_total}</td><td>{ci_fpr_str}</td>"
        fpr_rows += "</tr>"

    fpr_table_html = (
        f'<div style="overflow-x:auto;">'
        f'<table class="placebo-table">{fpr_header}{fpr_rows}</table></div>'
    )
    body += make_section(
        "False Positive Rate Summary",
        fpr_table_html,
        section_id="fpr_summary",
    )

    # --- P-value distribution chart ---
    fig = go.Figure()
    for col, label, _ in METRICS:
        p_values = []
        for row in results:
            p = row["metrics"][col]["mann_whitney"].get("p_value")
            if p is not None:
                p_values.append(p)
        if p_values:
            fig.add_trace(go.Histogram(
                x=p_values,
                name=label,
                nbinsx=10,
                opacity=0.7,
            ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="p-value",
        yaxis_title="Count",
        margin=dict(l=60, r=30, t=40, b=50),
        legend=dict(orientation="h", y=1.12),
        shapes=[
            dict(
                type="line",
                x0=ALPHA, x1=ALPHA,
                y0=0, y1=1,
                yref="paper",
                line=dict(color=ACCENT_RED, width=2, dash="dash"),
            )
        ],
        annotations=[
            dict(
                x=ALPHA, y=1.05, yref="paper",
                text=f"alpha={ALPHA}",
                showarrow=False,
                font=dict(color=ACCENT_RED, size=11),
            )
        ],
    )

    chart_html = fig.to_html(include_plotlyjs=False, full_html=False)
    body += make_section(
        "P-Value Distribution Under Null",
        f'<div style="color:{TEXT_SECONDARY};padding:0 20px 8px;font-size:0.85rem;">'
        "Under correct calibration, p-values should be approximately uniformly "
        "distributed (flat histogram). A spike near 0 suggests the model is liberal."
        f"</div>{chart_html}",
        section_id="pvalue_dist",
    )

    # --- Methodology ---
    method_html = f"""
    <div style="padding:16px 20px;line-height:1.8;color:{TEXT_SECONDARY};font-size:0.85rem;">
        <p><b>Purpose:</b> Placebo (falsification) tests check whether the statistical
        methods used for treatment-effect estimation produce false positives at the
        expected nominal rate. If they do, the p-values from the real analysis are
        trustworthy.</p>
        <p><b>Method:</b> {N_PLACEBO} random dates were drawn (seed={RNG_SEED}) from the
        pre-treatment period ({DATA_START} to {TREATMENT_START - timedelta(days=1)}),
        each at least {MIN_BUFFER_DAYS} days from the window edges.
        At each placebo date, the pre-treatment data was split and a two-sided
        Mann-Whitney U test was performed for each metric.
        {'CausalImpact (Bayesian structural time series) was also run where available.' if has_ci else 'CausalImpact was not available in this environment.'}</p>
        <p><b>Expected result:</b> ~{ALPHA:.0%} of placebo tests should be significant
        (1 out of {N_PLACEBO}). If the observed FPR is much higher, the real
        treatment-effect p-values may be overconfident.</p>
        <p><b>Interpretation:</b></p>
        <ul>
            <li><b>Well-calibrated:</b> FPR within ~2 percentage points of {ALPHA:.0%}</li>
            <li><b>Conservative:</b> FPR near 0% (tests are too strict, may miss real effects)</li>
            <li><b>Liberal:</b> FPR significantly above {ALPHA:.0%} (tests find "effects" where none exist)</li>
        </ul>
    </div>
    """
    body += make_section("Methodology", method_html, section_id="methodology")

    html = wrap_html(
        title="Placebo Calibration Tests",
        body_content=body,
        report_id="placebo",
        subtitle=(
            f"{N_PLACEBO} falsification tests across the pre-treatment period "
            f"({DATA_START} to {TREATMENT_START - timedelta(days=1)})"
        ),
        header_meta="Model calibration check",
        extra_css=table_css,
    )
    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("PLACEBO / FALSIFICATION TESTS")
    print("=" * 60)

    daily = load_daily_data()
    placebo_dates = generate_placebo_dates(N_PLACEBO, RNG_SEED)

    print(f"\n[ANALYSIS] Running {len(placebo_dates)} placebo tests for {len(METRICS)} metrics...")
    analysis = run_placebo_analysis(daily, placebo_dates)

    # Print summary to console
    print("\n--- False Positive Rate Summary ---")
    fpr = analysis["false_positive_rates"]
    for col, label, _ in METRICS:
        mw = fpr[col]["mann_whitney"]
        rate = f'{mw["fpr"]:.0%}' if mw["fpr"] is not None else "N/A"
        print(f"  {label}: {mw['significant']}/{mw['total']} significant (FPR={rate})")

    # Save JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n[OUTPUT] JSON: {JSON_OUTPUT}")

    # Save HTML
    html = generate_html_report(analysis)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    print(f"[OUTPUT] HTML: {HTML_OUTPUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()
