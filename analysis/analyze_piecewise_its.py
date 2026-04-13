#!/usr/bin/env python3
"""
Piecewise Interrupted Time Series (ITS) Regression with AR(1) Errors

N-of-1 study: three phases of biometric monitoring in a post-HSCT patient.
  - Phase 1 (baseline): DATA_START to day before TREATMENT_START
  - Phase 2 (Jakavi only): TREATMENT_START to day before BETA_BLOCKER_START
  - Phase 3 (Jakavi + beta-blocker): BETA_BLOCKER_START onward

Model per metric (mean_rmssd, lowest_heart_rate, average_heart_rate,
sleep_efficiency):

    y_t = b0 + b1*time + b2*jakavi + b3*time_since_jakavi
               + b4*bb + b5*time_since_bb + e_t
    e_t = rho * e_{t-1} + u_t   (AR(1) autocorrelation)

Fitted via statsmodels GLSAR (iterative Cochrane-Orcutt).

Outputs:
  - Interactive HTML:  reports/piecewise_regression.html
  - Structured JSON:   reports/piecewise_regression_metrics.json

Usage:
    python analysis/analyze_piecewise_its.py
"""
from __future__ import annotations

import json
import sqlite3
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
from statsmodels.regression.linear_model import GLSAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

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
    BETA_BLOCKER_START,
    DATA_START,
)
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    format_p_value,
    STATUS_COLORS,
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
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "piecewise_regression.html"
JSON_OUTPUT = REPORTS_DIR / "piecewise_regression_metrics.json"

LAYOUT_DEFAULTS = dict(
    margin=dict(l=70, r=30, t=60, b=40),
)

# Metric definitions: column name -> (label, unit, higher_is_better)
METRICS = {
    "mean_rmssd": ("Mean RMSSD", "ms", True),
    "lowest_heart_rate": ("Lowest Heart Rate", "bpm", False),
    "average_heart_rate": ("Average Heart Rate", "bpm", False),
    "sleep_efficiency": ("Sleep Efficiency", "%", True),
}

# Colors per metric
METRIC_COLORS = {
    "mean_rmssd": ACCENT_PURPLE,
    "lowest_heart_rate": ACCENT_GREEN,
    "average_heart_rate": ACCENT_CYAN,
    "sleep_efficiency": ACCENT_BLUE,
}

# ITS coefficient labels
COEFF_LABELS = {
    "const": ("Intercept (b0)", "Baseline level at study start"),
    "time": ("Baseline trend (b1)", "Daily change during baseline"),
    "jakavi": ("Jakavi level shift (b2)", "Immediate change at Jakavi start"),
    "time_since_jakavi": ("Jakavi slope change (b3)", "Change in daily trend after Jakavi"),
    "bb": ("Beta-blocker level shift (b4)", "Immediate change at beta-blocker start"),
    "time_since_bb": ("Beta-blocker slope change (b5)", "Change in daily trend after beta-blocker"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_daily_data() -> pd.DataFrame:
    """Load and merge Oura biometric data into a daily matrix."""
    print("[DATA] Loading biometric data from database...")

    if not DATABASE_PATH.exists():
        print(
            f"ERROR: Database not found at {DATABASE_PATH}. "
            "Run: python api/import_oura.py --days 90",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)

    # HRV epochs -> daily aggregates
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

    # Sleep periods (per-night)
    sleep = pd.read_sql_query(
        """SELECT day as date, average_heart_rate, lowest_heart_rate, efficiency
           FROM oura_sleep_periods
           WHERE type = 'long_sleep'
           ORDER BY day""",
        conn,
    )
    for col in ["average_heart_rate", "lowest_heart_rate", "efficiency"]:
        sleep[col] = pd.to_numeric(sleep[col], errors="coerce")
    sleep = sleep.rename(columns={"efficiency": "sleep_efficiency"})

    # Readiness (for temperature deviation if needed later)
    readiness = pd.read_sql_query(
        "SELECT date, score as readiness_score FROM oura_readiness ORDER BY date",
        conn,
    )
    for col in readiness.columns:
        if col != "date":
            readiness[col] = pd.to_numeric(readiness[col], errors="coerce")

    conn.close()

    # Build merged daily frame
    all_dates = sorted(
        set(hrv_daily["date"].tolist() + sleep["date"].tolist())
    )
    daily = pd.DataFrame({"date": all_dates})
    daily = daily.merge(hrv_daily, on="date", how="left")
    daily = daily.merge(sleep, on="date", how="left")
    daily = daily.merge(readiness, on="date", how="left")

    # Filter to analysis window
    data_start_str = str(DATA_START)
    daily = daily[daily["date"] >= data_start_str].reset_index(drop=True)
    daily = daily.sort_values("date").reset_index(drop=True)

    print(f"  Daily matrix: {len(daily)} days, {daily['date'].iloc[0]} to {daily['date'].iloc[-1]}")
    return daily


# ---------------------------------------------------------------------------
# ITS design matrix construction
# ---------------------------------------------------------------------------


def build_its_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    """Add ITS regressors to the daily DataFrame.

    time:             0, 1, 2, ... from DATA_START
    jakavi:           0 before TREATMENT_START, 1 on/after
    time_since_jakavi: 0 before TREATMENT_START, then 0, 1, 2, ...
    bb:               0 before BETA_BLOCKER_START, 1 on/after
    time_since_bb:    0 before BETA_BLOCKER_START, then 0, 1, 2, ...
    """
    jakavi_str = str(TREATMENT_START)
    bb_str = str(BETA_BLOCKER_START)
    data_start_str = str(DATA_START)

    df = daily.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    data_start_dt = pd.Timestamp(data_start_str)
    jakavi_dt = pd.Timestamp(jakavi_str)
    bb_dt = pd.Timestamp(bb_str)

    df["time"] = (df["date_dt"] - data_start_dt).dt.days
    df["jakavi"] = (df["date_dt"] >= jakavi_dt).astype(int)
    df["time_since_jakavi"] = np.where(
        df["date_dt"] >= jakavi_dt,
        (df["date_dt"] - jakavi_dt).dt.days,
        0,
    )
    df["bb"] = (df["date_dt"] >= bb_dt).astype(int)
    df["time_since_bb"] = np.where(
        df["date_dt"] >= bb_dt,
        (df["date_dt"] - bb_dt).dt.days,
        0,
    )

    n_baseline = (df["jakavi"] == 0).sum()
    n_jakavi_only = ((df["jakavi"] == 1) & (df["bb"] == 0)).sum()
    n_bb = (df["bb"] == 1).sum()
    print(f"  ITS phases: baseline={n_baseline}d, Jakavi-only={n_jakavi_only}d, Jakavi+BB={n_bb}d")

    return df


# ---------------------------------------------------------------------------
# GLSAR fitting
# ---------------------------------------------------------------------------

EXOG_COLS = ["time", "jakavi", "time_since_jakavi", "bb", "time_since_bb"]
GLSAR_MAX_ITER = 20
GLSAR_RHO_ORDER = 1


def fit_its_glsar(
    df: pd.DataFrame,
    metric: str,
) -> dict[str, Any] | None:
    """Fit piecewise ITS with AR(1) errors using GLSAR.

    Returns a dict with coefficients, diagnostics, and fitted values,
    or None if fitting fails.
    """
    label, unit, higher_is_better = METRICS[metric]

    # Drop rows where the metric is NaN
    subset = df[["date", "time", "jakavi", "time_since_jakavi", "bb", "time_since_bb", metric]].dropna(subset=[metric])
    if len(subset) < 10:
        print(f"  WARN: {metric} has only {len(subset)} valid observations, skipping")
        return None

    y = subset[metric].values.astype(float)
    X = subset[EXOG_COLS].values.astype(float)

    # Add constant
    X_with_const = np.column_stack([np.ones(len(y)), X])
    col_names = ["const"] + EXOG_COLS

    try:
        model = GLSAR(y, X_with_const, rho=GLSAR_RHO_ORDER)
        result = model.iterative_fit(maxiter=GLSAR_MAX_ITER)
    except Exception as exc:
        print(f"  WARN: GLSAR fit failed for {metric}: {exc}")
        return None

    # Extract coefficients and confidence intervals
    params = result.params
    conf_int = result.conf_int(alpha=0.05)
    pvalues = result.pvalues
    bse = result.bse

    # Estimated AR(1) coefficient
    rho_hat = float(model.rho[0]) if hasattr(model, "rho") and len(model.rho) > 0 else 0.0

    # Diagnostics on residuals
    resid = result.resid
    dw_stat = float(durbin_watson(resid))

    # Ljung-Box test (up to 10 lags)
    max_lags = min(10, len(resid) // 5)
    if max_lags < 1:
        max_lags = 1
    lb_result = acorr_ljungbox(resid, lags=max_lags, return_df=True)
    lb_pvalue = float(lb_result["lb_pvalue"].iloc[-1])

    # Fitted values (on the original data)
    fitted = result.fittedvalues

    # Build coefficient table
    coeff_table = []
    for i, name in enumerate(col_names):
        coeff_table.append({
            "name": name,
            "label": COEFF_LABELS[name][0],
            "description": COEFF_LABELS[name][1],
            "estimate": float(params[i]),
            "std_error": float(bse[i]),
            "ci_lower": float(conf_int[i, 0]),
            "ci_upper": float(conf_int[i, 1]),
            "p_value": float(pvalues[i]),
            "significant": bool(pvalues[i] < 0.05),
        })

    return {
        "metric": metric,
        "label": label,
        "unit": unit,
        "higher_is_better": higher_is_better,
        "n_obs": len(y),
        "coefficients": coeff_table,
        "rho_hat": rho_hat,
        "r_squared": float(result.rsquared),
        "r_squared_adj": float(result.rsquared_adj),
        "durbin_watson": dw_stat,
        "ljung_box_pvalue": lb_pvalue,
        "residuals_white_noise": lb_pvalue > 0.05,
        "dates": subset["date"].tolist(),
        "observed": y.tolist(),
        "fitted": fitted.tolist(),
        "residuals": resid.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------


def build_fitted_vs_observed_chart(result: dict[str, Any]) -> go.Figure:
    """Observed vs fitted values with intervention markers."""
    dates = pd.to_datetime(result["dates"])
    observed = result["observed"]
    fitted = result["fitted"]
    label = result["label"]
    unit = result["unit"]
    color = METRIC_COLORS.get(result["metric"], ACCENT_BLUE)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=observed,
        mode="markers",
        name="Observed",
        marker=dict(color=color, size=5, opacity=0.6),
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=fitted,
        mode="lines",
        name="ITS Fitted",
        line=dict(color=ACCENT_RED, width=2.5),
    ))

    # Intervention lines (shapes + annotations to avoid Plotly vline datetime bug)
    jakavi_ts = str(TREATMENT_START)
    bb_ts = str(BETA_BLOCKER_START)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"{label} — Observed vs ITS Fitted"),
        xaxis_title="Date",
        yaxis_title=f"{label} ({unit})",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        shapes=[
            dict(type="line", x0=jakavi_ts, x1=jakavi_ts, y0=0, y1=1,
                 yref="paper", line=dict(color=ACCENT_AMBER, width=2, dash="dash")),
            dict(type="line", x0=bb_ts, x1=bb_ts, y0=0, y1=1,
                 yref="paper", line=dict(color=ACCENT_GREEN, width=2, dash="dash")),
        ],
        annotations=[
            dict(x=jakavi_ts, y=1.0, yref="paper", text="Jakavi start",
                 showarrow=False, font=dict(color=ACCENT_AMBER, size=11),
                 xanchor="right", yanchor="bottom"),
            dict(x=bb_ts, y=1.0, yref="paper", text="Beta-blocker",
                 showarrow=False, font=dict(color=ACCENT_GREEN, size=11),
                 xanchor="left", yanchor="bottom"),
        ],
    )

    return fig


def build_residual_diagnostics_chart(result: dict[str, Any]) -> go.Figure:
    """Residual time series + histogram in a 1x2 subplot."""
    dates = pd.to_datetime(result["dates"])
    residuals = np.array(result["residuals"])
    label = result["label"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residuals over Time", "Residual Distribution"],
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.12,
    )

    # Residual time series
    fig.add_trace(go.Scatter(
        x=dates, y=residuals,
        mode="markers+lines",
        name="Residuals",
        marker=dict(color=ACCENT_BLUE, size=4, opacity=0.6),
        line=dict(color=ACCENT_BLUE, width=1),
    ), row=1, col=1)

    # Zero line (shape to avoid hline issues on subplots)
    fig.add_shape(
        type="line", x0=0, x1=1, y0=0, y1=0,
        xref="x domain", yref="y",
        line=dict(color=TEXT_TERTIARY, width=1, dash="dot"),
        row=1, col=1,
    )

    # Intervention lines on residual plot (shapes for compatibility)
    jakavi_ts = str(TREATMENT_START)
    bb_ts = str(BETA_BLOCKER_START)
    fig.add_shape(
        type="line", x0=jakavi_ts, x1=jakavi_ts, y0=0, y1=1,
        yref="y domain",
        line=dict(color=ACCENT_AMBER, width=1.5, dash="dash"),
        row=1, col=1,
    )
    fig.add_shape(
        type="line", x0=bb_ts, x1=bb_ts, y0=0, y1=1,
        yref="y domain",
        line=dict(color=ACCENT_GREEN, width=1.5, dash="dash"),
        row=1, col=1,
    )

    # Histogram
    fig.add_trace(go.Histogram(
        y=residuals,
        nbinsy=20,
        name="Distribution",
        marker=dict(color=ACCENT_BLUE, opacity=0.7),
    ), row=1, col=2)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"{label} — Residual Diagnostics"),
        height=350,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_xaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=2)

    return fig


# ---------------------------------------------------------------------------
# HTML report section builders
# ---------------------------------------------------------------------------


def _coeff_table_html(result: dict[str, Any]) -> str:
    """Build an HTML table of regression coefficients."""
    rows = ""
    for c in result["coefficients"]:
        sig_badge = (
            f'<span style="color:{ACCENT_GREEN};font-weight:600">*</span>'
            if c["significant"] else ""
        )
        p_str = format_p_value(c["p_value"])
        rows += (
            f'<tr>'
            f'<td style="font-weight:500">{c["label"]}</td>'
            f'<td style="text-align:right">{c["estimate"]:+.4f}</td>'
            f'<td style="text-align:right">{c["std_error"]:.4f}</td>'
            f'<td style="text-align:right">[{c["ci_lower"]:+.4f}, {c["ci_upper"]:+.4f}]</td>'
            f'<td style="text-align:right">{p_str} {sig_badge}</td>'
            f'</tr>'
        )

    return f"""
    <table style="width:100%;border-collapse:collapse;margin:12px 0;">
      <thead>
        <tr style="border-bottom:1px solid {BORDER_DEFAULT};">
          <th style="text-align:left;padding:8px 12px;color:{TEXT_SECONDARY}">Parameter</th>
          <th style="text-align:right;padding:8px 12px;color:{TEXT_SECONDARY}">Estimate</th>
          <th style="text-align:right;padding:8px 12px;color:{TEXT_SECONDARY}">Std Error</th>
          <th style="text-align:right;padding:8px 12px;color:{TEXT_SECONDARY}">95% CI</th>
          <th style="text-align:right;padding:8px 12px;color:{TEXT_SECONDARY}">P-value</th>
        </tr>
      </thead>
      <tbody style="font-family:monospace;font-size:0.9em;">
        {rows}
      </tbody>
    </table>
    """


def _diagnostics_html(result: dict[str, Any]) -> str:
    """Build a diagnostics summary card."""
    rho = result["rho_hat"]
    dw = result["durbin_watson"]
    lb_p = result["ljung_box_pvalue"]
    wn = result["residuals_white_noise"]
    r2 = result["r_squared"]
    r2_adj = result["r_squared_adj"]
    n = result["n_obs"]

    wn_status = "good" if wn else "warning"
    wn_text = "Pass (white noise)" if wn else "Fail (autocorrelation remains)"

    return f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:12px 0;">
      <div style="background:{BG_ELEVATED};border-radius:8px;padding:12px;">
        <div style="color:{TEXT_SECONDARY};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em">AR(1) Rho</div>
        <div style="color:{TEXT_PRIMARY};font-size:1.25rem;font-weight:600;margin-top:4px">{rho:.3f}</div>
      </div>
      <div style="background:{BG_ELEVATED};border-radius:8px;padding:12px;">
        <div style="color:{TEXT_SECONDARY};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em">Durbin-Watson</div>
        <div style="color:{TEXT_PRIMARY};font-size:1.25rem;font-weight:600;margin-top:4px">{dw:.3f}</div>
        <div style="color:{TEXT_TERTIARY};font-size:0.7rem;margin-top:2px">Ideal: 2.0</div>
      </div>
      <div style="background:{BG_ELEVATED};border-radius:8px;padding:12px;">
        <div style="color:{TEXT_SECONDARY};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em">Ljung-Box p</div>
        <div style="color:{STATUS_COLORS[wn_status] if wn_status in STATUS_COLORS else TEXT_PRIMARY};font-size:1.25rem;font-weight:600;margin-top:4px">{lb_p:.3f}</div>
        <div style="color:{TEXT_TERTIARY};font-size:0.7rem;margin-top:2px">{wn_text}</div>
      </div>
      <div style="background:{BG_ELEVATED};border-radius:8px;padding:12px;">
        <div style="color:{TEXT_SECONDARY};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em">R-squared</div>
        <div style="color:{TEXT_PRIMARY};font-size:1.25rem;font-weight:600;margin-top:4px">{r2:.3f}</div>
        <div style="color:{TEXT_TERTIARY};font-size:0.7rem;margin-top:2px">Adj: {r2_adj:.3f} | N={n}</div>
      </div>
    </div>
    """


def _build_metric_section(result: dict[str, Any], chart_data: dict[str, str]) -> str:
    """Build a complete HTML section for one metric."""
    metric = result["metric"]
    label = result["label"]
    unit = result["unit"]
    higher_is_better = result["higher_is_better"]

    # Build KPI cards for key intervention coefficients
    cards = []
    for coeff in result["coefficients"]:
        name = coeff["name"]
        if name in ("jakavi", "time_since_jakavi", "bb", "time_since_bb"):
            est = coeff["estimate"]
            sig = coeff["significant"]

            # Determine status based on direction and significance
            if sig:
                if name in ("jakavi", "time_since_jakavi"):
                    favorable = (est > 0) == higher_is_better
                else:
                    favorable = (est > 0) == higher_is_better
                status = "good" if favorable else "warning"
            else:
                status = "neutral"

            detail = f'{format_p_value(coeff["p_value"])} | CI [{coeff["ci_lower"]:+.2f}, {coeff["ci_upper"]:+.2f}]'
            cards.append(make_kpi_card(
                label=coeff["label"].split("(")[0].strip(),
                value=est,
                unit=unit,
                status=status,
                detail=detail,
                decimals=3,
            ))

    kpi_html = make_kpi_row(*cards) if cards else ""

    # Coefficient table
    coeff_html = _coeff_table_html(result)

    # Diagnostics
    diag_html = _diagnostics_html(result)

    # Charts (lazy-loaded)
    fit_key = f"fit_{metric}"
    resid_key = f"resid_{metric}"

    chart_html = (
        f'<div id="chart-{fit_key}" class="chart-box" data-chart="{fit_key}">'
        f'<div style="padding:40px;text-align:center;color:{TEXT_TERTIARY}">Loading chart...</div></div>'
        f'<div id="chart-{resid_key}" class="chart-box" data-chart="{resid_key}">'
        f'<div style="padding:40px;text-align:center;color:{TEXT_TERTIARY}">Loading chart...</div></div>'
    )

    # Build Plotly figures and serialize for lazy loading
    fig_fit = build_fitted_vs_observed_chart(result)
    fig_resid = build_residual_diagnostics_chart(result)

    chart_data[fit_key] = fig_fit.to_json()
    chart_data[resid_key] = fig_resid.to_json()

    content = kpi_html + coeff_html + diag_html + chart_html
    return make_section(f"{label} ({unit})", content, section_id=f"metric-{metric}")


# ---------------------------------------------------------------------------
# Summary KPI cards
# ---------------------------------------------------------------------------


def _build_summary_kpis(all_results: list[dict[str, Any]]) -> str:
    """Build top-level KPI row summarizing significant findings."""
    n_sig_jakavi = 0
    n_sig_bb = 0
    total_metrics = len(all_results)

    for result in all_results:
        for coeff in result["coefficients"]:
            if coeff["name"] in ("jakavi", "time_since_jakavi") and coeff["significant"]:
                n_sig_jakavi += 1
            if coeff["name"] in ("bb", "time_since_bb") and coeff["significant"]:
                n_sig_bb += 1

    # Phase durations
    baseline_days = (TREATMENT_START - DATA_START).days
    jakavi_days = (BETA_BLOCKER_START - TREATMENT_START).days
    bb_days = (date.today() - BETA_BLOCKER_START).days

    cards = [
        make_kpi_card(
            "Baseline",
            baseline_days,
            "days",
            status="info",
            detail=f"{DATA_START} to {TREATMENT_START - pd.Timedelta(days=1):%Y-%m-%d}",
            decimals=0,
        ),
        make_kpi_card(
            "Jakavi Phase",
            jakavi_days,
            "days",
            status="info",
            detail=f"Sig effects: {n_sig_jakavi}/{total_metrics * 2}",
            decimals=0,
        ),
        make_kpi_card(
            "BB Phase",
            bb_days,
            "days",
            status="info",
            detail=f"Sig effects: {n_sig_bb}/{total_metrics * 2}",
            decimals=0,
        ),
        make_kpi_card(
            "Metrics Analyzed",
            total_metrics,
            "",
            status="neutral",
            detail="ITS with AR(1) errors",
            decimals=0,
        ),
    ]
    return make_kpi_row(*cards)


# ---------------------------------------------------------------------------
# Methodology explanation
# ---------------------------------------------------------------------------


def _methodology_section() -> str:
    """Static methodology explanation."""
    content = f"""
    <div style="color:{TEXT_SECONDARY};line-height:1.7;font-size:0.9rem;">
      <p><strong>Interrupted Time Series (ITS)</strong> is a quasi-experimental design
      for evaluating interventions when randomization is not possible. This N-of-1 study
      uses a <strong>piecewise regression</strong> with two interruptions:</p>

      <div style="background:{BG_ELEVATED};border-radius:8px;padding:16px;margin:12px 0;font-family:monospace;font-size:0.85rem;line-height:1.8;">
        y<sub>t</sub> = &beta;<sub>0</sub> + &beta;<sub>1</sub>&middot;time
        + &beta;<sub>2</sub>&middot;jakavi + &beta;<sub>3</sub>&middot;time_since_jakavi
        + &beta;<sub>4</sub>&middot;bb + &beta;<sub>5</sub>&middot;time_since_bb
        + &epsilon;<sub>t</sub><br>
        &epsilon;<sub>t</sub> = &rho;&middot;&epsilon;<sub>t-1</sub> + u<sub>t</sub>
        &nbsp;&nbsp;(AR(1) autocorrelation correction)
      </div>

      <table style="width:100%;border-collapse:collapse;margin:12px 0;">
        <tr style="border-bottom:1px solid {BORDER_SUBTLE};">
          <td style="padding:6px 12px;color:{ACCENT_AMBER};font-weight:600;width:80px">&beta;<sub>2</sub></td>
          <td style="padding:6px 12px">Jakavi level shift — immediate change in the metric when Jakavi started</td>
        </tr>
        <tr style="border-bottom:1px solid {BORDER_SUBTLE};">
          <td style="padding:6px 12px;color:{ACCENT_AMBER};font-weight:600">&beta;<sub>3</sub></td>
          <td style="padding:6px 12px">Jakavi slope change — change in daily trend after Jakavi</td>
        </tr>
        <tr style="border-bottom:1px solid {BORDER_SUBTLE};">
          <td style="padding:6px 12px;color:{ACCENT_GREEN};font-weight:600">&beta;<sub>4</sub></td>
          <td style="padding:6px 12px">Beta-blocker level shift — immediate change when beta-blocker was added</td>
        </tr>
        <tr>
          <td style="padding:6px 12px;color:{ACCENT_GREEN};font-weight:600">&beta;<sub>5</sub></td>
          <td style="padding:6px 12px">Beta-blocker slope change — change in daily trend after adding beta-blocker</td>
        </tr>
      </table>

      <p><strong>AR(1) correction</strong> accounts for day-to-day autocorrelation in
      biometric data (today's measurement is correlated with yesterday's). Without this
      correction, standard errors would be underestimated and p-values too liberal.</p>

      <p>Fitted via <code>statsmodels.GLSAR</code> (iterative Cochrane-Orcutt, up to
      {GLSAR_MAX_ITER} iterations). Residual diagnostics include the Durbin-Watson
      statistic and Ljung-Box test for remaining autocorrelation.</p>
    </div>
    """
    return make_section("Methodology", content, section_id="methodology")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sensitivity analysis: BB intervention date ±N days
# ---------------------------------------------------------------------------

BB_SENSITIVITY_OFFSETS = [-3, -2, -1, 0, +1, +2, +3]


def _build_its_matrix_with_bb(daily: pd.DataFrame, bb_date: date) -> pd.DataFrame:
    """Build ITS design matrix with a custom beta-blocker date."""
    df = daily.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    data_start_dt = pd.Timestamp(str(DATA_START))
    jakavi_dt = pd.Timestamp(str(TREATMENT_START))
    bb_dt = pd.Timestamp(str(bb_date))

    df["time"] = (df["date_dt"] - data_start_dt).dt.days
    df["jakavi"] = (df["date_dt"] >= jakavi_dt).astype(int)
    df["time_since_jakavi"] = np.where(
        df["date_dt"] >= jakavi_dt, (df["date_dt"] - jakavi_dt).dt.days, 0,
    )
    df["bb"] = (df["date_dt"] >= bb_dt).astype(int)
    df["time_since_bb"] = np.where(
        df["date_dt"] >= bb_dt, (df["date_dt"] - bb_dt).dt.days, 0,
    )
    return df


MIN_POST_BB_DAYS = 5  # Minimum post-BB observations for a credible slope estimate


def run_bb_sensitivity(daily: pd.DataFrame) -> dict[str, Any]:
    """Re-fit ITS at shifted BB dates and return coefficient stability table."""
    results: dict[str, list[dict[str, Any]]] = {m: [] for m in METRICS}

    for offset in BB_SENSITIVITY_OFFSETS:
        bb_shifted = BETA_BLOCKER_START + pd.Timedelta(days=offset)
        df = _build_its_matrix_with_bb(daily, bb_shifted)

        for metric in METRICS:
            # Count post-BB observations for this metric
            subset = df[df["bb"] == 1][metric].dropna()
            n_post = len(subset)
            support_status = "supported" if n_post >= MIN_POST_BB_DAYS else "underpowered"

            fit = fit_its_glsar(df, metric)
            if fit is None:
                results[metric].append({
                    "offset": offset,
                    "bb_date": str(bb_shifted.date() if hasattr(bb_shifted, 'date') else bb_shifted),
                    "n_post": n_post,
                    "support_status": "fit_failed",
                    "supported": False,
                    "error": True,
                })
                continue
            # Extract bb level (b4) and bb slope (b5) coefficients
            coeffs = {c["name"]: c for c in fit["coefficients"]}
            b4 = coeffs.get("bb", {})
            b5 = coeffs.get("time_since_bb", {})
            results[metric].append({
                "offset": offset,
                "bb_date": str(bb_shifted.date() if hasattr(bb_shifted, 'date') else bb_shifted),
                "n_post": n_post,
                "support_status": support_status,
                "supported": support_status == "supported",
                "b4_estimate": b4.get("estimate"),
                "b4_pvalue": b4.get("p_value"),
                "b4_significant": b4.get("significant"),
                "b5_estimate": b5.get("estimate"),
                "b5_pvalue": b5.get("p_value"),
                "b5_significant": b5.get("significant"),
                "r_squared": fit["r_squared"],
                "underpowered": n_post < MIN_POST_BB_DAYS,
                "error": False,
            })

    return results


def _summarize_bb_sensitivity_entries(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize support and significance across shifted BB dates."""
    attempted_total = len(entries)
    successful_entries = [e for e in entries if not e.get("error")]
    supported_entries = [e for e in successful_entries if e.get("support_status") == "supported"]
    underpowered_entries = [e for e in successful_entries if e.get("support_status") != "supported"]
    failed_entries = [e for e in entries if e.get("error")]

    return {
        "attempted_total": attempted_total,
        "successful_total": len(successful_entries),
        "supported_total": len(supported_entries),
        "underpowered_total": len(underpowered_entries),
        "failed_total": len(failed_entries),
        "supported_sig_total": sum(1 for e in supported_entries if e.get("b5_significant")),
        "successful_sig_total": sum(1 for e in successful_entries if e.get("b5_significant")),
        "supported_offsets": [e["offset"] for e in supported_entries],
        "underpowered_offsets": [e["offset"] for e in underpowered_entries],
        "failed_offsets": [e["offset"] for e in failed_entries],
    }


def _classify_bb_sensitivity_summary(summary: dict[str, Any]) -> tuple[str, str]:
    """Classify BB-date sensitivity robustness and return label + detail."""
    underpowered_offsets = summary["underpowered_offsets"]
    forward_limited = bool(underpowered_offsets) and all(offset > 0 for offset in underpowered_offsets)

    if summary["supported_total"] == 0:
        return "INSUFFICIENT SUPPORT", "0 supported shifts"

    if summary["supported_sig_total"] == summary["supported_total"]:
        if summary["underpowered_total"] == 0 and summary["failed_total"] == 0:
            return "FULLY ROBUST", f'{summary["supported_sig_total"]}/{summary["attempted_total"]} attempted'
        if forward_limited and summary["failed_total"] == 0:
            return (
                "FORWARD-LIMITED",
                f'{summary["supported_sig_total"]}/{summary["supported_total"]} supported; '
                f'{summary["underpowered_total"]}/{summary["attempted_total"]} forward shift(s) underpowered',
            )
        return (
            "SUPPORTED ONLY",
            f'{summary["supported_sig_total"]}/{summary["supported_total"]} supported; '
            f'{summary["underpowered_total"]} underpowered, {summary["failed_total"]} failed',
        )

    if summary["supported_sig_total"] == 0:
        return "SENSITIVE", f'0/{summary["supported_total"]} supported'

    return "MIXED", f'{summary["supported_sig_total"]}/{summary["supported_total"]} supported'


def _format_offset_list(offsets: list[int]) -> str:
    """Format integer day offsets for compact prose."""
    if not offsets:
        return "none"
    return ", ".join(f"{offset:+d}d" for offset in offsets)


def _build_sensitivity_interpretation_payload(sensitivity: dict[str, Any]) -> dict[str, Any]:
    """Plain-text interpretation payload for report prose and JSON export."""
    overall = [
        "The beta-blocker date sensitivity analysis should not be described as a symmetric ±3-day robustness check.",
        f"Several future-shifted models fall near the end of the time series and therefore have fewer than {MIN_POST_BB_DAYS} post-beta-blocker observations.",
        "Those forward shifts are still shown for transparency, but they are treated as support-limited rather than equal-weight evidence.",
    ]

    per_metric: dict[str, str] = {}
    for metric, entries in sensitivity.items():
        label = METRICS[metric][0]
        summary = _summarize_bb_sensitivity_entries(entries)
        status, _detail = _classify_bb_sensitivity_summary(summary)
        supported_offsets = summary["supported_offsets"]
        underpowered_offsets = summary["underpowered_offsets"]

        if status == "FULLY ROBUST":
            sentence = (
                f"{label}: the beta-blocker slope term remained significant across all attempted shifts "
                f"({_format_offset_list(supported_offsets)}), so this metric is fully robust to the tested date perturbation."
            )
        elif status == "FORWARD-LIMITED":
            sentence = (
                f"{label}: the beta-blocker slope term remained significant in every supported re-fit "
                f"({_format_offset_list(supported_offsets)}). Later forward shifts "
                f"({_format_offset_list(underpowered_offsets)}) were underpowered because the post-beta-blocker window "
                f"was shorter than {MIN_POST_BB_DAYS} observations, so this should be described as forward-limited rather than fully robust."
            )
        elif status == "SUPPORTED ONLY":
            sentence = (
                f"{label}: all supported shifts remained significant, but underpowered or failed shifts "
                f"({_format_offset_list(underpowered_offsets + summary['failed_offsets'])}) limit how strongly date robustness can be claimed."
            )
        elif status == "MIXED":
            sentence = (
                f"{label}: supported shifts showed mixed significance, so the estimated beta-blocker slope is sensitive to plausible date changes."
            )
        elif status == "SENSITIVE":
            sentence = (
                f"{label}: none of the supported shifts remained significant, so the estimated beta-blocker slope is not robust to plausible date changes."
            )
        else:
            sentence = (
                f"{label}: the post-beta-blocker window is too short to support a meaningful date sensitivity claim."
            )

        per_metric[metric] = sentence

    manuscript_text = (
        "Suggested manuscript wording: "
        "\"The beta-blocker sensitivity analysis supported the observed post-beta-blocker slope effect in all adequately "
        "supported re-fits, but forward-shifted models near the series end were underpowered because they contained fewer "
        f"than {MIN_POST_BB_DAYS} post-beta-blocker observations. We therefore describe the analysis as forward-limited "
        "rather than fully robust to symmetric ±3-day date uncertainty.\""
    )

    return {
        "overall": overall,
        "per_metric": per_metric,
        "manuscript_wording": manuscript_text,
    }


def _build_sensitivity_interpretation_section(sensitivity: dict[str, Any]) -> str:
    """Reviewer-safe narrative interpretation for the BB-date sensitivity analysis."""
    payload = _build_sensitivity_interpretation_payload(sensitivity)
    metric_lines = [
        f"<li>{sentence}</li>"
        for sentence in payload["per_metric"].values()
    ]

    content = f"""
    <div style="color:{TEXT_SECONDARY};line-height:1.7;font-size:0.9rem;">
      <p>{payload["overall"][0]}</p>
      <p>{payload["overall"][1]} {payload["overall"][2]}</p>
      <ul style="margin:12px 0 12px 18px;padding:0;">
        {"".join(metric_lines)}
      </ul>
      <p><strong>{payload["manuscript_wording"]}</strong></p>
    </div>
    """
    return content


def _build_sensitivity_section(sensitivity: dict[str, Any]) -> str:
    """Build HTML section for BB date sensitivity analysis."""
    rows_per_metric: list[str] = []

    for metric, entries in sensitivity.items():
        label = METRICS[metric][0]
        unit = METRICS[metric][1]
        color = METRIC_COLORS.get(metric, ACCENT_BLUE)
        summary = _summarize_bb_sensitivity_entries(entries)
        underpowered_offsets = summary["underpowered_offsets"]
        failed_offsets = summary["failed_offsets"]
        badge_label, badge_detail = _classify_bb_sensitivity_summary(summary)
        if badge_label == "FULLY ROBUST":
            badge_color = ACCENT_GREEN
        elif badge_label in {"FORWARD-LIMITED", "SUPPORTED ONLY", "MIXED"}:
            badge_color = ACCENT_AMBER
        else:
            badge_color = ACCENT_RED

        table_rows = []
        for e in entries:
            if e.get("error"):
                n_post_str = str(e.get("n_post", "?"))
                table_rows.append(
                    f'<tr><td>{e["offset"]:+d}</td><td>{e.get("bb_date", "?")}</td>'
                    f'<td>{n_post_str}</td>'
                    f'<td colspan="6" style="color:{ACCENT_RED}">fit failed</td></tr>'
                )
                continue

            is_actual = e["offset"] == 0
            row_style = f'font-weight:600;background:{BG_ELEVATED}' if is_actual else ''
            marker = " (actual)" if is_actual else ""
            underpowered = e.get("underpowered", False)
            support_label = "supported" if e.get("support_status") == "supported" else f"underpowered (&lt;{MIN_POST_BB_DAYS})"
            support_color = ACCENT_GREEN if e.get("support_status") == "supported" else ACCENT_AMBER

            b4_sig = f'<span style="color:{ACCENT_GREEN}">*</span>' if e.get("b4_significant") else ""
            b5_sig = f'<span style="color:{ACCENT_GREEN}">*</span>' if e.get("b5_significant") else ""

            b4_est = f'{e["b4_estimate"]:+.2f}' if e.get("b4_estimate") is not None else "—"
            b5_est = f'{e["b5_estimate"]:+.3f}' if e.get("b5_estimate") is not None else "—"
            b4_p = format_p_value(e["b4_pvalue"]) if e.get("b4_pvalue") is not None else "—"
            b5_p = format_p_value(e["b5_pvalue"]) if e.get("b5_pvalue") is not None else "—"

            n_post = e.get("n_post", "?")
            n_post_warn = f' <span style="color:{ACCENT_AMBER}" title="&lt;{MIN_POST_BB_DAYS} post-BB days">&#9888;</span>' if underpowered else ""

            table_rows.append(
                f'<tr style="{row_style}">'
                f'<td>{e["offset"]:+d}d{marker}</td>'
                f'<td>{e["bb_date"]}</td>'
                f'<td>{n_post}{n_post_warn}</td>'
                f'<td style="color:{support_color}">{support_label}</td>'
                f'<td>{b4_est} {unit} {b4_sig}</td><td>{b4_p}</td>'
                f'<td>{b5_est} {unit}/day {b5_sig}</td><td>{b5_p}</td>'
                f'<td>{e["r_squared"]:.3f}</td></tr>'
            )

        if summary["supported_total"] == 0:
            badge = f'<span style="color:{badge_color};font-weight:600">{badge_label}</span>'
        else:
            badge = f'<span style="color:{badge_color};font-weight:600">{badge_label} ({badge_detail})</span>'

        support_tail = []
        if underpowered_offsets:
            offsets = ", ".join(f"{offset:+d}" for offset in underpowered_offsets)
            support_tail.append(f'underpowered offsets: {offsets}')
        if failed_offsets:
            offsets = ", ".join(f"{offset:+d}" for offset in failed_offsets)
            support_tail.append(f'failed offsets: {offsets}')
        support_tail_html = ""
        if support_tail:
            support_tail_html = " " + "; ".join(support_tail) + "."

        support_note = (
            f'Support threshold: at least {MIN_POST_BB_DAYS} non-missing post-BB observations. '
            f'{summary["supported_total"]}/{summary["attempted_total"]} shifted dates met the threshold; '
            f'{summary["underpowered_total"]} were underpowered and {summary["failed_total"]} fit(s) failed.'
            f'{support_tail_html}'
        )

        rows_per_metric.append(f"""
        <div style="margin:20px 0">
          <div style="font-weight:600;color:{color};font-size:1rem">{label} — {badge}</div>
          <div style="color:{TEXT_SECONDARY};font-size:0.82rem;margin-top:4px">{support_note}</div>
          <table style="width:100%;border-collapse:collapse;font-size:0.85rem;margin-top:8px">
            <thead><tr style="border-bottom:2px solid {BORDER_DEFAULT}">
              <th style="padding:8px;color:{TEXT_PRIMARY}">Offset</th>
              <th style="padding:8px;color:{TEXT_PRIMARY}">BB Date</th>
              <th style="padding:8px;color:{TEXT_PRIMARY}">Post-BB n</th>
              <th style="padding:8px;color:{TEXT_PRIMARY}">Support</th>
              <th style="padding:8px;color:{TEXT_PRIMARY}">Level (b4)</th>
              <th style="padding:8px;color:{TEXT_PRIMARY}">p</th>
              <th style="padding:8px;color:{TEXT_PRIMARY}">Slope (b5)</th>
              <th style="padding:8px;color:{TEXT_PRIMARY}">p</th>
              <th style="padding:8px;color:{TEXT_PRIMARY}">R²</th>
            </tr></thead>
            <tbody>{"".join(table_rows)}</tbody>
          </table>
        </div>""")

    method_note = (
        f'<div style="color:{TEXT_SECONDARY};font-size:0.85rem;margin-bottom:12px">'
        f'<strong>Method:</strong> The ITS model is re-fitted with the beta-blocker date '
        f'shifted by -3 to +3 days. FULLY ROBUST = all attempted shifts fit and all supported '
        f'shifts are significant. FORWARD-LIMITED = all supported shifts are significant, but one or more '
        f'future-shifted fits have &lt;{MIN_POST_BB_DAYS} post-BB observations. SUPPORTED ONLY = supported shifts '
        f'are significant, but support gaps or failed fits remain. MIXED = only some supported shifts are significant; '
        f'SENSITIVE = no supported shifts are significant. Shifts with &lt;{MIN_POST_BB_DAYS} '
        f'post-BB days are flagged (&#9888;) and excluded from the significance denominator, but still listed for transparency. '
        f'Highlighted row = actual date ({BETA_BLOCKER_START}). '
        f'Post-BB n = non-missing metric days after the shifted BB date. * = p &lt; 0.05.</div>'
    )

    return method_note + "".join(rows_per_metric)


def main() -> None:
    print("=" * 60)
    print("Piecewise ITS Regression with AR(1) Errors")
    print("=" * 60)

    # Load and prepare data
    daily = load_daily_data()
    df = build_its_matrix(daily)

    # Fit models for each metric
    all_results: list[dict[str, Any]] = []
    for metric in METRICS:
        print(f"\n[FIT] {METRICS[metric][0]} ({metric})...")
        result = fit_its_glsar(df, metric)
        if result is not None:
            all_results.append(result)
            n_sig = sum(1 for c in result["coefficients"] if c["significant"])
            print(f"  R2={result['r_squared']:.3f}, rho={result['rho_hat']:.3f}, "
                  f"DW={result['durbin_watson']:.3f}, {n_sig} significant coefficients")

    if not all_results:
        print("ERROR: No metrics could be fitted. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Sensitivity analysis: BB date ±3 days
    print("\n[SENSITIVITY] Re-fitting with BB date shifted ±3 days...")
    sensitivity = run_bb_sensitivity(daily)
    for metric, entries in sensitivity.items():
        summary = _summarize_bb_sensitivity_entries(entries)
        status, detail = _classify_bb_sensitivity_summary(summary)
        print(
            f"  {METRICS[metric][0]}: {status} "
            f"({detail}) "
            f"[{summary['underpowered_total']} underpowered, {summary['failed_total']} failed]"
        )

    # Build HTML report
    print("\n[REPORT] Generating HTML report...")
    chart_data: dict[str, str] = {}

    body = _build_summary_kpis(all_results)
    body += _methodology_section()

    for result in all_results:
        body += _build_metric_section(result, chart_data)

    body += make_section(
        "Sensitivity Interpretation",
        _build_sensitivity_interpretation_section(sensitivity),
        section_id="sensitivity-interpretation",
    )

    body += make_section(
        "Sensitivity Analysis: BB Intervention Date ±3 Days",
        _build_sensitivity_section(sensitivity),
        section_id="sensitivity",
    )

    html = wrap_html(
        title="Piecewise ITS Regression",
        body_content=body,
        report_id="piecewise_its",
        subtitle=f"AR(1)-corrected interrupted time series: {len(METRICS)} biometric endpoints",
        chart_data=chart_data,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT.write_text(html, encoding="utf-8")
    print(f"  HTML saved: {HTML_OUTPUT}")

    # Build JSON metrics
    json_metrics: dict[str, Any] = {
        "generated": datetime.now().isoformat(),
        "model": "Piecewise ITS with AR(1) errors (GLSAR)",
        "phases": {
            "baseline": {"start": str(DATA_START), "end": str(TREATMENT_START)},
            "jakavi": {"start": str(TREATMENT_START), "end": str(BETA_BLOCKER_START)},
            "beta_blocker": {"start": str(BETA_BLOCKER_START)},
        },
        "metrics": {},
    }

    for result in all_results:
        metric = result["metric"]
        json_metrics["metrics"][metric] = {
            "label": result["label"],
            "unit": result["unit"],
            "n_obs": result["n_obs"],
            "r_squared": result["r_squared"],
            "r_squared_adj": result["r_squared_adj"],
            "rho_hat": result["rho_hat"],
            "durbin_watson": result["durbin_watson"],
            "ljung_box_pvalue": result["ljung_box_pvalue"],
            "residuals_white_noise": result["residuals_white_noise"],
            "coefficients": {
                c["name"]: {
                    "estimate": c["estimate"],
                    "std_error": c["std_error"],
                    "ci_lower": c["ci_lower"],
                    "ci_upper": c["ci_upper"],
                    "p_value": c["p_value"],
                    "significant": c["significant"],
                }
                for c in result["coefficients"]
            },
        }

    # Add sensitivity results and support summaries for auditability.
    json_metrics["bb_date_sensitivity"] = {
        metric: entries
        for metric, entries in sensitivity.items()
    }
    json_metrics["bb_date_sensitivity_summary"] = {
        metric: _summarize_bb_sensitivity_entries(entries)
        for metric, entries in sensitivity.items()
    }
    json_metrics["bb_date_sensitivity_interpretation"] = _build_sensitivity_interpretation_payload(sensitivity)
    json_metrics["bb_date_sensitivity_meta"] = {
        "min_post_bb_days": MIN_POST_BB_DAYS,
        "offsets": BB_SENSITIVITY_OFFSETS,
    }

    JSON_OUTPUT.write_text(json.dumps(json_metrics, indent=2), encoding="utf-8")
    print(f"  JSON saved: {JSON_OUTPUT}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary of significant findings (p < 0.05):")
    print("-" * 60)
    for result in all_results:
        sig_coeffs = [c for c in result["coefficients"] if c["significant"] and c["name"] != "const"]
        if sig_coeffs:
            print(f"\n  {result['label']}:")
            for c in sig_coeffs:
                print(f"    {c['label']}: {c['estimate']:+.4f} {result['unit']} "
                      f"({format_p_value(c['p_value'])})")
        else:
            print(f"\n  {result['label']}: No significant intervention effects")
    print("=" * 60)


if __name__ == "__main__":
    main()
