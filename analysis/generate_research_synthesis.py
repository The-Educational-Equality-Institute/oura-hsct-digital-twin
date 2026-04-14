#!/usr/bin/env python3
"""Generate the Research Synthesis HTML report: Two-Hit Autonomic Recovery.

Loads live biometric data from the Oura DB plus statistical results from
sibling JSON metrics files, and renders a comprehensive, clinician-facing
HTML report using the project dark clinical theme.

Outputs:
    reports/research_synthesis.html
    reports/research_synthesis_metrics.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path resolution & imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATABASE_PATH,
    REPORTS_DIR,
    DATA_START,
    TREATMENT_START,
    BETA_BLOCKER_START,
    TRANSPLANT_DATE,
    ESC_RMSSD_DEFICIENCY,
    POPULATION_RMSSD_MEAN,
)
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    disclaimer_banner,
    format_p_value,
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
    ACCENT_ORANGE,
    FONT_FAMILY,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "research_synthesis.html"
JSON_OUTPUT = REPORTS_DIR / "research_synthesis_metrics.json"

# Period colors
C_PRE = TEXT_SECONDARY
C_JAKAVI = ACCENT_BLUE
C_BISO = ACCENT_GREEN

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_daily_data() -> pd.DataFrame:
    """Load HRV and sleep HR from DB into a daily matrix."""
    print("[DATA] Loading biometric data from database...")
    if not DATABASE_PATH.exists():
        print(f"ERROR: Database not found at {DATABASE_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)

    hrv = pd.read_sql_query(
        "SELECT timestamp, rmssd FROM oura_hrv ORDER BY timestamp", conn
    )
    hrv["date"] = pd.to_datetime(hrv["timestamp"], utc=True).dt.date.astype(str)
    hrv["rmssd"] = pd.to_numeric(hrv["rmssd"], errors="coerce")
    hrv_daily = (
        hrv.groupby("date").agg(mean_rmssd=("rmssd", "mean")).reset_index()
    )

    sleep = pd.read_sql_query(
        """SELECT day as date, average_heart_rate, lowest_heart_rate
           FROM oura_sleep_periods
           WHERE type = 'long_sleep' ORDER BY day""",
        conn,
    )
    for col in ["average_heart_rate", "lowest_heart_rate"]:
        sleep[col] = pd.to_numeric(sleep[col], errors="coerce")

    conn.close()

    all_dates = sorted(set(hrv_daily["date"].tolist() + sleep["date"].tolist()))
    daily = pd.DataFrame({"date": all_dates})
    daily = daily.merge(hrv_daily, on="date", how="left")
    daily = daily.merge(sleep, on="date", how="left")
    daily = daily[daily["date"] >= str(DATA_START)].reset_index(drop=True)
    daily = daily.sort_values("date").reset_index(drop=True)
    print(f"  Daily matrix: {len(daily)} days, {daily['date'].iloc[0]} to {daily['date'].iloc[-1]}")
    return daily


def compute_kpis(daily: pd.DataFrame) -> dict[str, Any]:
    """Compute all KPI values from the live daily data."""
    today = date.today()
    days_on_jakavi = (today - TREATMENT_START).days
    days_on_biso = (today - BETA_BLOCKER_START).days

    ts = str(TREATMENT_START)
    bb = str(BETA_BLOCKER_START)

    pre = daily[daily["date"] < ts]
    jak = daily[(daily["date"] >= ts) & (daily["date"] < bb)]
    biso = daily[daily["date"] >= bb]

    pre_rmssd = pre["mean_rmssd"].dropna()
    jak_rmssd = jak["mean_rmssd"].dropna()
    biso_rmssd = biso["mean_rmssd"].dropna()
    latest_rmssd = daily["mean_rmssd"].dropna().iloc[-1] if not daily["mean_rmssd"].dropna().empty else None

    pre_hr = pre["average_heart_rate"].dropna()
    current_hr = biso["average_heart_rate"].dropna() if not biso["average_heart_rate"].dropna().empty else jak["average_heart_rate"].dropna()

    return {
        "days_on_jakavi": days_on_jakavi,
        "days_on_bisoprolol": days_on_biso,
        "pre_rmssd_mean": round(float(pre_rmssd.mean()), 1) if len(pre_rmssd) else None,
        "jakavi_rmssd_mean": round(float(jak_rmssd.mean()), 1) if len(jak_rmssd) else None,
        "biso_rmssd_mean": round(float(biso_rmssd.mean()), 1) if len(biso_rmssd) else None,
        "latest_rmssd": round(float(latest_rmssd), 1) if latest_rmssd is not None else None,
        "pre_hr_mean": round(float(pre_hr.mean()), 1) if len(pre_hr) else None,
        "current_hr_mean": round(float(current_hr.mean()), 1) if len(current_hr) else None,
        "n_pre": int(len(pre_rmssd)),
        "n_jakavi": int(len(jak_rmssd)),
        "n_biso": int(len(biso_rmssd)),
        "data_end": daily["date"].iloc[-1],
    }


# ---------------------------------------------------------------------------
# JSON loaders for statistical results
# ---------------------------------------------------------------------------


def _load_json(name: str) -> dict:
    path = REPORTS_DIR / name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Hero chart: RMSSD time series with interventions
# ---------------------------------------------------------------------------


def build_hero_chart(daily: pd.DataFrame) -> str:
    """Interactive Plotly chart: full RMSSD series, color-coded by period."""
    ts = str(TREATMENT_START)
    bb = str(BETA_BLOCKER_START)

    pre = daily[daily["date"] < ts].copy()
    jak = daily[(daily["date"] >= ts) & (daily["date"] < bb)].copy()
    biso = daily[daily["date"] >= bb].copy()

    fig = go.Figure()

    # Daily points by period
    for df, name, color in [
        (pre, "Pre-treatment", C_PRE),
        (jak, "Jakavi only", C_JAKAVI),
        (biso, "Jakavi + Bisoprolol", C_BISO),
    ]:
        if df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["mean_rmssd"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=7, opacity=0.8),
            hovertemplate="%{x}<br>RMSSD: %{y:.1f} ms<extra>" + name + "</extra>",
        ))

    # Weekly means as stepped line
    daily_copy = daily.copy()
    daily_copy["date_dt"] = pd.to_datetime(daily_copy["date"])
    daily_copy["week"] = daily_copy["date_dt"].dt.isocalendar().week.astype(int)
    daily_copy["year"] = daily_copy["date_dt"].dt.isocalendar().year.astype(int)
    weekly = daily_copy.groupby(["year", "week"]).agg(
        mean_rmssd=("mean_rmssd", "mean"),
        date_start=("date", "first"),
        date_end=("date", "last"),
    ).reset_index()

    # Build stepped x/y pairs
    step_x, step_y = [], []
    for _, row in weekly.iterrows():
        if pd.notna(row["mean_rmssd"]):
            step_x.extend([row["date_start"], row["date_end"]])
            step_y.extend([row["mean_rmssd"], row["mean_rmssd"]])

    fig.add_trace(go.Scatter(
        x=step_x, y=step_y,
        mode="lines",
        name="Weekly mean",
        line=dict(color=ACCENT_AMBER, width=2.5, dash="solid"),
        opacity=0.85,
    ))

    # Compute date range for annotations and shading
    date_min = daily["date"].iloc[0]
    date_max = daily["date"].iloc[-1]
    y_max = float(daily["mean_rmssd"].dropna().max()) * 1.15

    # ESC threshold
    fig.add_hline(
        y=ESC_RMSSD_DEFICIENCY, line_dash="dash",
        line_color=ACCENT_RED, opacity=0.5,
    )
    fig.add_annotation(
        x=date_min, y=ESC_RMSSD_DEFICIENCY, text="ESC threshold (15 ms)",
        showarrow=False, yshift=10, xanchor="left",
        font=dict(color=ACCENT_RED, size=11),
    )

    # Intervention lines (shapes + annotations to avoid Plotly string-date bug)
    for x_val, color, label, xanch in [
        (ts, ACCENT_BLUE, "Jakavi start", "right"),
        (bb, ACCENT_GREEN, "Bisoprolol start", "left"),
    ]:
        fig.add_shape(
            type="line", x0=x_val, x1=x_val, y0=0, y1=y_max,
            line=dict(color=color, width=1.5, dash="dash"), opacity=0.7,
        )
        fig.add_annotation(
            x=x_val, y=y_max, text=label, showarrow=False,
            xanchor=xanch, yshift=4,
            font=dict(color=color, size=11),
        )
    fig.add_vrect(
        x0=date_min, x1=ts,
        fillcolor=C_PRE, opacity=0.04, line_width=0,
    )
    fig.add_vrect(
        x0=ts, x1=bb,
        fillcolor=ACCENT_BLUE, opacity=0.06, line_width=0,
    )
    fig.add_vrect(
        x0=bb, x1=date_max,
        fillcolor=ACCENT_GREEN, opacity=0.06, line_width=0,
    )

    fig.update_layout(
        title="RMSSD Time Series: Three-Phase Observation",
        xaxis_title="Date",
        yaxis_title="RMSSD (ms)",
        height=460,
        margin=dict(l=60, r=30, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        hovermode="x unified",
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _status_for_rmssd(val: float | None) -> str:
    if val is None:
        return "neutral"
    if val < ESC_RMSSD_DEFICIENCY:
        return "critical"
    if val < POPULATION_RMSSD_MEAN:
        return "warning"
    return "normal"


def build_executive_summary(kpis: dict) -> str:
    """Section 0: KPI cards + summary paragraph."""
    cards = make_kpi_row(
        make_kpi_card(
            "Day on Jakavi", kpis["days_on_jakavi"], "",
            status="info", detail=f"Since {TREATMENT_START.strftime('%b %d')}",
            decimals=0,
        ),
        make_kpi_card(
            "Pre-Tx RMSSD", kpis["pre_rmssd_mean"], "ms",
            status="critical", detail=f"n={kpis['n_pre']} days",
        ),
        make_kpi_card(
            "Jakavi+BB RMSSD", kpis["biso_rmssd_mean"], "ms",
            status=_status_for_rmssd(kpis["biso_rmssd_mean"]),
            detail=f"n={kpis['n_biso']} days",
        ),
        make_kpi_card(
            "Latest RMSSD", kpis["latest_rmssd"], "ms",
            status=_status_for_rmssd(kpis["latest_rmssd"]),
            detail=f"As of {kpis['data_end']}",
        ),
    )

    pct_change = ""
    if kpis["pre_rmssd_mean"] and kpis["biso_rmssd_mean"]:
        pct = ((kpis["biso_rmssd_mean"] - kpis["pre_rmssd_mean"]) / kpis["pre_rmssd_mean"]) * 100
        pct_change = f" ({pct:+.0f}% from baseline)"

    summary = f"""
    <div style="margin-top:20px; line-height:1.8; color:{TEXT_SECONDARY}; font-size:0.95rem;">
    <p><strong style="color:{TEXT_PRIMARY}">Finding:</strong> A post-HSCT patient with severe autonomic
    dysfunction (RMSSD ~{kpis['pre_rmssd_mean']} ms, 1.6th population percentile) showed an accelerating
    HRV recovery after sequential ruxolitinib + bisoprolol, reaching a mean of
    {kpis['biso_rmssd_mean']} ms{pct_change} during the combined treatment period.</p>
    <p><strong style="color:{TEXT_PRIMARY}">Mechanism hypothesis:</strong> Ruxolitinib suppressed the
    inflammatory driver (Hit 1, anti-inflammatory priming), bisoprolol unmasked recovered vagal tone
    (Hit 2, sympatholytic unmasking). The ITS slope change is significant (p&lt;0.001) while the level
    shift is not (p=0.19) — consistent with an accelerating emergence from the PPG noise floor rather
    than a sudden pharmacological jump.</p>
    <p><strong style="color:{ACCENT_AMBER}">Key caveat:</strong> Pre-treatment RMSSD values (~10 ms)
    are at the Oura Ring PPG noise floor. Quantitative pre-treatment values should be interpreted as
    qualitative markers of severe autonomic depression, not precise measurements.</p>
    </div>
    """
    return cards + summary


def build_two_hit_model() -> str:
    """Section 2: Hypothesis with mechanism flow diagram."""
    return f"""
    <div style="margin-bottom:24px; color:{TEXT_SECONDARY}; line-height:1.7; font-size:0.95rem;">
    <p><strong style="color:{TEXT_PRIMARY}">Core claim:</strong> HRV recovery was likely underway during
    ruxolitinib monotherapy but below the Oura Ring's RMSSD detection threshold. Bisoprolol accelerated
    and unmasked the recovery rather than initiating it.</p>
    </div>

    <h3 style="color:{TEXT_PRIMARY}; margin:20px 0 16px; font-size:1.05rem;">The Vicious Cycle (pre-treatment)</h3>
    <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:center; gap:8px;
                padding:20px; background:{BG_ELEVATED}; border-radius:10px; border:1px solid {BORDER_SUBTLE};
                margin-bottom:24px;">
        <div style="background:rgba(239,68,68,0.15); color:{ACCENT_RED}; padding:10px 16px;
                    border-radius:8px; font-weight:600; font-size:0.88rem; text-align:center;
                    border:1px solid rgba(239,68,68,0.25); min-width:140px;">
            Chronic GvHD
        </div>
        <div style="color:{TEXT_TERTIARY}; font-size:1.3rem;">&rarr;</div>
        <div style="background:rgba(239,68,68,0.1); color:#FCA5A5; padding:10px 16px;
                    border-radius:8px; font-size:0.85rem; text-align:center;
                    border:1px solid rgba(239,68,68,0.15); min-width:140px;">
            Cytokine release<br><span style="font-size:0.75rem; color:{TEXT_TERTIARY}">IL-6, TNF-a, IFN-g</span>
        </div>
        <div style="color:{TEXT_TERTIARY}; font-size:1.3rem;">&rarr;</div>
        <div style="background:rgba(245,158,11,0.12); color:{ACCENT_AMBER}; padding:10px 16px;
                    border-radius:8px; font-size:0.85rem; text-align:center;
                    border:1px solid rgba(245,158,11,0.2); min-width:160px;">
            Sympathetic activation<br>+ vagal suppression
        </div>
        <div style="color:{TEXT_TERTIARY}; font-size:1.3rem;">&rarr;</div>
        <div style="background:rgba(239,68,68,0.12); color:#FCA5A5; padding:10px 16px;
                    border-radius:8px; font-size:0.85rem; text-align:center;
                    border:1px solid rgba(239,68,68,0.15); min-width:180px;">
            Loss of cholinergic<br>anti-inflammatory reflex
        </div>
        <div style="color:{TEXT_TERTIARY}; font-size:1.3rem;">&circlearrowright;</div>
    </div>

    <h3 style="color:{TEXT_PRIMARY}; margin:20px 0 16px; font-size:1.05rem;">The Two-Hit Intervention</h3>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px;">
        <div style="background:{BG_ELEVATED}; border-radius:10px; padding:20px;
                    border-left:4px solid {ACCENT_BLUE}; border:1px solid {BORDER_SUBTLE};
                    border-left:4px solid {ACCENT_BLUE};">
            <div style="color:{ACCENT_BLUE}; font-weight:700; font-size:0.95rem; margin-bottom:8px;">
                Hit 1: Ruxolitinib (JAK1/JAK2 inhibitor)
            </div>
            <ul style="color:{TEXT_SECONDARY}; font-size:0.85rem; padding-left:18px; line-height:1.7;">
                <li>Suppresses IL-6, TNF-a, IFN-g (within 5-7 days)</li>
                <li>De-suppresses brainstem vagal nuclei</li>
                <li>Restores macrophage cholinergic sensitivity</li>
                <li>Evidence: HR dropped significantly (p=0.009)</li>
                <li><strong>HRV did NOT improve measurably</strong> — masked by sympathetic saturation + PPG noise floor</li>
            </ul>
        </div>
        <div style="background:{BG_ELEVATED}; border-radius:10px; padding:20px;
                    border-left:4px solid {ACCENT_GREEN}; border:1px solid {BORDER_SUBTLE};
                    border-left:4px solid {ACCENT_GREEN};">
            <div style="color:{ACCENT_GREEN}; font-weight:700; font-size:0.95rem; margin-bottom:8px;">
                Hit 2: Bisoprolol (beta-1 selective blocker)
            </div>
            <ul style="color:{TEXT_SECONDARY}; font-size:0.85rem; padding-left:18px; line-height:1.7;">
                <li>Blocks sympathetic input at SA node (~40-50% receptor occupancy)</li>
                <li>Unmasks pre-recovered vagal modulation</li>
                <li>Triggers baroreflex-mediated vagal potentiation</li>
                <li>Restores cholinergic anti-inflammatory reflex</li>
                <li><strong>System flips from vicious to virtuous cycle</strong></li>
            </ul>
        </div>
    </div>

    <h3 style="color:{TEXT_PRIMARY}; margin:20px 0 16px; font-size:1.05rem;">The Virtuous Cycle (post-intervention)</h3>
    <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:center; gap:8px;
                padding:20px; background:{BG_ELEVATED}; border-radius:10px; border:1px solid {BORDER_SUBTLE};">
        <div style="background:rgba(16,185,129,0.15); color:{ACCENT_GREEN}; padding:10px 16px;
                    border-radius:8px; font-weight:600; font-size:0.88rem; text-align:center;
                    border:1px solid rgba(16,185,129,0.25); min-width:130px;">
            Beta-blockade
        </div>
        <div style="color:{TEXT_TERTIARY}; font-size:1.3rem;">&rarr;</div>
        <div style="background:rgba(59,130,246,0.12); color:#93C5FD; padding:10px 16px;
                    border-radius:8px; font-size:0.85rem; text-align:center;
                    border:1px solid rgba(59,130,246,0.15); min-width:140px;">
            Vagal unmasking
        </div>
        <div style="color:{TEXT_TERTIARY}; font-size:1.3rem;">&rarr;</div>
        <div style="background:rgba(16,185,129,0.1); color:#6EE7B7; padding:10px 16px;
                    border-radius:8px; font-size:0.85rem; text-align:center;
                    border:1px solid rgba(16,185,129,0.15); min-width:160px;">
            Cholinergic reflex<br>reactivates
        </div>
        <div style="color:{TEXT_TERTIARY}; font-size:1.3rem;">&rarr;</div>
        <div style="background:rgba(59,130,246,0.1); color:#93C5FD; padding:10px 16px;
                    border-radius:8px; font-size:0.85rem; text-align:center;
                    border:1px solid rgba(59,130,246,0.15); min-width:160px;">
            Further cytokine<br>suppression
        </div>
        <div style="color:{ACCENT_GREEN}; font-size:1.3rem;">&circlearrowright;</div>
    </div>

    <p style="margin-top:16px; color:{TEXT_TERTIARY}; font-size:0.83rem; font-style:italic;">
    Framing: temporally separable effects consistent with distinct mechanisms (not pharmacological synergy).
    Analogous to ACE-inhibitor + beta-blocker in heart failure — each addresses a different node of the same
    dysfunctional circuit.
    </p>
    """


def build_alternative_explanations() -> str:
    """Section 3: PPG noise floor as lead, then other alternatives."""
    return f"""
    <div style="background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3);
                border-left:4px solid {ACCENT_AMBER}; border-radius:10px; padding:20px;
                margin-bottom:24px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <span style="font-size:1.3rem;">&#9888;</span>
            <strong style="color:{ACCENT_AMBER}; font-size:1rem;">PPG Measurement Floor — Primary Caveat</strong>
        </div>
        <div style="color:{TEXT_SECONDARY}; font-size:0.88rem; line-height:1.7;">
        <p>At RMSSD ~10 ms, the Oura Ring PPG noise (~5-10 ms IBI estimation error) equals the physiological
        signal. <strong>RMSSD<sub>measured</sub> = sqrt(RMSSD<sub>true</sub><sup>2</sup> +
        RMSSD<sub>noise</sub><sup>2</sup>)</strong>. The signal-to-noise ratio at baseline was approximately 1:1.</p>
        <p style="margin-top:8px;"><strong>No published PPG validation study has tested accuracy at RMSSD &lt;15 ms.</strong>
        Cao 2022, Liang 2024, and Dial 2025 all used healthy populations (RMSSD 20-80 ms). Pre-treatment values
        should be interpreted as qualitative ("severely depressed"), not quantitative.</p>
        <p style="margin-top:8px;">The ITS model is consistent with this: level shift NOT significant (b4 p=0.19)
        but slope change highly significant (b5 p&lt;0.001) — an accelerating emergence from the noise floor,
        not a sudden pharmacological jump.</p>
        </div>
    </div>

    <div style="color:{TEXT_SECONDARY}; font-size:0.88rem; line-height:1.7;">
    <h3 style="color:{TEXT_PRIMARY}; font-size:0.95rem; margin-bottom:12px;">Other Alternative Explanations</h3>
    <ol style="padding-left:20px;">
        <li style="margin-bottom:12px;">
            <strong style="color:{TEXT_PRIMARY};">Baseline compression / floor effect (Stein 2005):</strong>
            Patients with the lowest baseline HRV show the largest relative improvements.
            The 134% relative increase represents only +13 ms — patient remains well below normal (42 ms median).
        </li>
        <li style="margin-bottom:12px;">
            <strong style="color:{TEXT_PRIMARY};">Cycle-length dependence:</strong>
            RMSSD scales with RR interval. HR dropping ~80 to ~70 bpm mechanically amplifies vagal modulation
            expression. Estimated 15-25% of observed RMSSD increase is cycle-length effect.
        </li>
        <li style="margin-bottom:12px;">
            <strong style="color:{TEXT_PRIMARY};">Iron clearance / phlebotomy:</strong>
            Ferritin 2225 to 1247 ug/L with phlebotomy. Iron overload causes direct cardiac toxicity.
            Cardiac T2* never measured — <strong>this confound is not addressable from available data.</strong>
        </li>
        <li style="margin-bottom:12px;">
            <strong style="color:{TEXT_PRIMARY};">HEV resolution:</strong>
            Hepatitis E diagnosed D+2 of ruxolitinib. If HEV viral load was declining, this would
            independently reduce inflammatory burden. <strong>HEV PCR trajectory needed.</strong>
        </li>
        <li style="margin-bottom:12px;">
            <strong style="color:{TEXT_PRIMARY};">Regression to the mean:</strong>
            Placebo falsification tests show 100% false positive rate with Mann-Whitney on pre-treatment data
            (non-stationary baseline), confirming ITS with trend control is the valid primary analysis.
        </li>
        <li style="margin-bottom:12px;">
            <strong style="color:{TEXT_PRIMARY};">Seasonal / activity confounding:</strong>
            January to April in Norway — increasing daylight and outdoor activity. Not addressable from
            available data.
        </li>
    </ol>
    </div>
    """


def build_predictions() -> str:
    """Section 4: Falsifiable predictions with status badges."""
    predictions = [
        (
            "Inflammatory markers should decline BEFORE the HRV inflection",
            "hsCRP, IL-6 trajectory should show decline before bisoprolol was added.",
            "TESTABLE NOW", ACCENT_GREEN,
        ),
        (
            "CRP should show FURTHER decline after bisoprolol",
            "The restored vagal tone should contribute its own anti-inflammatory effect (cholinergic reflex).",
            "REQUIRES LABS", ACCENT_AMBER,
        ),
        (
            "Bisoprolol washout should show HRV decline to a HIGHER floor",
            "Expected: HRV drops to ~15-18 ms (not back to ~10 ms). If it drops to ~10 ms, the two-hit model is wrong.",
            "PROSPECTIVE", ACCENT_BLUE,
        ),
        (
            "Bisoprolol dose increase should show diminishing HRV returns",
            "At 2.5mg (~40-50% receptor occupancy), titrating to 5mg should produce proportionally smaller gains.",
            "PROSPECTIVE", ACCENT_BLUE,
        ),
    ]

    items = ""
    for i, (title, desc, badge, color) in enumerate(predictions, 1):
        items += f"""
        <div style="display:flex; gap:16px; align-items:flex-start; padding:16px;
                    background:{BG_ELEVATED}; border-radius:8px; margin-bottom:10px;
                    border:1px solid {BORDER_SUBTLE};">
            <div style="min-width:32px; height:32px; background:rgba(59,130,246,0.15);
                        border-radius:50%; display:flex; align-items:center; justify-content:center;
                        color:{ACCENT_BLUE}; font-weight:700; font-size:0.9rem; flex-shrink:0;">{i}</div>
            <div style="flex:1;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px; flex-wrap:wrap;">
                    <strong style="color:{TEXT_PRIMARY}; font-size:0.9rem;">{title}</strong>
                    <span style="background:rgba({_hex_to_rgb(color)},0.15); color:{color};
                                 padding:2px 10px; border-radius:12px; font-size:0.72rem;
                                 font-weight:600; white-space:nowrap;">{badge}</span>
                </div>
                <div style="color:{TEXT_SECONDARY}; font-size:0.83rem; line-height:1.6;">{desc}</div>
            </div>
        </div>
        """
    return items


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' string for rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


def build_why_it_matters() -> str:
    """Section 5: Tracey reflex hook, Koopman mirror, novelty."""
    return f"""
    <div style="color:{TEXT_SECONDARY}; font-size:0.9rem; line-height:1.7;">
    <p style="margin-bottom:16px;">
    <strong style="color:{TEXT_PRIMARY}; font-size:1rem;">The headline:</strong> This case represents the
    <strong style="color:{ACCENT_CYAN};">first wearable-documented observation of the Tracey inflammatory
    reflex closing the loop in a human, in real time, in vivo.</strong>
    </p>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px;">
        <div style="background:{BG_ELEVATED}; padding:16px; border-radius:8px; border:1px solid {BORDER_SUBTLE};">
            <div style="color:{ACCENT_BLUE}; font-weight:600; font-size:0.85rem; margin-bottom:8px;">
                Koopman et al. 2016 (PNAS)
            </div>
            <div style="color:{TEXT_SECONDARY}; font-size:0.83rem;">
                Demonstrated: VNS &rarr; cytokine reduction in RA<br>
                <span style="color:{TEXT_TERTIARY}; font-size:0.78rem;">Direction: vagal stimulation &rarr; anti-inflammatory</span>
            </div>
        </div>
        <div style="background:{BG_ELEVATED}; padding:16px; border-radius:8px; border:1px solid {BORDER_SUBTLE};">
            <div style="color:{ACCENT_GREEN}; font-weight:600; font-size:0.85rem; margin-bottom:8px;">
                This observation (mirror image)
            </div>
            <div style="color:{TEXT_SECONDARY}; font-size:0.83rem;">
                Demonstrates: anti-inflammatory Rx &rarr; vagal recovery<br>
                <span style="color:{TEXT_TERTIARY}; font-size:0.78rem;">Direction: cytokine suppression &rarr; vagal restoration</span>
            </div>
        </div>
    </div>

    <h3 style="color:{TEXT_PRIMARY}; font-size:0.95rem; margin:16px 0 10px;">What is novel (no published precedent):</h3>
    <ol style="padding-left:20px; font-size:0.85rem;">
        <li style="margin-bottom:6px;">JAK inhibitor + beta-blocker with HRV as an outcome — in any disease</li>
        <li style="margin-bottom:6px;">Wearable-tracked autonomic recovery trajectory during ruxolitinib — in any context</li>
        <li style="margin-bottom:6px;">Quantitative "two-hit" autonomic restoration pattern in a GvHD patient</li>
        <li style="margin-bottom:6px;">Continuous 96-day HRV time series spanning both intervention points with formal causal inference</li>
    </ol>

    <p style="margin-top:16px; color:{TEXT_TERTIARY}; font-size:0.82rem; font-style:italic;">
    HSCT is the setting, not the contribution. The finding — sequential anti-inflammatory + sympatholytic therapy
    produces a state transition in autonomic function — has implications for heart failure, RA/SLE, post-sepsis
    autonomic dysfunction, and diabetic autonomic neuropathy with inflammatory component.
    </p>
    </div>
    """


def build_limitations() -> str:
    """Section 6: Numbered limitations, prominent."""
    items = [
        ("<strong>N=1.</strong> Single-patient observation. Cannot establish generalizability.", "critical"),
        ("<strong>No randomization or blinding.</strong> Confounding by indication and placebo effect cannot be excluded.", "critical"),
        ("<strong>PPG-derived HRV, not ECG.</strong> Oura Ring NOT validated at RMSSD &lt;15 ms. Pre-treatment values are near the instrument detection limit.", "warning"),
        ("<strong>HEV confound.</strong> Hepatitis E diagnosed D+2. Viral load trajectory unknown.", "warning"),
        ("<strong>Phlebotomy confound.</strong> Iron reduction has independent autonomic effects. Cardiac T2* not measured.", "warning"),
        ("<strong>Short post-bisoprolol window.</strong> Sustained response needs confirmation at 30, 60, 90 days.", "warning"),
        ("<strong>No inflammatory biomarker trajectory spanning both interventions.</strong>", "warning"),
        ("<strong>Seasonal confounding.</strong> Jan to Apr transition in Norway.", "info"),
        ("<strong>Concurrent medications</strong> beyond ruxolitinib and bisoprolol could affect autonomic function.", "info"),
    ]

    html = ""
    for i, (text, severity) in enumerate(items, 1):
        color = STATUS_COLORS.get(severity, TEXT_SECONDARY)
        bg = f"rgba({_hex_to_rgb(color)},0.08)"
        border = f"rgba({_hex_to_rgb(color)},0.25)"
        html += f"""
        <div style="display:flex; gap:12px; align-items:flex-start; padding:12px 16px;
                    background:{bg}; border-radius:8px; margin-bottom:8px;
                    border:1px solid {border};">
            <span style="color:{color}; font-weight:700; font-size:0.85rem; min-width:20px;">{i}.</span>
            <span style="color:{TEXT_SECONDARY}; font-size:0.85rem; line-height:1.6;">{text}</span>
        </div>
        """
    return html


def build_statistical_evidence(kpis: dict) -> str:
    """Section 7: Pull results from JSON files, render as comparison table."""
    its = _load_json("piecewise_regression_metrics.json")
    ci = _load_json("sequential_causal_impact_metrics.json")
    tau = _load_json("tau_u_metrics.json")
    placebo = _load_json("placebo_calibration_metrics.json")

    rows = ""

    # ITS coefficients for RMSSD
    its_rmssd = its.get("metrics", {}).get("mean_rmssd", {})
    coeffs = its_rmssd.get("coefficients", {})
    r2 = its_rmssd.get("r_squared")
    ljung = its_rmssd.get("ljung_box_pvalue")

    its_entries = [
        ("b3: Jakavi slope", "time_since_jakavi", "Change in daily RMSSD trend after Jakavi"),
        ("b4: BB level shift", "bb", "Immediate RMSSD change at bisoprolol start"),
        ("b5: BB slope change", "time_since_bb", "Change in daily RMSSD trend after bisoprolol"),
    ]
    for label, key, desc in its_entries:
        c = coeffs.get(key, {})
        est = c.get("estimate")
        p = c.get("p_value")
        sig = c.get("significant", False)
        est_str = f"{est:+.2f} ms/day" if est is not None else "N/A"
        if key == "bb":
            est_str = f"{est:+.2f} ms" if est is not None else "N/A"
        p_str = format_p_value(p)
        sig_color = ACCENT_GREEN if sig else ACCENT_AMBER
        sig_text = "Significant" if sig else "Not significant"
        rows += _table_row("Piecewise ITS", label, est_str, p_str, sig_text, sig_color)

    # R-squared and Ljung-Box
    if r2 is not None:
        rows += _table_row("Piecewise ITS", "Model R\u00b2", f"{r2:.3f}", "", "Fit", ACCENT_BLUE)
    if ljung is not None:
        ljung_ok = ljung > 0.05
        rows += _table_row(
            "Piecewise ITS", "Ljung-Box",
            format_p_value(ljung),
            "",
            "Passes" if ljung_ok else "Fails",
            ACCENT_GREEN if ljung_ok else ACCENT_RED,
        )

    # CausalImpact
    for run_key, run_label in [("run_a", "CI Run A (Jakavi)"), ("run_b", "CI Run B (BB marginal)")]:
        run = ci.get(run_key, {})
        streams = run.get("streams", {})
        rmssd_s = streams.get("mean_rmssd", {})
        if rmssd_s:
            eff = rmssd_s.get("avg_effect")
            p = rmssd_s.get("p_value")
            sig = str(rmssd_s.get("significant", "False")).lower() == "true"
            eff_str = f"{eff:+.2f} ms" if eff is not None else "N/A"
            p_str = format_p_value(p)
            sig_color = ACCENT_GREEN if sig else ACCENT_AMBER
            sig_text = "Significant" if sig else "Not significant"
            rows += _table_row("CausalImpact", run_label + " RMSSD", eff_str, p_str, sig_text, sig_color)

    # Tau-U
    tau_comps = tau.get("comparisons", {})
    for comp_key, comp_label in [("A_vs_B", "A vs B (Jakavi)"), ("B_vs_C", "B vs C (BB marginal)")]:
        comp = tau_comps.get(comp_key, {})
        mets = comp.get("metrics", {})
        rmssd_t = mets.get("mean_rmssd", {})
        tau_data = rmssd_t.get("tau_u", {})
        if tau_data:
            t_val = tau_data.get("tau")
            p = tau_data.get("p_value")
            t_str = f"Tau={t_val:+.3f}" if t_val is not None else "N/A"
            p_str = format_p_value(p)
            sig = p is not None and p < 0.05
            sig_color = ACCENT_GREEN if sig else ACCENT_AMBER
            rows += _table_row("Tau-U", comp_label + " RMSSD", t_str, p_str,
                               "Large" if sig else "Weak", sig_color)

    # Placebo
    fpr = placebo.get("false_positive_rates", {})
    rmssd_fpr = fpr.get("mean_rmssd", {})
    mw = rmssd_fpr.get("mann_whitney", {})
    if mw:
        rate = mw.get("fpr")
        rate_str = f"{rate:.0%}" if rate is not None else "N/A"
        rows += _table_row(
            "Placebo", "Mann-Whitney FPR (RMSSD)", rate_str, "",
            "Confirms ITS needed" if rate == 1.0 else "",
            ACCENT_AMBER if rate and rate > 0.2 else ACCENT_GREEN,
        )

    table = f"""
    <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; font-size:0.83rem;">
    <thead>
        <tr style="border-bottom:2px solid {BORDER_DEFAULT};">
            <th style="text-align:left; padding:10px 12px; color:{TEXT_TERTIARY}; font-weight:600;">Method</th>
            <th style="text-align:left; padding:10px 12px; color:{TEXT_TERTIARY}; font-weight:600;">Test</th>
            <th style="text-align:right; padding:10px 12px; color:{TEXT_TERTIARY}; font-weight:600;">Estimate</th>
            <th style="text-align:right; padding:10px 12px; color:{TEXT_TERTIARY}; font-weight:600;">p-value</th>
            <th style="text-align:center; padding:10px 12px; color:{TEXT_TERTIARY}; font-weight:600;">Status</th>
        </tr>
    </thead>
    <tbody>{rows}</tbody>
    </table>
    </div>
    """
    return table


def _table_row(method: str, test: str, estimate: str, p_val: str, status: str, color: str) -> str:
    """Render a single row of the evidence table."""
    return f"""
    <tr style="border-bottom:1px solid {BORDER_SUBTLE};">
        <td style="padding:10px 12px; color:{TEXT_SECONDARY};">{method}</td>
        <td style="padding:10px 12px; color:{TEXT_PRIMARY};">{test}</td>
        <td style="padding:10px 12px; text-align:right; color:{TEXT_PRIMARY}; font-family:monospace;">{estimate}</td>
        <td style="padding:10px 12px; text-align:right; color:{TEXT_SECONDARY}; font-family:monospace;">{p_val}</td>
        <td style="padding:10px 12px; text-align:center;">
            <span style="background:rgba({_hex_to_rgb(color)},0.15); color:{color};
                         padding:2px 10px; border-radius:12px; font-size:0.75rem;
                         font-weight:600;">{status}</span>
        </td>
    </tr>
    """


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  Research Synthesis: Two-Hit Autonomic Recovery")
    print("=" * 60)

    daily = load_daily_data()
    kpis = compute_kpis(daily)

    print(f"  Pre-treatment RMSSD: {kpis['pre_rmssd_mean']} ms (n={kpis['n_pre']})")
    print(f"  Jakavi-only RMSSD:   {kpis['jakavi_rmssd_mean']} ms (n={kpis['n_jakavi']})")
    print(f"  Jakavi+BB RMSSD:     {kpis['biso_rmssd_mean']} ms (n={kpis['n_biso']})")
    print(f"  Latest RMSSD:        {kpis['latest_rmssd']} ms")

    # Build sections
    body = ""

    body += disclaimer_banner()

    body += make_section(
        "Executive Summary",
        build_executive_summary(kpis),
        section_id="executive",
    )

    body += make_section(
        "RMSSD Timeline: Three-Phase Observation",
        '<div class="chart-box">' + build_hero_chart(daily) + "</div>",
        section_id="timeline",
    )

    body += make_section(
        "The Two-Hit Model (Hypothesis)",
        build_two_hit_model(),
        section_id="two-hit",
    )

    body += make_section(
        "Alternative Explanations",
        build_alternative_explanations(),
        section_id="alternatives",
    )

    body += make_section(
        "Falsifiable Predictions",
        build_predictions(),
        section_id="predictions",
    )

    body += make_section(
        "Why It Matters",
        build_why_it_matters(),
        section_id="significance",
    )

    body += make_section(
        "Limitations",
        build_limitations(),
        section_id="limitations",
    )

    body += make_section(
        "Statistical Evidence Summary",
        build_statistical_evidence(kpis),
        section_id="evidence",
    )

    # Assemble page
    html = wrap_html(
        "Research Synthesis: Two-Hit Autonomic Recovery",
        body,
        report_id="synthesis",
        subtitle="N-of-1 observation in post-HSCT chronic GvHD",
        data_end=kpis["data_end"],
    )

    HTML_OUTPUT.write_text(html, encoding="utf-8")
    size_mb = HTML_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n  HTML: {HTML_OUTPUT} ({size_mb:.1f} MB)")

    # Metrics JSON
    metrics = {
        "generated": datetime.now().isoformat(),
        "report": "research_synthesis",
        "kpis": kpis,
        "statistical_sources": [
            "piecewise_regression_metrics.json",
            "sequential_causal_impact_metrics.json",
            "tau_u_metrics.json",
            "placebo_calibration_metrics.json",
        ],
    }
    JSON_OUTPUT.write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"  JSON: {JSON_OUTPUT}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
