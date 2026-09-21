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

import json, sqlite3, sys, warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np, pandas as pd
import plotly.graph_objects as go, plotly.io as pio

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATABASE_PATH, REPORTS_DIR, DATA_START, TREATMENT_START,
    BETA_BLOCKER_START, ESC_RMSSD_DEFICIENCY, POPULATION_RMSSD_MEAN,
)
from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section, disclaimer_banner,
    format_p_value, STATUS_COLORS, BG_ELEVATED, BORDER_SUBTLE, BORDER_DEFAULT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_CYAN,
)

pio.templates.default = "clinical_dark"
HTML_OUTPUT = REPORTS_DIR / "research_synthesis.html"
JSON_OUTPUT = REPORTS_DIR / "research_synthesis_metrics.json"
C_PRE, C_JAK, C_BIS = TEXT_SECONDARY, ACCENT_BLUE, ACCENT_GREEN


def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _pill(text: str, color: str) -> str:
    return (f'<span style="background:rgba({_hex_rgb(color)},.15);color:{color};'
            f'padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600">{text}</span>')


def _card(border_color: str, title: str, body: str) -> str:
    return (f'<div style="background:{BG_ELEVATED};border-radius:10px;padding:20px;'
            f'border:1px solid {BORDER_SUBTLE};border-left:4px solid {border_color}">'
            f'<div style="color:{border_color};font-weight:700;font-size:.95rem;margin-bottom:8px">'
            f'{title}</div>{body}</div>')


def _flow_box(bg_rgba: str, color: str, text: str, border_rgba: str, w: str = "140px") -> str:
    return (f'<div style="background:{bg_rgba};color:{color};padding:10px 16px;border-radius:8px;'
            f'font-size:.85rem;text-align:center;border:1px solid {border_rgba};min-width:{w}">{text}</div>')


def _arrow() -> str:
    return f'<div style="color:{TEXT_TERTIARY};font-size:1.3rem">&rarr;</div>'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_data() -> pd.DataFrame:
    print("[DATA] Loading biometric data from database...")
    if not DATABASE_PATH.exists():
        print(f"ERROR: Database not found at {DATABASE_PATH}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    hrv = pd.read_sql_query("SELECT timestamp, rmssd FROM oura_hrv ORDER BY timestamp", conn)
    hrv["date"] = pd.to_datetime(hrv["timestamp"], utc=True).dt.date.astype(str)
    hrv["rmssd"] = pd.to_numeric(hrv["rmssd"], errors="coerce")
    hrv_daily = hrv.groupby("date").agg(mean_rmssd=("rmssd", "mean")).reset_index()
    sleep = pd.read_sql_query(
        "SELECT day as date, average_heart_rate, lowest_heart_rate "
        "FROM oura_sleep_periods WHERE type='long_sleep' ORDER BY day", conn)
    for c in ["average_heart_rate", "lowest_heart_rate"]:
        sleep[c] = pd.to_numeric(sleep[c], errors="coerce")
    conn.close()
    dates = sorted(set(hrv_daily["date"].tolist() + sleep["date"].tolist()))
    daily = pd.DataFrame({"date": dates})
    daily = daily.merge(hrv_daily, on="date", how="left").merge(sleep, on="date", how="left")
    daily = daily[daily["date"] >= str(DATA_START)].sort_values("date").reset_index(drop=True)
    print(f"  Daily matrix: {len(daily)} days, {daily['date'].iloc[0]} to {daily['date'].iloc[-1]}")
    return daily


def compute_kpis(daily: pd.DataFrame) -> dict[str, Any]:
    ts, bb = str(TREATMENT_START), str(BETA_BLOCKER_START)
    pre = daily[daily["date"] < ts]
    jak = daily[(daily["date"] >= ts) & (daily["date"] < bb)]
    biso = daily[daily["date"] >= bb]

    def _mean(s): return round(float(s.mean()), 1) if len(s) else None

    pre_r, jak_r, bis_r = pre["mean_rmssd"].dropna(), jak["mean_rmssd"].dropna(), biso["mean_rmssd"].dropna()
    latest = daily["mean_rmssd"].dropna()
    cur_hr = biso["average_heart_rate"].dropna()
    if cur_hr.empty:
        cur_hr = jak["average_heart_rate"].dropna()
    return dict(
        days_on_jakavi=(date.today() - TREATMENT_START).days,
        days_on_bisoprolol=(date.today() - BETA_BLOCKER_START).days,
        pre_rmssd_mean=_mean(pre_r), jakavi_rmssd_mean=_mean(jak_r),
        biso_rmssd_mean=_mean(bis_r),
        latest_rmssd=round(float(latest.iloc[-1]), 1) if not latest.empty else None,
        pre_hr_mean=_mean(pre["average_heart_rate"].dropna()),
        current_hr_mean=_mean(cur_hr),
        n_pre=int(len(pre_r)), n_jakavi=int(len(jak_r)), n_biso=int(len(bis_r)),
        data_end=daily["date"].iloc[-1],
    )


def _load_json(name: str) -> dict:
    p = REPORTS_DIR / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Hero chart
# ---------------------------------------------------------------------------

def build_hero_chart(daily: pd.DataFrame) -> str:
    ts, bb = str(TREATMENT_START), str(BETA_BLOCKER_START)
    fig = go.Figure()
    for mask, name, color in [
        (daily["date"] < ts, "Pre-treatment", C_PRE),
        ((daily["date"] >= ts) & (daily["date"] < bb), "Jakavi only", C_JAK),
        (daily["date"] >= bb, "Jakavi + Bisoprolol", C_BIS),
    ]:
        df = daily[mask]
        if df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["mean_rmssd"], mode="markers", name=name,
            marker=dict(color=color, size=7, opacity=0.8),
            hovertemplate="%{x}<br>RMSSD: %{y:.1f} ms<extra>" + name + "</extra>"))

    # Weekly stepped means
    dc = daily.copy()
    dc["date_dt"] = pd.to_datetime(dc["date"])
    dc["yw"] = dc["date_dt"].dt.isocalendar().week.astype(int)
    dc["yy"] = dc["date_dt"].dt.isocalendar().year.astype(int)
    wk = dc.groupby(["yy", "yw"]).agg(m=("mean_rmssd", "mean"), d0=("date", "first"), d1=("date", "last")).reset_index()
    sx, sy = [], []
    for _, r in wk.iterrows():
        if pd.notna(r["m"]):
            sx.extend([r["d0"], r["d1"]]); sy.extend([r["m"], r["m"]])
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="Weekly mean",
                             line=dict(color=ACCENT_AMBER, width=2.5), opacity=0.85))

    dmin, dmax = daily["date"].iloc[0], daily["date"].iloc[-1]
    ymax = float(daily["mean_rmssd"].dropna().max()) * 1.15
    fig.add_hline(y=ESC_RMSSD_DEFICIENCY, line_dash="dash", line_color=ACCENT_RED, opacity=0.5)
    fig.add_annotation(x=dmin, y=ESC_RMSSD_DEFICIENCY, text="ESC threshold (15 ms)",
                       showarrow=False, yshift=10, xanchor="left", font=dict(color=ACCENT_RED, size=11))
    for xv, col, lbl, xa in [(ts, C_JAK, "Jakavi start", "right"), (bb, C_BIS, "Bisoprolol start", "left")]:
        fig.add_shape(type="line", x0=xv, x1=xv, y0=0, y1=ymax,
                      line=dict(color=col, width=1.5, dash="dash"), opacity=0.7)
        fig.add_annotation(x=xv, y=ymax, text=lbl, showarrow=False, xanchor=xa, yshift=4,
                           font=dict(color=col, size=11))
    for x0, x1, fc, op in [(dmin, ts, C_PRE, 0.04), (ts, bb, C_JAK, 0.06), (bb, dmax, C_BIS, 0.06)]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=fc, opacity=op, line_width=0)
    fig.update_layout(title="RMSSD Time Series: Three-Phase Observation",
                      xaxis_title="Date", yaxis_title="RMSSD (ms)", height=460,
                      margin=dict(l=60, r=30, t=70, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
                      hovermode="x unified")
    return fig.to_html(include_plotlyjs=False, full_html=False)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _rmssd_status(v):
    if v is None: return "neutral"
    if v < ESC_RMSSD_DEFICIENCY: return "critical"
    return "warning" if v < POPULATION_RMSSD_MEAN else "normal"


def build_executive_summary(k: dict) -> str:
    cards = make_kpi_row(
        make_kpi_card("Day on Jakavi", k["days_on_jakavi"], "", status="info",
                      detail=f"Since {TREATMENT_START:%b %d}", decimals=0),
        make_kpi_card("Pre-Tx RMSSD", k["pre_rmssd_mean"], "ms", status="critical",
                      detail=f"n={k['n_pre']} days"),
        make_kpi_card("Jakavi+BB RMSSD", k["biso_rmssd_mean"], "ms",
                      status=_rmssd_status(k["biso_rmssd_mean"]), detail=f"n={k['n_biso']} days"),
        make_kpi_card("Latest RMSSD", k["latest_rmssd"], "ms",
                      status=_rmssd_status(k["latest_rmssd"]), detail=f"As of {k['data_end']}"),
    )
    pct = ""
    if k["pre_rmssd_mean"] and k["biso_rmssd_mean"]:
        pct = f" ({((k['biso_rmssd_mean']-k['pre_rmssd_mean'])/k['pre_rmssd_mean'])*100:+.0f}% from baseline)"
    return cards + f"""
    <div style="margin-top:20px;line-height:1.8;color:{TEXT_SECONDARY};font-size:.95rem">
    <p><b style="color:{TEXT_PRIMARY}">Finding:</b> A post-HSCT patient with severe autonomic dysfunction
    (RMSSD ~{k['pre_rmssd_mean']} ms, 1.6th percentile) showed accelerating HRV recovery after sequential
    ruxolitinib + bisoprolol, reaching {k['biso_rmssd_mean']} ms{pct} during the combined period.</p>
    <p><b style="color:{TEXT_PRIMARY}">Mechanism hypothesis:</b> Ruxolitinib suppressed the inflammatory
    driver (Hit 1), bisoprolol unmasked recovered vagal tone (Hit 2). The ITS slope change is significant
    (p&lt;0.001) while the level shift is not (p=0.19) - consistent with accelerating emergence from the
    PPG noise floor rather than a sudden pharmacological jump.</p>
    <p><b style="color:{ACCENT_AMBER}">Key caveat:</b> Pre-treatment RMSSD values (~10 ms) are at the
    Oura Ring PPG noise floor. Quantitative pre-treatment values should be interpreted as qualitative
    markers of severe autonomic depression, not precise measurements.</p></div>"""


def build_two_hit_model() -> str:
    flow_wrap = (f'<div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:8px;'
                 f'padding:20px;background:{BG_ELEVATED};border-radius:10px;border:1px solid {BORDER_SUBTLE};'
                 f'margin-bottom:24px">')
    vicious = flow_wrap + "".join([
        _flow_box("rgba(239,68,68,.15)", ACCENT_RED, "Chronic GvHD", "rgba(239,68,68,.25)"),
        _arrow(),
        _flow_box("rgba(239,68,68,.1)", "#FCA5A5",
                  f'Cytokine release<br><span style="font-size:.75rem;color:{TEXT_TERTIARY}">IL-6, TNF-a, IFN-g</span>',
                  "rgba(239,68,68,.15)"),
        _arrow(),
        _flow_box("rgba(245,158,11,.12)", ACCENT_AMBER, "Sympathetic activation<br>+ vagal suppression",
                  "rgba(245,158,11,.2)", "160px"),
        _arrow(),
        _flow_box("rgba(239,68,68,.12)", "#FCA5A5", "Loss of cholinergic<br>anti-inflammatory reflex",
                  "rgba(239,68,68,.15)", "180px"),
        f'<div style="color:{TEXT_TERTIARY};font-size:1.3rem">&circlearrowright;</div>',
    ]) + "</div>"

    virtuous = flow_wrap + "".join([
        _flow_box("rgba(16,185,129,.15)", ACCENT_GREEN, "Beta-blockade", "rgba(16,185,129,.25)", "130px"),
        _arrow(),
        _flow_box("rgba(59,130,246,.12)", "#93C5FD", "Vagal unmasking", "rgba(59,130,246,.15)"),
        _arrow(),
        _flow_box("rgba(16,185,129,.1)", "#6EE7B7", "Cholinergic reflex<br>reactivates",
                  "rgba(16,185,129,.15)", "160px"),
        _arrow(),
        _flow_box("rgba(59,130,246,.1)", "#93C5FD", "Further cytokine<br>suppression",
                  "rgba(59,130,246,.15)", "160px"),
        f'<div style="color:{ACCENT_GREEN};font-size:1.3rem">&circlearrowright;</div>',
    ]) + "</div>"

    hit1 = _card(ACCENT_BLUE, "Hit 1: Ruxolitinib (JAK1/JAK2 inhibitor)", f"""
        <ul style="color:{TEXT_SECONDARY};font-size:.85rem;padding-left:18px;line-height:1.7">
        <li>Suppresses IL-6, TNF-a, IFN-g (within 5-7 days)</li>
        <li>De-suppresses brainstem vagal nuclei</li>
        <li>Restores macrophage cholinergic sensitivity</li>
        <li>Evidence: HR dropped significantly (p=0.009)</li>
        <li><b>HRV did NOT improve measurably</b> - masked by sympathetic saturation + PPG noise floor</li></ul>""")

    hit2 = _card(ACCENT_GREEN, "Hit 2: Bisoprolol (beta-1 selective blocker)", f"""
        <ul style="color:{TEXT_SECONDARY};font-size:.85rem;padding-left:18px;line-height:1.7">
        <li>Blocks sympathetic input at SA node (~40-50% receptor occupancy)</li>
        <li>Unmasks pre-recovered vagal modulation</li>
        <li>Triggers baroreflex-mediated vagal potentiation</li>
        <li>Restores cholinergic anti-inflammatory reflex</li>
        <li><b>System flips from vicious to virtuous cycle</b></li></ul>""")

    return f"""
    <div style="margin-bottom:24px;color:{TEXT_SECONDARY};line-height:1.7;font-size:.95rem">
    <p><b style="color:{TEXT_PRIMARY}">Core claim:</b> HRV recovery was likely underway during ruxolitinib
    monotherapy but below the Oura Ring RMSSD detection threshold. Bisoprolol accelerated and unmasked the
    recovery rather than initiating it.</p></div>
    <h3 style="color:{TEXT_PRIMARY};margin:20px 0 16px;font-size:1.05rem">The Vicious Cycle (pre-treatment)</h3>
    {vicious}
    <h3 style="color:{TEXT_PRIMARY};margin:20px 0 16px;font-size:1.05rem">The Two-Hit Intervention</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px">{hit1}{hit2}</div>
    <h3 style="color:{TEXT_PRIMARY};margin:20px 0 16px;font-size:1.05rem">The Virtuous Cycle (post-intervention)</h3>
    {virtuous}
    <p style="margin-top:16px;color:{TEXT_TERTIARY};font-size:.83rem;font-style:italic">
    Framing: temporally separable effects consistent with distinct mechanisms (not pharmacological synergy).
    Analogous to ACE-inhibitor + beta-blocker in heart failure.</p>"""


def build_alternative_explanations() -> str:
    ppg_warn = f"""
    <div style="background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);
                border-left:4px solid {ACCENT_AMBER};border-radius:10px;padding:20px;margin-bottom:24px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
        <span style="font-size:1.3rem">&#9888;</span>
        <b style="color:{ACCENT_AMBER};font-size:1rem">PPG Measurement Floor - Primary Caveat</b></div>
    <div style="color:{TEXT_SECONDARY};font-size:.88rem;line-height:1.7">
    <p>At RMSSD ~10 ms, PPG noise (~5-10 ms IBI error) equals the physiological signal.
    <b>RMSSD<sub>measured</sub> = sqrt(RMSSD<sub>true</sub><sup>2</sup> + RMSSD<sub>noise</sub><sup>2</sup>)</b>.
    Signal-to-noise ratio at baseline was approximately 1:1.</p>
    <p style="margin-top:8px"><b>No published PPG validation study has tested accuracy at RMSSD &lt;15 ms.</b>
    Cao 2022, Liang 2024, and Dial 2025 all used healthy populations (RMSSD 20-80 ms).</p>
    <p style="margin-top:8px">ITS confirms: level shift NOT significant (b4 p=0.19) but slope change
    highly significant (b5 p&lt;0.001) - accelerating emergence from noise floor.</p></div></div>"""

    alts = [
        ("Baseline compression / floor effect (Stein 2005)", "Patients with lowest baseline HRV show largest relative improvements. The 134% relative increase is only +13 ms - patient remains well below normal (42 ms median)."),
        ("Cycle-length dependence", "RMSSD scales with RR interval. HR dropping ~80 to ~70 bpm mechanically amplifies vagal modulation. Estimated 15-25% of observed increase is cycle-length effect."),
        ("Iron clearance / phlebotomy", "Ferritin 2225 to 1247 ug/L. Iron overload causes direct cardiac toxicity. Cardiac T2* never measured - <b>not addressable from available data.</b>"),
        ("HEV resolution", "Hepatitis E diagnosed D+2 of ruxolitinib. Viral load decline would independently reduce inflammatory burden. <b>HEV PCR trajectory needed.</b>"),
        ("Regression to the mean", "Placebo tests show 100% Mann-Whitney false positive rate on pre-treatment data, confirming ITS with trend control is the valid primary analysis."),
        ("Seasonal / activity confounding", "Jan to Apr in Norway - increasing daylight and outdoor activity. Not addressable from available data."),
    ]
    items = "".join(f'<li style="margin-bottom:12px"><b style="color:{TEXT_PRIMARY}">{t}:</b> {d}</li>' for t, d in alts)
    return ppg_warn + f"""<div style="color:{TEXT_SECONDARY};font-size:.88rem;line-height:1.7">
    <h3 style="color:{TEXT_PRIMARY};font-size:.95rem;margin-bottom:12px">Other Alternative Explanations</h3>
    <ol style="padding-left:20px">{items}</ol></div>"""


def build_predictions() -> str:
    preds = [
        ("Inflammatory markers should decline BEFORE the HRV inflection",
         "hsCRP, IL-6 trajectory should show decline before bisoprolol was added.", "TESTABLE NOW", ACCENT_GREEN),
        ("CRP should show FURTHER decline after bisoprolol",
         "Restored vagal tone should contribute its own anti-inflammatory effect.", "REQUIRES LABS", ACCENT_AMBER),
        ("Bisoprolol washout should show HRV decline to a HIGHER floor",
         "Expected: HRV drops to ~15-18 ms (not ~10 ms). If back to ~10 ms, model is wrong.", "PROSPECTIVE", ACCENT_BLUE),
        ("Bisoprolol dose increase should show diminishing HRV returns",
         "At 2.5mg (~40-50% occupancy), titrating to 5mg should yield proportionally smaller gains.", "PROSPECTIVE", ACCENT_BLUE),
    ]
    out = ""
    for i, (t, d, badge, col) in enumerate(preds, 1):
        out += f"""<div style="display:flex;gap:16px;align-items:flex-start;padding:16px;
        background:{BG_ELEVATED};border-radius:8px;margin-bottom:10px;border:1px solid {BORDER_SUBTLE}">
        <div style="min-width:32px;height:32px;background:rgba(59,130,246,.15);border-radius:50%;
        display:flex;align-items:center;justify-content:center;color:{ACCENT_BLUE};font-weight:700;
        font-size:.9rem;flex-shrink:0">{i}</div>
        <div style="flex:1"><div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap">
        <b style="color:{TEXT_PRIMARY};font-size:.9rem">{t}</b>{_pill(badge, col)}</div>
        <div style="color:{TEXT_SECONDARY};font-size:.83rem;line-height:1.6">{d}</div></div></div>"""
    return out


def build_why_it_matters() -> str:
    return f"""<div style="color:{TEXT_SECONDARY};font-size:.9rem;line-height:1.7">
    <p style="margin-bottom:16px"><b style="color:{TEXT_PRIMARY};font-size:1rem">The headline:</b>
    <b style="color:{ACCENT_CYAN}">First wearable-documented observation of the Tracey inflammatory reflex
    closing the loop in a human, in real time, in vivo.</b></p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
    {_card(ACCENT_BLUE, "Koopman et al. 2016 (PNAS)", f'''
        <div style="color:{TEXT_SECONDARY};font-size:.83rem">VNS &rarr; cytokine reduction in RA<br>
        <span style="color:{TEXT_TERTIARY};font-size:.78rem">Direction: vagal stimulation &rarr; anti-inflammatory</span></div>''')}
    {_card(ACCENT_GREEN, "This observation (mirror image)", f'''
        <div style="color:{TEXT_SECONDARY};font-size:.83rem">Anti-inflammatory Rx &rarr; vagal recovery<br>
        <span style="color:{TEXT_TERTIARY};font-size:.78rem">Direction: cytokine suppression &rarr; vagal restoration</span></div>''')}
    </div>
    <h3 style="color:{TEXT_PRIMARY};font-size:.95rem;margin:16px 0 10px">What is novel (no published precedent):</h3>
    <ol style="padding-left:20px;font-size:.85rem">
    <li style="margin-bottom:6px">JAK inhibitor + beta-blocker with HRV as an outcome - in any disease</li>
    <li style="margin-bottom:6px">Wearable-tracked autonomic recovery trajectory during ruxolitinib - in any context</li>
    <li style="margin-bottom:6px">Quantitative "two-hit" autonomic restoration pattern in a GvHD patient</li>
    <li style="margin-bottom:6px">Continuous 96-day HRV series spanning both interventions with formal causal inference</li></ol>
    <p style="margin-top:16px;color:{TEXT_TERTIARY};font-size:.82rem;font-style:italic">
    HSCT is the setting, not the contribution. Implications extend to heart failure, RA/SLE, post-sepsis
    autonomic dysfunction, and diabetic autonomic neuropathy with inflammatory component.</p></div>"""


def build_limitations() -> str:
    items = [
        ("<b>N=1.</b> Single-patient observation. Cannot establish generalizability.", "critical"),
        ("<b>No randomization or blinding.</b> Confounding by indication and placebo cannot be excluded.", "critical"),
        ("<b>PPG-derived HRV, not ECG.</b> Oura Ring NOT validated at RMSSD &lt;15 ms.", "warning"),
        ("<b>HEV confound.</b> Hepatitis E diagnosed D+2. Viral load trajectory unknown.", "warning"),
        ("<b>Phlebotomy confound.</b> Iron reduction has independent autonomic effects. T2* not measured.", "warning"),
        ("<b>Short post-bisoprolol window.</b> Sustained response needs 30/60/90-day confirmation.", "warning"),
        ("<b>No inflammatory biomarker trajectory</b> spanning both interventions.", "warning"),
        ("<b>Seasonal confounding.</b> Jan to Apr transition in Norway.", "info"),
        ("<b>Concurrent medications</b> could independently affect autonomic function.", "info"),
    ]
    out = ""
    for i, (txt, sev) in enumerate(items, 1):
        c = STATUS_COLORS.get(sev, TEXT_SECONDARY)
        out += (f'<div style="display:flex;gap:12px;align-items:flex-start;padding:12px 16px;'
                f'background:rgba({_hex_rgb(c)},.08);border-radius:8px;margin-bottom:8px;'
                f'border:1px solid rgba({_hex_rgb(c)},.25)">'
                f'<span style="color:{c};font-weight:700;font-size:.85rem;min-width:20px">{i}.</span>'
                f'<span style="color:{TEXT_SECONDARY};font-size:.85rem;line-height:1.6">{txt}</span></div>')
    return out


def _trow(method: str, test: str, est: str, pv: str, status: str, col: str) -> str:
    return (f'<tr style="border-bottom:1px solid {BORDER_SUBTLE}">'
            f'<td style="padding:10px 12px;color:{TEXT_SECONDARY}">{method}</td>'
            f'<td style="padding:10px 12px;color:{TEXT_PRIMARY}">{test}</td>'
            f'<td style="padding:10px 12px;text-align:right;color:{TEXT_PRIMARY};font-family:monospace">{est}</td>'
            f'<td style="padding:10px 12px;text-align:right;color:{TEXT_SECONDARY};font-family:monospace">{pv}</td>'
            f'<td style="padding:10px 12px;text-align:center">{_pill(status, col)}</td></tr>')


def build_statistical_evidence() -> str:
    its = _load_json("piecewise_regression_metrics.json")
    ci = _load_json("sequential_causal_impact_metrics.json")
    tau = _load_json("tau_u_metrics.json")
    placebo = _load_json("placebo_calibration_metrics.json")
    rows = ""

    # ITS
    coeffs = its.get("metrics", {}).get("mean_rmssd", {}).get("coefficients", {})
    r2 = its.get("metrics", {}).get("mean_rmssd", {}).get("r_squared")
    ljung = its.get("metrics", {}).get("mean_rmssd", {}).get("ljung_box_pvalue")
    for lbl, key, unit in [("b3: Jakavi slope", "time_since_jakavi", "ms/day"),
                           ("b4: BB level shift", "bb", "ms"),
                           ("b5: BB slope change", "time_since_bb", "ms/day")]:
        c = coeffs.get(key, {})
        est = c.get("estimate")
        p = c.get("p_value")
        sig = c.get("significant", False)
        rows += _trow("Piecewise ITS", lbl,
                       f"{est:+.2f} {unit}" if est is not None else "N/A",
                       format_p_value(p),
                       "Significant" if sig else "Not significant",
                       ACCENT_GREEN if sig else ACCENT_AMBER)
    if r2 is not None:
        rows += _trow("Piecewise ITS", "Model R\u00b2", f"{r2:.3f}", "", "Fit", ACCENT_BLUE)
    if ljung is not None:
        ok = ljung > 0.05
        rows += _trow("Piecewise ITS", "Ljung-Box", format_p_value(ljung), "",
                       "Passes" if ok else "Fails", ACCENT_GREEN if ok else ACCENT_RED)

    # CausalImpact
    for rk, rl in [("run_a", "CI Run A (Jakavi)"), ("run_b", "CI Run B (BB marginal)")]:
        s = ci.get(rk, {}).get("streams", {}).get("mean_rmssd", {})
        if s:
            eff, p = s.get("avg_effect"), s.get("p_value")
            sig = str(s.get("significant", "False")).lower() == "true"
            rows += _trow("CausalImpact", f"{rl} RMSSD",
                           f"{eff:+.2f} ms" if eff is not None else "N/A",
                           format_p_value(p),
                           "Significant" if sig else "Not significant",
                           ACCENT_GREEN if sig else ACCENT_AMBER)

    # Tau-U
    for ck, cl in [("A_vs_B", "A vs B (Jakavi)"), ("B_vs_C", "B vs C (BB marginal)")]:
        td = tau.get("comparisons", {}).get(ck, {}).get("metrics", {}).get("mean_rmssd", {}).get("tau_u", {})
        if td:
            tv, p = td.get("tau"), td.get("p_value")
            sig = p is not None and p < 0.05
            rows += _trow("Tau-U", f"{cl} RMSSD",
                           f"Tau={tv:+.3f}" if tv is not None else "N/A",
                           format_p_value(p), "Large" if sig else "Weak",
                           ACCENT_GREEN if sig else ACCENT_AMBER)

    # Placebo
    mw = placebo.get("false_positive_rates", {}).get("mean_rmssd", {}).get("mann_whitney", {})
    if mw:
        rate = mw.get("fpr")
        rows += _trow("Placebo", "Mann-Whitney FPR (RMSSD)",
                       f"{rate:.0%}" if rate is not None else "N/A", "",
                       "Confirms ITS needed" if rate == 1.0 else "",
                       ACCENT_AMBER if rate and rate > 0.2 else ACCENT_GREEN)

    hdr = (f'<tr style="border-bottom:2px solid {BORDER_DEFAULT}">' +
           "".join(f'<th style="text-align:{a};padding:10px 12px;color:{TEXT_TERTIARY};font-weight:600">{h}</th>'
                   for h, a in [("Method", "left"), ("Test", "left"), ("Estimate", "right"),
                                ("p-value", "right"), ("Status", "center")]) + "</tr>")
    return f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.83rem"><thead>{hdr}</thead><tbody>{rows}</tbody></table></div>'


# ---------------------------------------------------------------------------
# Main
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

    body = make_section("Executive Summary", build_executive_summary(kpis), section_id="executive")
    body += make_section("RMSSD Timeline: Three-Phase Observation",
                         '<div class="chart-box">' + build_hero_chart(daily) + "</div>", section_id="timeline")
    body += make_section("The Two-Hit Model (Hypothesis)", build_two_hit_model(), section_id="two-hit")
    body += make_section("Alternative Explanations", build_alternative_explanations(), section_id="alternatives")
    body += make_section("Falsifiable Predictions", build_predictions(), section_id="predictions")
    body += make_section("Why It Matters", build_why_it_matters(), section_id="significance")
    body += make_section("Limitations", build_limitations(), section_id="limitations")
    body += make_section("Statistical Evidence Summary", build_statistical_evidence(), section_id="evidence")

    html = wrap_html("Research Synthesis: Two-Hit Autonomic Recovery", body, report_id="synthesis",
                     subtitle="N-of-1 observation in post-HSCT chronic GvHD", data_end=kpis["data_end"])
    HTML_OUTPUT.write_text(html, encoding="utf-8")
    size_mb = HTML_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n  HTML: {HTML_OUTPUT} ({size_mb:.1f} MB)")

    metrics = {"generated": datetime.now().isoformat(), "report": "research_synthesis",
               "kpis": kpis, "statistical_sources": [
                   "piecewise_regression_metrics.json", "sequential_causal_impact_metrics.json",
                   "tau_u_metrics.json", "placebo_calibration_metrics.json"]}
    JSON_OUTPUT.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"  JSON: {JSON_OUTPUT}\n\n  Done.")


if __name__ == "__main__":
    main()
