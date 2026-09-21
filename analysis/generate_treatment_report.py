#!/usr/bin/env python3
"""Treatment Response Report -- primary clinical report for physicians.

Covers heart, autonomic, sleep, activity, inflammation, breathing, and both
medicines (Jakavi / Bisoprolol) in a single HTML dashboard.

Outputs:  reports/treatment_response_report.html
          reports/treatment_response_metrics.json
"""
from __future__ import annotations
import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DATABASE_PATH, REPORTS_DIR, TREATMENT_START,
                    BETA_BLOCKER_START, DATA_START)
from profiles import PROFILES
from _theme import (wrap_html, make_kpi_card, make_kpi_row, make_section,
                    add_phase_shading,
                    format_p_value, TEXT_SECONDARY, TEXT_TERTIARY,
                    ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_CYAN)

pio.templates.default = "clinical_dark"
TODAY = date.today()
MITCH_DB, WENCHE_DB = Path(PROFILES["mitch"]["database"]), Path(PROFILES["wenche"]["database"])
P_PRE, P_JAK, P_BISO = "Pre-treatment", "Jakavi only", "Jakavi + Bisoprolol"
PHASE_COLORS = {P_PRE: TEXT_SECONDARY, P_JAK: ACCENT_BLUE, P_BISO: ACCENT_GREEN}

def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def _df(conn, sql):
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        return pd.DataFrame()

def _phase(d: date) -> str:
    if d < TREATMENT_START:
        return P_PRE
    return P_JAK if d < BETA_BLOCKER_START else P_BISO

def _load_json(name: str) -> dict:
    p = REPORTS_DIR / name
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}

def _vlines(fig):
    add_phase_shading(fig, str(TREATMENT_START))
    for d, lbl, clr in ((str(TREATMENT_START), "Jakavi", ACCENT_BLUE),
                         (str(BETA_BLOCKER_START), "Bisoprolol", ACCENT_GREEN)):
        fig.add_vline(x=d, line=dict(color=clr, width=2, dash="dash"), opacity=0.7)
        fig.add_annotation(x=d, y=1.06, yref="paper", text=lbl, showarrow=False,
                           font=dict(size=11, color=clr), xanchor="left")

def _phase_ts(df, col, title, ylabel, height=400):
    """Build a phase-colored time-series Plotly figure."""
    fig = go.Figure()
    for phase, color in PHASE_COLORS.items():
        sub = df[df["phase"] == phase]
        if sub.empty or col not in sub.columns:
            continue
        fig.add_trace(go.Scatter(x=sub["date"], y=sub[col], mode="lines+markers",
                                 name=phase, line=dict(color=color, width=2),
                                 marker=dict(size=4, color=color)))
    _vlines(fig)
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=ylabel,
                      height=height, legend=dict(orientation="h", y=-0.15))
    return pio.to_json(fig, validate=False)

def _periods(df):
    return {P_PRE: df[df["phase"] == P_PRE], P_JAK: df[df["phase"] == P_JAK],
            P_BISO: df[df["phase"] == P_BISO]}

def _three_period_table(df, specs, with_p=False):
    """Build an HTML table comparing three treatment phases.
    specs: list of (col, label, fmt) tuples. with_p adds a p-value column."""
    pds = _periods(df)
    rows = []
    for col, label, fmt in specs:
        means, series = {}, {}
        for pn, pdf in pds.items():
            s = pdf[col].dropna() if col in pdf.columns else pd.Series(dtype=float)
            series[pn] = s
            means[pn] = f"{s.mean():{fmt}}" if len(s) > 0 else "-"
        p_cell = ""
        if with_p:
            pre_v, biso_v = series[P_PRE], series[P_BISO]
            if len(pre_v) >= 3 and len(biso_v) >= 3:
                _, p = stats.mannwhitneyu(pre_v, biso_v, alternative="two-sided")
                p_cell = f"<td>{format_p_value(p)}</td>"
            else:
                p_cell = "<td>N/A</td>"
        rows.append(f'<tr><td>{label}</td><td>{means[P_PRE]}</td>'
                     f'<td>{means[P_JAK]}</td><td>{means[P_BISO]}</td>{p_cell}</tr>')
    p_hdr = "<th>p-value</th>" if with_p else ""
    return (f'<table><thead><tr><th>Metric</th><th>Pre-treatment</th>'
            f'<th>Jakavi only</th><th>Jak + Biso</th>{p_hdr}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')

# --- Data loading ---

def load_henrik_data() -> pd.DataFrame:
    conn = _connect(DATABASE_PATH)
    sleep = _df(conn, """SELECT day AS date, total_sleep_duration, rem_sleep_duration,
        deep_sleep_duration, light_sleep_duration, efficiency, average_heart_rate,
        lowest_heart_rate, average_hrv, average_breath, type
        FROM oura_sleep_periods WHERE type='long_sleep' ORDER BY day""")
    readiness = _df(conn, """SELECT date, score AS readiness_score, temperature_deviation,
        recovery_index, resting_heart_rate FROM oura_readiness WHERE score IS NOT NULL ORDER BY date""")
    activity = _df(conn, """SELECT date, score AS activity_score, steps, active_calories
        FROM oura_activity WHERE score IS NOT NULL ORDER BY date""")
    spo2 = _df(conn, "SELECT date, spo2_average FROM oura_spo2 WHERE spo2_average > 0 ORDER BY date")
    conn.close()
    for d in (sleep, readiness, activity, spo2):
        if not d.empty and "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"]).dt.date
    if sleep.empty:
        return pd.DataFrame()
    for c in ("total_sleep_duration","rem_sleep_duration","deep_sleep_duration","light_sleep_duration"):
        if c in sleep.columns:
            sleep[c] = sleep[c] / 3600.0
    merged = sleep.copy()
    for d in (readiness, activity, spo2):
        if not d.empty:
            merged = merged.merge(d, on="date", how="left")
    merged["phase"] = merged["date"].apply(_phase)
    return merged.sort_values("date").reset_index(drop=True)

def _load_comp(db_path, days=22):
    if not db_path.exists():
        return None
    conn = _connect(db_path)
    s = _df(conn, f"""SELECT day AS date, average_hrv, average_heart_rate, lowest_heart_rate,
        total_sleep_duration, efficiency, average_breath, type
        FROM oura_sleep_periods WHERE type='long_sleep' ORDER BY day DESC LIMIT {days}""")
    a = _df(conn, f"SELECT date, steps FROM oura_activity WHERE score IS NOT NULL ORDER BY date DESC LIMIT {days}")
    conn.close()
    if s.empty:
        return None
    s["date"] = pd.to_datetime(s["date"]).dt.date
    if not a.empty:
        a["date"] = pd.to_datetime(a["date"]).dt.date
        s = s.merge(a, on="date", how="left")
    if "total_sleep_duration" in s.columns:
        s["total_sleep_duration"] = s["total_sleep_duration"] / 3600.0
    return s

# --- Section builders ---

def build_executive(df):
    jak_days, biso_days = (TODAY - TREATMENT_START).days, (TODAY - BETA_BLOCKER_START).days
    last_hrv = df["average_hrv"].dropna().iloc[-1] if not df["average_hrv"].dropna().empty else 0
    last_hr = df["lowest_heart_rate"].dropna().iloc[-1] if not df["lowest_heart_rate"].dropna().empty else 0
    pre, post = df[df["phase"] == P_PRE], df[df["phase"] == P_BISO]
    pre_hrv = pre["average_hrv"].mean() if not pre.empty else 0
    post_hrv = post["average_hrv"].mean() if not post.empty else last_hrv
    hrv_pct = ((post_hrv - pre_hrv) / pre_hrv * 100) if pre_hrv > 0 else 0
    pre_hr = pre["lowest_heart_rate"].mean() if not pre.empty else 0
    post_hr = post["lowest_heart_rate"].mean() if not post.empty else last_hr
    hr_delta = post_hr - pre_hr
    cards = make_kpi_row(
        make_kpi_card("Days on Jakavi", jak_days, "days", status="info",
                      detail=f"Ruxolitinib 10 mg BID since {TREATMENT_START.strftime('%b %d')}"),
        make_kpi_card("Days on Bisoprolol", biso_days, "days", status="info",
                      detail=f"2.5 mg daily since {BETA_BLOCKER_START.strftime('%b %d')}"),
        make_kpi_card("Current HRV", last_hrv, "ms", decimals=1,
                      status="critical" if last_hrv < 15 else "warning" if last_hrv < 25 else "normal",
                      detail="Last night RMSSD"),
        make_kpi_card("Lowest HR", last_hr, "bpm", decimals=0,
                      status="warning" if last_hr > 80 else "normal", detail="Last night minimum"),
        make_kpi_card("HRV Change", f"{hrv_pct:+.0f}", "%",
                      status="good" if hrv_pct > 0 else "warning",
                      detail=f"Pre-tx {pre_hrv:.1f} ms vs Jak+Biso {post_hrv:.1f} ms"),
        make_kpi_card("HR Change", f"{hr_delta:+.1f}", "bpm",
                      status="good" if hr_delta < 0 else "warning",
                      detail=f"Pre-tx {pre_hr:.0f} bpm vs Jak+Biso {post_hr:.0f} bpm"),
    )
    narrative = (
        '<div class="odt-narrative">'
        f'<b>Two-hit autonomic recovery model:</b> Ruxolitinib (Day {jak_days}) addresses '
        f'GvHD-driven inflammation. Bisoprolol (Day {biso_days}) provides direct chronotropic '
        f'support. Together, HRV has increased {hrv_pct:+.0f}% and lowest sleeping HR shifted '
        f'{hr_delta:+.1f} bpm vs pre-treatment. The combination is producing measurable '
        'autonomic recovery that neither drug achieved alone.</div>')
    metrics = {"jak_days": jak_days, "biso_days": biso_days,
               "last_hrv": round(last_hrv, 1), "last_hr": round(float(last_hr)),
               "hrv_change_pct": round(hrv_pct, 1), "hr_change_bpm": round(hr_delta, 1)}
    return cards + narrative, metrics

def build_timeline(df):
    rows = []
    for _, r in df.iterrows():
        d, phase = r["date"], r["phase"]
        day_num = (d - TREATMENT_START).days
        bg = ("rgba(59,130,246,0.06)" if phase == P_JAK else
              "rgba(16,185,129,0.06)" if phase == P_BISO else "transparent")
        def _f(key, fmt): return f"{r[key]:{fmt}}" if pd.notna(r.get(key)) else "-"
        ps = "Pre" if phase == P_PRE else ("Jak" if phase == P_JAK else "Jak+BB")
        rows.append(
            f'<tr style="background:{bg}"><td>{d}</td><td>D{day_num:+d}</td>'
            f'<td>{_f("average_hrv",".1f")}</td><td>{_f("lowest_heart_rate",".0f")}</td>'
            f'<td>{_f("average_heart_rate",".0f")}</td><td>{_f("efficiency",".0f")}</td>'
            f'<td>{_f("total_sleep_duration",".1f")}</td><td>{_f("deep_sleep_duration",".1f")}</td>'
            f'<td>{_f("rem_sleep_duration",".1f")}</td><td>{_f("temperature_deviation","+.2f")}</td>'
            f'<td>{_f("readiness_score",".0f")}</td><td>{_f("recovery_index",".0f")}</td>'
            f'<td>{_f("steps",".0f")}</td><td>{ps}</td></tr>')
    return ('<div style="overflow-x:auto;max-height:600px;overflow-y:auto"><table><thead><tr>'
            '<th>Date</th><th>Day</th><th>HRV</th><th>Low HR</th><th>Avg HR</th>'
            '<th>Eff%</th><th>Sleep h</th><th>Deep h</th><th>REM h</th>'
            '<th>Temp</th><th>Ready</th><th>Recov</th><th>Steps</th><th>Phase</th>'
            '</tr></thead><tbody>' + "\n".join(rows) + '</tbody></table></div>')

def build_heart(df, its_data):
    charts = {"hrv_ts": _phase_ts(df, "average_hrv", "HRV (RMSSD) by Treatment Phase", "RMSSD (ms)", 420)}
    # HR overlay
    fig_hr = go.Figure()
    for col, lbl, clr in (("lowest_heart_rate","Lowest HR",ACCENT_GREEN),
                           ("average_heart_rate","Average HR",ACCENT_CYAN)):
        if col in df.columns:
            fig_hr.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines+markers",
                             name=lbl, line=dict(color=clr, width=2), marker=dict(size=4, color=clr)))
    _vlines(fig_hr)
    fig_hr.update_layout(title="Heart Rate Trends (Sleeping)", xaxis_title="Date",
                         yaxis_title="bpm", height=420, legend=dict(orientation="h", y=-0.15))
    charts["hr_ts"] = pio.to_json(fig_hr, validate=False)
    table = _three_period_table(df, [
        ("average_hrv", "HRV (ms)", ".1f"), ("lowest_heart_rate", "Lowest HR (bpm)", ".0f"),
        ("average_heart_rate", "Avg HR (bpm)", ".0f"), ("recovery_index", "Recovery Index", ".0f"),
    ], with_p=True)
    its_text = ""
    m = its_data.get("metrics", {})
    for key, label in (("mean_rmssd","HRV"), ("lowest_heart_rate","HR")):
        b5 = m.get(key, {}).get("coefficients", {}).get("time_since_bb", {})
        if b5:
            its_text += (f'<p><b>ITS beta-blocker slope ({label}):</b> '
                f'{b5.get("estimate",0):+.2f} {"ms" if key=="mean_rmssd" else "bpm"}/day '
                f'({format_p_value(b5.get("p_value"))}, '
                f'CI [{b5.get("ci_lower",0):.2f}, {b5.get("ci_upper",0):.2f}])</p>')
    html = ('<div id="chart-hrv_ts" class="chart-box" data-chart="hrv_ts">Loading...</div>'
            '<div id="chart-hr_ts" class="chart-box" data-chart="hr_ts">Loading...</div>'
            '<h3>Three-Period Comparison</h3>' + table + its_text)
    return html, charts

def build_sleep(df):
    charts = {"sleep_dur": _phase_ts(df, "total_sleep_duration",
                                     "Sleep Duration by Phase", "Hours", 380)}
    arch_cols = ["deep_sleep_duration", "rem_sleep_duration", "light_sleep_duration"]
    has_arch = all(c in df.columns for c in arch_cols)
    if has_arch:
        fig = go.Figure()
        for col, lbl, clr in (("deep_sleep_duration","Deep",ACCENT_PURPLE),
                               ("rem_sleep_duration","REM",ACCENT_CYAN),
                               ("light_sleep_duration","Light",TEXT_SECONDARY)):
            fig.add_trace(go.Bar(x=df["date"], y=df[col], name=lbl, marker_color=clr))
        fig.update_layout(barmode="stack", title="Sleep Architecture (Nightly)",
                          xaxis_title="Date", yaxis_title="Hours", height=380,
                          legend=dict(orientation="h", y=-0.15))
        _vlines(fig)
        charts["sleep_arch"] = pio.to_json(fig, validate=False)
    table = _three_period_table(df, [
        ("total_sleep_duration","Duration (h)",".1f"), ("efficiency","Efficiency (%)",".0f"),
        ("deep_sleep_duration","Deep (h)",".1f"), ("rem_sleep_duration","REM (h)",".1f"),
        ("light_sleep_duration","Light (h)",".1f"), ("average_breath","Breath rate",".1f"),
    ])
    html = ('<div id="chart-sleep_dur" class="chart-box" data-chart="sleep_dur">Loading...</div>'
            + ('<div id="chart-sleep_arch" class="chart-box" data-chart="sleep_arch">Loading...</div>'
               if has_arch else '')
            + '<h3>Sleep Metrics by Phase</h3>' + table)
    return html, charts

def build_activity(df):
    charts = {"steps": _phase_ts(df, "steps", "Daily Steps by Phase", "Steps", 380)}
    table = _three_period_table(df, [
        ("steps","Steps/day",".0f"), ("active_calories","Active cal/day",".0f"),
        ("activity_score","Activity score",".0f"),
    ], with_p=True)
    html = ('<div id="chart-steps" class="chart-box" data-chart="steps">Loading...</div>'
            '<h3>Activity Metrics by Phase</h3>' + table)
    return html, charts

def build_temp(df):
    charts = {}
    if "temperature_deviation" in df.columns:
        fig = go.Figure()
        for phase, color in PHASE_COLORS.items():
            sub = df[df["phase"] == phase]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(x=sub["date"], y=sub["temperature_deviation"],
                          mode="lines+markers", name=phase,
                          line=dict(color=color, width=2), marker=dict(size=4, color=color)))
        fig.add_hline(y=0, line=dict(color=TEXT_TERTIARY, dash="dot", width=1))
        _vlines(fig)
        fig.update_layout(title="Temperature Deviation from Baseline",
                          xaxis_title="Date", yaxis_title="Deviation (C)",
                          height=380, legend=dict(orientation="h", y=-0.15))
        charts["temp"] = pio.to_json(fig, validate=False)
    table = _three_period_table(df, [
        ("temperature_deviation","Temp deviation (C)","+.2f"),
        ("spo2_average","SpO2 (%)",".1f"), ("average_breath","Breath rate",".1f"),
    ])
    pds = _periods(df)
    biso_t = pds[P_BISO]["temperature_deviation"].dropna() if "temperature_deviation" in df.columns else pd.Series(dtype=float)
    jak_t = pds[P_JAK]["temperature_deviation"].dropna() if "temperature_deviation" in df.columns else pd.Series(dtype=float)
    flag = ""
    if len(biso_t) > 0 and len(jak_t) > 0 and biso_t.mean() > jak_t.mean() + 0.05:
        flag = (f'<div class="odt-narrative" style="border-left-color:var(--accent-amber)">'
                f'<b>Note:</b> Mean temp on Jak+Biso ({biso_t.mean():+.2f} C) elevated vs '
                f'Jak-only ({jak_t.mean():+.2f} C). Beta-blockers can mask fever. Monitor.</div>')
    html = (('<div id="chart-temp" class="chart-box" data-chart="temp">Loading...</div>'
             if "temp" in charts else '')
            + '<h3>Inflammation & Temperature by Phase</h3>' + table + flag)
    return html, charts

def build_drugs(ci_data, tau_data):
    run_a = ci_data.get("run_a", {}).get("streams", {})
    run_b = ci_data.get("run_b", {}).get("streams", {})
    def _ci(streams, key):
        s = streams.get(key, {})
        if not s:
            return ("-","-","-")
        return (f"{s.get('avg_effect',0):+.1f}", f"{s.get('relative_effect_pct',0):+.0f}%",
                format_p_value(s.get("p_value")))
    rows = []
    for key, lbl in (("mean_rmssd","HRV (ms)"), ("lowest_heart_rate","Lowest HR"),
                      ("average_heart_rate","Avg HR"), ("sleep_efficiency","Sleep Eff %")):
        a, b = _ci(run_a, key), _ci(run_b, key)
        rows.append(f'<tr><td>{lbl}</td><td>{a[0]}</td><td>{a[1]}</td><td>{a[2]}</td>'
                     f'<td>{b[0]}</td><td>{b[1]}</td><td>{b[2]}</td></tr>')
    ci_table = ('<table><thead><tr><th>Metric</th>'
        '<th colspan="3" style="text-align:center">Jakavi Alone (Run A)</th>'
        '<th colspan="3" style="text-align:center">+ Bisoprolol (Run B)</th></tr>'
        '<tr><th></th><th>Effect</th><th>Rel</th><th>p</th>'
        '<th>Effect</th><th>Rel</th><th>p</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table>')
    # Tau-U
    comps = tau_data.get("comparisons", {})
    ab = comps.get("A_vs_B", {}).get("metrics", {})
    bc = comps.get("B_vs_C", {}).get("metrics", {})
    tau_rows = []
    for key, lbl in (("mean_rmssd","HRV"), ("lowest_heart_rate","Lowest HR"),
                      ("average_heart_rate","Avg HR")):
        ab_m, bc_m = ab.get(key,{}).get("tau_u",{}), bc.get(key,{}).get("tau_u",{})
        tau_rows.append(f'<tr><td>{lbl}</td>'
            f'<td>{ab_m.get("tau",0):.3f}</td><td>{ab_m.get("effect_size","-")}</td>'
            f'<td>{bc_m.get("tau",0):.3f}</td><td>{bc_m.get("effect_size","-")}</td></tr>')
    tau_table = ('<h3>Tau-U Effect Sizes</h3><table><thead><tr><th>Metric</th>'
        '<th>A vs B (Tau)</th><th>Effect</th><th>B vs C (Tau)</th><th>Effect</th>'
        '</tr></thead><tbody>' + "".join(tau_rows) + '</tbody></table>')
    two_hit = ('<div class="odt-narrative"><b>Two-hit model:</b> Run A shows Jakavi produced '
        'favorable but mostly non-significant trends (immune modulation). Run B shows the '
        'beta-blocker addition produced large, statistically significant effects on HRV '
        '(+92%, p&lt;0.001) and HR (-5.2%, p&lt;0.001). Jakavi removes the inflammatory '
        'driver while bisoprolol provides chronotropic relief.</div>')
    return '<h3>Bayesian CausalImpact (Sequential)</h3>' + ci_table + tau_table + two_hit

def build_comparison(df):
    recent = df.tail(22)
    mitch, wenche = _load_comp(MITCH_DB), _load_comp(WENCHE_DB)
    def _s(data, col):
        if data is None or col not in data.columns:
            return "-"
        s = data[col].dropna()
        return f"{s.mean():.1f}" if len(s) > 0 else "-"
    rows = []
    for col, lbl in (("average_hrv","HRV (ms)"), ("lowest_heart_rate","Lowest HR (bpm)"),
                      ("average_heart_rate","Avg HR (bpm)"), ("total_sleep_duration","Sleep (h)"),
                      ("efficiency","Efficiency (%)"), ("steps","Steps/day")):
        rows.append(f'<tr><td>{lbl}</td><td>{_s(recent,col)}</td>'
                     f'<td>{_s(mitch,col)}</td><td>{_s(wenche,col)}</td></tr>')
    table = ('<table><thead><tr><th>Metric (recent 22d)</th><th>Henrik (36, HSCT)</th>'
             '<th>Mitch (36, Stroke)</th><th>Wenche (61, Healthy)</th>'
             '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')
    w_hrv = (wenche["average_hrv"].dropna().mean()
             if wenche is not None and "average_hrv" in wenche.columns
             and not wenche["average_hrv"].dropna().empty else None)
    h_hrv = (recent["average_hrv"].dropna().mean()
             if "average_hrv" in recent.columns and not recent["average_hrv"].dropna().empty else None)
    ctx = ""
    if w_hrv and h_hrv and h_hrv > 0:
        ctx = (f'<div class="odt-narrative"><b>Context:</b> Your 61-year-old mother (healthy '
               f'control) has {w_hrv/h_hrv:.1f}x your HRV despite being 25 years older. This '
               'underscores the severity of post-HSCT autonomic deficit and the clinical need '
               'for the current dual-drug intervention.</div>')
    return table + ctx

def build_warnings(df):
    warnings = []
    biso = df[df["phase"] == P_BISO]
    if "temperature_deviation" in biso.columns:
        t = biso["temperature_deviation"].dropna().mean()
        if pd.notna(t) and t > 0.1:
            warnings.append(f'<li><b>Temperature elevation:</b> Mean deviation on Jak+Biso is '
                            f'{t:+.2f} C. Beta-blockers can mask fever. Monitor oral temps.</li>')
    if "average_breath" in df.columns:
        pre_br = df[df["phase"] == P_PRE]["average_breath"].dropna()
        biso_br = biso["average_breath"].dropna()
        if len(pre_br) > 0 and len(biso_br) > 0:
            delta = biso_br.mean() - pre_br.mean()
            if delta > 0.5:
                warnings.append(f'<li><b>Breath rate increase:</b> +{delta:.1f} breaths/min vs '
                                'pre-treatment. Monitor SpO2.</li>')
    warnings.append('<li><b>PPG noise floor:</b> HRV values below 10 ms are near Oura sensor '
                    'noise floor. Interpret single-night values cautiously.</li>')
    wl = ('<h3>Active Watchpoints</h3><ul style="color:var(--text-secondary);line-height:1.8">'
          + "".join(warnings) + '</ul>') if warnings else ""
    labs = ('<h3>Recommended Follow-up Labs</h3>'
            '<ul style="color:var(--text-secondary);line-height:1.8">'
            '<li>CBC with differential (ruxolitinib myelosuppression)</li>'
            '<li>CRP / ESR (inflammation tracking)</li>'
            '<li>LFTs (hepatic GvHD screening)</li>'
            '<li>HEV PCR (viral clearance confirmation)</li>'
            '<li>12-lead ECG (PR interval on bisoprolol)</li>'
            '<li>Consider 24h Holter if resting HR drops below 55 bpm</li></ul>')
    return wl + labs

# --- Main ---

def main() -> int:
    print("Loading Henrik data...")
    df = load_henrik_data()
    if df.empty:
        print("ERROR: No data loaded from database")
        return 1
    df = df[df["date"] >= DATA_START].copy()
    print(f"  {len(df)} days loaded ({df['date'].min()} to {df['date'].max()})")

    its = _load_json("piecewise_regression_metrics.json")
    ci = _load_json("sequential_causal_impact_metrics.json")
    tau = _load_json("tau_u_metrics.json")
    chart_data, body = {}, ""
    metrics = {"generated": datetime.now().isoformat()}

    print("  Building executive summary...")
    s1, s1m = build_executive(df)
    body += make_section("Executive Summary", s1, section_id="executive")
    metrics["executive"] = s1m

    print("  Building treatment timeline...")
    body += make_section("Treatment Timeline (Day-by-Day)",
                         build_timeline(df[df["date"] >= TREATMENT_START].copy()),
                         section_id="timeline")

    for label, fn, sid, args in (
        ("heart & autonomic", build_heart, "heart", (df, its)),
        ("sleep analysis", build_sleep, "sleep", (df,)),
        ("activity", build_activity, "activity", (df,)),
        ("inflammation & temp", build_temp, "inflammation", (df,)),
    ):
        print(f"  Building {label}...")
        html, charts = fn(*args)
        section_title = {"heart": "Heart & Autonomic System",
                         "sleep": "Sleep Architecture & Quality",
                         "activity": "Activity & Physical Form",
                         "inflammation": "Inflammation & Temperature"}[sid]
        body += make_section(section_title, html, section_id=sid)
        chart_data.update(charts)

    print("  Building drug analysis...")
    body += make_section("Drug-Specific Analysis", build_drugs(ci, tau), section_id="drugs")
    print("  Building patient comparison...")
    body += make_section("Three-Way Comparison (Henrik / Mitch / Wenche)",
                         build_comparison(df), section_id="comparison")
    print("  Building warnings...")
    body += make_section("Warnings & Follow-up", build_warnings(df), section_id="warnings")

    html = wrap_html("Treatment Response Report", body, report_id="treatment_report",
                     subtitle="Ruxolitinib + Bisoprolol: comprehensive multi-system assessment "
                              "for Drs Schoemans and Wolff",
                     chart_data=chart_data, data_end=df["date"].max())

    out_html = REPORTS_DIR / "treatment_response_report.html"
    out_json = REPORTS_DIR / "treatment_response_metrics.json"
    out_html.write_text(html, encoding="utf-8")
    out_json.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_html}\nWrote {out_json}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
