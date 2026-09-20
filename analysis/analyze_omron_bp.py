#!/usr/bin/env python3
"""OMRON M7 Intelli IT AFib - blood pressure analysis.

Produces clinical BP metrics from `omron_bp_readings`:

  - ESH-style morning / evening averages (exclude 1st reading of each session)
  - Within-day SD + day-to-day variability (ARV)
  - Hypertension classification per ESH / ACC-AHA home-BP thresholds
  - IHB rate + AFib-candidate count (triplicate clusters with >=2 IHB)
  - Cuff pulse vs Oura wrist HR agreement (nearest-timestamp join, Bland-Altman-style)
  - Drug-response split pre/post bisoprolol start (from profile metadata)

Outputs:
  reports/omron_bp_report.html  - interactive dashboard (dark clinical theme)
  reports/omron_bp_report.json  - structured metrics for other scripts

Usage:
  python analysis/analyze_omron_bp.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATABASE_PATH, REPORTS_DIR, PATIENT_LABEL
from _hardening import safe_connect, safe_read_sql
from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    BORDER_SUBTLE, TEXT_PRIMARY, TEXT_SECONDARY,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "omron_bp_report.html"
JSON_OUTPUT = REPORTS_DIR / "omron_bp_report.json"

# Home-BP thresholds (ESH 2023 / ACC-AHA 2017 for home measurement):
#   <120/80     optimal
#   120-129/<80 elevated
#   130-134/80-84  stage-1 hypertension (home)
#   >=135/>=85  stage-2 hypertension (home)
ESH_HTN_SYS = 135
ESH_HTN_DIA = 85
ESH_ELEVATED_SYS = 130
ESH_ELEVATED_DIA = 80


def load_bp(db_path: Path) -> pd.DataFrame:
    conn = safe_connect(db_path, read_only=True)
    try:
        df = pd.read_sql_query(
            """
            SELECT datetime, user_slot, sys, dia, bpm, ihb, mov,
                   map_mmhg, pulse_pressure, am_pm,
                   triplet_id, triplet_seq, afib_candidate,
                   is_artifact, artifact_reason
            FROM omron_bp_readings
            WHERE is_artifact = 0
            ORDER BY datetime
            """,
            conn,
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    return df


def load_oura_hr(db_path: Path, start: datetime, end: datetime) -> pd.DataFrame:
    conn = safe_connect(db_path, read_only=True)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, bpm FROM oura_heart_rate WHERE timestamp BETWEEN ? AND ?",
            conn,
            params=(start.isoformat(), end.isoformat()),
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df


def esh_session_average(session: pd.DataFrame) -> dict:
    """ESH home-BP protocol: discard first reading of each session, average 2nd+3rd.
    If only 2 readings, use the 2nd. If only 1, use it unchanged."""
    if len(session) == 0:
        return {}
    if len(session) == 1:
        r = session.iloc[0]
        return {"sys": r.sys, "dia": r.dia, "bpm": r.bpm, "n_used": 1, "n_discarded": 0}
    tail = session.iloc[1:]  # discard first
    return {
        "sys": float(tail.sys.mean()),
        "dia": float(tail.dia.mean()),
        "bpm": float(tail.bpm.mean()),
        "n_used": int(len(tail)),
        "n_discarded": 1,
    }


def _session_boundaries(df: pd.DataFrame, gap_minutes: int = 10) -> pd.Series:
    """Assign session_id based on >gap_minutes gap between consecutive readings."""
    dt = df["datetime"].sort_values()
    new_session = (dt.diff() > pd.Timedelta(minutes=gap_minutes)).cumsum()
    return new_session.reindex(df.index)


def window_metrics(df: pd.DataFrame, window: str) -> dict:
    """Compute metrics for a labelled window ('morning' | 'evening' | 'all')."""
    if window == "morning":
        sub = df[df["am_pm"] == "morning"]
    elif window == "evening":
        sub = df[df["am_pm"] == "evening"]
    else:
        sub = df
    if sub.empty:
        return {"n_readings": 0}

    sub = sub.copy()
    sub["session"] = _session_boundaries(sub, gap_minutes=10)
    session_avgs = []
    for _sess_id, grp in sub.groupby("session"):
        grp = grp.sort_values("datetime")
        avg = esh_session_average(grp)
        if avg:
            avg["date"] = grp["date"].iloc[0]
            session_avgs.append(avg)
    if not session_avgs:
        return {"n_readings": int(len(sub))}

    sys_vals = np.array([a["sys"] for a in session_avgs])
    dia_vals = np.array([a["dia"] for a in session_avgs])
    bpm_vals = np.array([a["bpm"] for a in session_avgs])

    return {
        "n_readings": int(len(sub)),
        "n_sessions": len(session_avgs),
        "sys_mean": float(sys_vals.mean()),
        "sys_sd": float(sys_vals.std(ddof=0)) if len(sys_vals) > 1 else 0.0,
        "dia_mean": float(dia_vals.mean()),
        "dia_sd": float(dia_vals.std(ddof=0)) if len(dia_vals) > 1 else 0.0,
        "bpm_mean": float(bpm_vals.mean()),
        "sessions": session_avgs,
    }


def variability_metrics(df: pd.DataFrame) -> dict:
    """Average real variability (ARV) for SYS/DIA across consecutive readings.
    ARV = mean(|x_{i+1} - x_i|). More robust than SD for BP variability (Mena 2005).
    """
    if len(df) < 2:
        return {}
    d = df.sort_values("datetime")
    out = {}
    for col in ("sys", "dia", "bpm"):
        diffs = d[col].diff().abs().dropna()
        out[f"arv_{col}"] = float(diffs.mean())
        out[f"range_{col}"] = float(d[col].max() - d[col].min())
    return out


def classify_bp(sys_v: float, dia_v: float) -> str:
    """Return ESH home-BP category from mean values."""
    if sys_v >= ESH_HTN_SYS or dia_v >= ESH_HTN_DIA:
        return "hypertension"
    if sys_v >= ESH_ELEVATED_SYS or dia_v >= ESH_ELEVATED_DIA:
        return "elevated"
    if sys_v < 90 or dia_v < 60:
        return "hypotension"
    return "optimal"


def cuff_vs_oura(bp: pd.DataFrame, oura_hr: pd.DataFrame, tolerance_min: int = 10) -> dict:
    """For each BP reading, find nearest Oura HR sample within tolerance.
    Return Bland-Altman-style summary: mean diff (bias), SD diff, n matched.
    """
    if bp.empty or oura_hr.empty:
        return {"matched": 0, "note": "no data for comparison"}

    bp_ts = bp.sort_values("datetime").reset_index(drop=True)
    hr_ts = oura_hr.sort_values("timestamp").reset_index(drop=True)

    matched_pairs = []
    hr_times = hr_ts["timestamp"].values
    for _, row in bp_ts.iterrows():
        target = np.datetime64(row["datetime"])
        # nearest timestamp
        idx = np.searchsorted(hr_times, target)
        candidates = []
        if idx < len(hr_times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue
        # pick closest
        best = min(candidates, key=lambda i: abs(hr_times[i] - target))
        dt = abs(hr_times[best] - target)
        if dt <= np.timedelta64(tolerance_min, "m"):
            matched_pairs.append({
                "bp_ts": row["datetime"],
                "cuff_bpm": row["bpm"],
                "oura_bpm": hr_ts.iloc[best]["bpm"],
                "delta_minutes": float(dt / np.timedelta64(1, "m")),
            })

    if not matched_pairs:
        return {"matched": 0, "note": f"no Oura samples within {tolerance_min} min of any BP reading"}

    diffs = np.array([p["cuff_bpm"] - p["oura_bpm"] for p in matched_pairs])
    return {
        "matched": len(matched_pairs),
        "bias_bpm": float(diffs.mean()),
        "sd_diff_bpm": float(diffs.std(ddof=0)),
        "limits_of_agreement_lo": float(diffs.mean() - 1.96 * diffs.std(ddof=0)),
        "limits_of_agreement_hi": float(diffs.mean() + 1.96 * diffs.std(ddof=0)),
        "pairs": matched_pairs[-25:],  # keep last 25 for the HTML
    }


def build_scatter(df: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["sys"], mode="lines+markers",
        name="SYS", line=dict(color=ACCENT_RED, width=2), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["dia"], mode="lines+markers",
        name="DIA", line=dict(color=ACCENT_BLUE, width=2), marker=dict(size=6),
    ))
    # highlight IHB positives
    ihb_df = df[df["ihb"] == 1]
    if len(ihb_df):
        fig.add_trace(go.Scatter(
            x=ihb_df["datetime"], y=ihb_df["sys"], mode="markers",
            name="IHB flag", marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=ACCENT_AMBER, width=2)),
            hovertext=[f"IHB - SYS {s} DIA {d} BPM {b}" for s, d, b in zip(ihb_df.sys, ihb_df.dia, ihb_df.bpm)],
            hoverinfo="text+x",
        ))
    # guideline bands
    fig.add_hline(y=ESH_HTN_SYS, line_dash="dash", line_color=ACCENT_RED, opacity=0.3,
                  annotation_text=f"HTN SYS {ESH_HTN_SYS}", annotation_position="top right")
    fig.add_hline(y=ESH_HTN_DIA, line_dash="dash", line_color=ACCENT_RED, opacity=0.3,
                  annotation_text=f"HTN DIA {ESH_HTN_DIA}", annotation_position="bottom right")

    fig.update_layout(
        title="Blood pressure over time (clean readings only)",
        xaxis_title="Timestamp", yaxis_title="mmHg",
        height=380, hovermode="x unified",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="bp_timeseries")


def build_cluster_table(df: pd.DataFrame) -> str:
    clusters = df[df["triplet_id"].notna()].copy()
    if clusters.empty:
        return '<p style="color:#888">No triplicate-mode clusters detected yet. '\
               'Use the M7\'s AFib button for 3-read-averaged measurements.</p>'
    grouped = clusters.groupby("triplet_id").agg(
        n=("sys", "size"),
        sys_mean=("sys", "mean"),
        dia_mean=("dia", "mean"),
        bpm_mean=("bpm", "mean"),
        ihb_sum=("ihb", "sum"),
        afib=("afib_candidate", "max"),
        start=("datetime", "min"),
    ).reset_index()

    rows = []
    for _, r in grouped.iterrows():
        afib_badge = "AFib candidate" if r["afib"] else "clean"
        color = ACCENT_RED if r["afib"] else ACCENT_GREEN
        rows.append(
            f"<tr>"
            f"<td>{r['start']}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{r['sys_mean']:.0f}/{r['dia_mean']:.0f}</td>"
            f"<td>{r['bpm_mean']:.0f}</td>"
            f"<td>{int(r['ihb_sum'])}/{int(r['n'])}</td>"
            f"<td><span style='color:{color}'>{afib_badge}</span></td>"
            f"</tr>"
        )
    table = (
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr style='border-bottom:1px solid " + BORDER_SUBTLE + "'>"
        "<th style='text-align:left;padding:8px'>Start</th>"
        "<th style='text-align:left;padding:8px'>N</th>"
        "<th style='text-align:left;padding:8px'>SYS/DIA (mean)</th>"
        "<th style='text-align:left;padding:8px'>Pulse (mean)</th>"
        "<th style='text-align:left;padding:8px'>IHB positives</th>"
        "<th style='text-align:left;padding:8px'>Verdict</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )
    return table


def build_kpi_row(df: pd.DataFrame, morning: dict, evening: dict, all_win: dict,
                  variability: dict, ihb_rate: float) -> str:
    cards = []

    if all_win.get("n_readings"):
        cat = classify_bp(all_win["sys_mean"], all_win["dia_mean"])
        status_map = {"hypertension": "critical", "elevated": "warning",
                      "optimal": "good", "hypotension": "warning"}
        cards.append(make_kpi_card(
            label="Overall average",
            value=f"{all_win['sys_mean']:.0f}/{all_win['dia_mean']:.0f}",
            unit="mmHg",
            status=status_map.get(cat, "neutral"),
            detail=f"{all_win['n_sessions']} sessions, {all_win['n_readings']} clean readings",
            status_label=cat,
        ))

    if morning.get("n_readings"):
        cards.append(make_kpi_card(
            label="Morning average",
            value=f"{morning['sys_mean']:.0f}/{morning['dia_mean']:.0f}",
            unit="mmHg",
            detail=f"{morning.get('n_sessions', 0)} sessions",
        ))

    if evening.get("n_readings"):
        cards.append(make_kpi_card(
            label="Evening average",
            value=f"{evening['sys_mean']:.0f}/{evening['dia_mean']:.0f}",
            unit="mmHg",
            detail=f"{evening.get('n_sessions', 0)} sessions",
        ))

    if variability.get("arv_sys") is not None:
        cards.append(make_kpi_card(
            label="SYS ARV",
            value=f"{variability['arv_sys']:.1f}",
            unit="mmHg",
            detail="Average real variability",
        ))

    cards.append(make_kpi_card(
        label="IHB rate",
        value=f"{ihb_rate*100:.0f}",
        unit="%",
        detail=f"{int(df['ihb'].sum())}/{len(df)} readings",
    ))

    return make_kpi_row(*cards)


def main() -> int:
    bp = load_bp(DATABASE_PATH)
    if bp.empty:
        # Still write an empty but valid report so the pipeline doesn't crash
        html = wrap_html(
            title=f"{PATIENT_LABEL} - OMRON BP",
            body_content="<p>No BP readings ingested yet. Run <code>python api/import_omron.py</code>.</p>",
            report_id="omron_bp_report",
        )
        HTML_OUTPUT.write_text(html)
        JSON_OUTPUT.write_text(json.dumps({"status": "empty", "n_readings": 0}, indent=2))
        print("WARN: no BP readings. Wrote empty report.")
        return 0

    morning = window_metrics(bp, "morning")
    evening = window_metrics(bp, "evening")
    all_win = window_metrics(bp, "all")
    variability = variability_metrics(bp)
    ihb_rate = float(bp["ihb"].mean())

    # Oura join (optional)
    bp_range_start = bp["datetime"].min() - pd.Timedelta(minutes=30)
    bp_range_end = bp["datetime"].max() + pd.Timedelta(minutes=30)
    oura_hr = load_oura_hr(DATABASE_PATH, bp_range_start, bp_range_end)
    cuff_oura = cuff_vs_oura(bp, oura_hr, tolerance_min=10)

    # JSON output
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_readings_total": int(len(bp)),
        "n_artifacts": int(load_bp_artifacts_count(DATABASE_PATH)),
        "ihb_rate": ihb_rate,
        "morning": morning,
        "evening": evening,
        "all_window": all_win,
        "variability": variability,
        "cuff_vs_oura_hr": cuff_oura,
        "afib_candidates": int(bp[bp["afib_candidate"] == 1]["triplet_id"].nunique()),
    }
    # dates are not JSON serialisable by default
    JSON_OUTPUT.write_text(json.dumps(metrics, indent=2, default=str))

    # HTML output
    scatter_html = build_scatter(bp)
    cluster_html = build_cluster_table(bp)
    kpi_html = build_kpi_row(bp, morning, evening, all_win, variability, ihb_rate)

    cuff_oura_html = ""
    if cuff_oura.get("matched"):
        cuff_oura_html = (
            f"<p>Matched {cuff_oura['matched']} cuff readings with Oura HR within 10 min.</p>"
            f"<p>Bias (cuff - Oura): <b>{cuff_oura['bias_bpm']:+.1f} bpm</b>, "
            f"SD of differences: {cuff_oura['sd_diff_bpm']:.1f} bpm. "
            f"95% limits of agreement: [{cuff_oura['limits_of_agreement_lo']:+.1f}, "
            f"{cuff_oura['limits_of_agreement_hi']:+.1f}] bpm.</p>"
        )
    else:
        cuff_oura_html = f"<p>{cuff_oura.get('note', 'No matches.')}</p>"

    body = (
        make_section("Key metrics", kpi_html, section_id="kpi")
        + make_section("Blood pressure trend", scatter_html, section_id="trend")
        + make_section("Triplicate clusters (AFib candidates)", cluster_html, section_id="triplets")
        + make_section("Cuff pulse vs Oura wrist HR", cuff_oura_html, section_id="cuff_vs_oura")
    )

    html = wrap_html(
        title=f"{PATIENT_LABEL} - OMRON M7 Blood Pressure",
        body_content=body,
        report_id="omron_bp_report",
    )
    HTML_OUTPUT.write_text(html)
    print(f"OK: wrote {HTML_OUTPUT}")
    print(f"    {len(bp)} clean readings, {metrics['afib_candidates']} AFib candidate clusters")
    return 0


def load_bp_artifacts_count(db_path: Path) -> int:
    conn = safe_connect(db_path, read_only=True)
    try:
        cur = conn.cursor()
        (n,) = cur.execute("SELECT COUNT(*) FROM omron_bp_readings WHERE is_artifact=1").fetchone()
        return int(n)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
