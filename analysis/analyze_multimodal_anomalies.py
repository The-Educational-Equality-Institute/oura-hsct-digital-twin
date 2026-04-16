#!/usr/bin/env python3
"""Multi-modal anomaly detection via STUMPY matrix profiles.

For each modality column in frame_1d, compute a univariate matrix profile
at two pattern lengths (weekly, biweekly). Then compute a multivariate
matrix profile (mSTUMP) over the z-scored feature matrix to flag days
where several channels deviate jointly from their usual pattern.

Output:
  reports/multimodal_anomaly_report.html
  reports/multimodal_anomaly_metrics.json

Interpretation guide:
  * Per-column discords = a single modality doing something unusual for N days
  * Multivariate discords = a joint pattern that doesn't match anything seen
    before (captures co-movement anomalies, e.g. rising RHR + rising temp
    + lower HRV, which a univariate detector would miss)

The matrix profile distance is purely geometric - it tells you the week
in question is unlike any other week in the series. It does NOT tell you
whether that unlikeness is clinically important. Always cross-reference
with drug-start dates (ruxolitinib 2026-03-16, bisoprolol 2026-04-08)
before reading it as a flare signal.

Usage:
  python analysis/analyze_multimodal_anomalies.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import stumpy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import REPORTS_DIR, PATIENT_LABEL, TREATMENT_START  # noqa: E402
from _frames import load_frame, load_baselines, with_baseline_z  # noqa: E402
from _theme import (  # noqa: E402
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    ACCENT_BLUE, ACCENT_RED, ACCENT_AMBER,
    TEXT_SECONDARY,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "multimodal_anomaly_report.html"
JSON_OUTPUT = REPORTS_DIR / "multimodal_anomaly_metrics.json"

# Columns we care about for anomaly detection. These must be in frame_1d.
CORE_COLS = [
    "oura_rmssd_sleep_avg",
    "oura_hr_rest",
    "oura_hr_lowest",
    "oura_sleep_efficiency",
    "oura_spo2_avg",
    "oura_temp_delta",
    "bp_sys_mean",
    "bp_dia_mean",
    "bp_pulse_mean",
]

# Pattern lengths in days. 7 = weekly, 14 = biweekly.
# Multivariate uses m=7 for balance between sensitivity and context.
M_WEEKLY = 7
M_BIWEEKLY = 14
M_MULTI = 7

# Discord threshold: flag top 5% of matrix profile values (higher = more unusual)
DISCORD_PCTILE = 95


def _fill_short_gaps(s: pd.Series, max_gap: int = 2) -> pd.Series:
    """Linear-interpolate gaps up to max_gap consecutive NaNs, leave longer ones."""
    mask = s.isna()
    gap_ids = (~mask).cumsum()
    gap_sizes = mask.groupby(gap_ids).sum()
    fill_mask = mask & gap_ids.map(lambda i: gap_sizes.get(i, 0) <= max_gap)
    return s.where(~fill_mask, s.interpolate(method="linear"))


def univariate_profile(series: pd.Series, m: int) -> pd.Series | None:
    """STUMPY matrix profile. Returns None if series is too short or too sparse."""
    clean = _fill_short_gaps(series.dropna())
    if len(clean) < 2 * m + 5:
        return None
    # stumpy cannot handle NaN; drop them after gap-filling
    clean = clean.dropna()
    if len(clean) < 2 * m + 5:
        return None
    try:
        mp = stumpy.stump(clean.astype("float64").to_numpy(), m=m)
    except Exception:
        return None
    # mp[:, 0] is the matrix profile distance (higher = more unusual)
    profile = pd.Series(mp[:, 0], index=clean.index[: len(mp)])
    return profile


def multivariate_profile(df: pd.DataFrame, m: int) -> tuple[np.ndarray | None, pd.DatetimeIndex]:
    """stumpy.mstump multivariate matrix profile over z-scored features."""
    # Drop columns that are too sparse, fill small gaps
    usable = []
    for col in df.columns:
        s = _fill_short_gaps(df[col])
        if s.notna().sum() >= 2 * m + 5:
            usable.append(col)
    if len(usable) < 2:
        return None, df.index
    clean = df[usable].apply(_fill_short_gaps).dropna()
    if len(clean) < 2 * m + 5:
        return None, clean.index
    try:
        # mstump expects shape (d, n) - rows=dimensions, cols=time
        mp, _idx = stumpy.mstump(clean.to_numpy().T, m=m)
    except Exception:
        return None, clean.index
    # mp is (d, n-m+1). Use per-dimension profile + an all-dimensions sum
    return mp, clean.index[: mp.shape[1]]


def find_discords(profile: pd.Series, pctile: int = DISCORD_PCTILE,
                  max_return: int = 10) -> list[dict]:
    """Return top-N days by matrix profile distance, annotating each."""
    if profile is None or profile.empty:
        return []
    threshold = np.nanpercentile(profile.dropna(), pctile)
    flagged = profile[profile >= threshold].sort_values(ascending=False)
    out = []
    for idx, dist in flagged.head(max_return).items():
        out.append({
            "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
            "distance": float(dist),
        })
    return out


def build_timeseries_plot(df: pd.DataFrame, profiles: dict[str, pd.Series],
                          discord_threshold: dict[str, float]) -> str:
    """One plot per tracked column with its matrix profile underneath."""
    key_cols = ["oura_rmssd_sleep_avg", "oura_hr_rest", "oura_spo2_avg",
                "oura_temp_delta", "bp_sys_mean"]
    from plotly.subplots import make_subplots
    n = sum(1 for c in key_cols if c in df.columns and df[c].notna().any())
    if n == 0:
        return "<p style='color:#888'>No data available for anomaly plotting.</p>"
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[c for c in key_cols if c in df.columns],
                        vertical_spacing=0.04)
    row = 0
    for col in key_cols:
        if col not in df.columns or df[col].notna().sum() == 0:
            continue
        row += 1
        s = df[col].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=col,
            line=dict(color=ACCENT_BLUE, width=1.5),
            showlegend=False,
        ), row=row, col=1)
        # overlay discord markers
        prof = profiles.get(f"{col}__m7")
        if prof is not None and not prof.empty:
            thresh = discord_threshold.get(f"{col}__m7")
            if thresh is not None:
                flagged = prof[prof >= thresh]
                if not flagged.empty:
                    vals = s.reindex(flagged.index)
                    fig.add_trace(go.Scatter(
                        x=flagged.index, y=vals.values, mode="markers",
                        marker=dict(size=10, color=ACCENT_RED, symbol="diamond"),
                        name="discord", showlegend=False,
                    ), row=row, col=1)
    # Treatment marker
    fig.add_vline(x=pd.Timestamp(TREATMENT_START), line_dash="dash",
                  line_color=ACCENT_AMBER, opacity=0.4)
    fig.update_layout(height=180 * n, hovermode="x unified",
                      title="Per-channel trend with univariate matrix profile discords (red)")
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="mmanom_ts")


def build_multivariate_plot(mp: np.ndarray, idx: pd.DatetimeIndex,
                            cols: list[str]) -> str:
    """Heatmap of per-dimension mSTUMP profile + summed profile."""
    if mp is None:
        return "<p style='color:#888'>Not enough overlapping multimodal data yet for mSTUMP.</p>"
    fig = go.Figure()
    # Summed profile across all dimensions - the "joint anomaly" signal
    summed = mp.sum(axis=0)
    summed_z = (summed - np.nanmean(summed)) / (np.nanstd(summed) + 1e-9)
    fig.add_trace(go.Scatter(
        x=idx, y=summed_z, mode="lines+markers",
        name="joint discord score (z-scored sum)",
        line=dict(color=ACCENT_RED, width=2),
    ))
    fig.add_hline(y=2.5, line_dash="dash", line_color=ACCENT_AMBER, opacity=0.4,
                  annotation_text="threshold 2.5 sigma", annotation_position="top right")
    fig.add_vline(x=pd.Timestamp(TREATMENT_START), line_dash="dash",
                  line_color=ACCENT_AMBER, opacity=0.3,
                  annotation_text="rux start", annotation_position="top right")
    fig.update_layout(
        title=f"Multivariate discord score across {len(cols)} channels",
        height=340, hovermode="x unified",
        yaxis_title="z-scored joint MP distance",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="mmanom_multi")


def main() -> int:
    df = load_frame("1d")
    if df.empty:
        html = wrap_html(
            title=f"{PATIENT_LABEL} - Multimodal anomaly detection",
            body_content="<p>frame_1d is empty. Run <code>python analysis/_frames.py --rebuild</code>.</p>",
            report_id="multimodal_anomaly_report",
        )
        HTML_OUTPUT.write_text(html)
        JSON_OUTPUT.write_text(json.dumps({"status": "empty"}, indent=2))
        print("WARN: empty frame_1d")
        return 0

    # Filter to columns we actually have data for
    present = [c for c in CORE_COLS if c in df.columns and df[c].notna().sum() >= 20]
    if not present:
        print("WARN: no columns have >=20 observations, skipping anomaly detection")
        JSON_OUTPUT.write_text(json.dumps({
            "status": "insufficient_data",
            "n_rows": int(len(df)),
            "candidate_cols": CORE_COLS,
        }, indent=2, default=str))
        return 0

    # Univariate matrix profiles
    profiles: dict[str, pd.Series] = {}
    discords: dict[str, list[dict]] = {}
    thresholds: dict[str, float] = {}
    for col in present:
        for m, tag in [(M_WEEKLY, "m7"), (M_BIWEEKLY, "m14")]:
            key = f"{col}__{tag}"
            prof = univariate_profile(df[col], m)
            if prof is None:
                continue
            profiles[key] = prof
            thresh = float(np.nanpercentile(prof.dropna(), DISCORD_PCTILE))
            thresholds[key] = thresh
            discords[key] = find_discords(prof, DISCORD_PCTILE, max_return=5)

    # Multivariate on z-scored subset
    zdf = with_baseline_z(df[present], cols=present)
    z_cols = [f"{c}_z" for c in present if f"{c}_z" in zdf.columns]
    # If baselines were NaN, z columns won't exist - fall back to raw
    if not z_cols:
        z_cols = present
        feat = df[z_cols]
    else:
        feat = zdf[z_cols]
    mp_multi, mp_idx = multivariate_profile(feat, M_MULTI)

    joint_discords: list[dict] = []
    if mp_multi is not None:
        summed = mp_multi.sum(axis=0)
        summed_z = (summed - np.nanmean(summed)) / (np.nanstd(summed) + 1e-9)
        # Flag days where joint score > 2.5 sigma
        flagged_ix = np.where(summed_z >= 2.5)[0]
        for i in flagged_ix:
            joint_discords.append({
                "date": mp_idx[i].isoformat() if hasattr(mp_idx[i], "isoformat") else str(mp_idx[i]),
                "joint_z": float(summed_z[i]),
                "per_channel_contrib": {
                    z_cols[d]: float(mp_multi[d, i]) for d in range(mp_multi.shape[0])
                },
            })

    # --- HTML assembly
    kpi_cards = [
        make_kpi_card(
            label="Days tracked",
            value=str(len(df)),
            unit="d",
            detail=f"{df.index.min().date()} - {df.index.max().date()}",
        ),
        make_kpi_card(
            label="Channels profiled",
            value=str(len(present)),
            unit="",
            detail=f"of {len(CORE_COLS)} candidates",
        ),
        make_kpi_card(
            label="Univariate discords (m=7)",
            value=str(sum(len(v) for k, v in discords.items() if k.endswith("__m7"))),
            unit="",
            detail=f"top {100-DISCORD_PCTILE}% by MP distance",
        ),
        make_kpi_card(
            label="Joint multivariate discords",
            value=str(len(joint_discords)),
            unit="",
            detail="z-scored sum > 2.5 sigma",
            status="warning" if joint_discords else "good",
        ),
    ]

    # Discord table
    univ_rows = []
    for key, items in discords.items():
        for d in items:
            col, m = key.rsplit("__", 1)
            univ_rows.append(
                f"<tr><td>{d['date']}</td><td>{col}</td><td>{m}</td><td>{d['distance']:.2f}</td></tr>"
            )
    univ_table = (
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr><th style='text-align:left;padding:6px'>Date</th>"
        "<th style='text-align:left;padding:6px'>Channel</th>"
        "<th style='text-align:left;padding:6px'>Window</th>"
        "<th style='text-align:left;padding:6px'>MP distance</th></tr></thead>"
        f"<tbody>{''.join(univ_rows[:30])}</tbody></table>"
    )
    if not univ_rows:
        univ_table = "<p style='color:#888'>No univariate discords flagged yet.</p>"

    joint_table = "<p style='color:#888'>No joint multivariate discords flagged.</p>"
    if joint_discords:
        rows = []
        for d in joint_discords[:20]:
            top_contrib = max(d["per_channel_contrib"].items(), key=lambda x: x[1])
            rows.append(
                f"<tr><td>{d['date']}</td><td>{d['joint_z']:.2f}</td>"
                f"<td>{top_contrib[0]} ({top_contrib[1]:.2f})</td></tr>"
            )
        joint_table = (
            "<table style='width:100%;border-collapse:collapse'>"
            "<thead><tr><th style='text-align:left;padding:6px'>Date</th>"
            "<th style='text-align:left;padding:6px'>Joint z</th>"
            "<th style='text-align:left;padding:6px'>Top channel contributor</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    body = (
        make_section("Summary", make_kpi_row(*kpi_cards), section_id="kpi")
        + make_section("Per-channel trend with univariate discords",
                       build_timeseries_plot(df, profiles, thresholds), section_id="trend")
        + make_section("Multivariate joint discord score",
                       build_multivariate_plot(mp_multi, mp_idx, z_cols), section_id="multi")
        + make_section("Joint anomaly days",
                       joint_table, section_id="joint_table")
        + make_section("Top univariate discords (all channels, m=7 and m=14)",
                       univ_table, section_id="univ_table")
    )

    html = wrap_html(
        title=f"{PATIENT_LABEL} - Multimodal anomaly detection",
        body_content=body,
        report_id="multimodal_anomaly_report",
    )
    HTML_OUTPUT.write_text(html)

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_days": int(len(df)),
        "channels_profiled": present,
        "univariate_discords": discords,
        "joint_discords": joint_discords,
        "thresholds": thresholds,
    }
    JSON_OUTPUT.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"OK: wrote {HTML_OUTPUT}")
    print(f"    {len(df)} days, {len(present)} channels, "
          f"{sum(len(v) for v in discords.values())} univariate discords, "
          f"{len(joint_discords)} joint discords")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
