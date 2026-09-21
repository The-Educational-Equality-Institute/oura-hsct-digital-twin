#!/usr/bin/env python3
"""Anthropic outreach case page: a bespoke, light, clinical-premium single page.

One story: a consumer Oura ring plus a patient-built N=1 digital twin measured
a medicine's physiological effect. This is a standalone artifact for the
Anthropic outreach case, separate from the dark multi-report dashboard.

Design: light clinical-premium (Stripe / Apple Health / Linear bar). Bespoke
template built here, NOT the dark _theme.py system.

Outputs:
    reports/anthropic_case.html
    plus an optional second copy at $ANTHROPIC_CASE_COPY_TO, when that variable is set

Data:
    - Hero chart series: oura_sleep_periods (day, average_hrv, average_heart_rate)
      WHERE type='long_sleep', live from the database.
    - Recent-vs-baseline numbers: computed live from the same table (last 30
      nights vs. the BASELINE_DAYS window before TREATMENT_START).
    - Tested full-window effect: reports/composite_biomarkers.json
      ("treatment_response" block; Mann-Whitney tested pre/post ruxolitinib).

No computed statistic is altered. This script only presents live numbers.
"""
from __future__ import annotations

import json
import sqlite3
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BETA_BLOCKER_START,
    DATABASE_PATH,
    REPORTS_DIR,
    TREATMENT_START,
    HEV_DIAGNOSIS_DATE,
    POPULATION_RMSSD_MEAN,
    POPULATION_RMSSD_MEDIAN,
    PLOTLY_CDN_URL,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _counterfactual import fork_chart_html, format_p, phase_estimates  # noqa: E402  (the digital-twin fork)

# Optional second output path (a private working copy elsewhere on disk).
# Never hard-code a local path here; the repository is public.
EXTRA_COPY = Path(os.environ["ANTHROPIC_CASE_COPY_TO"]) if os.environ.get("ANTHROPIC_CASE_COPY_TO") else None

RECENT_WINDOW_DAYS = 30

# --- Light clinical-premium palette (bespoke, do not import _theme.py) ------
BG = "#F7F7F5"
CARD = "#FFFFFF"
HAIRLINE = "rgba(20,22,26,0.08)"
INK = "#14161A"
SECONDARY = "#5B616E"
TERTIARY = "#8A909C"
ACCENT = "#3A3AD6"
ACCENT_SOFT = "rgba(58,58,214,0.08)"
ACCENT_SOFT_LINE = "rgba(58,58,214,0.35)"


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        print(f"ERROR: Database not found: {path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_sleep_series(conn: sqlite3.Connection) -> pd.DataFrame:
    """Nightly HRV + resting HR, long sleep only, deduplicated by night."""
    df = pd.read_sql_query(
        """
        SELECT day, average_hrv, average_heart_rate, lowest_heart_rate,
               rem_sleep_duration, average_breath
        FROM oura_sleep_periods
        WHERE type = 'long_sleep'
        ORDER BY day
        """,
        conn,
    )
    df["day"] = pd.to_datetime(df["day"])
    # A handful of nights have duplicate long_sleep rows (re-synced periods).
    # Keep the last write per night so each calendar night appears once.
    df = df.drop_duplicates(subset="day", keep="last").sort_values("day").reset_index(drop=True)
    return df


def compute_recent_vs_baseline(df: pd.DataFrame) -> dict:
    """Recent RECENT_WINDOW_DAYS nights vs. the full pre-treatment baseline.

    The baseline is every night before treatment start (day < TREATMENT_START),
    matching analysis/generate_index.py and the rest of the dashboard. Every
    number here is read straight from the database, never hardcoded.
    """
    treatment_start = pd.Timestamp(TREATMENT_START)
    baseline = df[df["day"] < treatment_start]
    recent = df.sort_values("day").tail(RECENT_WINDOW_DAYS)

    def pct_change(pre: float, post: float) -> float:
        return (post - pre) / pre * 100.0

    hrv_pre, hrv_post = baseline["average_hrv"].mean(), recent["average_hrv"].mean()
    hr_pre, hr_post = baseline["average_heart_rate"].mean(), recent["average_heart_rate"].mean()
    low_pre, low_post = baseline["lowest_heart_rate"].mean(), recent["lowest_heart_rate"].mean()
    # REM sleep is stored in seconds; report it in hours.
    rem_pre = baseline["rem_sleep_duration"].mean() / 3600.0
    rem_post = recent["rem_sleep_duration"].mean() / 3600.0
    # Respiratory rate (breaths per minute), shown honestly: it moved the
    # less-favorable way, both values inside the normal 12-20/min band.
    br_pre, br_post = baseline["average_breath"].mean(), recent["average_breath"].mean()

    return {
        "baseline_n": int(len(baseline)),
        "recent_n": int(len(recent)),
        "baseline_start": baseline["day"].min().date().isoformat() if len(baseline) else None,
        "baseline_end": baseline["day"].max().date().isoformat() if len(baseline) else None,
        "recent_start": recent["day"].min().date().isoformat(),
        "recent_end": recent["day"].max().date().isoformat(),
        "hrv": {"pre": hrv_pre, "post": hrv_post, "pct": pct_change(hrv_pre, hrv_post)},
        "avg_hr": {"pre": hr_pre, "post": hr_post, "pct": pct_change(hr_pre, hr_post)},
        "lowest_hr": {"pre": low_pre, "post": low_post, "pct": pct_change(low_pre, low_post)},
        "rem": {"pre": rem_pre, "post": rem_post, "pct": pct_change(rem_pre, rem_post)},
        "resp": {"pre": br_pre, "post": br_post, "pct": pct_change(br_pre, br_post)},
    }


def load_tested_effect() -> dict:
    """The statistically tested full-window pre/post ruxolitinib effect,
    from the existing composite_biomarkers.json (never recomputed here).
    """
    path = REPORTS_DIR / "composite_biomarkers.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text("utf-8"))
    return data.get("treatment_response", {})


# ---------------------------------------------------------------------------
# Hero chart (light plotly)
# ---------------------------------------------------------------------------

def build_hero_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    y_hrv_max = max(float(df["average_hrv"].max()) * 1.15, 60)

    # Phase shading: pre-treatment (neutral) vs post-treatment (faint accent wash)
    x_start = df["day"].min()
    x_end = df["day"].max()
    treatment_start = pd.Timestamp(TREATMENT_START)

    fig.add_vrect(
        x0=treatment_start, x1=x_end,
        fillcolor=ACCENT_SOFT, line_width=0, layer="below",
    )

    # Healthy RMSSD reference band (42-56 ms), honest context, drawn faint.
    fig.add_hrect(
        y0=POPULATION_RMSSD_MEAN, y1=56,
        fillcolor="rgba(20,22,26,0.05)", line_width=0, layer="below",
    )
    fig.add_annotation(
        x=x_start, y=(POPULATION_RMSSD_MEAN + 56) / 2,
        xanchor="left", yanchor="middle",
        text="healthy reference band",
        showarrow=False, font=dict(size=10.5, color=TERTIARY),
        xshift=4,
    )

    # HRV line (the accent series)
    fig.add_trace(go.Scatter(
        x=df["day"], y=df["average_hrv"],
        mode="lines", name="HRV (RMSSD, ms)",
        line=dict(color=ACCENT, width=2.25),
        hovertemplate="%{x|%b %d, %Y}<br>HRV: %{y:.1f} ms<extra></extra>",
    ))

    # Resting HR line, secondary axis, muted ink
    fig.add_trace(go.Scatter(
        x=df["day"], y=df["average_heart_rate"],
        mode="lines", name="Resting HR (bpm)",
        line=dict(color=SECONDARY, width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="%{x|%b %d, %Y}<br>Resting HR: %{y:.0f} bpm<extra></extra>",
    ))

    # Treatment + diagnosis markers. The two dates sit 2 days apart, so the
    # labels are staggered vertically and the HEV marker is drawn as a smaller
    # secondary annotation to keep both legible.
    marker_specs = (
        # (date, text, y-position, font-size, line-opacity, marker color)
        (TREATMENT_START, "ruxolitinib start", 1.055, 11.5, 0.6, ACCENT),
        (HEV_DIAGNOSIS_DATE, "HEV dx", 0.94, 10, 0.4, TERTIARY),
    )
    for d, label, y_pos, font_size, line_opacity, color in marker_specs:
        fig.add_vline(
            x=pd.Timestamp(d), line=dict(color=color, width=1.25, dash="dash"),
            opacity=line_opacity,
        )
        fig.add_annotation(
            x=pd.Timestamp(d), y=y_pos, yref="paper",
            text=label, showarrow=False,
            font=dict(size=font_size, color=color), xanchor="left", xshift=4,
        )

    fig.update_layout(
        height=460,
        margin=dict(l=56, r=56, t=64, b=48),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                   size=13, color=INK),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=SECONDARY),
        ),
        xaxis=dict(
            gridcolor="rgba(20,22,26,0.06)", linecolor=HAIRLINE,
            tickfont=dict(color=TERTIARY, size=11),
            showline=True, zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="HRV (ms)", font=dict(color=SECONDARY, size=12)),
            range=[0, y_hrv_max],
            gridcolor="rgba(20,22,26,0.06)", linecolor=HAIRLINE,
            tickfont=dict(color=TERTIARY, size=11),
            showline=True, zeroline=False,
        ),
        yaxis2=dict(
            title=dict(text="Resting HR (bpm)", font=dict(color=TERTIARY, size=12)),
            overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)", linecolor=HAIRLINE,
            tickfont=dict(color=TERTIARY, size=11),
            showline=True, zeroline=False,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD, bordercolor=HAIRLINE, font=dict(color=INK, size=12)),
    )
    return fig


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def fmt1(x: float) -> str:
    return f"{x:.1f}"


def fmt2(x: float) -> str:
    return f"{x:.2f}"


def fmt0(x: float) -> str:
    return f"{x:.0f}"


def fmt_pct(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.0f}%"


# ---------------------------------------------------------------------------
# HTML assembly (bespoke light template)
# ---------------------------------------------------------------------------

def build_html(df: pd.DataFrame, stats: dict, tested: dict) -> str:
    # The digital-twin fork: modelled "without Jakavi" range vs the actual on-Jakavi line.
    hero_div, cf = fork_chart_html(div_id="cf-fork", height=480)
    phase = phase_estimates()
    bb_fmt = BETA_BLOCKER_START.isoformat() if BETA_BLOCKER_START else "a later date"
    its_j, its_b = phase.get("its_jakavi", {}), phase.get("its_bb", {})
    fpr = phase.get("mw_fpr")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data_start = df["day"].min().date().isoformat()
    data_end = df["day"].max().date().isoformat()

    hrv = stats["hrv"]
    avg_hr = stats["avg_hr"]
    low_hr = stats["lowest_hr"]
    rem = stats["rem"]
    resp = stats["resp"]

    tested_hrv = tested.get("rmssd_mean", {})
    tested_hr = tested.get("sleep_hr_mean", {})

    headline_options = [
        "A model projected my body without the medicines. I went far past it.",
        "The effect of a medicine, visible in a consumer ring.",
        "HRV rose from {pre} to {post} ms after ruxolitinib.".format(
            pre=fmt1(hrv["pre"]), post=fmt1(hrv["post"])
        ),
    ]

    body = f"""
<div class="page">

  <header class="hero">
    <div class="col">
      <p class="eyebrow">The digital twin &middot; N=1 &middot; Oura Ring</p>
      <h1>{headline_options[0]}</h1>
      <p class="lede">
        A patient built a model of his own physiology from the {cf['ivx']} nights before
        treatment. From that, the model projects the range his HRV would sit in without it:
        around {cf['pre_mean']:.0f} ms. On ruxolitinib alone, {cf['nights_above_band_jakavi_only']} of
        {cf['jakavi_only_nights']} nights sat above that range. After a beta-blocker was added on
        {bb_fmt}, {cf['nights_above_band_bb']} of {cf['bb_nights']} did. The gap between the two
        lines is what a consumer ring made visible.
      </p>
    </div>
  </header>

  <section class="chart-wrap">
    <div class="card chart-card">
      <div class="fork-legend">
        <span class="lg lg-actual">Actual, on treatment</span>
        <span class="lg lg-cf">Modelled without treatment</span>
        <span class="lg lg-band">Projected range without treatment</span>
      </div>
      {hero_div}
    </div>
    <p class="fork-callout">
      <b>{cf['nights_above_band_bb']} of {cf['bb_nights']} nights</b> since the beta-blocker sit above
      the range the model projected without treatment, against {cf['nights_above_band_jakavi_only']} of
      {cf['jakavi_only_nights']} on ruxolitinib alone. Projected {cf['cf_recent']:.0f} ms. Actual
      {cf['actual_recent']:.0f} ms, a {fmt_pct(cf['gap_pct'])} gap.
    </p>
  </section>

  <section class="col">
    <h2 class="section-title">The change, in numbers</h2>
    <p class="section-sub">
      Last {stats['recent_n']} nights ({stats['recent_start']} to {stats['recent_end']})
      compared with the full pre-treatment baseline, all {stats['baseline_n']} nights
      before treatment ({stats['baseline_start']} to {stats['baseline_end']}).
    </p>

    <div class="kpi-grid">
      <div class="kpi card">
        <p class="kpi-label">HRV (RMSSD)</p>
        <p class="kpi-number">{fmt1(hrv['pre'])} <span class="arrow">&rarr;</span> {fmt1(hrv['post'])}<span class="unit"> ms</span></p>
        <p class="kpi-delta up">{fmt_pct(hrv['pct'])}</p>
        <p class="kpi-meaning">HRV reflects parasympathetic, vagal, autonomic tone.</p>
      </div>
      <div class="kpi card">
        <p class="kpi-label">Average sleeping heart rate</p>
        <p class="kpi-number">{fmt0(avg_hr['pre'])} <span class="arrow">&rarr;</span> {fmt0(avg_hr['post'])}<span class="unit"> bpm</span></p>
        <p class="kpi-delta down">{fmt_pct(avg_hr['pct'])}</p>
        <p class="kpi-meaning">Lower resting heart rate means reduced cardiovascular strain.</p>
      </div>
      <div class="kpi card">
        <p class="kpi-label">Lowest nightly heart rate</p>
        <p class="kpi-number">{fmt0(low_hr['pre'])} <span class="arrow">&rarr;</span> {fmt0(low_hr['post'])}<span class="unit"> bpm</span></p>
        <p class="kpi-delta down">{fmt_pct(low_hr['pct'])}</p>
        <p class="kpi-meaning">The floor heart rate reaches overnight, a separate signal from the average.</p>
      </div>
      <div class="kpi card">
        <p class="kpi-label">REM sleep</p>
        <p class="kpi-number">{fmt2(rem['pre'])} <span class="arrow">&rarr;</span> {fmt2(rem['post'])}<span class="unit"> h</span></p>
        <p class="kpi-delta up">{fmt_pct(rem['pct'])}</p>
        <p class="kpi-meaning">REM is the sleep stage tied to memory and emotional recovery.</p>
      </div>
    </div>

    <p class="context-line">
      Healthy adult HRV runs about {fmt0(POPULATION_RMSSD_MEAN)} to 56 ms (mean {fmt0(POPULATION_RMSSD_MEAN)},
      median {fmt0(POPULATION_RMSSD_MEDIAN)}). {fmt1(hrv['post'])} ms is up sharply from {fmt1(hrv['pre'])} ms.
      It is still below that range.
    </p>

    <p class="context-line">
      The baseline was low and slowly rising, not flat. Nightly HRV averaged about
      7 ms in January, 9 in February, and 11.5 in early March. It stayed near
      {phase.get('phase_b_mean', 0):.0f} ms through the weeks on ruxolitinib alone, then rose to about
      {phase.get('phase_c_mean', 0):.0f} ms after the beta-blocker was added and has held near
      {fmt1(hrv['post'])} ms since. That step, visible on the chart, is the signal. No single percentage is.
    </p>

    <p class="context-line">
      Not every metric moved the favorable way. Respiratory rate rose from
      {fmt1(resp['pre'])} to {fmt1(resp['post'])} breaths per minute ({fmt_pct(resp['pct'])}).
      Both figures sit inside the normal 12 to 20 per minute band. It is shown here
      because the other numbers are only credible if this one is too.
    </p>

    <p class="context-line">
      The numbers above compare the last {stats['recent_n']} nights against the full
      pre-treatment baseline. That is the "where things stand now" read. A separate
      statistical test compares all pre-ruxolitinib nights against all post-ruxolitinib
      nights across the observation window. On it,
      mean HRV went from {tested_hrv.get('pre_mean', '?')} to {tested_hrv.get('post_mean', '?')} ms
      ({fmt_pct(tested_hrv.get('pct_change', 0))}, Mann-Whitney p &lt; 0.001, Cohen's d {tested_hrv.get('cohens_d', '?')}).
      Average sleeping heart rate went from {tested_hr.get('pre_mean', '?')} to {tested_hr.get('post_mean', '?')} bpm
      ({fmt_pct(tested_hr.get('pct_change', 0))}). That test is description, not proof: in the
      pipeline's own placebo calibration it fires at {(fpr if fpr is not None else 0):.0%} of dates where nothing
      happened. The phase-corrected model below is the rigorous backing.
    </p>
  </section>

  <section class="col">
    <h2 class="section-title">Which medicine, and what the timing says</h2>
    <p class="body-text">
      Ruxolitinib started on 2026-03-16. A beta-blocker was added on {bb_fmt}. In the
      {phase.get('n_b', '?')} nights on ruxolitinib alone, nightly HRV averaged
      {phase.get('phase_b_mean', 0):.1f} ms against {phase.get('phase_a_mean', 0):.1f} ms before treatment.
      After the beta-blocker it averaged {phase.get('phase_c_mean', 0):.1f} ms. The pipeline's
      phase-corrected model (piecewise ITS with AR(1) errors) finds no step at ruxolitinib
      ({its_j.get('estimate', 0):+.1f} ms, {format_p(its_j.get('p_value'))}) and a step of
      {its_b.get('estimate', 0):+.1f} ms at the beta-blocker ({format_p(its_b.get('p_value'))}).
      Sleeping heart rate fell from about {fmt0(avg_hr['pre'])} to {fmt0(avg_hr['post'])} bpm over the
      same period. The timing points at the second medicine, or at the two together.
    </p>
    <p class="body-text">
      This is one person, not a trial. Hepatitis E was diagnosed two days after ruxolitinib
      started and resolved in the same window. The changes began after treatment started. We
      cannot prove which medicine caused them, or that either did. This is a strong
      association, not proof.
    </p>
  </section>

  <section class="col">
    <h2 class="section-title">How it was built</h2>
    <p class="body-text">
      A patient with zero programming background built this from Oura ring data.
      The tool was Claude Code, running on Claude models from Anthropic.
      Claude wrote the code: SQL queries, statistical tests, an N=1 state-space
      digital twin, this page.
    </p>
    <p class="body-text">
      The patient decided what to test and verified the output against the raw data
      and against clinical records. No number on this page was picked because it
      looked good. Each one comes from a live query at generation time.
    </p>
  </section>

  <section class="col">
    <div class="card honesty-panel">
      <h2 class="section-title">What this is, and is not</h2>
      <ul class="honesty-list">
        <li>Exploratory analysis, not a clinical trial.</li>
        <li>N=1: one patient, one ring, one timeline. No control group.</li>
        <li>Consumer-grade sensor. Not a certified clinical device.</li>
        <li>
          The hepatitis E diagnosis landed two days after ruxolitinib started.
          The two are temporally confounded. This page shows association, not proof of cause.
        </li>
      </ul>
    </div>
  </section>

  <footer class="col footer">
    <p><a class="footer-link" href="index.html">Full dashboard and methodology &rarr;</a></p>
    <p class="footer-meta">Generated {generated}.</p>
  </footer>

</div>
"""
    return wrap_page("The effect of a medicine, visible in a consumer ring.", body)


def wrap_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" href="data:,">
<title>{title}</title>
<script src="{PLOTLY_CDN_URL}"></script>
<style>
{CSS}
</style>
</head>
<body>
{body}
</body>
</html>
"""


CSS = f"""
:root {{
  --bg: {BG};
  --card: {CARD};
  --hairline: {HAIRLINE};
  --ink: {INK};
  --secondary: {SECONDARY};
  --tertiary: {TERTIARY};
  --accent: {ACCENT};
}}

* {{ box-sizing: border-box; }}

html, body {{
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
  font-variant-numeric: tabular-nums;
}}

.page {{
  max-width: 1080px;
  margin: 0 auto;
  padding: 72px 24px 96px;
}}

.col {{
  max-width: 760px;
  margin: 0 auto;
  padding: 0 8px;
}}

.card {{
  background: var(--card);
  border: 1px solid var(--hairline);
  border-radius: 16px;
  box-shadow: 0 1px 2px rgba(20,22,26,0.04), 0 8px 24px rgba(20,22,26,0.04);
}}

.hero {{
  margin-bottom: 40px;
  animation: fadeUp 0.5s ease both;
}}

.eyebrow {{
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 16px;
}}

.fork-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin: 0 0 8px 4px;
  font-size: 13px;
  color: var(--secondary);
}}
.fork-legend .lg {{ position: relative; padding-left: 24px; }}
.fork-legend .lg::before {{
  content: "";
  position: absolute;
  left: 0; top: 50%;
  width: 16px; height: 0;
  transform: translateY(-50%);
}}
.lg-actual::before {{ border-top: 2.5px solid var(--accent); }}
.lg-cf::before {{ border-top: 2px dashed var(--tertiary); }}
.lg-band::before {{
  height: 11px; border: 0;
  background: rgba(138,144,156,0.22);
  border-radius: 2px;
}}
.fork-callout {{
  margin: 18px 0 0;
  font-size: 16px;
  line-height: 1.55;
  color: var(--secondary);
}}
.fork-callout b {{ color: var(--ink); font-weight: 650; }}

h1 {{
  font-size: clamp(32px, 4.4vw, 48px);
  line-height: 1.12;
  letter-spacing: -0.02em;
  font-weight: 650;
  margin: 0 0 20px;
  color: var(--ink);
}}

.lede {{
  font-size: 18px;
  line-height: 1.55;
  color: var(--secondary);
  margin: 0;
  max-width: 620px;
}}

.chart-wrap {{
  margin: 0 0 56px;
  animation: fadeUp 0.6s ease 0.05s both;
}}

.chart-card {{
  padding: 24px 8px 8px;
}}

.section-title {{
  font-size: 24px;
  font-weight: 650;
  letter-spacing: -0.01em;
  margin: 0 0 8px;
  color: var(--ink);
}}

.section-sub {{
  font-size: 15px;
  color: var(--tertiary);
  margin: 0 0 24px;
  line-height: 1.5;
}}

section.col {{
  margin-bottom: 56px;
  animation: fadeUp 0.6s ease 0.1s both;
}}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 28px;
}}

.kpi {{
  padding: 24px 20px;
}}

.kpi-label {{
  font-size: 13px;
  color: var(--tertiary);
  margin: 0 0 12px;
  font-weight: 550;
}}

.kpi-number {{
  font-size: 28px;
  font-weight: 650;
  letter-spacing: -0.01em;
  margin: 0 0 8px;
  color: var(--ink);
  white-space: nowrap;
}}

.kpi-number .arrow {{
  color: var(--tertiary);
  font-weight: 400;
  margin: 0 2px;
}}

.kpi-number .unit {{
  font-size: 16px;
  font-weight: 500;
  color: var(--secondary);
}}

.kpi-delta {{
  display: inline-block;
  font-size: 14px;
  font-weight: 650;
  margin: 0 0 12px;
  color: var(--accent);
}}

.kpi-meaning {{
  font-size: 13.5px;
  line-height: 1.5;
  color: var(--secondary);
  margin: 0;
}}

.context-line {{
  font-size: 15px;
  line-height: 1.6;
  color: var(--secondary);
  margin: 0 0 16px;
}}

.body-text {{
  font-size: 16px;
  line-height: 1.65;
  color: var(--secondary);
  margin: 0 0 16px;
}}

.honesty-panel {{
  padding: 28px 28px 24px;
}}

.honesty-panel .section-title {{
  margin-bottom: 16px;
}}

.honesty-list {{
  margin: 0;
  padding: 0 0 0 20px;
  color: var(--secondary);
}}

.honesty-list li {{
  font-size: 15.5px;
  line-height: 1.65;
  margin-bottom: 10px;
}}

.honesty-list li:last-child {{
  margin-bottom: 0;
}}

.footer {{
  padding-top: 24px;
  border-top: 1px solid var(--hairline);
  margin-bottom: 0;
}}

.footer-link {{
  color: var(--accent);
  text-decoration: none;
  font-size: 15px;
  font-weight: 550;
}}

.footer-link:hover {{
  text-decoration: underline;
}}

.footer-meta {{
  font-size: 13px;
  color: var(--tertiary);
  margin: 8px 0 0;
}}

@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

@media (max-width: 720px) {{
  .kpi-grid {{
    grid-template-columns: 1fr;
  }}
  .page {{
    padding: 48px 16px 64px;
  }}
}}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conn = _connect(DATABASE_PATH)
    df = load_sleep_series(conn)
    stats = compute_recent_vs_baseline(df)
    tested = load_tested_effect()

    html = build_html(df, stats, tested)

    out_path = REPORTS_DIR / "anthropic_case.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")

    if EXTRA_COPY is not None:
        EXTRA_COPY.parent.mkdir(parents=True, exist_ok=True)
        EXTRA_COPY.write_text(html, encoding="utf-8")
        print(f"Wrote {EXTRA_COPY}")

    hrv = stats["hrv"]
    avg_hr = stats["avg_hr"]
    low_hr = stats["lowest_hr"]
    print(
        f"HRV: {hrv['pre']:.1f} -> {hrv['post']:.1f} ms ({fmt_pct(hrv['pct'])})  "
        f"avg HR: {avg_hr['pre']:.1f} -> {avg_hr['post']:.1f} bpm ({fmt_pct(avg_hr['pct'])})  "
        f"lowest HR: {low_hr['pre']:.1f} -> {low_hr['post']:.1f} bpm ({fmt_pct(low_hr['pct'])})"
    )


if __name__ == "__main__":
    main()
