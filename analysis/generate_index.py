#!/usr/bin/env python3
"""Generate the Dashboard homepage (index.html) for the Oura Digital Twin.

Rebuilt as a light, clinical-premium dashboard (Stripe / Apple Health / Linear
grade). Bespoke light template built inline here: it does NOT import the dark
get_base_css/wrap_html from _theme.py. Data helpers (REPORT_REGISTRY) are still
read from _theme.py, but every number on the page comes from a live query at
generation time against oura.db, or from reports/composite_biomarkers.json.

Sections:
    1. Compact meta line (day counts, ring, data-through), computed live.
    2. Hero: eyebrow + confident statement + the HRV headline number.
    3. Metric grid: 4 equal compact cards (HRV, avg sleeping HR, lowest HR, REM),
       each with before -> after, a big indigo % delta, and a sparkline.
    4. Drug-effect chart (light): nightly HRV + resting HR with the ruxolitinib
       marker. The centrepiece.
    5. One quiet tested-stat line (whole-period HRV effect) from composite JSON.
    6. Report directory (kept, restyled light).

No computed statistic is altered. Numbers are live; nothing is hardcoded.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import sys
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    BETA_BLOCKER_START,
    COMPANION_LABEL,
    COMPANION_URL,
    DATABASE_PATH,
    HEALTH_EQUITY_URL,
    PLOTLY_CDN_URL,
    POPULATION_RMSSD_MEAN,
    PROJECT_ROOT,
    REPO_URL,
    REPORTS_DIR,
    SITE_DESCRIPTION,
    SITE_NAME,
    TRANSPLANT_DATE,
    TREATMENT_START,
)
from _theme import REPORT_REGISTRY, site_head_meta  # noqa: E402  (data helpers only)
from _counterfactual import fork_chart_html, format_p, phase_estimates  # noqa: E402

# --- Light clinical-premium palette (bespoke, matches generate_anthropic_case) ---
BG = "#F7F7F5"
CARD = "#FFFFFF"
HAIRLINE = "rgba(20,22,26,0.08)"
INK = "#14161A"
SECONDARY = "#5B616E"
TERTIARY = "#666D7B"
ACCENT = "#3A3AD6"
ACCENT_SOFT = "rgba(58,58,214,0.08)"

RECENT_WINDOW_DAYS = 30
# The homepage draws scatter traces only, so the basic bundle (about a third of the
# full one) is enough. It is loaded with defer; the chart script below is a module,
# so it runs after the bundle in document order.
PLOTLY_INDEX_URL = PLOTLY_CDN_URL.replace("plotly-", "plotly-basic-")


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
    """Nightly HRV + heart-rate signals, long sleep only, one row per night."""
    df = pd.read_sql_query(
        """
        SELECT day, average_hrv, average_heart_rate, lowest_heart_rate,
               rem_sleep_duration
        FROM oura_sleep_periods
        WHERE type = 'long_sleep'
        ORDER BY day
        """,
        conn,
    )
    df["day"] = pd.to_datetime(df["day"])
    df = df.drop_duplicates(subset="day", keep="last").sort_values("day").reset_index(drop=True)
    return df


def compute_recent_vs_baseline(df: pd.DataFrame) -> dict:
    """Recent RECENT_WINDOW_DAYS nights vs. the full pre-treatment baseline.

    Baseline = every night before treatment start (day < TREATMENT_START),
    matching the rest of the dashboard. Every number here is read straight from
    the database, never hardcoded. Sparkline series are the recent-window
    nightly values, returned so each card can draw its own trend.
    """
    treatment_start = pd.Timestamp(TREATMENT_START)
    baseline = df[df["day"] < treatment_start]
    recent = df.sort_values("day").tail(RECENT_WINDOW_DAYS)

    def pct_change(pre: float, post: float) -> float:
        return (post - pre) / pre * 100.0

    def spark(series: pd.Series, scale: float = 1.0) -> list[float]:
        return [float(v) * scale for v in series.dropna().tolist()]

    hrv_pre, hrv_post = baseline["average_hrv"].mean(), recent["average_hrv"].mean()
    hr_pre, hr_post = baseline["average_heart_rate"].mean(), recent["average_heart_rate"].mean()
    low_pre, low_post = baseline["lowest_heart_rate"].mean(), recent["lowest_heart_rate"].mean()
    rem_pre = baseline["rem_sleep_duration"].mean() / 3600.0
    rem_post = recent["rem_sleep_duration"].mean() / 3600.0

    return {
        "baseline_n": int(len(baseline)),
        "recent_n": int(len(recent)),
        "baseline_start": baseline["day"].min().date().isoformat() if len(baseline) else None,
        "baseline_end": baseline["day"].max().date().isoformat() if len(baseline) else None,
        "recent_start": recent["day"].min().date().isoformat(),
        "recent_end": recent["day"].max().date().isoformat(),
        "hrv": {"pre": hrv_pre, "post": hrv_post, "pct": pct_change(hrv_pre, hrv_post),
                "spark": spark(recent["average_hrv"])},
        "avg_hr": {"pre": hr_pre, "post": hr_post, "pct": pct_change(hr_pre, hr_post),
                   "spark": spark(recent["average_heart_rate"])},
        "lowest_hr": {"pre": low_pre, "post": low_post, "pct": pct_change(low_pre, low_post),
                      "spark": spark(recent["lowest_heart_rate"])},
        "rem": {"pre": rem_pre, "post": rem_post, "pct": pct_change(rem_pre, rem_post),
                "spark": spark(recent["rem_sleep_duration"], 1.0 / 3600.0)},
    }


def load_tested_effect() -> dict:
    """The statistically tested whole-period pre/post ruxolitinib effect,
    from the existing composite_biomarkers.json (never recomputed here).
    """
    path = REPORTS_DIR / "composite_biomarkers.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text("utf-8"))
    return data.get("treatment_response", {})


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt0(x: float) -> str:
    return f"{x:.0f}"


def fmt1(x: float) -> str:
    return f"{x:.1f}"


def fmt2(x: float) -> str:
    return f"{x:.2f}"


def fmt_pct(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.0f}%"


def _fmt_date(iso: str) -> str:
    return date.fromisoformat(iso[:10]).strftime("%b %d %Y")


# ---------------------------------------------------------------------------
# Sparkline (inline SVG, indigo, no dependencies)
# ---------------------------------------------------------------------------

def sparkline_svg(values: list[float], *, width: int = 120, height: int = 30) -> str:
    """Compact indigo sparkline as inline SVG. Empty string if too few points."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return ""
    lo, hi = min(clean), max(clean)
    span = hi - lo or 1.0
    n = len(clean)
    pad = 2.0
    usable_h = height - 2 * pad
    pts = []
    for i, v in enumerate(clean):
        x = (i / (n - 1)) * (width - 2 * pad) + pad
        y = height - pad - ((v - lo) / span) * usable_h
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(pts)
    last_x, last_y = pts[-1].split(",")
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'preserveAspectRatio="none" aria-hidden="true">'
        f'<polyline fill="none" stroke="{ACCENT}" stroke-width="1.6" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{polyline}"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.1" fill="{ACCENT}"/>'
        '</svg>'
    )


# ---------------------------------------------------------------------------
# Hero drug-effect chart (light plotly)
# ---------------------------------------------------------------------------

def build_effect_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    x_end = df["day"].max()
    treatment_start = pd.Timestamp(TREATMENT_START)
    y_hrv_max = max(float(df["average_hrv"].max()) * 1.15, 45)

    # Post-treatment wash, faint accent.
    fig.add_vrect(
        x0=treatment_start, x1=x_end,
        fillcolor=ACCENT_SOFT, line_width=0, layer="below",
    )

    # Healthy RMSSD reference band, drawn faint for honest context.
    fig.add_hrect(
        y0=POPULATION_RMSSD_MEAN, y1=56,
        fillcolor="rgba(20,22,26,0.05)", line_width=0, layer="below",
    )
    fig.add_annotation(
        x=df["day"].min(), y=(POPULATION_RMSSD_MEAN + 56) / 2,
        xanchor="left", yanchor="middle",
        text="healthy reference band", showarrow=False,
        font=dict(size=10.5, color=TERTIARY), xshift=4,
    )

    # HRV line, the accent series.
    fig.add_trace(go.Scatter(
        x=df["day"], y=df["average_hrv"],
        mode="lines", name="HRV (RMSSD, ms)",
        line=dict(color=ACCENT, width=2.25),
        hovertemplate="%{x|%b %d, %Y}<br>HRV: %{y:.1f} ms<extra></extra>",
    ))

    # Resting HR, secondary axis, muted ink.
    fig.add_trace(go.Scatter(
        x=df["day"], y=df["average_heart_rate"],
        mode="lines", name="Resting HR (bpm)",
        line=dict(color=SECONDARY, width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="%{x|%b %d, %Y}<br>Resting HR: %{y:.0f} bpm<extra></extra>",
    ))

    # Ruxolitinib marker: the step at this line is the point of the chart.
    fig.add_vline(
        x=treatment_start, line=dict(color=ACCENT, width=1.25, dash="dash"),
        opacity=0.6,
    )
    fig.add_annotation(
        x=treatment_start, y=1.055, yref="paper",
        text="ruxolitinib start", showarrow=False,
        font=dict(size=11.5, color=ACCENT), xanchor="left", xshift=4,
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
# Light navigation (built from REPORT_REGISTRY, restyled light)
# ---------------------------------------------------------------------------

NAV_PRIMARY = ["weekly", "full_analysis", "comp_treatment", "piecewise_its"]


def build_nav() -> str:
    by_id = {r["id"]: r for r in REPORT_REGISTRY}
    links = []
    for rid in NAV_PRIMARY:
        r = by_id.get(rid)
        if not r:
            continue
        links.append(f'<a class="nav-link" href="{escape(r["file"])}">{escape(r["title"])}</a>')
    return (
        '<nav class="topnav">'
        '<a class="nav-brand" href="index.html">Oura Digital Twin</a>'
        f'<div class="nav-links">{"".join(links)}</div>'
        '</nav>'
    )


# ---------------------------------------------------------------------------
# Report directory (kept, restyled light)
# ---------------------------------------------------------------------------

def build_report_directory() -> str:
    groups: dict[str, list[dict]] = {}
    for report in REPORT_REGISTRY:
        if report["id"] == "home":
            continue
        groups.setdefault(report["group"], []).append(report)

    preferred = ["Core", "Clinical", "Advanced", "Comparative", "Statistical", "Individual", "Context"]
    ordered = [g for g in preferred if g in groups]
    ordered.extend(g for g in groups if g not in ordered)

    blocks = []
    for group in ordered:
        rows = []
        for report in groups[group]:
            desc = report.get("desc") or f"{report['title']} report."
            search = escape((report["title"] + " " + desc + " " + group).lower())
            rows.append(
                f'<a class="report-row" href="{escape(report["file"])}" data-report-row data-search="{search}">'
                f'<span class="report-title">{escape(report["title"])}</span>'
                f'<span class="report-desc">{escape(desc)}</span>'
                '</a>'
            )
        blocks.append(
            '<details class="report-group" open data-report-group>'
            f'<summary><span>{escape(group)}</span><em>{len(rows)}</em></summary>'
            f'<div class="report-rows">{"".join(rows)}</div>'
            '</details>'
        )

    return (
        '<section class="col directory">'
        '<div class="directory-head">'
        '<h2 class="section-title">All reports</h2>'
        '<input id="report-filter" type="search" placeholder="Filter by title, topic, or group">'
        '</div>'
        f'<div class="report-directory">{"".join(blocks)}</div>'
        '</section>'
    )


# ---------------------------------------------------------------------------
# HTML assembly (bespoke light template)
# ---------------------------------------------------------------------------

def build_html(df: pd.DataFrame, stats: dict, tested: dict, data_end: str) -> str:
    # The digital-twin fork: modelled "without Jakavi" range vs the actual on-Jakavi line.
    fork_div, cf = fork_chart_html(div_id="cf-fork", height=480)
    fork_div = re.sub(r'<script[^>]*>', '<script type="module">', fork_div, count=1)

    end = date.fromisoformat(data_end[:10])
    days_post_hsct = (end - TRANSPLANT_DATE).days
    days_on_rux = (end - TREATMENT_START).days + 1
    days_on_bb = (end - BETA_BLOCKER_START).days + 1 if BETA_BLOCKER_START else 0
    bb_fmt = BETA_BLOCKER_START.strftime("%B %d").replace(" 0", " ") if BETA_BLOCKER_START else ""
    phase = phase_estimates()

    hrv = stats["hrv"]
    avg_hr = stats["avg_hr"]
    low_hr = stats["lowest_hr"]
    rem = stats["rem"]

    tested_hrv = tested.get("rmssd_mean", {})
    tested_pct = tested_hrv.get("pct_change")
    tested_p = tested_hrv.get("mann_whitney_p")
    tested_p_txt = "p&lt;0.001" if (tested_p is not None and tested_p < 0.001) else (
        f"p={tested_p:.3f}" if tested_p is not None else "p=n/a"
    )
    tested_line = ""
    its_j, its_b = phase.get("its_jakavi", {}), phase.get("its_bb", {})
    if its_j.get("estimate") is not None and its_b.get("estimate") is not None:
        tested_line += (
            '<p class="tested-line">'
            'Phase-corrected estimate (piecewise ITS with AR(1) errors): '
            f'ruxolitinib step {its_j["estimate"]:+.1f} ms ({format_p(its_j.get("p_value"))}), '
            f'beta-blocker step {its_b["estimate"]:+.1f} ms ({format_p(its_b.get("p_value"))}).'
        )
        if phase.get("tau_ab") is not None and phase.get("tau_bc") is not None:
            tested_line += (
                f' Tau-U, trend-corrected: pre to ruxolitinib alone {phase["tau_ab"]:+.2f}; '
                f'ruxolitinib alone to plus beta-blocker {phase["tau_bc"]:+.2f}.'
            )
        tested_line += '</p>'
    if tested_pct is not None:
        fpr = phase.get("mw_fpr")
        fpr_txt = (
            f' In this pipeline\'s own placebo calibration that test fires at {fpr:.0%} of dates where nothing happened, '
            'so it is shown as description, not evidence.' if fpr is not None else ''
        )
        tested_line += (
            '<p class="tested-line">'
            f'Whole-period test: HRV {fmt_pct(tested_pct)} ({tested_p_txt}), '
            f'all pre-ruxolitinib nights vs all post-ruxolitinib nights.{fpr_txt} '
            '<a href="placebo_calibration.html">Placebo tests</a> &middot; '
            '<a href="piecewise_regression.html">Piecewise ITS</a> &middot; '
            '<a href="tau_u_effects.html">Tau-U</a>.'
            '</p>'
        )

    # Metric cards. Each: label, before -> after, big indigo delta, sparkline.
    cards = [
        {
            "label": "HRV (RMSSD)",
            "before": f"{fmt1(hrv['pre'])} <span class=\"arrow\">&rarr;</span> {fmt1(hrv['post'])}",
            "unit": "ms",
            "pct": hrv["pct"],
            "spark": sparkline_svg(hrv["spark"]),
            "meaning": "Parasympathetic, vagal, autonomic tone. Higher is better recovery.",
        },
        {
            "label": "Average sleeping HR",
            "before": f"{fmt0(avg_hr['pre'])} <span class=\"arrow\">&rarr;</span> {fmt0(avg_hr['post'])}",
            "unit": "bpm",
            "pct": avg_hr["pct"],
            "spark": sparkline_svg(avg_hr["spark"]),
            "meaning": "Lower resting heart rate means less cardiovascular strain.",
        },
        {
            "label": "Lowest nightly HR",
            "before": f"{fmt0(low_hr['pre'])} <span class=\"arrow\">&rarr;</span> {fmt0(low_hr['post'])}",
            "unit": "bpm",
            "pct": low_hr["pct"],
            "spark": sparkline_svg(low_hr["spark"]),
            "meaning": "The floor heart rate reaches overnight.",
        },
        {
            "label": "REM sleep",
            "before": f"{fmt2(rem['pre'])} <span class=\"arrow\">&rarr;</span> {fmt2(rem['post'])}",
            "unit": "h",
            "pct": rem["pct"],
            "spark": sparkline_svg(rem["spark"]),
            "meaning": "The sleep stage tied to memory and emotional recovery.",
        },
    ]
    card_html = "".join(
        '<div class="metric card">'
        f'<p class="metric-label">{escape(c["label"])}</p>'
        f'<p class="metric-before">{c["before"]}<span class="unit"> {escape(c["unit"])}</span></p>'
        f'<p class="metric-delta">{fmt_pct(c["pct"])}</p>'
        f'{c["spark"]}'
        f'<p class="metric-meaning">{escape(c["meaning"])}</p>'
        '</div>'
        for c in cards
    )

    hrv_pct_txt = fmt_pct(hrv["pct"])

    body = f"""
{build_nav()}
<div class="page">

  <p class="meta">Day {days_on_rux} on ruxolitinib &middot; day {days_on_bb} on a beta-blocker &middot; {days_post_hsct} days post-transplant &middot; Oura ring &middot; data through {_fmt_date(data_end)}</p>

  <p class="intro">I had a stem-cell transplant for leukaemia in {TRANSPLANT_DATE.strftime("%B %Y")}. This site reads my
    Oura ring every night and turns the data into the reports below. Claude Code wrote the code; I described what I
    needed in Norwegian and checked what came back. <a href="{REPO_URL}">Open source, MIT licensed</a>, part of
    <a href="{HEALTH_EQUITY_URL}">TEEI Health Equity</a>. Companion project: <a href="{COMPANION_URL}">{escape(COMPANION_LABEL)}</a>.
    <a href="how_built.html">How this was built</a>.</p>

  <header class="hero">
    <p class="eyebrow">The digital twin</p>
    <h1>A model projected my body without the medicines. I went far past it.</h1>
    <p class="lede">
      This twin learned my nightly physiology from the {cf['ivx']} nights before treatment and
      projects the range my HRV would sit in without it: around {cf['pre_mean']:.0f} ms. On
      ruxolitinib alone, {cf['nights_above_band_jakavi_only']} of {cf['jakavi_only_nights']} nights
      sat above that range. After a beta-blocker was added on {bb_fmt}, {cf['nights_above_band_bb']}
      of {cf['bb_nights']} did. The gap between the two lines is what a consumer ring made
      visible; the phase analysis below says which medicine carries it.
    </p>
  </header>

  <section class="chart-wrap">
    <div class="card chart-card">
      <div class="fork-legend">
        <span class="lg lg-actual">Actual, on treatment</span>
        <span class="lg lg-cf">Modelled without treatment</span>
        <span class="lg lg-band">Projected range without treatment</span>
      </div>
      {fork_div}
    </div>
    <p class="fork-callout">
      <b>{cf['nights_above_band_bb']} of {cf['bb_nights']} nights</b> since the beta-blocker sit above
      the range the twin projected without treatment, against {cf['nights_above_band_jakavi_only']} of
      {cf['jakavi_only_nights']} on ruxolitinib alone. Projected {cf['cf_recent']:.0f} ms. Actual
      {cf['actual_recent']:.0f} ms, a {fmt_pct(cf['gap_pct'])} gap.
    </p>
    {tested_line}
  </section>

  <section class="col">
    <h2 class="section-title">What moved, and by how much</h2>
    <p class="section-sub">Last {stats['recent_n']} nights vs the pre-treatment baseline. Every figure is a live query.</p>
    <div class="metric-grid">
      {card_html}
    </div>
  </section>

  {build_report_directory()}

  <footer class="col footer">
    <p class="disclaimer">N=1 &middot; Oura ring &middot; exploratory, not a clinical device.</p>
    <p class="disclaimer"><a href="{REPO_URL}">Source code on GitHub</a> &middot; Built with Claude Code &middot;
      <a href="{HEALTH_EQUITY_URL}">TEEI Health Equity</a> &middot; Companion project: <a href="{COMPANION_URL}">{escape(COMPANION_LABEL)}</a> &middot;
      <a href="how_built.html">How this was built</a> &middot; <a href="claims.html">Every number, checked</a> &middot;
      MIT License &middot; &copy; 2026 The Educational Equality Institute</p>
  </footer>

</div>
"""
    return wrap_page("Oura Digital Twin", body)


def wrap_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
{site_head_meta(SITE_NAME, SITE_DESCRIPTION, "")}
<title>{escape(title)}</title>
<script defer src="{PLOTLY_INDEX_URL}"></script>
<style>
{CSS}
</style>
</head>
<body>
{body}
<script>
{JS}
</script>
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

.topnav {{
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  height: 56px;
  padding: 0 24px;
  background: rgba(247,247,245,0.85);
  backdrop-filter: saturate(180%) blur(12px);
  border-bottom: 1px solid var(--hairline);
}}
.nav-brand {{
  font-size: 15px;
  font-weight: 650;
  letter-spacing: -0.01em;
  color: var(--ink);
  text-decoration: none;
  white-space: nowrap;
}}
.nav-links {{
  display: flex;
  gap: 4px;
  overflow-x: auto;
  scrollbar-width: none;
}}
.nav-links::-webkit-scrollbar {{ display: none; }}
.nav-link {{
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--secondary);
  text-decoration: none;
  white-space: nowrap;
  transition: background 120ms ease, color 120ms ease;
}}
.nav-link:hover {{
  background: var(--card);
  color: var(--ink);
}}

.page {{
  max-width: 1080px;
  margin: 0 auto;
  padding: 40px 24px 96px;
}}

.col {{
  max-width: 820px;
  margin: 0 auto;
  padding: 0 8px;
}}

.card {{
  background: var(--card);
  border: 1px solid var(--hairline);
  border-radius: 16px;
  box-shadow: 0 1px 2px rgba(20,22,26,0.04), 0 8px 24px rgba(20,22,26,0.04);
}}

.meta {{
  max-width: 820px;
  margin: 0 auto 32px;
  padding: 0 8px;
  font-size: 13px;
  color: var(--tertiary);
  letter-spacing: 0.01em;
}}

.intro {{
  max-width: 820px;
  margin: -16px auto 40px;
  padding: 0 8px;
  font-size: 15px;
  line-height: 1.6;
  color: var(--secondary);
}}
.intro a {{ color: var(--accent); text-decoration: none; border-bottom: 1px solid rgba(58,58,214,0.25); }}
.intro a:hover {{ border-bottom-color: var(--accent); }}
.disclaimer a {{ color: var(--secondary); text-decoration: none; border-bottom: 1px solid var(--hairline); }}
.disclaimer a:hover {{ color: var(--accent); }}
.disclaimer + .disclaimer {{ margin-top: 8px; }}

.hero {{
  max-width: 820px;
  margin: 0 auto 48px;
  padding: 0 8px;
}}

.eyebrow {{
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 16px;
}}

h1 {{
  font-size: clamp(34px, 4.6vw, 52px);
  line-height: 1.08;
  letter-spacing: -0.025em;
  font-weight: 680;
  margin: 0 0 24px;
  color: var(--ink);
}}

.hero-number {{
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 4px 16px;
  font-size: clamp(56px, 8vw, 84px);
  line-height: 1;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--accent);
  margin: 0 0 24px;
}}
.hero-unit {{
  font-size: 15px;
  font-weight: 500;
  letter-spacing: 0;
  color: var(--tertiary);
}}

.lede {{
  font-size: 18px;
  line-height: 1.6;
  color: var(--secondary);
  margin: 0;
  max-width: 640px;
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
  border-radius: 2px; top: 50%;
}}

.fork-callout {{
  max-width: 820px;
  margin: 18px auto 0;
  padding: 0 8px;
  font-size: 16px;
  line-height: 1.55;
  color: var(--secondary);
}}
.fork-callout b {{ color: var(--ink); font-weight: 650; }}

.section-title {{
  font-size: 24px;
  font-weight: 650;
  letter-spacing: -0.015em;
  margin: 0 0 8px;
  color: var(--ink);
}}

.section-sub {{
  font-size: 15px;
  color: var(--tertiary);
  margin: 0 0 20px;
  line-height: 1.5;
}}

section.col {{
  margin-bottom: 56px;
}}

.metric-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 16px;
}}

.metric {{
  display: flex;
  flex-direction: column;
  padding: 20px 18px;
  min-height: 190px;
}}

.metric-label {{
  font-size: 12.5px;
  color: var(--tertiary);
  margin: 0 0 12px;
  font-weight: 550;
  letter-spacing: 0.01em;
}}

.metric-before {{
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0 0 6px;
  color: var(--ink);
  white-space: nowrap;
}}
.metric-before .arrow {{
  color: var(--tertiary);
  font-weight: 400;
  margin: 0 2px;
}}
.metric-before .unit {{
  font-size: 13px;
  font-weight: 500;
  color: var(--secondary);
}}

.metric-delta {{
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0 0 12px;
  color: var(--accent);
}}

.spark {{
  display: block;
  width: 100%;
  height: 30px;
  margin: 0 0 12px;
}}

.metric-meaning {{
  font-size: 12.5px;
  line-height: 1.45;
  color: var(--secondary);
  margin: auto 0 0;
}}

.metric-sub {{
  max-width: 640px;
  font-size: 14px;
  color: var(--tertiary);
  line-height: 1.5;
  margin: 0;
}}

.chart-wrap {{
  margin: 0 auto 56px;
  max-width: 1064px;
}}
.chart-head {{
  margin-bottom: 20px;
}}
.chart-card {{
  padding: 24px 8px 8px;
}}
.tested-line {{
  max-width: 820px;
  margin: 16px auto 0;
  padding: 0 8px;
  font-size: 13.5px;
  color: var(--tertiary);
  line-height: 1.5;
}}

.directory {{
  margin-bottom: 56px;
}}
.directory-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 20px;
}}
.directory-head .section-title {{
  margin: 0;
}}
#report-filter {{
  width: min(340px, 100%);
  padding: 9px 12px;
  border: 1px solid var(--hairline);
  border-radius: 10px;
  background: var(--card);
  color: var(--ink);
  font: inherit;
  font-size: 14px;
}}
#report-filter:focus {{
  outline: none;
  border-color: var(--accent);
}}
.report-directory {{
  display: grid;
  gap: 8px;
}}
.report-group {{
  border-top: 1px solid var(--hairline);
  padding-top: 8px;
}}
.report-group summary {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  cursor: pointer;
  list-style: none;
  padding: 6px 4px;
  color: var(--ink);
  font-size: 14px;
  font-weight: 600;
}}
.report-group summary::-webkit-details-marker {{ display: none; }}
.report-group summary em {{
  color: var(--tertiary);
  font-size: 12px;
  font-style: normal;
  font-weight: 500;
}}
.report-rows {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px 16px;
  margin-top: 6px;
}}
.report-row {{
  display: grid;
  gap: 2px;
  padding: 10px 12px;
  border-radius: 10px;
  text-decoration: none;
  transition: background 120ms ease;
}}
.report-row:hover {{
  background: var(--card);
}}
.report-title {{
  color: var(--ink);
  font-size: 14px;
  font-weight: 600;
}}
.report-desc {{
  color: var(--secondary);
  font-size: 12.5px;
  line-height: 1.4;
}}

.footer {{
  padding-top: 24px;
  border-top: 1px solid var(--hairline);
  margin-bottom: 0;
}}
.disclaimer {{
  font-size: 13px;
  color: var(--tertiary);
  margin: 0;
}}

@media (max-width: 900px) {{
  .metric-grid {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
}}

@media (max-width: 620px) {{
  .metric-grid {{
    grid-template-columns: 1fr;
  }}
  .report-rows {{
    grid-template-columns: 1fr;
  }}
  .page {{
    padding: 32px 16px 64px;
  }}
  .metric {{
    min-height: 0;
  }}
}}

"""


JS = """
const setupReportFilter = () => {
  const input = document.getElementById("report-filter");
  if (!input) return;
  const rows = Array.from(document.querySelectorAll("[data-report-row]"));
  const groups = Array.from(document.querySelectorAll("[data-report-group]"));
  input.addEventListener("input", () => {
    const query = input.value.trim().toLowerCase();
    rows.forEach((row) => {
      row.hidden = query.length > 0 && !row.dataset.search.includes(query);
    });
    groups.forEach((group) => {
      const visible = Array.from(group.querySelectorAll("[data-report-row]")).some((row) => !row.hidden);
      group.hidden = !visible;
      if (query) group.open = true;
    });
  });
};
window.addEventListener("DOMContentLoaded", setupReportFilter);
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conn = _connect(DATABASE_PATH)
    df = load_sleep_series(conn)
    conn.close()

    stats = compute_recent_vs_baseline(df)
    tested = load_tested_effect()
    data_end = df["day"].max().date().isoformat()

    html = build_html(df, stats, tested, data_end)

    assets_src = PROJECT_ROOT / "assets"
    if assets_src.is_dir():
        assets_dst = REPORTS_DIR / "assets"
        assets_dst.mkdir(parents=True, exist_ok=True)
        for asset in assets_src.iterdir():
            if asset.is_file():
                shutil.copy2(asset, assets_dst / asset.name)

    out = REPORTS_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
