"""Light clinical theme for Oura Digital Twin HTML reports.

Complete design system: Plotly template, CSS, navigation bar, KPI cards,
section components, and full-page assembly. Every report generator imports
from this module for visual consistency.

Usage:
    from _theme import (
        wrap_html, make_kpi_card, make_kpi_row, make_section,
        make_hero_timeseries, disclaimer_banner, metric_explainer, format_p_value,
        METRIC_DESCRIPTIONS, STATUS_COLORS, COLORWAY,
    )
    import plotly.io as pio
    pio.templates.default = "clinical_dark"

    body = make_kpi_row(
        make_kpi_card("RMSSD", 18.3, "ms", status="critical", detail="Below ESC threshold"),
        make_kpi_card("Mean HR", 72, "bpm", status="normal"),
    )
    hero_fig = make_hero_timeseries()  # queries oura.db directly; or pass metrics={...}
    body += make_chart_panel(
        "HRV and heart rate", "Full observation window",
        hero_fig.to_html(include_plotlyjs=False, full_html=False),
    )
    body += make_section("HRV Trends", fig.to_html(include_plotlyjs=False, full_html=False))
    html = wrap_html("Advanced HRV", body, report_id="hrv")
"""

import json
import re
import sqlite3
import sys
from datetime import date, datetime
from functools import lru_cache
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH, FONT_FAMILY, PLOTLY_CDN_URL, PATIENT_LABEL, DATA_START,
    TREATMENT_START, HEV_DIAGNOSIS_DATE, BETA_BLOCKER_START,
    SITE_NAME, SITE_URL, SITE_DESCRIPTION, SITE_INDEXABLE, OG_IMAGE_PATH,
    REPO_URL, COMPANION_URL, COMPANION_LABEL, HEALTH_EQUITY_URL,
)

# ---------------------------------------------------------------------------
# Report Registry - navigation metadata
# ---------------------------------------------------------------------------

REPORT_REGISTRY = [
    {"id": "home", "file": "index.html", "title": "Dashboard", "group": "Core", "desc": "Current dashboard, report directory, and live status overview."},
    {"id": "about", "file": "roadmap.html#honest", "title": "About", "group": "Context", "desc": "Methodology, limitations, and honest assessment of what this system can and cannot do."},
    {"id": "roadmap", "file": "roadmap.html#roadmap", "title": "Next Steps", "group": "Context", "desc": "Planned analyses, validation targets, and next steps for the digital twin platform."},
    {"id": "how_built", "file": "how_built.html", "title": "How this was built", "group": "Context", "desc": "Who did what, how a number gets onto a page, and how the pipeline checks itself. Measured, not claimed."},
    {"id": "claims", "file": "claims.html", "title": "Every number, checked", "group": "Context", "desc": "Every statistic printed on this site, cross-checked against the JSON the pipeline computed it from."},
    {"id": "full_analysis", "file": "oura_full_analysis.html", "title": "Full Analysis", "group": "Core", "desc": "Heart rate, HRV, sleep, activity, SpO2, and readiness trends across the full observation window."},
    {"id": "biomarkers", "file": "composite_biomarkers.html", "title": "Biomarker Trends", "group": "Core", "desc": "Composite biomarker indices combining multiple Oura signals into research-use summary scores."},
    {"id": "sleep", "file": "advanced_sleep_analysis.html", "title": "Sleep Analysis", "group": "Core", "desc": "Sleep architecture, staging distribution, efficiency, and circadian rhythm analysis."},
    {"id": "causal", "file": "causal_inference_report.html", "title": "Causal: Ruxolitinib", "group": "Clinical", "desc": "Bayesian causal impact and interrupted time-series analysis of ruxolitinib response."},
    {"id": "gvhd", "file": "gvhd_prediction_report.html", "title": "GvHD Prediction", "group": "Clinical", "desc": "Hidden Markov and state-space models predicting GvHD flare probability from wearable signals."},
    {"id": "spo2", "file": "spo2_bos_screening.html", "title": "SpO2 & BOS", "group": "Clinical", "desc": "SpO2 trend monitoring and bronchiolitis obliterans syndrome screening thresholds."},
    {"id": "hrv", "file": "advanced_hrv_analysis.html", "title": "Advanced HRV", "group": "Advanced", "desc": "Frequency-domain HRV, Poincare plots, DFA, sample entropy, and autonomic balance metrics."},
    {"id": "digital_twin", "file": "digital_twin_report.html", "title": "Digital Twin", "group": "Advanced", "desc": "Unscented Kalman Filter digital twin tracking latent inflammatory and autonomic states."},
    {"id": "foundation", "file": "foundation_model_report.html", "title": "Foundation Model", "group": "Advanced", "desc": "Chronos foundation model forecasting with prediction intervals and anomaly scoring."},
    {"id": "anomalies", "file": "anomaly_detection_report.html", "title": "Anomaly Detection", "group": "Advanced", "desc": "Matrix Profile, Isolation Forest, and CUSUM anomaly detection across biometric channels."},
    {"id": "3d_dashboard", "file": "oura_3d_dashboard.html", "title": "3D Dashboard", "group": "Advanced", "desc": "Interactive 3D scatter of sleep, HRV, and activity with treatment phase coloring."},
    {"id": "comp_autonomic", "file": "comparative_autonomic_report.html", "title": "Autonomic Comparison", "group": "Comparative", "desc": "HRV and resting HR recovery trajectories compared between post-HSCT and post-stroke patients."},
    {"id": "comp_treatment", "file": "comparative_treatment_response.html", "title": "Treatment Response", "group": "Comparative", "desc": "Changepoint detection and pre/post treatment response with Mann-Whitney U tests."},
    {"id": "comp_sleep", "file": "comparative_sleep_analysis.html", "title": "Sleep Architecture", "group": "Comparative", "desc": "Sleep architecture, efficiency, and timing compared against clinical benchmarks."},
    {"id": "comp_coupling", "file": "comparative_activity_recovery_coupling.html", "title": "Activity-Recovery", "group": "Comparative", "desc": "Activity-recovery coupling analysis: does day N activity predict day N+1 recovery?"},
    {"id": "comp_anomalies", "file": "comparative_anomaly_report.html", "title": "Anomaly Patterns", "group": "Comparative", "desc": "Anomaly fingerprinting and clustering: how bad days manifest differently."},
    {"id": "comp_breathing", "file": "comparative_breathing_analysis.html", "title": "Breathing Analysis", "group": "Comparative", "desc": "Respiratory-rate trends, week-over-week shifts, and outlier nights against recent baseline."},
    {"id": "comp_temperature", "file": "comparative_temperature_analysis.html", "title": "Temperature Analysis", "group": "Comparative", "desc": "Temperature deviation tracking, excursion alerts, and post-treatment change patterns."},
    {"id": "mitch_standalone", "file": "mitch_standalone_report.html", "title": "P2 Dashboard", "group": "Individual", "desc": "Post-stroke patient P2: HRV, HR, sleep, and activity dashboard."},
    {"id": "wenche_standalone", "file": "wenche_standalone_report.html", "title": "P3 Dashboard", "group": "Individual", "desc": "Healthy control P3: baseline HRV, HR, sleep, and activity reference."},
    {"id": "mitch_changepoints", "file": "mitch_changepoint_investigation.html", "title": "P2 Changepoints", "group": "Comparative", "desc": "Patient 2 changepoint scan of HRV, sleep, and recovery markers around key timeline events."},
    {"id": "weekly", "file": "weekly_tracker.html", "title": "Weekly Tracker", "group": "Core", "desc": "One-page weekly tracker with watchpoints, week-over-week deltas, and clinician-style summary text."},
    {"id": "forecast", "file": "rux_forecast.html", "title": "Rux Forecast", "group": "Clinical", "desc": "Near-term HRV and heart-rate recovery forecast from the current post-treatment trajectory."},
    {"id": "piecewise_its", "file": "piecewise_regression.html", "title": "Piecewise ITS", "group": "Statistical", "desc": "Piecewise ITS regression with AR(1) errors: two-intervention model with date sensitivity analysis."},
    {"id": "sequential_ci", "file": "sequential_causal_impact.html", "title": "Sequential CI", "group": "Statistical", "desc": "Sequential Bayesian CausalImpact isolating Jakavi and beta-blocker effects in separate runs."},
    {"id": "placebo", "file": "placebo_calibration.html", "title": "Placebo Tests", "group": "Statistical", "desc": "Falsification tests at 20 random pre-treatment dates to calibrate false positive rates."},
    {"id": "tau_u", "file": "tau_u_effects.html", "title": "Tau-U Effects", "group": "Statistical", "desc": "Tau-U and NAP effect sizes for single-case experimental design with baseline trend correction."},
    {"id": "synthesis", "file": "research_synthesis.html", "title": "Research Synthesis", "group": "Clinical", "desc": "Two-hit autonomic recovery hypothesis with live KPIs, timeline, and statistical evidence."},
    {"id": "treatment_report", "file": "treatment_response_report.html", "title": "Treatment Report", "group": "Clinical", "desc": "Primary clinical report covering all systems and both medicines for specialist review."},
]

NAV_PRIMARY_IDS = [
    "home",
    "weekly",
    "full_analysis",
    "comp_treatment",
    "piecewise_its",
]

# ---------------------------------------------------------------------------
# Color Palette - clinical LIGHT tokens
# ---------------------------------------------------------------------------

# Approved light clinical-premium palette (matches analysis/generate_index.py
# and analysis/generate_anthropic_case.py exactly):
#   page bg #F7F7F5, card/surface #FFFFFF, hairline rgba(20,22,26,0.08),
#   ink #14161A, secondary #5B616E, tertiary #8A909C, ONE accent deep indigo
#   #3A3AD6 (+ soft tint rgba(58,58,214,0.08)). No green, no teal, no cyan.

# Surfaces and ink
BG_PRIMARY = "#F7F7F5"
BG_SURFACE = "#FFFFFF"
BG_ELEVATED = "#F2F2EF"
TEXT_PRIMARY = "#14161A"
TEXT_SECONDARY = "#5B616E"
TEXT_TERTIARY = "#666D7B"
BORDER_SUBTLE = "rgba(20,22,26,0.06)"
BORDER_DEFAULT = "rgba(20,22,26,0.12)"

# --- Signature accent: ONE deep indigo hue ---
# The whole system runs on a single accent, deep indigo #3A3AD6, plus a soft
# tint for fills. ACCENT_TEAL is kept ONLY as a name (imports depend on it)
# but now resolves to the indigo accent; ACCENT_TEAL_DEEP is a deeper indigo
# used for the darker end of tonal ramps. No teal/cyan/green pigment remains.
ACCENT_TEAL = "#3A3AD6"
ACCENT_TEAL_DEEP = "#2A2AA6"

# Categorical slots. Legacy names are kept as real (non-duplicate) aliases so
# the 25+ report generators that import ACCENT_CYAN/ACCENT_INDIGO/etc. keep
# working. On light we lead with indigo and fill the rest of the wheel with
# ink/grey and a sparing amber; green/teal/cyan are replaced by indigo or ink.
ACCENT_BLUE = ACCENT_TEAL          # the signature accent (deep indigo #3A3AD6)
ACCENT_CYAN = "#6E6EE0"            # lighter indigo tint, distinct from ACCENT_BLUE
ACCENT_GREEN = "#3A3AD6"           # improve reads as indigo on light, not green
ACCENT_AMBER = "#A34A08"           # muted amber, used sparingly for caution (AA on its own tint)
ACCENT_RED = "#B4231F"             # muted clinical red for decline/critical
ACCENT_PURPLE = "#5B5BD0"          # indigo-violet, distinct from ACCENT_BLUE
ACCENT_INDIGO = "#3A3AD6"          # canonical indigo accent
ACCENT_PINK = "#8A4FB0"            # muted violet, distinct from ACCENT_PURPLE
ACCENT_ORANGE = "#B45309"          # shares the amber tone for warm channels

# Status mapping. One improve / one decline / one caution colour. On light we
# prefer indigo + ink; improve is indigo (not green), caution/decline lean on
# muted amber/red used sparingly.
STATUS_COLORS = {
    "normal": ACCENT_INDIGO,
    "good": ACCENT_INDIGO,
    "warning": ACCENT_AMBER,
    "serious": "#C2410C",
    "critical": ACCENT_RED,
    "info": ACCENT_BLUE,
    "neutral": "transparent",
}

# Semantic status aliases (0.1): use these in new code instead of raw
# STATUS_COLORS keys or hex literals.
IMPROVE = ACCENT_INDIGO
DECLINE = ACCENT_RED
CAUTION = ACCENT_AMBER
NEUTRAL = TEXT_SECONDARY
ACCENT = ACCENT_INDIGO

# Semantic channels, reconciled to the light palette so a chart and its KPI
# card share a hue per signal. HRV leads in indigo; HR is a muted ink.
C_HR = TEXT_SECONDARY
C_HRV = ACCENT_INDIGO
C_SLEEP = ACCENT_PURPLE
C_TEMP = ACCENT_ORANGE
C_ACTIVITY = ACCENT_INDIGO
C_SPO2 = TEXT_PRIMARY
C_BREATH = ACCENT_AMBER

# Treatment overlay colors
C_PRE_TX = TEXT_TERTIARY
C_POST_TX = ACCENT_INDIGO
C_COUNTERFACTUAL = "rgba(58,58,214,0.35)"
C_FORECAST = ACCENT_INDIGO
C_BASELINE = "#8A909C"
C_RUX_LINE = C_POST_TX
C_EFFECT = C_FORECAST

# Plotly colorway, fixed order. Indigo leads, then ink/greys and a sparing
# amber; no green/teal/cyan.
COLORWAY = [
    ACCENT_INDIGO, TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_PURPLE,
    ACCENT_AMBER, ACCENT_RED, ACCENT_PINK, TEXT_TERTIARY,
]

RADIUS_SM = 6
RADIUS_MD = 10
RADIUS_LG = 14

# Backward-compatible aliases for old config.py light-theme names.
C_PRIMARY = ACCENT_BLUE
C_SECONDARY = ACCENT_CYAN
C_MUTED = TEXT_SECONDARY
C_LIGHT = TEXT_SECONDARY
C_DARK = TEXT_PRIMARY
C_ACCENT = ACCENT_BLUE
C_CRITICAL = STATUS_COLORS["critical"]
C_GOOD = STATUS_COLORS["good"]
C_WARNING = STATUS_COLORS["warning"]
C_NEUTRAL = TEXT_SECONDARY
C_BG = BG_PRIMARY
C_CARD = BG_SURFACE
C_TEXT = TEXT_PRIMARY
C_GRID = BORDER_SUBTLE
C_TEXT_MUTED = TEXT_SECONDARY
C_BG_LIGHT = BG_ELEVATED
C_CAUTION = STATUS_COLORS["warning"]

# ---------------------------------------------------------------------------
# Metric Descriptions - reusable across reports
# ---------------------------------------------------------------------------

METRIC_DESCRIPTIONS = {
    "GVHD_SCORE": "Composite from HRV, HR, sleep fragmentation, temperature. Ring-derived, not clinical diagnosis.",
    "ADSI": "Autonomic Dysfunction Severity Index. Higher = more dysfunction.",
    "CV_RISK": "Cardiovascular risk proxy from resting HR, HRV, SpO2. Not equivalent to Framingham or SCORE2.",
    "RECOVERY_INDEX": "Overall recovery trajectory. Higher = better.",
    "ALLOSTATIC_LOAD": "Cumulative physiological stress burden. Scale 0-7.",
    "PHARMA_RESPONSE": "Pharmacodynamic response to ruxolitinib. Z-score relative to pre-treatment baseline.",
}

# ---------------------------------------------------------------------------
# Plotly Template
# ---------------------------------------------------------------------------


def create_clinical_dark_template() -> go.layout.Template:
    """Premium LIGHT clinical dashboard Plotly template.

    Name kept as create_clinical_dark_template (and template id "clinical_dark")
    so the 30+ importing report generators keep working; the palette is the
    approved light clinical-premium one (white plot on #F7F7F5 page, ink text,
    single indigo accent).
    """
    template = go.layout.Template()

    template.layout.font = dict(
        family=FONT_FAMILY,
        size=13,
        color=TEXT_PRIMARY,
    )
    template.layout.paper_bgcolor = BG_SURFACE
    template.layout.plot_bgcolor = BG_SURFACE
    template.layout.hovermode = "x unified"
    template.layout.hoverlabel = dict(
        bgcolor=BG_SURFACE,
        font_size=13,
        font_family=FONT_FAMILY,
        font_color=TEXT_PRIMARY,
        bordercolor="rgba(20,22,26,0.08)",
        namelength=-1,
    )
    template.layout.margin = dict(l=56, r=16, t=48, b=40, pad=0)

    template.layout.title = dict(
        font=dict(size=16, color=TEXT_PRIMARY, family=FONT_FAMILY, weight=600),
        x=0.0, xanchor="left",
        pad=dict(l=0, t=0, b=10),
    )

    template.layout.xaxis = dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=BORDER_DEFAULT,
        linewidth=1,
        tickfont=dict(size=11, color=TEXT_TERTIARY),
        title=dict(font=dict(size=12, color=TEXT_TERTIARY), standoff=12),
        rangeselector=dict(
            bgcolor="rgba(0,0,0,0)",
            activecolor=BG_ELEVATED,
            bordercolor="rgba(20,22,26,0.08)",
            borderwidth=1,
            font=dict(size=12, color=TEXT_SECONDARY, family=FONT_FAMILY),
        ),
        automargin=True,
    )
    template.layout.yaxis = dict(
        showgrid=True,
        gridcolor=BORDER_SUBTLE,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=BORDER_DEFAULT,
        linewidth=1,
        tickfont=dict(size=11, color=TEXT_TERTIARY),
        title=dict(font=dict(size=12, color=TEXT_TERTIARY), standoff=12),
        automargin=True,
    )

    template.layout.legend = dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=12, color=TEXT_PRIMARY),
        orientation="h",
        yanchor="top",
        y=1,
        xanchor="right",
        x=1,
        itemsizing="constant",
    )

    template.layout.colorway = COLORWAY

    # Single-hue tonal ramp: light-to-indigo on light backgrounds.
    template.layout.colorscale.sequential = [
        [0, "#EDEDFB"], [0.25, "#C3C3F1"], [0.5, ACCENT_INDIGO],
        [0.75, ACCENT_TEAL_DEEP], [1.0, TEXT_PRIMARY],
    ]
    template.layout.colorscale.diverging = [
        [0, ACCENT_RED], [0.25, "#E0A9A7"], [0.5, "#FFFFFF"],
        [0.75, "#9E9EEC"], [1.0, ACCENT_INDIGO],
    ]

    # Annotation defaults
    template.layout.annotationdefaults = dict(
        font=dict(size=12, color=TEXT_SECONDARY),
        arrowcolor=TEXT_TERTIARY,
        arrowhead=2, arrowwidth=1,
    )

    # Shape defaults (reference bands): faint indigo tint on light.
    template.layout.shapedefaults = dict(
        fillcolor="rgba(58,58,214,0.06)",
        line=dict(color=BORDER_DEFAULT, width=1),
    )

    # Trace defaults
    template.data.scatter = [go.Scatter(
        line=dict(width=2),
        marker=dict(size=8, line=dict(width=2, color=BG_SURFACE)),
    )]
    template.data.bar = [go.Bar(
        marker=dict(line=dict(width=0), opacity=0.9),
    )]
    template.data.heatmap = [go.Heatmap(
        colorscale=[
            [0, "#FFFFFF"], [0.25, "#E3E3F9"], [0.5, "#9E9EEC"],
            [0.75, ACCENT_INDIGO], [1.0, ACCENT_TEAL_DEEP],
        ],
    )]

    return template


# Auto-register at import time (does NOT set as default - each script opts in)
pio.templates["clinical_dark"] = create_clinical_dark_template()

PLOTLY_CONFIG = {
    "responsive": True,
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def add_treatment_marker(fig: go.Figure, x, label: str) -> go.Figure:
    """Add the standardized treatment/event marker to a Plotly figure."""
    fig.add_vline(
        x=x,
        line_width=1,
        line_dash="dot",
        line_color=TEXT_PRIMARY,
        opacity=1,
    )
    fig.add_annotation(
        x=x,
        y=1,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        yanchor="top",
        xanchor="left",
        xshift=6,
        yshift=-8,
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(20,22,26,0.08)",
        borderwidth=1,
        borderpad=4,
        font=dict(size=11, color=TEXT_SECONDARY, family=FONT_FAMILY),
    )
    return fig


def _last_trace_x(fig: go.Figure):
    """Best-effort maximum x-value across Plotly traces for phase spans."""
    import math

    def _sort_key(value):
        if value is None:
            return None
        try:
            if hasattr(value, "to_pydatetime"):
                value = value.to_pydatetime()
        except (TypeError, ValueError):
            return None

        if isinstance(value, datetime):
            return (0, value.timestamp())
        if isinstance(value, date):
            return (0, datetime(value.year, value.month, value.day).timestamp())

        try:
            number = float(value)
            if math.isfinite(number):
                return (1, number)
        except (TypeError, ValueError):
            pass

        text = str(value).strip()
        if not text or text.lower() == "nat":
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return (0, parsed.timestamp())
        except ValueError:
            pass
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
            return (0, parsed.timestamp())
        except ValueError:
            return (2, text)

    best_key = None
    best_value = None
    for trace in fig.data:
        xs = getattr(trace, "x", None)
        if xs is None:
            continue
        try:
            values = list(xs)
        except TypeError:
            continue
        for value in values:
            key = _sort_key(value)
            if key is None:
                continue
            if best_key is None or key > best_key:
                best_key = key
                best_value = value
    return best_value


def add_phase_shading(
    fig: go.Figure,
    x0,
    x1=None,
    *,
    row: int | str | None = None,
    col: int | str | None = None,
) -> go.Figure:
    """Add a quiet post-intervention phase band from x0 to x1 or chart end."""
    resolved_x1 = x1 if x1 is not None else _last_trace_x(fig)
    if resolved_x1 is None:
        return fig

    kwargs = {}
    if row is not None:
        kwargs["row"] = row
    if col is not None:
        kwargs["col"] = col

    fig.add_vrect(
        x0=x0,
        x1=resolved_x1,
        fillcolor="rgba(58,58,214,0.06)",
        line_width=0,
        layer="below",
        **kwargs,
    )
    return fig


def add_end_labels(fig: go.Figure, max_series: int = 4) -> go.Figure:
    """Direct-label up to max_series visible x/y traces at their last point."""
    xy_traces = [
        trace for trace in fig.data
        if getattr(trace, "visible", True) is not False
        and getattr(trace, "showlegend", True) is not False
        and getattr(trace, "x", None) is not None
        and getattr(trace, "y", None) is not None
        and getattr(trace, "name", None)
    ]
    if not xy_traces or len(xy_traces) > max_series:
        return fig

    colorway_idx = 0
    for trace in xy_traces:
        xs = list(trace.x)
        ys = list(trace.y)
        last = next(
            (
                (x_val, y_val)
                for x_val, y_val in zip(reversed(xs), reversed(ys))
                if y_val is not None
            ),
            None,
        )
        if last is None:
            continue
        color = None
        if getattr(trace, "line", None) is not None:
            color = getattr(trace.line, "color", None)
        if not color and getattr(trace, "marker", None) is not None:
            color = getattr(trace.marker, "color", None)
        if not color:
            color = COLORWAY[colorway_idx % len(COLORWAY)]
        colorway_idx += 1
        fig.add_annotation(
            x=last[0],
            y=last[1],
            xref=getattr(trace, "xaxis", None) or "x",
            yref=getattr(trace, "yaxis", None) or "y",
            text=escape(str(trace.name)),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            xshift=8,
            font=dict(size=12, color=color, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=2,
            name="odt-end-label",
        )
    return fig


# ---------------------------------------------------------------------------
# Hero Time-series (0.4)
# ---------------------------------------------------------------------------


def _query_daily_hrv_hr(
    db_path: Path | str | None = None,
) -> tuple[list[str], list[float | None], list[float | None]]:
    """Query daily mean HRV (RMSSD) and heart rate from long-sleep periods.

    Mirrors the query already used by analyze_patient_standalone.py /
    analyze_mitch_changepoints.py: one row per night from oura_sleep_periods
    where type='long_sleep', ordered by day. Returns parallel lists
    (dates, hrv_values, hr_values); missing values are None, never fabricated.
    """
    path = str(db_path) if db_path is not None else str(DATABASE_PATH)
    dates: list[str] = []
    hrv: list[float | None] = []
    hr: list[float | None] = []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT day, average_hrv, average_heart_rate "
            "FROM oura_sleep_periods WHERE type = 'long_sleep' ORDER BY day"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return dates, hrv, hr

    for day, avg_hrv, avg_hr in rows:
        dates.append(day)
        hrv.append(float(avg_hrv) if avg_hrv is not None else None)
        hr.append(float(avg_hr) if avg_hr is not None else None)
    return dates, hrv, hr


def make_hero_timeseries(
    metrics: dict | None = None,
    *,
    db_path: Path | str | None = None,
    show_hr: bool = True,
    title: str = "HRV and heart rate over the full observation window",
    height: int = 460,
    div_id: str = "odt-hero-timeseries",
) -> go.Figure:
    """Build the flagship full-window HRV (+ optional HR) hero time-series.

    This is the ONE reusable hero chart Phase-1 pages should embed above
    their KPI grid: the full observation window, both treatment markers
    (Ruxolitinib start + HEV diagnosis), and quiet shading for the three
    clinical phases (baseline / Jakavi-only / Jakavi+beta-blocker). It never
    computes or infers a data point; it only ever plots values it was given
    or found verbatim in the database.

    Args:
        metrics: Optional dict with an explicit series, shaped as:
            {
                "dates": ["2026-01-08", ...],          # required, ISO YYYY-MM-DD
                "hrv":   [16.2, 15.8, ...] | None,       # same length as dates
                "hr":    [58.1, 59.0, ...] | None,       # same length as dates
            }
            Any of "hrv"/"hr" may be omitted or contain None for missing
            nights; a missing/absent series is simply not plotted (never
            fabricated). If `metrics` is None, the function queries
            oura_sleep_periods (long_sleep rows) from `db_path` (defaults
            to config.DATABASE_PATH) using the same query already used by
            analyze_patient_standalone.py / analyze_mitch_changepoints.py.
        db_path: Override DB path when metrics is None. Defaults to
            config.DATABASE_PATH.
        show_hr: Whether to add the heart-rate trace on a secondary y-axis
            in addition to HRV. If no HR data exists, this is a no-op.
        title: Plotly figure title.
        height: Figure height in px.
        div_id: Unused by the caller directly (Plotly figures don't carry
            a DOM id) but documented here for callers that render via
            fig.to_html(div_id=...) / include it in a chart_data lazy-load
            key, so page generators have one canonical name to reach for.

    Returns:
        A go.Figure using create_clinical_dark_template, with
        add_treatment_marker for Ruxolitinib (config.TREATMENT_START) and
        HEV diagnosis (config.HEV_DIAGNOSIS_DATE), and add_phase_shading
        covering the Jakavi-only window (TREATMENT_START to
        BETA_BLOCKER_START) and the Jakavi+beta-blocker window
        (BETA_BLOCKER_START to the last observed date). The baseline phase
        (before TREATMENT_START) is left unshaded by design, consistent
        with add_phase_shading's "quiet post-intervention band" contract.

    Callers (Phase 1) should still wrap the returned figure in
    make_chart_panel(...) / fig.to_html(...) themselves, since embedding
    (full_html vs lazy-loaded chart_data) is a per-page decision.
    """
    if metrics is not None:
        dates = metrics.get("dates") or []
        hrv_values = metrics.get("hrv")
        hr_values = metrics.get("hr")
    else:
        dates, hrv_values, hr_values = _query_daily_hrv_hr(db_path)

    fig = go.Figure()
    fig.update_layout(template="clinical_dark", height=height, title=dict(text=title))

    has_hrv = bool(dates) and hrv_values and any(v is not None for v in hrv_values)
    has_hr = bool(dates) and show_hr and hr_values and any(v is not None for v in hr_values)

    if not has_hrv and not has_hr:
        # No real series to plot. Render an empty, honestly-labeled figure
        # rather than a fabricated line.
        fig.add_annotation(
            text="No HRV/HR series available for this window",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color=TEXT_TERTIARY),
        )
        return fig

    if has_hrv:
        fig.add_trace(go.Scatter(
            x=dates, y=hrv_values, mode="lines+markers", name="HRV (RMSSD, ms)",
            line=dict(color=C_HRV, width=2),
            marker=dict(size=5, color=C_HRV, line=dict(color=BG_SURFACE, width=1)),
            connectgaps=False,
        ))

    if has_hr:
        fig.add_trace(go.Scatter(
            x=dates, y=hr_values, mode="lines+markers", name="Heart rate (bpm)",
            line=dict(color=C_HR, width=2),
            marker=dict(size=5, color=C_HR, line=dict(color=BG_SURFACE, width=1)),
            yaxis="y2",
            connectgaps=False,
        ))
        fig.update_layout(
            yaxis2=dict(
                overlaying="y", side="right", showgrid=False,
                title=dict(text="bpm", font=dict(size=12, color=TEXT_TERTIARY)),
                tickfont=dict(size=11, color=TEXT_TERTIARY),
            ),
        )

    # Treatment / event markers - never inferred, always the config dates.
    add_treatment_marker(fig, str(TREATMENT_START), "Ruxolitinib start")
    if HEV_DIAGNOSIS_DATE:
        add_treatment_marker(fig, str(HEV_DIAGNOSIS_DATE), "HEV diagnosed")

    # Phase shading: baseline is left unshaded; Jakavi-only and
    # Jakavi+beta-blocker each get a quiet band up to the next boundary
    # (or to the last observed date for the final phase).
    add_phase_shading(fig, str(TREATMENT_START), str(BETA_BLOCKER_START))
    add_phase_shading(fig, str(BETA_BLOCKER_START))

    add_end_labels(fig, max_series=2)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=56, r=56, t=64, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Inter Font Embed
# ---------------------------------------------------------------------------

_INTER_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"'
    ' rel="stylesheet" media="print" onload="this.media=\'all\'">\n'
    '<noscript><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"'
    ' rel="stylesheet"></noscript>'
)


def get_plotly_enhancer_js() -> str:
    """Return runtime Plotly defaults and responsive chart polishing."""
    config_json = json.dumps(PLOTLY_CONFIG)
    return f"""
<script type="module">
(() => {{
  const ODT_CONFIG = {config_json};
  const ODT_COLORWAY = {json.dumps(COLORWAY)};
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.__ODT_PLOTLY_CONFIG = ODT_CONFIG;

	  const mergeConfig = (config) => {{
	    const next = Object.assign({{}}, ODT_CONFIG, config || {{}});
	    next.displayModeBar = false;
	    next.displaylogo = false;
	    next.modeBarButtonsToRemove = Array.from(new Set([
	      ...(ODT_CONFIG.modeBarButtonsToRemove || []),
	      ...((config && config.modeBarButtonsToRemove) || []),
	    ]));
    return next;
  }};

  const patchNewPlot = () => {{
    if (!window.Plotly || window.Plotly.__odtPatched) return;
    const originalNewPlot = window.Plotly.newPlot.bind(window.Plotly);
    const LAZY_MARGIN = 400;
    const plotNow = (gd, data, layout, config) => {{
      const nextLayout = Object.assign({{}}, layout || {{}});
      nextLayout.hovermode = nextLayout.hovermode || "x unified";
      if (reduceMotion) nextLayout.transition = {{ duration: 0 }};
      return originalNewPlot(gd, data, nextLayout, mergeConfig(config)).then((graphDiv) => {{
        window.__odtEnhancePlotly?.(graphDiv);
        return graphDiv;
      }});
    }};
    // Charts below the fold wait until they are about to scroll into view. Every
    // caller gets the same promise it always got; it just resolves later.
    window.Plotly.newPlot = (gd, data, layout, config) => {{
      const el = typeof gd === "string" ? document.getElementById(gd) : gd;
      if (!el || !("IntersectionObserver" in window)) return plotNow(gd, data, layout, config);
      const rect = el.getBoundingClientRect();
      const near = rect.bottom > -LAZY_MARGIN && rect.top < window.innerHeight + LAZY_MARGIN;
      if (near) return plotNow(gd, data, layout, config);
      const wanted = (layout && layout.height) || 420;
      if (!el.style.minHeight) el.style.minHeight = wanted + "px";
      return new Promise((resolve, reject) => {{
        const io = new IntersectionObserver((entries) => {{
          if (!entries.some((entry) => entry.isIntersecting)) return;
          io.disconnect();
          plotNow(gd, data, layout, config).then(resolve, reject);
        }}, {{ rootMargin: LAZY_MARGIN + "px" }});
        io.observe(el);
      }});
    }};
    window.Plotly.__odtPatched = true;
  }};

  const asArray = (value) => {{
    if (Array.isArray(value)) return value;
    if (value && typeof value !== "string" && typeof value.length === "number") {{
      return Array.from(value);
    }}
    return [];
  }};

  const escapeHtml = (value) => String(value || "").replace(/[&<>"']/g, (char) => ({{
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }}[char]));

  const lastPoint = (trace) => {{
    const xs = asArray(trace.x);
    const ys = asArray(trace.y);
    for (let index = Math.min(xs.length, ys.length) - 1; index >= 0; index -= 1) {{
      const x = xs[index];
      const y = ys[index];
      if (x === null || x === undefined || y === null || y === undefined) continue;
      if (typeof y === "number" && !Number.isFinite(y)) continue;
      return {{ x, y }};
    }}
    return null;
  }};

  const traceColor = (trace, index) => {{
    const lineColor = trace.line && typeof trace.line.color === "string" ? trace.line.color : "";
    const markerColor = trace.marker && typeof trace.marker.color === "string" ? trace.marker.color : "";
    return lineColor || markerColor || ODT_COLORWAY[index % ODT_COLORWAY.length];
  }};

  const chartTitleText = (layout) => {{
    const title = layout && layout.title;
    if (!title) return "";
    if (typeof title === "string") return title.trim();
    return String(title.text || "").trim();
  }};

  const normalizeTitle = (value) => String(value || "")
    .replace(/<[^>]*>/g, " ")
    .replace(/\\s+/g, " ")
    .trim()
    .toLowerCase();

  const ensureChartPanel = (graphDiv, titleText) => {{
    if (!graphDiv || graphDiv.dataset.odtPanelized === "true") return;
    const parent = graphDiv.parentElement;
    if (!parent) return;

    if (parent.classList.contains("odt-chart-panel")) {{
      graphDiv.dataset.odtPanelized = "true";
      return;
    }}

    const sectionTitle = graphDiv
      .closest(".odt-section")
      ?.querySelector(".odt-section-heading h2")
      ?.textContent || "";
    const showHeader = titleText && normalizeTitle(titleText) !== normalizeTitle(sectionTitle);

    const panel = document.createElement("div");
    panel.className = `odt-chart-panel${{showHeader ? "" : " odt-chart-panel--compact"}}`;

    if (showHeader) {{
      const header = document.createElement("div");
      header.className = "odt-chart-panel-header";
      header.innerHTML = `<div class="odt-chart-panel-title">${{escapeHtml(titleText)}}</div>`;
      panel.appendChild(header);
    }}

    parent.insertBefore(panel, graphDiv);
    panel.appendChild(graphDiv);
    graphDiv.dataset.odtPanelized = "true";
  }};

	  const labelableTraces = (graphDiv) => {{
	    const blockedTypes = new Set(["heatmap", "contour", "histogram2d", "surface", "table", "pie", "indicator", "sunburst", "treemap"]);
	    return (Array.isArray(graphDiv.data) ? graphDiv.data : []).filter((trace) => {{
      if (!trace || trace.visible === false || trace.visible === "legendonly") return false;
      if (trace.showlegend === false) return false;
      if (!trace.name || String(trace.name).trim() === "") return false;
      if (blockedTypes.has(trace.type)) return false;
      if (!asArray(trace.x).length || !asArray(trace.y).length) return false;
	      return Boolean(lastPoint(trace));
	    }});
	  }};

	  const compactEventText = (text) => {{
	    const normalized = normalizeTitle(text);
	    if (normalized.includes("acute")) return "Acute";
	    if (normalized.includes("rux")) return "Rux";
	    if (normalized.includes("hev")) return "HEV";
	    return String(text || "").replace(/\\s+/g, " ").trim();
	  }};

	  const baseAnnotations = (layout, isNarrow) => {{
	    const source = Array.isArray(layout.annotations)
	      ? layout.annotations.filter((annotation) => annotation && annotation.name !== "odt-end-label")
	      : [];
	    if (!isNarrow) return {{ annotations: source, changed: false }};

	    let eventIndex = 0;
	    let changed = false;
	    const annotations = source.map((annotation) => {{
	      const y = Number(annotation.y);
	      const yref = String(annotation.yref || "");
	      const isTopEvent = annotation.text && yref === "paper" && Number.isFinite(y) && y >= 0.95;
	      if (!isTopEvent) return annotation;

	      const next = Object.assign({{}}, annotation);
	      next.text = compactEventText(next.text);
	      next.textangle = -45;
	      next.xanchor = "left";
	      next.yanchor = "bottom";
	      next.yshift = -8 - (eventIndex % 3) * 14;
	      next.font = Object.assign({{}}, next.font || {{}}, {{ size: 10 }});
	      eventIndex += 1;
	      changed = true;
	      return next;
	    }});
	    return {{ annotations, changed }};
	  }};

	  const endLabelAnnotations = (graphDiv, layout, isNarrow) => {{
	    const base = baseAnnotations(layout, isNarrow);
	    const traces = labelableTraces(graphDiv);
	    if (traces.length < 2 || traces.length > 4) {{
	      return base.changed ? base.annotations : null;
	    }}
	    const labels = traces.map((trace, index) => {{
	      const point = lastPoint(trace);
	      return {{
        x: point.x,
        y: point.y,
        xref: trace.xaxis || "x",
        yref: trace.yaxis || "y",
        text: escapeHtml(trace.name),
        showarrow: false,
        xanchor: "left",
        yanchor: "middle",
        xshift: 8,
        font: {{ size: 12, color: traceColor(trace, index), family: "{FONT_FAMILY}" }},
        bgcolor: "rgba(255,255,255,0.85)",
        borderpad: 2,
	        name: "odt-end-label",
	      }};
	    }});
	    return base.annotations.concat(labels);
	  }};

	  window.__odtEnhancePlotly = function(graphDiv) {{
	    if (!window.Plotly || !graphDiv || !graphDiv.layout) return;
	    try {{
      const chartBox = graphDiv.closest(".chart-box");
      if (chartBox) {{
        chartBox.style.display = "block";
        chartBox.style.width = "100%";
        chartBox.style.overflowX = "hidden";
      }}

      graphDiv.style.marginLeft = "auto";
      graphDiv.style.marginRight = "auto";
      graphDiv.style.display = "block";
      graphDiv.style.width = "100%";
      graphDiv.style.maxWidth = "100%";

      const plotContainer = graphDiv.querySelector(".plot-container");
      if (plotContainer) {{
        plotContainer.style.marginLeft = "auto";
        plotContainer.style.marginRight = "auto";
        plotContainer.style.width = "100%";
        plotContainer.style.maxWidth = "100%";
      }}
    }} catch (e) {{
      // Non-fatal: continue with relayout adjustments.
    }}

	    const layout = graphDiv.layout || {{}};
	    const externalTitle = chartTitleText(layout);
	    ensureChartPanel(graphDiv, externalTitle);
	    const isNarrow = window.matchMedia("(max-width: 560px)").matches;

	    const traceCount = Array.isArray(graphDiv.data) ? graphDiv.data.length : 0;
	    const updates = {{
	      "title.text": "",
      "paper_bgcolor": "{BG_SURFACE}",
      "plot_bgcolor": "{BG_SURFACE}",
      "hovermode": "x unified",
      "font.family": "{FONT_FAMILY}",
      "font.size": Math.max(layout.font?.size || 0, 13),
      "font.color": "{TEXT_PRIMARY}",
      "hoverlabel.bgcolor": "{BG_ELEVATED}",
      "hoverlabel.font.size": 13,
      "hoverlabel.font.family": "{FONT_FAMILY}",
      "hoverlabel.font.color": "{TEXT_PRIMARY}",
	      "margin.t": Math.max(layout.margin?.t || 0, 48),
	      "margin.b": Math.max(layout.margin?.b || 0, isNarrow ? 104 : 40),
	      "margin.l": Math.max(layout.margin?.l || 0, 56),
	      "margin.r": Math.max(layout.margin?.r || 0, 16),
	      "legend.orientation": "h",
	      "legend.x": isNarrow ? 0 : 1,
	      "legend.xanchor": isNarrow ? "left" : "right",
	      "legend.y": isNarrow ? -0.24 : 1,
	      "legend.yanchor": isNarrow ? "top" : "top",
	      "legend.bgcolor": "rgba(0,0,0,0)",
	      "legend.borderwidth": 0,
	      "legend.font.size": isNarrow ? 11 : 12,
	      "showlegend": traceCount > 1,
	    }};

	    const annotations = endLabelAnnotations(graphDiv, layout, isNarrow);
    if (annotations) updates.annotations = annotations;

    Object.keys(layout)
      .filter((key) => /^(x|y)axis\\d*$/.test(key))
      .forEach((key) => {{
        const isX = key.startsWith("x");
        updates[`${{key}}.automargin`] = true;
        updates[`${{key}}.showgrid`] = !isX;
        updates[`${{key}}.gridcolor`] = "{BORDER_SUBTLE}";
        updates[`${{key}}.gridwidth`] = 1;
        updates[`${{key}}.zeroline`] = false;
        updates[`${{key}}.showline`] = true;
        updates[`${{key}}.linecolor`] = "{BORDER_DEFAULT}";
        updates[`${{key}}.linewidth`] = 1;
        updates[`${{key}}.tickfont.size`] = 11;
        updates[`${{key}}.tickfont.color`] = "{TEXT_TERTIARY}";
        updates[`${{key}}.title.font.size`] = 12;
        updates[`${{key}}.title.font.color`] = "{TEXT_TERTIARY}";
        updates[`${{key}}.title.standoff`] = Math.max(layout[key]?.title?.standoff || 0, 12);
        updates[`${{key}}.rangeselector.bgcolor`] = "rgba(0,0,0,0)";
        updates[`${{key}}.rangeselector.activecolor`] = "{BG_ELEVATED}";
        updates[`${{key}}.rangeselector.bordercolor`] = "rgba(20,22,26,0.08)";
        updates[`${{key}}.rangeselector.borderwidth`] = 1;
        updates[`${{key}}.rangeselector.font.size`] = 12;
        updates[`${{key}}.rangeselector.font.color`] = "{TEXT_SECONDARY}";
        updates[`${{key}}.rangeselector.font.family`] = "{FONT_FAMILY}";
      }});

    Plotly.relayout(graphDiv, updates).catch(() => {{}});
  }};

  patchNewPlot();
  window.addEventListener("DOMContentLoaded", patchNewPlot);
  window.addEventListener("load", () => {{
    patchNewPlot();
    window.setTimeout(() => {{
      document.querySelectorAll(".js-plotly-plot").forEach((graphDiv) => {{
        window.__odtEnhancePlotly?.(graphDiv);
      }});
    }}, 80);
  }});

  window.addEventListener("resize", () => {{
    if (!window.Plotly) return;
    document.querySelectorAll(".js-plotly-plot").forEach((graphDiv) => {{
      Plotly.Plots.resize(graphDiv);
    }});
  }});
}})();
</script>"""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def get_base_css() -> str:
    """Return full <style> block for dark clinical reports."""
    return f"""<style>
:root {{
  /* Color tokens from the redesign spec */
  --bg-page: {BG_PRIMARY};
  --bg-card: {BG_SURFACE};
  --bg-primary: var(--bg-page);
  --bg-surface: var(--bg-card);
  --bg-elevated: {BG_ELEVATED};
  --ink-primary: {TEXT_PRIMARY};
  --ink-secondary: {TEXT_SECONDARY};
  --ink-muted: {TEXT_TERTIARY};
  --text-primary: {TEXT_PRIMARY};
  --text-secondary: {TEXT_SECONDARY};
  --text-tertiary: {TEXT_TERTIARY};
  --grid-hairline: {BORDER_SUBTLE};
  --axis-baseline: {BORDER_DEFAULT};
  --border-ring: rgba(20,22,26,0.08);
  --border-subtle: var(--grid-hairline);
  --border-default: var(--axis-baseline);
  --accent-teal: {ACCENT_TEAL};
  --accent-teal-deep: {ACCENT_TEAL_DEEP};
  --accent-blue: {ACCENT_BLUE};
  --accent-aqua: {ACCENT_GREEN};
  --accent-yellow: {ACCENT_AMBER};
  --accent-green: {ACCENT_GREEN};
  --accent-amber: {ACCENT_AMBER};
  --accent-red: {ACCENT_RED};
  --accent-purple: {ACCENT_PURPLE};
  --accent-cyan: {ACCENT_CYAN};
  --accent-pink: {ACCENT_PINK};
  --accent-orange: {ACCENT_ORANGE};
  --accent-indigo: {ACCENT_INDIGO};
  --series-blue: {ACCENT_BLUE};
  --series-aqua: {ACCENT_GREEN};
  --series-yellow: {ACCENT_AMBER};
  --series-green: {C_SPO2};
  --series-violet: {ACCENT_PURPLE};
  --series-red: {ACCENT_RED};
  --series-magenta: {ACCENT_PINK};
  --series-orange: {ACCENT_ORANGE};
  --channel-hr: {C_HR};
  --channel-hrv: {C_HRV};
  --channel-sleep: {C_SLEEP};
  --channel-temperature: {C_TEMP};
  --channel-activity: {C_ACTIVITY};
  --channel-spo2: {C_SPO2};
  --channel-breath: {C_BREATH};
  --status-good: {STATUS_COLORS["good"]};
  --status-warning: {STATUS_COLORS["warning"]};
  --status-serious: {STATUS_COLORS["serious"]};
  --status-critical: {STATUS_COLORS["critical"]};
  /* Semantic status aliases (0.1) */
  --improve: {IMPROVE};
  --decline: {DECLINE};
  --caution: {CAUTION};
  --neutral: {NEUTRAL};
  --accent: {ACCENT};

  /* Elevation: three real surface levels (0.2) */
  --elevation-0: var(--bg-page);
  --elevation-1: var(--bg-surface);
  --elevation-2: var(--bg-elevated);
  --elevation-1-highlight: rgba(255,255,255,0);
  --elevation-1-border: rgba(20,22,26,0.08);
  --elevation-2-highlight: rgba(255,255,255,0);
  --elevation-2-border: rgba(20,22,26,0.10);
  --shadow-1: 0 1px 2px rgba(20,22,26,0.04), 0 8px 24px rgba(20,22,26,0.04);
  --shadow-2: 0 1px 2px rgba(20,22,26,0.05), 0 12px 32px rgba(20,22,26,0.06);
  --hero-gradient: linear-gradient(180deg, rgba(58,58,214,0.03) 0%, rgba(58,58,214,0) 42%);

  /* Modular type scale (0.3): ~1.125 ratio, clamp()-based for hero numerals */
  --text-12: 12px;
  --text-13: 13px;
  --text-14: 14px;
  --text-16: 16px;
  --text-18: 18px;
  --text-22: 22px;
  --text-28: 28px;
  --text-40: 40px;
  --text-48: 48px;
  --text-3xs: var(--text-12);
  --text-2xs: var(--text-12);
  --text-xs: var(--text-12);
  --text-sm: var(--text-14);
  --text-base: var(--text-16);
  --text-lg: var(--text-18);
  --text-xl: var(--text-22);
  --text-2xl: var(--text-28);
  --scale-ratio: 1.125;
  --step-eyebrow: var(--text-12);
  --step-body: var(--text-14);
  --step-heading: var(--text-22);
  --step-title: var(--text-28);
  --step-hero: clamp(2.25rem, 1.9rem + 1.4vw, 3rem);
  --step-hero-lg: clamp(2.75rem, 2.2rem + 2vw, 3.75rem);
  --tracking-eyebrow: 0.06em;
  --tracking-hero: -0.02em;
  --tabular-nums: "tnum" 1, "lnum" 1;

  /* Spacing scale */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-6: 24px;
  --space-8: 32px;
  --space-12: 48px;
  --space-16: 64px;
  --space-xs: var(--space-1);
  --space-sm: var(--space-2);
  --space-md: var(--space-4);
  --space-lg: var(--space-6);
  --space-xl: var(--space-8);
  --space-2xl: var(--space-12);
  --space-3xl: var(--space-16);

  /* Radius, layout, and motion */
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --radius-xl: var(--radius-lg);
  --nav-height: 56px;
  --container-max: 1200px;
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --duration-fast: 150ms;
  --duration-normal: 250ms;
  --duration-slow: 400ms;
  --shadow-overlay: 0 8px 24px rgba(20,22,26,0.10);
  /* Motion-ready hooks (0.6): reusable transition shorthands. No new
     animation is wired up yet; Phase 2 owns actual motion/stagger. */
  --transition-elevation: background var(--duration-fast) ease,
    border-color var(--duration-fast) ease, box-shadow var(--duration-normal) var(--ease-out),
    transform var(--duration-normal) var(--ease-out);
  --transition-color: color var(--duration-fast) ease, background var(--duration-fast) ease,
    border-color var(--duration-fast) ease;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

html {{
  color-scheme: light;
  scroll-behavior: smooth;
}}

::selection {{
  background: rgba(58,58,214,0.16);
  color: var(--text-primary);
}}

:focus-visible {{
  outline: 2px solid var(--accent-blue);
  outline-offset: 2px;
  border-radius: var(--radius-sm);
}}

body {{
  font-family: {FONT_FAMILY};
  background:
    radial-gradient(ellipse at top center, rgba(58,58,214,0.03) 0, rgba(58,58,214,0) 600px),
    var(--bg-page);
  color: var(--text-primary);
  font-size: var(--text-16);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  min-height: 100vh;
}}

/* === Scrollbar (light theme) === */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
::-webkit-scrollbar-thumb {{
  background: var(--border-default);
  border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{ background: var(--text-tertiary); }}

/* === Navigation === */
.odt-nav {{
  position: sticky;
  top: 0;
  z-index: 1000;
  background: var(--bg-page);
  border-bottom: 1px solid var(--border-ring);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--space-6);
  height: var(--nav-height);
  gap: var(--space-4);
}}
.odt-nav-brand {{
  font-size: var(--text-16);
  font-weight: 600;
  color: var(--text-primary);
  text-decoration: none;
  white-space: nowrap;
  display: flex;
  align-items: center;
  gap: var(--space-2);
  letter-spacing: 0;
  transition: opacity var(--duration-fast) ease;
}}
.odt-nav-brand:hover {{ opacity: 0.85; }}
.odt-nav-brand .odt-logo {{
  width: 24px;
  height: 24px;
  border-radius: var(--radius-sm);
  background: var(--bg-elevated);
  border: 1px solid var(--border-ring);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--text-12);
  font-weight: 700;
  color: var(--accent-blue);
  letter-spacing: 0;
  flex-shrink: 0;
}}
.odt-nav-brand span {{ color: var(--accent-blue); }}
.odt-nav-links {{
  display: flex;
  align-items: center;
  gap: var(--space-3);
  flex: 1;
  min-width: 0;
  justify-content: flex-end;
}}
.odt-nav-primary {{
  display: flex;
  align-items: center;
  gap: var(--space-2);
  min-width: 0;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
  flex: 1;
}}
.odt-nav-primary::-webkit-scrollbar {{ display: none; }}
.odt-nav-current {{
  flex-shrink: 0;
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-tertiary);
  padding: 0 2px;
  white-space: nowrap;
}}
.odt-nav-link {{
  font-size: var(--text-13);
  font-weight: 600;
  color: var(--text-secondary);
  text-decoration: none;
  padding: var(--space-2) var(--space-3);
  border: 1px solid transparent;
  white-space: nowrap;
  transition: color var(--duration-fast) ease, border-color var(--duration-fast) ease, background var(--duration-fast) ease;
  border-radius: var(--radius-sm);
}}
.odt-nav-link:hover {{
  color: var(--text-primary);
  background: var(--bg-elevated);
}}
.odt-nav-link.active {{
  color: var(--accent-blue);
  border-color: var(--border-ring);
  background: var(--bg-surface);
}}
.odt-nav-browse {{
  position: relative;
  flex-shrink: 0;
}}
.odt-nav-browse summary {{
  list-style: none;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  color: var(--text-primary);
  font-size: var(--text-13);
  font-weight: 600;
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-ring);
  background: var(--bg-surface);
  transition: border-color var(--duration-fast) ease, background var(--duration-fast) ease;
}}
.odt-nav-browse summary::-webkit-details-marker {{ display: none; }}
.odt-nav-browse summary::after {{
  content: "";
  width: 6px;
  height: 6px;
  border-right: 1px solid var(--text-tertiary);
  border-bottom: 1px solid var(--text-tertiary);
  transform: rotate(45deg);
  margin-top: -3px;
  transition: transform var(--duration-fast) ease;
}}
.odt-nav-browse summary:hover,
.odt-nav-browse[open] summary {{
  border-color: var(--border-ring);
  background: var(--bg-elevated);
}}
.odt-nav-browse[open] summary::after {{
  transform: rotate(225deg);
  margin-top: 3px;
}}
.odt-nav-panel {{
  position: absolute;
  top: calc(100% + 10px);
  right: 0;
  width: min(860px, calc(100vw - 32px));
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-ring);
  background: var(--bg-elevated);
  box-shadow: var(--shadow-overlay);
}}
.odt-nav-panel-header {{
  margin-bottom: var(--space-3);
  color: var(--text-tertiary);
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
.odt-nav-panel-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-4);
  max-height: 70vh;
  overflow-y: auto;
  padding-right: 4px;
}}
.odt-nav-panel-group {{
  min-width: 0;
}}
.odt-nav-group {{
  display: block;
  font-size: var(--text-12);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-tertiary);
  margin-bottom: 8px;
}}
.odt-nav-panel-links {{
  display: grid;
  gap: 6px;
}}
.odt-nav-panel-link {{
  display: block;
  font-size: var(--text-13);
  font-weight: 500;
  color: var(--text-secondary);
  text-decoration: none;
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  border: 1px solid transparent;
  transition: color var(--duration-fast) ease, background var(--duration-fast) ease, border-color var(--duration-fast) ease;
}}
.odt-nav-panel-link:hover {{
  color: var(--text-primary);
  background: var(--bg-surface);
}}
.odt-nav-panel-link.active {{
  color: var(--accent-blue);
  background: var(--bg-surface);
  border-color: var(--border-ring);
}}
.odt-nav-toggle {{
  display: none;
  background: none;
  border: 1px solid var(--border-ring);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: var(--text-18);
  cursor: pointer;
  padding: 6px 10px;
  margin-left: auto;
  transition: color var(--duration-fast) ease, border-color var(--duration-fast) ease, background var(--duration-fast) ease;
}}
.odt-nav-toggle:hover {{
  color: var(--text-primary);
  border-color: var(--text-tertiary);
  background: var(--bg-elevated);
}}

/* === Container === */
.odt-container {{
  max-width: var(--container-max);
  margin: 0 auto;
  padding: var(--space-8) var(--space-6) var(--space-12);
}}

/* === Report Header === */
.odt-header {{
  padding: var(--space-12) var(--space-6) var(--space-8);
  max-width: var(--container-max);
  margin: 0 auto;
  position: relative;
}}
.odt-header::after {{
  content: '';
  position: absolute;
  bottom: 0;
  left: var(--space-6);
  right: var(--space-6);
  height: 1px;
  background: var(--border-ring);
}}
.odt-header h1 {{
  font-size: var(--text-28);
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--space-2);
  letter-spacing: 0;
  line-height: 1.2;
  max-width: 980px;
}}
.odt-header .subtitle {{
  font-size: var(--text-16);
  font-weight: 500;
  color: var(--text-secondary);
  line-height: 1.5;
  max-width: 760px;
}}
.odt-header .metadata {{
  font-size: var(--text-13);
  color: var(--text-tertiary);
  margin-top: var(--space-4);
  display: flex;
  align-items: center;
  gap: var(--space-2);
}}
.odt-header .metadata::before {{
  content: '';
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent-green);
  flex-shrink: 0;
}}

/* === KPI Cards === */
.odt-kpi-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--space-4);
  margin-bottom: var(--space-6);
}}
.odt-kpi {{
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  padding: var(--space-4);
  border: 1px solid var(--border-ring);
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: background var(--duration-fast) ease, border-color var(--duration-fast) ease;
}}
.odt-kpi:hover {{
  background: var(--bg-elevated);
  border-color: var(--border-ring);
}}
.odt-kpi-status {{
  display: none;
}}
.odt-kpi--critical,
.odt-kpi--warning,
.odt-kpi--normal,
.odt-kpi--good,
.odt-kpi--info {{ border-color: var(--border-ring); }}
.odt-kpi-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2);
  margin-bottom: var(--space-2);
}}
.odt-kpi-status-label {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: var(--text-12);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 3px var(--space-2);
  border-radius: var(--radius-sm);
  flex-shrink: 0;
  line-height: 1.2;
}}
.odt-kpi-status-label::before {{
  content: '';
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
  flex-shrink: 0;
}}
.odt-kpi-label {{
  font-size: var(--text-12);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-tertiary);
  font-weight: 600;
}}
.odt-kpi-value {{
  font-size: var(--text-40);
  font-weight: 600;
  margin-top: var(--space-1);
  color: var(--text-primary);
  line-height: 1.05;
  letter-spacing: var(--tracking-hero);
  overflow-wrap: break-word;
  display: flex;
  align-items: baseline;
  gap: var(--space-1);
  flex-wrap: wrap;
  font-variant-numeric: tabular-nums;
  font-feature-settings: var(--tabular-nums);
}}
.odt-kpi-unit {{
  font-size: var(--text-14);
  color: var(--text-secondary);
  font-weight: 400;
}}
.odt-kpi-detail {{
  font-size: var(--text-13);
  color: var(--text-secondary);
  margin-top: var(--space-2);
  line-height: 1.5;
  padding-top: var(--space-2);
  border-top: 1px solid var(--border-ring);
}}

/* === Sections === */
.odt-section {{
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-ring);
  padding: var(--space-6);
  margin-bottom: var(--space-6);
  position: relative;
  scroll-margin-top: 112px;
  transition: border-color var(--duration-fast) ease;
}}
.odt-section::before {{
  content: none;
}}
.odt-section h2 {{
  font-size: var(--text-22);
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--space-4);
  padding-bottom: var(--space-3);
  padding-left: 0;
  border-bottom: 1px solid var(--border-ring);
  border-left: 0;
  letter-spacing: 0;
  line-height: 1.2;
}}
.odt-section h3 {{
  font-size: var(--text-18);
  font-weight: 600;
  color: var(--text-primary);
  margin-top: var(--space-6);
  margin-bottom: var(--space-2);
}}
.odt-section p {{
  margin-bottom: var(--space-4);
  font-size: var(--text-14);
  color: var(--text-secondary);
  line-height: 1.5;
}}

/* === Tables === */
table {{
  width: 100%;
  max-width: 100%;
  border-collapse: collapse;
  margin: var(--space-md) 0;
  font-size: var(--text-sm);
}}
th {{
  text-align: left;
  padding: var(--space-3) var(--space-4);
  background: var(--bg-elevated);
  color: var(--text-primary);
  font-weight: 600;
  font-size: var(--text-12);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border-bottom: 1px solid var(--border-default);
  font-variant-numeric: tabular-nums;
}}
td {{
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  transition: background var(--duration-fast) ease;
  font-variant-numeric: tabular-nums;
}}
tr:hover td {{ background: var(--bg-elevated); }}

/* === Plotly overrides === */
.plotly-graph-div {{
  margin: 0 auto !important;
  width: 100% !important;
  max-width: 100%;
  min-height: 340px;
}}
.js-plotly-plot {{
  margin-left: auto !important;
  margin-right: auto !important;
  width: 100% !important;
  max-width: 100%;
}}
.js-plotly-plot .plot-container {{
  margin-left: auto !important;
  margin-right: auto !important;
  width: 100% !important;
  max-width: 100%;
}}
.js-plotly-plot .plotly .modebar {{
  right: 8px !important;
  display: none !important;
  opacity: 0;
  pointer-events: none;
}}
.js-plotly-plot:hover .plotly .modebar {{ opacity: 0; }}
.js-plotly-plot .plotly .modebar-btn {{ font-size: 14px; }}
.js-plotly-plot .scatterlayer .js-line {{
  stroke-linecap: round;
  stroke-linejoin: round;
}}
/* Prevent subplot-title annotations from being clipped */
.odt-section .js-plotly-plot {{ overflow: visible; }}
.odt-section .plot-container {{ overflow: visible; }}
.odt-section .svg-container {{ overflow: visible !important; }}
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text,
.odt-axis,
.odt-tick {{
  font-variant-numeric: tabular-nums;
}}
.js-plotly-plot .main-svg text {{
  text-rendering: geometricPrecision;
}}

/* === Chart boxes (lazy-loaded) === */
.chart-box {{
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-ring);
  min-height: 200px;
  margin-bottom: var(--space-md);
  display: block;
  width: 100%;
  overflow-x: hidden;
  padding: 0;
  color: var(--text-tertiary);
  font-size: var(--text-sm);
}}
.chart-box > .js-plotly-plot {{
  display: block !important;
  width: 100% !important;
  max-width: 100%;
  margin: 0 auto !important;
}}
.chart-box > .plotly-graph-div,
.chart-box .plot-container,
.chart-box .svg-container {{
  display: block !important;
  width: 100% !important;
  max-width: 100%;
  margin: 0 auto !important;
}}

/* === Narrative callout === */
.odt-narrative {{
  padding: var(--space-4);
  background: var(--bg-surface);
  border: 1px solid var(--border-ring);
  border-radius: var(--radius-md);
  font-size: var(--text-14);
  color: var(--text-primary);
  line-height: 1.5;
  margin-bottom: var(--space-lg);
}}

/* === Context Strip (disclaimer + confound merged) === */
.odt-context-strip {{
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-ring);
  min-height: 32px;
  padding: 0 var(--space-6);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-3);
  font-size: var(--text-12);
  line-height: 1.4;
  flex-wrap: wrap;
}}
.odt-context-strip .odt-ctx-item {{
  color: var(--text-tertiary);
  display: flex;
  align-items: center;
  gap: 6px;
}}
.odt-context-strip .odt-ctx-item::before {{
  content: '';
  display: inline-block;
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: var(--border-default);
  flex-shrink: 0;
}}
.odt-context-strip .odt-ctx-item:first-child::before {{ display: none; }}
.odt-context-strip .odt-ctx-item.warn {{
  color: var(--accent-amber);
}}
.odt-context-strip .odt-ctx-item.warn::before {{
  background: var(--accent-amber);
}}
.odt-context-strip .odt-ctx-dot {{
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--border-subtle);
  flex-shrink: 0;
}}

/* === Metric Explainer === */
.odt-kpi-explainer {{
  font-size: var(--text-12);
  color: var(--text-tertiary);
  margin-top: var(--space-2);
  line-height: 1.5;
}}
.odt-kpi-explainer b {{
  color: var(--text-secondary);
  font-weight: 600;
}}

/* === Footer === */
.odt-footer {{
  text-align: center;
  padding: var(--space-8) var(--space-6);
  color: var(--text-tertiary);
  font-size: var(--text-13);
  margin-top: var(--space-3xl);
  position: relative;
}}
.odt-footer::before {{
  content: '';
  position: absolute;
  top: 0;
  left: 10%;
  right: 10%;
  height: 1px;
  background: var(--border-ring);
}}
.odt-footer div {{
  margin-bottom: var(--space-sm);
  line-height: 1.5;
}}
.odt-footer div:last-child {{ margin-bottom: 0; }}
.odt-footer .odt-footer-fine {{
  font-size: var(--text-12);
  opacity: 0.7;
}}
.odt-footer a {{
  color: var(--accent-blue);
  text-decoration: none;
  transition: color var(--duration-fast) ease;
}}
.odt-footer a:hover {{
  color: var(--accent-blue);
  text-decoration: underline;
  text-underline-offset: 2px;
}}

/* === Responsive === */
@media (max-width: 1200px) {{
  .odt-nav-current {{ display: none; }}
  .odt-nav {{
    padding: 0 20px;
    gap: 12px;
  }}
}}
@media (max-width: 900px) {{
  .odt-nav {{
    flex-wrap: wrap;
    height: auto;
    padding: 10px 16px;
  }}
  .odt-nav-toggle {{ display: block; }}
  .odt-nav-links {{
    display: none;
    width: 100%;
    flex-direction: column;
    align-items: stretch;
    padding: var(--space-sm) 0 4px;
    border-top: 1px solid var(--border-ring);
  }}
  .odt-nav-links.open {{ display: flex; }}
  .odt-nav-primary {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    overflow: visible;
  }}
  .odt-nav-link {{
    padding: 10px 12px;
    border-color: var(--border-ring);
    background: var(--bg-surface);
  }}
  .odt-nav-link.active {{
    background: var(--bg-surface);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
  }}
  .odt-nav-current {{ display: none; }}
  .odt-nav-browse {{ width: 100%; }}
  .odt-nav-browse summary {{
    width: 100%;
    justify-content: space-between;
  }}
  .odt-nav-panel {{
    position: static;
    width: 100%;
    margin-top: 10px;
    max-height: none;
    box-shadow: none;
  }}
  .odt-nav-panel-grid {{ grid-template-columns: 1fr; max-height: none; }}
  .odt-context-strip {{ padding: var(--space-sm) 16px; gap: var(--space-sm); flex-direction: column; }}
  .odt-context-strip .odt-ctx-dot {{ display: none; }}
  .odt-container {{ padding: var(--space-6) var(--space-4) var(--space-8); }}
  .odt-header {{ padding: var(--space-8) var(--space-4) var(--space-6); }}
  .odt-header::after {{ left: 16px; right: 16px; }}
  .odt-header h1 {{ line-height: 1.1; }}
  .odt-kpi-row {{ grid-template-columns: repeat(2, 1fr); gap: var(--space-3); }}
  .odt-kpi {{ padding: var(--space-4); }}
  .odt-kpi-value {{ font-size: var(--text-28); }}
  .odt-section {{
    padding: var(--space-4);
    scroll-margin-top: 132px;
  }}
  .odt-section::before {{ left: 16px; right: 16px; }}
  .odt-section h2 {{ margin-bottom: 16px; }}
  .odt-footer::before {{ left: 5%; right: 5%; }}
}}
@media (max-width: 480px) {{
  .odt-kpi-row {{ grid-template-columns: 1fr; }}
  .odt-nav-primary {{ grid-template-columns: 1fr; }}
  .odt-header h1 {{ font-size: var(--text-28); }}
  .odt-kpi-head {{
    flex-direction: column;
    align-items: flex-start;
  }}
  .odt-kpi-status-label {{ margin-top: 2px; }}
}}

/* === Animations === */
@keyframes fadeInUp {{
  from {{ opacity: 0; transform: translateY(24px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
  from {{ opacity: 0; }}
  to {{ opacity: 1; }}
}}
@keyframes shimmer {{
  0% {{ background-position: -200% 0; }}
  100% {{ background-position: 200% 0; }}
}}
@keyframes pulseGlow {{
  0%, 100% {{ opacity: 0.6; }}
  50% {{ opacity: 1; }}
}}
@keyframes slideInRight {{
  from {{ opacity: 0; transform: translateX(-12px); }}
  to {{ opacity: 1; transform: translateX(0); }}
}}

/* === Skeleton Loading === */
.odt-skeleton {{
  background: linear-gradient(90deg, var(--bg-surface) 25%, var(--bg-elevated) 50%, var(--bg-surface) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.8s ease-in-out infinite;
  border-radius: var(--radius-sm);
}}

/* === Scroll Reveal === */
.odt-reveal {{
  opacity: 0;
  transform: translateY(20px);
  transition: opacity 0.7s var(--ease-out), transform 0.7s var(--ease-out);
}}
.odt-reveal.visible {{
  opacity: 1;
  transform: translateY(0);
}}

/* === Badge/Chip === */
.odt-badge {{
  display: inline-flex;
  align-items: center;
  padding: 3px var(--space-2);
  border-radius: var(--radius-sm);
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: var(--bg-elevated);
  border: 1px solid var(--border-ring);
  transition: opacity var(--duration-fast) ease;
}}
.odt-badge:hover {{ opacity: 0.85; }}
.odt-badge-blue {{ color: var(--accent-blue); }}
.odt-badge-green {{ color: var(--status-good); }}
.odt-badge-red {{ color: var(--status-critical); }}
.odt-badge-amber {{ color: var(--status-warning); }}

/* === Redesign core layer === */
.odt-nav {{
  height: var(--nav-height);
  padding: 0 var(--space-6);
  background: rgba(247, 247, 245, 0.85);
  -webkit-backdrop-filter: blur(12px);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border-ring);
  box-shadow: none;
  gap: var(--space-6);
  flex-wrap: nowrap;
}}
@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {{
  .odt-nav {{ background: var(--bg-page); }}
}}
.odt-nav-brand {{
  font-size: var(--text-16);
  font-weight: 600;
  letter-spacing: 0;
  gap: 10px;
}}
.odt-nav-brand .odt-logo {{
  width: 24px;
  height: 24px;
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  border: 1px solid var(--border-ring);
  color: var(--ink-primary);
  box-shadow: none;
}}
.odt-brand-text {{ color: var(--ink-primary); }}
.odt-brand-mark {{
  color: var(--ink-muted);
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
.odt-nav-links {{
  justify-content: flex-end;
  gap: var(--space-4);
}}
.odt-nav-primary {{
  justify-content: flex-end;
  gap: 2px;
  overflow: visible;
}}
.odt-nav-current,
.odt-nav-toggle {{ display: none !important; }}
.odt-nav-link {{
  height: var(--nav-height);
  display: inline-flex;
  align-items: center;
  padding: 0 10px;
  border: 0;
  border-bottom: 2px solid transparent;
  border-radius: 0;
  background: transparent;
  color: var(--ink-secondary);
  font-size: var(--text-13);
  font-weight: 500;
  letter-spacing: 0;
  transform: none;
  transition: color 120ms ease, background 120ms ease, border-color 120ms ease;
}}
.odt-nav-link:hover {{
  color: var(--ink-primary);
  background: rgba(20,22,26,0.04);
  transform: none;
}}
.odt-nav-link.active {{
  color: var(--ink-primary);
  border-bottom-color: var(--accent-blue);
  background: transparent;
  box-shadow: none;
}}
.odt-nav-browse summary {{
  height: 36px;
  padding: 0 12px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-ring);
  background: rgba(255,255,255,0.9);
  color: var(--ink-primary);
  font-size: var(--text-13);
  font-weight: 600;
  transition: background 120ms ease, border-color 120ms ease;
}}
.odt-nav-browse summary::after {{ content: ""; }}
.odt-nav-panel {{
  top: calc(100% + 8px);
  width: min(840px, calc(100vw - 32px));
  padding: var(--space-4);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-ring);
  background: rgba(255,255,255,0.98);
  box-shadow: var(--shadow-overlay);
}}
.odt-nav-panel-header {{
  margin-bottom: var(--space-4);
  color: var(--ink-muted);
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: 0.06em;
}}
.odt-nav-panel-grid {{
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px 20px;
}}
.odt-nav-group {{
  font-size: var(--text-12);
  letter-spacing: 0.06em;
  color: var(--ink-muted);
}}
.odt-nav-panel-link {{
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  color: var(--ink-secondary);
}}
.odt-nav-panel-title {{
  display: inline;
  color: inherit;
  font-size: var(--text-13);
  font-weight: 600;
  line-height: 1.3;
}}
.odt-nav-panel-separator {{
  color: var(--ink-muted);
}}
.odt-nav-panel-desc {{
  display: inline;
  margin-top: 0;
  color: var(--ink-muted);
  font-size: var(--text-12);
  line-height: 1.35;
}}
.odt-nav-panel-link:hover,
.odt-nav-panel-link:focus-visible {{
  color: var(--ink-primary);
  background: rgba(20,22,26,0.04);
}}
.odt-nav-panel-link.active {{
  color: var(--ink-primary);
  background: rgba(58,58,214,0.08);
  border-color: rgba(58,58,214,0.24);
}}

.odt-context-strip {{
  display: block;
  height: 32px;
  min-height: 32px;
  padding: 0 var(--space-6);
  background: var(--bg-surface);
  border-bottom: 0;
  box-sizing: border-box;
  color: var(--ink-secondary);
  font-size: var(--text-12);
  line-height: 1.4;
  overflow: hidden;
}}
.odt-context-strip summary {{
  min-height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  list-style: none;
  cursor: pointer;
}}
.odt-context-strip summary::-webkit-details-marker {{ display: none; }}
.odt-context-strip .odt-ctx-hev {{
  color: var(--status-warning);
  font-weight: 600;
}}
.odt-context-strip .odt-ctx-dot {{
  width: auto;
  height: auto;
  border-radius: 0;
  background: transparent;
  color: var(--ink-muted);
}}
.odt-ctx-more-label {{
  display: inline-flex;
  color: var(--ink-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-left: 2px;
}}
.odt-context-more {{
  display: none;
  max-width: var(--container-max);
  margin: 0 auto;
  padding: 0 0 10px;
  color: var(--ink-muted);
  text-align: center;
}}
.odt-context-strip[open] {{
  height: auto;
  overflow: visible;
}}
.odt-context-strip[open] .odt-context-more {{ display: block; }}

.odt-container {{
  max-width: var(--container-max);
  padding: var(--space-8) var(--space-6) var(--space-12);
}}
.odt-header {{
  max-width: var(--container-max);
  padding: var(--space-12) var(--space-6) var(--space-6);
}}
.odt-header::after {{ display: none; }}
.odt-header h1 {{
  max-width: 920px;
  margin-bottom: 8px;
  color: var(--ink-primary);
  background: none;
  -webkit-text-fill-color: currentColor;
  font-size: var(--text-28);
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.2;
}}
.odt-header .subtitle {{
  max-width: 760px;
  color: var(--ink-secondary);
  font-size: var(--text-14);
  line-height: 1.5;
}}
.odt-header .metadata {{
  margin-top: 12px;
  color: var(--ink-muted);
  font-size: var(--text-13);
}}
.odt-header .metadata::before {{ display: none; }}

.odt-kpi-row {{
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: var(--space-3);
  margin-bottom: var(--space-6);
}}
.odt-kpi {{
  min-height: 164px;
  padding: var(--space-4);
  background: var(--elevation-1);
  border: 1px solid var(--elevation-1-border);
  border-radius: var(--radius-md);
  box-shadow: inset 0 1px 0 var(--elevation-1-highlight), var(--shadow-1);
  backdrop-filter: none;
  overflow: visible;
  transition: var(--transition-elevation);
}}
.odt-kpi:hover {{
  transform: none;
  box-shadow: inset 0 1px 0 var(--elevation-1-highlight), var(--shadow-1);
  border-color: rgba(20,22,26,0.14);
  background: var(--elevation-2);
}}
.odt-kpi--critical::before {{
  content: '';
  position: absolute;
  left: 0;
  top: var(--space-3);
  bottom: var(--space-3);
  width: 2px;
  border-radius: 0 2px 2px 0;
  background: var(--status-critical);
}}
.odt-kpi-status {{ display: none; }}
.odt-kpi-head {{
  align-items: flex-start;
  flex-wrap: wrap;
  margin-bottom: 10px;
}}
.odt-kpi-label {{
  color: var(--ink-muted);
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: 0.06em;
  line-height: 1.2;
}}
.odt-kpi-status-label {{
  gap: 6px;
  padding: 3px 7px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-ring);
  background: var(--bg-elevated);
  color: var(--ink-secondary);
  font-size: var(--text-12);
  font-weight: 500;
  text-transform: none;
  letter-spacing: 0;
}}
.odt-kpi-status-label::before {{
  content: none;
}}
.odt-kpi-status-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--status-color, var(--ink-muted));
}}
.odt-kpi-status-label--dot {{
  width: 18px;
  height: 18px;
  justify-content: center;
  padding: 0;
}}
.odt-kpi-value {{
  margin-top: 0;
  color: var(--ink-primary);
  font-size: 32px;
  font-weight: 650;
  letter-spacing: var(--tracking-hero);
  line-height: 1.05;
  overflow-wrap: normal;
  font-variant-numeric: tabular-nums;
  font-feature-settings: var(--tabular-nums);
}}
.odt-kpi-unit {{
  color: var(--ink-secondary);
  font-size: var(--text-13);
  font-weight: 400;
}}
.odt-kpi-delta {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin-top: 10px;
  padding: 3px 7px;
  border-radius: var(--radius-sm);
  background: rgba(20,22,26,0.05);
  color: var(--ink-secondary);
  font-size: 11px;
  font-weight: 500;
}}
.odt-kpi-delta b {{ font-weight: 600; }}
.odt-kpi-delta em {{
  color: var(--ink-muted);
  font-style: normal;
}}
.odt-kpi-delta--good {{
  color: var(--status-good);
  background: rgba(58,58,214,0.08);
}}
.odt-kpi-delta--bad {{
  color: var(--status-critical);
  background: rgba(180,35,31,0.10);
}}
.odt-kpi-delta--flat {{ color: var(--ink-secondary); }}
.odt-kpi-detail {{
  margin-top: 8px;
  padding-top: 0;
  border-top: 0;
  color: var(--ink-secondary);
  font-size: var(--text-13);
}}
.odt-kpi-trend {{
  width: 100%;
  height: 42px;
  margin-top: auto;
  padding-top: var(--space-2);
  display: block;
}}

.odt-section {{
  margin: 0 0 var(--space-16);
  padding: 0;
  background: transparent;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  scroll-margin-top: 104px;
}}
.odt-section::before {{ display: none; }}
.odt-section-heading {{ margin-bottom: var(--space-6); }}
.odt-section-kicker {{
  color: var(--ink-muted);
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: var(--tracking-eyebrow);
  text-transform: uppercase;
  margin-bottom: 8px;
}}
.odt-section-title-row {{
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: var(--space-4);
}}
.odt-section-title-row span {{
  height: 1px;
  background: linear-gradient(90deg, var(--border-ring), rgba(20,22,26,0));
}}
.odt-section h2 {{
  margin: 0;
  padding: 0;
  border: 0;
  color: var(--ink-primary);
  font-size: var(--text-22);
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.2;
}}
.odt-section h3 {{
  color: var(--ink-primary);
  font-size: var(--text-18);
  font-weight: 600;
}}
.odt-section p {{
  color: var(--ink-secondary);
  font-size: var(--text-14);
  line-height: 1.5;
}}
.chart-box,
.odt-chart-panel {{
  background: var(--elevation-1);
  border: 1px solid var(--elevation-1-border);
  border-radius: var(--radius-md);
  box-shadow: inset 0 1px 0 var(--elevation-1-highlight), var(--shadow-1);
  transition: var(--transition-elevation);
}}
.chart-box:hover,
.odt-chart-panel:hover {{
  border-color: rgba(20,22,26,0.14);
}}
.odt-chart-panel {{
  margin-bottom: var(--space-6);
  overflow: hidden;
}}
.odt-chart-panel .chart-box {{
  margin-bottom: 0;
  border: 0;
  border-radius: 0;
  box-shadow: none;
}}
.odt-chart-panel > .js-plotly-plot,
.odt-chart-panel > .plotly-graph-div {{
  margin-bottom: 0;
}}
.odt-chart-panel-header {{
  display: grid;
  gap: 4px;
  padding: 18px 20px 0;
}}
.odt-chart-panel-title {{
  color: var(--ink-primary);
  font-size: var(--text-14);
  font-weight: 600;
  line-height: 1.35;
}}
.odt-chart-panel-subtitle {{
  color: var(--ink-muted);
  font-size: var(--text-12);
  line-height: 1.45;
}}
.odt-chart-panel--compact {{
  padding-top: 0;
}}

/* === Hero card (0.2): reusable elevation-2 surface with top-down gradient,
   used by hero KPI/summary blocks (e.g. the Bayesian-twin hero, the index
   dashboard hero). Generalised from the ad-hoc styling that already worked
   well on the digital-twin hero. === */
.odt-hero-card {{
  position: relative;
  background: var(--elevation-2);
  background-image: var(--hero-gradient);
  border: 1px solid var(--elevation-2-border);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  box-shadow: inset 0 1px 0 var(--elevation-2-highlight), var(--shadow-2);
  transition: var(--transition-elevation);
}}
.odt-hero-card:hover {{
  border-color: rgba(20,22,26,0.16);
}}
table {{
  border-collapse: collapse;
  border-spacing: 0;
  margin: var(--space-4) 0 var(--space-6);
  font-variant-numeric: tabular-nums;
}}
th {{
  background: transparent;
  color: var(--ink-muted);
  border-bottom: 1px solid var(--border-ring);
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
td {{
  border-bottom: 1px solid rgba(20,22,26,0.06);
  color: var(--ink-secondary);
}}
tr:hover td {{
  background: rgba(20,22,26,0.025);
}}
.odt-table-scroll {{
  width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}}
.odt-narrative {{
  background: transparent;
  border-left: 1px solid var(--border-default);
  border-radius: 0;
  box-shadow: none;
  color: var(--ink-secondary);
  font-size: var(--text-13);
}}
.odt-kpi-explainer {{
  margin-top: 8px;
  padding-left: 10px;
  border-left: 1px solid var(--border-default);
  color: var(--ink-secondary);
  font-size: var(--text-13);
}}
.odt-kpi-explainer b {{
  color: var(--ink-primary);
  font-weight: 600;
}}
.odt-footer {{
  max-width: var(--container-max);
  margin: var(--space-4) auto 0;
  padding: var(--space-6);
  color: var(--ink-muted);
  text-align: left;
  font-size: var(--text-12);
}}
.odt-footer::before {{ display: none; }}
.odt-footer div {{
  margin-bottom: 4px;
  line-height: 1.45;
}}

@media (max-width: 900px) {{
  .odt-nav {{
    height: var(--nav-height);
    padding: 0 var(--space-4);
  }}
  .odt-nav-links {{
    display: flex;
    width: auto;
    padding: 0;
    border-top: 0;
  }}
  .odt-nav-primary {{ display: none; }}
  .odt-nav-browse {{ width: auto; }}
  .odt-nav-browse summary {{
    width: auto;
    justify-content: center;
    font-size: var(--text-13);
  }}
  .odt-nav-panel {{
    position: absolute;
    right: 0;
    width: min(360px, calc(100vw - 32px));
    max-height: calc(100vh - 88px);
    overflow: auto;
  }}
  .odt-nav-panel-grid {{ grid-template-columns: 1fr; }}
  .odt-context-strip {{ padding: 0 var(--space-4); }}
  .odt-context-strip[open] {{
    height: auto;
    overflow: visible;
  }}
  .odt-context-strip summary {{
    justify-content: flex-start;
    overflow: hidden;
    white-space: nowrap;
    cursor: pointer;
  }}
  .odt-context-strip summary span:not(:first-child):not(.odt-ctx-more-label),
  .odt-context-strip .odt-ctx-dot {{
    display: none;
  }}
  .odt-ctx-more-label {{
    display: inline-flex;
    margin-left: auto;
  }}
  .odt-context-more {{ text-align: left; }}
  .odt-container {{ padding: var(--space-6) var(--space-4) var(--space-12); }}
  .odt-header {{ padding: var(--space-8) var(--space-4) var(--space-6); }}
  .odt-kpi-row {{ grid-template-columns: 1fr; }}
  .odt-section {{ margin-bottom: var(--space-16); }}
  table {{
    display: block;
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    white-space: nowrap;
  }}
  .odt-footer {{ padding: var(--space-6) var(--space-4) var(--space-8); }}
}}

@media (prefers-reduced-motion: reduce) {{
  html {{
    scroll-behavior: auto;
  }}
  *,
  *::before,
  *::after {{
    animation: none !important;
    transition: none !important;
  }}
}}

/* === Clinical Summary v2 === */
.cs-verdict {{
  position: relative;
  padding: var(--space-4) var(--space-6);
  border-radius: var(--radius-md);
  margin-bottom: var(--space-6);
  display: flex;
  align-items: center;
  gap: var(--space-4);
  background: var(--bg-surface);
  border: 1px solid var(--border-ring);
}}
.cs-verdict::before,
.cs-verdict::after {{ content: none; }}
.cs-verdict-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--status-critical);
  flex-shrink: 0;
}}
.cs-verdict-text {{
  font-size: var(--text-14);
  color: var(--text-primary);
  line-height: 1.5;
}}
.cs-verdict-text strong {{
  color: var(--text-primary);
  font-weight: 700;
}}

.cs-dev-card,
.cs-finding,
.cs-stat {{
  background: var(--bg-surface);
  border: 1px solid var(--border-ring);
  border-radius: var(--radius-md);
  transition: background var(--duration-fast) ease, border-color var(--duration-fast) ease;
}}
.cs-dev-card:hover,
.cs-finding:hover,
.cs-stat:hover {{
  background: var(--bg-elevated);
  border-color: var(--border-ring);
}}

.cs-dev-grid,
.cs-findings-grid,
.cs-stats-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--space-3);
  margin-bottom: var(--space-6);
  padding-bottom: var(--space-6);
  position: relative;
}}
.cs-dev-grid::after,
.cs-findings-grid::after,
.cs-stats-row::after {{
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: var(--border-ring);
}}
.cs-dev-card,
.cs-finding,
.cs-stat {{
  padding: var(--space-4);
}}
.cs-dev-header,
.cs-finding-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--border-ring);
}}
.cs-dev-label,
.cs-stat-label {{
  font-size: var(--text-12);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-tertiary);
  font-weight: 600;
}}
.cs-dev-pct,
.cs-sev {{
  display: inline-flex;
  align-items: center;
  padding: 2px var(--space-2);
  border-radius: var(--radius-sm);
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: var(--bg-elevated);
  border: 1px solid var(--border-ring);
}}
.cs-dev-pct.critical,
.cs-sev.critical,
.cs-sev.severe {{ color: var(--status-critical); }}
.cs-dev-pct.warning,
.cs-sev.moderate {{ color: var(--status-warning); }}
.cs-dev-pct.info,
.cs-sev.low-normal {{ color: var(--accent-blue); }}
.cs-dev-value {{
  font-size: var(--text-28);
  font-weight: 600;
  line-height: 1.2;
  margin-bottom: var(--space-2);
  color: var(--text-primary);
  letter-spacing: var(--tracking-hero);
  font-variant-numeric: tabular-nums;
  font-feature-settings: var(--tabular-nums);
}}
.cs-dev-value .unit {{
  font-size: var(--text-14);
  color: var(--text-secondary);
  font-weight: 400;
  margin-left: var(--space-1);
}}

.cs-bar {{
  position: relative;
  width: 100%;
  height: 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-ring);
  border-radius: var(--radius-sm);
  overflow: visible;
  margin-bottom: var(--space-1);
}}
.cs-bar::before,
.cs-bar-normal::after,
.cs-bar-marker::before {{ content: none; }}
.cs-bar-normal {{
  position: absolute;
  height: 100%;
  background: var(--status-good);
  border-radius: var(--radius-sm);
  z-index: 1;
  opacity: 0.35;
}}
.cs-bar-marker {{
  position: absolute;
  top: -4px;
  width: 6px;
  height: 20px;
  border-radius: 3px;
  z-index: 3;
  transform: translateX(-3px);
  background: var(--text-tertiary);
}}
.cs-bar-marker.critical {{ background: var(--status-critical); }}
.cs-bar-marker.warning {{ background: var(--status-warning); }}
.cs-bar-marker.info {{ background: var(--accent-blue); }}
.cs-bar-scale {{
  display: flex;
  justify-content: space-between;
  font-size: var(--text-12);
  color: var(--text-tertiary);
  margin-top: var(--space-1);
}}
.cs-bar-context {{
  font-size: var(--text-12);
  color: var(--text-tertiary);
  margin-top: var(--space-1);
}}

.cs-finding-title {{
  font-size: var(--text-14);
  font-weight: 600;
  color: var(--text-primary);
}}
.cs-finding:has(.cs-sev.critical),
.cs-finding:has(.cs-sev.severe),
.cs-finding:has(.cs-sev.moderate),
.cs-finding:has(.cs-sev.low-normal) {{
  border-color: var(--border-ring);
  background: var(--bg-surface);
}}

.cs-metric {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: var(--space-2) 0;
  border-bottom: 1px solid var(--border-ring);
  font-size: var(--text-13);
}}
.cs-metric:last-child {{ border-bottom: none; }}
.cs-metric-name {{ color: var(--text-secondary); }}
.cs-metric-val {{
  font-weight: 600;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
  font-feature-settings: var(--tabular-nums);
}}
.cs-metric-val.critical {{ color: var(--status-critical); }}
.cs-metric-val.warning {{ color: var(--status-warning); }}

.cs-stat {{
  text-align: center;
}}
.cs-stat-number {{
  font-size: var(--text-40);
  font-weight: 600;
  line-height: 1.05;
  margin-bottom: var(--space-1);
  color: var(--text-primary);
  letter-spacing: var(--tracking-hero);
  font-variant-numeric: tabular-nums;
  font-feature-settings: var(--tabular-nums);
}}
.cs-stat-number.critical {{ color: var(--status-critical); }}
.cs-stat-number.warning {{ color: var(--status-warning); }}
.cs-stat-number.info {{ color: var(--accent-blue); }}

.cs-conclusion {{
  position: relative;
  padding: var(--space-4);
  background: var(--bg-surface);
  border: 1px solid var(--border-ring);
  border-radius: var(--radius-md);
  font-size: var(--text-13);
  line-height: 1.5;
  color: var(--text-secondary);
  margin-bottom: 0;
}}
.cs-conclusion strong {{
  color: var(--text-primary);
  background: var(--bg-elevated);
  padding: 2px var(--space-2);
  border-radius: var(--radius-sm);
  font-weight: 600;
  letter-spacing: 0;
}}

.cs-refs {{
  margin-top: var(--space-4);
  border-radius: var(--radius-md);
  overflow: hidden;
}}
.cs-refs summary {{
  display: flex;
  align-items: center;
  gap: var(--space-2);
  cursor: pointer;
  padding: var(--space-3) var(--space-4);
  font-size: var(--text-12);
  font-weight: 600;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  background: var(--bg-surface);
  border: 1px solid var(--border-ring);
  border-radius: var(--radius-md);
  transition: color var(--duration-fast) ease, background var(--duration-fast) ease;
  list-style: none;
  user-select: none;
}}
.cs-refs summary::-webkit-details-marker {{ display: none; }}
.cs-refs summary::before {{
  content: '+';
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  font-size: var(--text-12);
  font-weight: 700;
  color: var(--accent-blue);
  background: var(--bg-elevated);
  border-radius: var(--radius-sm);
  flex-shrink: 0;
}}
.cs-refs[open] summary::before {{
  content: '\\2212';
}}
.cs-refs summary:hover,
.cs-refs[open] summary {{
  color: var(--text-primary);
  background: var(--bg-elevated);
}}
.cs-refs .cs-refs-inner {{
  overflow: hidden;
  max-height: 0;
  opacity: 0;
  transition: max-height var(--duration-normal) ease, opacity var(--duration-fast) ease, padding var(--duration-fast) ease;
  padding: 0 var(--space-4);
}}
.cs-refs[open] .cs-refs-inner {{
  max-height: 600px;
  opacity: 1;
  padding: var(--space-3) var(--space-4) var(--space-4);
}}
.cs-refs ol,
.cs-refs ul {{
  margin: 0;
  padding-left: 0;
  list-style: none;
  counter-reset: ref-counter;
}}
.cs-refs li {{
  counter-increment: ref-counter;
  position: relative;
  padding: var(--space-2) var(--space-3) var(--space-2) var(--space-8);
  margin-bottom: var(--space-1);
  font-size: var(--text-12);
  line-height: 1.5;
  color: var(--text-secondary);
  background: var(--bg-surface);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-ring);
  transition: background var(--duration-fast) ease;
}}
.cs-refs li:hover {{
  background: var(--bg-elevated);
}}
.cs-refs li::before {{
  content: counter(ref-counter);
  position: absolute;
  left: var(--space-3);
  top: var(--space-2);
  font-size: var(--text-12);
  font-weight: 700;
  color: var(--accent-blue);
  opacity: 0.9;
}}
.cs-refs li:last-child {{ margin-bottom: 0; }}

/* === Page-specific legacy card overrides === */
.idx-hero,
.idx-panel,
.idx-fact-card,
.idx-priority-card,
.idx-card,
.tx-summary-box,
.tx-kpi,
.weekly-card,
.doctor-summary {{
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-ring) !important;
  border-left: 1px solid var(--border-ring) !important;
  border-radius: var(--radius-md) !important;
  box-shadow: none !important;
}}
.idx-priority-card:hover,
.idx-priority-card:focus-visible,
.idx-card:hover,
.idx-card:focus-visible,
.weekly-card:hover {{
  background: var(--bg-elevated) !important;
  border-color: var(--border-ring) !important;
  transform: none !important;
  box-shadow: none !important;
}}
.weekly-card .status-bar {{
  display: none !important;
}}
.weekly-card .card-header,
.weekly-card .metric-value,
.weekly-card .last-week-label {{
  padding-left: 0 !important;
}}
.idx-hero-kicker,
.idx-fact-label,
.idx-panel-label,
.idx-priority-eyebrow,
.idx-group-chip,
.tx-kpi-label,
.tx-traj-title,
.weekly-card .metric-name,
.odt-section-kicker {{
  font-size: var(--text-12) !important;
  font-weight: 600 !important;
  letter-spacing: var(--tracking-eyebrow) !important;
  text-transform: uppercase !important;
  color: var(--text-tertiary) !important;
}}
.idx-hero-title {{
  font-size: var(--text-28) !important;
  font-weight: 600 !important;
  line-height: 1.2 !important;
  letter-spacing: var(--tracking-hero) !important;
}}
.idx-hero-copy,
.idx-group-copy,
.idx-card-desc,
.idx-priority-reason,
.idx-priority-desc,
.tx-intro,
.tx-kpi-detail,
.weekly-card .metric-unit,
.weekly-card .last-week-label {{
  color: var(--text-secondary) !important;
  line-height: 1.5 !important;
}}
.idx-fact-value,
.idx-card-title,
.idx-priority-title,
.tx-kpi-value,
.weekly-card .metric-value,
.doctor-summary h3,
.doctor-summary li {{
  color: var(--text-primary) !important;
}}
.idx-fact-value,
.tx-kpi-value,
.weekly-card .metric-value {{
  font-variant-numeric: tabular-nums !important;
  font-feature-settings: var(--tabular-nums) !important;
}}
.tx-kpi-value,
.weekly-card .metric-value {{
  font-size: var(--text-28) !important;
  font-weight: 600 !important;
  line-height: 1.05 !important;
  letter-spacing: var(--tracking-hero) !important;
}}
.tx-traj-table th,
.tx-traj-table td {{
  border-color: var(--border-ring) !important;
  color: var(--text-secondary) !important;
  font-variant-numeric: tabular-nums !important;
}}

@media (prefers-reduced-motion: reduce) {{
  html {{
    scroll-behavior: auto;
  }}
  *,
  *::before,
  *::after {{
    animation: none !important;
    transition: none !important;
  }}
}}

/* === Print === */
@media print {{
  .odt-nav, .odt-context-strip {{ display: none; }}
  body {{
    background: var(--bg-page);
    color: var(--text-primary);
  }}
  .odt-header h1,
  .odt-section h2,
  .odt-section h3,
  .odt-kpi-value {{
    color: var(--text-primary);
  }}
  .odt-header .subtitle,
  .odt-section p,
  .odt-kpi-detail,
  td {{
    color: var(--text-secondary);
  }}
  .odt-header .metadata,
  .odt-kpi-label,
  .odt-footer {{
    color: var(--text-tertiary);
  }}
  .odt-header::after {{
    background: var(--border-ring);
  }}
  .odt-section,
  .odt-kpi,
  .odt-narrative {{
    border: 1px solid var(--border-ring);
    background: var(--bg-surface);
  }}
  .odt-section h2,
  th,
  td {{
    border-color: var(--border-ring);
  }}
  th {{
    background: var(--bg-elevated);
    color: var(--text-primary);
  }}
  .odt-footer a {{
    color: var(--accent-blue);
  }}
  .odt-badge {{
    border-color: var(--border-ring);
  }}
}}
</style>"""


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def get_navigation_html(current_report_id: str) -> str:
    """Sticky top navigation with native details fallback and JS keyboard polish."""
    groups: dict[str, list[dict]] = {}
    for r in REPORT_REGISTRY:
        groups.setdefault(r["group"], []).append(r)

    report_lookup = {r["id"]: r for r in REPORT_REGISTRY}

    def _is_active(report: dict) -> bool:
        return report["id"] == current_report_id

    primary_links = []
    for report_id in NAV_PRIMARY_IDS:
        report = report_lookup.get(report_id)
        if report is None:
            continue
        active = " active" if _is_active(report) else ""
        primary_links.append(
            f'<a class="odt-nav-link{active}" data-nav-report-id="{escape(report["id"])}" '
            f'href="{report["file"]}">{escape(report["title"])}</a>'
        )

    group_blocks = []
    preferred_group_order = ["Core", "Clinical", "Advanced", "Comparative", "Statistical", "Context"]
    ordered_group_names = [g for g in preferred_group_order if g in groups]
    ordered_group_names.extend(g for g in groups if g not in ordered_group_names)

    for group_name in ordered_group_names:
        reports = groups[group_name]
        panel_links = []
        for r in reports:
            active = " active" if _is_active(r) else ""
            desc = r.get("desc") or "Clinical research report."
            panel_links.append(
                f'<a class="odt-nav-panel-link{active}" data-nav-report-id="{escape(r["id"])}" href="{r["file"]}">'
                f'<span class="odt-nav-panel-title">{escape(r["title"])}</span>'
                f'<span class="odt-nav-panel-separator" aria-hidden="true"> - </span>'
                f'<span class="odt-nav-panel-desc">{escape(desc)}</span>'
                f'</a>'
            )
        group_blocks.append(
            f'<div class="odt-nav-panel-group">'
            f'<span class="odt-nav-group">{escape(group_name)}</span>'
            f'<div class="odt-nav-panel-links">{"".join(panel_links)}</div>'
            f'</div>'
        )

    total_reports = len(REPORT_REGISTRY)

    return (
        '<nav class="odt-nav">\n'
        '  <a class="odt-nav-brand" href="index.html">'
        '<span class="odt-logo">DT</span>'
        '<span class="odt-brand-text">Digital Twin</span><span class="odt-brand-mark">Oura</span></a>\n'
        f'  <div class="odt-nav-links">'
        f'<div class="odt-nav-primary">{"".join(primary_links)}</div>'
        f'<details class="odt-nav-browse">'
        f'<summary aria-label="Open report navigation">All reports</summary>'
        f'<div class="odt-nav-panel">'
        f'<div class="odt-nav-panel-header">All reports · {total_reports} destinations</div>'
        f'<div class="odt-nav-panel-grid">{"".join(group_blocks)}</div>'
        f'</div>'
        f'</details>'
        f'</div>\n'
        '</nav>'
    )


# ---------------------------------------------------------------------------
# Disclaimer Banner
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _resolve_latest_data_date() -> date:
    """Best-effort latest observed data date across core Oura tables."""
    try:
        conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT MAX(d) FROM ("
            "  SELECT MAX(substr(timestamp,1,10)) AS d FROM oura_heart_rate"
            "  UNION ALL SELECT MAX(day) FROM oura_sleep_periods"
            "  UNION ALL SELECT MAX(date) FROM oura_readiness"
            ")"
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        return datetime.now().date()

    if not row or not row[0]:
        return datetime.now().date()
    return datetime.strptime(row[0], "%Y-%m-%d").date()


def _coerce_date(value: str | date | datetime | None) -> date | None:
    """Normalize string/datetime inputs to a plain date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def disclaimer_banner(post_days: int | None = None) -> str:
    """Compact context strip: data source + confound note in one expandable line."""
    if post_days is None:
        latest = _resolve_latest_data_date()
        post_days = max(0, (latest - TREATMENT_START).days + 1)

    hev_date = HEV_DIAGNOSIS_DATE.strftime("%b %d") if HEV_DIAGNOSIS_DATE else "N/A"
    return (
        '<details class="odt-context-strip">'
        '<summary>'
        '<span>Oura Ring Gen 4 sensor data, not clinical measurements</span>'
        '<span class="odt-ctx-dot" aria-hidden="true">&middot;</span>'
        '<span>N=1 case study, not validated for clinical decisions</span>'
        '<span class="odt-ctx-dot" aria-hidden="true">&middot;</span>'
        f'<span>HEV diagnosed <span class="odt-ctx-hev">{escape(hev_date)}</span>; '
        f'Day {post_days} post-ruxolitinib</span>'
        '<span class="odt-ctx-more-label">More</span>'
        '</summary>'
        '<div class="odt-context-more">'
        'Consumer wearable data can support exploratory review only. '
        'The HEV diagnosis, temporally confounded with treatment start, remains a material confounder.'
        '</div>'
        '</details>'
    )


# ---------------------------------------------------------------------------
# Metric Explainer
# ---------------------------------------------------------------------------


def metric_explainer(name: str, description: str) -> str:
    """Inline explainer for a metric. Use inside sections or tables."""
    return (
        '<div class="odt-kpi-explainer odt-metric-explainer">'
        f'<b>{escape(name)}:</b> {escape(description)}</div>'
    )


# ---------------------------------------------------------------------------
# P-value Formatting - single source of truth for all reports
# ---------------------------------------------------------------------------


def format_p_value(value: float | None, decimals: int = 3) -> str:
    """Format p-values consistently across all reports.

    Returns "N/A" for missing/NaN, "p<0.001" for very small,
    otherwise "p=X.XXX" at the specified decimal precision.
    """
    if value is None:
        return "N/A"
    try:
        import math
        if not math.isfinite(value):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.{decimals}f}"


# ---------------------------------------------------------------------------
# KPI Components
# ---------------------------------------------------------------------------


def _format_kpi_value(value: float | str, decimals: int) -> str:
    """Compact KPI values without forcing tabular figures."""
    if isinstance(value, (int, float)):
        if abs(float(value)) >= 1000:
            return f"{value:,.0f}"
        if decimals <= 0:
            return f"{value:.0f}"
        return f"{value:.{decimals}f}"
    return str(value)


def _status_label(status: str, override: str | None = None) -> str:
    if override is not None:
        return override
    return {
        "normal": "In range",
        "good": "In range",
        "warning": "Watch",
        "serious": "Elevated",
        "critical": "Alert",
        "info": "Info",
    }.get(status, "")


_SPARKLINE_SEQ = 0


def _sparkline_svg(values: list[float] | None, color: str) -> str:
    """Render a KPI sparkline: gradient area fill, rounded caps, end-dot.

    The whole line and its area fill are drawn in the metric's own channel
    hue (`color`), fading to transparent toward the top of the fill so it
    reads as a soft gradient rather than a flat tint. Never fabricates
    points; only real values passed in are plotted, in order.
    """
    if not values:
        return ""
    import math

    clean = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    if len(clean) < 2:
        return ""

    global _SPARKLINE_SEQ
    _SPARKLINE_SEQ += 1
    gradient_id = f"odt-spark-grad-{_SPARKLINE_SEQ}"

    width = 120
    height = 36
    pad = 4
    lo = min(clean)
    hi = max(clean)
    span = hi - lo if hi != lo else 1.0
    step = (width - pad * 2) / (len(clean) - 1)
    points = []
    for idx, value in enumerate(clean):
        x = pad + idx * step
        y = height - pad - ((value - lo) / span) * (height - pad * 2)
        points.append((x, y))

    def _path(segment: list[tuple[float, float]]) -> str:
        start = segment[0]
        rest = " ".join(f"L{x:.1f},{y:.1f}" for x, y in segment[1:])
        return f"M{start[0]:.1f},{start[1]:.1f} {rest}".strip()

    base_path = _path(points)
    last_path = _path(points[-2:])
    x_last, y_last = points[-1]
    area_path = (
        f"M{points[0][0]:.1f},{height - pad:.1f} "
        f"L{points[0][0]:.1f},{points[0][1]:.1f} "
        + " ".join(f"L{x:.1f},{y:.1f}" for x, y in points[1:])
        + f" L{points[-1][0]:.1f},{height - pad:.1f} Z"
    )
    safe_color = escape(color)
    return (
        '<svg class="odt-kpi-trend" viewBox="0 0 120 36" aria-hidden="true" focusable="false">'
        '<defs>'
        f'<linearGradient id="{gradient_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{safe_color}" stop-opacity="0.32"/>'
        f'<stop offset="100%" stop-color="{safe_color}" stop-opacity="0"/>'
        '</linearGradient>'
        '</defs>'
        f'<path d="{area_path}" fill="url(#{gradient_id})"/>'
        f'<path d="{base_path}" fill="none" stroke="{safe_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.55"/>'
        f'<path d="{last_path}" fill="none" stroke="{safe_color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{x_last:.1f}" cy="{y_last:.1f}" r="3" fill="{safe_color}" stroke="{BG_SURFACE}" stroke-width="2"/>'
        '</svg>'
    )


def make_kpi_card(
    label: str,
    value: float | str,
    unit: str = "",
    status: str = "neutral",
    detail: str = "",
    decimals: int = 1,
    explainer: str = "",
    status_label: str | None = None,
    delta: float | None = None,
    delta_label: str = "vs last week",
    good_direction: str = "up",
    trend: list[float] | None = None,
    channel_color: str = ACCENT_BLUE,
) -> str:
    """Single KPI card with neutral border, optional delta, and status chip.

    Args:
        label: Short uppercase label (e.g. "MEAN RMSSD")
        value: Number or string to display large
        unit: Unit suffix (e.g. "ms", "bpm", "%")
        status: "critical", "warning", "normal"/"good", "info", or "neutral"
        detail: Optional small text below the value
        decimals: Decimal places when value is numeric
        explainer: Optional one-liner explaining what this metric means
        status_label: Override display text (e.g. "Low", "Elevated", "Insufficient").
            Pass an empty string to suppress the chip entirely.
        delta: Signed percent delta against a named period.
        delta_label: Label for the delta comparison period.
        good_direction: "up" or "down" indicating which delta direction is clinically good.
        trend: Optional 12-30 point sparkline.
        channel_color: Color for the sparkline last segment.
    """
    val_str = _format_kpi_value(value, decimals)
    unit_html = f'<span class="odt-kpi-unit">{unit}</span>' if unit else ""
    detail_html = (
        f'<div class="odt-kpi-detail">{escape(detail)}</div>' if detail else ""
    )
    explainer_html = (
        f'<div class="odt-kpi-explainer">{escape(explainer)}</div>' if explainer else ""
    )

    normalized_status = "good" if status == "normal" else status
    color = STATUS_COLORS.get(normalized_status, "transparent")
    status_label_html = ""
    label_text = _status_label(status, status_label)
    if normalized_status == "neutral" and status_label is None:
        status_label_html = (
            f'<span class="odt-kpi-status-label odt-kpi-status-label--dot" '
            f'style="--status-color:{TEXT_TERTIARY}" aria-label="Neutral">'
            f'<span class="odt-kpi-status-dot"></span></span>'
        )
    elif label_text:
        status_label_html = (
            f'<span class="odt-kpi-status-label" style="--status-color:{color}">'
            f'<span class="odt-kpi-status-dot"></span>{escape(label_text)}</span>'
        )

    delta_html = ""
    if delta is not None:
        try:
            delta_value = float(delta)
            if abs(delta_value) < 1:
                delta_state = "flat"
                triangle = "&#9656;"
            else:
                is_good = (delta_value > 0 and good_direction == "up") or (
                    delta_value < 0 and good_direction == "down"
                )
                delta_state = "good" if is_good else "bad"
                triangle = "&#9650;" if delta_value > 0 else "&#9660;"
            delta_html = (
                f'<div class="odt-kpi-delta odt-kpi-delta--{delta_state}">'
                f'<span>{triangle}</span><b>{delta_value:+.1f}%</b>'
                f'<em>{escape(delta_label)}</em></div>'
            )
        except (TypeError, ValueError):
            delta_html = ""

    trend_html = _sparkline_svg(trend, channel_color)
    status_cls = f" odt-kpi--{normalized_status}" if normalized_status != "neutral" else ""

    return (
        f'<div class="odt-kpi{status_cls}">'
        f'<div class="odt-kpi-head">'
        f'<div class="odt-kpi-label">{escape(label.rstrip(":"))}</div>'
        f'{status_label_html}</div>'
        f'<div class="odt-kpi-value">{val_str}{unit_html}</div>'
        f'{delta_html}{detail_html}{explainer_html}{trend_html}</div>'
    )


def make_kpi_row(*cards: str) -> str:
    """Wrap KPI cards in a responsive grid row."""
    return f'<div class="odt-kpi-row">{"".join(cards)}</div>'


# ---------------------------------------------------------------------------
# Section Component
# ---------------------------------------------------------------------------


def make_section(title: str, content: str, section_id: str = "") -> str:
    """Wrap content in a report section with kicker, title, and hairline rule."""
    id_attr = f' id="{section_id}"' if section_id else ""
    kicker = section_id.replace("-", " ").upper() if section_id else "REPORT SECTION"
    return (
        f'<section class="odt-section"{id_attr}>'
        f'<div class="odt-section-heading">'
        f'<div class="odt-section-kicker">{escape(kicker)}</div>'
        f'<div class="odt-section-title-row"><h2>{escape(title)}</h2><span></span></div>'
        f'</div>{content}</section>'
    )


def make_chart_panel(title: str, subtitle: str = "", fig_html: str = "") -> str:
    """Wrap a Plotly embed in an external-title chart panel."""
    title_html = (
        f'<div class="odt-chart-panel-title">{escape(title)}</div>' if title else ""
    )
    subtitle_html = (
        f'<div class="odt-chart-panel-subtitle">{escape(subtitle)}</div>'
        if subtitle else ""
    )
    header_html = (
        f'<div class="odt-chart-panel-header">{title_html}{subtitle_html}</div>'
        if title_html or subtitle_html else ""
    )
    compact = " odt-chart-panel--compact" if not header_html else ""
    return f'<div class="odt-chart-panel{compact}">{header_html}{fig_html}</div>'


# ---------------------------------------------------------------------------
# Page Assembly
# ---------------------------------------------------------------------------


def site_head_meta(title: str, description: str, page_file: str = "") -> str:
    """Robots, description, canonical and Open Graph tags shared by every page.

    ``SITE_INDEXABLE`` in config.py is the single switch for indexing.
    ``page_file`` is the path relative to the site root (empty for the homepage).
    """
    url = f"{SITE_URL}/{page_file}" if page_file else f"{SITE_URL}/"
    robots = "index, follow" if SITE_INDEXABLE else "noindex, nofollow"
    image = f"{SITE_URL}/{OG_IMAGE_PATH}"
    t = escape(title)
    d = escape(description)
    return (
        f'<meta name="robots" content="{robots}">\n'
        f'<meta name="description" content="{d}">\n'
        f'<link rel="canonical" href="{url}">\n'
        '<link rel="icon" href="data:,">\n'
        '<meta property="og:type" content="website">\n'
        f'<meta property="og:site_name" content="{escape(SITE_NAME)}">\n'
        f'<meta property="og:title" content="{t}">\n'
        f'<meta property="og:description" content="{d}">\n'
        f'<meta property="og:url" content="{url}">\n'
        f'<meta property="og:image" content="{image}">\n'
        '<meta property="og:image:width" content="1200">\n'
        '<meta property="og:image:height" content="630">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:title" content="{t}">\n'
        f'<meta name="twitter:description" content="{d}">\n'
        f'<meta name="twitter:image" content="{image}">'
    )


def site_footer_links() -> str:
    """The shared identity line: source, tool, programme, companion site."""
    return (
        '<div class="odt-footer-links">'
        f'<a href="{REPO_URL}">Source code on GitHub</a> &middot; '
        'Built with Claude Code &middot; '
        f'<a href="{HEALTH_EQUITY_URL}">TEEI Health Equity</a> &middot; '
        f'Companion project: <a href="{COMPANION_URL}">{escape(COMPANION_LABEL)}</a> &middot; '
        '<a href="how_built.html">How this was built</a> &middot; '
        '<a href="claims.html">Every number, checked</a>'
        '</div>'
    )


def wrap_html(
    title: str,
    body_content: str,
    report_id: str,
    subtitle: str = "",
    header_meta: str | None = None,
    chart_data: dict | None = None,
    extra_css: str = "",
    extra_js: str = "",
    data_end: str | date | datetime | None = None,
    post_days: int | None = None,
) -> str:
    """Assemble a complete HTML page with nav, theme, and optional lazy-load.

    Args:
        title: Report title (shown in header and <title>)
        body_content: Main HTML - KPI rows, sections, chart divs
        report_id: Must match an id in REPORT_REGISTRY for nav highlighting
        subtitle: Optional subtitle below title
        header_meta: Optional header metadata text after the generated timestamp.
            Defaults to PATIENT_LABEL. Pass "" to suppress it.
        chart_data: Dict of {key: plotly_json_str} for IntersectionObserver
            lazy loading. Chart containers should be:
            <div id="chart-{key}" class="chart-box" data-chart="{key}">Loading...</div>
        extra_css: Additional CSS rules (without <style> tags)
        extra_js: Additional JS (without <script> tags)
        data_end: Last observed data date for footer/context strip
        post_days: Inclusive number of post-treatment days represented
    """
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_end_date = _coerce_date(data_end) or _resolve_latest_data_date()
    footer_post_days = (
        max(0, (data_end_date - TREATMENT_START).days + 1)
        if post_days is None else post_days
    )
    escaped_title = escape(title)
    registry_entry = next((r for r in REPORT_REGISTRY if r["id"] == report_id), None)
    page_file = registry_entry["file"].split("#")[0] if registry_entry else ""
    if page_file == "index.html":
        page_file = ""
    page_description = subtitle or (registry_entry["desc"] if registry_entry else SITE_DESCRIPTION)
    head_meta = site_head_meta(f"{title} | {SITE_NAME}", page_description, page_file)

    subtitle_html = (
        f'\n      <div class="subtitle">{subtitle}</div>' if subtitle else ""
    )
    meta_label = PATIENT_LABEL if header_meta is None else header_meta
    meta_items = [f"Generated {generated}"]
    if meta_label:
        meta_items.append(str(meta_label))
    meta_items.append(
        f"Data {DATA_START.strftime('%b %d, %Y')} to {data_end_date.strftime('%b %d, %Y')}"
    )
    meta_items.append(f"Day {footer_post_days} post-treatment")
    metadata_html = (
        f'\n      <div class="metadata">'
        f'{" &middot; ".join(escape(item) for item in meta_items)}</div>'
    )

    extra_style = f"\n<style>\n{extra_css}\n</style>" if extra_css else ""

    # Lazy-load JS for chart_data dict
    chart_js = ""
    if chart_data:
        chart_json = json.dumps(chart_data)
        chart_js = f"""
<script>
const chartData = {chart_json};
const observer = new IntersectionObserver((entries) => {{
  entries.forEach(entry => {{
    if (entry.isIntersecting) {{
      const el = entry.target;
      const key = el.dataset.chart;
      if (chartData[key] && el.dataset.rendered !== 'true') {{
        try {{
          const d = JSON.parse(chartData[key]);
          el.innerHTML = '';
          el.style.height = (d.layout.height || 450) + 'px';
          Plotly.newPlot(el.id, d.data, d.layout, window.__ODT_PLOTLY_CONFIG || {{}}).then((graphDiv) => {{
            window.__odtEnhancePlotly?.(graphDiv);
            Plotly.Plots.resize(graphDiv);
          }});
          el.dataset.rendered = 'true';
        }} catch (e) {{
          el.innerHTML = '<div style="padding:20px;color:{ACCENT_RED}">Error rendering chart</div>';
          console.error('Chart render error:', key, e);
        }}
      }}
    }}
  }});
}}, {{ rootMargin: '200px' }});
document.querySelectorAll('.chart-box').forEach(el => observer.observe(el));
</script>"""

    nav_hash_script = """
<script>
const setupOdtNavigation = () => {
  const browse = document.querySelector(".odt-nav-browse");
  if (!browse) return;
  const summary = browse.querySelector("summary");
  const links = Array.from(browse.querySelectorAll(".odt-nav-panel-link"));

  summary?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      browse.open = !browse.open;
      if (browse.open) links[0]?.focus();
    }
  });

  browse.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      browse.open = false;
      summary?.focus();
      return;
    }
    if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
    event.preventDefault();
    if (!browse.open) browse.open = true;
    const activeIndex = links.indexOf(document.activeElement);
    if (activeIndex === -1) {
      links[event.key === "ArrowDown" ? 0 : links.length - 1]?.focus();
      return;
    }
    const offset = event.key === "ArrowDown" ? 1 : -1;
    links[(activeIndex + offset + links.length) % links.length]?.focus();
  });

  document.addEventListener("click", (event) => {
    if (!browse.contains(event.target)) browse.open = false;
  });
};

const syncRoadmapNavState = () => {
  const path = window.location.pathname || "";
  if (!path.endsWith("/roadmap.html") && !path.endsWith("roadmap.html")) return;

  const activeId = window.location.hash === "#honest" ? "about" : "roadmap";
  const currentLabel = activeId === "about" ? "About" : "Next Steps";

  document.querySelectorAll("[data-nav-report-id='about'], [data-nav-report-id='roadmap']").forEach((el) => {
    el.classList.toggle("active", el.dataset.navReportId === activeId);
  });
};

window.addEventListener("DOMContentLoaded", setupOdtNavigation);
window.addEventListener("DOMContentLoaded", syncRoadmapNavState);
window.addEventListener("hashchange", syncRoadmapNavState);
</script>"""

    extra_script = f"\n<script>\n{extra_js}\n</script>" if extra_js else ""
    plotly_bundle_url = _choose_plotly_bundle(
        body_content + (json.dumps(chart_data) if chart_data else "") + extra_js
    )

    return _defer_plotly_scripts(f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="{BG_PRIMARY}">
{head_meta}
<title>{escaped_title} | {escape(SITE_NAME)}</title>
{_INTER_FONT_LINK}
<script defer src="{plotly_bundle_url}"></script>
{get_plotly_enhancer_js()}
{get_base_css()}{extra_style}
</head>
<body>
{get_navigation_html(report_id)}
{disclaimer_banner(post_days=footer_post_days)}
<main id="main-content" class="odt-main">

<div class="odt-header">
  <h1>{escaped_title}</h1>{subtitle_html}{metadata_html}
</div>

<div class="odt-container">
{body_content}
</div>

<div class="odt-footer">
  <div>Data window: {DATA_START.strftime('%B %d')} to {data_end_date.strftime('%B %d, %Y')} &middot; Post-treatment: day {footer_post_days} on ruxolitinib &middot; Generated: {generated}</div>
  <div>Oura Ring Gen 4 consumer wearable data; N=1 exploratory case study; not a medical device. MIT License &middot; &copy; 2026 <a href="https://theeducationalequalityinstitute.org">The Educational Equality Institute</a> &middot; Oura&reg; is a registered trademark of Oura Health Oy.</div>
  {site_footer_links()}
</div>
</main>
{chart_js}{nav_hash_script}{extra_script}
</body>
</html>""")


_TRACE_TYPE_RE = re.compile(r'\\?"type\\?"\s*:\s*\\?"([a-z0-9]+)\\?"')
# "type" also names layout shapes, axis kinds and updatemenus; none of those need a trace module.
_NOT_TRACE_TYPES = {
    "rect", "line", "circle", "path", "date", "linear", "log", "category",
    "multicategory", "data", "layout", "buttons", "dropdown", "auto", "domain",
}
_BASIC_TRACES = {"scatter", "bar", "pie"}
_CARTESIAN_TRACES = _BASIC_TRACES | {
    "box", "heatmap", "histogram", "histogram2d", "histogram2dcontour", "image",
    "contour", "scatterternary", "violin",
}


def _choose_plotly_bundle(page_source: str) -> str:
    """Return the CDN URL of the smallest Plotly bundle that draws every trace on the page.

    basic (about 1 MB) covers scatter, bar and pie; cartesian (about 2 MB) adds box,
    heatmap, histogram, contour and violin; anything else (3D, WebGL, maps, finance,
    indicators) gets the full bundle. Trace types are read from the serialised figures.
    """
    types = set(_TRACE_TYPE_RE.findall(page_source)) - _NOT_TRACE_TYPES
    if not types or types <= _BASIC_TRACES:
        return PLOTLY_CDN_URL.replace("plotly-", "plotly-basic-")
    if types <= _CARTESIAN_TRACES:
        return PLOTLY_CDN_URL.replace("plotly-", "plotly-cartesian-")
    return PLOTLY_CDN_URL


_INLINE_SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", re.S)


def _defer_plotly_scripts(page: str) -> str:
    """Make every inline script that calls Plotly a module script.

    The Plotly bundle is loaded with ``defer`` so the page paints before the
    3.6 MB library arrives. Module scripts execute after deferred scripts, in
    document order, so a chart script that was written as a classic inline
    script keeps working once it is a module. The enhancer in <head> stays
    classic: it guards on ``window.Plotly`` and re-runs on DOMContentLoaded.
    """
    def _convert(match: re.Match) -> str:
        attrs, body = match.group(1), match.group(2)
        if "Plotly." not in body or "ODT_CONFIG" in body or "module" in attrs:
            return match.group(0)
        return f'<script type="module">{body}</script>'

    return _INLINE_SCRIPT_RE.sub(_convert, page)
