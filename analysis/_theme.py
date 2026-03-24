"""Dark clinical theme for Oura Digital Twin HTML reports.

Complete design system: Plotly template, CSS, navigation bar, KPI cards,
section components, and full-page assembly. All 11 analysis scripts import
from this module for visual consistency.

Usage:
    from _theme import (
        wrap_html, make_kpi_card, make_kpi_row, make_section,
        disclaimer_banner, metric_explainer, format_p_value,
        METRIC_DESCRIPTIONS, STATUS_COLORS, COLORWAY,
    )
    import plotly.io as pio
    pio.templates.default = "clinical_dark"

    body = make_kpi_row(
        make_kpi_card("RMSSD", 18.3, "ms", status="critical", detail="Below ESC threshold"),
        make_kpi_card("Mean HR", 72, "bpm", status="normal"),
    )
    body += make_section("HRV Trends", fig.to_html(include_plotlyjs=False, full_html=False))
    html = wrap_html("Advanced HRV", body, report_id="hrv")
"""

import json
import sqlite3
import sys
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH,
    FONT_FAMILY,
    PLOTLY_CDN_URL,
    PATIENT_LABEL,
    DATA_START,
    TREATMENT_START,
)

# ---------------------------------------------------------------------------
# Report Registry — 11 reports in 3 groups
# ---------------------------------------------------------------------------

REPORT_REGISTRY = [
    {
        "id": "full_analysis",
        "file": "oura_full_analysis.html",
        "title": "Full Analysis",
        "group": "Core",
    },
    {
        "id": "biomarkers",
        "file": "composite_biomarkers.html",
        "title": "Biomarker Trends",
        "group": "Core",
    },
    {
        "id": "sleep",
        "file": "advanced_sleep_analysis.html",
        "title": "Sleep Analysis",
        "group": "Core",
    },
    {
        "id": "causal",
        "file": "causal_inference_report.html",
        "title": "Causal: Ruxolitinib",
        "group": "Clinical",
    },
    {
        "id": "gvhd",
        "file": "gvhd_prediction_report.html",
        "title": "GvHD Prediction",
        "group": "Clinical",
    },
    {
        "id": "spo2",
        "file": "spo2_bos_screening.html",
        "title": "SpO2 & BOS",
        "group": "Clinical",
    },
    {
        "id": "hrv",
        "file": "advanced_hrv_analysis.html",
        "title": "Advanced HRV",
        "group": "Advanced",
    },
    {
        "id": "digital_twin",
        "file": "digital_twin_report.html",
        "title": "Digital Twin",
        "group": "Advanced",
    },
    {
        "id": "foundation",
        "file": "foundation_model_report.html",
        "title": "Foundation Model",
        "group": "Advanced",
    },
    {
        "id": "anomalies",
        "file": "anomaly_detection_report.html",
        "title": "Anomaly Detection",
        "group": "Advanced",
    },
    {
        "id": "3d_dashboard",
        "file": "oura_3d_dashboard.html",
        "title": "3D Dashboard",
        "group": "Advanced",
    },
    {
        "id": "about",
        "file": "roadmap.html#honest",
        "title": "About",
        "group": "Context",
    },
    {
        "id": "roadmap",
        "file": "roadmap.html#roadmap",
        "title": "Next Steps",
        "group": "Context",
    },
]

# ---------------------------------------------------------------------------
# Color Palette — premium dark theme
# ---------------------------------------------------------------------------

# Backgrounds
BG_PRIMARY = "#0F1117"
BG_SURFACE = "#1A1D27"
BG_ELEVATED = "#242837"

# Text
TEXT_PRIMARY = "#E8E8ED"
TEXT_SECONDARY = "#9CA3AF"
TEXT_TERTIARY = "#6B7280"

# Borders & grid
BORDER_SUBTLE = "#2D3348"
BORDER_DEFAULT = "#374151"

# Accent colors
ACCENT_BLUE = "#3B82F6"
ACCENT_GREEN = "#10B981"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#EF4444"
ACCENT_PURPLE = "#8B5CF6"
ACCENT_CYAN = "#06B6D4"
ACCENT_PINK = "#EC4899"
ACCENT_ORANGE = "#F97316"
ACCENT_INDIGO = "#6366F1"

# Status mapping
STATUS_COLORS = {
    "normal": ACCENT_GREEN,
    "good": ACCENT_GREEN,
    "warning": ACCENT_AMBER,
    "critical": ACCENT_RED,
    "info": ACCENT_BLUE,
    "neutral": "transparent",
}

# Biometric-specific (clinical monitor standard)
C_HR = ACCENT_GREEN  # Heart rate — green on patient monitors
C_SPO2 = ACCENT_CYAN  # SpO2 — cyan on pulse oximeters
C_HRV = ACCENT_PURPLE  # HRV/RMSSD — autonomic nervous system
C_SLEEP = ACCENT_INDIGO  # Sleep — calming, sleep-associated
C_TEMP = ACCENT_ORANGE  # Temperature — warmth association
C_ACTIVITY = "#34D399"  # Activity — energy/movement (light emerald)

# Period/series colors (for treatment effect plots)
C_PRE_TX = TEXT_SECONDARY
C_POST_TX = ACCENT_BLUE
C_BASELINE = "#60A5FA"
C_COUNTERFACTUAL = "#93C5FD"
C_FORECAST = ACCENT_CYAN
C_RUX_LINE = ACCENT_BLUE
C_EFFECT = ACCENT_GREEN

# Plotly colorway (8 colors, colorblind-safe)
COLORWAY = [
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_AMBER,
    ACCENT_PURPLE,
    ACCENT_PINK,
    ACCENT_CYAN,
    ACCENT_ORANGE,
    ACCENT_INDIGO,
]

# Backward-compatible aliases (old config.py light-theme names → dark equivalents)
C_PRIMARY = ACCENT_BLUE
C_SECONDARY = ACCENT_CYAN
C_MUTED = TEXT_SECONDARY
C_LIGHT = "#60A5FA"
C_DARK = "#DBEAFE"
C_ACCENT = ACCENT_BLUE
C_CRITICAL = ACCENT_RED
C_GOOD = ACCENT_GREEN
C_WARNING = ACCENT_AMBER
C_NEUTRAL = TEXT_SECONDARY
C_BG = BG_PRIMARY
C_CARD = BG_SURFACE
C_TEXT = TEXT_PRIMARY
C_GRID = BORDER_SUBTLE
C_TEXT_MUTED = TEXT_SECONDARY
C_BG_LIGHT = BG_ELEVATED
C_CAUTION = ACCENT_AMBER

# ---------------------------------------------------------------------------
# Metric Descriptions — reusable across reports
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
    """Premium dark clinical dashboard Plotly template."""
    template = go.layout.Template()

    # Layout
    template.layout.font = dict(
        family=FONT_FAMILY,
        size=14,
        color=TEXT_PRIMARY,
    )
    template.layout.paper_bgcolor = BG_PRIMARY
    template.layout.plot_bgcolor = BG_SURFACE
    template.layout.hovermode = "x unified"
    template.layout.hoverlabel = dict(
        bgcolor="rgba(30, 35, 50, 0.92)",
        font_size=13,
        font_family=FONT_FAMILY,
        bordercolor="rgba(255, 255, 255, 0.1)",
        namelength=-1,
    )
    template.layout.margin = dict(l=64, r=34, t=112, b=60, pad=4)

    # Title
    template.layout.title = dict(
        font=dict(size=20, color="#FFFFFF", family=FONT_FAMILY),
        x=0.0,
        xanchor="left",
        y=0.98,
        yanchor="top",
        pad=dict(l=10, t=10),
    )

    # Axes
    for axis_name in ("xaxis", "yaxis"):
        axis_obj = getattr(template.layout, axis_name)
        axis_obj.showgrid = True
        axis_obj.gridcolor = "rgba(255, 255, 255, 0.05)"
        axis_obj.gridwidth = 1
        axis_obj.griddash = "dot"
        axis_obj.zeroline = False
        axis_obj.showline = True
        axis_obj.linecolor = BORDER_DEFAULT
        axis_obj.linewidth = 1
        axis_obj.tickfont = dict(size=12, color=TEXT_SECONDARY)
        axis_obj.title = dict(
            font=dict(size=13, color=TEXT_SECONDARY),
            standoff=16,
        )
        axis_obj.automargin = True

    # Legend
    template.layout.legend = dict(
        bgcolor="rgba(26, 29, 39, 0.85)",
        bordercolor="rgba(255, 255, 255, 0.06)",
        borderwidth=1,
        font=dict(size=12, color=TEXT_PRIMARY),
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="left",
        x=0,
        itemsizing="constant",
        tracegroupgap=10,
    )

    # Colorway
    template.layout.colorway = COLORWAY

    # Colorscales
    template.layout.colorscale.sequential = [
        [0, BG_PRIMARY],
        [0.25, "#1E3A5F"],
        [0.5, ACCENT_BLUE],
        [0.75, "#93C5FD"],
        [1.0, "#DBEAFE"],
    ]
    template.layout.colorscale.diverging = [
        [0, ACCENT_RED],
        [0.25, "#FCA5A5"],
        [0.5, "#F3F4F6"],
        [0.75, "#6EE7B7"],
        [1.0, ACCENT_GREEN],
    ]

    # Annotation defaults
    template.layout.annotationdefaults = dict(
        font=dict(size=12, color=TEXT_SECONDARY),
        arrowcolor=TEXT_TERTIARY,
        arrowhead=2,
        arrowwidth=1,
    )

    # Shape defaults (reference bands)
    template.layout.shapedefaults = dict(
        fillcolor="rgba(59, 130, 246, 0.1)",
        line=dict(color="rgba(59, 130, 246, 0.3)", width=1),
    )

    # Trace defaults
    template.data.scatter = [
        go.Scatter(
            line=dict(width=2),
            marker=dict(size=7, line=dict(width=0)),
        )
    ]
    template.data.bar = [
        go.Bar(
            marker=dict(line=dict(width=0), opacity=0.9),
        )
    ]
    template.data.heatmap = [
        go.Heatmap(
            colorscale=[
                [0, BG_PRIMARY],
                [0.25, "#1E3A5F"],
                [0.5, ACCENT_BLUE],
                [0.75, "#93C5FD"],
                [1.0, "#DBEAFE"],
            ],
        )
    ]

    return template


# Auto-register at import time (does NOT set as default — each script opts in)
pio.templates["clinical_dark"] = create_clinical_dark_template()

# ---------------------------------------------------------------------------
# Inter Font Embed
# ---------------------------------------------------------------------------

_INTER_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2'
    '?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
)


def get_plotly_enhancer_js() -> str:
    """Return a small runtime patch that improves chart spacing and legibility."""
    return f"""
<script>
window.__odtEnhancePlotly = function(graphDiv) {{
  if (!window.Plotly || !graphDiv || !graphDiv.layout) return;
  let containerWidth = null;
  try {{
    const chartBox = graphDiv.closest(".chart-box");
    if (chartBox) {{
      chartBox.style.display = "block";
      chartBox.style.width = "100%";
      chartBox.style.overflowX = "hidden";
      containerWidth = chartBox.clientWidth || null;
    }} else {{
      containerWidth = graphDiv.parentElement?.clientWidth || null;
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
      plotContainer.style.display = "block";
      plotContainer.style.width = "100%";
      plotContainer.style.maxWidth = "100%";
    }}

    const svgContainer = graphDiv.querySelector(".svg-container");
    if (svgContainer) {{
      svgContainer.style.marginLeft = "auto";
      svgContainer.style.marginRight = "auto";
      svgContainer.style.display = "block";
      svgContainer.style.width = "100%";
      svgContainer.style.maxWidth = "100%";
    }}
  }} catch (e) {{
    // Non-fatal: continue with relayout adjustments.
  }}

  const layout = graphDiv.layout || {{}};
  const updates = {{}};
  const axisKeys = Object.keys(layout).filter((key) => /^(x|y)axis\\d*$/.test(key));

  axisKeys.forEach((key) => {{
    const axis = layout[key] || {{}};
    updates[`${{key}}.automargin`] = true;
    updates[`${{key}}.tickfont.size`] = Math.max(axis.tickfont?.size || 0, 11);
    updates[`${{key}}.tickfont.color`] = axis.tickfont?.color || "{TEXT_SECONDARY}";
    updates[`${{key}}.title.font.size`] = Math.max(axis.title?.font?.size || 0, 12);
    updates[`${{key}}.title.font.color`] = axis.title?.font?.color || "{TEXT_SECONDARY}";
    updates[`${{key}}.title.standoff`] = Math.max(axis.title?.standoff || 0, 14);
  }});

  updates["font.size"] = Math.max(layout.font?.size || 0, 13);
  updates["legend.font.size"] = Math.max(layout.legend?.font?.size || 0, 12);
  updates["hoverlabel.font.size"] = Math.max(layout.hoverlabel?.font?.size || 0, 12);
  updates["margin.t"] = Math.max(layout.margin?.t || 0, 138);
  updates["margin.b"] = Math.max(layout.margin?.b || 0, 68);
  updates["margin.l"] = Math.max(layout.margin?.l || 0, 64);
  updates["margin.r"] = Math.max(layout.margin?.r || 0, 34);
  updates["autosize"] = true;
  if (typeof containerWidth === "number" && containerWidth > 0) {{
    updates["width"] = Math.floor(containerWidth);
  }}

  if (layout.title) {{
    updates["title.xanchor"] = layout.title.xanchor || "left";
    updates["title.yanchor"] = "top";
    updates["title.y"] =
      typeof layout.title?.y === "number" ? Math.min(layout.title.y, 0.985) : 0.985;
    updates["title.pad.t"] = Math.max(layout.title?.pad?.t || 0, 8);
    updates["title.pad.b"] = Math.max(layout.title?.pad?.b || 0, 14);
  }}

  if (typeof layout.legend?.y === "number" && layout.legend.y < 0) {{
    updates["legend.y"] = Math.min(layout.legend.y, -0.14);
  }}

  if (Array.isArray(layout.annotations) && layout.annotations.length) {{
    updates.annotations = layout.annotations.map((annotation) => {{
      const next = {{ ...annotation }};
      const isSubplotTitle =
        next.showarrow === false &&
        next.xref === "paper" &&
        next.yref === "paper" &&
        typeof next.y === "number" &&
        next.y >= 0.95;
      if (isSubplotTitle) {{
        next.font = {{
          ...(next.font || {{}}),
          size: Math.max(next.font?.size || 0, 13),
          color: "{TEXT_PRIMARY}",
          family: "{FONT_FAMILY}",
        }};
        next.yanchor = "bottom";
        if (typeof next.y === "number") {{
          next.y = Math.min(Math.max(next.y, 1.02), 1.08);
        }} else {{
          next.y = 1.02;
        }}
      }}
      return next;
    }});
  }}

  Plotly.relayout(graphDiv, updates)
    .then(() => Plotly.Plots.resize(graphDiv))
    .catch(() => {{
      try {{ Plotly.Plots.resize(graphDiv); }} catch (_) {{}}
    }});
}};

window.addEventListener("load", () => {{
  window.setTimeout(() => {{
    document.querySelectorAll(".js-plotly-plot").forEach((graphDiv) => {{
      window.__odtEnhancePlotly?.(graphDiv);
    }});
  }}, 120);
  window.setTimeout(() => {{
    document.querySelectorAll(".js-plotly-plot").forEach((graphDiv) => {{
      window.__odtEnhancePlotly?.(graphDiv);
    }});
  }}, 900);
}});

window.addEventListener("resize", () => {{
  if (!window.Plotly) return;
  document.querySelectorAll(".js-plotly-plot").forEach((graphDiv) => {{
    Plotly.Plots.resize(graphDiv);
  }});
}});
</script>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def get_base_css() -> str:
    """Return full <style> block for dark clinical reports."""
    return f"""<style>
:root {{
  /* --- Color palette --- */
  --bg-primary: {BG_PRIMARY};
  --bg-surface: {BG_SURFACE};
  --bg-elevated: {BG_ELEVATED};
  --text-primary: {TEXT_PRIMARY};
  --text-secondary: {TEXT_SECONDARY};
  --text-tertiary: {TEXT_TERTIARY};
  --border-subtle: {BORDER_SUBTLE};
  --border-default: {BORDER_DEFAULT};
  --accent-blue: {ACCENT_BLUE};
  --accent-green: {ACCENT_GREEN};
  --accent-amber: {ACCENT_AMBER};
  --accent-red: {ACCENT_RED};
  --accent-purple: {ACCENT_PURPLE};
  --accent-cyan: {ACCENT_CYAN};
  --accent-pink: {ACCENT_PINK};
  --accent-orange: {ACCENT_ORANGE};
  --accent-indigo: {ACCENT_INDIGO};

  /* --- Type scale --- */
  --text-2xl: 1.75rem;
  --text-xl: 1.25rem;
  --text-lg: 1.125rem;
  --text-base: 0.9375rem;
  --text-sm: 0.875rem;
  --text-xs: 0.75rem;
  --text-2xs: 0.6875rem;
  --text-3xs: 0.625rem;

  /* --- Spacing rhythm --- */
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
  --space-2xl: 48px;
  --space-3xl: 72px;

  /* --- Radii --- */
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --radius-xl: 20px;

  /* --- Layout --- */
  --nav-height: 56px;
  --container-max: 1540px;

  /* --- Transitions --- */
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --duration-fast: 150ms;
  --duration-normal: 250ms;
  --duration-slow: 400ms;

  /* --- Glass morphism base --- */
  --glass-bg: rgba(26, 29, 39, 0.55);
  --glass-border: rgba(255, 255, 255, 0.06);
  --glass-shadow: 0 4px 24px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04);
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

html {{
  color-scheme: dark;
  scroll-behavior: smooth;
}}

::selection {{
  background: rgba(59, 130, 246, 0.3);
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
    radial-gradient(circle at top left, rgba(59, 130, 246, 0.08), transparent 32%),
    radial-gradient(circle at top right, rgba(139, 92, 246, 0.05), transparent 28%),
    linear-gradient(180deg, #0F1117 0%, #0C1018 100%);
  color: var(--text-primary);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-variant-numeric: tabular-nums;
  min-height: 100vh;
}}

/* === Scrollbar (dark theme) === */
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
  background: rgba(15, 17, 23, 0.88);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border-bottom: 1px solid rgba(45, 51, 72, 0.5);
  box-shadow: 0 1px 12px rgba(0, 0, 0, 0.2);
  display: flex;
  align-items: center;
  padding: 0 28px;
  height: var(--nav-height);
  gap: 20px;
}}
.odt-nav-brand {{
  font-size: var(--text-base);
  font-weight: 700;
  color: var(--text-primary);
  text-decoration: none;
  white-space: nowrap;
  display: flex;
  align-items: center;
  gap: 10px;
  letter-spacing: -0.02em;
  transition: opacity var(--duration-fast) ease;
}}
.odt-nav-brand:hover {{ opacity: 0.85; }}
.odt-nav-brand .odt-logo {{
  width: 24px;
  height: 24px;
  border-radius: var(--radius-sm);
  background: linear-gradient(135deg, {ACCENT_BLUE}, {ACCENT_PURPLE});
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--text-2xs);
  font-weight: 800;
  color: white;
  letter-spacing: 0;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
}}
.odt-nav-brand span {{ color: var(--accent-blue); }}
.odt-nav-links {{
  display: flex;
  align-items: center;
  gap: 4px;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
}}
.odt-nav-links::-webkit-scrollbar {{ display: none; }}
.odt-nav-group {{
  font-size: 0.5625rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-tertiary);
  padding: 0 8px 0 16px;
  white-space: nowrap;
  position: relative;
}}
.odt-nav-group::before {{
  content: '';
  position: absolute;
  left: 4px;
  top: 50%;
  transform: translateY(-50%);
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: var(--border-subtle);
}}
.odt-nav-group:first-child::before {{ display: none; }}
.odt-nav-link {{
  font-size: var(--text-xs);
  font-weight: 500;
  color: var(--text-secondary);
  text-decoration: none;
  padding: 16px 10px;
  border-bottom: 2px solid transparent;
  white-space: nowrap;
  transition: color var(--duration-fast) ease, border-color var(--duration-normal) ease, background var(--duration-fast) ease;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0;
}}
.odt-nav-link:hover {{
  color: var(--text-primary);
  background: rgba(255, 255, 255, 0.04);
}}
.odt-nav-link.active {{
  color: var(--accent-blue);
  border-bottom-color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.06);
  box-shadow: 0 1px 0 0 var(--accent-blue);
}}
.odt-nav-toggle {{
  display: none;
  background: none;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 1.125rem;
  cursor: pointer;
  padding: 6px 10px;
  margin-left: auto;
  transition: all var(--duration-fast) ease;
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
  padding: var(--space-xl) 32px var(--space-2xl);
}}

/* === Report Header === */
.odt-header {{
  padding: 52px 32px 36px;
  max-width: var(--container-max);
  margin: 0 auto;
  position: relative;
}}
.odt-header::after {{
  content: '';
  position: absolute;
  bottom: 0;
  left: 32px;
  right: 32px;
  height: 1px;
  background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-purple) 30%, var(--border-subtle) 60%, transparent 100%);
}}
.odt-header h1 {{
  font-size: clamp(2rem, 3vw, 2.75rem);
  font-weight: 800;
  background: linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 50%, #C7D2FE 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 10px;
  letter-spacing: -0.03em;
  line-height: 1.08;
  max-width: 980px;
}}
.odt-header .subtitle {{
  font-size: 1rem;
  font-weight: 500;
  color: var(--text-secondary);
  line-height: 1.65;
  max-width: 760px;
}}
.odt-header .metadata {{
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-variant-numeric: tabular-nums;
}}
.odt-header .metadata::before {{
  content: '';
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent-green);
  box-shadow: 0 0 6px rgba(16, 185, 129, 0.5);
  flex-shrink: 0;
  animation: metadataPulse 3s ease-in-out infinite;
}}
@keyframes metadataPulse {{
  0%, 100% {{ opacity: 0.7; }}
  50% {{ opacity: 1; }}
}}

/* === KPI Cards === */
.odt-kpi-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px;
  margin-bottom: 22px;
}}
.odt-kpi {{
  background: rgba(17, 20, 30, 0.75);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-radius: var(--radius-md);
  padding: 20px 22px 18px;
  border: 1px solid rgba(255, 255, 255, 0.05);
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: transform var(--duration-normal) var(--ease-out),
              box-shadow var(--duration-normal) ease,
              border-color var(--duration-normal) ease;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
}}
.odt-kpi:hover {{
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4);
  border-color: rgba(255, 255, 255, 0.1);
}}
.odt-kpi-status {{
  position: absolute; top: 0; left: 0;
  width: 100%; height: 3px;
}}
.odt-kpi--critical {{ border-left: 3px solid {ACCENT_RED}; }}
.odt-kpi--warning  {{ border-left: 3px solid {ACCENT_AMBER}; }}
.odt-kpi--normal   {{ border-left: 3px solid {ACCENT_GREEN}; }}
.odt-kpi--good     {{ border-left: 3px solid {ACCENT_GREEN}; }}
.odt-kpi--info     {{ border-left: 3px solid {ACCENT_BLUE}; }}
.odt-kpi-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 10px;
}}
.odt-kpi-status-label {{
  display: inline-flex;
  align-items: center;
  font-size: 0.6875rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 3px 10px;
  border-radius: 4px;
  flex-shrink: 0;
  line-height: 1.2;
}}
.odt-kpi-label {{
  font-size: 0.6875rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(255, 255, 255, 0.5);
  font-weight: 600;
}}
.odt-kpi-value {{
  font-size: clamp(1.75rem, 2.4vw, 2.125rem);
  font-weight: 700;
  margin-top: 2px;
  color: #FFFFFF;
  line-height: 1.15;
  letter-spacing: -0.02em;
  overflow-wrap: break-word;
  font-variant-numeric: tabular-nums;
  display: flex;
  align-items: baseline;
  gap: 5px;
  flex-wrap: wrap;
}}
.odt-kpi-unit {{
  font-size: 0.8125rem;
  color: rgba(255, 255, 255, 0.6);
  font-weight: 400;
}}
.odt-kpi-detail {{
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.6);
  margin-top: 8px;
  line-height: 1.45;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}}

/* === Sections === */
.odt-section {{
  background: var(--bg-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-subtle);
  padding: 28px 30px 30px;
  margin-bottom: 22px;
  position: relative;
  scroll-margin-top: 112px;
  transition: border-color var(--duration-normal) ease;
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
}}
.odt-section::before {{
  content: '';
  position: absolute;
  top: 0;
  left: 30px;
  right: 30px;
  height: 2px;
  border-radius: 0 0 2px 2px;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple) 50%, transparent 100%);
  opacity: 0.5;
}}
.odt-section h2 {{
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 20px;
  padding-bottom: 12px;
  padding-left: 16px;
  border-bottom: 1px solid var(--border-subtle);
  border-left: 4px solid var(--accent-blue);
  letter-spacing: -0.01em;
  line-height: 1.2;
}}
.odt-section h3 {{
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-top: 20px;
  margin-bottom: 10px;
}}
.odt-section p {{
  margin-bottom: 18px;
  font-size: 0.96875rem;
  color: var(--text-secondary);
  line-height: 1.75;
}}

/* === Tables === */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: var(--space-md) 0;
  font-size: var(--text-sm);
  font-variant-numeric: tabular-nums;
}}
th {{
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elevated);
  color: var(--text-primary);
  font-weight: 600;
  font-size: var(--text-2xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 2px solid var(--border-default);
}}
td {{
  padding: 11px 14px;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  transition: background var(--duration-fast) ease;
}}
tr:hover td {{ background: rgba(36, 40, 55, 0.5); }}

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
  opacity: 0.5;
  transition: opacity var(--duration-fast) ease;
}}
.js-plotly-plot:hover .plotly .modebar {{ opacity: 1; }}
.js-plotly-plot .plotly .modebar-btn {{ font-size: 14px; }}
/* Prevent subplot-title annotations from being clipped */
.odt-section .js-plotly-plot {{ overflow: visible; }}
.odt-section .plot-container {{ overflow: visible; }}
.odt-section .svg-container {{ overflow: visible !important; }}
.js-plotly-plot .main-svg text {{
  text-rendering: geometricPrecision;
}}

/* === Chart boxes (lazy-loaded) === */
.chart-box {{
  background: var(--bg-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-subtle);
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
  padding: 20px 22px;
  background: rgba(59, 130, 246, 0.06);
  border-left: 3px solid var(--accent-blue);
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
  font-size: 0.875rem;
  color: var(--text-primary);
  line-height: 1.7;
  margin-bottom: var(--space-lg);
  box-shadow: inset 4px 0 12px -4px rgba(59, 130, 246, 0.1);
}}

/* === Context Strip (disclaimer + confound merged) === */
.odt-context-strip {{
  background: rgba(26, 29, 39, 0.7);
  border-bottom: 1px solid var(--border-subtle);
  padding: 10px 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  font-size: var(--text-2xs);
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
  font-size: 0.6875rem;
  color: rgba(255, 255, 255, 0.55);
  margin-top: 6px;
  line-height: 1.45;
}}
.odt-kpi-explainer b {{
  color: rgba(255, 255, 255, 0.5);
  font-weight: 600;
}}

/* === Footer === */
.odt-footer {{
  text-align: center;
  padding: 34px 32px 36px;
  color: var(--text-tertiary);
  font-size: var(--text-xs);
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
  background: linear-gradient(90deg, transparent 0%, var(--border-subtle) 20%, var(--accent-blue) 50%, var(--border-subtle) 80%, transparent 100%);
  opacity: 0.6;
}}
.odt-footer div {{
  margin-bottom: var(--space-sm);
  line-height: 1.5;
}}
.odt-footer div:last-child {{ margin-bottom: 0; }}
.odt-footer .odt-footer-fine {{
  font-size: var(--text-2xs);
  opacity: 0.7;
}}
.odt-footer a {{
  color: var(--accent-blue);
  text-decoration: none;
  transition: color var(--duration-fast) ease;
}}
.odt-footer a:hover {{
  color: #60A5FA;
  text-decoration: underline;
  text-underline-offset: 2px;
}}

/* === Responsive === */
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
    padding: var(--space-sm) 0;
  }}
  .odt-nav-links.open {{ display: flex; }}
  .odt-nav-link {{ padding: var(--space-sm) 0; border-bottom: none; }}
  .odt-nav-link.active {{ background: rgba(59, 130, 246, 0.08); border-radius: var(--radius-sm); padding: var(--space-sm) 10px; }}
  .odt-nav-group {{ padding: var(--space-sm) 0 var(--space-xs); }}
  .odt-nav-group::before {{ display: none; }}
  .odt-context-strip {{ padding: var(--space-sm) 16px; gap: var(--space-sm); flex-direction: column; }}
  .odt-context-strip .odt-ctx-dot {{ display: none; }}
  .odt-container {{ padding: 20px 16px 32px; }}
  .odt-header {{ padding: 34px 16px 24px; }}
  .odt-header::after {{ left: 16px; right: 16px; }}
  .odt-header h1 {{ line-height: 1.1; }}
  .odt-kpi-row {{ grid-template-columns: repeat(2, 1fr); gap: 10px; }}
  .odt-kpi {{ padding: 16px 16px 14px; }}
  .odt-kpi:hover {{ transform: none; }}
  .odt-kpi-value {{ font-size: 1.5rem; }}
  .odt-section {{
    padding: 22px 18px 20px;
    scroll-margin-top: 132px;
  }}
  .odt-section::before {{ left: 16px; right: 16px; }}
  .odt-section h2 {{ margin-bottom: 16px; }}
  .odt-footer::before {{ left: 5%; right: 5%; }}
}}
@media (max-width: 480px) {{
  .odt-kpi-row {{ grid-template-columns: 1fr; }}
  .odt-header h1 {{ font-size: 1.625rem; }}
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

/* === Utility: Glass Morphism === */
.odt-glass {{
  background: var(--glass-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  box-shadow: var(--glass-shadow);
}}

/* === Utility: Gradient Text === */
.odt-gradient-text {{
  background: linear-gradient(135deg, {ACCENT_BLUE}, {ACCENT_PURPLE}, {ACCENT_CYAN});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}

/* === Utility: Glow Effects === */
.odt-glow-blue {{ box-shadow: 0 0 20px rgba(59,130,246,0.15), 0 0 60px rgba(59,130,246,0.05); }}
.odt-glow-green {{ box-shadow: 0 0 20px rgba(16,185,129,0.15), 0 0 60px rgba(16,185,129,0.05); }}
.odt-glow-red {{ box-shadow: 0 0 20px rgba(239,68,68,0.15), 0 0 60px rgba(239,68,68,0.05); }}
.odt-glow-amber {{ box-shadow: 0 0 20px rgba(245,158,11,0.15), 0 0 60px rgba(245,158,11,0.05); }}

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
  padding: 3px 10px;
  border-radius: 20px;
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.04em;
  transition: opacity var(--duration-fast) ease;
}}
.odt-badge:hover {{ opacity: 0.85; }}
.odt-badge-blue {{
  background: rgba(59,130,246,0.12);
  color: #60A5FA;
  border: 1px solid rgba(59,130,246,0.25);
}}
.odt-badge-green {{
  background: rgba(16,185,129,0.12);
  color: #34D399;
  border: 1px solid rgba(16,185,129,0.25);
}}
.odt-badge-red {{
  background: rgba(239,68,68,0.12);
  color: #FCA5A5;
  border: 1px solid rgba(239,68,68,0.25);
}}
.odt-badge-amber {{
  background: rgba(245,158,11,0.12);
  color: #FCD34D;
  border: 1px solid rgba(245,158,11,0.25);
}}

/* === Clinical Summary v2 — Enhanced === */

/* CSS Houdini: register custom angle for animated border */
@property --verdict-angle {{
  syntax: '<angle>';
  initial-value: 0deg;
  inherits: false;
}}

/* --- CS Keyframes --- */
@keyframes csVerdictPulse {{
  0%, 100% {{ opacity: 0.6; box-shadow: 0 0 8px rgba(239,68,68,0.4); }}
  50%      {{ opacity: 1;   box-shadow: 0 0 16px rgba(239,68,68,0.7); }}
}}
@keyframes verdictBorderSpin {{
  to {{ --verdict-angle: 360deg; }}
}}
@keyframes verdictGlow {{
  0%, 100% {{
    box-shadow: 0 0 18px 2px rgba(239,68,68,0.18),
                0 0 40px 4px rgba(239,120,50,0.10),
                0 0 80px 8px rgba(239,68,68,0.06);
  }}
  50% {{
    box-shadow: 0 0 24px 4px rgba(239,68,68,0.28),
                0 0 50px 8px rgba(239,120,50,0.16),
                0 0 90px 12px rgba(239,68,68,0.10);
  }}
}}
@keyframes cs-fade-in-up {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes cs-scale-in {{
  from {{ opacity: 0; transform: scale(0.8); }}
  to   {{ opacity: 1; transform: scale(1.0); }}
}}
@keyframes csMarkerSlideIn {{
  0%   {{ left: 0% !important; opacity: 0; transform: translateX(-3px) scaleY(0.6); }}
  60%  {{ opacity: 1; transform: translateX(-3px) scaleY(1); }}
  100% {{ opacity: 1; transform: translateX(-3px) scaleY(1); }}
}}
@keyframes csMarkerGlow {{
  0%, 100% {{ filter: brightness(1); }}
  50%      {{ filter: brightness(1.2); }}
}}
@keyframes csBarTrackReveal {{
  0%   {{ opacity: 0; transform: scaleX(0); transform-origin: left; }}
  100% {{ opacity: 1; transform: scaleX(1); transform-origin: left; }}
}}

/* --- Verdict Banner — Animated gradient border with outer glow --- */
.cs-verdict {{
  border: none;
  position: relative;
  isolation: isolate;
  padding: 20px 24px;
  border-radius: 12px;
  margin-bottom: 28px;
  display: flex;
  align-items: center;
  gap: 16px;
  background: rgba(15, 17, 23, 0.92);
  box-shadow:
    0 0 18px 2px rgba(239,68,68,0.18),
    0 0 40px 4px rgba(239,120,50,0.10),
    0 0 80px 8px rgba(239,68,68,0.06);
  animation:
    verdictBorderSpin 7s linear infinite,
    verdictGlow 7s ease-in-out infinite;
}}
.cs-verdict::before {{
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 12px;
  padding: 1.5px;
  z-index: -1;
  background: conic-gradient(
    from var(--verdict-angle),
    rgba(239,68,68,0.85), rgba(239,120,50,0.70),
    rgba(220,80,40,0.55), rgba(239,68,68,0.40),
    rgba(200,60,60,0.55), rgba(239,120,50,0.70),
    rgba(239,68,68,0.85)
  );
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask-composite: exclude;
}}
.cs-verdict::after {{
  content: '';
  position: absolute;
  inset: -2px;
  border-radius: 14px;
  z-index: -2;
  opacity: 0.35;
  background: conic-gradient(
    from var(--verdict-angle),
    rgba(239,68,68,0.50), rgba(239,120,50,0.35),
    rgba(239,68,68,0.20), rgba(239,120,50,0.35),
    rgba(239,68,68,0.50)
  );
  filter: blur(10px);
  animation: verdictBorderSpin 7s linear infinite;
}}
.cs-verdict-dot {{
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--accent-red);
  box-shadow: 0 0 8px rgba(239,68,68,0.5);
  animation: csVerdictPulse 2s ease-in-out infinite;
  flex-shrink: 0;
}}
.cs-verdict-text {{
  font-size: 0.9375rem;
  color: var(--text-primary);
  line-height: 1.5;
}}
.cs-verdict-text strong {{
  color: var(--accent-red);
  font-weight: 700;
}}

/* --- Glassmorphism base for all card types --- */
.cs-dev-card, .cs-finding, .cs-stat {{
  background: rgba(30, 34, 49, 0.55);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 2px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03);
  transition: transform 0.2s ease-out, box-shadow 0.2s ease-out, border-color 0.2s ease-out;
  animation: cs-fade-in-up 0.4s ease-out backwards;
}}
/* Staggered entrance — dev cards */
.cs-dev-card:nth-child(1) {{ animation-delay: 0s; }}
.cs-dev-card:nth-child(2) {{ animation-delay: 0.1s; }}
.cs-dev-card:nth-child(3) {{ animation-delay: 0.2s; }}
.cs-dev-card:nth-child(4) {{ animation-delay: 0.3s; }}
/* Staggered entrance — findings */
.cs-finding:nth-child(1) {{ animation-delay: 0.15s; }}
.cs-finding:nth-child(2) {{ animation-delay: 0.25s; }}
.cs-finding:nth-child(3) {{ animation-delay: 0.35s; }}
.cs-finding:nth-child(4) {{ animation-delay: 0.45s; }}
/* Staggered entrance — stats */
.cs-stat:nth-child(1) {{ animation-delay: 0.4s; }}
.cs-stat:nth-child(2) {{ animation-delay: 0.5s; }}
.cs-stat:nth-child(3) {{ animation-delay: 0.6s; }}

/* --- Hover effects --- */
.cs-dev-card:hover {{
  transform: translateY(-2px);
  border-color: rgba(251,191,36,0.35);
  box-shadow: 0 4px 16px rgba(251,191,36,0.12), 0 1px 4px rgba(0,0,0,0.4);
}}
.cs-finding:hover {{
  transform: translateY(-2px);
  border-color: rgba(96,165,250,0.3);
  box-shadow: 0 4px 16px rgba(96,165,250,0.1), 0 1px 4px rgba(0,0,0,0.4);
}}
.cs-stat:hover {{
  transform: translateY(-2px);
  border-color: rgba(45,212,191,0.3);
  box-shadow: 0 4px 16px rgba(45,212,191,0.1), 0 1px 4px rgba(0,0,0,0.4);
}}

/* --- Deviation bar grid --- */
.cs-dev-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
  margin-bottom: 28px;
  padding-bottom: 28px;
  position: relative;
}}
.cs-dev-grid::after {{
  content: '';
  position: absolute;
  bottom: 0; left: 8%; right: 8%;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 15%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.06) 85%, transparent 100%);
}}
.cs-dev-card {{
  background: rgba(30, 34, 49, 0.5);
  border-radius: 10px;
  padding: 16px 18px;
}}
.cs-dev-header {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 2px;
}}
.cs-dev-label {{
  font-size: 0.6875rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-secondary);
  font-weight: 500;
}}
.cs-dev-pct {{
  font-size: 0.6875rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
}}
.cs-dev-pct.critical {{ color: #FCA5A5; background: rgba(239,68,68,0.12); }}
.cs-dev-pct.warning {{ color: #FCD34D; background: rgba(245,158,11,0.12); }}
.cs-dev-pct.info {{ color: #93C5FD; background: rgba(59,130,246,0.12); }}
/* Gradient text on deviation values */
.cs-dev-value {{
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1.2;
  margin-bottom: 8px;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  background-image: linear-gradient(135deg, #94A3B8, #CBD5E1);
}}
.cs-dev-value .unit {{
  font-size: 0.75rem;
  -webkit-text-fill-color: var(--text-tertiary);
  font-weight: 400;
  margin-left: 2px;
}}

/* --- Enhanced deviation bars --- */
.cs-bar {{
  position: relative;
  width: 100%;
  height: 12px;
  background: linear-gradient(90deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.025) 100%);
  border-radius: 6px;
  overflow: visible;
  margin-bottom: 4px;
  animation: csBarTrackReveal 0.5s ease-out both;
}}
.cs-bar::before {{
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 6px;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.3), inset 0 -1px 1px rgba(255,255,255,0.02);
  pointer-events: none;
  z-index: 0;
}}
.cs-bar-normal {{
  position: absolute;
  height: 100%;
  background: linear-gradient(90deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.28) 30%, rgba(52,211,153,0.32) 50%, rgba(16,185,129,0.28) 70%, rgba(16,185,129,0.08) 100%);
  border-radius: 6px;
  z-index: 1;
  transition: opacity 0.3s ease;
}}
.cs-bar-normal::after {{
  content: '';
  position: absolute;
  inset: 1px;
  border-radius: 5px;
  box-shadow: inset 0 0 6px rgba(16,185,129,0.15);
  pointer-events: none;
}}
.cs-bar-marker {{
  position: absolute;
  top: -4px;
  width: 6px;
  height: 20px;
  border-radius: 3px;
  z-index: 3;
  transform: translateX(-3px);
  transition: box-shadow 0.3s ease;
  animation: csMarkerSlideIn 0.8s cubic-bezier(0.16,1,0.3,1) both;
}}
.cs-bar-marker::before {{
  content: '';
  position: absolute;
  top: 1px; left: 1px; right: 1px;
  height: 4px;
  border-radius: 2px 2px 0 0;
  background: linear-gradient(180deg, rgba(255,255,255,0.35) 0%, transparent 100%);
  pointer-events: none;
}}
.cs-dev-card:nth-child(2) .cs-bar-marker {{ animation-delay: 0.06s; }}
.cs-dev-card:nth-child(3) .cs-bar-marker {{ animation-delay: 0.12s; }}
.cs-dev-card:nth-child(4) .cs-bar-marker {{ animation-delay: 0.18s; }}
.cs-bar-marker.critical {{
  background: linear-gradient(180deg, #F87171 0%, #EF4444 40%, #DC2626 100%);
  box-shadow: 0 0 10px rgba(239,68,68,0.7), 0 0 20px rgba(239,68,68,0.3), 0 0 2px rgba(239,68,68,0.9);
  animation: csMarkerSlideIn 0.8s cubic-bezier(0.16,1,0.3,1) both, csMarkerGlow 2.5s ease-in-out 1s infinite;
}}
.cs-bar-marker.critical:hover {{
  box-shadow: 0 0 14px rgba(239,68,68,0.85), 0 0 28px rgba(239,68,68,0.4), 0 0 3px rgba(239,68,68,1);
}}
.cs-bar-marker.warning {{
  background: linear-gradient(180deg, #FBBF24 0%, #F59E0B 40%, #D97706 100%);
  box-shadow: 0 0 10px rgba(245,158,11,0.6), 0 0 18px rgba(245,158,11,0.2), 0 0 2px rgba(245,158,11,0.8);
}}
.cs-bar-marker.warning:hover {{
  box-shadow: 0 0 14px rgba(245,158,11,0.75), 0 0 24px rgba(245,158,11,0.35), 0 0 3px rgba(245,158,11,0.9);
}}
.cs-bar-marker.info {{
  background: linear-gradient(180deg, #60A5FA 0%, #3B82F6 40%, #2563EB 100%);
  box-shadow: 0 0 10px rgba(59,130,246,0.6), 0 0 18px rgba(59,130,246,0.2), 0 0 2px rgba(59,130,246,0.8);
}}
.cs-bar-marker.info:hover {{
  box-shadow: 0 0 14px rgba(59,130,246,0.75), 0 0 24px rgba(59,130,246,0.35), 0 0 3px rgba(59,130,246,0.9);
}}
.cs-bar-scale {{
  display: flex;
  justify-content: space-between;
  font-size: 0.5625rem;
  color: var(--text-tertiary);
  opacity: 0.7;
  margin-top: 1px;
}}
.cs-bar-context {{
  font-size: 0.6875rem;
  color: var(--text-tertiary);
  margin-top: 4px;
}}

/* --- Findings grid --- */
.cs-findings-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
  margin-bottom: 28px;
  padding-bottom: 28px;
  position: relative;
}}
.cs-findings-grid::after {{
  content: '';
  position: absolute;
  bottom: 0; left: 8%; right: 8%;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 15%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.06) 85%, transparent 100%);
}}
.cs-finding {{
  background: rgba(32, 38, 56, 0.5);
  border-radius: 10px;
  padding: 16px 18px;
}}
.cs-finding-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border-subtle);
}}
.cs-finding-title {{
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-primary);
}}
/* Severity badges */
.cs-sev {{
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.5625rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
.cs-sev.critical {{ background: rgba(239,68,68,0.12); color: #FCA5A5; }}
.cs-sev.moderate {{ background: rgba(245,158,11,0.12); color: #FCD34D; }}
.cs-sev.severe   {{ background: rgba(249,115,22,0.12); color: #FDBA74; }}
.cs-sev.low-normal {{ background: rgba(6,182,212,0.12); color: #67E8F9; }}
/* Severity-colored left border accents (:has) */
.cs-finding:has(.cs-sev.critical) {{
  border-left: 3px solid rgba(252,165,165,0.4);
  background: linear-gradient(to right, rgba(252,165,165,0.03), transparent 40%), rgba(32,38,56,0.5);
}}
.cs-finding:has(.cs-sev.severe) {{
  border-left: 3px solid rgba(253,186,116,0.4);
  background: linear-gradient(to right, rgba(253,186,116,0.03), transparent 40%), rgba(32,38,56,0.5);
}}
.cs-finding:has(.cs-sev.moderate) {{
  border-left: 3px solid rgba(252,211,77,0.4);
  background: linear-gradient(to right, rgba(252,211,77,0.03), transparent 40%), rgba(32,38,56,0.5);
}}
.cs-finding:has(.cs-sev.low-normal) {{
  border-left: 3px solid rgba(103,232,249,0.4);
  background: linear-gradient(to right, rgba(103,232,249,0.03), transparent 40%), rgba(32,38,56,0.5);
}}

/* Metrics */
.cs-metric {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 6px 0;
  border-bottom: 1px solid rgba(45,51,72,0.3);
  font-size: 0.8125rem;
}}
.cs-metric:last-child {{ border-bottom: none; }}
.cs-metric-name {{ color: var(--text-secondary); }}
.cs-metric-val {{
  font-weight: 600;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}}
.cs-metric-val.critical {{ color: #FCA5A5; }}
.cs-metric-val.warning  {{ color: #FCD34D; }}

/* --- Stat callouts --- */
.cs-stats-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 28px;
  padding-bottom: 28px;
  position: relative;
}}
.cs-stats-row::after {{
  content: '';
  position: absolute;
  bottom: 0; left: 10%; right: 10%;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.05) 20%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.05) 80%, transparent 100%);
}}
.cs-stat {{
  background: rgba(34, 40, 58, 0.6);
  border: 1px solid rgba(255,255,255,0.08);
  text-align: center;
  padding: 20px 16px;
  border-radius: 10px;
}}
/* Gradient text stat numbers */
.cs-stat-number {{
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 4px;
  font-variant-numeric: tabular-nums;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.cs-stat-number.critical {{ background-image: linear-gradient(135deg, #EF4444, #F97316); }}
.cs-stat-number.warning  {{ background-image: linear-gradient(135deg, #F59E0B, #FCD34D); }}
.cs-stat-number.info     {{ background-image: linear-gradient(135deg, #3B82F6, #06B6D4); }}
.cs-stat-label {{
  font-size: 0.6875rem;
  color: var(--text-secondary);
  line-height: 1.4;
}}
/* Scale-in animation for stat numbers */
.cs-stat .cs-stat-number {{
  animation: cs-scale-in 0.35s ease-out backwards;
}}
.cs-stat:nth-child(1) .cs-stat-number {{ animation-delay: 0.5s; }}
.cs-stat:nth-child(2) .cs-stat-number {{ animation-delay: 0.6s; }}
.cs-stat:nth-child(3) .cs-stat-number {{ animation-delay: 0.7s; }}

/* --- Conclusion --- */
.cs-conclusion {{
  position: relative;
  padding: 18px 24px 18px 27px;
  background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(99,102,241,0.06) 100%);
  border: 1px solid rgba(59,130,246,0.15);
  border-left: 3px solid #6366F1;
  border-radius: 0 8px 8px 0;
  font-size: 0.8125rem;
  line-height: 1.6;
  color: var(--text-secondary);
  margin-bottom: 0;
  animation: cs-fade-in-up 0.4s ease-out 0.7s backwards;
}}
.cs-conclusion strong {{
  color: #93C5FD;
  background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(99,102,241,0.12));
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
  letter-spacing: 0.01em;
}}

/* --- Collapsible references --- */
.cs-refs {{
  margin-top: 16px;
  border-radius: 8px;
  overflow: hidden;
}}
.cs-refs summary {{
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 10px 16px;
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  color: rgba(148,163,184,0.7);
  background: rgba(30,34,49,0.5);
  border-radius: 8px;
  transition: color 0.2s ease, background 0.2s ease;
  list-style: none;
  user-select: none;
}}
.cs-refs summary::-webkit-details-marker {{ display: none; }}
.cs-refs summary::before {{
  content: '+';
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px; height: 18px;
  font-size: 0.75rem;
  font-weight: 700;
  color: #6366F1;
  background: rgba(99,102,241,0.1);
  border-radius: 4px;
  transition: transform 0.3s cubic-bezier(0.4,0,0.2,1), background 0.2s ease;
  flex-shrink: 0;
}}
.cs-refs[open] summary::before {{
  content: '\\2212';
  transform: rotate(180deg);
  background: rgba(99,102,241,0.18);
}}
.cs-refs summary:hover {{
  color: rgba(148,163,184,0.95);
  background: rgba(30,34,49,0.8);
}}
.cs-refs[open] summary {{
  border-radius: 8px 8px 0 0;
  background: rgba(30,34,49,0.7);
}}
.cs-refs .cs-refs-inner {{
  overflow: hidden;
  max-height: 0;
  opacity: 0;
  transition: max-height 0.4s cubic-bezier(0.4,0,0.2,1), opacity 0.3s ease 0.05s, padding 0.3s ease;
  padding: 0 16px;
}}
.cs-refs[open] .cs-refs-inner {{
  max-height: 600px;
  opacity: 1;
  padding: 14px 16px 16px;
}}
.cs-refs ol, .cs-refs ul {{
  margin: 0;
  padding-left: 0;
  list-style: none;
  counter-reset: ref-counter;
}}
.cs-refs li {{
  counter-increment: ref-counter;
  position: relative;
  padding: 8px 12px 8px 36px;
  margin-bottom: 4px;
  font-size: 0.75rem;
  line-height: 1.55;
  color: rgba(148,163,184,0.75);
  background: rgba(30,34,49,0.35);
  border-radius: 6px;
  border-left: 2px solid rgba(99,102,241,0.15);
  transition: background 0.15s ease, border-color 0.15s ease;
}}
.cs-refs li:hover {{
  background: rgba(30,34,49,0.6);
  border-left-color: rgba(99,102,241,0.35);
}}
.cs-refs li::before {{
  content: counter(ref-counter);
  position: absolute;
  left: 10px; top: 8px;
  font-size: 0.6875rem;
  font-weight: 700;
  color: #6366F1;
  opacity: 0.7;
  font-variant-numeric: tabular-nums;
}}
.cs-refs li:last-child {{ margin-bottom: 0; }}

/* --- Accessibility: disable all CS animations --- */
@media (prefers-reduced-motion: reduce) {{
  .cs-dev-card, .cs-finding, .cs-stat, .cs-conclusion {{ animation: none; }}
  .cs-stat .cs-stat-number {{ animation: none; }}
  .cs-verdict {{ animation: none; }}
  .cs-verdict::before, .cs-verdict::after {{ animation: none; }}
  .cs-bar {{ animation: none; }}
  .cs-bar-marker {{ animation: none; }}
}}

/* === Print === */
@media print {{
  .odt-nav, .odt-context-strip {{ display: none; }}
  body {{ background: white; color: #111; }}
  .odt-header h1 {{ color: #111; }}
  .odt-header .subtitle {{ color: #333; }}
  .odt-header .metadata {{ color: #555; }}
  .odt-header::after {{ background: #ccc; }}
  .odt-section, .odt-kpi {{ border: 1px solid #ddd; background: white; }}
  .odt-section h2 {{ color: #111; border-bottom-color: #ddd; }}
  .odt-section h3 {{ color: #222; }}
  .odt-section p {{ color: #333; }}
  .odt-kpi-value {{ color: #111; }}
  .odt-kpi-label {{ color: #555; }}
  .odt-kpi-detail {{ color: #444; }}
  th {{ background: #f3f4f6; color: #111; border-bottom-color: #ccc; }}
  td {{ color: #333; border-bottom-color: #eee; }}
  .odt-footer {{ color: #666; border-top-color: #ddd; }}
  .odt-footer a {{ color: #2563EB; }}
  .odt-narrative {{ background: #f0f4ff; border-left-color: #2563EB; color: #111; }}
  .clinical-subsection {{ background: #f9f9f9; border-color: #ddd; }}
  .clinical-patient {{ background: #f3f4f6; }}
  .clinical-severity {{ border-color: #ccc; }}
  .odt-badge {{ border-color: #ccc; }}
}}
</style>"""


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def get_navigation_html(current_report_id: str) -> str:
    """Sticky top navigation bar with grouped report links."""
    groups: dict[str, list[dict]] = {}
    for r in REPORT_REGISTRY:
        groups.setdefault(r["group"], []).append(r)

    links = []
    preferred_group_order = ["Core", "Clinical", "Advanced", "Context"]
    ordered_group_names = [g for g in preferred_group_order if g in groups]
    ordered_group_names.extend(g for g in groups if g not in ordered_group_names)

    for group_name in ordered_group_names:
        reports = groups[group_name]
        links.append(f'<span class="odt-nav-group">{group_name}</span>')
        for r in reports:
            active = " active" if r["id"] == current_report_id else ""
            links.append(
                f'<a class="odt-nav-link{active}" href="{r["file"]}">{r["title"]}</a>'
            )

    return (
        '<nav class="odt-nav">\n'
        '  <a class="odt-nav-brand" href="oura_full_analysis.html">'
        '<span class="odt-logo">DT</span>'
        "Oura <span>Digital Twin</span></a>\n"
        '  <button class="odt-nav-toggle" '
        "onclick=\"let n=this.nextElementSibling;n.classList.toggle('open');"
        "this.setAttribute('aria-expanded',n.classList.contains('open'))\" "
        'aria-label="Menu" aria-expanded="false">&#9776;</button>\n'
        f'  <div class="odt-nav-links">{"".join(links)}</div>\n'
        "</nav>"
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
    """Compact context strip: data source + confound note in one line."""
    if post_days is None:
        latest = _resolve_latest_data_date()
        post_days = max(0, (latest - TREATMENT_START).days + 1)

    return (
        '<div class="odt-context-strip">'
        '<span class="odt-ctx-item">Oura Ring Gen 4 sensor data — not clinical measurements</span>'
        '<span class="odt-ctx-dot"></span>'
        '<span class="odt-ctx-item">N=1 case study — not validated for clinical decisions</span>'
        '<span class="odt-ctx-dot"></span>'
        '<span class="odt-ctx-item warn">HEV diagnosed Mar 18; interpret findings cautiously '
        f"in this Day {post_days} post-ruxolitinib window</span>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Metric Explainer
# ---------------------------------------------------------------------------


def metric_explainer(name: str, description: str) -> str:
    """Inline explainer for a metric. Use inside sections or tables."""
    return f'<div class="odt-kpi-explainer"><b>{name}:</b> {description}</div>'


# ---------------------------------------------------------------------------
# P-value Formatting — single source of truth for all reports
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


def make_kpi_card(
    label: str,
    value: float | str,
    unit: str = "",
    status: str = "neutral",
    detail: str = "",
    decimals: int = 1,
    explainer: str = "",
    status_label: str = "",
) -> str:
    """Single KPI card with optional colored status bar on the right edge.

    Args:
        label: Short uppercase label (e.g. "MEAN RMSSD")
        value: Number or string to display large
        unit: Unit suffix (e.g. "ms", "bpm", "%")
        status: "critical", "warning", "normal"/"good", "info", or "neutral"
        detail: Optional small text below the value
        decimals: Decimal places when value is numeric
        explainer: Optional one-liner explaining what this metric means
        status_label: Override display text (e.g. "Low", "Elevated", "Insufficient")
    """
    if isinstance(value, (int, float)):
        val_str = f"{value:.{decimals}f}"
    else:
        val_str = str(value)

    color = STATUS_COLORS.get(status, "transparent")
    status_bar = (
        f'<div class="odt-kpi-status" style="background:{color}"></div>'
        if status != "neutral"
        else ""
    )
    unit_html = f'<span class="odt-kpi-unit">{unit}</span>' if unit else ""
    detail_html = f'<div class="odt-kpi-detail">{detail}</div>' if detail else ""
    explainer_html = (
        f'<div class="odt-kpi-explainer">{explainer}</div>' if explainer else ""
    )

    # Status text label with colored badge
    STATUS_LABELS = {
        "critical": ("Critical", "color:#FCA5A5;background:rgba(239,68,68,0.15)"),
        "warning": ("Abnormal", "color:#FCD34D;background:rgba(245,158,11,0.15)"),
        "normal": ("Normal", "color:#34D399;background:rgba(16,185,129,0.15)"),
        "good": ("Normal", "color:#34D399;background:rgba(16,185,129,0.15)"),
        "info": ("Info", "color:#93C5FD;background:rgba(59,130,246,0.15)"),
    }
    status_label_html = ""
    if status_label:
        _, label_style = STATUS_LABELS.get(status, ("", ""))
        status_label_html = (
            f'<span class="odt-kpi-status-label" style="{label_style}">'
            f"{status_label}</span>"
        )
    elif status in STATUS_LABELS:
        label_text, label_style = STATUS_LABELS[status]
        status_label_html = (
            f'<span class="odt-kpi-status-label" style="{label_style}">'
            f"{label_text}</span>"
        )

    status_cls = f" odt-kpi--{status}" if status != "neutral" else ""

    return (
        f'<div class="odt-kpi{status_cls}">{status_bar}'
        f'<div class="odt-kpi-head">'
        f'<div class="odt-kpi-label">{label}</div>'
        f"{status_label_html}</div>"
        f'<div class="odt-kpi-value">{val_str}{unit_html}</div>'
        f"{detail_html}{explainer_html}</div>"
    )


def make_kpi_row(*cards: str) -> str:
    """Wrap KPI cards in a responsive grid row."""
    return f'<div class="odt-kpi-row">{"".join(cards)}</div>'


# ---------------------------------------------------------------------------
# Section Component
# ---------------------------------------------------------------------------


def make_section(title: str, content: str, section_id: str = "") -> str:
    """Wrap content in a styled card section with title."""
    id_attr = f' id="{section_id}"' if section_id else ""
    return f'<div class="odt-section"{id_attr}><h2>{title}</h2>{content}</div>'


# ---------------------------------------------------------------------------
# Page Assembly
# ---------------------------------------------------------------------------


def wrap_html(
    title: str,
    body_content: str,
    report_id: str,
    subtitle: str = "",
    header_meta: str | None = None,
    chart_data: dict | None = None,
    extra_css: str = "",
    extra_js: str = "",
    data_start: str | date | datetime | None = None,
    data_end: str | date | datetime | None = None,
    post_days: int | None = None,
) -> str:
    """Assemble a complete HTML page with nav, theme, and optional lazy-load.

    Args:
        title: Report title (shown in header and <title>)
        body_content: Main HTML — KPI rows, sections, chart divs
        report_id: Must match an id in REPORT_REGISTRY for nav highlighting
        subtitle: Optional subtitle below title
        header_meta: Optional header metadata text after the generated timestamp.
            Defaults to PATIENT_LABEL. Pass "" to suppress it.
        chart_data: Dict of {key: plotly_json_str} for IntersectionObserver
            lazy loading. Chart containers should be:
            <div id="chart-{key}" class="chart-box" data-chart="{key}">Loading...</div>
        extra_css: Additional CSS rules (without <style> tags)
        extra_js: Additional JS (without <script> tags)
        data_start: First observed data date for footer/context strip
        data_end: Last observed data date for footer/context strip
        post_days: Inclusive number of post-treatment days represented
    """
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_start_date = _coerce_date(data_start) or DATA_START
    data_end_date = _coerce_date(data_end) or _resolve_latest_data_date()
    footer_post_days = (
        max(0, (data_end_date - TREATMENT_START).days + 1)
        if post_days is None
        else post_days
    )

    subtitle_html = (
        f'\n      <div class="subtitle">{subtitle}</div>' if subtitle else ""
    )
    meta_label = PATIENT_LABEL if header_meta is None else header_meta
    meta_suffix = f" &middot; {meta_label}" if meta_label else ""
    metadata_html = (
        f'\n      <div class="metadata">Generated {generated}{meta_suffix}</div>'
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
          Plotly.newPlot(el.id, d.data, d.layout, {{
            responsive: true,
            displayModeBar: true,
            scrollZoom: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false,
          }}).then((graphDiv) => {{
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

    extra_script = f"\n<script>\n{extra_js}\n</script>" if extra_js else ""

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="{BG_PRIMARY}">
<meta name="robots" content="noindex, nofollow">
<title>{title} — Oura Digital Twin</title>
{_INTER_FONT_LINK}
<script src="{PLOTLY_CDN_URL}"></script>
{get_base_css()}{extra_style}
</head>
<body>
{get_navigation_html(report_id)}
{disclaimer_banner(post_days=footer_post_days)}

<div class="odt-header">
  <h1>{title}</h1>{subtitle_html}{metadata_html}
</div>

<div class="odt-container">
{body_content}
</div>

<div class="odt-footer">
  <div>All metrics derived from Oura Ring Gen 4 consumer wearable data. Not clinical-grade measurements.</div>
  <div>Single-patient case study (N=1). Not validated for clinical decision-making. Not a medical device.</div>
  <div>Data: {data_start_date.strftime("%B %-d")} &ndash; {data_end_date.strftime("%B %-d, %Y")} &middot; Post-intervention: {footer_post_days} days (ruxolitinib, {TREATMENT_START.strftime("%B %-d")})</div>
  <div>Open source under MIT License &middot; &copy; 2026 <a href="https://theeducationalequalityinstitute.org">The Educational Equality Institute</a> &middot; <a href="https://github.com/The-Educational-Equality-Institute/oura-hsct-digital-twin">GitHub</a></div>
  <div class="odt-footer-fine">Updated daily at 06:15 CET &middot; Last generated: {generated}</div>
  <div class="odt-footer-fine">This project is not affiliated with, endorsed by, or sponsored by Oura Health Oy. Oura&reg; is a registered trademark of Oura Health Oy.</div>
</div>
{get_plotly_enhancer_js()}{chart_js}{extra_script}
</body>
</html>"""
