#!/usr/bin/env python3
"""Generate the Dashboard homepage (index.html) for Oura Digital Twin.

Reads JSON metrics from sibling reports to display headline KPIs,
then renders a card grid linking to all 12 analysis reports.

Must run AFTER all other analysis scripts so JSON metrics exist.
"""
import json
import sys
from datetime import date
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR, TREATMENT_START  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import (  # noqa: E402
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    REPORT_REGISTRY,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_AMBER,
    ACCENT_RED,
    ACCENT_PURPLE,
    BG_ELEVATED,
    BG_SURFACE,
    BORDER_SUBTLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
)


def _load_json(name: str) -> dict:
    """Load a JSON metrics file, returning empty dict on failure."""
    path = REPORTS_DIR / name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Report descriptions for dashboard cards
# ---------------------------------------------------------------------------

REPORT_META: dict[str, dict] = {
    "full_analysis": {
        "desc": "Heart rate, HRV, sleep, activity, SpO2, and readiness trends across the full observation window.",
        "icon": "&#9829;",  # heart
    },
    "biomarkers": {
        "desc": "Composite biomarker indices combining multiple Oura signals into research-use summary scores.",
        "icon": "&#9733;",  # star
    },
    "sleep": {
        "desc": "Sleep architecture, staging distribution, efficiency, and circadian rhythm analysis.",
        "icon": "&#9790;",  # moon
    },
    "causal": {
        "desc": "Bayesian causal impact and interrupted time-series analysis of ruxolitinib response.",
        "icon": "&#8694;",  # arrow
    },
    "gvhd": {
        "desc": "Hidden Markov and state-space models predicting GvHD flare probability from wearable signals.",
        "icon": "&#9888;",  # warning
    },
    "spo2": {
        "desc": "SpO2 trend monitoring and bronchiolitis obliterans syndrome screening thresholds.",
        "icon": "&#9673;",  # circle
    },
    "hrv": {
        "desc": "Frequency-domain HRV, Poincare plots, DFA, sample entropy, and autonomic balance metrics.",
        "icon": "&#8766;",  # wave
    },
    "digital_twin": {
        "desc": "Unscented Kalman Filter digital twin tracking latent inflammatory and autonomic states.",
        "icon": "&#9881;",  # gear
    },
    "foundation": {
        "desc": "Chronos foundation model forecasting with prediction intervals and anomaly scoring.",
        "icon": "&#9041;",  # brain-like
    },
    "anomalies": {
        "desc": "Matrix Profile, Isolation Forest, and CUSUM anomaly detection across biometric channels.",
        "icon": "&#9889;",  # lightning
    },
    "3d_dashboard": {
        "desc": "Interactive 3D scatter of sleep, HRV, and activity with treatment phase coloring.",
        "icon": "&#9674;",  # diamond
    },
    "about": {
        "desc": "Methodology, limitations, and honest assessment of what this system can and cannot do.",
        "icon": "&#9432;",  # info
    },
    "roadmap": {
        "desc": "Planned analyses, validation targets, and next steps for the digital twin platform.",
        "icon": "&#10132;",  # right arrow
    },
    "comp_autonomic": {
        "desc": "HRV and resting HR recovery trajectories compared between post-HSCT and post-stroke patients.",
        "icon": "&#9829;",
    },
    "comp_treatment": {
        "desc": "Changepoint detection and pre/post treatment response with Mann-Whitney U tests.",
        "icon": "&#8694;",
    },
    "comp_sleep": {
        "desc": "Sleep architecture, efficiency, and timing compared against clinical benchmarks.",
        "icon": "&#9790;",
    },
    "comp_coupling": {
        "desc": "Activity-recovery coupling analysis: does day N activity predict day N+1 recovery?",
        "icon": "&#9107;",
    },
    "comp_anomalies": {
        "desc": "Anomaly fingerprinting and clustering: how do bad days manifest differently?",
        "icon": "&#9889;",
    },
    "comp_breathing": {
        "desc": "Respiratory-rate trends, week-over-week shifts, and outlier nights against recent baseline.",
        "icon": "&#8767;",
    },
    "comp_temperature": {
        "desc": "Temperature deviation tracking, excursion alerts, and post-treatment change patterns.",
        "icon": "&#9788;",
    },
    "mitch_standalone": {
        "desc": "Post-stroke patient (P2): HRV, HR, sleep, and activity dashboard.",
        "icon": "&#9829;",
    },
    "wenche_standalone": {
        "desc": "Healthy control (P3): baseline HRV, HR, sleep, and activity reference.",
        "icon": "&#9829;",
    },
    "mitch_changepoints": {
        "desc": "Patient 2 changepoint scan of HRV, sleep, and recovery markers around key timeline events.",
        "icon": "&#10697;",
    },
    "weekly": {
        "desc": "One-page weekly tracker with watchpoints, week-over-week deltas, and clinician-style summary text.",
        "icon": "&#128197;",
    },
    "forecast": {
        "desc": "Near-term HRV and heart-rate recovery forecast from the current post-treatment trajectory.",
        "icon": "&#128200;",
    },
    "piecewise_its": {
        "desc": "Piecewise ITS regression with AR(1) errors: two-intervention model with BB date sensitivity analysis.",
        "icon": "&#8982;",  # segmented bar
    },
    "sequential_ci": {
        "desc": "Sequential Bayesian CausalImpact isolating Jakavi and beta-blocker effects in separate runs.",
        "icon": "&#8623;",  # branching arrow
    },
    "placebo": {
        "desc": "Falsification tests at 20 random pre-treatment dates to calibrate false positive rates.",
        "icon": "&#9746;",  # x-mark
    },
    "tau_u": {
        "desc": "Tau-U and NAP effect sizes for single-case experimental design with baseline trend correction.",
        "icon": "&#964;",  # tau
    },
}

GROUP_SUMMARIES: dict[str, str] = {
    "Core": "Daily-use entry points for current status, longitudinal trends, and the interactive 3D overview.",
    "Clinical": "Treatment-response, risk-screening, and forecasting reports most relevant for direct follow-up.",
    "Advanced": "Model-heavy analyses for latent state tracking, anomaly detection, and higher-order physiology.",
    "Individual": "Per-patient standalone dashboards for P2 (post-stroke) and P3 (healthy control).",
    "Comparative": "Cross-patient and cross-domain comparisons that add benchmark context to the primary case.",
    "Statistical": "Inference and falsification modules testing whether intervention timing holds up under scrutiny.",
    "Context": "Method notes, limitations, and roadmap material for interpreting the rest of the dashboard responsibly.",
}

START_HERE_REPORTS: list[dict[str, str]] = [
    {
        "id": "weekly",
        "eyebrow": "Current status",
        "reason": "Fastest way to see this week's watchpoints, deltas, and ruxolitinib day count.",
    },
    {
        "id": "full_analysis",
        "eyebrow": "Full context",
        "reason": "Best overview of baseline, trend, and burden across the full observation window.",
    },
    {
        "id": "piecewise_its",
        "eyebrow": "Intervention effect",
        "reason": "Most direct test of Jakavi and beta-blocker timing, including sensitivity analysis.",
    },
]


def _format_date(value: str | None) -> str:
    """Return a readable date label, preserving unknown inputs."""
    if not value:
        return "Unknown"
    try:
        return date.fromisoformat(value).strftime("%b %d, %Y")
    except ValueError:
        return value


def _report_count(group: str | None = None) -> int:
    """Count reports shown on the homepage, optionally within one group."""
    return sum(
        1
        for report in REPORT_REGISTRY
        if report["id"] != "home" and (group is None or report["group"] == group)
    )


def _report_lookup(report_id: str) -> dict | None:
    """Resolve a report entry from the registry."""
    for report in REPORT_REGISTRY:
        if report["id"] == report_id:
            return report
    return None


def _fallback_description(report: dict) -> str:
    """Provide a reasonable description when card metadata is missing."""
    return (
        f"{report['title']} report in the {report['group'].lower()} section."
    )


def hero_overview() -> str:
    """Landing-page overview with current watchpoints and quick facts."""
    full = _load_json("oura_full_analysis.json")
    weekly = _load_json("weekly_tracker.json")

    data_start = full.get("data_start") or full.get("data_range", {}).get("start")
    data_end = full.get("data_end") or full.get("data_range", {}).get("end")
    n_days = full.get("data_range", {}).get("n_days")
    total_reports = _report_count()
    days_on_rux = weekly.get("days_on_ruxolitinib")

    badges = [
        f'<span class="idx-badge">Exploratory decision support</span>',
        f'<span class="idx-badge">Consumer wearable data</span>',
        f'<span class="idx-badge">{total_reports} linked reports</span>',
    ]
    if days_on_rux is not None:
        badges.append(
            f'<span class="idx-badge">Day {int(days_on_rux)} on ruxolitinib</span>'
        )

    facts = [
        (
            "Window",
            f"{_format_date(data_start)} to {_format_date(data_end)}"
            if data_start and data_end
            else "Unavailable",
        ),
        ("Coverage", f"{int(n_days)} days" if n_days is not None else "Unavailable"),
        ("Latest sync", _format_date(data_end)),
        ("Reports live", str(total_reports)),
    ]

    fact_html = "".join(
        f'<div class="idx-fact-card">'
        f'<div class="idx-fact-label">{escape(label)}</div>'
        f'<div class="idx-fact-value">{escape(value)}</div>'
        f'</div>'
        for label, value in facts
    )

    watchpoints = weekly.get("doctor_summary") or []
    if watchpoints:
        watch_html = "".join(
            f'<li>{escape(item)}</li>' for item in watchpoints[:4]
        )
    else:
        watch_html = (
            "<li>Weekly tracker summary is not available yet. Re-run the report suite to populate live watchpoints.</li>"
        )

    mean_hr = full.get("hr_daily_mean")
    sleep_hours = full.get("sleep_duration_avg_hrs")
    quick_lines = []
    if mean_hr is not None:
        quick_lines.append(f"Mean HR {float(mean_hr):.1f} bpm across the full window.")
    if sleep_hours is not None:
        quick_lines.append(f"Average sleep duration {float(sleep_hours):.1f} h/night.")
    if not quick_lines:
        quick_lines.append("Full-window summary metrics are not available yet.")

    quick_html = "".join(f"<li>{escape(line)}</li>" for line in quick_lines)

    return (
        f'<section class="idx-hero">'
        f'<div class="idx-hero-main">'
        f'<div class="idx-hero-kicker">Oura Digital Twin</div>'
        f'<h1 class="idx-hero-title">Start with the signal, then drill into treatment response and causal evidence.</h1>'
        f'<p class="idx-hero-copy">'
        f'This dashboard organizes Oura ring time-series into a single review surface for weekly status, longitudinal context, intervention follow-up, and statistical stress tests. '
        f'It is designed for exploratory monitoring and hypothesis generation, not as a clinical-grade device.'
        f'</p>'
        f'<div class="idx-badge-row">{"".join(badges)}</div>'
        f'</div>'
        f'<div class="idx-fact-grid">{fact_html}</div>'
        f'</section>'
        f'<div class="idx-summary-grid">'
        f'<div class="idx-panel">'
        f'<div class="idx-panel-label">Current watchpoints</div>'
        f'<ul class="idx-panel-list">{watch_html}</ul>'
        f'</div>'
        f'<div class="idx-panel">'
        f'<div class="idx-panel-label">Quick context</div>'
        f'<ul class="idx-panel-list">{quick_html}</ul>'
        f'</div>'
        f'</div>'
    )


def hero_kpis() -> str:
    """Top-level KPI row from the full analysis JSON."""
    full = _load_json("oura_full_analysis.json")
    if not full:
        return ""

    def _status(key: str, thresholds: tuple) -> str:
        """Assign status based on value and (warn, critical) thresholds."""
        val = full.get(key)
        if val is None:
            return "neutral"
        val = float(val)
        lo, hi = thresholds
        if lo < hi:  # higher is worse (e.g. HR)
            if val >= hi:
                return "critical"
            if val >= lo:
                return "warning"
            return "good"
        else:  # lower is worse (e.g. HRV, sleep score)
            if val <= hi:
                return "critical"
            if val <= lo:
                return "warning"
            return "good"

    cards = []

    rmssd = full.get("rmssd_mean")
    if rmssd is not None:
        cards.append(make_kpi_card(
            "HRV (RMSSD)", float(rmssd), "ms",
            status=_status("rmssd_mean", (20, 15)),
            detail="ESC threshold: 15 ms",
        ))

    hr = full.get("hr_daily_mean")
    if hr is not None:
        cards.append(make_kpi_card(
            "Mean HR", float(hr), "bpm",
            status=_status("hr_daily_mean", (80, 90)),
            detail=f"{full.get('hr_pct_tachycardic', '?')}% days tachycardic",
        ))

    sleep = full.get("sleep_score_mean")
    if sleep is not None:
        cards.append(make_kpi_card(
            "Sleep Score", float(sleep), "",
            status=_status("sleep_score_mean", (70, 60)),
            detail=f"{full.get('sleep_duration_avg_hrs', '?')} hrs avg duration",
        ))

    readiness = full.get("readiness_mean")
    if readiness is not None:
        cards.append(make_kpi_card(
            "Readiness", float(readiness), "",
            status=_status("readiness_mean", (70, 60)),
        ))

    spo2 = full.get("spo2_mean")
    if spo2 is not None:
        cards.append(make_kpi_card(
            "SpO2", float(spo2), "%",
            status="good" if float(spo2) >= 95 else "warning",
        ))

    cv_age = full.get("cv_age_mean")
    if cv_age is not None:
        cards.append(make_kpi_card(
            "CV Age", float(cv_age), "yrs",
            status="warning" if float(cv_age) > 40 else "good",
            detail="Oura cardiovascular age estimate",
        ))

    return make_kpi_row(*cards) if cards else ""


# ---------------------------------------------------------------------------
# Treatment response highlight
# ---------------------------------------------------------------------------

# Metric display order for the treatment-response summary
_TREATMENT_METRIC_ORDER = [
    "hrv_average",
    "hr_lowest",
    "hr_average",
    "efficiency",
    "deep_sleep_hours",
    "steps",
]

# Trajectory metrics to show in the three-period table
_TRAJECTORY_METRICS = ["hrv_average", "hr_lowest", "hr_average"]


def treatment_response_summary() -> str:
    """Prominent summary of Ruxolitinib treatment response findings."""
    data = _load_json("comparative_treatment_response.json")
    henrik = data.get("patients", {}).get("henrik", {})
    metrics = henrik.get("metrics", {})
    three_period = henrik.get("three_period", {})

    if not metrics:
        return ""

    significant_metrics = []
    for key in _TREATMENT_METRIC_ORDER:
        metric = metrics.get(key)
        if metric is None:
            continue
        comparison = metric.get("comparison", {})
        if comparison.get("significant_corrected"):
            significant_metrics.append((key, metric))

    # Build KPI cards for corrected-significant metrics only
    sig_cards = []
    for key, m in significant_metrics:
        comp = m.get("comparison", {})
        if not comp:
            continue

        pct = comp.get("pct_change", 0)
        p_raw = comp.get("mann_whitney_p", 1)
        d_val = abs(comp.get("cohens_d", 0))
        effect = comp.get("effect_label", "")
        display_name = m.get("display_name", key)

        # Direction arrow
        arrow = "&#9650;" if pct > 0 else "&#9660;"

        sig_cards.append(
            f'<div class="tx-kpi">'
            f'<div class="tx-kpi-label">{display_name}</div>'
            f'<div class="tx-kpi-value">'
            f'<span class="tx-arrow">{arrow}</span> '
            f'{abs(pct):.1f}%'
            f'</div>'
            f'<div class="tx-kpi-detail">'
            f'p={p_raw:.3f} &middot; d={d_val:.2f} ({effect})'
            f'</div>'
            f'</div>'
        )

    if not sig_cards:
        return ""

    # Build three-period trajectory table
    traj_rows = ""
    if three_period:
        for key in _TRAJECTORY_METRICS:
            tp = three_period.get(key)
            if tp is None:
                continue
            periods = tp.get("periods", {})
            pre_a = periods.get("pre_acute", {}).get("mean")
            post_a = periods.get("post_acute_pre_rux", {}).get("mean")
            post_r = periods.get("post_rux", {}).get("mean")
            if pre_a is None or post_a is None or post_r is None:
                continue
            unit = tp.get("unit", "")
            name = tp.get("display_name", key)
            traj_rows += (
                f'<tr>'
                f'<td class="tx-traj-label">{name}</td>'
                f'<td>{pre_a:.1f} {unit}</td>'
                f'<td>{post_a:.1f} {unit}</td>'
                f'<td class="tx-traj-final">{post_r:.1f} {unit}</td>'
                f'</tr>'
            )

    traj_html = ""
    if traj_rows:
        traj_html = (
            f'<div class="tx-trajectory">'
            f'<div class="tx-traj-title">Three-Period Trajectory</div>'
            f'<table class="tx-traj-table">'
            f'<thead><tr>'
            f'<th></th>'
            f'<th>Pre-Acute</th>'
            f'<th>Post-Acute</th>'
            f'<th>Post-Rux</th>'
            f'</tr></thead>'
            f'<tbody>{traj_rows}</tbody>'
            f'</table>'
            f'</div>'
        )

    inner = (
        f'<div class="tx-summary-box">'
        f'<p class="tx-intro">'
        f'3-week post-ruxolitinib assessment &mdash; '
        f'{len(significant_metrics)} of {len(metrics)} biometric metrics show statistically significant improvement '
        f'(Mann-Whitney U, Bonferroni-corrected)'
        f'</p>'
        f'<div class="tx-kpi-row">{"".join(sig_cards)}</div>'
        f'{traj_html}'
        f'<a href="comparative_treatment_response.html" class="tx-link">'
        f'View full analysis &#8250;'
        f'</a>'
        f'</div>'
    )

    return make_section("Treatment Response: Ruxolitinib", inner)


def start_here_section() -> str:
    """Curated entry points so the landing page feels navigable."""
    cards = []
    for item in START_HERE_REPORTS:
        report = _report_lookup(item["id"])
        if report is None:
            continue
        meta = REPORT_META.get(report["id"], {})
        desc = meta.get("desc") or _fallback_description(report)
        icon = meta.get("icon", "&#9654;")
        cards.append(
            f'<a href="{report["file"]}" class="idx-priority-card">'
            f'<div class="idx-priority-eyebrow">{escape(item["eyebrow"])}</div>'
            f'<div class="idx-priority-head">'
            f'<div class="idx-priority-icon">{icon}</div>'
            f'<div class="idx-priority-title">{escape(report["title"])}</div>'
            f'</div>'
            f'<div class="idx-priority-reason">{escape(item["reason"])}</div>'
            f'<div class="idx-priority-desc">{escape(desc)}</div>'
            f'</a>'
        )

    if not cards:
        return ""

    return make_section("Start Here", f'<div class="idx-priority-grid">{"".join(cards)}</div>')


def report_cards() -> str:
    """Grid of cards linking to each report, grouped by category."""
    groups: dict[str, list[dict]] = {}
    for r in REPORT_REGISTRY:
        if r["id"] == "home":
            continue
        groups.setdefault(r["group"], []).append(r)

    preferred_order = ["Core", "Clinical", "Advanced", "Comparative", "Statistical", "Context"]
    ordered = [g for g in preferred_order if g in groups]
    ordered.extend(g for g in groups if g not in ordered)

    group_colors = {
        "Core": ACCENT_BLUE,
        "Clinical": ACCENT_RED,
        "Advanced": ACCENT_PURPLE,
        "Comparative": ACCENT_CYAN,
        "Statistical": ACCENT_AMBER,
        "Context": ACCENT_GREEN,
    }

    html_parts = []
    for group_name in ordered:
        reports = groups[group_name]
        color = group_colors.get(group_name, ACCENT_AMBER)
        group_summary = GROUP_SUMMARIES.get(group_name, "Supporting reports in this section.")
        intro = (
            f'<div class="idx-group-intro">'
            f'<div class="idx-group-copy">{escape(group_summary)}</div>'
            f'<div class="idx-group-chip">{_report_count(group_name)} reports</div>'
            f'</div>'
        )

        cards_html = []
        for r in reports:
            meta = REPORT_META.get(r["id"], {})
            desc = meta.get("desc") or _fallback_description(r)
            icon = meta.get("icon", "&#9654;")

            cards_html.append(
                f'<a href="{r["file"]}" class="idx-card" '
                f'style="border-left: 3px solid {color};">'
                f'<div class="idx-card-icon" style="color: {color};">{icon}</div>'
                f'<div class="idx-card-body">'
                f'<div class="idx-card-title">{r["title"]}</div>'
                f'<div class="idx-card-desc">{desc}</div>'
                f'</div>'
                f'<div class="idx-card-arrow">&#8250;</div>'
                f'</a>'
            )

        section_html = intro + f'<div class="idx-card-grid">{"".join(cards_html)}</div>'
        html_parts.append(make_section(group_name, section_html))

    return "\n".join(html_parts)


EXTRA_CSS = f"""
/* Homepage hero */
.idx-hero {{
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(280px, 1fr);
  gap: 18px;
  padding: 22px 24px;
  margin-bottom: 18px;
  background:
    radial-gradient(circle at top right, rgba(34, 211, 238, 0.10), transparent 36%),
    linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(16, 185, 129, 0.04)),
    {BG_ELEVATED};
  border: 1px solid {BORDER_SUBTLE};
  border-radius: 16px;
}}
.idx-hero-kicker {{
  color: {ACCENT_CYAN};
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 10px;
}}
.idx-hero-title {{
  margin: 0 0 10px 0;
  font-size: clamp(1.7rem, 2.5vw, 2.4rem);
  line-height: 1.12;
}}
.idx-hero-copy {{
  margin: 0;
  max-width: 740px;
  color: {TEXT_SECONDARY};
  font-size: 0.97rem;
  line-height: 1.7;
}}
.idx-badge-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}}
.idx-badge {{
  display: inline-flex;
  align-items: center;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid rgba(34, 211, 238, 0.18);
  background: rgba(15, 23, 42, 0.42);
  color: {TEXT_PRIMARY};
  font-size: 0.78rem;
}}
.idx-fact-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}}
.idx-fact-card {{
  padding: 14px 16px;
  border-radius: 12px;
  background: rgba(2, 6, 23, 0.35);
  border: 1px solid {BORDER_SUBTLE};
}}
.idx-fact-label {{
  color: {TEXT_TERTIARY};
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
}}
.idx-fact-value {{
  color: {TEXT_PRIMARY};
  font-size: 0.95rem;
  font-weight: 600;
  line-height: 1.45;
}}
.idx-summary-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 10px;
}}
.idx-panel {{
  padding: 16px 18px;
  border-radius: 12px;
  background: {BG_ELEVATED};
  border: 1px solid {BORDER_SUBTLE};
}}
.idx-panel-label {{
  margin-bottom: 10px;
  color: {TEXT_PRIMARY};
  font-size: 0.82rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}
.idx-panel-list {{
  margin: 0;
  padding-left: 18px;
  color: {TEXT_SECONDARY};
}}
.idx-panel-list li {{
  margin-bottom: 8px;
  line-height: 1.55;
}}

/* Start-here cards */
.idx-priority-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 14px;
}}
.idx-priority-card {{
  display: block;
  padding: 18px 18px 20px 18px;
  border-radius: 14px;
  border: 1px solid {BORDER_SUBTLE};
  background:
    linear-gradient(180deg, rgba(59, 130, 246, 0.06), rgba(15, 23, 42, 0)),
    {BG_ELEVATED};
  text-decoration: none;
  color: {TEXT_PRIMARY};
  transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}}
.idx-priority-card:hover,
.idx-priority-card:focus-visible {{
  transform: translateY(-3px);
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.28);
  border-color: rgba(59, 130, 246, 0.35);
}}
.idx-priority-eyebrow {{
  color: {ACCENT_BLUE};
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 10px;
}}
.idx-priority-head {{
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}}
.idx-priority-icon {{
  width: 38px;
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  background: rgba(59, 130, 246, 0.12);
  color: {ACCENT_CYAN};
  font-size: 1.2rem;
}}
.idx-priority-title {{
  font-size: 1.05rem;
  font-weight: 700;
}}
.idx-priority-reason {{
  color: {TEXT_PRIMARY};
  font-size: 0.92rem;
  line-height: 1.55;
  margin-bottom: 8px;
}}
.idx-priority-desc {{
  color: {TEXT_SECONDARY};
  font-size: 0.82rem;
  line-height: 1.5;
}}

/* Group summaries and report cards */
.idx-group-intro {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}}
.idx-group-copy {{
  color: {TEXT_SECONDARY};
  font-size: 0.85rem;
  line-height: 1.55;
  max-width: 780px;
}}
.idx-group-chip {{
  flex-shrink: 0;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.10);
  border: 1px solid {BORDER_SUBTLE};
  color: {TEXT_TERTIARY};
  font-size: 0.76rem;
}}
.idx-card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
  gap: 14px;
}}
.idx-card {{
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 16px 18px;
  background: {BG_ELEVATED};
  border: 1px solid {BORDER_SUBTLE};
  border-radius: 10px;
  text-decoration: none;
  color: {TEXT_PRIMARY};
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
.idx-card:hover,
.idx-card:focus-visible {{
  transform: translateY(-3px);
  box-shadow: 0 8px 22px rgba(59, 130, 246, 0.12);
  border-color: rgba(59, 130, 246, 0.28);
}}
.idx-card-icon {{
  font-size: 1.6rem;
  flex-shrink: 0;
  width: 36px;
  text-align: center;
}}
.idx-card-body {{
  flex: 1;
  min-width: 0;
}}
.idx-card-title {{
  font-weight: 600;
  font-size: 0.95rem;
  margin-bottom: 4px;
}}
.idx-card-desc {{
  font-size: 0.8rem;
  color: {TEXT_SECONDARY};
  line-height: 1.4;
}}
.idx-card-arrow {{
  font-size: 1.4rem;
  color: {TEXT_TERTIARY};
  flex-shrink: 0;
}}

/* Treatment response summary */
.tx-summary-box {{
  background: {BG_ELEVATED};
  border: 1px solid {BORDER_SUBTLE};
  border-left: 4px solid {ACCENT_GREEN};
  border-radius: 10px;
  padding: 20px 24px;
}}
.tx-intro {{
  color: {TEXT_SECONDARY};
  font-size: 0.85rem;
  line-height: 1.5;
  margin: 0 0 16px 0;
}}
.tx-kpi-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}}
.tx-kpi {{
  background: {BG_SURFACE};
  border: 1px solid {BORDER_SUBTLE};
  border-radius: 8px;
  padding: 14px 16px;
  text-align: center;
}}
.tx-kpi-label {{
  font-size: 0.75rem;
  color: {TEXT_SECONDARY};
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 6px;
}}
.tx-kpi-value {{
  font-size: 1.3rem;
  font-weight: 700;
  color: {ACCENT_GREEN};
}}
.tx-arrow {{
  font-size: 0.85rem;
}}
.tx-kpi-detail {{
  font-size: 0.7rem;
  color: {TEXT_TERTIARY};
  margin-top: 4px;
}}
.tx-trajectory {{
  margin-bottom: 14px;
}}
.tx-traj-title {{
  font-size: 0.8rem;
  font-weight: 600;
  color: {TEXT_SECONDARY};
  margin-bottom: 8px;
}}
.tx-traj-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
}}
.tx-traj-table th {{
  color: {TEXT_TERTIARY};
  font-weight: 500;
  text-align: center;
  padding: 6px 10px;
  border-bottom: 1px solid {BORDER_SUBTLE};
}}
.tx-traj-table th:first-child {{
  text-align: left;
}}
.tx-traj-table td {{
  color: {TEXT_SECONDARY};
  text-align: center;
  padding: 6px 10px;
}}
.tx-traj-label {{
  text-align: left !important;
  color: {TEXT_PRIMARY};
  font-weight: 500;
}}
.tx-traj-final {{
  color: {ACCENT_GREEN} !important;
  font-weight: 600;
}}
.tx-link {{
  display: inline-block;
  font-size: 0.85rem;
  color: {ACCENT_GREEN};
  text-decoration: none;
  font-weight: 500;
}}
.tx-link:hover {{
  text-decoration: underline;
}}

@media (max-width: 920px) {{
  .idx-hero,
  .idx-summary-grid {{
    grid-template-columns: 1fr;
  }}
}}

@media (max-width: 720px) {{
  .idx-group-intro {{
    flex-direction: column;
    align-items: flex-start;
  }}
  .idx-fact-grid {{
    grid-template-columns: 1fr;
  }}
}}
"""


def main() -> None:
    full = _load_json("oura_full_analysis.json")
    post_days = full.get("post_days")
    data_end = full.get("data_end") or full.get("data_range", {}).get("end")

    body = (
        hero_overview()
        + hero_kpis()
        + start_here_section()
        + treatment_response_summary()
        + report_cards()
    )

    html = wrap_html(
        title="Dashboard",
        body_content=body,
        report_id="home",
        subtitle="Oura Digital Twin",
        data_end=data_end,
        post_days=int(post_days) if post_days is not None else None,
        extra_css=EXTRA_CSS,
    )

    out = REPORTS_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
