#!/usr/bin/env python3
"""Generate the Dashboard homepage (index.html) for Oura Digital Twin.

Reads JSON metrics from sibling reports to display headline KPIs,
then renders a card grid linking to all 12 analysis reports.

Must run AFTER all other analysis scripts so JSON metrics exist.
"""
import json
import sys
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
        "desc": "Composite biomarker indices combining multiple Oura signals into clinical-grade scores.",
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
}


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
            detail=f"ESC threshold: 15 ms",
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


def report_cards() -> str:
    """Grid of cards linking to each report, grouped by category."""
    groups: dict[str, list[dict]] = {}
    for r in REPORT_REGISTRY:
        if r["id"] == "home":
            continue
        groups.setdefault(r["group"], []).append(r)

    preferred_order = ["Core", "Clinical", "Advanced", "Context"]
    ordered = [g for g in preferred_order if g in groups]
    ordered.extend(g for g in groups if g not in ordered)

    group_colors = {
        "Core": ACCENT_BLUE,
        "Clinical": ACCENT_RED,
        "Advanced": ACCENT_PURPLE,
        "Context": ACCENT_GREEN,
    }

    html_parts = []
    for group_name in ordered:
        reports = groups[group_name]
        color = group_colors.get(group_name, ACCENT_AMBER)

        cards_html = []
        for r in reports:
            meta = REPORT_META.get(r["id"], {})
            desc = meta.get("desc", "")
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

        section_html = f'<div class="idx-card-grid">{"".join(cards_html)}</div>'
        html_parts.append(make_section(group_name, section_html))

    return "\n".join(html_parts)


EXTRA_CSS = f"""
/* Dashboard card grid */
.idx-card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 12px;
}}
.idx-card {{
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 18px;
  background: {BG_ELEVATED};
  border-radius: 10px;
  text-decoration: none;
  color: {TEXT_PRIMARY};
  transition: transform 0.15s, box-shadow 0.15s;
}}
.idx-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
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
.idx-hero-text {{
  color: {TEXT_SECONDARY};
  font-size: 0.9rem;
  line-height: 1.6;
  max-width: 720px;
  margin-bottom: 8px;
}}
"""


def main() -> None:
    full = _load_json("oura_full_analysis.json")
    post_days = full.get("post_days")
    data_end = full.get("data_end") or full.get("data_range", {}).get("end")

    hero_text = (
        '<div class="idx-hero-text">'
        "Continuous wearable monitoring via Oura Gen 3. "
        "12 analysis modules run daily, producing interactive HTML reports "
        "with clinical-grade signal processing, causal inference, "
        "and predictive modeling."
        "</div>"
    )

    body = hero_text + hero_kpis() + report_cards()

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
