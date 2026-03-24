#!/usr/bin/env python3
"""Generate the What's Next / Roadmap page for Oura Digital Twin.

Outputs reports/roadmap.html using the shared _theme.py design system.
No database access required — pure static content with real numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import wrap_html, make_section, make_kpi_card, make_kpi_row  # noqa: E402


def subject_cards() -> str:
    """Three subject profile cards."""
    primary = make_kpi_card(
        "Primary Subject",
        "79",
        "days",
        status="info",
        detail="36M &middot; Post-HSCT &middot; Resting HR 79-84 &middot; HRV 9-10 ms",
    )
    mamma = make_kpi_card(
        "Family Control",
        "1",
        "night",
        status="good",
        detail="61F &middot; Healthy &middot; Resting HR 56-68 &middot; Same genetics",
    )
    subject_b = make_kpi_card(
        "Disease Control",
        "34",
        "nights",
        status="info",
        status_label="Recruiting",
        detail="36M &middot; Post-stroke &middot; Sparse data &middot; Restarting nightly wear",
    )
    return make_kpi_row(primary, mamma, subject_b)


def honest_assessment() -> str:
    """What day 8 can and cannot tell us."""
    return """
<h3>What we have</h3>
<p>79 days of continuous Oura data. 67 days pre-intervention baseline.
8 days post-ruxolitinib. 11 core analysis modules plus a roadmap appendix.
A CausalImpact model that
returns a statistically significant treatment signal (see causal inference report for current p-values).</p>

<h3>What that p-value actually means</h3>
<p>The model detected a statistically significant change in the biometric
signal after March 16. It cannot tell us <em>why</em>.</p>

<h3>The problem</h3>
<p>Ruxolitinib started March 16. Hepatitis E was diagnosed March 18.
Two days apart. At day 8, we cannot separate the two.</p>

<table>
<thead>
<tr>
  <th>Signal</th>
  <th>Pre-rux (67 days)</th>
  <th>Post-rux (8 days)</th>
  <th>Could be rux?</th>
  <th>Could be HEV?</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Temperature deviation</td>
  <td>+0.07 °C</td>
  <td>-0.16 °C</td>
  <td>Yes (immunosuppression)</td>
  <td>Yes (acute viral fever resolving)</td>
</tr>
<tr>
  <td>Sleep HR</td>
  <td>85.0 bpm</td>
  <td>81.8 bpm</td>
  <td>Yes (reduced inflammation)</td>
  <td>Yes (acute phase resolving)</td>
</tr>
<tr>
  <td>Resting HR (readiness)</td>
  <td>79.1 bpm</td>
  <td style="color:#F59E0B;">84.2 bpm</td>
  <td>No (went up, not down)</td>
  <td>Yes (HEV-driven tachycardia)</td>
</tr>
<tr>
  <td>HRV (RMSSD)</td>
  <td>9.2 ms</td>
  <td>10.1 ms</td>
  <td>Marginal (still severely depressed)</td>
  <td>Unclear</td>
</tr>
</tbody>
</table>

<p style="margin-top:16px;">The resting HR went <em>up</em> after ruxolitinib, not down. That is
more consistent with acute HEV infection than with JAK inhibitor response.
The temperature drop could be either. The HRV change is within noise at
these sample sizes.</p>

<h3>What resolves this</h3>
<p>Time. HEV is acute. It resolves in weeks. Ruxolitinib is sustained.
If the signals persist and strengthen at day 28 (~April 13), it is
ruxolitinib. If they fade, it was HEV. We cannot know before then.</p>

<p>There is one shortcut: <strong>Oura has the answer now.</strong>
Their cohort contains users on ruxolitinib without HEV. Comparing
those users' autonomic trajectories against this patient's would
resolve the confound immediately. That comparison is impossible with
N=1. It is trivial at cohort scale.</p>
"""


def day1_comparison() -> str:
    """Family control's first night vs primary subject — the contrast that's already visible."""
    return """
<p>Ring received March 23. First full night: March 24. One night of data.</p>

<table>
<thead>
<tr>
  <th>Metric</th>
  <th>Subject A (36M, post-HSCT)</th>
  <th>Family Control (61F, healthy)</th>
  <th>What it means</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Resting HR</td>
  <td>79-84 bpm</td>
  <td style="color:#10B981;">56-68 bpm</td>
  <td>20+ bpm gap. Subject A's autonomic system is measurably damaged.</td>
</tr>
<tr>
  <td>HRV (RMSSD)</td>
  <td>9-10 ms</td>
  <td>TBD (collecting)</td>
  <td>Population median: 49 ms. Subject A is at the 1st percentile. Family control will likely be near normal.</td>
</tr>
<tr>
  <td>Age</td>
  <td>36</td>
  <td>61</td>
  <td>The 61-year-old has a healthier heart rate than the 36-year-old. That is the disease signal.</td>
</tr>
<tr>
  <td>Genetics</td>
  <td colspan="2" style="text-align:center;">~50% shared genome</td>
  <td>Same genetic background eliminates inherited autonomic traits as confounders.</td>
</tr>
</tbody>
</table>

<p style="margin-top:16px;">One night. The contrast is already visible.
The 61-year-old family control rests at 56-68 bpm. The 36-year-old subject,
28 months post-transplant, rests at 79-84 bpm. That gap is unlikely to be
explained by shared baseline alone, but still needs cohort-level confirmation.</p>
"""


def stroke_control() -> str:
    """Subject B — what the disease control adds."""
    return """
<p>36M. Same age as Subject A. Post-stroke with neurovascular autonomic damage.
Has worn an Oura Ring intermittently — 34 nights over 2 years. That is not
enough for meaningful analysis.</p>

<p>Starting now: nightly wear. Once we have 30+ consecutive nights, the
same 11-module pipeline runs on his data. No code changes. Different
config file, different OAuth token, same analyses.</p>

<h3>What this adds</h3>
<p>Subject A has immune-mediated autonomic damage (GVHD, 10 organ systems).
Subject B has neurovascular autonomic damage (post-stroke). Both are 36M.
Same hardware, same software, different pathology.</p>

<p>If the autonomic signatures separate — different HRV patterns, different
sleep architecture, different circadian disruption — that is evidence that
a consumer ring can distinguish between disease types, not just detect
"something is wrong."</p>
"""


def roadmap_table() -> str:
    """Phase roadmap as an HTML table."""
    return """
<table>
<thead>
<tr>
  <th>Phase</th>
  <th>Timeline</th>
  <th>What</th>
</tr>
</thead>
<tbody>
<tr>
  <td><span style="color:#3B82F6;font-weight:700;">v1.0</span></td>
  <td>Now</td>
  <td>Single-patient: 79 days, 8 days post-rux, 11 core modules. HEV confound unresolved.</td>
</tr>
<tr>
  <td><span style="color:#3B82F6;font-weight:700;">v1.1</span></td>
  <td>Continuous (resolves ~April 13)</td>
  <td>Reports update daily. Day 28 is when HEV vs. ruxolitinib should be easier to separate statistically if the current patterns persist.</td>
</tr>
<tr>
  <td><span style="color:#10B981;font-weight:700;">v2.0</span></td>
  <td>April</td>
  <td>Healthy family control (61F). 30+ nights. Same pipeline and shared genetics, providing a cleaner baseline comparison.</td>
</tr>
<tr>
  <td><span style="color:#10B981;font-weight:700;">v2.1</span></td>
  <td>April-May</td>
  <td>Post-stroke control (36M). 30+ consecutive nights. Different pathology, same age.</td>
</tr>
<tr>
  <td><span style="color:#8B5CF6;font-weight:700;">v3.0</span></td>
  <td>Q2 2026</td>
  <td>Multi-user comparison: GVHD vs. stroke vs. healthy. Between-subjects CausalImpact.</td>
</tr>
</tbody>
</table>
"""


def planned_analyses() -> str:
    """List of planned analyses."""
    analyses = [
        (
            "Between-subjects CausalImpact",
            "Same intervention window, three physiologies. "
            "Does ruxolitinib produce a detectable autonomic signal "
            "only in the immune-mediated patient? The controls make this answerable.",
        ),
        (
            "Family genetic control comparison",
            "Mother and son, ~50% shared genome. Baseline HRV, sleep architecture, "
            "circadian patterns. What is inherited and what is disease?",
        ),
        (
            "Disease-specific autonomic signatures",
            "Immune-mediated (GVHD) vs. neurovascular (post-stroke). "
            "Different damage, same ring, same pipeline. Do the signatures separate?",
        ),
        (
            "Cohort-ready pipeline",
            "Any Oura user, any condition. Config file + OAuth token = "
            "full 11-module analysis. Open source.",
        ),
    ]

    items = []
    for title, desc in analyses:
        items.append(
            f'<div style="margin-bottom:20px;">'
            f'<h3 style="margin-bottom:6px;">{title}</h3>'
            f'<p style="margin:0;">{desc}</p>'
            f'</div>'
        )
    return "\n".join(items)


def oura_team_cta() -> str:
    """Call-to-action for Oura's team — factual, not pushy."""
    return """
<p>This pipeline runs in 193 seconds on a single patient. Config file +
OAuth token is all it needs. If your cohort contains even 50 users on
ruxolitinib, you can validate or falsify every finding on this site in
an afternoon. The code is MIT-licensed. The confound table above tells
you exactly which comparisons resolve which questions.</p>
"""


def main():
    body = ""

    # Subjects
    body += make_section("Current Subjects", subject_cards(), section_id="subjects")

    # Honest assessment — the most important section
    body += make_section(
        "What This Is and What It Isn't",
        honest_assessment(),
        section_id="honest",
    )

    # Family control day 1
    body += make_section(
        "Day 1: Family Control",
        day1_comparison(),
        section_id="day1",
    )

    # Stroke control
    body += make_section(
        "Disease Control: Post-Stroke",
        stroke_control(),
        section_id="stroke",
    )

    # Roadmap
    body += make_section("Roadmap", roadmap_table(), section_id="roadmap")

    # Planned analyses
    body += make_section(
        "Planned Analyses", planned_analyses(), section_id="analyses"
    )

    # CTA for Oura — last section before footer
    body += make_section(
        "For Oura's Team", oura_team_cta(), section_id="oura-team"
    )

    html = wrap_html(
        title="What's Next",
        body_content=body,
        report_id="roadmap",
        subtitle="From single-patient proof-of-concept to cohort-ready pipeline",
    )

    out = REPORTS_DIR / "roadmap.html"
    out.write_text(html)
    print(f"Roadmap written to {out}")


if __name__ == "__main__":
    main()
