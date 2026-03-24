#!/usr/bin/env python3
"""Generate the What's Next / Roadmap page for Oura Digital Twin.

Outputs reports/roadmap.html using the shared _theme.py design system.
Queries the database and reads JSON metrics for dynamic values.
"""

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    DATABASE_PATH,
    REPORTS_DIR,
    TREATMENT_START,
    DATA_START,
    TRANSPLANT_DATE,
    PATIENT_AGE,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import wrap_html, make_section, make_kpi_card, make_kpi_row  # noqa: E402


def load_dynamic_stats() -> dict:
    """Load all dynamic values from database, config, and JSON metrics."""
    tx_str = str(TREATMENT_START)
    ds_str = str(DATA_START)

    # --- Determine data end date from the actual dataset (not wall clock) ---
    conn_tmp = sqlite3.connect(str(DATABASE_PATH))
    try:
        row = conn_tmp.execute(
            "SELECT MAX(day) FROM oura_sleep_periods"
        ).fetchone()
        data_end_str = row[0] if row and row[0] else None
    finally:
        conn_tmp.close()

    if data_end_str:
        data_end = date.fromisoformat(data_end_str)
    else:
        data_end = date.today()  # fallback only when DB is empty

    # --- Date-derived counts ---
    pre_days = (TREATMENT_START - DATA_START).days
    post_days = (data_end - TREATMENT_START).days
    total_days = (data_end - DATA_START).days
    months_post_hsct = round((data_end - TRANSPLANT_DATE).days / 30.44)

    # --- Pre/post means from database ---
    stats = {
        "total_days": total_days,
        "pre_days": pre_days,
        "post_days": post_days,
        "months_post_hsct": months_post_hsct,
        "patient_age": PATIENT_AGE,
        "treatment_start": TREATMENT_START,
    }

    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        # Sleep HR and HRV pre/post (oura_sleep_periods uses 'day')
        rows = conn.execute(
            """
            SELECT
                CASE WHEN day < ? THEN 'pre' ELSE 'post' END as period,
                AVG(average_heart_rate) as avg_hr,
                AVG(average_hrv) as avg_hrv
            FROM oura_sleep_periods
            WHERE day >= ?
            GROUP BY period
        """,
            (tx_str, ds_str),
        ).fetchall()
        for r in rows:
            p = r["period"]
            stats[f"sleep_hr_{p}"] = round(r["avg_hr"], 1) if r["avg_hr"] else 0
            stats[f"hrv_{p}"] = round(r["avg_hrv"], 1) if r["avg_hrv"] else 0

        # Temperature deviation and resting HR pre/post (oura_readiness uses 'date')
        rows = conn.execute(
            """
            SELECT
                CASE WHEN date < ? THEN 'pre' ELSE 'post' END as period,
                AVG(resting_heart_rate) as rhr,
                AVG(temperature_deviation) as temp_dev
            FROM oura_readiness
            WHERE date >= ?
            GROUP BY period
        """,
            (tx_str, ds_str),
        ).fetchall()
        for r in rows:
            p = r["period"]
            stats[f"resting_hr_{p}"] = round(r["rhr"], 1) if r["rhr"] else 0
            stats[f"temp_dev_{p}"] = (
                round(r["temp_dev"], 2) if r["temp_dev"] is not None else 0
            )
    finally:
        conn.close()

    # --- Module count from JSON metrics ---
    json_files = [
        "digital_twin_metrics.json",
        "causal_inference_metrics.json",
        "gvhd_prediction_metrics.json",
        "anomaly_detection_metrics.json",
        "composite_biomarkers.json",
        "advanced_hrv_metrics.json",
        "advanced_sleep_metrics.json",
        "spo2_bos_metrics.json",
        "foundation_model_metrics.json",
        "oura_full_analysis.json",
    ]
    loaded = sum(1 for f in json_files if (REPORTS_DIR / f).exists())
    stats["n_modules"] = loaded
    stats["n_modules_total"] = len(json_files)

    # --- Run time from full analysis JSON ---
    try:
        fa = json.loads((REPORTS_DIR / "oura_full_analysis.json").read_text())
        stats["run_time_sec"] = fa.get("run_time_seconds")
    except (FileNotFoundError, json.JSONDecodeError):
        stats["run_time_sec"] = None

    return stats


def _fmt_temp(val: float) -> str:
    """Format temperature deviation with sign."""
    return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"


def subject_cards(s: dict) -> str:
    """Three subject profile cards."""
    primary = make_kpi_card(
        "Primary Subject",
        str(s["total_days"]),
        "days",
        status="info",
        detail=f"{s['patient_age']}M &middot; Post-HSCT &middot; "
        f"Resting HR {s.get('resting_hr_pre', '?')}-{s.get('resting_hr_post', '?')} "
        f"&middot; HRV {s.get('hrv_pre', '?')}-{s.get('hrv_post', '?')} ms",
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


def honest_assessment(s: dict) -> str:
    """What the current post-rux window can and cannot tell us."""
    temp_pre = _fmt_temp(s.get("temp_dev_pre", 0))
    temp_post = _fmt_temp(s.get("temp_dev_post", 0))
    rhr_post = s.get("resting_hr_post", 0)
    rhr_pre = s.get("resting_hr_pre", 0)
    rhr_direction = "up" if rhr_post > rhr_pre else "down"

    return f"""
<h3>What we have</h3>
<p>{s["total_days"]} days of continuous Oura data. {s["pre_days"]} days pre-intervention baseline.
{s["post_days"]} days post-ruxolitinib. {s["n_modules"]} core analysis modules plus a roadmap appendix.
A CausalImpact model that
returns a statistically significant treatment signal (see causal inference report for current p-values).</p>

<h3>What that p-value actually means</h3>
<p>The model detected a statistically significant change in the biometric
signal after March 16. It cannot tell us <em>why</em>.</p>

<h3>The problem</h3>
<p>Ruxolitinib started March 16. Hepatitis E was diagnosed March 18.
Two days apart. At day {s["post_days"]}, we cannot separate the two.</p>

<table>
<thead>
<tr>
  <th>Signal</th>
  <th>Pre-rux ({s["pre_days"]} days)</th>
  <th>Post-rux ({s["post_days"]} days)</th>
  <th>Could be rux?</th>
  <th>Could be HEV?</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Temperature deviation</td>
  <td>{temp_pre} &deg;C</td>
  <td>{temp_post} &deg;C</td>
  <td>Yes (immunosuppression)</td>
  <td>Yes (acute viral fever resolving)</td>
</tr>
<tr>
  <td>Sleep HR</td>
  <td>{s.get("sleep_hr_pre", "?")} bpm</td>
  <td>{s.get("sleep_hr_post", "?")} bpm</td>
  <td>Yes (reduced inflammation)</td>
  <td>Yes (acute phase resolving)</td>
</tr>
<tr>
  <td>Resting HR (readiness)</td>
  <td>{rhr_pre} bpm</td>
  <td style="color:#F59E0B;">{rhr_post} bpm</td>
  <td>No (went {rhr_direction}, not {"down" if rhr_direction == "up" else "up"})</td>
  <td>Yes (HEV-driven tachycardia)</td>
</tr>
<tr>
  <td>HRV (RMSSD)</td>
  <td>{s.get("hrv_pre", "?")} ms</td>
  <td>{s.get("hrv_post", "?")} ms</td>
  <td>Marginal (still severely depressed)</td>
  <td>Unclear</td>
</tr>
</tbody>
</table>

<p style="margin-top:16px;">The resting HR went <em>{rhr_direction}</em> after ruxolitinib, not {"down" if rhr_direction == "up" else "up"}. That is
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


def day1_comparison(s: dict) -> str:
    """Family control's first night vs primary subject."""
    return f"""
<p>Ring received March 23. First full night: March 24. One night of data.</p>

<table>
<thead>
<tr>
  <th>Metric</th>
  <th>Subject A ({s["patient_age"]}M, post-HSCT)</th>
  <th>Family Control (61F, healthy)</th>
  <th>What it means</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Resting HR</td>
  <td>{s.get("resting_hr_pre", "?")}-{s.get("resting_hr_post", "?")} bpm</td>
  <td style="color:#10B981;">56-68 bpm</td>
  <td>20+ bpm gap. Subject A's autonomic system is measurably damaged.</td>
</tr>
<tr>
  <td>HRV (RMSSD)</td>
  <td>{s.get("hrv_pre", "?")}-{s.get("hrv_post", "?")} ms</td>
  <td>TBD (collecting)</td>
  <td>Population median: 49 ms. Subject A is at the 1st percentile. Family control will likely be near normal.</td>
</tr>
<tr>
  <td>Age</td>
  <td>{s["patient_age"]}</td>
  <td>61</td>
  <td>The 61-year-old has a healthier heart rate than the {s["patient_age"]}-year-old. That is the disease signal.</td>
</tr>
<tr>
  <td>Genetics</td>
  <td colspan="2" style="text-align:center;">~50% shared genome</td>
  <td>Same genetic background eliminates inherited autonomic traits as confounders.</td>
</tr>
</tbody>
</table>

<p style="margin-top:16px;">One night. The contrast is already visible.
The 61-year-old family control rests at 56-68 bpm. The {s["patient_age"]}-year-old subject,
{s["months_post_hsct"]} months post-transplant, rests at {s.get("resting_hr_pre", "?")}-{s.get("resting_hr_post", "?")} bpm. That gap is unlikely to be
explained by shared baseline alone, but still needs cohort-level confirmation.</p>
"""


def stroke_control(s: dict) -> str:
    """Subject B - what the disease control adds."""
    return f"""
<p>36M. Same age as Subject A. Post-stroke with neurovascular autonomic damage.
Has worn an Oura Ring intermittently - 34 nights over 2 years. That is not
enough for meaningful analysis.</p>

<p>Starting now: nightly wear. Once we have 30+ consecutive nights, the
same {s["n_modules"]}-module pipeline runs on his data. No code changes. Different
config file, different OAuth token, same analyses.</p>

<h3>What this adds</h3>
<p>Subject A has immune-mediated autonomic damage (GVHD, 10 organ systems).
Subject B has neurovascular autonomic damage (post-stroke). Both are 36M.
Same hardware, same software, different pathology.</p>

<p>If the autonomic signatures separate - different HRV patterns, different
sleep architecture, different circadian disruption - that is evidence that
a consumer ring can distinguish between disease types, not just detect
"something is wrong."</p>
"""


def roadmap_table(s: dict) -> str:
    """Phase roadmap as an HTML table."""
    return f"""
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
  <td>Single-patient: {s["total_days"]} days, {s["post_days"]} days post-rux, {s["n_modules"]} core modules. HEV confound unresolved.</td>
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


def planned_analyses(s: dict) -> str:
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
            f"Any Oura user, any condition. Config file + OAuth token = "
            f"full {s['n_modules']}-module analysis. Open source.",
        ),
    ]

    items = []
    for title, desc in analyses:
        items.append(
            f'<div style="margin-bottom:20px;">'
            f'<h3 style="margin-bottom:6px;">{title}</h3>'
            f'<p style="margin:0;">{desc}</p>'
            f"</div>"
        )
    return "\n".join(items)


def oura_team_cta() -> str:
    """Call-to-action for Oura's team - factual, not pushy."""
    return """
<p>This pipeline runs on a single patient in minutes. Config file +
OAuth token is all it needs. If your cohort contains even 50 users on
ruxolitinib, you can validate or falsify every finding on this site in
an afternoon. The code is MIT-licensed. The confound table above tells
you exactly which comparisons resolve which questions.</p>
"""


def main():
    stats = load_dynamic_stats()

    print(
        f"  Dynamic stats: {stats['total_days']} days, "
        f"{stats['pre_days']} pre / {stats['post_days']} post, "
        f"{stats['n_modules']}/{stats['n_modules_total']} modules loaded"
    )

    body = ""

    # Subjects
    body += make_section(
        "Current Subjects", subject_cards(stats), section_id="subjects"
    )

    # Honest assessment - the most important section
    body += make_section(
        "What This Is and What It Isn't",
        honest_assessment(stats),
        section_id="honest",
    )

    # Family control day 1
    body += make_section(
        "Day 1: Family Control",
        day1_comparison(stats),
        section_id="day1",
    )

    # Stroke control
    body += make_section(
        "Disease Control: Post-Stroke",
        stroke_control(stats),
        section_id="stroke",
    )

    # Roadmap
    body += make_section("Roadmap", roadmap_table(stats), section_id="roadmap")

    # Planned analyses
    body += make_section(
        "Planned Analyses", planned_analyses(stats), section_id="analyses"
    )

    # CTA for Oura - last section before footer
    body += make_section("For Oura's Team", oura_team_cta(), section_id="oura-team")

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
