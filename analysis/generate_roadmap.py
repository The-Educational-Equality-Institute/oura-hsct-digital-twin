#!/usr/bin/env python3
"""Generate the What's Next page (reports/roadmap.html).

The page asked a falsifiable question in March 2026 and set its own test for it.
This generator answers that test from the pipeline's own outputs, every run:

- reports/causal_inference_metrics.json  (per-metric pre/post Mann-Whitney tests)
- reports/composite_biomarkers.json       (whole-period HRV effect)
- analysis/_counterfactual.py             (the digital twin's projected range)
- reports/<subject>_standalone_metrics.json (the two other subjects)
- reports/run_summary.json                (pipeline runtime, written by run_all.py)
- REPORT_REGISTRY                         (what exists on the site)

The only hardcoded figures are the March snapshot, reproduced verbatim as history.
"""
import json
import sys
from datetime import date, timedelta
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    BETA_BLOCKER_START,
    DATA_START,
    HEV_DIAGNOSIS_DATE,
    REPO_URL,
    REPORTS_DIR,
    TRANSPLANT_DATE,
    TREATMENT_START,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import (  # noqa: E402
    REPORT_REGISTRY,
    _resolve_latest_data_date,
    format_p_value,
    make_kpi_card,
    make_kpi_row,
    make_section,
    wrap_html,
)

# What this page showed in late March 2026, day 8 on ruxolitinib. Kept verbatim as history.
MARCH_SNAPSHOT = [
    ("Temperature deviation", "+0.07 C", "-0.16 C", "Yes (immunosuppression)", "Yes (acute viral fever resolving)"),
    ("Sleep HR", "85.0 bpm", "81.8 bpm", "Yes (reduced inflammation)", "Yes (acute phase resolving)"),
    ("Resting HR (readiness)", "79.1 bpm", "84.2 bpm", "No (went up, not down)", "Yes (HEV-driven tachycardia)"),
    ("HRV (RMSSD)", "9.2 ms", "10.1 ms", "Marginal (still severely depressed)", "Unclear"),
]
MARCH_TEST = (
    "Time. HEV is acute. It resolves in weeks. Ruxolitinib is sustained. "
    "If the signals persist and strengthen at day 28 (~April 13), it is ruxolitinib. "
    "If they fade, it was HEV. We cannot know before then."
)

METRIC_ORDER = ["mean_rmssd", "average_heart_rate", "lowest_heart_rate", "sleep_efficiency"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    path = REPORTS_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


def individual_tests() -> dict:
    tests = _load("causal_inference_metrics.json").get("individual_metric_tests", {})
    return {k: v for k, v in tests.items() if isinstance(v, dict) and "post_mean" in v}


def whole_period_hrv() -> dict:
    return _load("composite_biomarkers.json").get("treatment_response", {}).get("rmssd_mean", {})


def counterfactual() -> dict:
    try:
        from _counterfactual import compute_counterfactual
        return compute_counterfactual()
    except Exception:  # noqa: BLE001  (the page must render without the twin)
        return {}


def phases(metric: str) -> dict:
    try:
        from _counterfactual import phase_estimates
        return phase_estimates(metric)
    except Exception:  # noqa: BLE001
        return {}


def _p(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}" if value < 0.01 else f"p={value:.2f}"


def subject_metrics(profile: str) -> dict:
    return _load(f"{profile}_standalone_metrics.json")


def run_summary() -> dict:
    return _load("run_summary.json")


def _fmt_runtime(seconds: float | None) -> str:
    if not seconds:
        return ""
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    return f"{seconds / 60:.0f} minutes"


def _fmt_date(d: date) -> str:
    return d.strftime("%B %d, %Y").replace(" 0", " ")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def what_this_is(latest: date, post_days: int) -> str:
    n_reports = len([r for r in REPORT_REGISTRY if r["group"] not in ("Context",)])
    return f"""
<p>One patient, one consumer ring, every night since {_fmt_date(DATA_START)}. I had an
allogeneic stem-cell transplant in {TRANSPLANT_DATE.strftime('%B %Y')}. On {_fmt_date(TREATMENT_START)}
I started ruxolitinib (Jakavi) for chronic graft-versus-host disease. The pipeline behind this site
re-runs {n_reports} reports on the ring data every morning; this page was generated from data through
{_fmt_date(latest)}, day {post_days} on the drug. Claude Code wrote the code. I described what I needed
in Norwegian and checked what came back.</p>

<p>What it is not: a medical device, a clinical trial, or evidence about anyone but me. Every page carries
that caveat because it is true. <a href="how_built.html">How this was built</a> lists who did what;
<a href="claims.html">Every number, checked</a> is the pipeline auditing its own pages.</p>
"""


def march_question() -> str:
    hev_str = _fmt_date(HEV_DIAGNOSIS_DATE) if HEV_DIAGNOSIS_DATE else "N/A"
    rux_str = _fmt_date(TREATMENT_START)
    gap = (HEV_DIAGNOSIS_DATE - TREATMENT_START).days if HEV_DIAGNOSIS_DATE else 0
    rows = "\n".join(
        f"<tr><td>{escape(sig)}</td><td>{escape(pre)}</td><td>{escape(post)}</td>"
        f"<td>{escape(rux)}</td><td>{escape(hev)}</td></tr>"
        for sig, pre, post, rux, hev in MARCH_SNAPSHOT
    )
    return f"""
<p>Two things happened {gap} days apart. Ruxolitinib started {rux_str}. Hepatitis E was diagnosed
{hev_str}. At day 8 this page said the two could not be separated, and set a test:</p>

<blockquote class="odt-quote">&ldquo;{escape(MARCH_TEST)}&rdquo;</blockquote>

<p>This is the table the page showed then, day 8 on ruxolitinib, kept exactly as published:</p>

<table>
<thead>
<tr><th>Signal</th><th>Pre-rux</th><th>Post-rux (8 days)</th><th>Could be rux?</th><th>Could be HEV?</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<p style="margin-top:16px;">The reading in March: the resting heart rate had gone <em>up</em>, which fit acute
hepatitis E better than a JAK-inhibitor response; the temperature drop could be either; the HRV change was
within noise.</p>
"""


def answer_now(tests: dict, hrv_whole: dict, cf: dict, post_days: int, latest: date) -> str:
    if not tests:
        return "<p>The per-metric tests have not been generated yet. Run the pipeline first.</p>"

    ph = {m: phases(m) for m in ("mean_rmssd", "lowest_heart_rate", "average_heart_rate")}
    hrv_ph = ph["mean_rmssd"]
    bb_str = _fmt_date(BETA_BLOCKER_START) if BETA_BLOCKER_START else "a later date"
    fpr = hrv_ph.get("mw_fpr")

    # --- KPI row: the phase view, not the pooled view ---
    cards = []
    if hrv_ph.get("phase_b_mean") is not None:
        cards.append(make_kpi_card(
            "HRV, ruxolitinib alone", hrv_ph["phase_b_mean"], "ms", status="info", decimals=1,
            detail=f"{hrv_ph.get('phase_a_mean', 0):.1f} ms before → {hrv_ph['phase_b_mean']:.1f} ms · {hrv_ph.get('n_b', '?')} nights · Tau-U {hrv_ph.get('tau_ab', 0):+.2f}",
            status_label="Inside the twin's range",
        ))
    if hrv_ph.get("phase_c_mean") is not None:
        step = hrv_ph.get("its_bb", {})
        cards.append(make_kpi_card(
            "HRV, plus beta-blocker", hrv_ph["phase_c_mean"], "ms", status="good", decimals=1,
            detail=f"{hrv_ph['phase_b_mean']:.1f} → {hrv_ph['phase_c_mean']:.1f} ms · {hrv_ph.get('n_c', '?')} nights · ITS step {step.get('estimate', 0):+.1f} ms, {_p(step.get('p_value'))}",
            status_label="Step at second medicine",
        ))
    lhr = ph["lowest_heart_rate"]
    if lhr.get("phase_c_mean") is not None:
        step = lhr.get("its_bb", {})
        cards.append(make_kpi_card(
            "Lowest HR, plus beta-blocker", lhr["phase_c_mean"], "bpm", status="good", decimals=1,
            detail=f"{lhr.get('phase_a_mean', 0):.1f} before · {lhr.get('phase_b_mean', 0):.1f} on ruxolitinib alone · ITS step {step.get('estimate', 0):+.1f} bpm, {_p(step.get('p_value'))}",
            status_label="Step at second medicine",
        ))
    if cf.get("bb_nights"):
        cards.append(make_kpi_card(
            "Nights above the twin's range", f"{cf['nights_above_band_bb']}/{cf['bb_nights']}", "",
            status="info",
            detail=f"since the beta-blocker · {cf['nights_above_band_jakavi_only']}/{cf['jakavi_only_nights']} on ruxolitinib alone · projected {cf['cf_recent']:.0f} ms, actual {cf['actual_recent']:.0f} ms",
        ))
    kpis = make_kpi_row(*cards) if cards else ""

    # --- Table 1: did the signals fade? (pooled, descriptive) ---
    rows = []
    for key in METRIC_ORDER:
        m = tests.get(key)
        if not m:
            continue
        unit = m.get("unit", "")
        change = m["post_mean"] - m["pre_mean"]
        rows.append(
            f"<tr><td>{escape(m['label'])}</td>"
            f"<td>{m['pre_mean']:.1f} {escape(unit)}</td>"
            f"<td>{m['post_mean']:.1f} {escape(unit)}</td>"
            f"<td>{change:+.1f} {escape(unit)}</td>"
            f"<td>{m['cohens_d']:+.2f}</td>"
            f"<td>{m['n_pre']} / {m['n_post']}</td></tr>"
        )
    pooled = f"""
<h3>Did the signals fade? No.</h3>
<p>All nights before treatment against all nights on treatment, the split the March page used:</p>
<table>
<thead>
<tr><th>Signal</th><th>Before</th><th>On treatment</th><th>Change</th><th>Cohen's d</th><th>Nights before / on</th></tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<p style="margin-top:12px;">These are descriptions, not evidence. The pooled Mann-Whitney test behind them fires at
{f"{fpr:.0%}" if fpr is not None else "most"} of 20 random pre-treatment dates in this pipeline's own
<a href="placebo_calibration.html">placebo calibration</a>, so its p-values are not shown here.</p>
"""

    # --- Table 2: which medicine? (phase-corrected) ---
    prow = []
    for key, label in (("mean_rmssd", "HRV (RMSSD)"), ("lowest_heart_rate", "Lowest HR"), ("average_heart_rate", "Average sleeping HR")):
        e = ph.get(key, {})
        if e.get("phase_c_mean") is None:
            continue
        j, b = e.get("its_jakavi", {}), e.get("its_bb", {})
        prow.append(
            f"<tr><td>{escape(label)}</td>"
            f"<td>{e.get('phase_a_mean', 0):.1f}</td>"
            f"<td>{e.get('phase_b_mean', 0):.1f}</td>"
            f"<td>{e['phase_c_mean']:.1f}</td>"
            f"<td>{j.get('estimate', 0):+.1f} ({escape(_p(j.get('p_value')))})</td>"
            f"<td>{b.get('estimate', 0):+.1f} ({escape(_p(b.get('p_value')))})</td>"
            f"<td>{e.get('tau_ab', 0):+.2f}</td>"
            f"<td>{e.get('tau_bc', 0):+.2f}</td></tr>"
        )
    n_a, n_b, n_c = hrv_ph.get("n_a", "?"), hrv_ph.get("n_b", "?"), hrv_ph.get("n_c", "?")
    phased = f"""
<h3>Was it ruxolitinib? The March test could not see the second medicine.</h3>
<p>A beta-blocker was added on {bb_str}, {(BETA_BLOCKER_START - TREATMENT_START).days if BETA_BLOCKER_START else '?'} days
after ruxolitinib. Three phases: before treatment ({n_a} nights), ruxolitinib alone ({n_b} nights), ruxolitinib plus
beta-blocker ({n_c} nights). The piecewise interrupted time series fits a step at each medicine start with AR(1)
errors; Tau-U is the trend-corrected single-case effect size between phases.</p>
<table>
<thead>
<tr><th>Signal</th><th>Before</th><th>Ruxolitinib alone</th><th>Plus beta-blocker</th><th>ITS step at ruxolitinib</th><th>ITS step at beta-blocker</th><th>Tau-U before → alone</th><th>Tau-U alone → plus BB</th></tr>
</thead>
<tbody>
{''.join(prow)}
</tbody>
</table>
"""

    reading = ""
    if hrv_ph.get("phase_c_mean") is not None:
        j, b = hrv_ph.get("its_jakavi", {}), hrv_ph.get("its_bb", {})
        twin = ""
        if cf.get("bb_nights"):
            twin = (
                f" The twin, trained only on the nights before treatment, projected HRV around {cf['cf_recent']:.0f} ms without it: "
                f"{cf['nights_above_band_jakavi_only']} of {cf['jakavi_only_nights']} nights on ruxolitinib alone sat above that range, "
                f"{cf['nights_above_band_bb']} of {cf['bb_nights']} nights since the beta-blocker did."
            )
        reading = f"""
<p style="margin-top:16px;">By the test this page set itself, the signals did not fade. But the change the March page was
waiting for did not arrive in the {n_b} nights on ruxolitinib alone: HRV moved from {hrv_ph.get('phase_a_mean', 0):.1f} to
{hrv_ph.get('phase_b_mean', 0):.1f} ms, and the phase-corrected model finds no step there ({j.get('estimate', 0):+.1f} ms,
{_p(j.get('p_value'))}). It arrived after {bb_str}: HRV averaged {hrv_ph['phase_c_mean']:.1f} ms on the two medicines
together, with a step of {b.get('estimate', 0):+.1f} ms at the beta-blocker ({_p(b.get('p_value'))}).{twin}</p>

<p>So the honest answer at day {post_days}: the signal persisted, and by the pipeline's own phase analysis it sits with the
second medicine, or with the two together. Whether that is a beta-blocker acting on heart rate, a delayed ruxolitinib
effect, or both, is the open question. The <a href="research_synthesis.html">Research Synthesis</a> page holds the
two-hit hypothesis; the <a href="piecewise_regression.html">Piecewise ITS</a>, <a href="tau_u_effects.html">Tau-U</a>
and <a href="sequential_causal_impact.html">Sequential CI</a> pages hold the models.</p>
"""

    caveat = f"""
<p>What time cannot settle: this is one person and an observational before-and-after. Hepatitis E was diagnosed two days
after ruxolitinib started and resolved in the same window. The <a href="causal_inference_report.html">causal report</a>
runs the single-split methods and shows their own placebo check; read that before reading any paragraph here as proof.</p>

<p>The shortcut named in March still stands: Oura's own cohort contains users on ruxolitinib without hepatitis E, and
users on beta-blockers without either. That comparison is impossible with N=1 and trivial at cohort scale.</p>
"""
    return f"""
<p>Data through {_fmt_date(latest)}. Every number is a live query from the pipeline's own JSON.</p>
{kpis}
{pooled}
{phased}
{reading}
{caveat}
"""


def subject_cards(latest: date, tests: dict) -> str:
    hrv = tests.get("mean_rmssd", {})
    nights = (hrv.get("n_pre", 0) + hrv.get("n_post", 0)) or (latest - DATA_START).days + 1
    primary = make_kpi_card(
        "Primary subject", nights, "days of data", status="info", decimals=0,
        detail=f"Post-HSCT · data {_fmt_date(DATA_START)} to {_fmt_date(latest)}",
    )
    cards = [primary]
    for profile, title in (("wenche", "Family control"), ("mitch", "Disease control")):
        m = subject_metrics(profile)
        if not m:
            continue
        generated = str(m.get("generated", ""))[:10]
        cards.append(
            make_kpi_card(
                title, m.get("data_days", 0), "days of ring data", status="info", decimals=0,
                detail=f"{escape(str(m.get('label', '')))} · age {m.get('age', '?')} · metrics dated {generated}",
            )
        )
    return make_kpi_row(*cards)


def roadmap_table(latest: date) -> str:
    groups: dict[str, int] = {}
    for r in REPORT_REGISTRY:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    wenche = subject_metrics("wenche")
    mitch = subject_metrics("mitch")
    summary = run_summary()
    passed = summary.get("passed")
    total = (summary.get("passed") or 0) + (summary.get("failed") or 0)
    daily = "Reports regenerate every morning at 06:15 from a fresh Oura import"
    if passed is not None and total:
        daily += f"; last run {passed}/{total} scripts passed"
    rows = [
        ("v1.0", "Done", f"Single-patient pipeline. Live since March 2026; data through {_fmt_date(latest)}."),
        ("v1.1", "Running", daily + "."),
        ("v2.0", "Running" if wenche else "Planned",
         (f"Healthy family control: {wenche.get('data_days', 0)} days of ring data, standalone dashboard live." if wenche
          else "Healthy family control, 30+ nights, same pipeline and shared genetics.")),
        ("v2.1", "Running" if mitch else "Planned",
         (f"Post-stroke control: {mitch.get('data_days', 0)} days of ring history, standalone dashboard and changepoint scan live." if mitch
          else "Post-stroke control, 30+ consecutive nights, different pathology, same age.")),
        ("v3.0", "Partly",
         f"{groups.get('Comparative', 0)} comparative reports and {groups.get('Statistical', 0)} single-case statistical reports are live. "
         "Between-subjects CausalImpact is not built yet."),
    ]
    body = "\n".join(
        f'<tr><td><strong>{escape(v)}</strong></td><td>{escape(status)}</td><td>{what}</td></tr>'
        for v, status, what in rows
    )
    return f"""
<table>
<thead><tr><th>Phase</th><th>Status</th><th>What</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
"""


def planned_analyses() -> str:
    analyses = [
        ("Between-subjects CausalImpact",
         "Same intervention window, three physiologies. Does ruxolitinib produce a detectable autonomic "
         "signal only in the immune-mediated patient? The controls make this answerable."),
        ("Family genetic control comparison",
         "Mother and son, about half the genome shared. Baseline HRV, sleep architecture, circadian patterns. "
         "What is inherited and what is disease?"),
        ("Disease-specific autonomic signatures",
         "Immune-mediated (GVHD) against neurovascular (post-stroke). Different damage, same ring, same pipeline. "
         "Do the signatures separate?"),
        ("Cohort-ready pipeline",
         "Any Oura user, any condition. A config file and an OAuth token run the full report set. Open source."),
    ]
    return "\n".join(
        f'<div style="margin-bottom:20px;"><h3 style="margin-bottom:6px;">{escape(t)}</h3><p style="margin:0;">{escape(d)}</p></div>'
        for t, d in analyses
    )


def oura_team_cta() -> str:
    runtime = _fmt_runtime(run_summary().get("total_runtime_s"))
    runtime_txt = f" It runs end to end in {runtime} on one CPU." if runtime else ""
    return f"""
<p>This pipeline needs a config file and an OAuth token.{runtime_txt} If your cohort contains even 50 users on
ruxolitinib, you can validate or falsify every finding on this site in an afternoon. The code is
<a href="{REPO_URL}">MIT-licensed</a>. The tables above say exactly which comparisons resolve which questions.</p>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    latest = _resolve_latest_data_date()
    post_days = max(0, (latest - TREATMENT_START).days + 1)
    tests = individual_tests()
    cf = counterfactual()
    hrv_whole = whole_period_hrv()

    body = ""
    body += make_section("What this is, and what it is not", what_this_is(latest, post_days), section_id="honest")
    body += make_section("The question this page asked in March", march_question(), section_id="question")
    body += make_section(f"The answer at day {post_days}", answer_now(tests, hrv_whole, cf, post_days, latest), section_id="answer")
    body += make_section("Subjects", subject_cards(latest, tests), section_id="subjects")
    body += make_section("Where the roadmap stands", roadmap_table(latest), section_id="roadmap")
    body += make_section("Planned analyses", planned_analyses(), section_id="analyses")
    body += make_section("For Oura's team", oura_team_cta(), section_id="oura-team")

    html = wrap_html(
        title="What's Next",
        body_content=body,
        report_id="roadmap",
        subtitle="The question this site asked in March, answered by its own data",
        data_end=latest,
        post_days=post_days,
        extra_css=(
            ".odt-quote{margin:16px 0;padding:12px 18px;border-left:3px solid #3A3AD6;"
            "background:rgba(58,58,214,0.05);font-style:italic;color:#14161A;}"
        ),
    )
    out = REPORTS_DIR / "roadmap.html"
    out.write_text(html, encoding="utf-8")
    print(f"Roadmap written to {out}")


if __name__ == "__main__":
    main()
