#!/usr/bin/env python3
"""Generate reports/how_built.html: who did what, and how a number gets onto a page.

Every figure on this page is measured at generation time from the repository, the
database and the pipeline's own outputs. Nothing here is typed in by hand.
"""
import json
import sqlite3
import subprocess
import sys
from datetime import date
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    DATABASE_PATH,
    DATA_START,
    PROJECT_ROOT,
    REPO_URL,
    REPORTS_DIR,
    TREATMENT_START,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import (  # noqa: E402
    REPORT_REGISTRY,
    _resolve_latest_data_date,
    make_kpi_card,
    make_kpi_row,
    make_section,
    wrap_html,
)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _load_json(name: str) -> dict:
    path = REPORTS_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


def count_pipeline_scripts() -> int:
    try:
        import run_all  # noqa: PLC0415  (only the SCRIPTS list is needed)
        return len(run_all.SCRIPTS)
    except Exception:  # noqa: BLE001
        return len(list((PROJECT_ROOT / "analysis").glob("analyze_*.py"))) + len(
            list((PROJECT_ROOT / "analysis").glob("generate_*.py"))
        )


def count_code() -> tuple[int, int]:
    files = sorted((PROJECT_ROOT / "analysis").glob("*.py"))
    lines = sum(len(f.read_text("utf-8", errors="ignore").splitlines()) for f in files)
    return len(files), lines


def count_tests() -> int:
    return len(list((PROJECT_ROOT / "tests").rglob("test_*.py")))


def database_span() -> tuple[str, str, int]:
    try:
        with sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM oura_sleep").fetchone()
        return row[0] or "", row[1] or "", int(row[2] or 0)
    except sqlite3.Error:
        return "", "", 0


def git_facts() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commits = run("rev-list", "--count", "HEAD")
    first = run("log", "--reverse", "--format=%cs")
    first_date = first.splitlines()[0] if first else ""
    return {"commits": int(commits) if commits.isdigit() else 0, "first": first_date}


def timer_schedule() -> str:
    timer = Path.home() / ".config" / "systemd" / "user" / "oura-daily-pipeline.timer"
    if not timer.exists():
        return ""
    for line in timer.read_text("utf-8", errors="ignore").splitlines():
        if line.startswith("OnCalendar="):
            return line.split("=", 1)[1].strip()
    return ""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def who_did_what() -> str:
    rows = [
        ("Describing what to build, in Norwegian, one report at a time", "Me"),
        ("Writing the code: every analysis module, the theme, the pipeline, this page", "Claude Code"),
        ("Supplying the data: my Oura account, the clinical dates in config.py", "Me"),
        ("Checking outputs against the source data and the clinical record", "Me"),
        ("Deciding which statistical methods to run and how to report their limits", "Both, in conversation"),
        ("Every clinical decision", "My clinicians"),
    ]
    body = "\n".join(f"<tr><td>{escape(a)}</td><td><strong>{escape(b)}</strong></td></tr>" for a, b in rows)
    return f"""
<p>I have no programming background. Claude Code, Anthropic's command-line agent, wrote the code in this
repository. I told it what I needed, gave it the data, and checked what came back. The placebo tests, the
statistical audit and the link checker exist because outputs were not trusted on sight.</p>
<table>
<thead><tr><th>Task</th><th>Who</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
"""


def by_the_numbers() -> str:
    n_scripts = count_pipeline_scripts()
    n_files, n_lines = count_code()
    n_tests = count_tests()
    first, last, nights = database_span()
    git = git_facts()
    n_reports = len(REPORT_REGISTRY)
    cards = [
        make_kpi_card("Pipeline scripts", n_scripts, "", status="info", decimals=0, detail="run in order by run_all.py"),
        make_kpi_card("Report pages", n_reports, "", status="info", decimals=0, detail="in the navigation registry"),
        make_kpi_card("Analysis code", n_lines, "lines", status="info", decimals=0, detail=f"{n_files} Python files in analysis/"),
        make_kpi_card("Nights of data", nights, "", status="info", decimals=0, detail=f"{first} to {last}" if first else "database not found"),
    ]
    if git["commits"]:
        cards.append(
            make_kpi_card("Commits", git["commits"], "", status="info", decimals=0, detail=f"since {git['first']}")
        )
    cards.append(make_kpi_card("Test files", n_tests, "", status="info", decimals=0, detail="run by run_tests.py"))
    return make_kpi_row(*cards)


def how_a_number_gets_onto_a_page() -> str:
    schedule = timer_schedule()
    schedule_txt = f" A systemd user timer fires it at {escape(schedule)}." if schedule else ""
    steps = [
        ("Import", "The Oura API is read for the whole window and written to a SQLite database. Device imports (blood pressure, glucose, ECG) are preserved across rebuilds."),
        ("Analyse", "run_all.py runs every script in order. Each writes one HTML page and one JSON file with the numbers it printed."),
        ("Audit", "statcheck_reports.py re-reads every page, extracts every p-value, effect size and test statistic, and matches it against the JSON. A mismatch stops the deploy. The result is published as <a href=\"claims.html\">Every number, checked</a>."),
        ("Check links", "check-links.py walks every page and fails on any link that does not resolve to a file."),
        ("Publish", "wrangler uploads the folder to Cloudflare Pages. Nothing is edited by hand between the database and the page."),
    ]
    items = "\n".join(f"<li><strong>{escape(t)}.</strong> {d}</li>" for t, d in steps)
    return f"""
<p>The same five steps run every morning.{schedule_txt} If any step fails, the previous day's site stays up and I get a
desktop notification with the failing step.</p>
<ol style="line-height:1.7;">
{items}
</ol>
"""


def verification_status() -> str:
    audit = _load_json("statcheck_audit.json")
    summary = _load_json("run_summary.json")
    parts = []
    if audit:
        ok = "passes" if audit.get("pass") else "does not pass"
        mismatches = audit.get("mismatches", "?")
        if isinstance(mismatches, list):
            mismatches = len(mismatches)
        parts.append(
            f"The latest statistical audit checked {audit.get('reports_checked', '?')} pages, extracted "
            f"{audit.get('claims_extracted', '?')} claims and found {mismatches} mismatches; it {ok}."
        )
    if summary:
        parts.append(
            f"The latest pipeline run passed {summary.get('passed', '?')} of "
            f"{(summary.get('passed') or 0) + (summary.get('failed') or 0)} scripts in "
            f"{float(summary.get('total_runtime_s') or 0) / 60:.0f} minutes."
        )
    if not parts:
        parts.append("No audit output is present yet. Run the pipeline to produce it.")
    return "<p>" + " ".join(escape(p) for p in parts) + "</p>"


def limits() -> str:
    latest = _resolve_latest_data_date()
    return f"""
<p>One person. A consumer ring, not a clinical monitor. An observational before-and-after with two co-interventions
(hepatitis E resolving, a beta-blocker added). The models are exploratory and the site says so on every page.
Data run from {DATA_START.strftime('%B %d, %Y')} to {latest.strftime('%B %d, %Y')}; treatment split at
{TREATMENT_START.strftime('%B %d, %Y')}.</p>
"""


def source() -> str:
    return f"""
<p>The code is public under the MIT licence at <a href="{REPO_URL}">{escape(REPO_URL.replace('https://', ''))}</a>.
To run it on your own ring: copy <code>config.example.py</code> to <code>config.py</code>, put an Oura token in
<code>.env</code>, then <code>pip install -r requirements-full.txt</code> and <code>python run_all.py</code>.</p>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    body = ""
    body += make_section("Who did what", who_did_what(), section_id="who")
    body += make_section("By the numbers", by_the_numbers(), section_id="numbers")
    body += make_section("How a number gets onto a page", how_a_number_gets_onto_a_page(), section_id="pipeline")
    body += make_section("Does it check itself?", verification_status(), section_id="verification")
    body += make_section("Limits", limits(), section_id="limits")
    body += make_section("Source", source(), section_id="source")

    html = wrap_html(
        title="How this was built",
        body_content=body,
        report_id="how_built",
        subtitle="Measured from the repository and the pipeline's own outputs, generated with the rest of the site",
        header_meta="",
    )
    out = REPORTS_DIR / "how_built.html"
    out.write_text(html, encoding="utf-8")
    print(f"How-built page written to {out} (generated {date.today().isoformat()})")


if __name__ == "__main__":
    main()
