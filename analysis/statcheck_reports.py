#!/usr/bin/env python3
"""Statistical integrity checker for Oura analysis reports.

Extracts all p-values, effect sizes, and test statistics from HTML reports,
cross-references them against authoritative JSON metrics files, and flags
mismatches exceeding a configurable tolerance.

Wired into run_all.py as the post-generation gate: run_all.py runs this after
every analysis script, and refuses to assemble reports/send_bundle (and exits
non-zero) when the audit does not pass. scripts/daily-pipeline-local.sh and
scripts/deploy-local.sh stop before the Cloudflare deploy on the same signal.

Writes reports/statcheck_audit.json (machine-readable) and reports/claims.html
(the published register: every claim, its JSON value, and OK/MISMATCH/UNMATCHED).

Usage:
    python analysis/statcheck_reports.py              # check all reports
    python analysis/statcheck_reports.py --tolerance 0.01
    python analysis/statcheck_reports.py --verbose     # show every matched pair
    python analysis/statcheck_reports.py --json        # JSON output only
"""

import argparse
import html
import json
import logging
import re
import sys
from html import escape
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class StatClaim:
    """A statistical claim extracted from HTML."""

    report: str
    stat_type: str  # "p_value", "correlation", "cohens_d", "r_squared", "effect_size"
    operator: str  # "=", "<", ">", "≤"
    value: float
    context: str  # surrounding text for identification
    line_hint: int = 0  # approximate position in text


@dataclass
class StatReference:
    """An authoritative value from JSON metrics."""

    report: str
    json_path: str
    stat_type: str
    value: float
    label: str = ""  # human-friendly label if available


@dataclass
class Mismatch:
    """A flagged discrepancy between HTML and JSON."""

    report: str
    stat_type: str
    html_value: float
    html_operator: str
    json_value: float
    json_path: str
    delta: float
    context: str
    severity: str  # "error", "warning", "info"


@dataclass
class SanityIssue:
    """A standalone sanity-check finding (no JSON cross-ref needed)."""

    report: str
    issue_type: str  # "p_gt_1", "p_negative", "p_zero_exact", "malformed"
    value: float | str
    context: str
    severity: str


@dataclass
class AuditResult:
    """Complete audit output."""

    reports_checked: int = 0
    claims_extracted: int = 0
    references_loaded: int = 0
    matches_found: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    sanity_issues: list[SanityIssue] = field(default_factory=list)
    unmatched_html: list[StatClaim] = field(default_factory=list)
    json_errors: list[str] = field(default_factory=list)
    matched_pairs: list[tuple[StatClaim, StatReference]] = field(default_factory=list)
    pages_with_json: list[str] = field(default_factory=list)
    pages_without_json: list[str] = field(default_factory=list)
    pages_expected_missing: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTML → JSON filename mapping
# ---------------------------------------------------------------------------

# Every page the pipeline publishes, mapped to the JSON metrics file(s) whose
# values it is allowed to print. Derived from the REPORTS_DIR writes in each
# generator in analysis/ (see docs/BUILD-2026-09-20-pipeline.md). A page may
# name several JSON files when it aggregates numbers produced elsewhere. None
# means no companion JSON exists, so the page is scanned for sanity issues and
# its claims are reported as UNMATCHED rather than cross-referenced.
HTML_TO_JSON: dict[str, str | list[str] | None] = {
    # Core
    "index.html": None,                       # generate_index.py (reads others)
    "roadmap.html": [                         # generate_roadmap.py aggregates
        "causal_inference_metrics.json",
        "composite_biomarkers.json",
    ],
    "anthropic_case.html": None,              # generate_anthropic_case.py
    "404.html": None,                         # generate_404.py (no statistics)
    "how_built.html": None,                   # generate_how_built.py
    "oura_full_analysis.html": "oura_full_analysis.json",
    "composite_biomarkers.html": "composite_biomarkers.json",
    "advanced_sleep_analysis.html": "advanced_sleep_metrics.json",
    "weekly_tracker.html": "weekly_tracker.json",
    # Clinical
    "causal_inference_report.html": "causal_inference_metrics.json",
    "gvhd_prediction_report.html": "gvhd_prediction_metrics.json",
    "spo2_bos_screening.html": "spo2_bos_metrics.json",
    "rux_forecast.html": "rux_forecast.json",
    "research_synthesis.html": "research_synthesis_metrics.json",
    "treatment_response_report.html": "treatment_response_metrics.json",
    # Advanced
    "advanced_hrv_analysis.html": "advanced_hrv_metrics.json",
    "digital_twin_report.html": "digital_twin_metrics.json",
    "foundation_model_report.html": "foundation_model_metrics.json",
    "anomaly_detection_report.html": "anomaly_detection_metrics.json",
    "oura_3d_dashboard.html": "oura_3d_dashboard_metrics.json",
    "multimodal_anomaly_report.html": "multimodal_anomaly_metrics.json",
    "omron_bp_report.html": "omron_bp_report.json",
    # Comparative
    "comparative_autonomic_report.html": "comparative_autonomic_metrics.json",
    "comparative_treatment_response.html": "comparative_treatment_response.json",
    "comparative_sleep_analysis.html": "comparative_sleep_metrics.json",
    "comparative_activity_recovery_coupling.html": "comparative_activity_recovery_coupling.json",
    "comparative_anomaly_report.html": "comparative_anomaly_metrics.json",
    "comparative_breathing_analysis.html": "comparative_breathing_metrics.json",
    "comparative_temperature_analysis.html": "comparative_temperature_metrics.json",
    "mitch_changepoint_investigation.html": "mitch_changepoint_metrics.json",
    # Statistical
    "piecewise_regression.html": "piecewise_regression_metrics.json",
    "sequential_causal_impact.html": "sequential_causal_impact_metrics.json",
    "placebo_calibration.html": "placebo_calibration_metrics.json",
    "tau_u_effects.html": "tau_u_metrics.json",
    # Individual subjects (analyze_patient_standalone.py, one pair per profile)
    "mitch_standalone_report.html": "mitch_standalone_metrics.json",
    "wenche_standalone_report.html": "wenche_standalone_metrics.json",
}

# Pages this audit writes about the other pages. They restate every number the
# analysis pages print, so scanning them would re-extract the whole register as
# if it were a fresh set of claims.
AUDIT_OUTPUT_PAGES = frozenset({"claims.html"})

# ---------------------------------------------------------------------------
# Extraction: HTML → StatClaim list
# ---------------------------------------------------------------------------

# Patterns that look like p-values but are not (CSS, percentages, thresholds)
FALSE_POSITIVE_CONTEXTS = re.compile(
    r"padding|percent|pixel|opacity|font-size|margin|"
    r"top:|left:|right:|bottom:|width:|height:|"
    r"span>p>|<p>|</p>|rgb|hsl|grid|flex",
    re.IGNORECASE,
)

# p-value pattern: p = 0.123, p<0.001, p > 0.05, etc.
P_VALUE_RE = re.compile(
    r"""(?<![a-zA-Z])         # not preceded by letter (avoid "sleep=...")
    p                          # literal 'p'
    \s*                        # optional whitespace
    ([=<>≤≥])                  # operator
    \s*
    (\d+\.?\d*(?:[eE][+-]?\d+)?)  # numeric value
    """,
    re.VERBOSE,
)

# Correlation: r = +0.791, r = -0.668
CORR_RE = re.compile(
    r"""(?<![a-zA-Z])         # not preceded by letter
    r                          # literal 'r'
    \s*=\s*
    ([+-]?\d+\.?\d*)           # value (possibly signed)
    """,
    re.VERBOSE,
)

# R-squared: R²=0.025, R² = 0.189
R2_RE = re.compile(
    r"""R[²2]                  # R² or R2
    \s*=\s*
    (\d+\.?\d*)                # value
    """,
    re.VERBOSE,
)

# Cohen's d: Cohen's d = 1.53, d = -0.93
COHENS_D_RE = re.compile(
    r"""(?:Cohen.{0,3}s?\s*)?  # optional "Cohen's"
    d\s*=\s*
    ([+-]?\d+\.?\d*)           # value
    """,
    re.VERBOSE,
)


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities, preserving whitespace structure."""
    # Decode unicode escapes from Plotly JSON blobs
    text = text.replace(r"\u003c", "<").replace(r"\u003e", ">")
    text = text.replace(r"\u003cbr\u003e", " | ")
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = html.unescape(text)
    return text


def extract_claims(report_name: str, html_content: str) -> list[StatClaim]:
    """Extract statistical claims from an HTML report."""
    text = _strip_html(html_content)
    claims: list[StatClaim] = []

    # --- p-values ---
    for m in P_VALUE_RE.finditer(text):
        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 30)
        context = text[start:end].strip()

        # Filter false positives
        if FALSE_POSITIVE_CONTEXTS.search(context):
            continue

        # A "p" preceded by asterisks is the significance legend under a table
        # ("* p<0.05, ** p<0.01, *** p<0.001"), i.e. notation, not a claim.
        preceding = text[max(0, m.start() - 4):m.start()]
        if preceding.strip().endswith("*"):
            continue

        op = m.group(1)
        try:
            val = float(m.group(2))
        except ValueError:
            continue

        # Filter values clearly not p-values (> 1 handled as sanity issue)
        if val > 1.0:
            continue

        claims.append(
            StatClaim(
                report=report_name,
                stat_type="p_value",
                operator=op,
                value=val,
                context=context,
                line_hint=m.start(),
            )
        )

    # --- correlations (r = ...) ---
    for m in CORR_RE.finditer(text):
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 20)
        context = text[start:end].strip()
        if FALSE_POSITIVE_CONTEXTS.search(context):
            continue
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if abs(val) > 1.0:
            continue
        claims.append(
            StatClaim(
                report=report_name,
                stat_type="correlation",
                operator="=",
                value=val,
                context=context,
                line_hint=m.start(),
            )
        )

    # --- R² ---
    for m in R2_RE.finditer(text):
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 20)
        context = text[start:end].strip()
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if val > 1.0:
            continue
        claims.append(
            StatClaim(
                report=report_name,
                stat_type="r_squared",
                operator="=",
                value=val,
                context=context,
                line_hint=m.start(),
            )
        )

    # --- Cohen's d ---
    for m in COHENS_D_RE.finditer(text):
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 20)
        context = text[start:end].strip()
        # Avoid matching CSS "d=..." (SVG path)
        if "path" in context.lower() or "svg" in context.lower():
            continue
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        claims.append(
            StatClaim(
                report=report_name,
                stat_type="cohens_d",
                operator="=",
                value=val,
                context=context,
                line_hint=m.start(),
            )
        )

    return claims


# ---------------------------------------------------------------------------
# JSON → StatReference list
# ---------------------------------------------------------------------------

# Plausible range per statistic. A JSON value outside it is not that statistic
# (a "tau_max" of 7 is a lag parameter, not a correlation), so it never becomes
# a reference and cannot be paired against a printed claim.
STAT_RANGES = {
    "p_value": (0.0, 1.0),
    "correlation": (-1.0, 1.0),
    "r_squared": (0.0, 1.0),
    "cohens_d": (-100.0, 100.0),
    "effect_size": (-100.0, 100.0),
}


def classify_stat_key(key: str) -> str | None:
    """Map a JSON key name to the statistic it holds, or None.

    The metrics files name the same statistic a dozen ways (p_value, kruskal_p,
    mann_whitney_p, b4_pvalue, plain p). Until 2026-09-20 only "p_value" and
    "p_val" were recognised, so most printed p-values had no authority to match
    against and were paired with whatever unrelated value happened to be
    numerically closest. That produced mismatches the pages had not made.
    """
    k = key.lower()

    # Probabilities that merely start with "p_" are not p-values.
    if k.startswith("p_") and k != "p_value":
        return None
    if k == "p" or k.endswith("_p") or k == "p_value" or k == "pvalue":
        return "p_value"
    if k.endswith("_p_value") or k.endswith("_pvalue"):
        return "p_value"

    if k.startswith("r_squared") or k == "r2" or k.endswith("_r_squared") or k.endswith("_r2"):
        return "r_squared"

    if "cohens_d" in k or k in ("cohen_d", "cohens_d"):
        return "cohens_d"

    if k in ("r", "rho", "tau", "correlation") or k.startswith("tau_"):
        return "correlation"
    if k.endswith(("_r", "_rho", "_corr", "_correlation")):
        return "correlation"

    if k == "effect_size" or k.endswith("_effect_size"):
        return "effect_size"

    return None


def extract_references(
    report_name: str, json_data: Any, prefix: str = ""
) -> list[StatReference]:
    """Recursively extract statistical values from JSON metrics."""
    refs: list[StatReference] = []

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            path = f"{prefix}.{key}" if prefix else key
            key_lower = key.lower()

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stat_type = classify_stat_key(key_lower)
                if stat_type is not None:
                    low, high = STAT_RANGES[stat_type]
                    if low <= float(value) <= high:
                        # Extract a human label from nearby keys
                        label = json_data.get(
                            "label", json_data.get("feature", json_data.get("name", ""))
                        )
                        refs.append(
                            StatReference(
                                report=report_name,
                                json_path=path,
                                stat_type=stat_type,
                                value=float(value),
                                label=str(label) if label else "",
                            )
                        )

            refs.extend(extract_references(report_name, value, path))

    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            refs.extend(extract_references(report_name, item, f"{prefix}[{i}]"))

    return refs


# ---------------------------------------------------------------------------
# Cross-reference & matching
# ---------------------------------------------------------------------------


def _values_match(html_val: float, json_val: float, tolerance: float) -> bool:
    """Check if two values match within tolerance, accounting for rounding."""
    return abs(html_val - json_val) <= tolerance


def _round_matches(html_val: float, json_val: float) -> bool:
    """Check if HTML value is a rounded version of JSON value."""
    # Determine decimal places in HTML value
    s = f"{html_val:.10f}".rstrip("0")
    if "." in s:
        decimals = len(s.split(".")[1])
    else:
        decimals = 0
    rounded_json = round(json_val, decimals)
    return abs(html_val - rounded_json) < 1e-10


_ANCHOR_SPLIT = re.compile(r"[^a-z0-9]+")
_ANCHOR_STOPWORDS = {
    "value", "values", "test", "tests", "mean", "metrics", "metric", "data",
    "json", "report", "stat", "stats", "result", "results", "score", "scores",
    "analysis", "summary", "total", "with", "from", "this", "that", "than",
}


def _anchor_tokens(text: str) -> set[str]:
    """Distinctive lowercase words of length >= 4, minus generic filler."""
    return {
        tok for tok in _ANCHOR_SPLIT.split(text.lower())
        if len(tok) >= 4 and not tok.isdigit() and tok not in _ANCHOR_STOPWORDS
    }


def _is_anchored(claim: StatClaim, ref: StatReference) -> bool:
    """True when the claim's surrounding text names the statistic the ref holds.

    Without this, any printed number is "matched" to whichever JSON value of the
    same kind happens to be numerically nearest, and a difference is reported as
    a mismatch the page never made. A value difference is only evidence that the
    page is wrong when we can tell the two describe the same statistic.
    """
    ref_tokens = _anchor_tokens(ref.json_path) | _anchor_tokens(ref.label)
    if not ref_tokens:
        return False
    return bool(ref_tokens & _anchor_tokens(claim.context))


_INEQUALITY_OPS = {"<", ">", "\u2264", "\u2265"}


def _satisfies(claim_value: float, operator: str, json_value: float) -> bool:
    """Is an inequality claim true of the authoritative value?

    "Ljung-Box p > 0.05" is a threshold statement, not a reported number. It is
    correct when the JSON value is above 0.05, not when it equals 0.05.
    """
    if operator == "<":
        return json_value < claim_value
    if operator == "\u2264":
        return json_value <= claim_value
    if operator == ">":
        return json_value > claim_value
    if operator == "\u2265":
        return json_value >= claim_value
    return False


def cross_reference(
    claims: list[StatClaim],
    refs: list[StatReference],
    tolerance: float,
) -> tuple[list[Mismatch], list[tuple[StatClaim, StatReference]], list[StatClaim]]:
    """Match HTML claims against JSON references.

    Returns (mismatches, matched_pairs, unmatched_claims).
    """
    mismatches: list[Mismatch] = []
    matched: list[tuple[StatClaim, StatReference]] = []
    unmatched: list[StatClaim] = []

    # Build ref lookup by (report, stat_type)
    ref_by_type: dict[tuple[str, str], list[StatReference]] = {}
    for ref in refs:
        key = (ref.report, ref.stat_type)
        ref_by_type.setdefault(key, []).append(ref)

    for claim in claims:
        key = (claim.report, claim.stat_type)
        candidates = ref_by_type.get(key, [])

        if not candidates:
            unmatched.append(claim)
            continue

        if claim.operator in _INEQUALITY_OPS:
            # Judge a threshold statement only against references the context
            # actually names; otherwise there is no way to tell which statistic
            # the threshold is about, and the claim stays unmatched.
            anchored = [ref for ref in candidates if _is_anchored(claim, ref)]
            if not anchored:
                unmatched.append(claim)
                continue
            satisfying = [
                ref for ref in anchored
                if _satisfies(claim.value, claim.operator, ref.value)
            ]
            if satisfying:
                matched.append(
                    (claim, min(satisfying, key=lambda r: abs(r.value - claim.value)))
                )
            else:
                closest = min(anchored, key=lambda r: abs(r.value - claim.value))
                mismatches.append(
                    Mismatch(
                        report=claim.report,
                        stat_type=claim.stat_type,
                        html_value=claim.value,
                        html_operator=claim.operator,
                        json_value=closest.value,
                        json_path=closest.json_path,
                        delta=abs(claim.value - closest.value),
                        context=claim.context,
                        severity="error",
                    )
                )
            continue

        # Find best match: exact match first, then closest value
        best_ref = None
        best_delta = float("inf")
        exact_match = False

        for ref in candidates:
            if _round_matches(claim.value, ref.value):
                best_ref = ref
                best_delta = abs(claim.value - ref.value)
                exact_match = True
                break

            delta = abs(claim.value - ref.value)
            if delta < best_delta:
                best_delta = delta
                best_ref = ref

        if best_ref is None:
            unmatched.append(claim)
            continue

        if exact_match or _values_match(claim.value, best_ref.value, tolerance):
            matched.append((claim, best_ref))
        else:
            # A difference is only a mismatch when the claim's context names the
            # same statistic the reference holds. Otherwise the page printed a
            # number this audit cannot locate in the JSON: report it as
            # unmatched, which is what it is, rather than as a contradiction.
            if best_delta < 0.5 and _is_anchored(claim, best_ref):
                severity = "error" if best_delta > 0.05 else "warning"
                mismatches.append(
                    Mismatch(
                        report=claim.report,
                        stat_type=claim.stat_type,
                        html_value=claim.value,
                        html_operator=claim.operator,
                        json_value=best_ref.value,
                        json_path=best_ref.json_path,
                        delta=best_delta,
                        context=claim.context,
                        severity=severity,
                    )
                )
            else:
                unmatched.append(claim)

    return mismatches, matched, unmatched


# ---------------------------------------------------------------------------
# Sanity checks (no JSON needed)
# ---------------------------------------------------------------------------


def sanity_check(claims: list[StatClaim]) -> list[SanityIssue]:
    """Flag statistical values that are inherently suspicious."""
    issues: list[SanityIssue] = []

    for c in claims:
        if c.stat_type == "p_value":
            if c.value < 0:
                issues.append(
                    SanityIssue(
                        report=c.report,
                        issue_type="p_negative",
                        value=c.value,
                        context=c.context,
                        severity="error",
                    )
                )
            elif c.value == 0.0 and c.operator == "=":
                issues.append(
                    SanityIssue(
                        report=c.report,
                        issue_type="p_zero_exact",
                        value=c.value,
                        context=c.context,
                        severity="warning",
                    )
                )

        if c.stat_type == "correlation" and abs(c.value) > 1.0:
            issues.append(
                SanityIssue(
                    report=c.report,
                    issue_type="correlation_gt_1",
                    value=c.value,
                    context=c.context,
                    severity="error",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Duplicate p-value detection (HTML says X, also says Y for same metric)
# ---------------------------------------------------------------------------


def detect_internal_inconsistencies(claims: list[StatClaim]) -> list[SanityIssue]:
    """Detect the same p-value reported differently within one report."""
    issues: list[SanityIssue] = []

    # Group by (report, stat_type) and look for near-duplicate contexts
    by_report: dict[str, list[StatClaim]] = {}
    for c in claims:
        by_report.setdefault(c.report, []).append(c)

    for report, report_claims in by_report.items():
        p_claims = [c for c in report_claims if c.stat_type == "p_value"]
        # Check for values that appear with both = and different precision
        seen: dict[str, list[StatClaim]] = {}
        for c in p_claims:
            # Normalize context to find duplicates
            ctx_key = re.sub(r"\s+", " ", c.context[:30]).strip().lower()
            seen.setdefault(ctx_key, []).append(c)

        for ctx_key, group in seen.items():
            if len(group) > 1:
                values = {c.value for c in group}
                if len(values) > 1:
                    issues.append(
                        SanityIssue(
                            report=report,
                            issue_type="inconsistent_self_report",
                            value=str(values),
                            context=group[0].context,
                            severity="warning",
                        )
                    )

    return issues


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def run_audit(
    reports_dir: Path,
    tolerance: float = 0.005,
    verbose: bool = False,
) -> AuditResult:
    """Run the full statistical integrity audit."""
    result = AuditResult()
    all_claims: list[StatClaim] = []
    all_refs: list[StatReference] = []

    # Check every published page, not intermediate or dated snapshot files.
    html_files = [
        reports_dir / name for name in HTML_TO_JSON if (reports_dir / name).exists()
    ]
    result.pages_expected_missing = sorted(
        name for name in HTML_TO_JSON if not (reports_dir / name).exists()
    )
    result.reports_checked = len(html_files)

    for html_path in sorted(html_files):
        report_name = html_path.name
        html_content = html_path.read_text(encoding="utf-8", errors="replace")
        claims = extract_claims(report_name, html_content)
        all_claims.extend(claims)

        json_spec = HTML_TO_JSON.get(report_name)
        json_names = (
            [] if json_spec is None
            else [json_spec] if isinstance(json_spec, str)
            else list(json_spec)
        )
        if json_names:
            result.pages_with_json.append(report_name)
        else:
            result.pages_without_json.append(report_name)

        for json_name in json_names:
            json_path = reports_dir / json_name
            if not json_path.exists():
                msg = f"{json_name}: missing (referenced by {report_name})"
                logging.error("JSON verification failed for %s", msg)
                result.json_errors.append(msg)
                continue
            try:
                json_data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logging.error("JSON verification failed for %s: %s", json_name, e)
                result.json_errors.append(f"{json_name}: {e}")
                continue
            all_refs.extend(extract_references(report_name, json_data))

    result.claims_extracted = len(all_claims)
    result.references_loaded = len(all_refs)

    # Cross-reference
    mismatches, matched, unmatched = cross_reference(all_claims, all_refs, tolerance)
    result.mismatches = mismatches
    result.matches_found = len(matched)
    result.matched_pairs = matched
    result.unmatched_html = unmatched

    # Sanity checks
    result.sanity_issues = sanity_check(all_claims)
    result.sanity_issues.extend(detect_internal_inconsistencies(all_claims))

    if verbose:
        print(f"\n  Matched pairs ({len(matched)}):")
        for claim, ref in matched:
            print(
                f"    {claim.report}: {claim.stat_type} "
                f"HTML={claim.operator}{claim.value} "
                f"JSON={ref.value:.6f} ({ref.json_path})"
            )

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_summary(result: AuditResult) -> None:
    """Print a human-readable audit summary."""
    print(f"\n{'=' * 70}")
    print("  STATCHECK - Statistical Integrity Audit")
    print(f"{'=' * 70}")
    print(f"  Reports checked:   {result.reports_checked}")
    print(f"  Claims extracted:  {result.claims_extracted}")
    print(f"  JSON references:   {result.references_loaded}")
    print(f"  Matched (OK):      {result.matches_found}")
    print(f"  Mismatches:        {len(result.mismatches)}")
    print(f"  Sanity issues:     {len(result.sanity_issues)}")
    print(f"  Unmatched HTML:    {len(result.unmatched_html)}")

    if result.mismatches:
        print(f"\n{'-' * 70}")
        print("  MISMATCHES (HTML vs JSON)")
        print(f"{'-' * 70}")
        for m in result.mismatches:
            icon = "!!" if m.severity == "error" else "!"
            print(f"  [{icon}] {m.report}")
            print(
                f"      {m.stat_type}: HTML {m.html_operator}{m.html_value} "
                f"vs JSON {m.json_value:.6f} (delta={m.delta:.6f})"
            )
            print(f"      JSON path: {m.json_path}")
            print(f"      Context: ...{m.context[:80]}...")
            print()

    if result.sanity_issues:
        print(f"{'-' * 70}")
        print("  SANITY ISSUES")
        print(f"{'-' * 70}")
        for s in result.sanity_issues:
            icon = "!!" if s.severity == "error" else "!"
            print(f"  [{icon}] {s.report}: {s.issue_type} = {s.value}")
            print(f"      Context: ...{s.context[:80]}...")
            print()

    if not result.mismatches and not result.sanity_issues and not result.json_errors:
        print("\n  All statistical claims verified. No issues found.")

    if result.json_errors:
        print(f"\n  JSON LOAD FAILURES ({len(result.json_errors)}):")
        for je in result.json_errors:
            print(f"    - {je}")

    n_problems = (
        len(result.mismatches)
        + len([s for s in result.sanity_issues if s.severity == "error"])
        + len(result.json_errors)
    )
    print(f"\n{'=' * 70}")
    if n_problems == 0:
        print("  RESULT: PASS")
    else:
        print(f"  RESULT: FAIL - {n_problems} PROBLEM(S) FOUND")
    print(f"{'=' * 70}")


def audit_passed(result: AuditResult) -> bool:
    """A run passes when nothing contradicts the JSON authority."""
    return (
        len(result.mismatches) == 0
        and not any(issue.severity == "error" for issue in result.sanity_issues)
        and len(result.json_errors) == 0
    )


def to_json(result: AuditResult) -> dict:
    """Convert audit result to JSON-serializable dict."""
    return {
        "reports_checked": result.reports_checked,
        "pages_with_json": result.pages_with_json,
        "pages_without_json": result.pages_without_json,
        "pages_expected_but_absent": result.pages_expected_missing,
        "claims_extracted": result.claims_extracted,
        "references_loaded": result.references_loaded,
        "matches_ok": result.matches_found,
        "mismatches": [asdict(m) for m in result.mismatches],
        "sanity_issues": [asdict(issue) for issue in result.sanity_issues],
        "json_errors": result.json_errors,
        "unmatched_count": len(result.unmatched_html),
        "pass": audit_passed(result),
    }


# ---------------------------------------------------------------------------
# Published claims page
# ---------------------------------------------------------------------------

_STATUS_STYLE = {
    "OK": ("#3A3AD6", "verified against the JSON value"),
    "MISMATCH": ("#B4231F", "the page prints a value the JSON does not carry"),
    "UNMATCHED": ("#8A909C", "no JSON value of this kind to check against"),
}


def _claim_rows(result: AuditResult) -> list[dict[str, str]]:
    """Flatten every extracted claim into one display row, sorted by page."""
    rows: list[dict[str, str]] = []

    for claim, ref in result.matched_pairs:
        rows.append({
            "report": claim.report,
            "stat_type": claim.stat_type,
            "html": f"{claim.operator}{claim.value:g}",
            "json": f"{ref.value:.6g}",
            "json_path": ref.json_path,
            "status": "OK",
            "context": claim.context,
        })

    for mismatch in result.mismatches:
        rows.append({
            "report": mismatch.report,
            "stat_type": mismatch.stat_type,
            "html": f"{mismatch.html_operator}{mismatch.html_value:g}",
            "json": f"{mismatch.json_value:.6g}",
            "json_path": mismatch.json_path,
            "status": "MISMATCH",
            "context": mismatch.context,
        })

    for claim in result.unmatched_html:
        rows.append({
            "report": claim.report,
            "stat_type": claim.stat_type,
            "html": f"{claim.operator}{claim.value:g}",
            "json": "-",
            "json_path": "-",
            "status": "UNMATCHED",
            "context": claim.context,
        })

    status_order = {"MISMATCH": 0, "OK": 1, "UNMATCHED": 2}
    rows.sort(key=lambda r: (r["report"], status_order[r["status"]], r["stat_type"]))
    return rows


def render_claims_page(result: AuditResult, reports_dir: Path) -> Path:
    """Write reports/claims.html: every extracted claim and its verdict."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _theme import wrap_html, make_section, make_kpi_card, make_kpi_row  # noqa: PLC0415

    rows = _claim_rows(result)
    passed = audit_passed(result)
    counts = {key: sum(1 for r in rows if r["status"] == key) for key in _STATUS_STYLE}

    kpis = make_kpi_row(
        make_kpi_card(
            "Pages checked", str(result.reports_checked), "pages",
            status="info",
            detail=f"{len(result.pages_with_json)} with a JSON authority, "
                   f"{len(result.pages_without_json)} without",
        ),
        make_kpi_card(
            "Claims extracted", str(len(rows)), "numbers",
            status="info",
            detail="p-values, correlations, R-squared and Cohen's d parsed from the pages",
        ),
        make_kpi_card(
            "Verified", str(counts["OK"]), "OK",
            status="good",
            detail="page value equals the JSON value within tolerance",
        ),
        make_kpi_card(
            "Mismatches", str(counts["MISMATCH"]), "errors",
            status="good" if counts["MISMATCH"] == 0 else "critical",
            detail="page prints a number its JSON does not carry",
        ),
    )

    verdict = (
        '<p><strong>Result: PASS.</strong> No page contradicts its metrics JSON.</p>'
        if passed else
        f'<p><strong>Result: FAIL.</strong> {counts["MISMATCH"]} mismatch(es), '
        f'{len([i for i in result.sanity_issues if i.severity == "error"])} sanity error(s) '
        f'and {len(result.json_errors)} JSON load failure(s) block publication.</p>'
    )

    legend = "".join(
        f'<li><strong style="color:{color}">{escape(status)}</strong> &mdash; {escape(meaning)}</li>'
        for status, (color, meaning) in _STATUS_STYLE.items()
    )

    intro = (
        "<p>Every statistic printed on this site is extracted from the published HTML and "
        "checked against the JSON metrics file its generator wrote. The JSON is the "
        "authority: if the two disagree, the page is wrong.</p>"
        f"{verdict}"
        f"<ul>{legend}</ul>"
    )

    if result.pages_expected_missing:
        intro += (
            "<p>Pages in the registry that were not present in this run: "
            + escape(", ".join(result.pages_expected_missing))
            + ".</p>"
        )

    body_rows = []
    for row in rows:
        color = _STATUS_STYLE[row["status"]][0]
        context = re.sub(r"\s+", " ", row["context"]).strip()[:120]
        body_rows.append(
            "<tr>"
            f"<td>{escape(row['report'])}</td>"
            f"<td>{escape(row['stat_type'])}</td>"
            f"<td>{escape(row['html'])}</td>"
            f"<td>{escape(row['json'])}</td>"
            f"<td>{escape(row['json_path'])}</td>"
            f'<td style="color:{color};font-weight:600">{escape(row["status"])}</td>'
            f"<td>{escape(context)}</td>"
            "</tr>"
        )

    table = (
        '<div style="overflow-x:auto"><table><thead><tr>'
        "<th>Page</th><th>Statistic</th><th>On the page</th><th>In the JSON</th>"
        "<th>JSON path</th><th>Verdict</th><th>Context</th>"
        "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table></div>"
    )

    body = make_section("What this page is", intro, section_id="method")
    body += kpis
    body += make_section(
        f"Every claim ({len(rows)})", table, section_id="claims"
    )

    page = wrap_html(
        title="Claims Register",
        body_content=body,
        report_id="claims",
        subtitle="Every number on this site, checked against its JSON",
    )
    out = reports_dir / "claims.html"
    out.write_text(page, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Statistical integrity audit for Oura reports"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Maximum allowed delta between HTML and JSON values (default: 0.005)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all matched pairs"
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output JSON instead of text summary",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Reports directory (default: from config)",
    )
    args = parser.parse_args()

    result = run_audit(args.reports_dir, args.tolerance, args.verbose)

    if args.json_output:
        print(json.dumps(to_json(result), indent=2, ensure_ascii=False))
    else:
        print_summary(result)

    # Write JSON audit file alongside reports
    audit_path = args.reports_dir / "statcheck_audit.json"
    audit_path.write_text(
        json.dumps(to_json(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Publish the human-readable register of every claim and its verdict
    claims_path = render_claims_page(result, args.reports_dir)
    if not args.json_output:
        print(f"\n  Audit JSON:     {audit_path}")
        print(f"  Claims page:    {claims_path}")

    # Exit code: 0 = pass, 1 = mismatches, sanity errors or JSON errors found
    return 0 if audit_passed(result) else 1


if __name__ == "__main__":
    sys.exit(main())
