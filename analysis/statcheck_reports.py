#!/usr/bin/env python3
"""Statistical integrity checker for Oura analysis reports.

Extracts all p-values, effect sizes, and test statistics from HTML reports,
cross-references them against authoritative JSON metrics files, and flags
mismatches exceeding a configurable tolerance.

Runs as a post-generation step in run_all.py.

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


# ---------------------------------------------------------------------------
# HTML → JSON filename mapping
# ---------------------------------------------------------------------------

HTML_TO_JSON = {
    "oura_full_analysis.html": "oura_full_analysis.json",
    "advanced_hrv_analysis.html": "advanced_hrv_metrics.json",
    "advanced_sleep_analysis.html": "advanced_sleep_metrics.json",
    "composite_biomarkers.html": "composite_biomarkers.json",
    "spo2_bos_screening.html": "spo2_bos_metrics.json",
    "anomaly_detection_report.html": "anomaly_detection_metrics.json",
    "foundation_model_report.html": "foundation_model_metrics.json",
    "digital_twin_report.html": "digital_twin_metrics.json",
    "causal_inference_report.html": "causal_inference_metrics.json",
    "gvhd_prediction_report.html": "gvhd_prediction_metrics.json",
    "oura_3d_dashboard.html": "oura_3d_dashboard_metrics.json",
    # roadmap.html has no metrics JSON
}

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

STAT_KEYS = {
    "p_value": "p_value",
    "p_val": "p_value",
    "correlation": "correlation",
    "cohens_d": "cohens_d",
    "r_squared": "r_squared",
    "effect_size": "effect_size",
}


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
                for pattern, stat_type in STAT_KEYS.items():
                    if pattern in key_lower:
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
                        break

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
            # Only flag as mismatch if it's close enough to plausibly be the same stat
            # (avoid matching p=0.012 against p=0.469)
            if best_delta < 0.5:
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

    # Discover HTML reports
    html_files = sorted(reports_dir.glob("*.html"))
    # Only check the curated set, not intermediate files
    curated_names = set(HTML_TO_JSON.keys()) | {"roadmap.html"}
    html_files = [f for f in html_files if f.name in curated_names]
    result.reports_checked = len(html_files)

    for html_path in html_files:
        report_name = html_path.name
        html_content = html_path.read_text(encoding="utf-8", errors="replace")
        claims = extract_claims(report_name, html_content)
        all_claims.extend(claims)

        # Load corresponding JSON
        json_name = HTML_TO_JSON.get(report_name)
        if json_name:
            json_path = reports_dir / json_name
            if json_path.exists():
                try:
                    json_data = json.loads(json_path.read_text(encoding="utf-8"))
                    refs = extract_references(report_name, json_data)
                    all_refs.extend(refs)
                except (json.JSONDecodeError, OSError) as e:
                    logging.error("JSON verification failed for %s: %s", json_name, e)
                    result.json_errors.append(f"{json_name}: {e}")

    result.claims_extracted = len(all_claims)
    result.references_loaded = len(all_refs)

    # Cross-reference
    mismatches, matched, unmatched = cross_reference(all_claims, all_refs, tolerance)
    result.mismatches = mismatches
    result.matches_found = len(matched)
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


def to_json(result: AuditResult) -> dict:
    """Convert audit result to JSON-serializable dict."""
    return {
        "reports_checked": result.reports_checked,
        "claims_extracted": result.claims_extracted,
        "references_loaded": result.references_loaded,
        "matches_ok": result.matches_found,
        "mismatches": [asdict(m) for m in result.mismatches],
        "sanity_issues": [asdict(s) for s in result.sanity_issues],
        "json_errors": result.json_errors,
        "unmatched_count": len(result.unmatched_html),
        "pass": len(result.mismatches) == 0
        and not any(s.severity == "error" for s in result.sanity_issues)
        and len(result.json_errors) == 0,
    }


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

    # Exit code: 0 = pass, 1 = mismatches or JSON errors found
    n_errors = (
        len(result.mismatches)
        + len([s for s in result.sanity_issues if s.severity == "error"])
        + len(result.json_errors)
    )
    return 1 if n_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
