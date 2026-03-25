#!/usr/bin/env python3
"""Run all Oura analysis modules and generate reports.

Executes 12 pipeline scripts sequentially, outputting all HTML reports
and JSON metrics to oura-digital-twin/reports/.

Usage:
    cd oura-digital-twin
    python run_all.py           # exit 0 if at least one script passes
    python run_all.py --strict  # exit 1 if ANY script fails
"""
import argparse
import os
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPORTS_DIR  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SEND_BUNDLE_DIR = REPORTS_DIR / "send_bundle"
SUBPROCESS_ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}

SCRIPTS = [
    "analyze_oura_full.py",
    "analyze_oura_advanced_hrv.py",
    "analyze_oura_sleep_advanced.py",
    "analyze_oura_biomarkers.py",
    "analyze_oura_spo2_trend.py",
    "analyze_oura_anomalies.py",
    "analyze_oura_foundation_models.py",
    "analyze_oura_digital_twin.py",
    "analyze_oura_causal.py",
    "analyze_oura_gvhd_predict.py",
    "generate_oura_3d_dashboard.py",
    "generate_roadmap.py",
]

SEND_BUNDLE_HTML = [
    "oura_full_analysis.html",
    "advanced_hrv_analysis.html",
    "advanced_sleep_analysis.html",
    "composite_biomarkers.html",
    "spo2_bos_screening.html",
    "anomaly_detection_report.html",
    "foundation_model_report.html",
    "digital_twin_report.html",
    "causal_inference_report.html",
    "gvhd_prediction_report.html",
    "oura_3d_dashboard.html",
    "roadmap.html",
]

SEND_BUNDLE_JSON = [
    "oura_full_analysis.json",
    "advanced_hrv_metrics.json",
    "advanced_sleep_metrics.json",
    "composite_biomarkers.json",
    "spo2_bos_metrics.json",
    "anomaly_detection_metrics.json",
    "foundation_model_metrics.json",
    "digital_twin_metrics.json",
    "causal_inference_metrics.json",
    "gvhd_prediction_metrics.json",
    "oura_3d_dashboard_metrics.json",
]


def log(message: str = "") -> None:
    """Write orchestration logs without stdout buffering surprises."""
    print(message, flush=True)


def assemble_send_bundle() -> tuple[list[str], list[str]]:
    """Copy the curated release surface to reports/send_bundle."""
    SEND_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    missing_html = [name for name in SEND_BUNDLE_HTML if not (REPORTS_DIR / name).exists()]
    missing_json = [name for name in SEND_BUNDLE_JSON if not (REPORTS_DIR / name).exists()]
    if missing_html or missing_json:
        missing = missing_html + missing_json
        raise FileNotFoundError(
            "Cannot assemble send bundle. Missing expected report artifacts: "
            + ", ".join(missing)
        )

    expected_names = set(SEND_BUNDLE_HTML + SEND_BUNDLE_JSON + ["SEND_MANIFEST.md"])
    for existing in SEND_BUNDLE_DIR.iterdir():
        if existing.is_file() and existing.name not in expected_names:
            existing.unlink()

    copied_html: list[str] = []
    copied_json: list[str] = []
    for name in SEND_BUNDLE_HTML:
        src = REPORTS_DIR / name
        if src.exists():
            shutil.copy2(src, SEND_BUNDLE_DIR / name)
            copied_html.append(name)

    for name in SEND_BUNDLE_JSON:
        src = REPORTS_DIR / name
        shutil.copy2(src, SEND_BUNDLE_DIR / name)
        copied_json.append(name)

    # The send bundle intentionally excludes dated snapshots; keep references internal.
    full_json_path = SEND_BUNDLE_DIR / "oura_full_analysis.json"
    if full_json_path.exists():
        payload = json.loads(full_json_path.read_text(encoding="utf-8"))
        canonical_html = payload.get("canonical_html", "oura_full_analysis.html")
        payload["report_html"] = canonical_html
        payload["date_stamped_html"] = canonical_html
        full_json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest_lines = [
        "# Oura Send Bundle",
        "",
        "Curated release bundle for external sharing.",
        "",
        "Included HTML reports:",
        *[f"- `{name}`" for name in copied_html],
        "",
        "Included JSON metrics:",
        *[f"- `{name}`" for name in copied_json],
        "",
        "Excluded on purpose:",
        "- historical dated snapshots such as `oura_full_analysis_20260323.html`",
        "- internal/demo artifacts such as `css_clinical_patterns.html`",
        "- screenshots and intermediate files not needed by the recipient",
        "- `causal_timeseries.json` because it is an internal generation input, not a send artifact",
    ]
    (SEND_BUNDLE_DIR / "SEND_MANIFEST.md").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    return copied_html, copied_json

def main():
    parser = argparse.ArgumentParser(description="Run all Oura analysis scripts.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if ANY script fails (default: exit 0 if at least one passes)",
    )
    args = parser.parse_args()

    t_total = time.perf_counter()
    log("=" * 70)
    n_scripts = len(SCRIPTS)
    log(f"  OURA ANALYSIS PIPELINE - ALL {n_scripts} SCRIPTS")
    log(f"  Output: {REPORTS_DIR}")
    log(f"  Mode: {'strict (any failure = exit 1)' if args.strict else 'resilient (exit 0 if any pass)'}")
    log("=" * 70)

    results = []
    for i, script in enumerate(SCRIPTS, 1):
        script_path = ANALYSIS_DIR / script
        if not script_path.exists():
            log(f"\n[{i}/{n_scripts}] SKIP {script} - file not found")
            results.append((script, "MISSING"))
            continue

        log(f"\n{'─' * 70}")
        log(f"[{i}/{n_scripts}] Running {script}...")
        log(f"{'─' * 70}")
        t0 = time.perf_counter()

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,
                env=SUBPROCESS_ENV,
                timeout=600,  # 10 min per script
            )
            elapsed = time.perf_counter() - t0
            status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
            results.append((script, status, elapsed))
            log(f"\n  -> {status} ({elapsed:.1f}s)")
        except subprocess.TimeoutExpired:
            results.append((script, "TIMEOUT"))
            log("\n  -> TIMEOUT (>600s)")
        except Exception as e:
            results.append((script, f"ERROR: {e}"))
            log(f"\n  -> ERROR: {e}")

    # Summary
    total_time = time.perf_counter() - t_total
    successes = sum(1 for e in results if e[1] == "OK")
    failures = n_scripts - successes
    log(f"\n{'=' * 70}")
    log("  PIPELINE COMPLETE")
    log(f"{'=' * 70}")
    for entry in results:
        script = entry[0]
        status = entry[1]
        elapsed = f" ({entry[2]:.1f}s)" if len(entry) > 2 else ""
        log(f"  {script:45s} {status}{elapsed}")

    log(f"\n  Total runtime: {total_time:.1f}s")
    log(f"  Reports: {REPORTS_DIR}")
    log(f"  Passed: {successes}/{n_scripts}  Failed: {failures}/{n_scripts}")

    if failures:
        log(f"\n  {failures}/{n_scripts} script(s) failed.")

    # Assemble send bundle only when all scripts passed
    if failures == 0:
        try:
            copied_html, copied_json = assemble_send_bundle()
            log(f"\n  Curated HTML reports: {len(copied_html)}")
            for name in copied_html:
                f = REPORTS_DIR / name
                size_mb = f.stat().st_size / (1024 * 1024)
                log(f"    {name} ({size_mb:.1f} MB)")
            log(f"  Curated JSON metrics: {len(copied_json)}")
            for name in copied_json:
                log(f"    {name}")
            log(f"\n  Send bundle: {SEND_BUNDLE_DIR}")
            log(f"  Manifest: {SEND_BUNDLE_DIR / 'SEND_MANIFEST.md'}")
        except FileNotFoundError as exc:
            log(f"\n  Send bundle assembly failed: {exc}")
    else:
        log("  Send bundle skipped (not all scripts passed).")

    # Exit code logic:
    # --strict: exit 1 if any script failed
    # default:  exit 1 only if ALL scripts failed (partial success = exit 0)
    if successes == 0:
        log("\n  All scripts failed. Exiting with code 1.")
        sys.exit(1)
    if args.strict and failures > 0:
        log("\n  Strict mode: exiting with code 1 due to failures.")
        sys.exit(1)


if __name__ == "__main__":
    main()
