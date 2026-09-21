#!/usr/bin/env python3
"""Multi-patient pipeline orchestrator.

Runs Henrik's full pipeline, standalone dashboards for Mitch and Wenche,
all comparative scripts, and regenerates the index.

Usage:
    python run_multi.py                  # everything
    python run_multi.py --standalone-only  # just Mitch + Wenche standalone
    python run_multi.py --comparative-only # just the 7 comparative scripts
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
SUBPROCESS_ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}

HENRIK_SCRIPTS = [
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
    "analyze_weekly_tracker.py",
    "analyze_rux_forecast.py",
    "analyze_piecewise_its.py",
    "analyze_sequential_ci.py",
    "analyze_placebo_tests.py",
    "analyze_tau_u.py",
    "generate_roadmap.py",
]

STANDALONE_CMDS = [
    [sys.executable, str(ANALYSIS_DIR / "analyze_patient_standalone.py"), "--profile", "mitch"],
    [sys.executable, str(ANALYSIS_DIR / "analyze_patient_standalone.py"), "--profile", "wenche"],
]

COMPARATIVE_SCRIPTS = [
    "analyze_comparative_autonomic.py",
    "analyze_comparative_treatment.py",
    "analyze_comparative_sleep.py",
    "analyze_comparative_coupling.py",
    "analyze_comparative_anomalies.py",
    "analyze_comparative_breathing.py",
    "analyze_comparative_temperature.py",
]

EXTRA_SCRIPTS = [
    "analyze_mitch_changepoints.py",
    "generate_index.py",
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def run_step(label: str, cmd: list[str]) -> tuple[str, str, float]:
    """Run a single subprocess and return (label, status, elapsed)."""
    log(f"\n{'─' * 60}\n  {label}\n{'─' * 60}")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=False, env=SUBPROCESS_ENV, timeout=600)
        elapsed = time.perf_counter() - t0
        status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        status = "TIMEOUT"
    except Exception as e:
        elapsed = time.perf_counter() - t0
        status = f"ERROR: {e}"
    log(f"  -> {status} ({elapsed:.1f}s)")
    return (label, status, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-patient pipeline orchestrator.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--standalone-only", action="store_true", help="Run only Mitch + Wenche standalone dashboards.")
    mode.add_argument("--comparative-only", action="store_true", help="Run only the 7 comparative scripts.")
    args = parser.parse_args()

    t_total = time.perf_counter()
    results: list[tuple[str, str, float]] = []
    log("=" * 60)
    log("  MULTI-PATIENT PIPELINE")
    log("=" * 60)

    if not args.standalone_only and not args.comparative_only:
        # Phase 1: Henrik's full pipeline
        for script in HENRIK_SCRIPTS:
            cmd = [sys.executable, str(ANALYSIS_DIR / script)]
            results.append(run_step(script, cmd))

    if not args.comparative_only:
        # Phase 2: Standalone dashboards
        for cmd in STANDALONE_CMDS:
            label = f"standalone --profile {cmd[-1]}"
            results.append(run_step(label, cmd))

    if not args.standalone_only:
        # Phase 3: Comparative scripts
        for script in COMPARATIVE_SCRIPTS:
            cmd = [sys.executable, str(ANALYSIS_DIR / script)]
            results.append(run_step(script, cmd))

    if not args.standalone_only and not args.comparative_only:
        # Phase 4: Extras (changepoints + index)
        for script in EXTRA_SCRIPTS:
            cmd = [sys.executable, str(ANALYSIS_DIR / script)]
            results.append(run_step(script, cmd))

    # Summary
    total = time.perf_counter() - t_total
    ok = sum(1 for _, s, _ in results if s == "OK")
    log(f"\n{'=' * 60}")
    log("  SUMMARY")
    log(f"{'=' * 60}")
    for label, status, elapsed in results:
        log(f"  {label:50s} {status} ({elapsed:.1f}s)")
    log(f"\n  Total: {total:.1f}s  Passed: {ok}/{len(results)}")

    if ok == 0 and results:
        sys.exit(1)


if __name__ == "__main__":
    main()
