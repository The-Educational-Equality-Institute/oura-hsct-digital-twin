#!/usr/bin/env python3
"""Run every ``test_*.py`` under ``tests/`` and report a pass/fail summary.

Stdlib only — works without pytest. Each test file has its own ``_run_all()``
that prints per-test status and exits non-zero on any failure. This runner
invokes each as a subprocess, so one failing suite doesn't stop the rest.

Usage:
    python run_tests.py
    python run_tests.py --fast   # stop on first failing suite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Stop on first failing suite",
    )
    args = parser.parse_args()

    test_files = sorted(TESTS_DIR.rglob("test_*.py"))
    if not test_files:
        print(f"No test files found under {TESTS_DIR}", file=sys.stderr)
        return 1

    results: list[tuple[Path, int, str]] = []
    for test_file in test_files:
        rel = test_file.relative_to(ROOT)
        print(f"\n=== {rel} ===")
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        last_line = ""
        if result.stdout:
            lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
            if lines:
                last_line = lines[-1]
        results.append((rel, result.returncode, last_line))

        if args.fast and result.returncode != 0:
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    failed = 0
    for rel, rc, last in results:
        status = "PASS" if rc == 0 else "FAIL"
        print(f"  {status}  {rel}  {last}")
        if rc != 0:
            failed += 1

    print("-" * 60)
    print(f"  {len(results) - failed}/{len(results)} suites passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
