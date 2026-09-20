"""Smoke tests over the reports the pipeline has already produced.

These tests READ reports/. They never regenerate anything, so they are cheap
enough to run after every pipeline run and they fail on exactly the condition
that shipped to production for months: a page that renders, links correctly and
looks finished while its body says a module could not run.

The page registry is analysis/statcheck_reports.HTML_TO_JSON, so there is one
list of published pages rather than two that drift apart.

Run directly (python tests/analysis/test_report_outputs.py) or through
python run_tests.py.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from statcheck_reports import AUDIT_OUTPUT_PAGES, HTML_TO_JSON

from config import REPORTS_DIR

try:
    from _theme import REPORT_REGISTRY
except ImportError:  # pragma: no cover - theme is always present in this repo
    REPORT_REGISTRY = []


# ---------------------------------------------------------------------------
# What counts as a broken page
# ---------------------------------------------------------------------------

# Case-sensitive on purpose. "failed fits remain" and "failed_total": 0 are
# legitimate prose and legitimate field names; "No module named" is not.
RUNTIME_FAILURE_STRINGS = (
    "No module named",
    "No model backend",
    "not installed",
    "Optional dependency unavailable",
    "dependency unavailable",
    "Traceback (most recent call last)",
    "FAILED",
)

# A themed page carries the shared CSS and navigation, so anything materially
# smaller than this never finished rendering. 404.html is deliberately lean.
MIN_HTML_BYTES = 20_000
MIN_HTML_BYTES_OVERRIDE = {
    "404.html": 4_000,
}

# Top-level keys each metrics JSON promises, taken from the dict its generator
# writes (structural sections plus the provenance stamp). A page that loses one
# of these has lost a section, whatever the HTML still looks like.
REQUIRED_JSON_KEYS: dict[str, tuple[str, ...]] = {
    "advanced_hrv_metrics.json": (
        "allostatic_load", "baevsky", "cosinor", "data_range", "dfa", "entropy",
        "frequency_domain", "generated", "generated_at", "hjorth", "hr_complexity", "mse",
        "nightly_hr_variability", "rqa", "toichi",
    ),
    "advanced_sleep_metrics.json": (
        "clinical_interpretation", "data_range", "efficiency", "fragmentation",
        "generated_at", "hrv_coupling", "meta", "movement", "rem_latency",
        "ruxolitinib_comparison", "sleep_cycles", "transition_matrix", "ultradian_rhythm",
    ),
    "anomaly_detection_metrics.json": (
        "data_range", "ensemble", "generated_at", "methods", "validation",
    ),
    "causal_inference_metrics.json": (
        "causal_impact", "confounder_analysis", "data_range", "generated", "generated_at",
        "individual_metric_tests", "mediation", "pcmci", "placebo_tests",
        "transfer_entropy",
    ),
    "comparative_activity_recovery_coupling.json": (
        "correlation_matrices", "coupling_assessment", "cross_correlations",
        "cross_patient_comparison", "dose_response", "generated_at", "lag1_correlations",
        "regression",
    ),
    "comparative_anomaly_metrics.json": ("comparison", "generated_at", "patients",),
    "comparative_autonomic_metrics.json": (
        "comparison", "generated_at", "patients", "recent_window",
        "severity_classification",
    ),
    "comparative_breathing_metrics.json": (
        "anomalies", "bos_screening", "comparison", "generated_at", "patients",
        "ruxolitinib_effect",
    ),
    "comparative_sleep_metrics.json": (
        "architecture", "benchmarks", "comparison", "efficiency", "meta",
        "recovery_indicators", "timing",
    ),
    "comparative_temperature_metrics.json": (
        "anomaly_counts", "baseline_stats", "cross_patient", "generated", "known_events",
        "lag_correlations", "rux_pre_post", "top_spikes",
    ),
    "comparative_treatment_response.json": (
        "generated_at", "methods", "multi_metric_convergence", "patients",
    ),
    "composite_biomarkers.json": (
        "biomarkers", "data_range", "generated", "generated_at", "three_period",
        "treatment_response",
    ),
    "digital_twin_metrics.json": (
        "data_range", "drug_response", "generated_at", "kalman", "prediction", "scalers",
        "sensor_fusion", "ukf",
    ),
    "foundation_model_metrics.json": (
        "chronos_available", "data_range", "ensemble_consensus", "generated_at",
    ),
    "gvhd_prediction_metrics.json": (
        "alerts", "bos", "composite", "data_range", "features", "generated_at",
        "progress_log", "rslds", "temperature",
    ),
    "mitch_changepoint_metrics.json": (
        "australia_analysis", "changepoints", "classifications", "data_gaps",
        "generated_at", "known_events", "questions_for_mitchell", "trajectory_segments",
    ),
    "mitch_standalone_metrics.json": (
        "clinical_flags", "generated", "hrv_trajectory", "kpis",
    ),
    "multimodal_anomaly_metrics.json": (
        "channels_profiled", "generated_at", "joint_discords", "thresholds",
        "univariate_discords",
    ),
    "omron_bp_report.json": (
        "all_window", "cuff_vs_oura_hr", "evening", "generated_at", "morning",
        "variability",
    ),
    "oura_3d_dashboard_metrics.json": (
        "analysis_modules_loaded", "data_points", "data_range", "generated_at",
        "phase_distribution", "pre_vs_post_treatment", "summary_stats",
    ),
    "oura_full_analysis.json": (
        "canonical_html", "data_end", "data_range", "data_start", "generated_at",
        "hr_daily_mean", "post_days", "rmssd_daily_mean", "rmssd_mean", "sleep_score_mean",
    ),
    "piecewise_regression_metrics.json": (
        "bb_date_sensitivity", "bb_date_sensitivity_interpretation",
        "bb_date_sensitivity_meta", "bb_date_sensitivity_summary", "generated", "metrics",
        "phases",
    ),
    "placebo_calibration_metrics.json": ("false_positive_rates", "results_by_date",),
    "research_synthesis_metrics.json": ("generated", "kpis", "statistical_sources",),
    "rux_forecast.json": ("forecasts", "generated_at", "phase_summary",),
    "sequential_causal_impact_metrics.json": ("run_a", "run_b",),
    "spo2_bos_metrics.json": (
        "bdi", "bos_risk", "data_range", "desaturation", "generated_at", "ruxolitinib",
        "spo2_hr_coupling", "temp_coupling", "trend", "variability",
    ),
    "tau_u_metrics.json": ("comparisons", "generated", "phases",),
    "treatment_response_metrics.json": ("executive", "generated",),
    "weekly_tracker.json": ("doctor_summary", "generated_at", "metrics",),
    "wenche_standalone_metrics.json": (
        "clinical_flags", "generated", "hrv_trajectory", "kpis",
    ),
}


def _json_names(spec: str | list[str] | None) -> list[str]:
    if spec is None:
        return []
    return [spec] if isinstance(spec, str) else list(spec)


def _present_pages() -> list[str]:
    return [name for name in HTML_TO_JSON if (REPORTS_DIR / name).exists()]


class ReportPagesExist(unittest.TestCase):
    """Every page in the registry must have been generated."""

    def test_reports_directory_exists(self) -> None:
        self.assertTrue(
            REPORTS_DIR.is_dir(),
            f"{REPORTS_DIR} does not exist. Run: python run_all.py",
        )

    def test_at_least_one_page_present(self) -> None:
        self.assertTrue(
            _present_pages(),
            "No registry page found in reports/. Run: python run_all.py",
        )

    def test_registry_pages_are_generated(self) -> None:
        missing = [name for name in HTML_TO_JSON if not (REPORTS_DIR / name).exists()]
        self.assertEqual(
            missing, [], f"Registry pages missing from reports/: {missing}"
        )

    def test_nav_targets_are_in_the_registry(self) -> None:
        """A page reachable from the nav must be one the registry knows about."""
        nav_files = {
            entry.get("file", "").split("#", 1)[0]
            for entry in REPORT_REGISTRY
        }
        nav_files = {name for name in nav_files if name.endswith(".html")}
        unknown = sorted(nav_files - set(HTML_TO_JSON) - set(AUDIT_OUTPUT_PAGES))
        self.assertEqual(
            unknown,
            [],
            "Pages in the navigation that the page registry does not cover: "
            f"{unknown}. Add them to HTML_TO_JSON in analysis/statcheck_reports.py.",
        )


class ReportPagesAreComplete(unittest.TestCase):
    """Each generated page must be a finished render, not a stub."""

    def test_pages_are_large_enough(self) -> None:
        too_small = []
        for name in _present_pages():
            floor = MIN_HTML_BYTES_OVERRIDE.get(name, MIN_HTML_BYTES)
            size = (REPORTS_DIR / name).stat().st_size
            if size < floor:
                too_small.append(f"{name} ({size} bytes < {floor})")
        self.assertEqual(too_small, [], f"Pages too small to be complete: {too_small}")

    def test_pages_have_no_runtime_failure_text(self) -> None:
        hits = []
        for name in _present_pages():
            text = (REPORTS_DIR / name).read_text(encoding="utf-8", errors="replace")
            for marker in RUNTIME_FAILURE_STRINGS:
                if marker in text:
                    index = text.index(marker)
                    context = " ".join(
                        text[max(0, index - 60):index + len(marker) + 60].split()
                    )
                    hits.append(f"{name}: {marker!r} in ...{context}...")
        self.assertEqual(
            hits,
            [],
            "Pages report a module that could not run:\n  " + "\n  ".join(hits),
        )

    def test_pages_close_their_html(self) -> None:
        truncated = [
            name for name in _present_pages()
            if "</html>" not in (REPORTS_DIR / name).read_text(
                encoding="utf-8", errors="replace"
            )[-2000:]
        ]
        self.assertEqual(truncated, [], f"Pages without a closing </html>: {truncated}")


class MetricsJsonContract(unittest.TestCase):
    """Each page's metrics JSON must parse and still carry its sections."""

    def test_companion_json_exists(self) -> None:
        missing = []
        for name in _present_pages():
            for json_name in _json_names(HTML_TO_JSON[name]):
                if not (REPORTS_DIR / json_name).exists():
                    missing.append(f"{json_name} (for {name})")
        self.assertEqual(missing, [], f"Metrics JSON missing: {missing}")

    def test_companion_json_parses(self) -> None:
        broken = []
        for name in _present_pages():
            for json_name in _json_names(HTML_TO_JSON[name]):
                path = REPORTS_DIR / json_name
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    broken.append(f"{json_name}: {exc}")
                    continue
                if not isinstance(payload, dict):
                    broken.append(f"{json_name}: top level is {type(payload).__name__}")
        self.assertEqual(broken, [], f"Metrics JSON unreadable: {broken}")

    def test_required_top_level_keys_present(self) -> None:
        missing = []
        for json_name, required in sorted(REQUIRED_JSON_KEYS.items()):
            path = REPORTS_DIR / json_name
            if not path.exists():
                missing.append(f"{json_name}: file missing")
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                missing.append(f"{json_name}: {exc}")
                continue
            absent = [key for key in required if key not in payload]
            if absent:
                missing.append(f"{json_name}: missing {absent}")
        self.assertEqual(missing, [], "Metrics JSON lost keys:\n  " + "\n  ".join(missing))

    def test_every_checked_json_has_a_contract(self) -> None:
        """A new metrics file must declare which keys its page depends on."""
        referenced = {
            json_name
            for spec in HTML_TO_JSON.values()
            for json_name in _json_names(spec)
        }
        undeclared = sorted(referenced - set(REQUIRED_JSON_KEYS))
        self.assertEqual(
            undeclared,
            [],
            f"Metrics files with no key contract in this test: {undeclared}",
        )


class ChronosRanForReal(unittest.TestCase):
    """The foundation-model page must carry a real forecast, not a fallback."""

    def test_chronos_available(self) -> None:
        path = REPORTS_DIR / "foundation_model_metrics.json"
        if not path.exists():
            self.skipTest("foundation_model_metrics.json not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(
            payload.get("chronos_available"),
            "chronos_available is false: the Chronos forecast did not run "
            f"({payload.get('chronos_error')}).",
        )


def _run_all() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    failed = len(result.failures) + len(result.errors)
    print(f"\n{result.testsRun - failed}/{result.testsRun} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
