#!/usr/bin/env python3
"""Glucose-autonomic coupling analysis (CGM × Oura).

Scaffolding for the analyses specified in
reports/cgm_hypotheses_pre_registered.md. Hypothesis-specific tests are
deliberately stubbed until real CGM data is imported after sensor activation
(~2026-04-21). Running this script on an empty glucose_readings table
produces a data-availability report, not fabricated stats.

Data contract:
  - glucose_readings    — created by api/import_glucose.py
  - symptom_events      — created by api/import_symptom.py
  - oura_hrv            — existing, 5-min RMSSD during sleep
  - oura_heart_rate     — existing, 5-min HR samples

Usage:
    python analysis/analyze_glucose_autonomic_coupling.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH, REPORTS_DIR, PATIENT_LABEL

REPORT_ID = "glucose_autonomic_coupling"
REPORT_HTML = REPORTS_DIR / f"{REPORT_ID}.html"
REPORT_JSON = REPORTS_DIR / f"{REPORT_ID}.json"


@dataclass
class DataAvailability:
    """Snapshot of what CGM-trial data is currently in the database."""

    glucose_rows: int
    glucose_start: str | None
    glucose_end: str | None
    symptom_rows: int
    hrv_rows: int
    hr_rows: int

    def trial_ready(self) -> bool:
        """True when enough data is loaded to run any hypothesis test."""
        return self.glucose_rows >= 100 and self.hrv_rows > 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def load_availability(conn: sqlite3.Connection) -> DataAvailability:
    """Summarize what's in the database without assuming any table exists yet."""
    if _table_exists(conn, "glucose_readings"):
        row = conn.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM glucose_readings"
        ).fetchone()
        glucose_rows, glucose_start, glucose_end = row
    else:
        glucose_rows, glucose_start, glucose_end = 0, None, None

    symptom_rows = (
        conn.execute("SELECT COUNT(*) FROM symptom_events").fetchone()[0]
        if _table_exists(conn, "symptom_events")
        else 0
    )
    hrv_rows = (
        conn.execute("SELECT COUNT(*) FROM oura_hrv").fetchone()[0]
        if _table_exists(conn, "oura_hrv")
        else 0
    )
    hr_rows = (
        conn.execute("SELECT COUNT(*) FROM oura_heart_rate").fetchone()[0]
        if _table_exists(conn, "oura_heart_rate")
        else 0
    )

    return DataAvailability(
        glucose_rows=glucose_rows,
        glucose_start=glucose_start,
        glucose_end=glucose_end,
        symptom_rows=symptom_rows,
        hrv_rows=hrv_rows,
        hr_rows=hr_rows,
    )


def load_glucose(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load glucose readings as a timestamp-indexed DataFrame."""
    if not _table_exists(conn, "glucose_readings"):
        return pd.DataFrame(columns=["timestamp", "glucose_mmol_l", "record_type"])
    df = pd.read_sql(
        "SELECT timestamp, glucose_mmol_l, record_type, sensor_serial "
        "FROM glucose_readings ORDER BY timestamp",
        conn,
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_symptom_events(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load symptom events as a timestamp-indexed DataFrame."""
    if not _table_exists(conn, "symptom_events"):
        return pd.DataFrame(
            columns=["timestamp", "symptom_type", "severity", "context", "note"]
        )
    df = pd.read_sql(
        "SELECT timestamp, symptom_type, severity, context, note "
        "FROM symptom_events ORDER BY timestamp",
        conn,
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Descriptive summary (runnable the moment any glucose data arrives)
# ---------------------------------------------------------------------------
def summarize_glucose(df: pd.DataFrame) -> dict:
    """Basic descriptive stats. Returns an empty dict when df is empty."""
    if df.empty:
        return {}
    g = df["glucose_mmol_l"]
    return {
        "n": int(len(g)),
        "mean_mmol_l": float(g.mean()),
        "median_mmol_l": float(g.median()),
        "min_mmol_l": float(g.min()),
        "max_mmol_l": float(g.max()),
        "cv_percent": float(100 * g.std() / g.mean()) if g.mean() > 0 else None,
        "time_in_range_3_9_10_0_pct": float(
            100 * ((g >= 3.9) & (g <= 10.0)).mean()
        ),
        "time_above_10_0_pct": float(100 * (g > 10.0).mean()),
        "time_below_3_9_pct": float(100 * (g < 3.9).mean()),
    }


# ---------------------------------------------------------------------------
# Hypothesis tests — stubbed until real data exists
# Each function maps to a hypothesis in reports/cgm_hypotheses_pre_registered.md
# ---------------------------------------------------------------------------
def test_h1_postprandial_hyperglycemia(
    glucose: pd.DataFrame,
    symptoms: pd.DataFrame,
) -> dict:
    """H1: ≥1 postprandial spike ≥10.0 mmol/L within 120 min after a meal timestamp.

    Requires symptom_events rows with context='postprandial' or a dedicated
    meal_events log. Not implemented until trial data arrives.
    """
    raise NotImplementedError("H1 test pending trial data (~2026-04-21)")


def test_h2_glycemic_autonomic_coupling(
    glucose: pd.DataFrame,
    hrv: pd.DataFrame,
) -> dict:
    """H2: Rapid glucose excursions followed by transient HRV depression.

    Excursion = Δglucose ≥ 2.0 mmol/L within 30 min.
    Outcome = ΔRMSSD (post 60min vs pre 60min baseline), Wilcoxon signed-rank.
    Not implemented until trial data arrives.
    """
    raise NotImplementedError("H2 test pending trial data (~2026-04-21)")


def test_h3_chest_pain_coincidence(
    glucose: pd.DataFrame,
    symptoms: pd.DataFrame,
) -> dict:
    """H3: Chest-pain events cluster after glucose readings ≥ 9.0 mmol/L.

    Binomial test of observed vs baseline rate of post-hyperglycemic windows.
    Not implemented until trial data arrives.
    """
    raise NotImplementedError("H3 test pending trial data (~2026-04-21)")


def test_h4_variability_autonomic_correlation(
    glucose: pd.DataFrame,
    hrv: pd.DataFrame,
) -> dict:
    """H4: Daily glucose CV correlates negatively with nightly RMSSD (Spearman).

    Not implemented until trial data arrives.
    """
    raise NotImplementedError("H4 test pending trial data (~2026-04-21)")


# ---------------------------------------------------------------------------
# Report writing (minimal until data exists)
# ---------------------------------------------------------------------------
def _render_availability_html(
    availability: DataAvailability,
    summary: dict,
) -> str:
    ready = availability.trial_ready()
    status_color = "#28A745" if ready else "#6C757D"
    status_label = "Trial data loaded" if ready else "Awaiting trial data"

    summary_rows = (
        "".join(
            f"<tr><td>{k.replace('_', ' ')}</td><td>{v:.2f}</td></tr>"
            for k, v in summary.items()
            if isinstance(v, (int, float))
        )
        if summary
        else '<tr><td colspan="2"><em>No glucose readings imported yet.</em></td></tr>'
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Glucose-Autonomic Coupling — {PATIENT_LABEL}</title>
<style>
  body {{ font-family: Inter, system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; color: #343A40; }}
  h1 {{ color: #0056B3; }}
  .status {{ padding: 0.5em 1em; border-left: 4px solid {status_color}; background: #F8F9FA; margin: 1em 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  td, th {{ padding: 0.5em; border-bottom: 1px solid #E9ECEF; text-align: left; }}
  .muted {{ color: #6C757D; font-size: 0.9em; }}
</style>
</head><body>
<h1>Glucose–Autonomic Coupling</h1>
<p class="muted">Patient: {PATIENT_LABEL} · Generated {datetime.now().isoformat(timespec="seconds")}</p>

<div class="status"><strong>{status_label}.</strong></div>

<h2>Data availability</h2>
<table>
  <tr><td>glucose_readings rows</td><td>{availability.glucose_rows}</td></tr>
  <tr><td>glucose range</td><td>{availability.glucose_start or "—"} → {availability.glucose_end or "—"}</td></tr>
  <tr><td>symptom_events rows</td><td>{availability.symptom_rows}</td></tr>
  <tr><td>oura_hrv rows</td><td>{availability.hrv_rows}</td></tr>
  <tr><td>oura_heart_rate rows</td><td>{availability.hr_rows}</td></tr>
</table>

<h2>Glucose summary</h2>
<table>{summary_rows}</table>

<h2>Hypotheses (pre-registered)</h2>
<p>See <a href="cgm_hypotheses_pre_registered.md">cgm_hypotheses_pre_registered.md</a>
for the analysis plan. Hypothesis tests are stubbed in this script until
the trial produces data.</p>
<ul>
  <li>H1 — Postprandial hyperglycemia (&ge; 10.0 mmol/L within 120 min of meal)</li>
  <li>H2 — Glycemic excursions followed by HRV depression (paired Wilcoxon)</li>
  <li>H3 — Chest-pain events cluster in post-hyperglycemic windows (binomial)</li>
  <li>H4 — Daily glucose CV correlates negatively with nightly RMSSD (Spearman)</li>
</ul>

<p class="muted">This report re-computes from the current database on every run.
Not validated for clinical decision-making. Not a medical device.</p>
</body></html>
"""


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    try:
        availability = load_availability(conn)
        glucose = load_glucose(conn)
    finally:
        conn.close()

    summary = summarize_glucose(glucose)

    REPORT_HTML.write_text(
        _render_availability_html(availability, summary), encoding="utf-8"
    )
    REPORT_JSON.write_text(
        json.dumps(
            {
                "patient": PATIENT_LABEL,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "availability": availability.__dict__,
                "glucose_summary": summary,
                "hypothesis_tests": {
                    "h1_postprandial_hyperglycemia": "pending_data",
                    "h2_glycemic_autonomic_coupling": "pending_data",
                    "h3_chest_pain_coincidence": "pending_data",
                    "h4_variability_autonomic_correlation": "pending_data",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {REPORT_HTML}")
    print(f"Wrote {REPORT_JSON}")
    print(
        f"  glucose_readings={availability.glucose_rows}  "
        f"symptom_events={availability.symptom_rows}  "
        f"oura_hrv={availability.hrv_rows}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
