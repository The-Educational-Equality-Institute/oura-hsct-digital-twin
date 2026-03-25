"""Patient and project configuration.

Copy this file to config.py and update the values for your own data.
All analysis scripts import constants from config.py so patient-specific
details live in one place.

    cp config.example.py config.py
"""
import os
import sys
import sqlite3
from datetime import date
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = PROJECT_ROOT / "data" / "oura.db"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Patient-specific dates (update for your case) ---
TRANSPLANT_DATE = date(2023, 1, 1)       # Major clinical event / baseline anchor
TREATMENT_START = date(2026, 1, 15)      # Intervention start (for causal analysis)
KNOWN_EVENT_DATE = date(2026, 1, 10)     # Known acute episode
HEV_DIAGNOSIS_DATE = None                # Leave as None if not applicable, or date(YYYY, M, D)
DATA_START = date(2025, 12, 1)           # Analysis window start
TREATMENT_START_STR = str(TREATMENT_START)
PATIENT_AGE = 40
PATIENT_TIMEZONE = "Europe/Oslo"

# --- Clinical context (used in report titles and labels) ---
PATIENT_LABEL = "Patient"

# --- Clinical thresholds (referenced across multiple scripts) ---
ESC_RMSSD_DEFICIENCY = 15     # RMSSD < 15 ms = severe autonomic deficiency (Kleiger 1987)
NOCTURNAL_HR_ELEVATED = 80    # Sleeping HR concern threshold (bpm) - above this is abnormal
IST_HR_THRESHOLD = 90         # IST criterion: mean 24-hour HR (HRS/EHRA 2015) - NOT for sleep-only data
POPULATION_RMSSD_MEDIAN = 49  # General population median RMSSD (ms) - Nunan 2010
NORM_RMSSD_P25 = 36           # Population 25th percentile (young adults, Nunan 2010)
NORM_RMSSD_P75 = 72           # Population 75th percentile (young adults, Nunan 2010)
POPULATION_RMSSD_MEAN = 42.0  # Shaffer & Ginsberg 2017, healthy adults
POPULATION_RMSSD_SD = 15.0    # Shaffer & Ginsberg 2017
# NOTE: Median (49) and mean (42) differ because RMSSD is right-skewed in the population.
HSCT_RMSSD_RANGE = (25, 40)   # Estimated RMSSD range for HSCT patients (ms) - clinical estimate
BASELINE_DAYS = 14            # Days before treatment start for baseline window

# --- SpO2 / BOS clinical constants ---
SPO2_NORMAL_MIN = 95.0        # Lower bound of normal adult SpO2 range (%)
SPO2_NORMAL_MAX = 100.0       # Upper bound of normal adult SpO2 range (%)
SPO2_CONCERN_THRESHOLD = 94.0 # Desaturation cutoff (%)
SPO2_CONCERN_SLOPE = -0.02    # Concerning SpO2 slope (%/day) - 1% decline per 50 days

# Breathing Disturbance Index thresholds (events/hour)
BDI_NORMAL = 5.0
BDI_MILD = 15.0
BDI_MODERATE = 30.0

# BOS (Bronchiolitis Obliterans Syndrome) risk score component weights
BOS_WEIGHTS = {
    "spo2_slope": 0.30,
    "spo2_variability": 0.20,
    "desaturation_freq": 0.20,
    "bdi": 0.15,
    "hr_decoupling": 0.15,
}

# DLCO measurements from medical records (date, DLCO%, context)
# Update with your own pulmonary function test results
DLCO_MEASUREMENTS = [
    (date(2024, 1, 1), 80.0, "Baseline"),
    (date(2025, 1, 1), 75.0, "Follow-up"),
]

# --- Visual identity: clinical white/blue palette ---
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# Primary palette (blue family)
C_PRIMARY = "#0056B3"
C_SECONDARY = "#007BFF"
C_MUTED = "#6C757D"
C_LIGHT = "#A7D9F7"
C_DARK = "#003366"
C_ACCENT = "#3399FF"

# Semantic colors
C_CRITICAL = "#DC3545"      # Clinical threshold violations
C_GOOD = "#28A745"          # Within normal range
C_WARNING = "#FFC107"       # Caution

# Layout colors
C_BG = "#F8F9FA"
C_CARD = "#FFFFFF"
C_TEXT = "#343A40"
C_GRID = "#E9ECEF"
C_TEXT_MUTED = "#6C757D"

# Backwards-compatible aliases
C_NEUTRAL = C_MUTED
C_BG_LIGHT = C_BG
C_CAUTION = "#F7B267"

# Biometric-specific colors (clinical monitor standard)
C_HR = "#10B981"            # Heart rate
C_SPO2 = "#06B6D4"         # SpO2
C_HRV = "#8B5CF6"          # HRV/RMSSD
C_SLEEP = "#6366F1"        # Sleep
C_TEMP = "#F97316"          # Temperature

# Period/series colors
C_PRE_TX = "#6C757D"
C_POST_TX = "#0056B3"
C_BASELINE = "#003366"
C_COUNTERFACTUAL = "#A7D9F7"
C_FORECAST = "#3399FF"
C_RUX_LINE = "#0056B3"
C_EFFECT = "#007BFF"

PLOTLY_COLORWAY = ["#0056B3", "#007BFF", "#6C757D", "#A7D9F7", "#003366", "#3399FF"]

PLOTLY_LAYOUT = dict(
    font=dict(family=FONT_FAMILY, size=13, color=C_TEXT),
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=C_CARD,
    plot_bgcolor=C_CARD,
    margin=dict(l=60, r=40, t=60, b=50),
    colorway=PLOTLY_COLORWAY,
    title=dict(font=dict(color=C_PRIMARY, size=16), x=0.05, xanchor="left"),
    xaxis=dict(gridcolor=C_GRID, linecolor=C_GRID, tickfont=dict(color=C_TEXT)),
    yaxis=dict(gridcolor=C_GRID, linecolor=C_GRID, tickfont=dict(color=C_TEXT)),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor=C_GRID, borderwidth=1),
    hoverlabel=dict(bgcolor=C_TEXT, font=dict(color="white")),
)


def validate_config() -> bool:
    """Validate that all required paths and resources exist."""
    ok = True

    db = DATABASE_PATH.resolve() if DATABASE_PATH.is_symlink() else DATABASE_PATH
    if not DATABASE_PATH.exists():
        print(
            f"ERROR: Database not found at {DATABASE_PATH}\n"
            f"  Resolved path: {db}\n"
            f"  Create the database by running: python api/import_oura.py"
        )
        ok = False
    elif db.stat().st_size == 0:
        print(
            f"ERROR: Database file exists but is empty (0 bytes): {db}\n"
            "  This usually means data has not been imported yet.\n"
            "  Fix: python api/import_oura.py --days 90"
        )
        ok = False
    elif not os.access(str(db), os.R_OK):
        print(f"ERROR: Database exists but is not readable: {db}")
        ok = False
    else:
        required_tables = [
            "oura_heart_rate",
            "oura_sleep_periods",
            "oura_readiness",
        ]
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                found = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            missing = [t for t in required_tables if t not in found]
            if missing:
                print(
                    f"ERROR: Database is missing required tables: {', '.join(missing)}\n"
                    "  Fix: re-run importer to populate schema and data:\n"
                    "  python api/import_oura.py --days 90"
                )
                ok = False
        except sqlite3.DatabaseError as e:
            print(
                f"ERROR: Database exists but is not a valid SQLite dataset: {db}\n"
                f"  Details: {type(e).__name__}: {e}"
            )
            ok = False

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not os.access(str(REPORTS_DIR), os.W_OK):
        print(f"ERROR: Reports directory is not writable: {REPORTS_DIR}")
        ok = False

    if ok:
        print(f"Config OK - DB: {db}  Reports: {REPORTS_DIR}")
    return ok
