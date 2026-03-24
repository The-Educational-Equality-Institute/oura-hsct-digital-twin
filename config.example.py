"""Patient and project configuration.

Copy this file to config.py and update the values for your own data.
All analysis scripts import constants from config.py so patient-specific
details live in one place.

    cp config.example.py config.py
"""
import os
import sys
from datetime import date
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = PROJECT_ROOT / "data" / "oura.db"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Patient-specific dates (update for your case) ---
TRANSPLANT_DATE = date(2023, 1, 1)        # Major clinical event / baseline anchor
TREATMENT_START = date(2026, 1, 15)       # Intervention start (e.g. new medication, for causal analysis)
KNOWN_EVENT_DATE = date(2026, 1, 10)      # Known acute episode (e.g. ER visit, infection)
HEV_DIAGNOSIS_DATE = date(2026, 2, 1)     # Secondary diagnosis date (leave as-is if not applicable)
DATA_START = date(2025, 12, 1)            # Analysis window start (first day of usable Oura data)
TREATMENT_START_STR = str(TREATMENT_START)
PATIENT_AGE = 40                          # Patient age in years (used in population-norm comparisons)
PATIENT_TIMEZONE = "Europe/Oslo"          # IANA timezone for the patient's location

# --- Clinical context (used in report titles and labels) ---
PATIENT_LABEL = "Post-HSCT Patient"       # Short label shown in report headers

# --- Clinical thresholds (referenced across multiple scripts) ---
ESC_RMSSD_DEFICIENCY = 15     # Clinical concern threshold (ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017): RMSSD < 15 ms = severe autonomic deficiency

# Heart rate thresholds
# IST diagnostic criteria (HRS/EHRA 2015): mean 24-hour HR > 90 bpm OR resting awake HR > 100 bpm.
# For nocturnal wearable data, 90 bpm is the WRONG threshold — normal sleeping HR for a 36M is 40-70 bpm.
# We use 80 bpm as the nocturnal concern threshold (sleeping HR > 80 warrants medical evaluation).
NOCTURNAL_HR_ELEVATED = 80    # Nocturnal HR concern threshold (bpm) — sleeping HR above this is abnormal
IST_HR_THRESHOLD = 90         # IST criterion: mean 24-hour HR (HRS/EHRA 2015) — NOT for sleep-only data

# RMSSD population norms (Nunan 2010 meta-analysis, short-term clinical 5-min ECG)
# NOTE: Oura PPG-derived RMSSD may differ ~7-10% from clinical ECG (MAPE 6.84%, CCC 0.91).
# Nocturnal values trend higher than resting 5-min recordings due to parasympathetic dominance.
POPULATION_RMSSD_MEDIAN = 49  # General population median RMSSD (ms) — Nunan 2010
POPULATION_RMSSD_MEAN = 42.0  # Shaffer & Ginsberg 2017, healthy adults
POPULATION_RMSSD_SD = 15.0    # Shaffer & Ginsberg 2017
# NOTE: Median (49) and mean (42) differ because RMSSD is right-skewed in the population.
NORM_RMSSD_P25 = 36           # Population 25th percentile (young adults IQR: 36-79 ms, Nunan 2010)
NORM_RMSSD_P75 = 72           # Population 75th percentile (young adults IQR: 36-79 ms, Nunan 2010)
HSCT_RMSSD_RANGE = (25, 40)   # Estimated RMSSD range for HSCT patients (ms) — no validated literature; clinical estimate
BASELINE_DAYS = 14            # Days before treatment start for baseline window

# --- Visual identity ---
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def validate_config() -> bool:
    """Validate that all required paths and resources exist.

    Returns True if everything is OK. Prints clear diagnostics and
    returns False (or exits) when something is wrong.
    """
    ok = True

    # 1. Database must exist and be readable
    db = DATABASE_PATH.resolve() if DATABASE_PATH.is_symlink() else DATABASE_PATH
    if not DATABASE_PATH.exists():
        print(
            f"ERROR: Database not found at {DATABASE_PATH}\n"
            f"  Resolved path: {db}\n"
            f"  Expected a symlink in data/ pointing to the main database.\n"
            f"  Fix: copy or symlink your SQLite database to "
            f"{PROJECT_ROOT / 'data' / 'oura.db'}"
        )
        ok = False
    elif not os.access(str(db), os.R_OK):
        print(f"ERROR: Database exists but is not readable: {db}")
        ok = False

    # 2. Create reports dir if missing
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Verify reports dir is writable
    if not os.access(str(REPORTS_DIR), os.W_OK):
        print(f"ERROR: Reports directory is not writable: {REPORTS_DIR}")
        ok = False

    if ok:
        print(f"Config OK — DB: {db}  Reports: {REPORTS_DIR}")
    return ok
