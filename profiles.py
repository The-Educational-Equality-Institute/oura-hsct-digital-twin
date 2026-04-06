"""Patient profiles for multi-user Oura data collection.

Henrik is the primary patient — config.py holds his clinical parameters.
Secondary profiles store their own clinical context and use separate databases
so the analysis pipeline can run independently for each patient.

Usage:
    python api/import_oura.py --days 30                  # Henrik (default)
    python api/import_oura.py --days 30 --profile mitch  # Mitch
"""
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

PROFILES = {
    "henrik": {
        "label": "Post-HSCT Patient",
        "age": 36,
        "condition": "ALL at 14, allogeneic HSCT 2023-11-23",
        "transplant_date": date(2023, 11, 23),
        "treatment_start": date(2026, 3, 16),
        "treatment": "Ruxolitinib 10mg BID",
        "database": PROJECT_ROOT / "data" / "demo.db",
        "major_event_date": date(2023, 11, 23),
        "major_event_label": "HSCT",
        # Uses default env vars: OURA_ACCESS_TOKEN, OURA_REFRESH_TOKEN
        "token_env": "OURA_ACCESS_TOKEN",
        "refresh_env": "OURA_REFRESH_TOKEN",
        "ring_gen": 4,
    },
    "mitch": {
        "label": "Post-Stroke Patient",
        "age": 33,
        "condition": "Stroke at 33, bilateral carotid/vertebral artery dissection (left main)",
        "database": PROJECT_ROOT / "data" / "mitch.db",
        "major_event_date": date(2024, 12, 15),
        "major_event_label": "Stroke",
        # Mitch's tokens use prefixed env vars
        "token_env": "MITCH_OURA_ACCESS_TOKEN",
        "refresh_env": "MITCH_OURA_REFRESH_TOKEN",
        "ring_gen": 3,
    },
}
