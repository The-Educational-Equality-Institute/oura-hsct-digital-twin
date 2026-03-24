"""Pandera DataFrame validation schemas for Oura analysis pipeline.

These schemas validate data at system boundaries — after SQL load, before
analysis. If the daily pipeline produces an impossible value (negative SpO2,
HR of 500, p-value of 2.0), it crashes loudly instead of publishing garbage.

Usage:
    from data_schemas import validate_daily_metrics, validate_stats

    df = load_data()
    df = validate_daily_metrics(df)  # raises SchemaError on bad data
"""
import sys
from pathlib import Path

import pandera as pa
from pandera import Column, Check, DataFrameSchema

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ESC_RMSSD_DEFICIENCY  # noqa: E402

# ---------------------------------------------------------------------------
# HRV data (oura_hrv table)
# ---------------------------------------------------------------------------
hrv_schema = DataFrameSchema(
    {
        "rmssd": Column(
            float,
            Check.in_range(0, 300),
            nullable=True,
            description="Root mean square of successive differences (ms)",
        ),
    },
    coerce=True,
)

# ---------------------------------------------------------------------------
# Heart rate data (oura_heart_rate / oura_sleep_periods)
# ---------------------------------------------------------------------------
heart_rate_schema = DataFrameSchema(
    {
        "heart_rate": Column(
            float,
            Check.in_range(25, 250),
            nullable=True,
            description="Heart rate (bpm) — 25 to 250 covers bradycardia to SVT",
        ),
    },
    coerce=True,
)

# ---------------------------------------------------------------------------
# SpO2 data (oura_spo2 table)
# ---------------------------------------------------------------------------
spo2_schema = DataFrameSchema(
    {
        "spo2_average": Column(
            float,
            Check.in_range(70, 100),
            nullable=True,
            description="Blood oxygen saturation (%) — <70 is not physiologically plausible from Oura",
        ),
    },
    coerce=True,
)

# ---------------------------------------------------------------------------
# Sleep data (oura_sleep_periods table)
# ---------------------------------------------------------------------------
sleep_schema = DataFrameSchema(
    {
        "total_sleep_duration": Column(
            float,
            Check.in_range(0, 86400),
            nullable=True,
            description="Total sleep duration (seconds) — max 24h",
        ),
        "efficiency": Column(
            float,
            Check.in_range(0, 100),
            nullable=True,
            description="Sleep efficiency (%)",
        ),
    },
    coerce=True,
)

# ---------------------------------------------------------------------------
# Temperature deviation (oura_readiness contributors or sleep periods)
# ---------------------------------------------------------------------------
temperature_schema = DataFrameSchema(
    {
        "temperature_deviation": Column(
            float,
            Check.in_range(-5.0, 5.0),
            nullable=True,
            description="Skin temperature deviation from baseline (deg C)",
        ),
    },
    coerce=True,
)

# ---------------------------------------------------------------------------
# Statistical results — for validating computed p-values and effect sizes
# ---------------------------------------------------------------------------
stats_result_schema = DataFrameSchema(
    {
        "p_value": Column(
            float,
            Check.in_range(0, 1),
            nullable=False,
            description="P-value must be between 0 and 1",
        ),
        "effect_size": Column(
            float,
            Check.in_range(-50, 50),
            nullable=True,
            description="Effect size (Cohen's d or similar)",
        ),
    },
    coerce=True,
)


# ---------------------------------------------------------------------------
# Convenience validators — call these after loading data
# ---------------------------------------------------------------------------
def validate_hrv(df):
    """Validate HRV DataFrame. Returns the validated (coerced) DataFrame."""
    if "rmssd" in df.columns:
        return hrv_schema.validate(df, lazy=True)
    return df


def validate_heart_rate(df):
    """Validate heart rate DataFrame."""
    if "heart_rate" in df.columns:
        return heart_rate_schema.validate(df, lazy=True)
    return df


def validate_spo2(df):
    """Validate SpO2 DataFrame. Filters sentinel values first."""
    if "spo2_average" in df.columns:
        df = df[df["spo2_average"] > 0]  # Remove Oura's 0.0 sentinel
        return spo2_schema.validate(df, lazy=True)
    return df


def validate_sleep(df):
    """Validate sleep DataFrame."""
    cols = {"total_sleep_duration", "efficiency"}
    if cols & set(df.columns):
        return sleep_schema.validate(df, lazy=True)
    return df
