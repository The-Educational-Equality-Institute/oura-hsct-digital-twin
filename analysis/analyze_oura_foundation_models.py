#!/usr/bin/env python3
"""
Foundation Model Time Series Analysis for Oura Ring Biometric Data

Applies Amazon Chronos-2 (pretrained time series foundation model) and a
statistical baseline (ARIMA / Prophet) to post-HSCT biometrics for:
  1. Probabilistic forecasting with uncertainty quantification
  2. Anomaly detection via prediction residuals
  3. Ensemble consensus scoring (foundation model + statistical baseline)
  4. Retrospective validation against known acute event
  5. Pre vs post treatment regime change detection

See config.py for patient details, event dates, and treatment dates.

Output:
  - Interactive HTML report: reports/foundation_model_report.html
  - JSON metrics: reports/foundation_model_metrics.json

Usage:
    python analysis/analyze_oura_foundation_models.py
"""

from __future__ import annotations

import io
import gc
import enum
import json
import shutil
import os
import sqlite3
import sys
import time
import traceback
import types
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATABASE_PATH, REPORTS_DIR, KNOWN_EVENT_DATE, TREATMENT_START,
    TREATMENT_START_STR,
    FONT_FAMILY,
)
from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    COLORWAY, STATUS_COLORS, BG_PRIMARY, BG_SURFACE, BG_ELEVATED,
    BORDER_SUBTLE, TEXT_PRIMARY, TEXT_SECONDARY,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    ACCENT_PURPLE, ACCENT_CYAN,
    C_PRE_TX, C_POST_TX, C_FORECAST, C_RUX_LINE,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "foundation_model_report.html"
JSON_OUTPUT = REPORTS_DIR / "foundation_model_metrics.json"

# ---------------------------------------------------------------------------
# Clinical context
# ---------------------------------------------------------------------------
RUXOLITINIB_START = TREATMENT_START_STR  # string form for date comparisons
KNOWN_EVENT_DATE_STR = str(KNOWN_EVENT_DATE)  # string form for date comparisons
FORECAST_HORIZON = 14  # days
CONTEXT_LENGTH = 55  # nights for training context

# Chronos model configuration
CHRONOS_MODEL = "amazon/chronos-bolt-base"  # Bolt variant (faster, native quantiles)
CHRONOS_FALLBACK = "amazon/chronos-t5-base"  # Original T5 fallback
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Visualization — from theme, with dark-theme-aware band colors
COLOR_PRE = C_PRE_TX
COLOR_POST = C_POST_TX
COLOR_RUX_LINE = C_RUX_LINE
COLOR_FORECAST = C_FORECAST
COLOR_ANOMALY = ACCENT_RED
COLOR_BAND_OUTER = "rgba(59, 130, 246, 0.12)"
COLOR_BAND_INNER = "rgba(59, 130, 246, 0.25)"


def _install_torchvision_compat_stub() -> None:
    """Mask a broken torchvision install during Chronos imports.

    This environment has torchvision metadata present, but importing it fails
    before Chronos can load because the installed binary is incompatible with
    the current torch operator registry. Chronos itself does not need image or
    video ops, so a minimal stub is sufficient for transformers import-time
    checks in this script.
    """
    if "torchvision" in sys.modules:
        return

    vision = types.ModuleType("torchvision")
    vision.__spec__ = ModuleSpec("torchvision", loader=None)
    vision.__path__ = []

    transforms = types.ModuleType("torchvision.transforms")
    transforms.__spec__ = ModuleSpec("torchvision.transforms", loader=None)

    io = types.ModuleType("torchvision.io")
    io.__spec__ = ModuleSpec("torchvision.io", loader=None)

    class InterpolationMode(enum.Enum):
        NEAREST = 0
        NEAREST_EXACT = 1
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 6

    transforms.InterpolationMode = InterpolationMode
    vision.transforms = transforms
    vision.io = io

    sys.modules["torchvision"] = vision
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.io"] = io


def _resolve_local_hf_snapshot(repo_id: str) -> Optional[Path]:
    """Return a complete local HF snapshot path if it already exists."""
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir = hub_root / f"models--{repo_id.replace('/', '--')}"
    if not cache_dir.exists():
        return None

    refs_main = cache_dir / "refs" / "main"
    candidates: list[Path] = []
    if refs_main.exists():
        ref = refs_main.read_text(encoding="utf-8").strip()
        if ref:
            candidates.append(cache_dir / "snapshots" / ref)

    snapshots_dir = cache_dir / "snapshots"
    if snapshots_dir.exists():
        candidates.extend(sorted(p for p in snapshots_dir.iterdir() if p.is_dir()))

    seen: set[Path] = set()
    for snapshot in candidates:
        if snapshot in seen:
            continue
        seen.add(snapshot)
        if (snapshot / "config.json").exists() and (snapshot / "model.safetensors").exists():
            return snapshot

    return None


def _import_base_chronos_pipeline():
    """Import Chronos while filtering one known non-fatal torchao warning."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            from chronos import BaseChronosPipeline
    finally:
        noisy_fragment = "Skipping import of cpp extensions due to incompatible torch version"
        for line in stdout_buffer.getvalue().splitlines():
            if noisy_fragment not in line:
                print(line)
        for line in stderr_buffer.getvalue().splitlines():
            if noisy_fragment not in line:
                print(line, file=sys.stderr)

    return BaseChronosPipeline


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_data() -> dict[str, pd.DataFrame]:
    """Load all Oura tables into DataFrames."""
    print("[DATA] Loading biometric data from database...")
    db_path = Path(DATABASE_PATH).resolve()
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}. Run: python api/import_oura.py --days 90", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # HRV epochs (5-min intervals)
        hrv = pd.read_sql_query(
            "SELECT timestamp, rmssd FROM oura_hrv ORDER BY timestamp", conn
        )
        hrv["datetime"] = pd.to_datetime(hrv["timestamp"], utc=True)
        hrv["date"] = hrv["datetime"].dt.date.astype(str)
        hrv["rmssd"] = pd.to_numeric(hrv["rmssd"], errors="coerce")

        # Heart rate (continuous)
        hr = pd.read_sql_query(
            "SELECT timestamp, bpm FROM oura_heart_rate ORDER BY timestamp", conn
        )
        hr["datetime"] = pd.to_datetime(hr["timestamp"], utc=True)
        hr["date"] = hr["datetime"].dt.date.astype(str)
        hr["bpm"] = pd.to_numeric(hr["bpm"], errors="coerce")

        # Sleep periods (per-night aggregates)
        sleep_periods = pd.read_sql_query(
            """SELECT day as date, average_hrv, average_heart_rate, average_breath,
                      total_sleep_duration, rem_sleep_duration, deep_sleep_duration,
                      light_sleep_duration, awake_time, efficiency, lowest_heart_rate
               FROM oura_sleep_periods
               WHERE type = 'long_sleep'
               ORDER BY day""",
            conn,
        )
        for col in sleep_periods.columns:
            if col != "date":
                sleep_periods[col] = pd.to_numeric(sleep_periods[col], errors="coerce")

        # SpO2
        spo2 = pd.read_sql_query(
            "SELECT date, spo2_average FROM oura_spo2 WHERE spo2_average > 0 ORDER BY date",
            conn,
        )
        spo2["spo2_average"] = pd.to_numeric(spo2["spo2_average"], errors="coerce")

        # Readiness (temperature deviation)
        readiness = pd.read_sql_query(
            "SELECT date, temperature_deviation FROM oura_readiness ORDER BY date",
            conn,
        )
        readiness["temperature_deviation"] = pd.to_numeric(
            readiness["temperature_deviation"], errors="coerce"
        )
    finally:
        conn.close()

    data = {
        "hrv": hrv,
        "hr": hr,
        "sleep_periods": sleep_periods,
        "spo2": spo2,
        "readiness": readiness,
    }

    for name, df in data.items():
        print(f"  {name}: {len(df)} rows, date range: "
              f"{df['date'].min()} to {df['date'].max()}")

    return data


def build_nightly_series(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build per-night time series from HRV epochs and sleep periods."""
    print("[DATA] Building nightly aggregated series...")

    # Aggregate HRV by night
    hrv_daily = (
        data["hrv"]
        .groupby("date")
        .agg(
            nightly_rmssd_mean=("rmssd", "mean"),
            nightly_rmssd_std=("rmssd", "std"),
            nightly_rmssd_min=("rmssd", "min"),
            nightly_rmssd_max=("rmssd", "max"),
            nightly_rmssd_median=("rmssd", "median"),
            hrv_epoch_count=("rmssd", "count"),
        )
        .reset_index()
    )

    # Aggregate HR by night
    hr_daily = (
        data["hr"]
        .groupby("date")
        .agg(
            nightly_hr_mean=("bpm", "mean"),
            nightly_hr_std=("bpm", "std"),
            nightly_hr_min=("bpm", "min"),
            nightly_hr_max=("bpm", "max"),
        )
        .reset_index()
    )

    # Sleep periods - already per-night
    sp = data["sleep_periods"].copy()
    sp.rename(columns={
        "average_hrv": "sleep_avg_hrv",
        "average_heart_rate": "sleep_avg_hr",
        "average_breath": "sleep_avg_breath",
    }, inplace=True)
    sp["total_hours"] = sp["total_sleep_duration"] / 3600
    sp["rem_pct"] = (
        sp["rem_sleep_duration"]
        / sp["total_sleep_duration"].replace(0, np.nan)
        * 100
    )

    # Merge all nightly data
    nightly = hrv_daily.copy()
    nightly = nightly.merge(hr_daily, on="date", how="outer")
    nightly = nightly.merge(
        sp[["date", "sleep_avg_hrv", "sleep_avg_hr", "sleep_avg_breath",
            "efficiency", "lowest_heart_rate", "total_hours", "rem_pct"]],
        on="date", how="outer",
    )
    nightly = nightly.merge(data["spo2"], on="date", how="left")
    nightly = nightly.merge(data["readiness"], on="date", how="left")

    nightly.sort_values("date", inplace=True)
    nightly.reset_index(drop=True, inplace=True)

    print(f"  Nightly series: {len(nightly)} nights, "
          f"{nightly['date'].min()} to {nightly['date'].max()}")

    return nightly


def build_hourly_hr(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Downsample continuous HR to hourly means."""
    print("[DATA] Building hourly HR series...")
    hr = data["hr"].copy()
    hr.set_index("datetime", inplace=True)
    hourly = hr["bpm"].resample("1h").agg(["mean", "std", "count"])
    hourly.columns = ["hr_mean", "hr_std", "hr_count"]
    # Drop hours with too few readings
    hourly = hourly[hourly["hr_count"] >= 3].copy()
    hourly.reset_index(inplace=True)
    print(f"  Hourly HR: {len(hourly)} hours")
    return hourly


# ===========================================================================
# CHRONOS FOUNDATION MODEL
# ===========================================================================

def load_chronos_pipeline():
    """Load the Chronos-2 foundation model pipeline."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    _install_torchvision_compat_stub()

    import torch
    BaseChronosPipeline = _import_base_chronos_pipeline()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CHRONOS] Loading model on {device}...")

    local_primary = _resolve_local_hf_snapshot(CHRONOS_MODEL)
    if local_primary is not None:
        print(f"[CHRONOS] Using local cache: {local_primary}")
        pipeline = BaseChronosPipeline.from_pretrained(
            str(local_primary),
            device_map=device,
            dtype=torch.float32,
        )
        print(f"[CHRONOS] Loaded local snapshot for {CHRONOS_MODEL}")
        print(f"  Context length: {pipeline.model_context_length}")
        if pipeline.model_prediction_length:
            print(f"  Max prediction length: {pipeline.model_prediction_length}")
        return pipeline

    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            CHRONOS_MODEL,
            device_map=device,
            dtype=torch.float32,
        )
        print(f"[CHRONOS] Loaded {CHRONOS_MODEL}")
    except Exception as e:
        print(f"[CHRONOS] Failed to load {CHRONOS_MODEL}: {e}")
        local_fallback = _resolve_local_hf_snapshot(CHRONOS_FALLBACK)
        if local_fallback is not None:
            print(f"[CHRONOS] Falling back to local cache: {local_fallback}")
            pipeline = BaseChronosPipeline.from_pretrained(
                str(local_fallback),
                device_map=device,
                dtype=torch.float32,
            )
            print(f"[CHRONOS] Loaded local snapshot for {CHRONOS_FALLBACK}")
        else:
            print(f"[CHRONOS] Falling back to {CHRONOS_FALLBACK}...")
            pipeline = BaseChronosPipeline.from_pretrained(
                CHRONOS_FALLBACK,
                device_map=device,
                dtype=torch.float32,
            )
            print(f"[CHRONOS] Loaded {CHRONOS_FALLBACK}")

    print(f"  Context length: {pipeline.model_context_length}")
    if pipeline.model_prediction_length:
        print(f"  Max prediction length: {pipeline.model_prediction_length}")

    return pipeline


def chronos_forecast_with_quantiles(
    pipeline,
    context: np.ndarray,
    prediction_length: int,
    quantile_levels: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Run Chronos forecast and return quantile predictions.

    Returns dict with keys: 'median', 'q10', 'q25', 'q75', 'q90', 'mean'.
    """
    import torch

    if quantile_levels is None:
        quantile_levels = QUANTILE_LEVELS

    ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)  # (1, T)

    # predict_quantiles returns (quantiles, mean) tensors
    quantiles_tensor, mean_tensor = pipeline.predict_quantiles(
        ctx,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )

    # quantiles_tensor shape: (1, prediction_length, len(quantile_levels))
    # mean_tensor shape: (1, prediction_length)
    q = quantiles_tensor[0].cpu().numpy()  # (prediction_length, n_quantiles)
    m = mean_tensor[0].cpu().numpy()  # (prediction_length,)

    result = {"mean": m}
    qmap = {0.1: "q10", 0.25: "q25", 0.5: "median", 0.75: "q75", 0.9: "q90"}
    for i, ql in enumerate(quantile_levels):
        key = qmap.get(ql, f"q{int(ql*100)}")
        result[key] = q[:, i]

    return result


# ===========================================================================
# ANALYSIS 1: CHRONOS ON NIGHTLY RMSSD AND HR
# ===========================================================================

def run_chronos_nightly(
    pipeline,
    nightly: pd.DataFrame,
    metrics: dict,
) -> dict[str, Any]:
    """Apply Chronos to nightly RMSSD and HR series.

    Uses first CONTEXT_LENGTH nights as context, forecasts FORECAST_HORIZON.
    """
    import torch

    print("\n" + "=" * 70)
    print("[CHRONOS] Analysis 1: Nightly RMSSD and HR Forecasting")
    print("=" * 70)

    results = {}

    for series_name, col in [("rmssd", "nightly_rmssd_mean"), ("hr", "nightly_hr_mean")]:
        print(f"\n[CHRONOS] Forecasting nightly {series_name.upper()}...")

        # Extract series, drop NaN
        s = nightly[["date", col]].dropna().copy()
        s.reset_index(drop=True, inplace=True)
        n = len(s)

        if n < CONTEXT_LENGTH + 5:
            print(f"  WARNING: Only {n} valid nights, need {CONTEXT_LENGTH + 5}. Skipping.")
            continue

        context_end = min(CONTEXT_LENGTH, n - FORECAST_HORIZON)
        actual_horizon = min(FORECAST_HORIZON, n - context_end)

        context = s[col].values[:context_end]
        actual = s[col].values[context_end:context_end + actual_horizon]
        context_dates = s["date"].values[:context_end]
        forecast_dates = s["date"].values[context_end:context_end + actual_horizon]

        print(f"  Context: {context_end} nights ({context_dates[0]} to {context_dates[-1]})")
        print(f"  Forecast: {actual_horizon} nights ({forecast_dates[0]} to {forecast_dates[-1]})")

        t0 = time.time()
        forecast = chronos_forecast_with_quantiles(
            pipeline, context, actual_horizon
        )
        elapsed = time.time() - t0
        print(f"  Inference time: {elapsed:.2f}s")

        # Compute metrics
        median_forecast = forecast["median"]
        mae = np.mean(np.abs(actual - median_forecast))
        rmse = np.sqrt(np.mean((actual - median_forecast) ** 2))
        mape = np.mean(np.abs((actual - median_forecast) / np.where(actual == 0, 1, actual))) * 100

        # Coverage: % of actuals within 90% PI
        in_90pi = np.sum(
            (actual >= forecast["q10"]) & (actual <= forecast["q90"])
        ) / len(actual) * 100
        # 50% PI coverage
        in_50pi = np.sum(
            (actual >= forecast["q25"]) & (actual <= forecast["q75"])
        ) / len(actual) * 100

        # Anomaly detection: points outside 90% PI
        outside_90pi = (actual < forecast["q10"]) | (actual > forecast["q90"])
        anomaly_dates = forecast_dates[outside_90pi]
        anomaly_residuals = actual[outside_90pi] - median_forecast[outside_90pi]

        # Width of prediction interval
        pi_width_90 = np.mean(forecast["q90"] - forecast["q10"])
        pi_width_50 = np.mean(forecast["q75"] - forecast["q25"])

        series_metrics = {
            "context_nights": int(context_end),
            "forecast_nights": int(actual_horizon),
            "context_start": str(context_dates[0]),
            "context_end": str(context_dates[-1]),
            "forecast_start": str(forecast_dates[0]),
            "forecast_end": str(forecast_dates[-1]),
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "mape": round(float(mape), 2),
            "coverage_90pi": round(float(in_90pi), 1),
            "coverage_50pi": round(float(in_50pi), 1),
            "pi_width_90_mean": round(float(pi_width_90), 2),
            "pi_width_50_mean": round(float(pi_width_50), 2),
            "n_anomalies_outside_90pi": int(np.sum(outside_90pi)),
            "anomaly_dates": [str(d) for d in anomaly_dates],
            "inference_time_s": round(elapsed, 2),
        }

        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")
        print(f"  90% PI coverage: {in_90pi:.0f}%, 50% PI coverage: {in_50pi:.0f}%")
        print(f"  Anomalies (outside 90% PI): {np.sum(outside_90pi)}")
        if len(anomaly_dates) > 0:
            for d, r in zip(anomaly_dates, anomaly_residuals):
                direction = "above" if r > 0 else "below"
                print(f"    {d}: {abs(r):.1f} {series_name} {direction} median forecast")

        results[series_name] = {
            "context": context,
            "actual": actual,
            "forecast": forecast,
            "context_dates": context_dates,
            "forecast_dates": forecast_dates,
            "metrics": series_metrics,
            "outside_90pi": outside_90pi,
        }

        metrics[f"chronos_nightly_{series_name}"] = series_metrics

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ===========================================================================
# ANALYSIS 2: CHRONOS ON CONTINUOUS HOURLY HR
# ===========================================================================

def run_chronos_hourly_hr(
    pipeline,
    hourly: pd.DataFrame,
    metrics: dict,
) -> dict[str, Any]:
    """Apply Chronos to hourly HR for sub-daily anomaly detection."""
    import torch

    print("\n" + "=" * 70)
    print("[CHRONOS] Analysis 2: Continuous Hourly HR Forecasting")
    print("=" * 70)

    hr_series = hourly["hr_mean"].values
    hr_times = hourly["datetime"].values
    n = len(hr_series)

    # Use 80% as context, forecast remaining 20%
    context_len = int(n * 0.8)
    forecast_len = min(48, n - context_len)  # Cap at 48 hours

    # Limit context to model's max context window
    max_ctx = pipeline.model_context_length or 512
    if context_len > max_ctx:
        # Use last max_ctx hours as context
        context_start = context_len - max_ctx
        context_len = max_ctx
    else:
        context_start = 0

    context = hr_series[context_start:context_start + context_len]
    actual = hr_series[context_start + context_len:context_start + context_len + forecast_len]
    forecast_times = hr_times[context_start + context_len:context_start + context_len + forecast_len]

    print(f"  Total hourly readings: {n}")
    print(f"  Context: {context_len} hours")
    print(f"  Forecast horizon: {forecast_len} hours")

    t0 = time.time()
    forecast = chronos_forecast_with_quantiles(pipeline, context, forecast_len)
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.2f}s")

    # Metrics
    median_f = forecast["median"]
    residuals = actual - median_f
    residual_std = np.std(residuals)
    mae = np.mean(np.abs(residuals))

    # Anomalous hours: > 2 SD from median forecast
    anomalous_mask = np.abs(residuals) > 2 * residual_std
    n_anomalous = int(np.sum(anomalous_mask))

    # Coverage
    in_90pi = np.sum(
        (actual >= forecast["q10"]) & (actual <= forecast["q90"])
    ) / len(actual) * 100

    # Also run sliding window anomaly detection over full series
    # Use a rolling context window approach
    print("\n[CHRONOS] Running sliding window anomaly detection on hourly HR...")
    window_size = min(168, max_ctx)  # 7 days or model max
    step_size = 24  # Advance by 1 day
    predict_ahead = 24  # Predict 24h ahead

    sliding_anomalies = []

    # Only do sliding window if we have enough data
    n_windows = (n - window_size - predict_ahead) // step_size
    if n_windows > 0:
        print(f"  Windows: {n_windows} (size={window_size}h, step={step_size}h, predict={predict_ahead}h)")
        for i in range(0, min(n_windows, 30), 1):  # Cap at 30 windows to limit compute
            start = i * step_size
            ctx_slice = hr_series[start:start + window_size]
            act_slice = hr_series[start + window_size:start + window_size + predict_ahead]
            time_slice = hr_times[start + window_size:start + window_size + predict_ahead]

            if len(act_slice) < predict_ahead:
                break

            try:
                sw_forecast = chronos_forecast_with_quantiles(
                    pipeline, ctx_slice, predict_ahead
                )
                sw_residuals = act_slice - sw_forecast["median"]
                sw_outside = (act_slice < sw_forecast["q10"]) | (act_slice > sw_forecast["q90"])

                for j, (is_out, ts, res) in enumerate(zip(sw_outside, time_slice, sw_residuals)):
                    if is_out:
                        sliding_anomalies.append({
                            "datetime": str(ts),
                            "residual": float(res),
                            "actual": float(act_slice[j]),
                            "forecast_median": float(sw_forecast["median"][j]),
                        })
            except Exception as e:
                print(f"    Window {i} failed: {e}")
                continue

        print(f"  Sliding window anomalies found: {len(sliding_anomalies)}")
    else:
        print("  Insufficient data for sliding window analysis")

    # Check if any Feb 9 anomalies were found
    feb9_anomalies = [a for a in sliding_anomalies if KNOWN_EVENT_DATE_STR in a["datetime"]]
    if feb9_anomalies:
        print(f"  Feb 9 anomalies in hourly HR: {len(feb9_anomalies)}")

    hourly_metrics = {
        "total_hours": int(n),
        "context_hours": int(context_len),
        "forecast_hours": int(forecast_len),
        "mae": round(float(mae), 2),
        "residual_std": round(float(residual_std), 2),
        "coverage_90pi": round(float(in_90pi), 1),
        "n_anomalous_hours_2sd": n_anomalous,
        "sliding_window_anomalies": len(sliding_anomalies),
        "feb9_hourly_anomalies": len(feb9_anomalies),
        "inference_time_s": round(elapsed, 2),
    }
    metrics["chronos_hourly_hr"] = hourly_metrics

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "context": context,
        "actual": actual,
        "forecast": forecast,
        "forecast_times": forecast_times,
        "residuals": residuals,
        "anomalous_mask": anomalous_mask,
        "sliding_anomalies": sliding_anomalies,
        "metrics": hourly_metrics,
    }


# ===========================================================================
# ANALYSIS 3: STATISTICAL BASELINE (ARIMA/Prophet) + ENSEMBLE
# ===========================================================================

def run_statistical_baseline(
    nightly: pd.DataFrame,
    metrics: dict,
) -> dict[str, Any]:
    """Run ARIMA baseline forecast on nightly series for ensemble comparison."""

    print("\n" + "=" * 70)
    print("[ENSEMBLE] Analysis 3: Statistical Baseline + Ensemble Scoring")
    print("=" * 70)

    results = {}

    for series_name, col in [("rmssd", "nightly_rmssd_mean"), ("hr", "nightly_hr_mean")]:
        print(f"\n[ENSEMBLE] ARIMA baseline for nightly {series_name.upper()}...")

        s = nightly[["date", col]].dropna().copy()
        s.reset_index(drop=True, inplace=True)
        n = len(s)

        if n < CONTEXT_LENGTH + 5:
            print("  Insufficient data. Skipping.")
            continue

        context_end = min(CONTEXT_LENGTH, n - FORECAST_HORIZON)
        actual_horizon = min(FORECAST_HORIZON, n - context_end)

        context = s[col].values[:context_end]
        actual = s[col].values[context_end:context_end + actual_horizon]
        forecast_dates = s["date"].values[context_end:context_end + actual_horizon]

        # Try ARIMA first, fall back to simple exponential smoothing
        forecast_median = None
        forecast_lower = None
        forecast_upper = None
        model_used = "none"

        try:
            from statsmodels.tsa.arima.model import ARIMA

            # Auto-select order with simple heuristic
            best_aic = np.inf
            best_order = (1, 0, 1)

            for p in range(0, 4):
                for d in range(0, 2):
                    for q in range(0, 4):
                        try:
                            model = ARIMA(context, order=(p, d, q))
                            fit = model.fit()
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_order = (p, d, q)
                        except Exception:
                            continue

            print(f"  Best ARIMA order: {best_order} (AIC: {best_aic:.1f})")
            model = ARIMA(context, order=best_order)
            fit = model.fit()
            fc = fit.get_forecast(steps=actual_horizon)
            forecast_median = fc.predicted_mean
            ci = fc.conf_int(alpha=0.1)  # 90% CI
            forecast_lower = ci[:, 0]
            forecast_upper = ci[:, 1]
            model_used = f"ARIMA{best_order}"
            print(f"  {model_used} forecast complete")

        except Exception as e:
            print(f"  ARIMA failed: {e}")
            # Simple fallback: exponential smoothing
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing

                model = ExponentialSmoothing(
                    context, trend="add", seasonal=None, damped_trend=True
                )
                fit = model.fit()
                forecast_median = fit.forecast(actual_horizon)
                # Approximate CI using residual std
                resid_std = np.std(fit.resid)
                forecast_lower = forecast_median - 1.645 * resid_std
                forecast_upper = forecast_median + 1.645 * resid_std
                model_used = "ExpSmoothing"
                print("  ExpSmoothing fallback complete")
            except Exception as e2:
                print(f"  All statistical models failed: {e2}")
                # Last resort: naive forecast (last value repeated)
                forecast_median = np.full(actual_horizon, context[-1])
                rolling_std = np.std(context[-14:]) if len(context) >= 14 else np.std(context)
                forecast_lower = forecast_median - 1.645 * rolling_std
                forecast_upper = forecast_median + 1.645 * rolling_std
                model_used = "Naive"
                print("  Using naive forecast fallback")

        # Compute metrics
        mae = np.mean(np.abs(actual - forecast_median))
        rmse = np.sqrt(np.mean((actual - forecast_median) ** 2))

        in_90ci = np.sum(
            (actual >= forecast_lower) & (actual <= forecast_upper)
        ) / len(actual) * 100

        # Anomalies: outside 90% CI
        outside = (actual < forecast_lower) | (actual > forecast_upper)
        anomaly_dates = forecast_dates[outside]

        stat_metrics = {
            "model": model_used,
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "coverage_90ci": round(float(in_90ci), 1),
            "n_anomalies": int(np.sum(outside)),
            "anomaly_dates": [str(d) for d in anomaly_dates],
        }

        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"  90% CI coverage: {in_90ci:.0f}%")
        print(f"  Anomalies: {np.sum(outside)}")

        results[series_name] = {
            "actual": actual,
            "forecast_median": forecast_median,
            "forecast_lower": forecast_lower,
            "forecast_upper": forecast_upper,
            "forecast_dates": forecast_dates,
            "outside": outside,
            "metrics": stat_metrics,
        }

        metrics[f"statistical_{series_name}"] = stat_metrics

    return results


def compute_ensemble_consensus(
    chronos_results: dict,
    stat_results: dict,
    nightly: pd.DataFrame,
    metrics: dict,
) -> pd.DataFrame:
    """Compute ensemble consensus from Chronos + statistical model.

    For each forecast day:
      - Both flag anomaly = HIGH confidence
      - One flags = MEDIUM confidence
      - Neither = NORMAL
    """
    print("\n[ENSEMBLE] Computing ensemble consensus scoring...")

    rows = []
    for series_name in ["rmssd", "hr"]:
        if series_name not in chronos_results or series_name not in stat_results:
            continue

        cr = chronos_results[series_name]
        sr = stat_results[series_name]

        # Align on forecast dates
        chrono_dates = set(str(d) for d in cr["forecast_dates"])
        stat_dates = set(str(d) for d in sr["forecast_dates"])
        common_dates = sorted(chrono_dates & stat_dates)

        for d in common_dates:
            c_idx = list(cr["forecast_dates"]).index(d)
            s_idx = list(sr["forecast_dates"]).index(d)

            c_anomaly = bool(cr["outside_90pi"][c_idx])
            s_anomaly = bool(sr["outside"][s_idx])

            if c_anomaly and s_anomaly:
                confidence = "HIGH"
            elif c_anomaly or s_anomaly:
                confidence = "MEDIUM"
            else:
                confidence = "NORMAL"

            actual_val = float(cr["actual"][c_idx])
            chrono_median = float(cr["forecast"]["median"][c_idx])
            stat_median = float(sr["forecast_median"][s_idx])

            rows.append({
                "date": d,
                "series": series_name,
                "actual": actual_val,
                "chronos_median": chrono_median,
                "stat_median": stat_median,
                "chronos_anomaly": c_anomaly,
                "stat_anomaly": s_anomaly,
                "consensus": confidence,
                "chronos_residual": actual_val - chrono_median,
                "stat_residual": actual_val - stat_median,
            })

    ensemble_df = pd.DataFrame(rows)

    if len(ensemble_df) > 0:
        # Summary
        for level in ["HIGH", "MEDIUM", "NORMAL"]:
            cnt = len(ensemble_df[ensemble_df["consensus"] == level])
            print(f"  {level}: {cnt} date-series pairs")

        # Check Feb 9 alignment
        feb9 = ensemble_df[ensemble_df["date"] == KNOWN_EVENT_DATE_STR]
        if len(feb9) > 0:
            print("\n  Feb 9 event consensus:")
            for _, row in feb9.iterrows():
                print(f"    {row['series']}: {row['consensus']} "
                      f"(Chronos: {'ANOMALY' if row['chronos_anomaly'] else 'normal'}, "
                      f"Stat: {'ANOMALY' if row['stat_anomaly'] else 'normal'})")

        high_dates = sorted(ensemble_df[ensemble_df["consensus"] == "HIGH"]["date"].unique())
        metrics["ensemble_consensus"] = {
            "high_confidence_anomaly_dates": high_dates,
            "medium_confidence_count": int(len(ensemble_df[ensemble_df["consensus"] == "MEDIUM"])),
            "normal_count": int(len(ensemble_df[ensemble_df["consensus"] == "NORMAL"])),
            "feb9_detected": KNOWN_EVENT_DATE_STR in high_dates,
        }
    else:
        metrics["ensemble_consensus"] = {"error": "no overlapping forecasts"}

    return ensemble_df


# ===========================================================================
# ANALYSIS 4: FEB 9 EVENT RETROSPECTIVE VALIDATION
# ===========================================================================

def run_feb9_retrospective(
    pipeline,
    nightly: pd.DataFrame,
    metrics: dict,
) -> dict[str, Any]:
    """Retrospective validation: train up to day before known event, forecast event + 6 days."""
    import torch

    print("\n" + "=" * 70)
    print("[VALIDATION] Analysis 4: Known Event Retrospective")
    print("=" * 70)

    # Derive the day before the known event date
    _pre_event_date = str(KNOWN_EVENT_DATE - timedelta(days=1))

    results = {}

    for series_name, col in [("rmssd", "nightly_rmssd_mean"), ("hr", "nightly_hr_mean")]:
        print(f"\n[VALIDATION] {series_name.upper()} retrospective...")

        s = nightly[["date", col]].dropna().copy()
        s.reset_index(drop=True, inplace=True)

        # Find pre-event index
        pre_event_mask = s["date"] <= _pre_event_date
        if not pre_event_mask.any():
            print(f"  No data before {_pre_event_date}. Skipping.")
            continue

        context = s.loc[pre_event_mask, col].values
        context_dates = s.loc[pre_event_mask, "date"].values

        # Forecast event date + 6 days (7 days total)
        retro_horizon = 7
        post_feb8 = s[s["date"] > _pre_event_date].head(retro_horizon)
        if len(post_feb8) == 0:
            print("  No data after Feb 8. Skipping.")
            continue

        actual = post_feb8[col].values
        actual_dates = post_feb8["date"].values

        # Limit context to model max
        max_ctx = pipeline.model_context_length or 512
        if len(context) > max_ctx:
            context = context[-max_ctx:]

        print(f"  Context: {len(context)} nights up to {context_dates[-1]}")
        print(f"  Forecasting: {len(actual)} nights ({actual_dates[0]} to {actual_dates[-1]})")

        t0 = time.time()
        forecast = chronos_forecast_with_quantiles(
            pipeline, context, len(actual)
        )
        elapsed = time.time() - t0

        median_f = forecast["median"]
        residuals = actual - median_f

        # Check if Feb 9 specifically is anomalous
        feb9_idx = None
        for i, d in enumerate(actual_dates):
            if str(d) == KNOWN_EVENT_DATE_STR:
                feb9_idx = i
                break

        feb9_detected = False
        feb9_residual = None
        feb9_direction = None

        if feb9_idx is not None:
            feb9_actual = actual[feb9_idx]
            feb9_median = median_f[feb9_idx]
            feb9_q10 = forecast["q10"][feb9_idx]
            feb9_q90 = forecast["q90"][feb9_idx]
            feb9_residual = float(feb9_actual - feb9_median)
            feb9_detected = bool(feb9_actual < feb9_q10 or feb9_actual > feb9_q90)
            feb9_direction = "above" if feb9_residual > 0 else "below"

            print("\n  === Feb 9 Analysis ===")
            print(f"  Actual {series_name}: {feb9_actual:.1f}")
            print(f"  Median forecast: {feb9_median:.1f}")
            print(f"  90% PI: [{feb9_q10:.1f}, {feb9_q90:.1f}]")
            print(f"  Residual: {feb9_residual:+.1f} ({feb9_direction})")
            print(f"  Outside 90% PI: {'YES - ANOMALY DETECTED' if feb9_detected else 'No'}")

        # Compute overall rate of observations outside 90% PI
        outside_90 = (actual < forecast["q10"]) | (actual > forecast["q90"])
        detection_rate = np.sum(outside_90) / len(actual) * 100

        # Compare against anomaly detection report methods
        retro_metrics = {
            "context_nights": int(len(context)),
            "forecast_nights": int(len(actual)),
            "forecast_dates": [str(d) for d in actual_dates],
            "feb9_detected": feb9_detected,
            "feb9_residual": round(float(feb9_residual), 2) if feb9_residual is not None else None,
            "feb9_direction": feb9_direction,
            "feb9_actual": round(float(actual[feb9_idx]), 2) if feb9_idx is not None else None,
            "feb9_median_forecast": round(float(median_f[feb9_idx]), 2) if feb9_idx is not None else None,
            "n_anomalies_in_window": int(np.sum(outside_90)),
            "detection_rate_pct": round(float(detection_rate), 1),
            "inference_time_s": round(elapsed, 2),
        }

        results[series_name] = {
            "context": context,
            "actual": actual,
            "actual_dates": actual_dates,
            "forecast": forecast,
            "feb9_idx": feb9_idx,
            "feb9_detected": feb9_detected,
            "outside_90": outside_90,
            "metrics": retro_metrics,
        }

        metrics[f"feb9_retro_{series_name}"] = retro_metrics

    torch.cuda.empty_cache()
    gc.collect()

    return results


# ===========================================================================
# ANALYSIS 5: PRE vs POST RUXOLITINIB
# ===========================================================================

def run_ruxolitinib_analysis(
    pipeline,
    nightly: pd.DataFrame,
    metrics: dict,
) -> dict[str, Any]:
    """Forecast into the ruxolitinib period using pre-period context."""
    import torch

    print("\n" + "=" * 70)
    print("[CHRONOS] Analysis 5: Pre vs Post Ruxolitinib Forecast")
    print("=" * 70)

    results = {}

    for series_name, col in [("rmssd", "nightly_rmssd_mean"), ("hr", "nightly_hr_mean")]:
        print(f"\n[CHRONOS] Ruxolitinib analysis: {series_name.upper()}...")

        s = nightly[["date", col]].dropna().copy()
        s.reset_index(drop=True, inplace=True)

        # Split at ruxolitinib start
        pre_mask = s["date"] < RUXOLITINIB_START
        post_mask = s["date"] >= RUXOLITINIB_START

        pre_data = s[pre_mask]
        post_data = s[post_mask]

        if len(pre_data) < 10:
            print(f"  Insufficient pre-ruxolitinib data ({len(pre_data)} nights). Skipping.")
            continue

        context = pre_data[col].values
        max_ctx = pipeline.model_context_length or 512
        if len(context) > max_ctx:
            context = context[-max_ctx:]

        # Forecast into post-period (or up to 14 days if post data is limited)
        n_post = len(post_data)
        forecast_len = max(n_post, 7)  # At least 7 days forecast even if no post data yet

        print(f"  Pre-period: {len(pre_data)} nights (context: {len(context)})")
        print(f"  Post-period actual: {n_post} nights")
        print(f"  Forecast length: {forecast_len} nights")

        t0 = time.time()
        forecast = chronos_forecast_with_quantiles(
            pipeline, context, forecast_len
        )
        elapsed = time.time() - t0

        # Pre-period forecast uncertainty (use last 14 days of pre-period)
        pre_last14_ctx = pre_data[col].values[:-14] if len(pre_data) > 14 else pre_data[col].values[:len(pre_data)//2]
        pre_last14_actual = pre_data[col].values[-14:] if len(pre_data) > 14 else pre_data[col].values[len(pre_data)//2:]

        if len(pre_last14_ctx) > 5 and len(pre_last14_actual) > 0:
            pre_forecast = chronos_forecast_with_quantiles(
                pipeline, pre_last14_ctx, len(pre_last14_actual)
            )
            pre_pi_width = np.mean(pre_forecast["q90"] - pre_forecast["q10"])
        else:
            pre_pi_width = None

        # Post-period PI width
        post_pi_width = np.mean(forecast["q90"][:forecast_len] - forecast["q10"][:forecast_len])

        # Compare pre vs post uncertainty
        if pre_pi_width is not None:
            uncertainty_change = ((post_pi_width - pre_pi_width) / pre_pi_width) * 100
            print(f"  Pre-period 90% PI width: {pre_pi_width:.2f}")
            print(f"  Post-period 90% PI width: {post_pi_width:.2f}")
            print(f"  Uncertainty change: {uncertainty_change:+.1f}%")
            if uncertainty_change < -10:
                print("  -> Uncertainty NARROWING: suggests stabilization")
            elif uncertainty_change > 10:
                print("  -> Uncertainty WIDENING: suggests increased unpredictability")
        else:
            uncertainty_change = None

        # If we have actual post-ruxolitinib data, compare
        post_comparison = None
        if n_post > 0:
            post_actual = post_data[col].values
            post_forecast_slice = forecast["median"][:n_post]
            post_residuals = post_actual - post_forecast_slice
            mean_shift = np.mean(post_residuals)

            print("\n  Post-ruxolitinib actual vs forecast:")
            print(f"  Mean shift: {mean_shift:+.2f} (actual - forecast)")
            if series_name == "rmssd" and mean_shift > 0:
                print("  -> RMSSD increased: possible parasympathetic improvement")
            elif series_name == "hr" and mean_shift < 0:
                print("  -> HR decreased: possible autonomic improvement")

            post_comparison = {
                "n_post_nights": n_post,
                "mean_shift": round(float(mean_shift), 2),
                "post_dates": [str(d) for d in post_data["date"].values],
                "post_actual": [round(float(v), 2) for v in post_actual],
                "post_forecast_median": [round(float(v), 2) for v in post_forecast_slice],
            }

        rux_metrics = {
            "pre_period_nights": int(len(pre_data)),
            "post_period_nights": n_post,
            "pre_pi_width_90": round(float(pre_pi_width), 2) if pre_pi_width else None,
            "post_pi_width_90": round(float(post_pi_width), 2),
            "uncertainty_change_pct": round(float(uncertainty_change), 1) if uncertainty_change is not None else None,
            "inference_time_s": round(elapsed, 2),
        }
        if post_comparison:
            rux_metrics["post_comparison"] = post_comparison

        results[series_name] = {
            "pre_context": context,
            "forecast": forecast,
            "forecast_len": forecast_len,
            "pre_dates": pre_data["date"].values,
            "post_data": post_data,
            "pre_pi_width": pre_pi_width,
            "post_pi_width": post_pi_width,
            "metrics": rux_metrics,
        }

        metrics[f"ruxolitinib_{series_name}"] = rux_metrics

    torch.cuda.empty_cache()
    gc.collect()

    return results


# ===========================================================================
# VISUALIZATION
# ===========================================================================

def create_forecast_figure(
    title: str,
    context_dates,
    context_values,
    forecast_dates,
    actual_values,
    forecast: dict,
    series_label: str,
    unit: str,
    known_event_date: str = KNOWN_EVENT_DATE_STR,
    rux_date: str = RUXOLITINIB_START,
    outside_90pi=None,
) -> go.Figure:
    """Create a Plotly figure with forecast bands and anomaly markers."""

    fig = go.Figure()

    ctx_dt = pd.to_datetime(context_dates)
    fc_dt = pd.to_datetime(forecast_dates)

    # --- Context / forecast boundary ---
    boundary_dt = ctx_dt[-1] if len(ctx_dt) > 0 else fc_dt[0]
    fig.add_shape(
        type="line", x0=boundary_dt, x1=boundary_dt,
        y0=0, y1=1, yref="paper",
        line=dict(color="rgba(255,255,255,0.25)", dash="dot", width=1),
    )

    # --- Bands first (behind lines) ---

    # 90% PI band — very subtle
    fig.add_trace(go.Scatter(
        x=np.concatenate([fc_dt, fc_dt[::-1]]),
        y=np.concatenate([forecast["q90"], forecast["q10"][::-1]]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.08)",
        line=dict(width=0),
        name="90% PI",
        hoverinfo="skip",
    ))

    # 50% PI band — slightly more visible
    fig.add_trace(go.Scatter(
        x=np.concatenate([fc_dt, fc_dt[::-1]]),
        y=np.concatenate([forecast["q75"], forecast["q25"][::-1]]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.18)",
        line=dict(width=0),
        name="50% PI",
        hoverinfo="skip",
    ))

    # --- Lines on top of bands ---

    # Context (historical) — bold solid
    fig.add_trace(go.Scatter(
        x=ctx_dt, y=context_values,
        mode="lines+markers",
        name=f"Context ({series_label})",
        line=dict(color=COLOR_PRE, width=2.5),
        marker=dict(size=3),
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            + f"{series_label}: %{{y:.1f}} {unit}"
            + "<extra></extra>"
        ),
    ))

    # Actual forecast period — bold solid
    fig.add_trace(go.Scatter(
        x=fc_dt, y=actual_values,
        mode="lines+markers",
        name="Actual",
        line=dict(color=COLOR_POST, width=2.5),
        marker=dict(size=5),
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            + f"Actual: %{{y:.1f}} {unit}"
            + "<extra></extra>"
        ),
    ))

    # Median forecast — dashed to distinguish from actual
    fig.add_trace(go.Scatter(
        x=fc_dt, y=forecast["median"],
        mode="lines",
        name="Chronos Median",
        line=dict(color=COLOR_FORECAST, width=2.5, dash="dash"),
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            + f"Forecast: %{{y:.1f}} {unit}"
            + "<extra></extra>"
        ),
    ))

    # --- Anomaly markers on top ---
    if outside_90pi is not None and np.any(outside_90pi):
        anomaly_idx = np.where(outside_90pi)[0]
        fig.add_trace(go.Scatter(
            x=fc_dt[anomaly_idx],
            y=actual_values[anomaly_idx],
            mode="markers",
            name="Anomaly (outside 90% PI)",
            marker=dict(
                color=COLOR_ANOMALY, size=13, symbol="x-thin-open",
                line=dict(width=3, color=COLOR_ANOMALY),
            ),
            hovertemplate=(
                "<b>ANOMALY</b><br>"
                + "<b>%{x|%b %d, %Y}</b><br>"
                + f"Actual: %{{y:.1f}} {unit}"
                + "<extra></extra>"
            ),
        ))

    # Known event line
    event_dt = pd.Timestamp(known_event_date)

    fig.add_shape(
        type="line", x0=event_dt, x1=event_dt,
        y0=0, y1=1, yref="paper",
        line=dict(color=ACCENT_RED, dash="dot", width=2),
    )
    fig.add_annotation(
        x=event_dt, y=1, yref="paper",
        text="<b>Feb 9 akutt hendelse</b>",
        showarrow=False, yanchor="bottom",
        font=dict(size=10, color=ACCENT_RED),
        bgcolor="rgba(15, 17, 23, 0.7)",
    )

    # Ruxolitinib line
    rux_dt = pd.Timestamp(rux_date)
    if rux_dt <= fc_dt.max():
        fig.add_shape(
            type="line", x0=rux_dt, x1=rux_dt,
            y0=0, y1=1, yref="paper",
            line=dict(color=COLOR_RUX_LINE, dash="dashdot", width=2),
        )
        fig.add_annotation(
            x=rux_dt, y=1, yref="paper",
            text="<b>Ruxolitinib start</b>",
            showarrow=False, yanchor="bottom",
            font=dict(size=10, color=COLOR_RUX_LINE),
            bgcolor="rgba(15, 17, 23, 0.7)",
        )

    fig.update_layout(
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis_title="Dato",
        yaxis_title=f"{series_label} ({unit})",
        height=450,
        hovermode="x unified",
        xaxis=dict(
            spikemode="across", spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)", spikesnap="cursor",
            spikedash="dot", showspikes=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True, spikemode="across",
            spikethickness=1, spikecolor="rgba(255,255,255,0.15)",
            spikedash="dot", spikesnap="cursor",
        ),
    )

    return fig


def create_ensemble_heatmap(ensemble_df: pd.DataFrame) -> go.Figure:
    """Create a heatmap showing ensemble consensus across dates and series."""
    if len(ensemble_df) == 0:
        return go.Figure().add_annotation(text="No ensemble data", showarrow=False)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("RMSSD Ensemble Consensus", "HR Ensemble Consensus"),
        row_heights=[0.5, 0.5],
    )

    for i, sname in enumerate(["rmssd", "hr"], 1):
        sdf = ensemble_df[ensemble_df["series"] == sname].copy()
        if len(sdf) == 0:
            continue

        dates = pd.to_datetime(sdf["date"])

        # Color by consensus (dark-theme-friendly) with subtle edge definition
        color_map = {"HIGH": ACCENT_RED, "MEDIUM": ACCENT_AMBER, "NORMAL": ACCENT_GREEN}
        edge_map = {
            "HIGH": "rgba(239, 68, 68, 0.6)",
            "MEDIUM": "rgba(245, 158, 11, 0.6)",
            "NORMAL": "rgba(16, 185, 129, 0.6)",
        }
        colors = [color_map.get(c, TEXT_SECONDARY) for c in sdf["consensus"]]
        edges = [edge_map.get(c, "rgba(255,255,255,0.1)") for c in sdf["consensus"]]

        # Residual bars with refined hover and edge lines
        fig.add_trace(go.Bar(
            x=dates,
            y=sdf["chronos_residual"],
            marker=dict(
                color=colors,
                line=dict(width=1, color=edges),
            ),
            name=f"Chronos residual ({sname.upper()})",
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Residual: %{y:.2f}<br>"
                "Level: <b>%{text}</b>"
                "<extra></extra>"
            ),
            text=sdf["consensus"],
            showlegend=False,
        ), row=i, col=1)

    # Add legend entries for consensus levels
    for level, color in [("HIGH", ACCENT_RED), ("MEDIUM", ACCENT_AMBER), ("NORMAL", ACCENT_GREEN)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=f"Consensus: {level}",
        ))

    fig.update_layout(
        margin=dict(l=50, r=30, t=100, b=40),
        height=500,
        barmode="overlay",
        hovermode="x unified",
    )

    # Crosshair spikes and subtle gridlines on both panels
    for ax_suffix in ["", "2"]:
        fig.update_layout(**{
            f"xaxis{ax_suffix}": dict(
                spikemode="across", spikethickness=1,
                spikecolor="rgba(255,255,255,0.15)", spikesnap="cursor",
                spikedash="dot", showspikes=True,
                gridcolor="rgba(255,255,255,0.05)",
            ),
            f"yaxis{ax_suffix}": dict(
                gridcolor="rgba(255,255,255,0.05)",
            ),
        })

    return fig


def create_ruxolitinib_figure(
    rux_results: dict,
    series_name: str,
    col_label: str,
    unit: str,
) -> go.Figure:
    """Create figure comparing pre and post ruxolitinib forecasts."""
    if series_name not in rux_results:
        return go.Figure().add_annotation(text=f"No {series_name} data", showarrow=False)

    r = rux_results[series_name]
    fig = go.Figure()

    pre_dates = pd.to_datetime(r["pre_dates"])
    forecast = r["forecast"]
    forecast_len = r["forecast_len"]

    # Generate forecast dates starting from day after last pre-period date
    last_pre_date = pre_dates[-1]
    fc_dates = pd.date_range(start=last_pre_date + pd.Timedelta(days=1), periods=forecast_len)

    rux_dt = pd.Timestamp(RUXOLITINIB_START)

    # --- Background shading: pre (subtle red) vs post (subtle green) ---
    all_dates = list(pre_dates) + list(fc_dates)
    x_min = min(all_dates) - pd.Timedelta(days=1)
    x_max = max(all_dates) + pd.Timedelta(days=1)
    fig.add_vrect(
        x0=x_min, x1=rux_dt,
        fillcolor="rgba(239, 68, 68, 0.04)", line_width=0,
        layer="below",
    )
    fig.add_vrect(
        x0=rux_dt, x1=x_max,
        fillcolor="rgba(16, 185, 129, 0.04)", line_width=0,
        layer="below",
    )

    # --- Band first (behind lines) ---
    fig.add_trace(go.Scatter(
        x=np.concatenate([fc_dates, fc_dates[::-1]]),
        y=np.concatenate([forecast["q90"][:forecast_len], forecast["q10"][:forecast_len][::-1]]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.08)",
        line=dict(width=0),
        name="90% PI",
        hoverinfo="skip",
    ))

    # --- Lines on top ---

    # Pre-period actual
    pre_values = r["pre_context"]
    # Show last 30 days of pre-period for visual clarity
    show_n = min(30, len(pre_values))
    fig.add_trace(go.Scatter(
        x=pre_dates[-show_n:],
        y=pre_values[-show_n:],
        mode="lines+markers",
        name="Pre-ruxolitinib actual",
        line=dict(color=COLOR_PRE, width=2.5),
        marker=dict(size=3),
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            + f"Pre: %{{y:.1f}} {unit}"
            + "<extra></extra>"
        ),
    ))

    # Post-period actual (if available)
    post_data = r["post_data"]
    if len(post_data) > 0:
        post_dates = pd.to_datetime(post_data["date"].values)
        # Determine which column to plot
        if series_name == "rmssd":
            post_vals = post_data["nightly_rmssd_mean"].values
        else:
            post_vals = post_data["nightly_hr_mean"].values

        fig.add_trace(go.Scatter(
            x=post_dates,
            y=post_vals,
            mode="lines+markers",
            name="Post-ruxolitinib actual",
            line=dict(color=COLOR_POST, width=2.5),
            marker=dict(size=5),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                + f"Post: %{{y:.1f}} {unit}"
                + "<extra></extra>"
            ),
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_dates, y=forecast["median"],
        mode="lines",
        name="Chronos forecast (median)",
        line=dict(color=COLOR_FORECAST, width=2.5, dash="dash"),
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            + f"Forecast: %{{y:.1f}} {unit}"
            + "<extra></extra>"
        ),
    ))

    # Ruxolitinib intervention line
    fig.add_shape(
        type="line", x0=rux_dt, x1=rux_dt,
        y0=0, y1=1, yref="paper",
        line=dict(color=COLOR_RUX_LINE, dash="dashdot", width=2),
    )
    fig.add_annotation(
        x=rux_dt, y=1, yref="paper",
        text="<b>Ruxolitinib 10mg BID</b>",
        showarrow=False, yanchor="bottom",
        font=dict(size=11, color=COLOR_RUX_LINE, family=FONT_FAMILY),
        bgcolor="rgba(15, 17, 23, 0.7)",
    )

    # PI width annotation
    pi_text = f"Pre PI width: {r['pre_pi_width']:.1f}" if r['pre_pi_width'] else ""
    pi_text += f"<br>Post PI width: {r['post_pi_width']:.1f}"
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=pi_text, showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=11, family=FONT_FAMILY),
        bgcolor=BG_ELEVATED, bordercolor=BORDER_SUBTLE, borderwidth=1,
    )

    fig.update_layout(
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis_title="Dato",
        yaxis_title=f"{col_label} ({unit})",
        height=450,
        hovermode="x unified",
        xaxis=dict(
            spikemode="across", spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)", spikesnap="cursor",
            spikedash="dot", showspikes=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True, spikemode="across",
            spikethickness=1, spikecolor="rgba(255,255,255,0.15)",
            spikedash="dot", spikesnap="cursor",
        ),
    )

    return fig


def create_hourly_hr_figure(hourly_results: dict) -> go.Figure:
    """Create figure for hourly HR forecast analysis."""
    fig = go.Figure()

    forecast_times = pd.to_datetime(hourly_results["forecast_times"])
    actual = hourly_results["actual"]
    forecast = hourly_results["forecast"]
    anomalous = hourly_results["anomalous_mask"]

    # --- Band first (behind lines) — very subtle ---
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast_times, forecast_times[::-1]]),
        y=np.concatenate([forecast["q90"], forecast["q10"][::-1]]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.08)",
        line=dict(width=0),
        name="90% PI",
        hoverinfo="skip",
    ))

    # --- Lines on top ---
    # Actual HR: subtle fine-grained data (thinner, smaller markers)
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=actual,
        mode="lines+markers",
        name="Actual HR",
        line=dict(color=COLOR_POST, width=1.2),
        marker=dict(size=2),
        opacity=0.7,
        hovertemplate=(
            "<b>%{x|%b %d %H:%M}</b><br>"
            "Actual: %{y:.0f} bpm"
            "<extra></extra>"
        ),
    ))

    # Forecast: prominent (bolder, dashed)
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=forecast["median"],
        mode="lines",
        name="Chronos Median",
        line=dict(color=COLOR_FORECAST, width=2.5, dash="dash"),
        hovertemplate=(
            "<b>%{x|%b %d %H:%M}</b><br>"
            "Forecast: %{y:.0f} bpm"
            "<extra></extra>"
        ),
    ))

    # --- Anomaly markers on top — red with white outlines ---
    if np.any(anomalous):
        fig.add_trace(go.Scatter(
            x=forecast_times[anomalous],
            y=actual[anomalous],
            mode="markers",
            name="Anomalous (>2 SD)",
            marker=dict(
                color=COLOR_ANOMALY, size=11, symbol="x-thin-open",
                line=dict(width=3, color=COLOR_ANOMALY),
            ),
            hovertemplate=(
                "<b>ANOMALY</b><br>"
                "<b>%{x|%b %d %H:%M}</b><br>"
                "Actual: %{y:.0f} bpm"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis_title="Time",
        yaxis_title="Heart Rate (bpm)",
        height=450,
        hovermode="x unified",
        xaxis=dict(
            spikemode="across", spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)", spikesnap="cursor",
            spikedash="dot", showspikes=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True, spikemode="across",
            spikethickness=1, spikecolor="rgba(255,255,255,0.15)",
            spikedash="dot", spikesnap="cursor",
        ),
    )

    return fig


# ===========================================================================
# HTML REPORT GENERATION
# ===========================================================================

def generate_simple_html_report(
    figures: list[tuple[str, go.Figure]],
    metrics: dict,
    feb9_results: dict,
) -> str:
    """Generate dark-themed HTML report using _theme design system."""
    print("\n[REPORT] Generating interactive HTML report...")

    # Build chart divs from figures
    chart_divs: dict[str, str] = {}
    for name, fig in figures:
        chart_divs[name] = fig.to_html(full_html=False, include_plotlyjs=False)

    # --- KPI row: top-level summary cards ---
    model_name = CHRONOS_MODEL.split("/")[-1]
    chronos_available = metrics.get("chronos_available", True)
    chronos_error = metrics.get("chronos_error")

    # Extract key metrics for KPI cards
    rmssd_m = metrics.get("chronos_nightly_rmssd", {})
    hr_m = metrics.get("chronos_nightly_hr", {})
    ens = metrics.get("ensemble_consensus", {})
    high_dates = ens.get("high_confidence_anomaly_dates", [])
    feb9_ens = ens.get("feb9_detected", False)
    feb9_retro_hits = sum(
        1
        for key in ("feb9_retro_rmssd", "feb9_retro_hr")
        if metrics.get(key, {}).get("feb9_detected", False)
    )
    feb9_retro_total = sum(1 for key in ("feb9_retro_rmssd", "feb9_retro_hr") if key in metrics)
    if not chronos_available or feb9_retro_total == 0:
        feb9_retro_value = "Not run"
        feb9_retro_status = "warning" if not chronos_available else "info"
        feb9_retro_detail = "Chronos retrospective holdout unavailable"
    else:
        feb9_retro_value = f"{feb9_retro_hits}/{feb9_retro_total}"
        feb9_retro_status = (
            "normal" if feb9_retro_hits == feb9_retro_total else
            "warning" if feb9_retro_hits > 0 else
            "critical"
        )
        feb9_retro_detail = "Chronos retrospective holdout detected Feb 9"

    kpi_cards = [
        make_kpi_card(
            "Foundation Model",
            model_name if chronos_available else "unavailable",
            status="info" if chronos_available else "warning",
            decimals=0,
            detail=None if chronos_available else chronos_error,
        ),
        make_kpi_card(
            "RMSSD MAE",
            rmssd_m.get("mae", "N/A"),
            unit="ms",
            status="normal" if rmssd_m.get("mae", 99) < 5 else "warning",
            detail=f'90% PI coverage: {rmssd_m.get("coverage_90pi", "N/A")}%',
        ),
        make_kpi_card(
            "HR MAE",
            hr_m.get("mae", "N/A"),
            unit="bpm",
            status="normal" if hr_m.get("mae", 99) < 3 else "warning",
            detail=f'90% PI coverage: {hr_m.get("coverage_90pi", "N/A")}%',
        ),
        make_kpi_card(
            "Feb 9 Chronos Holdout",
            feb9_retro_value,
            status=feb9_retro_status,
            decimals=0,
            detail=feb9_retro_detail,
        ),
        make_kpi_card(
            "Ruxolitinib Start",
            TREATMENT_START.strftime("%d. %b %Y").lower(),
            status="info",
            decimals=0,
        ),
        make_kpi_card(
            "March Ensemble",
            "Chronos + ARIMA" if chronos_available else "ARIMA only",
            status="neutral",
            decimals=0,
            detail=f'{len(high_dates)} HIGH confidence anomalies | Feb 9 {"detected" if feb9_ens else "not detected"}',
        ),
    ]
    body = make_kpi_row(*kpi_cards)

    # --- Section 1: Nightly RMSSD and HR ---
    if chronos_available:
        sec1_title = "1. Nightly RMSSD and HR - Chronos-2 Probabilistic Forecast"
        sec1_desc = (
            f"<p>{CONTEXT_LENGTH} nights context, {FORECAST_HORIZON} nights forecast. "
            f"Shaded bands: 90% (light) and 50% (dark) prediction intervals. "
            f"Cross = anomaly outside 90% PI.</p>"
        )
    else:
        sec1_title = "1. Nightly RMSSD and HR - Chronos unavailable"
        sec1_desc = (
            "<p>Chronos could not be loaded in this run, so nightly foundation-model "
            "forecast charts were skipped. Statistical baseline outputs remain below.</p>"
        )
    sec1_charts = chart_divs.get("nightly_rmssd", "") + chart_divs.get("nightly_hr", "")
    body += make_section(
        sec1_title,
        sec1_desc + sec1_charts,
        section_id="nightly-forecast",
    )

    # --- Section 2: Continuous Hourly HR ---
    sec2_desc = (
        "<p>21K+ HR readings downsampled to hourly means. "
        "Sliding window detection identifies anomalous segments.</p>"
        if chronos_available
        else "<p>Hourly Chronos analysis was skipped because the Chronos pipeline was unavailable.</p>"
    )
    sec2_chart = chart_divs.get("hourly_hr", "")
    body += make_section(
        "2. Continuous HR - Hourly Resolution Forecast" if chronos_available else "2. Continuous HR - Chronos unavailable",
        sec2_desc + sec2_chart,
        section_id="hourly-hr",
    )

    # --- Section 3: Ensemble Consensus ---
    sec3_desc = (
        "<p>Two independent models score each date. "
        "HIGH = both flag anomaly, MEDIUM = one, NORMAL = none.</p>"
        if chronos_available
        else "<p>Chronos was unavailable, so the consensus view is reduced to the statistical baseline only.</p>"
    )
    sec3_chart = chart_divs.get("ensemble", "")
    body += make_section(
        "3. Ensemble Consensus - Foundation Model + ARIMA Baseline"
        if chronos_available else
        "3. Statistical Baseline - ARIMA",
        sec3_desc + sec3_chart,
        section_id="ensemble",
    )

    # --- Section 4: Feb 9 Retrospective ---
    # Build Feb 9 detection table
    feb9_rows = ""
    for sname in ["rmssd", "hr"]:
        key = f"feb9_retro_{sname}"
        if key in metrics:
            m = metrics[key]
            detected = m.get("feb9_detected", False)
            residual = m.get("feb9_residual", "N/A")
            det_color = ACCENT_RED if detected else TEXT_SECONDARY
            feb9_rows += (
                f'<tr><td>Chronos Retrospective</td><td>{sname.upper()}</td>'
                f'<td style="color:{det_color};font-weight:700">'
                f'{"YES" if detected else "No"}</td>'
                f'<td>{residual}</td></tr>'
            )

    summary_box = f"""
    <div class="summary-box">
        <strong>Feb 9 Detection Summary</strong>
        <table style="margin-top: 8px;">
            <tr><th>Method</th><th>Series</th><th>Detected?</th><th>Residual</th></tr>
            {feb9_rows if feb9_rows else '<tr><td colspan="4">Chronos retrospective not available in this run.</td></tr>'}
        </table>
        <p style="margin-top: 8px;">
            Prospective March ensemble HIGH-confidence anomalies:
            <strong>{', '.join(high_dates) if high_dates else 'None detected'}</strong><br>
            Feb 9 in March ensemble consensus:
            <strong style="color:{ACCENT_RED if feb9_ens else TEXT_SECONDARY}">
            {'YES' if feb9_ens else 'No'}</strong>
        </p>
    </div>"""

    sec4_desc = (
        "<p>Model trained on all data through February 8, forecast for February 9-15. "
        "Tests whether the foundation model would have flagged the acute event prospectively.</p>"
        if chronos_available else
        "<p>Retrospective Chronos validation was skipped because the foundation model was unavailable.</p>"
    )
    sec4_charts = chart_divs.get("feb9_rmssd", "") + chart_divs.get("feb9_hr", "")
    body += make_section(
        "4. Feb 9 Retrospective Validation",
        sec4_desc + summary_box + sec4_charts,
        section_id="feb9-retro",
    )

    # --- Section 5: Pre vs Post Ruxolitinib ---
    sec5_desc = (
        "<p>Model trained on pre-ruxolitinib period and forecasts into treatment period. "
        "Narrower prediction intervals indicate stabilization, "
        "systematic shift indicates treatment effect.</p>"
        if chronos_available else
        "<p>Pre/post-ruxolitinib Chronos regime analysis was skipped because the foundation model was unavailable.</p>"
    )
    sec5_charts = chart_divs.get("rux_rmssd", "") + chart_divs.get("rux_hr", "")
    body += make_section(
        "5. Pre vs Post Ruxolitinib Regime Analysis",
        sec5_desc + sec5_charts,
        section_id="ruxolitinib",
    )

    # --- Section 6: Detailed Metrics ---
    metrics_rows = ""
    for section_name, data in sorted(metrics.items()):
        if isinstance(data, dict):
            metrics_rows += (
                f'<tr><td colspan="3" style="background:{BG_ELEVATED};'
                f'font-weight:600;color:{ACCENT_BLUE}">{section_name}</td></tr>\n'
            )
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    if isinstance(v, list) and len(v) <= 10:
                        display = ", ".join(str(x) for x in v)
                    else:
                        continue
                elif isinstance(v, float):
                    display = f"{v:.3f}"
                elif isinstance(v, bool):
                    display = "YES" if v else "No"
                else:
                    display = str(v)
                metrics_rows += f"<tr><td></td><td>{k}</td><td>{display}</td></tr>\n"

    metrics_table = (
        '<table><tr><th>Section</th><th>Metric</th><th>Value</th></tr>\n'
        + metrics_rows
        + "</table>"
    )
    body += make_section("Detailed Metrics", metrics_table, section_id="metrics")

    # --- N=1 disclaimer ---
    disclaimer = (
        '<div style="margin:16px 0;padding:12px 16px;border-left:3px solid '
        f'{ACCENT_AMBER};background:rgba(245,158,11,0.06);border-radius:4px;'
        f'font-size:0.85rem;color:{TEXT_SECONDARY}">'
        '<strong>N=1 retrospective case study.</strong> '
        'All detection metrics are descriptive, not inferential. '
        'The model was trained and evaluated on a single patient\'s data. '
        'Validation requires an external multi-patient cohort.'
        '</div>'
    )
    body += disclaimer

    # --- Extra CSS for this report's custom elements ---
    extra_css = f"""
.summary-box {{
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid {ACCENT_AMBER};
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}}
.summary-box strong {{
    color: {TEXT_PRIMARY};
}}
.detection-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 16px 0;
}}
.detection-card {{
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    background: var(--bg-surface);
}}
.detection-card.detected {{
    border-color: {ACCENT_RED};
    background: rgba(239, 68, 68, 0.08);
}}"""

    return wrap_html(
        title="Foundation Model Forecasting",
        body_content=body,
        report_id="foundation",
        subtitle="Amazon Chronos-2 + ARIMA Statistical Baseline - Oura Ring Biometrics, Post-HSCT",
        extra_css=extra_css,
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> int:
    """Run all foundation model analyses and generate report."""
    print("=" * 70)
    print("  FOUNDATION MODEL TIME SERIES ANALYSIS")
    print(f"  Model: {CHRONOS_MODEL}")
    print(f"  Known event: {KNOWN_EVENT_DATE}")
    print(f"  Ruxolitinib: {RUXOLITINIB_START}")
    print("=" * 70)

    total_start = time.time()
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics: dict[str, Any] = {
        "generated_at": generated_at,
        "run_timestamp": generated_at,
        "model": CHRONOS_MODEL,
        "known_event": KNOWN_EVENT_DATE_STR,
        "ruxolitinib_start": RUXOLITINIB_START,
    }

    # --- Load data ---
    try:
        data = load_data()
        nightly = build_nightly_series(data)
        hourly = build_hourly_hr(data)
        if not nightly.empty:
            metrics["data_range"] = {
                "start": str(nightly["date"].min()),
                "end": str(nightly["date"].max()),
                "n_nights": int(len(nightly)),
            }
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        traceback.print_exc()
        return 1

    # --- Load Chronos ---
    pipeline = None
    try:
        pipeline = load_chronos_pipeline()
    except Exception as e:
        chronos_error = f"{type(e).__name__}: {e}"
        metrics["chronos_available"] = False
        metrics["chronos_error"] = chronos_error
        print(f"[WARN] Chronos loading failed: {chronos_error}")
        print("[WARN] Continuing with ARIMA/statistical baseline only")
    else:
        metrics["chronos_available"] = True

    figures: list[tuple[str, go.Figure]] = []

    # --- Analysis 1: Nightly RMSSD + HR ---
    if pipeline is None:
        print("[WARN] Skipping nightly Chronos analysis: Chronos pipeline unavailable")
        chronos_nightly = {}
    else:
        try:
            chronos_nightly = run_chronos_nightly(pipeline, nightly, metrics)
            for sname, label, unit in [("rmssd", "RMSSD", "ms"), ("hr", "Heart Rate", "bpm")]:
                if sname in chronos_nightly:
                    r = chronos_nightly[sname]
                    fig = create_forecast_figure(
                        title=f"Chronos-2 Nightly {label} Forecast",
                        context_dates=r["context_dates"],
                        context_values=r["context"],
                        forecast_dates=r["forecast_dates"],
                        actual_values=r["actual"],
                        forecast=r["forecast"],
                        series_label=label,
                        unit=unit,
                        outside_90pi=r["outside_90pi"],
                    )
                    figures.append((f"nightly_{sname}", fig))
        except Exception as e:
            print(f"[ERROR] Nightly Chronos analysis failed: {e}")
            traceback.print_exc()
            chronos_nightly = {}

    # --- Analysis 2: Hourly HR ---
    hourly_hr_results = {}
    if pipeline is None:
        print("[WARN] Skipping hourly HR Chronos analysis: Chronos pipeline unavailable")
    else:
        try:
            hourly_hr_results = run_chronos_hourly_hr(pipeline, hourly, metrics)
            fig_hr = create_hourly_hr_figure(hourly_hr_results)
            figures.append(("hourly_hr", fig_hr))
        except Exception as e:
            print(f"[ERROR] Hourly HR analysis failed: {e}")
            traceback.print_exc()

    # --- Analysis 3: Statistical baseline + ensemble ---
    try:
        stat_results = run_statistical_baseline(nightly, metrics)
        ensemble_df = compute_ensemble_consensus(
            chronos_nightly, stat_results, nightly, metrics
        )
        fig_ens = create_ensemble_heatmap(ensemble_df)
        figures.append(("ensemble", fig_ens))
    except Exception as e:
        print(f"[ERROR] Statistical baseline / ensemble failed: {e}")
        traceback.print_exc()
        stat_results = {}
        ensemble_df = pd.DataFrame()

    # --- Analysis 4: Feb 9 retrospective ---
    feb9_results = {}
    if pipeline is None:
        print("[WARN] Skipping Feb 9 Chronos retrospective: Chronos pipeline unavailable")
    else:
        try:
            feb9_results = run_feb9_retrospective(pipeline, nightly, metrics)
            for sname, label, unit in [("rmssd", "RMSSD", "ms"), ("hr", "Heart Rate", "bpm")]:
                if sname in feb9_results:
                    r = feb9_results[sname]
                    fig = create_forecast_figure(
                        title=f"Feb 9 Retrospective: {label}",
                        context_dates=[],
                        context_values=np.array([]),
                        forecast_dates=r["actual_dates"],
                        actual_values=r["actual"],
                        forecast=r["forecast"],
                        series_label=label,
                        unit=unit,
                        outside_90pi=r["outside_90"],
                    )
                    # Add context tail
                    ctx_tail = r["context"][-14:]
                    ctx_dates = pd.date_range(
                        end=pd.Timestamp(r["actual_dates"][0]) - pd.Timedelta(days=1),
                        periods=len(ctx_tail),
                    )
                    fig.add_trace(go.Scatter(
                        x=ctx_dates, y=ctx_tail,
                        mode="lines+markers",
                        name="Context (last 14 nights)",
                        line=dict(color=COLOR_PRE, width=2),
                        marker=dict(size=3), opacity=0.6,
                    ))
                    figures.append((f"feb9_{sname}", fig))
        except Exception as e:
            print(f"[ERROR] Feb 9 retrospective failed: {e}")
            traceback.print_exc()

    # --- Analysis 5: Ruxolitinib ---
    rux_results = {}
    if pipeline is None:
        print("[WARN] Skipping ruxolitinib Chronos analysis: Chronos pipeline unavailable")
    else:
        try:
            rux_results = run_ruxolitinib_analysis(pipeline, nightly, metrics)
            for sname, label, unit in [("rmssd", "RMSSD", "ms"), ("hr", "Heart Rate", "bpm")]:
                fig_rux = create_ruxolitinib_figure(rux_results, sname, label, unit)
                figures.append((f"rux_{sname}", fig_rux))
        except Exception as e:
            print(f"[ERROR] Ruxolitinib analysis failed: {e}")
            traceback.print_exc()

    # --- Generate outputs ---
    total_elapsed = time.time() - total_start
    metrics["total_runtime_s"] = round(total_elapsed, 1)

    print(f"\n[REPORT] Total runtime: {total_elapsed:.1f}s")
    print(f"[REPORT] Generating {len(figures)} charts...")

    # Generate HTML
    try:
        html = generate_simple_html_report(figures, metrics, feb9_results)
        HTML_OUTPUT.write_text(html, encoding="utf-8")
        print(f"[REPORT] HTML report saved: {HTML_OUTPUT}")
    except Exception as e:
        print(f"[ERROR] HTML generation failed: {e}")
        traceback.print_exc()
        raise SystemExit(1) from e

    # Save JSON metrics
    try:
        # Clean metrics for JSON serialization
        def _json_clean(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (pd.Timestamp, datetime, date)):
                return str(obj)
            return obj

        clean_metrics = json.loads(json.dumps(metrics, default=_json_clean))
        JSON_OUTPUT.write_text(
            json.dumps(clean_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[REPORT] JSON metrics saved: {JSON_OUTPUT}")
    except Exception as e:
        print(f"[ERROR] JSON save failed: {e}")
        traceback.print_exc()
        raise SystemExit(1) from e

    # Summary
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print(f"  Runtime: {total_elapsed:.1f}s")
    print(f"  Charts: {len(figures)}")
    print(f"  HTML: {HTML_OUTPUT}")
    print(f"  JSON: {JSON_OUTPUT}")

    # Feb 9 detection summary
    for sname in ["rmssd", "hr"]:
        key = f"feb9_retro_{sname}"
        if key in metrics:
            m = metrics[key]
            status = "DETECTED" if m.get("feb9_detected") else "not detected"
            print(f"  Feb 9 ({sname.upper()}): {status} (residual: {m.get('feb9_residual', 'N/A')})")

    ens = metrics.get("ensemble_consensus", {})
    if "high_confidence_anomaly_dates" in ens:
        print(f"  HIGH confidence anomaly dates: {ens['high_confidence_anomaly_dates']}")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
