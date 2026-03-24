#!/usr/bin/env python3
"""
ML-Powered Anomaly Detection Engine for Oura Ring Biometric Streams

Five complementary algorithms detect anomalies in post-HSCT biometrics:
  1. Matrix Profile (stumpy) - discord detection in time series
  2. Isolation Forest (sklearn) - multivariate outlier detection
  3. LSTM Autoencoder (PyTorch) - reconstruction error anomaly detection
  4. Statistical Process Control - Shewhart / CUSUM / EWMA charts
  5. tsfresh Feature Extraction - night clustering for outlier identification

See config.py for patient details and known event dates.

Output:
  - Interactive HTML report: reports/anomaly_detection_report.html
  - JSON metrics: reports/anomaly_detection_metrics.json

Usage:
    python analysis/analyze_oura_anomalies.py
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Keep STUMPY on CPU in environments with a CUDA-enabled numba build but no driver.
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATABASE_PATH,
    REPORTS_DIR,
    KNOWN_EVENT_DATE,
    TREATMENT_START,
    ESC_RMSSD_DEFICIENCY,
    NOCTURNAL_HR_ELEVATED,
    POPULATION_RMSSD_MEDIAN,
    HSCT_RMSSD_RANGE,
    DATA_START,
    BASELINE_DAYS,
)

# Config dates are datetime.date objects; this script uses string keys throughout
KNOWN_EVENT_DATE = str(KNOWN_EVENT_DATE)
TREATMENT_START_STR = str(TREATMENT_START)
from _hardening import safe_divide
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    ACCENT_BLUE,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    C_HR,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "anomaly_detection_report.html"
JSON_OUTPUT = REPORTS_DIR / "anomaly_detection_metrics.json"

# ---------------------------------------------------------------------------
# Clinical context
# ---------------------------------------------------------------------------
CONTAMINATION_RATE = 0.1  # Isolation Forest expected anomaly rate
HSCT_TYPICAL_RMSSD = HSCT_RMSSD_RANGE  # (25, 40) ms range — imported from config

# Anomaly detection parameters
MP_WINDOW_SIZES = [3, 5, 7]
IF_CONTAMINATION = 0.1
LSTM_WINDOW = 7
LSTM_EPOCHS = 100
LSTM_LR = 0.001
LSTM_SEED = 42
SPC_SIGMA_WARN = 2
SPC_SIGMA_ACTION = 3
CUSUM_K = 0.5  # allowance (in sigma units)
CUSUM_H = 4.0  # decision threshold (in sigma units)
EWMA_LAMBDA = 0.2  # smoothing factor
AGREEMENT_THRESHOLD = 0.6  # fraction of methods required for "sufficient" agreement
METHOD_SCORE_THRESHOLD = 0.5  # per-method score threshold for counting support


# ===========================================================================
# DATA LOADING
# ===========================================================================


def load_data() -> dict[str, pd.DataFrame]:
    """Load all Oura tables into DataFrames."""
    print("[DATA] Loading biometric data from database...")

    if not Path(DATABASE_PATH).exists():
        print(f"Database not found: {DATABASE_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    try:
        # HRV epochs (5-min intervals)
        hrv = pd.read_sql_query(
            "SELECT timestamp, rmssd FROM oura_hrv ORDER BY timestamp", conn
        )
        hrv["date"] = pd.to_datetime(hrv["timestamp"]).dt.date.astype(str)
        hrv["rmssd"] = pd.to_numeric(hrv["rmssd"], errors="coerce")

        # Heart rate (continuous)
        hr = pd.read_sql_query(
            "SELECT timestamp, bpm FROM oura_heart_rate ORDER BY timestamp", conn
        )
        hr["date"] = pd.to_datetime(hr["timestamp"]).dt.date.astype(str)
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

        # Sleep nightly summaries
        sleep = pd.read_sql_query(
            "SELECT date, score, hrv_average FROM oura_sleep ORDER BY date", conn
        )
        sleep["score"] = pd.to_numeric(sleep["score"], errors="coerce")
        sleep["hrv_average"] = pd.to_numeric(sleep["hrv_average"], errors="coerce")

        # SpO2
        spo2 = pd.read_sql_query(
            "SELECT date, spo2_average FROM oura_spo2 WHERE spo2_average > 0 ORDER BY date",
            conn,
        )
        spo2["spo2_average"] = pd.to_numeric(spo2["spo2_average"], errors="coerce")

        # Readiness
        readiness = pd.read_sql_query(
            """SELECT date, score as readiness_score, temperature_deviation,
                      recovery_index, resting_heart_rate
               FROM oura_readiness ORDER BY date""",
            conn,
        )
        for col in readiness.columns:
            if col != "date":
                readiness[col] = pd.to_numeric(readiness[col], errors="coerce")
    finally:
        conn.close()

    data = {
        "hrv": hrv,
        "hr": hr,
        "sleep_periods": sleep_periods,
        "sleep": sleep,
        "spo2": spo2,
        "readiness": readiness,
    }

    for name, df in data.items():
        if df.empty:
            print(f"  {name}: 0 rows (empty)")
        else:
            print(
                f"  {name}: {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}"
            )

    return data


def build_daily_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a unified daily feature matrix from all biometric sources."""
    print("[DATA] Building daily feature matrix...")

    # Aggregate HRV by day
    hrv_daily = (
        data["hrv"]
        .groupby("date")
        .agg(
            mean_rmssd=("rmssd", "mean"),
            std_rmssd=("rmssd", "std"),
            min_rmssd=("rmssd", "min"),
            max_rmssd=("rmssd", "max"),
            median_rmssd=("rmssd", "median"),
            hrv_count=("rmssd", "count"),
        )
        .reset_index()
    )

    # Aggregate HR by day
    hr_daily = (
        data["hr"]
        .groupby("date")
        .agg(
            mean_hr=("bpm", "mean"),
            std_hr=("bpm", "std"),
            min_hr=("bpm", "min"),
            max_hr=("bpm", "max"),
            median_hr=("bpm", "median"),
            hr_count=("bpm", "count"),
        )
        .reset_index()
    )

    # Sleep periods - already per-night
    sp = data["sleep_periods"].copy()
    sp["total_hours"] = sp["total_sleep_duration"] / 3600
    sp["rem_pct"] = (
        sp["rem_sleep_duration"] / sp["total_sleep_duration"].replace(0, np.nan) * 100
    )
    sp["deep_pct"] = (
        sp["deep_sleep_duration"] / sp["total_sleep_duration"].replace(0, np.nan) * 100
    )
    sp["light_pct"] = (
        sp["light_sleep_duration"] / sp["total_sleep_duration"].replace(0, np.nan) * 100
    )
    sp_features = sp[
        [
            "date",
            "average_hrv",
            "average_heart_rate",
            "average_breath",
            "efficiency",
            "lowest_heart_rate",
            "total_hours",
            "rem_pct",
            "deep_pct",
            "light_pct",
        ]
    ].copy()

    # Sleep score
    sleep_score = data["sleep"][["date", "score"]].rename(
        columns={"score": "sleep_score"}
    )

    # SpO2
    spo2 = data["spo2"][["date", "spo2_average"]].copy()

    # Readiness
    readiness = data["readiness"][
        ["date", "readiness_score", "temperature_deviation", "recovery_index"]
    ].copy()

    # Merge all on date
    # Start with HRV daily (most complete)
    all_dates = sorted(
        set(
            hrv_daily["date"].tolist()
            + hr_daily["date"].tolist()
            + sp_features["date"].tolist()
        )
    )
    daily = pd.DataFrame({"date": all_dates})

    daily = daily.merge(hrv_daily, on="date", how="left")
    daily = daily.merge(hr_daily, on="date", how="left")
    daily = daily.merge(sp_features, on="date", how="left")
    daily = daily.merge(sleep_score, on="date", how="left")
    daily = daily.merge(spo2, on="date", how="left")
    daily = daily.merge(readiness, on="date", how="left")

    daily = daily.sort_values("date").reset_index(drop=True)
    data_start_str = str(DATA_START)
    daily = daily[daily["date"] >= data_start_str].reset_index(drop=True)
    print(f"  Daily matrix: {len(daily)} days x {len(daily.columns)} features")
    if not daily.empty:
        print(f"  Date range: {daily['date'].iloc[0]} to {daily['date'].iloc[-1]}")
    else:
        print("  WARNING: Daily feature matrix is empty")

    return daily


# ===========================================================================
# METHOD 1: MATRIX PROFILE (stumpy)
# ===========================================================================


def run_matrix_profile(daily: pd.DataFrame) -> dict[str, Any]:
    """Matrix Profile discord detection on nightly RMSSD and HR time series."""
    print("\n" + "=" * 70)
    print("[1/5] MATRIX PROFILE (stumpy)")
    print("=" * 70)
    t0 = time.time()

    try:
        import stumpy
    except ImportError:
        logging.warning("stumpy not installed — Matrix Profile will be skipped")
        return {
            "method": "Matrix Profile",
            "anomalies_by_window": {},
            "feb9_detected": False,
            "runtime_s": 0,
            "error": "stumpy not installed",
        }

    results: dict[str, Any] = {
        "method": "Matrix Profile",
        "anomalies_by_window": {},
        "feb9_detected": False,
        "runtime_s": 0,
    }

    # Prepare signals - use sleep_periods average_hrv and average_heart_rate
    # (more complete than aggregated HRV epochs)
    hrv_series = daily[["date", "mean_rmssd"]].dropna(subset=["mean_rmssd"]).copy()
    hr_series = daily[["date", "mean_hr"]].dropna(subset=["mean_hr"]).copy()

    signals = {
        "RMSSD": (hrv_series, "mean_rmssd"),
        "HR": (hr_series, "mean_hr"),
    }

    for signal_name, (df, col) in signals.items():
        signal = df[col].values.astype(np.float64)
        dates = df["date"].values

        if len(signal) < 10:
            print(f"  {signal_name}: too few points ({len(signal)}), skipping")
            continue

        for m in MP_WINDOW_SIZES:
            if len(signal) <= m:
                continue

            key = f"{signal_name}_w{m}"
            print(f"  Computing MP for {signal_name}, window={m}, n={len(signal)}...")

            try:
                mp = stumpy.stump(signal, m=m)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logging.warning(
                    "stumpy Matrix Profile unavailable in this environment; "
                    "skipping remaining Matrix Profile analysis (%s)",
                    error_msg,
                )
                results["runtime_s"] = round(time.time() - t0, 2)
                results["error"] = error_msg
                return results
            mp_values = mp[:, 0].astype(float)

            # Top anomalies (discords)
            top_k = min(5, len(mp_values))
            discord_indices = np.argsort(mp_values)[-top_k:][::-1]

            anomaly_dates = []
            anomaly_scores = []
            for idx in discord_indices:
                if idx < len(dates):
                    anomaly_dates.append(str(dates[idx]))
                    anomaly_scores.append(float(mp_values[idx]))

            # Normalize scores to 0-1 range
            mp_min, mp_max = mp_values.min(), mp_values.max()
            if mp_max > mp_min:
                norm_scores = (mp_values - mp_min) / (mp_max - mp_min)
            else:
                norm_scores = np.zeros_like(mp_values)

            # Per-date scores
            date_scores = {}
            for i, d in enumerate(dates[: len(norm_scores)]):
                date_scores[str(d)] = float(norm_scores[i])

            results["anomalies_by_window"][key] = {
                "window_size": m,
                "signal": signal_name,
                "top_discords": list(zip(anomaly_dates, anomaly_scores)),
                "date_scores": date_scores,
                "raw_mp": mp_values.tolist(),
            }

            # Check if Feb 9 is in top anomalies
            feb9_in_top = KNOWN_EVENT_DATE in anomaly_dates
            if feb9_in_top:
                results["feb9_detected"] = True
                rank = anomaly_dates.index(KNOWN_EVENT_DATE) + 1
                print(f"    ** Feb 9 detected as #{rank} discord! **")
            else:
                # Check if any date +-1 day from Feb 9 is there
                adjacent = [
                    str(
                        (
                            datetime.strptime(KNOWN_EVENT_DATE, "%Y-%m-%d")
                            + timedelta(days=d)
                        ).date()
                    )
                    for d in [-1, 0, 1]
                ]
                for adj in adjacent:
                    if adj in anomaly_dates:
                        results["feb9_detected"] = True
                        rank = anomaly_dates.index(adj) + 1
                        print(
                            f"    ** Feb 8-10 region detected as #{rank} discord ({adj})! **"
                        )
                        break

            print(f"    Top 3 discords: {anomaly_dates[:3]}")

    results["runtime_s"] = round(time.time() - t0, 2)
    print(f"  Matrix Profile complete in {results['runtime_s']}s")
    return results


# ===========================================================================
# METHOD 2: ISOLATION FOREST
# ===========================================================================


def run_isolation_forest(daily: pd.DataFrame) -> dict[str, Any]:
    """Multivariate Isolation Forest anomaly detection."""
    print("\n" + "=" * 70)
    print("[2/5] ISOLATION FOREST (sklearn)")
    print("=" * 70)
    t0 = time.time()

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    results: dict[str, Any] = {
        "method": "Isolation Forest",
        "anomalies": [],
        "feb9_detected": False,
        "runtime_s": 0,
        "date_scores": {},
    }

    # Select features for multivariate detection
    feature_cols = [
        "mean_rmssd",
        "std_rmssd",
        "mean_hr",
        "std_hr",
        "min_hr",
        "average_breath",
        "efficiency",
        "lowest_heart_rate",
        "total_hours",
        "rem_pct",
        "deep_pct",
        "spo2_average",
        "readiness_score",
        "temperature_deviation",
        "recovery_index",
        "sleep_score",
    ]

    available_cols = [c for c in feature_cols if c in daily.columns]
    df = daily[["date"] + available_cols].copy()

    # Forward-fill then drop remaining NaN rows
    df[available_cols] = df[available_cols].ffill().bfill()
    df = df.dropna(subset=available_cols)

    if len(df) < 10:
        print("  Not enough complete rows for Isolation Forest")
        results["error"] = "Insufficient data"
        return results

    print(f"  Using {len(available_cols)} features, {len(df)} days")
    print(f"  Features: {available_cols}")

    X = df[available_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_estimators=200,
        max_samples="auto",
    )
    labels = clf.fit_predict(X_scaled)
    scores = clf.decision_function(X_scaled)

    # Normalize scores: lower = more anomalous; invert so higher = more anomalous
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        norm_scores = 1.0 - (scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.zeros_like(scores)

    anomaly_mask = labels == -1
    anomaly_dates = df["date"].values[anomaly_mask]
    anomaly_scores = norm_scores[anomaly_mask]

    # Sort by score (most anomalous first)
    sort_idx = np.argsort(anomaly_scores)[::-1]
    anomaly_dates = anomaly_dates[sort_idx]
    anomaly_scores = anomaly_scores[sort_idx]

    results["anomalies"] = [
        {"date": str(d), "score": float(s)}
        for d, s in zip(anomaly_dates, anomaly_scores)
    ]

    # All date scores
    for i, d in enumerate(df["date"].values):
        results["date_scores"][str(d)] = float(norm_scores[i])

    # Check Feb 9
    feb9_adjacent = [
        str(
            (datetime.strptime(KNOWN_EVENT_DATE, "%Y-%m-%d") + timedelta(days=d)).date()
        )
        for d in [-1, 0, 1]
    ]
    for d in feb9_adjacent:
        if d in [str(x) for x in anomaly_dates]:
            results["feb9_detected"] = True
            break

    n_anomalies = len(anomaly_dates)
    print(
        f"  Detected {n_anomalies} anomalous days ({safe_divide(n_anomalies, len(df)) * 100:.1f}%)"
    )
    print(f"  Top 5 anomalies: {[str(d) for d in anomaly_dates[:5]]}")
    if results["feb9_detected"]:
        print("  ** Feb 8-10 region detected as anomaly! **")
    else:
        print(
            f"  Feb 9 NOT detected (score: {results['date_scores'].get(KNOWN_EVENT_DATE, 'N/A')})"
        )

    # Feature importances (via isolation depth proxy)
    results["n_anomalies"] = int(n_anomalies)
    results["total_days"] = int(len(df))
    results["features_used"] = available_cols
    results["runtime_s"] = round(time.time() - t0, 2)
    print(f"  Isolation Forest complete in {results['runtime_s']}s")
    return results


# ===========================================================================
# METHOD 3: LSTM AUTOENCODER
# ===========================================================================


def run_lstm_autoencoder(daily: pd.DataFrame) -> dict[str, Any]:
    """LSTM Autoencoder for sequence-based anomaly detection."""
    print("\n" + "=" * 70)
    print("[3/5] LSTM AUTOENCODER (PyTorch)")
    print("=" * 70)
    t0 = time.time()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logging.warning("torch not installed — LSTM Autoencoder will be skipped")
        return {
            "method": "LSTM Autoencoder",
            "anomalies": [],
            "feb9_detected": False,
            "runtime_s": 0,
            "date_scores": {},
            "training_loss": [],
            "error": "torch not installed",
        }

    results: dict[str, Any] = {
        "method": "LSTM Autoencoder",
        "anomalies": [],
        "feb9_detected": False,
        "runtime_s": 0,
        "date_scores": {},
        "training_loss": [],
    }

    # Fix weight init and shuffled minibatch order so CPU training is reproducible.
    random.seed(LSTM_SEED)
    np.random.seed(LSTM_SEED)
    torch.manual_seed(LSTM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(LSTM_SEED)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(LSTM_SEED)
    print(f"  Deterministic seed: {LSTM_SEED}")

    # Features for LSTM
    feature_cols = [
        "mean_rmssd",
        "mean_hr",
        "lowest_heart_rate",
        "efficiency",
        "deep_pct",
        "rem_pct",
        "readiness_score",
        "recovery_index",
    ]
    available_cols = [c for c in feature_cols if c in daily.columns]
    df = daily[["date"] + available_cols].copy()
    df[available_cols] = df[available_cols].ffill().bfill()
    df = df.dropna(subset=available_cols)

    if len(df) < LSTM_WINDOW + 5:
        print("  Not enough data for LSTM Autoencoder")
        results["error"] = "Insufficient data"
        return results

    n_features = len(available_cols)
    print(f"  Using {n_features} features, {len(df)} days, window={LSTM_WINDOW}")

    # Normalize
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[available_cols].values)

    # Create sliding windows
    windows = []
    window_dates = []  # date of the LAST day in each window
    for i in range(len(X_all) - LSTM_WINDOW + 1):
        windows.append(X_all[i : i + LSTM_WINDOW])
        window_dates.append(df["date"].iloc[i + LSTM_WINDOW - 1])

    X_tensor = torch.FloatTensor(np.array(windows))
    print(f"  Created {len(windows)} sliding windows of shape {X_tensor.shape}")

    # Find stable period (exclude Feb 8-10 from training)
    exclude_dates = set()
    for delta in range(-2, 3):
        d = (
            datetime.strptime(KNOWN_EVENT_DATE, "%Y-%m-%d") + timedelta(days=delta)
        ).strftime("%Y-%m-%d")
        exclude_dates.add(d)

    train_mask = []
    for wd in window_dates:
        # Exclude if any day in window overlaps with event
        idx = window_dates.index(wd)
        window_start_idx = idx  # window goes from here - LSTM_WINDOW + 1 to here
        day_range_start = max(0, len(df) - len(windows) + idx - LSTM_WINDOW + 1)
        # Simpler: exclude if the window end date is near the event
        train_mask.append(str(wd) not in exclude_dates)

    train_indices = [i for i, m in enumerate(train_mask) if m]
    test_indices = list(range(len(windows)))

    X_train = X_tensor[train_indices]
    print(f"  Training on {len(X_train)} windows (excluding event region)")

    # Define LSTM Autoencoder
    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features: int, hidden_dim: int = 32, latent_dim: int = 16):
            super().__init__()
            self.encoder_lstm1 = nn.LSTM(n_features, hidden_dim, batch_first=True)
            self.encoder_lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
            self.decoder_lstm1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
            self.decoder_lstm2 = nn.LSTM(hidden_dim, n_features, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Encode
            enc1, _ = self.encoder_lstm1(x)
            enc2, _ = self.encoder_lstm2(enc1)
            # Decode
            dec1, _ = self.decoder_lstm1(enc2)
            dec2, _ = self.decoder_lstm2(dec1)
            return dec2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = LSTMAutoencoder(n_features, hidden_dim=32, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train, X_train)
    loader = DataLoader(
        dataset,
        batch_size=min(16, len(X_train)),
        shuffle=True,
        generator=loader_generator,
    )

    # Train
    model.train()
    for epoch in range(LSTM_EPOCHS):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        results["training_loss"].append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{LSTM_EPOCHS}, Loss: {avg_loss:.6f}")

    # Compute reconstruction errors for all windows
    model.eval()
    with torch.no_grad():
        X_all_tensor = X_tensor.to(device)
        reconstructed = model(X_all_tensor)
        errors = torch.mean(
            (X_all_tensor.cpu() - reconstructed.cpu()) ** 2, dim=(1, 2)
        ).numpy()

    # Normalize errors to 0-1
    err_min, err_max = errors.min(), errors.max()
    if err_max > err_min:
        norm_errors = (errors - err_min) / (err_max - err_min)
    else:
        norm_errors = np.zeros_like(errors)

    # Threshold: mean + 2*std of training errors
    train_errors = errors[train_indices]
    threshold = train_errors.mean() + 2 * train_errors.std()
    print(f"  Reconstruction error threshold: {threshold:.6f}")
    print(
        f"  Training error: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}"
    )

    anomaly_mask = errors > threshold
    for i in test_indices:
        d = str(window_dates[i])
        results["date_scores"][d] = float(norm_errors[i])
        if anomaly_mask[i]:
            results["anomalies"].append(
                {
                    "date": d,
                    "score": float(norm_errors[i]),
                    "raw_error": float(errors[i]),
                }
            )

    # Sort anomalies by score
    results["anomalies"].sort(key=lambda x: x["score"], reverse=True)

    # Check Feb 9
    feb9_adjacent = [
        str(
            (datetime.strptime(KNOWN_EVENT_DATE, "%Y-%m-%d") + timedelta(days=d)).date()
        )
        for d in [-1, 0, 1]
    ]
    for a in results["anomalies"]:
        if a["date"] in feb9_adjacent:
            results["feb9_detected"] = True
            break

    print(f"  Detected {len(results['anomalies'])} anomalous windows")
    print(f"  Top 5: {[a['date'] for a in results['anomalies'][:5]]}")
    if results["feb9_detected"]:
        print("  ** Feb 8-10 region detected as anomaly! **")

    results["threshold"] = float(threshold)
    results["runtime_s"] = round(time.time() - t0, 2)
    print(f"  LSTM Autoencoder complete in {results['runtime_s']}s")
    return results


# ===========================================================================
# METHOD 4: STATISTICAL PROCESS CONTROL
# ===========================================================================


def run_spc(daily: pd.DataFrame) -> dict[str, Any]:
    """Statistical Process Control: Shewhart, CUSUM, and EWMA charts."""
    print("\n" + "=" * 70)
    print("[4/5] STATISTICAL PROCESS CONTROL (Shewhart / CUSUM / EWMA)")
    print("=" * 70)
    t0 = time.time()

    results: dict[str, Any] = {
        "method": "Statistical Process Control",
        "shewhart": {},
        "cusum": {},
        "ewma": {},
        "feb9_detected": False,
        "runtime_s": 0,
        "date_scores": {},
    }

    metrics = {
        "RMSSD": ("mean_rmssd", "lower"),  # low is bad
        "HR": ("mean_hr", "upper"),  # high is bad
    }

    # Add SpO2 if available
    if (
        "spo2_average" in daily.columns
        and daily["spo2_average"].notna().sum() > BASELINE_DAYS
    ):
        metrics["SpO2"] = ("spo2_average", "lower")

    for metric_name, (col, direction) in metrics.items():
        series = daily[["date", col]].dropna(subset=[col]).copy()
        if len(series) < BASELINE_DAYS + 5:
            print(f"  {metric_name}: insufficient data, skipping")
            continue

        dates = series["date"].values
        values = series[col].values.astype(float)

        # Baseline from first N days
        # Note: SPC baseline represents patient's observed values during early
        # January 2026, not a healthy population reference. Control limits reflect
        # individual patient variability.
        baseline = values[:BASELINE_DAYS]
        mu = baseline.mean()
        sigma = baseline.std()
        if sigma < 1e-10:
            sigma = 1.0  # avoid division by zero

        print(
            f"  {metric_name}: baseline mu={mu:.2f}, sigma={sigma:.2f} (from first {BASELINE_DAYS} days)"
        )

        # --- Shewhart Chart ---
        z_scores = (values - mu) / sigma
        shewhart_violations_2s = []
        shewhart_violations_3s = []

        for i, (d, z) in enumerate(zip(dates, z_scores)):
            if direction == "upper" and z > SPC_SIGMA_WARN:
                shewhart_violations_2s.append(str(d))
            elif direction == "lower" and z < -SPC_SIGMA_WARN:
                shewhart_violations_2s.append(str(d))
            if direction == "upper" and z > SPC_SIGMA_ACTION:
                shewhart_violations_3s.append(str(d))
            elif direction == "lower" and z < -SPC_SIGMA_ACTION:
                shewhart_violations_3s.append(str(d))

        results["shewhart"][metric_name] = {
            "mu": float(mu),
            "sigma": float(sigma),
            "warn_2sigma": shewhart_violations_2s,
            "action_3sigma": shewhart_violations_3s,
            "z_scores": {str(d): float(z) for d, z in zip(dates, z_scores)},
        }

        # --- CUSUM ---
        k = CUSUM_K * sigma
        h = CUSUM_H * sigma
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        cusum_alarms = []

        for i in range(1, len(values)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + (values[i] - mu) - k)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - (values[i] - mu) - k)

            if direction == "upper" and cusum_pos[i] > h:
                cusum_alarms.append(str(dates[i]))
            elif direction == "lower" and cusum_neg[i] > h:
                cusum_alarms.append(str(dates[i]))
            elif direction == "both" and (cusum_pos[i] > h or cusum_neg[i] > h):
                cusum_alarms.append(str(dates[i]))

        results["cusum"][metric_name] = {
            "k": float(k),
            "h": float(h),
            "alarms": cusum_alarms,
            "cusum_pos": {str(d): float(v) for d, v in zip(dates, cusum_pos)},
            "cusum_neg": {str(d): float(v) for d, v in zip(dates, cusum_neg)},
        }

        # --- EWMA ---
        ewma = np.zeros(len(values))
        ewma[0] = values[0]
        lam = EWMA_LAMBDA

        for i in range(1, len(values)):
            ewma[i] = lam * values[i] + (1 - lam) * ewma[i - 1]

        # EWMA control limits widen then stabilize
        ewma_sigma = sigma * np.sqrt(
            (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * np.arange(1, len(values) + 1)))
        )
        ucl = mu + SPC_SIGMA_ACTION * ewma_sigma
        lcl = mu - SPC_SIGMA_ACTION * ewma_sigma

        ewma_violations = []
        for i, (d, e) in enumerate(zip(dates, ewma)):
            if direction == "upper" and e > ucl[i]:
                ewma_violations.append(str(d))
            elif direction == "lower" and e < lcl[i]:
                ewma_violations.append(str(d))

        results["ewma"][metric_name] = {
            "lambda": float(lam),
            "violations": ewma_violations,
            "ewma_values": {str(d): float(v) for d, v in zip(dates, ewma)},
            "ucl": {str(d): float(v) for d, v in zip(dates, ucl)},
            "lcl": {str(d): float(v) for d, v in zip(dates, lcl)},
        }

        # Per-date composite SPC score (based on z-score magnitude)
        for i, d in enumerate(dates):
            ds = str(d)
            z_mag = abs(z_scores[i]) / SPC_SIGMA_ACTION  # normalize so 3-sigma = 1.0
            z_mag = min(z_mag, 1.5)  # cap at 1.5
            if ds not in results["date_scores"]:
                results["date_scores"][ds] = float(z_mag)
            else:
                results["date_scores"][ds] = max(
                    results["date_scores"][ds], float(z_mag)
                )

    # Check Feb 9 across all SPC methods
    for metric_name in metrics:
        for method_key in ["shewhart", "cusum", "ewma"]:
            method_data = results[method_key].get(metric_name, {})
            alarm_lists = []
            if method_key == "shewhart":
                alarm_lists = [
                    method_data.get("warn_2sigma", []),
                    method_data.get("action_3sigma", []),
                ]
            elif method_key == "cusum":
                alarm_lists = [method_data.get("alarms", [])]
            elif method_key == "ewma":
                alarm_lists = [method_data.get("violations", [])]

            for alarms in alarm_lists:
                for adj_d in range(-1, 2):
                    d = str(
                        (
                            datetime.strptime(KNOWN_EVENT_DATE, "%Y-%m-%d")
                            + timedelta(days=adj_d)
                        ).date()
                    )
                    if d in alarms:
                        results["feb9_detected"] = True

    # Print summary
    for metric_name in metrics:
        sh = results["shewhart"].get(metric_name, {})
        cu = results["cusum"].get(metric_name, {})
        ew = results["ewma"].get(metric_name, {})
        print(
            f"  {metric_name}: Shewhart 2s={len(sh.get('warn_2sigma', []))}, "
            f"3s={len(sh.get('action_3sigma', []))}, "
            f"CUSUM={len(cu.get('alarms', []))}, "
            f"EWMA={len(ew.get('violations', []))}"
        )

    if results["feb9_detected"]:
        print("  ** Feb 8-10 region detected by SPC! **")

    results["runtime_s"] = round(time.time() - t0, 2)
    print(f"  SPC complete in {results['runtime_s']}s")
    return results


# ===========================================================================
# METHOD 5: TSFRESH FEATURE EXTRACTION + CLUSTERING
# ===========================================================================


def run_tsfresh_clustering(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Extract features per night using tsfresh, then cluster to find outlier nights."""
    print("\n" + "=" * 70)
    print("[5/5] TSFRESH FEATURE EXTRACTION + CLUSTERING")
    print("=" * 70)
    t0 = time.time()

    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
    except ImportError:
        logging.warning("tsfresh not installed — tsfresh clustering will be skipped")
        return {
            "method": "tsfresh + DBSCAN",
            "anomalies": [],
            "feb9_detected": False,
            "runtime_s": 0,
            "date_scores": {},
            "error": "tsfresh not installed",
        }

    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    results: dict[str, Any] = {
        "method": "tsfresh + DBSCAN",
        "anomalies": [],
        "feb9_detected": False,
        "runtime_s": 0,
        "date_scores": {},
    }

    # Use HR data grouped by date
    hr = data["hr"][["date", "timestamp", "bpm"]].copy()
    hr["timestamp_dt"] = pd.to_datetime(hr["timestamp"])
    hr = hr.sort_values("timestamp_dt")

    # Get unique dates with enough data
    date_counts = hr.groupby("date").size()
    valid_dates = date_counts[date_counts >= 30].index.tolist()
    print(f"  Nights with >= 30 HR readings: {len(valid_dates)}")

    if len(valid_dates) < 10:
        print("  Not enough nights for tsfresh analysis")
        results["error"] = "Insufficient data"
        results["runtime_s"] = round(time.time() - t0, 2)
        return results

    # Build tsfresh input: id = date, sort = timestamp, value = bpm
    # Use numeric IDs for efficiency
    date_to_id = {d: i for i, d in enumerate(sorted(valid_dates))}
    id_to_date = {i: d for d, i in date_to_id.items()}

    ts_df = hr[hr["date"].isin(valid_dates)].copy()
    ts_df["night_id"] = ts_df["date"].map(date_to_id)
    ts_df = ts_df[["night_id", "timestamp_dt", "bpm"]].rename(
        columns={"timestamp_dt": "time", "bpm": "value"}
    )
    ts_df = ts_df.sort_values(["night_id", "time"]).reset_index(drop=True)

    print("  Extracting tsfresh features (MinimalFCParameters)...")

    try:
        features = extract_features(
            ts_df,
            column_id="night_id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
            n_jobs=1,
        )
    except Exception as e:
        print(f"  tsfresh extraction failed: {e}")
        results["error"] = str(e)
        results["runtime_s"] = round(time.time() - t0, 2)
        return results

    # Remove columns with all NaN
    features = features.dropna(axis=1, how="all")
    # Fill remaining NaN with 0
    features = features.fillna(0)

    print(f"  Extracted {features.shape[1]} features for {features.shape[0]} nights")

    if features.shape[0] < 5 or features.shape[1] < 2:
        print("  Not enough features extracted")
        results["error"] = "Insufficient features"
        results["runtime_s"] = round(time.time() - t0, 2)
        return results

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    # DBSCAN clustering
    db = DBSCAN(eps=3.0, min_samples=3)
    labels = db.fit_predict(X_scaled)

    outlier_mask = labels == -1
    outlier_indices = features.index[outlier_mask].tolist()
    outlier_dates = [id_to_date[i] for i in outlier_indices]

    # Compute distance from cluster centroid as anomaly score
    if len(X_scaled) > 0:
        centroid = (
            X_scaled[~outlier_mask].mean(axis=0)
            if (~outlier_mask).any()
            else X_scaled.mean(axis=0)
        )
        distances = np.linalg.norm(X_scaled - centroid, axis=1)
        d_min, d_max = distances.min(), distances.max()
        if d_max > d_min:
            norm_distances = (distances - d_min) / (d_max - d_min)
        else:
            norm_distances = np.zeros_like(distances)

        for idx in features.index:
            d = id_to_date[idx]
            pos = list(features.index).index(idx)
            results["date_scores"][d] = float(norm_distances[pos])
    else:
        norm_distances = np.array([])

    for od in outlier_dates:
        idx = date_to_id[od]
        pos = list(features.index).index(idx)
        results["anomalies"].append(
            {
                "date": od,
                "score": float(norm_distances[pos])
                if len(norm_distances) > pos
                else 0.0,
            }
        )

    results["anomalies"].sort(key=lambda x: x["score"], reverse=True)

    # Check Feb 9
    feb9_adjacent = set()
    for delta in range(-1, 2):
        feb9_adjacent.add(
            str(
                (
                    datetime.strptime(KNOWN_EVENT_DATE, "%Y-%m-%d")
                    + timedelta(days=delta)
                ).date()
            )
        )
    for a in results["anomalies"]:
        if a["date"] in feb9_adjacent:
            results["feb9_detected"] = True
            break

    n_clusters = len(set(labels) - {-1})
    print(f"  DBSCAN found {n_clusters} clusters, {len(outlier_dates)} outlier nights")
    print(f"  Outlier dates: {outlier_dates}")
    if results["feb9_detected"]:
        print("  ** Feb 8-10 region detected as outlier! **")

    results["n_clusters"] = n_clusters
    results["n_outliers"] = len(outlier_dates)
    results["n_features_extracted"] = int(features.shape[1])
    results["runtime_s"] = round(time.time() - t0, 2)
    print(f"  tsfresh complete in {results['runtime_s']}s")
    return results


# ===========================================================================
# VALIDATION & SCORING
# ===========================================================================


def validate_feb9(all_results: dict[str, dict]) -> dict[str, Any]:
    """Validate all methods against the known Feb 9 acute event."""
    print("\n" + "=" * 70)
    print("FEB 9 VALIDATION SUMMARY")
    print("=" * 70)

    validation = {
        "known_event": KNOWN_EVENT_DATE,
        "methods_tested": len(all_results),
        "methods_detected": 0,
        "detection_details": {},
    }

    for method_name, result in all_results.items():
        detected = result.get("feb9_detected", False)
        if detected:
            validation["methods_detected"] += 1

        # Get score for Feb 9
        score = None
        date_scores = result.get("date_scores", {})
        if KNOWN_EVENT_DATE in date_scores:
            score = date_scores[KNOWN_EVENT_DATE]
        elif isinstance(result.get("anomalies_by_window"), dict):
            # Matrix Profile: check across windows
            for window_key, wd in result["anomalies_by_window"].items():
                ds = wd.get("date_scores", {})
                if KNOWN_EVENT_DATE in ds:
                    score = max(score or 0, ds[KNOWN_EVENT_DATE])

        validation["detection_details"][method_name] = {
            "detected": detected,
            "score": score,
        }

        status = "DETECTED" if detected else "MISSED"
        score_str = f"{score:.3f}" if score is not None else "N/A"
        print(f"  {method_name}: {status} (score: {score_str})")

    agreement_rate = validation["methods_detected"] / max(
        validation["methods_tested"], 1
    )
    validation["method_agreement_rate"] = agreement_rate
    validation["study_type"] = "N=1 retrospective case study"
    print(
        f"\n  Method agreement: {validation['methods_detected']}/{validation['methods_tested']} ({agreement_rate:.0%}) [N=1 case study]"
    )

    return validation


def compute_ensemble_scores(
    daily: pd.DataFrame, all_results: dict[str, dict]
) -> pd.DataFrame:
    """Compute ensemble anomaly scores across all methods."""
    print("\n[ENSEMBLE] Computing combined anomaly scores...")

    dates = sorted(daily["date"].unique())
    ensemble = pd.DataFrame({"date": dates})

    method_weights = {
        "matrix_profile": 0.20,
        "isolation_forest": 0.25,
        "lstm_autoencoder": 0.20,
        "spc": 0.20,
        "tsfresh": 0.15,
    }

    for method_name, result in all_results.items():
        col = f"score_{method_name}"
        date_scores = result.get("date_scores", {})

        # Matrix Profile: merge across windows
        if method_name == "matrix_profile" and "anomalies_by_window" in result:
            merged_scores = {}
            for window_key, wd in result.get("anomalies_by_window", {}).items():
                for d, s in wd.get("date_scores", {}).items():
                    merged_scores[d] = max(merged_scores.get(d, 0), s)
            date_scores = merged_scores

        scores = [date_scores.get(d, np.nan) for d in dates]
        ensemble[col] = scores

    # Compute weighted ensemble score
    score_cols = [c for c in ensemble.columns if c.startswith("score_")]
    for _, row in ensemble.iterrows():
        total_weight = 0
        weighted_sum = 0
        for col in score_cols:
            method = col.replace("score_", "")
            if pd.notna(row[col]):
                w = method_weights.get(method, 0.15)
                weighted_sum += row[col] * w
                total_weight += w
        idx = ensemble.index[ensemble["date"] == row["date"]]
        if total_weight > 0:
            ensemble.loc[idx, "ensemble_score"] = weighted_sum / total_weight
        else:
            ensemble.loc[idx, "ensemble_score"] = np.nan

    # Rank
    ensemble["rank"] = ensemble["ensemble_score"].rank(ascending=False, method="min")

    # Flag top anomalies
    threshold = ensemble["ensemble_score"].quantile(0.9)
    ensemble["is_anomaly"] = ensemble["ensemble_score"] >= threshold

    n_anomalies = ensemble["is_anomaly"].sum()
    print(f"  Ensemble threshold (90th pct): {threshold:.3f}")
    print(f"  Anomalous days: {n_anomalies}")

    top5 = ensemble.nlargest(5, "ensemble_score").dropna(subset=["ensemble_score"])
    print("  Top 5 anomaly days:")
    for _, row in top5.iterrows():
        rank_str = str(int(row["rank"])) if pd.notna(row["rank"]) else "N/A"
        print(
            f"    {row['date']}: ensemble={row['ensemble_score']:.3f}, rank={rank_str}"
        )

    return ensemble


# ===========================================================================
# HTML REPORT GENERATION
# ===========================================================================


def generate_html_report(
    daily: pd.DataFrame,
    ensemble: pd.DataFrame,
    all_results: dict[str, dict],
    validation: dict[str, Any],
) -> str:
    """Generate interactive HTML report with Plotly visualizations."""
    print("\n[REPORT] Generating interactive HTML report...")

    # --- Fig 1: Ensemble Anomaly Timeline ---
    fig1 = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Ensemble Anomaly Score",
            "Nightly RMSSD (ms)",
            "Nightly Mean Heart Rate (bpm)",
            "Method Comparison",
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    dates_dt = pd.to_datetime(ensemble["date"])
    event_date = pd.Timestamp(KNOWN_EVENT_DATE)

    # Row 1: Ensemble score
    colors = [ACCENT_RED if a else ACCENT_BLUE for a in ensemble["is_anomaly"]]
    fig1.add_trace(
        go.Bar(
            x=dates_dt,
            y=ensemble["ensemble_score"],
            marker_color=colors,
            marker_line_width=0,
            opacity=0.85,
            name="Ensemble Score",
            hovertemplate="<b>%{x|%b %d}</b><br>Ensemble Score: %{y:.3f}<br>Status: "
            + "<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # Threshold line
    threshold = ensemble["ensemble_score"].quantile(0.9)
    fig1.add_shape(
        type="line",
        x0=dates_dt.min(),
        x1=dates_dt.max(),
        y0=threshold,
        y1=threshold,
        line=dict(color=ACCENT_RED, dash="dashdot", width=1.5),
        row=1,
        col=1,
    )
    fig1.add_annotation(
        x=dates_dt.max(),
        y=threshold,
        text=f"90th percentile ({threshold:.2f})",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=9, color=ACCENT_RED),
        row=1,
        col=1,
    )

    # Row 2: RMSSD with gradient fill
    rmssd_dates = pd.to_datetime(daily["date"])
    # Gradient fill under RMSSD
    fig1.add_trace(
        go.Scatter(
            x=rmssd_dates,
            y=daily["mean_rmssd"],
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.08)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=rmssd_dates,
            y=daily["mean_rmssd"],
            mode="lines+markers",
            name="RMSSD",
            line=dict(color=ACCENT_PURPLE, width=2.5),
            marker=dict(size=3, color=ACCENT_PURPLE),
            hovertemplate="<b>%{x|%b %d}</b><br>RMSSD: %{y:.1f} ms<extra></extra>",
        ),
        row=2,
        col=1,
    )
    # Clinical thresholds
    fig1.add_shape(
        type="line",
        x0=rmssd_dates.min(),
        x1=rmssd_dates.max(),
        y0=ESC_RMSSD_DEFICIENCY,
        y1=ESC_RMSSD_DEFICIENCY,
        line=dict(color=ACCENT_RED, dash="dot", width=1),
        row=2,
        col=1,
    )
    fig1.add_annotation(
        x=rmssd_dates.max(),
        y=ESC_RMSSD_DEFICIENCY,
        text="Parasympathetic deficiency (ESC/NASPE 1996; Shaffer & Ginsberg 2017)",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=9, color=ACCENT_RED),
        row=2,
        col=1,
    )

    # Row 3: Heart Rate with gradient fill
    fig1.add_trace(
        go.Scatter(
            x=rmssd_dates,
            y=daily["mean_hr"],
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.06)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=rmssd_dates,
            y=daily["mean_hr"],
            mode="lines+markers",
            name="HR",
            line=dict(color=C_HR, width=2.5),
            marker=dict(size=3, color=C_HR),
            hovertemplate="<b>%{x|%b %d}</b><br>HR: %{y:.0f} bpm<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig1.add_shape(
        type="line",
        x0=rmssd_dates.min(),
        x1=rmssd_dates.max(),
        y0=NOCTURNAL_HR_ELEVATED,
        y1=NOCTURNAL_HR_ELEVATED,
        line=dict(color=ACCENT_AMBER, dash="dot", width=1),
        row=3,
        col=1,
    )
    fig1.add_annotation(
        x=rmssd_dates.max(),
        y=NOCTURNAL_HR_ELEVATED,
        text=f"Nocturnal concern ({NOCTURNAL_HR_ELEVATED} bpm)",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=9, color=ACCENT_AMBER),
        row=3,
        col=1,
    )

    # Anomaly highlighting: red markers on RMSSD and HR for flagged days
    anomaly_dates_set = set(ensemble.loc[ensemble["is_anomaly"], "date"].values)
    daily_with_flags = daily.copy()
    daily_with_flags["_is_anomaly"] = daily_with_flags["date"].isin(anomaly_dates_set)
    anom_rows = daily_with_flags[daily_with_flags["_is_anomaly"]]

    if len(anom_rows) > 0:
        anom_rmssd_dates = pd.to_datetime(anom_rows["date"])
        # Merge ensemble scores for hover info
        anom_scores = []
        for d in anom_rows["date"].values:
            row_match = ensemble[ensemble["date"] == d]
            anom_scores.append(
                float(row_match["ensemble_score"].iloc[0]) if len(row_match) > 0 else 0
            )

        # RMSSD anomaly markers - large and unmistakable
        fig1.add_trace(
            go.Scatter(
                x=anom_rmssd_dates,
                y=anom_rows["mean_rmssd"],
                mode="markers",
                marker=dict(
                    size=14,
                    color=ACCENT_RED,
                    symbol="diamond",
                    line=dict(width=2, color="#FFFFFF"),
                    opacity=0.95,
                ),
                name="Anomaly",
                customdata=list(zip(anom_scores)),
                hovertemplate="<b>ANOMALY</b><br>%{x|%b %d}<br>RMSSD: %{y:.1f} ms<br>Score: %{customdata[0]:.3f}<extra></extra>",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        # HR anomaly markers
        fig1.add_trace(
            go.Scatter(
                x=anom_rmssd_dates,
                y=anom_rows["mean_hr"],
                mode="markers",
                marker=dict(
                    size=14,
                    color=ACCENT_RED,
                    symbol="diamond",
                    line=dict(width=2, color="#FFFFFF"),
                    opacity=0.95,
                ),
                name="Anomaly",
                customdata=list(zip(anom_scores)),
                hovertemplate="<b>ANOMALY</b><br>%{x|%b %d}<br>HR: %{y:.0f} bpm<br>Score: %{customdata[0]:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # Shade top-3 discord regions across RMSSD and HR rows
    mp_result = all_results.get("matrix_profile", {})
    discord_dates_seen: set[str] = set()
    for _wk, wd in mp_result.get("anomalies_by_window", {}).items():
        for d_str, _score in wd.get("top_discords", [])[:3]:
            if d_str not in discord_dates_seen:
                discord_dates_seen.add(d_str)
                d0 = pd.Timestamp(d_str) - pd.Timedelta(hours=12)
                d1 = pd.Timestamp(d_str) + pd.Timedelta(hours=12)
                for shade_row in [2, 3]:
                    yref = f"y{shade_row} domain"
                    fig1.add_shape(
                        type="rect",
                        x0=d0,
                        x1=d1,
                        y0=0,
                        y1=1,
                        yref=yref,
                        fillcolor="rgba(239, 68, 68, 0.12)",
                        line=dict(width=0),
                        layer="below",
                    )

    # Row 4: Method comparison heatmap-style scatter
    method_names = []
    method_scores_by_date = {}
    score_cols = [c for c in ensemble.columns if c.startswith("score_")]
    for col in score_cols:
        mname = col.replace("score_", "").replace("_", " ").title()
        method_names.append(mname)
        for _, row in ensemble.iterrows():
            d = row["date"]
            if d not in method_scores_by_date:
                method_scores_by_date[d] = {}
            method_scores_by_date[d][mname] = row[col]

    for i, mname in enumerate(method_names):
        y_vals = [
            method_scores_by_date.get(d, {}).get(mname, None) for d in ensemble["date"]
        ]
        fig1.add_trace(
            go.Scatter(
                x=dates_dt,
                y=[i] * len(dates_dt),
                mode="markers",
                marker=dict(
                    size=10,
                    color=y_vals,
                    colorscale="RdYlGn_r",
                    cmin=0,
                    cmax=1,
                    showscale=(i == 0),
                    colorbar=dict(title="Score", len=0.2, y=0.12) if i == 0 else None,
                ),
                name=mname,
                hovertemplate=f"{mname}<br>Date: %{{x|%Y-%m-%d}}<br>Score: %{{marker.color:.3f}}<extra></extra>",
            ),
            row=4,
            col=1,
        )

    fig1.update_yaxes(
        tickvals=list(range(len(method_names))),
        ticktext=method_names,
        row=4,
        col=1,
    )

    # Add event marker and treatment start on all rows
    treatment_date = pd.Timestamp(TREATMENT_START)
    for row_n in range(1, 5):
        yref = "y domain" if row_n == 1 else f"y{row_n} domain"
        # Known event vertical line
        fig1.add_shape(
            type="line",
            x0=event_date,
            x1=event_date,
            y0=0,
            y1=1,
            yref=yref,
            line=dict(color=ACCENT_RED, width=2, dash="dash"),
        )
        # Treatment start vertical line
        fig1.add_shape(
            type="line",
            x0=treatment_date,
            x1=treatment_date,
            y0=0,
            y1=1,
            yref=yref,
            line=dict(color=ACCENT_GREEN, width=2, dash="dash"),
        )
        if row_n == 1:
            fig1.add_annotation(
                x=event_date,
                y=1,
                yref=yref,
                text="Acute event Feb 9",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-30,
                font=dict(color=ACCENT_RED, size=11),
            )
            fig1.add_annotation(
                x=treatment_date,
                y=0.9,
                yref=yref,
                text=f"Ruxolitinib start {TREATMENT_START}",
                showarrow=True,
                arrowhead=2,
                ax=-40,
                ay=-30,
                font=dict(color=ACCENT_GREEN, size=11),
            )

    fig1.update_layout(
        height=1100,
        margin=dict(l=80, r=30, t=120, b=40),
        showlegend=True,
        legend=dict(orientation="h", y=-0.03, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    fig1.update_yaxes(title_text="Score", zeroline=False, row=1, col=1)
    fig1.update_yaxes(title_text="RMSSD (ms)", zeroline=False, row=2, col=1)
    fig1.update_yaxes(title_text="HR (bpm)", zeroline=False, row=3, col=1)
    # Subtle gridlines and crosshair spikes on all axes
    for row_n in range(1, 5):
        fig1.update_xaxes(
            tickformat="%d %b",
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikedash="dot",
            row=row_n,
            col=1,
        )
        fig1.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikedash="dot",
            row=row_n,
            col=1,
        )

    # --- Fig 2: SPC Charts ---
    spc_result = all_results.get("spc", {})
    fig2 = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Shewhart Chart: RMSSD",
            "CUSUM Chart: RMSSD",
            "EWMA Chart: RMSSD",
        ),
    )

    # Shewhart for RMSSD
    sh_rmssd = spc_result.get("shewhart", {}).get("RMSSD", {})
    if sh_rmssd:
        z_data = sh_rmssd.get("z_scores", {})
        z_dates = sorted(z_data.keys())
        z_vals = [z_data[d] for d in z_dates]
        z_dates_dt = pd.to_datetime(z_dates)

        mu = sh_rmssd["mu"]
        sigma = sh_rmssd["sigma"]

        rmssd_actual = [mu + z * sigma for z in z_vals]
        ucl_3s = mu + 3 * sigma
        lcl_3s = mu - 3 * sigma
        ucl_2s = mu + 2 * sigma
        lcl_2s = mu - 2 * sigma

        # Fill between 2-sigma limits (warning zone)
        fig2.add_trace(
            go.Scatter(
                x=z_dates_dt,
                y=[ucl_2s] * len(z_dates_dt),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=z_dates_dt,
                y=[lcl_2s] * len(z_dates_dt),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.06)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        # Identify out-of-control points (beyond 3-sigma)
        ooc_dates = []
        ooc_vals = []
        normal_dates = []
        normal_vals = []
        for dt, val in zip(z_dates_dt, rmssd_actual):
            if val > ucl_3s or val < lcl_3s:
                ooc_dates.append(dt)
                ooc_vals.append(val)
            else:
                normal_dates.append(dt)
                normal_vals.append(val)

        # Main trace
        fig2.add_trace(
            go.Scatter(
                x=z_dates_dt,
                y=rmssd_actual,
                mode="lines",
                name="RMSSD",
                line=dict(color=ACCENT_PURPLE, width=2.5),
                hovertemplate="<b>%{x|%b %d}</b><br>RMSSD: %{y:.1f} ms<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Out-of-control points highlighted dramatically
        if ooc_dates:
            fig2.add_trace(
                go.Scatter(
                    x=ooc_dates,
                    y=ooc_vals,
                    mode="markers",
                    name="Out of Control",
                    marker=dict(
                        size=12,
                        color=ACCENT_RED,
                        symbol="x-thin-open",
                        line=dict(width=3, color=ACCENT_RED),
                    ),
                    hovertemplate="<b>OUT OF CONTROL</b><br>%{x|%b %d}<br>RMSSD: %{y:.1f} ms<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Control limits with refined dash patterns
        fig2.add_shape(
            type="line",
            x0=z_dates_dt.min(),
            x1=z_dates_dt.max(),
            y0=ucl_2s,
            y1=ucl_2s,
            line=dict(color=ACCENT_AMBER, dash="dashdot", width=1),
            row=1,
            col=1,
        )
        fig2.add_shape(
            type="line",
            x0=z_dates_dt.min(),
            x1=z_dates_dt.max(),
            y0=lcl_2s,
            y1=lcl_2s,
            line=dict(color=ACCENT_AMBER, dash="dashdot", width=1),
            row=1,
            col=1,
        )
        fig2.add_shape(
            type="line",
            x0=z_dates_dt.min(),
            x1=z_dates_dt.max(),
            y0=ucl_3s,
            y1=ucl_3s,
            line=dict(color=ACCENT_RED, dash="longdash", width=1.5),
            row=1,
            col=1,
        )
        fig2.add_shape(
            type="line",
            x0=z_dates_dt.min(),
            x1=z_dates_dt.max(),
            y0=lcl_3s,
            y1=lcl_3s,
            line=dict(color=ACCENT_RED, dash="longdash", width=1.5),
            row=1,
            col=1,
        )
        # Center line
        fig2.add_shape(
            type="line",
            x0=z_dates_dt.min(),
            x1=z_dates_dt.max(),
            y0=mu,
            y1=mu,
            line=dict(color=ACCENT_GREEN, dash="solid", width=1.5),
            row=1,
            col=1,
        )
        # Limit labels
        fig2.add_annotation(
            x=z_dates_dt.max(),
            y=ucl_3s,
            text="UCL (3s)",
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color=ACCENT_RED),
            row=1,
            col=1,
        )
        fig2.add_annotation(
            x=z_dates_dt.max(),
            y=lcl_3s,
            text="LCL (3s)",
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color=ACCENT_RED),
            row=1,
            col=1,
        )
        fig2.add_annotation(
            x=z_dates_dt.max(),
            y=mu,
            text=f"CL ({mu:.1f})",
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color=ACCENT_GREEN),
            row=1,
            col=1,
        )

    # CUSUM for RMSSD
    cu_rmssd = spc_result.get("cusum", {}).get("RMSSD", {})
    if cu_rmssd:
        cp_data = cu_rmssd.get("cusum_pos", {})
        cn_data = cu_rmssd.get("cusum_neg", {})
        cu_dates = sorted(cp_data.keys())
        cu_dates_dt = pd.to_datetime(cu_dates)
        cp_vals = [cp_data[d] for d in cu_dates]
        cn_vals = [cn_data[d] for d in cu_dates]
        h_val = cu_rmssd["h"]

        # Fill under CUSUM+ (subtle)
        fig2.add_trace(
            go.Scatter(
                x=cu_dates_dt,
                y=cp_vals,
                mode="none",
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.06)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=cu_dates_dt,
                y=cp_vals,
                mode="lines",
                name="CUSUM+",
                line=dict(color=ACCENT_BLUE, width=2.5),
                hovertemplate="<b>%{x|%b %d}</b><br>CUSUM+: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=cu_dates_dt,
                y=cn_vals,
                mode="lines",
                name="CUSUM-",
                line=dict(color=ACCENT_RED, width=2.5),
                hovertemplate="<b>%{x|%b %d}</b><br>CUSUM-: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig2.add_shape(
            type="line",
            x0=cu_dates_dt.min(),
            x1=cu_dates_dt.max(),
            y0=h_val,
            y1=h_val,
            line=dict(color=ACCENT_RED, dash="longdash", width=1.5),
            row=2,
            col=1,
        )
        fig2.add_annotation(
            x=cu_dates_dt.max(),
            y=h_val,
            text=f"Decision boundary (h={h_val:.1f})",
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color=ACCENT_RED),
            row=2,
            col=1,
        )

    # EWMA for RMSSD
    ew_rmssd = spc_result.get("ewma", {}).get("RMSSD", {})
    if ew_rmssd:
        ew_data = ew_rmssd.get("ewma_values", {})
        ucl_data = ew_rmssd.get("ucl", {})
        lcl_data = ew_rmssd.get("lcl", {})
        ew_dates = sorted(ew_data.keys())
        ew_dates_dt = pd.to_datetime(ew_dates)
        ew_vals = [ew_data[d] for d in ew_dates]
        ucl_vals = [ucl_data[d] for d in ew_dates]
        lcl_vals = [lcl_data[d] for d in ew_dates]

        # Fill between UCL and LCL (control band)
        fig2.add_trace(
            go.Scatter(
                x=ew_dates_dt,
                y=ucl_vals,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=3,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=ew_dates_dt,
                y=lcl_vals,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(139, 92, 246, 0.06)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=3,
            col=1,
        )

        fig2.add_trace(
            go.Scatter(
                x=ew_dates_dt,
                y=ew_vals,
                mode="lines",
                name="EWMA",
                line=dict(color=ACCENT_PURPLE, width=2.5),
                hovertemplate="<b>%{x|%b %d}</b><br>EWMA: %{y:.1f} ms<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=ew_dates_dt,
                y=ucl_vals,
                mode="lines",
                name="UCL",
                line=dict(color=ACCENT_RED, dash="longdash", width=1.5),
            ),
            row=3,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=ew_dates_dt,
                y=lcl_vals,
                mode="lines",
                name="LCL",
                line=dict(color=ACCENT_RED, dash="longdash", width=1.5),
            ),
            row=3,
            col=1,
        )

    # Event + treatment markers
    for row_n in range(1, 4):
        yref = "y domain" if row_n == 1 else f"y{row_n} domain"
        fig2.add_shape(
            type="line",
            x0=event_date,
            x1=event_date,
            y0=0,
            y1=1,
            yref=yref,
            line=dict(color=ACCENT_RED, width=2, dash="dash"),
        )
        fig2.add_shape(
            type="line",
            x0=treatment_date,
            x1=treatment_date,
            y0=0,
            y1=1,
            yref=yref,
            line=dict(color=ACCENT_GREEN, width=2, dash="dash"),
        )

    fig2.update_layout(
        height=800,
        showlegend=True,
        margin=dict(l=60, r=40, t=120, b=50),
        hovermode="x unified",
    )
    # Subtle gridlines, crosshair spikes, zeroline off, date formatting on all rows
    for spc_row in range(1, 4):
        fig2.update_xaxes(
            tickformat="%d %b",
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikedash="dot",
            row=spc_row,
            col=1,
        )
        fig2.update_yaxes(
            zeroline=False,
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikedash="dot",
            row=spc_row,
            col=1,
        )
    fig2.update_yaxes(title_text="RMSSD (ms)", row=1, col=1)
    fig2.update_yaxes(title_text="CUSUM", row=2, col=1)
    fig2.update_yaxes(title_text="EWMA (ms)", row=3, col=1)
    fig2.add_annotation(
        text="Note: SPC baseline = patient's early January 2026 values, not population reference.",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.08,
        showarrow=False,
        font=dict(size=10, color=TEXT_TERTIARY),
    )

    # --- Fig 3: LSTM Training Loss ---
    lstm_result = all_results.get("lstm_autoencoder", {})
    fig3 = go.Figure()
    if lstm_result.get("training_loss"):
        training_loss = lstm_result["training_loss"]
        epochs = list(range(1, len(training_loss) + 1))
        # Subtle fill below the curve
        fig3.add_trace(
            go.Scatter(
                x=epochs,
                y=training_loss,
                mode="none",
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.08)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=epochs,
                y=training_loss,
                mode="lines",
                name="Training Loss",
                line=dict(color=ACCENT_BLUE, width=2.5, shape="spline"),
                hovertemplate="<b>Epoch %{x}</b><br>MSE Loss: %{y:.6f}<extra></extra>",
            )
        )
        # Final loss annotation
        final_loss = training_loss[-1]
        fig3.add_annotation(
            x=epochs[-1],
            y=final_loss,
            text=f"Final: {final_loss:.6f}",
            showarrow=True,
            arrowhead=2,
            ax=-60,
            ay=-25,
            font=dict(size=10, color=ACCENT_BLUE),
        )
        fig3.update_layout(
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            height=400,
            margin=dict(l=60, r=30, t=50, b=40),
            hovermode="x unified",
        )
        fig3.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.15)",
            spikedash="dot",
        )
        fig3.update_yaxes(
            zeroline=False,
            gridcolor="rgba(255,255,255,0.05)",
        )

    # --- Build HTML ---
    validation_html = _build_validation_html(validation, all_results, daily)
    clinical_html = _build_clinical_interpretation(daily, ensemble, all_results)

    # Executive summary stats
    n_anomaly_days = (
        int(ensemble["is_anomaly"].sum()) if "is_anomaly" in ensemble.columns else 0
    )
    n_high = (
        int((ensemble["ensemble_score"] >= 0.7).sum())
        if "ensemble_score" in ensemble.columns
        else 0
    )
    n_moderate = (
        int(
            (
                (ensemble["ensemble_score"] >= 0.4) & (ensemble["ensemble_score"] < 0.7)
            ).sum()
        )
        if "ensemble_score" in ensemble.columns
        else 0
    )
    feb9_detected = validation.get("methods_detected", 0) > 0
    feb9_agreement = validation.get("method_agreement_rate", 0)

    # Post-ruxolitinib changes
    tx_date_str = str(TREATMENT_START)
    post_tx = (
        ensemble[ensemble["date"] >= tx_date_str]
        if "date" in ensemble.columns
        else pd.DataFrame()
    )
    pre_tx = (
        ensemble[ensemble["date"] < tx_date_str]
        if "date" in ensemble.columns
        else pd.DataFrame()
    )
    post_tx_mean = (
        post_tx["ensemble_score"].mean() if len(post_tx) > 0 else float("nan")
    )
    pre_tx_mean = pre_tx["ensemble_score"].mean() if len(pre_tx) > 0 else float("nan")
    post_tx_change = ""
    if len(post_tx) > 0 and not pd.isna(pre_tx_mean) and not pd.isna(post_tx_mean):
        diff_pct = (
            ((post_tx_mean - pre_tx_mean) / pre_tx_mean * 100)
            if pre_tx_mean != 0
            else 0
        )
        direction = "lower" if diff_pct < 0 else "higher"
        post_tx_change = f"{abs(diff_pct):.0f}% {direction} mean anomaly score after ruxolitinib ({len(post_tx)} days)"
    else:
        post_tx_change = "Insufficient data after treatment start"

    feb9_status = "normal" if feb9_detected else "critical"
    feb9_label = (
        f"{validation.get('methods_detected', 0)}/{validation.get('methods_tested', 0)}"
    )

    anomaly_days_status = "warning" if n_anomaly_days > 5 else "info"
    anomaly_days_label = "Elevated" if n_anomaly_days > 5 else ""
    feb9_status_label = "Missed" if feb9_status == "critical" else "Detected"
    high_status = "critical" if n_high > 0 else "normal"
    high_label = "Detected" if n_high > 0 else ""
    moderate_status = "warning" if n_moderate > 0 else "normal"
    moderate_label = "Detected" if n_moderate > 0 else ""

    # --- Build body using theme components ---
    body = ""

    # KPI row
    body += make_kpi_row(
        make_kpi_card(
            "Anomaly Days",
            n_anomaly_days,
            "",
            status=anomaly_days_status,
            detail="Above 90th percentile",
            status_label=anomaly_days_label,
        ),
        make_kpi_card(
            "Feb 9 Detection",
            feb9_label,
            "",
            status=feb9_status,
            detail=f"{feb9_agreement:.0%} method agreement (N=1)",
            status_label=feb9_status_label,
        ),
        make_kpi_card(
            "High Severity",
            n_high,
            "",
            status=high_status,
            detail="Score >= 0.7",
            status_label=high_label,
        ),
        make_kpi_card(
            "Moderate",
            n_moderate,
            "",
            status=moderate_status,
            detail="Score 0.4-0.7",
            status_label=moderate_label,
        ),
    )

    # Post-tx narrative
    body += f'<div class="odt-narrative">{post_tx_change}</div>'

    # Data period info
    body += (
        f'<div class="odt-narrative">'
        f"<strong>Data period:</strong> {daily['date'].iloc[0]} to {daily['date'].iloc[-1]} "
        f"({len(daily)} days) &middot; "
        f"<strong>Known acute event:</strong> {KNOWN_EVENT_DATE}"
        f"</div>"
    )

    # Validation section
    body += make_section("Validation: February 9 Event", validation_html)

    # Charts using fig.to_html() with include_plotlyjs=False
    fig1_html = fig1.to_html(include_plotlyjs=False, full_html=False)
    fig2_html = fig2.to_html(include_plotlyjs=False, full_html=False)
    fig3_html = fig3.to_html(include_plotlyjs=False, full_html=False)

    body += make_section("Anomaly Timeline", fig1_html)
    body += make_section("Statistical Process Control (SPC)", fig2_html)
    body += make_section("LSTM Autoencoder - Training Curve", fig3_html)

    # Top anomalies table
    body += make_section(
        "Top Anomalies (Ensemble Scoring)", _build_top_anomalies_table(ensemble, daily)
    )

    # Clinical interpretation
    body += make_section("Clinical Interpretation", clinical_html)

    # Method summary
    body += make_section("Method Summary", _build_method_summary(all_results))

    # Extra CSS for script-specific classes
    extra_css = f"""
.anomaly-high {{ background: rgba(239, 68, 68, 0.15); }}
.method-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 2px;
}}
.badge-detected {{ background: rgba(16, 185, 129, 0.2); color: {ACCENT_GREEN}; }}
.badge-missed {{ background: rgba(239, 68, 68, 0.2); color: {ACCENT_RED}; }}
.detected {{ color: {ACCENT_GREEN}; }}
.missed {{ color: {ACCENT_RED}; }}
.clinical-callout {{
    padding: 16px 20px;
    background: rgba(245, 158, 11, 0.08);
    border-left: 3px solid {ACCENT_AMBER};
    border-radius: 0 8px 8px 0;
    margin-bottom: 16px;
}}
"""

    html_content = wrap_html(
        title="Anomaly Detection",
        body_content=body,
        report_id="anomalies",
        subtitle="ML-powered analysis: Matrix Profile, Isolation Forest, LSTM Autoencoder, SPC, tsfresh+DBSCAN",
        extra_css=extra_css,
    )

    return html_content


def _build_validation_html(
    validation: dict, all_results: dict, daily: pd.DataFrame
) -> str:
    """Build validation section HTML."""
    n_detected = validation["methods_detected"]
    n_total = validation["methods_tested"]
    agreement_rate = validation["method_agreement_rate"]

    badges = []
    for method, detail in validation["detection_details"].items():
        label = method.replace("_", " ").title()
        if detail["detected"]:
            badges.append(f'<span class="method-badge badge-detected">{label}</span>')
        else:
            badges.append(f'<span class="method-badge badge-missed">{label}</span>')

    methods_status = "normal" if agreement_rate >= AGREEMENT_THRESHOLD else "critical"
    methods_label = "Sufficient" if agreement_rate >= AGREEMENT_THRESHOLD else "Low"
    agreement_status = "normal" if agreement_rate >= AGREEMENT_THRESHOLD else "warning"
    agreement_label = "Strong" if agreement_rate >= AGREEMENT_THRESHOLD else "Weak"

    kpi_row = make_kpi_row(
        make_kpi_card(
            "Methods Detected",
            f"{n_detected}/{n_total}",
            "",
            status=methods_status,
            status_label=methods_label,
        ),
        make_kpi_card(
            "Method Agreement (N=1)",
            f"{agreement_rate:.0%}",
            "",
            status=agreement_status,
            status_label=agreement_label,
        ),
    )

    event_row = daily[daily["date"] == KNOWN_EVENT_DATE]
    if not event_row.empty:
        event_rmssd = event_row["mean_rmssd"].iloc[0]
        event_hr = event_row["mean_hr"].iloc[0]
        event_readiness = event_row["readiness_score"].iloc[0]
        rmssd_txt = f"{event_rmssd:.1f} ms" if pd.notna(event_rmssd) else "N/A"
        hr_txt = f"{event_hr:.0f} bpm" if pd.notna(event_hr) else "N/A"
        ready_txt = f"{event_readiness:.0f}" if pd.notna(event_readiness) else "N/A"
    else:
        rmssd_txt, hr_txt, ready_txt = "N/A", "N/A", "N/A"

    return f"""
    {kpi_row}
    <div style="margin-top: 12px;">
        <p><strong>Detection results per method:</strong></p>
        <p>{"".join(badges)}</p>
        <p style="color: {TEXT_SECONDARY}; font-size: 0.875rem;"><em>The Feb 8-9, 2026 event is a clinically confirmed acute episode
        (nightly mean HR {hr_txt}, nightly mean RMSSD {rmssd_txt}, readiness score {ready_txt}).
        All anomaly detectors are validated against this event.</em></p>
    </div>"""


def _build_top_anomalies_table(ensemble: pd.DataFrame, daily: pd.DataFrame) -> str:
    """Build top anomalies HTML table."""
    top = ensemble.nlargest(15, "ensemble_score").copy()
    top = top.merge(
        daily[["date", "mean_rmssd", "mean_hr", "efficiency", "readiness_score"]],
        on="date",
        how="left",
    )

    rows = []
    for _, r in top.iterrows():
        is_event = r["date"] == KNOWN_EVENT_DATE
        cls = ' class="anomaly-high"' if is_event else ""
        event_mark = " **" if is_event else ""
        rmssd = f"{r['mean_rmssd']:.1f}" if pd.notna(r.get("mean_rmssd")) else "-"
        hr = f"{r['mean_hr']:.0f}" if pd.notna(r.get("mean_hr")) else "-"
        eff = f"{r['efficiency']:.0f}" if pd.notna(r.get("efficiency")) else "-"
        ready = (
            f"{r['readiness_score']:.0f}" if pd.notna(r.get("readiness_score")) else "-"
        )

        # Count how many methods flagged this date
        score_cols = [c for c in ensemble.columns if c.startswith("score_")]
        n_methods = sum(
            1
            for c in score_cols
            if pd.notna(r.get(c)) and r[c] > METHOD_SCORE_THRESHOLD
        )

        rows.append(
            f"<tr{cls}>"
            f"<td>{int(r['rank'])}</td>"
            f"<td><strong>{r['date']}{event_mark}</strong></td>"
            f"<td>{r['ensemble_score']:.3f}</td>"
            f"<td>{rmssd}</td>"
            f"<td>{hr}</td>"
            f"<td>{eff}%</td>"
            f"<td>{ready}</td>"
            f"<td>{n_methods}/{len(score_cols)}</td>"
            f"</tr>"
        )

    return f"""
        <table>
            <thead>
                <tr>
                    <th>#</th><th>Date</th><th>Ensemble Score</th><th>RMSSD (ms)</th>
                    <th>HR (bpm)</th><th>Efficiency</th><th>Readiness</th><th>Methods (score &gt; 0.5)</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
        <p style="font-size: 0.8125rem; color: {TEXT_SECONDARY};"><em>** = known acute event. Methods (score &gt; 0.5) is a soft support count and is not the same as strict Feb 9 detection validation above.</em></p>"""


def _build_method_summary(all_results: dict) -> str:
    """Build method summary table."""
    rows = []
    for method_name, result in all_results.items():
        label = method_name.replace("_", " ").title()
        runtime = result.get("runtime_s", "N/A")
        detected = result.get("feb9_detected", False)
        status = (
            '<span class="detected">Yes</span>'
            if detected
            else '<span class="missed">No</span>'
        )

        if method_name == "matrix_profile":
            detail = f"{len(result.get('anomalies_by_window', {}))} signals x windows"
        elif method_name == "isolation_forest":
            detail = f"{result.get('n_anomalies', 0)} anomalies / {result.get('total_days', 0)} days"
        elif method_name == "lstm_autoencoder":
            detail = f"{len(result.get('anomalies', []))} anomalies, threshold={result.get('threshold', 0):.4f}"
        elif method_name == "spc":
            detail = "Shewhart + CUSUM + EWMA"
        elif method_name == "tsfresh":
            detail = f"{result.get('n_features_extracted', 0)} features, {result.get('n_outliers', 0)} outliers"
        else:
            detail = ""

        error = result.get("error", "")
        if error:
            detail = f'<span class="missed">{error}</span>'

        rows.append(
            f"<tr><td>{label}</td><td>{runtime}s</td><td>{status}</td><td>{detail}</td></tr>"
        )

    return f"""
        <table>
            <thead><tr><th>Method</th><th>Runtime</th><th>Feb 9 Detected</th><th>Details</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>"""


def _build_clinical_interpretation(
    daily: pd.DataFrame,
    ensemble: pd.DataFrame,
    all_results: dict,
) -> str:
    """Build clinical interpretation with proper design-system styling."""
    top5 = ensemble.nlargest(5, "ensemble_score")
    top_dates = top5["date"].tolist()

    feb9_rank = None
    if KNOWN_EVENT_DATE in ensemble["date"].values:
        feb9_row = ensemble[ensemble["date"] == KNOWN_EVENT_DATE]
        if not feb9_row.empty:
            feb9_rank = int(feb9_row["rank"].iloc[0])

    n_methods = len(all_results)
    n_methods_detecting = sum(
        1 for r in all_results.values() if r.get("feb9_detected", False)
    )

    event_row = daily[daily["date"] == KNOWN_EVENT_DATE]
    if not event_row.empty:
        event_rmssd = event_row["mean_rmssd"].iloc[0]
        event_hr = event_row["mean_hr"].iloc[0]
        event_readiness = event_row["readiness_score"].iloc[0]
        event_recovery = (
            event_row["recovery_index"].iloc[0]
            if "recovery_index" in event_row.columns
            else np.nan
        )
    else:
        event_rmssd, event_hr, event_readiness, event_recovery = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    rmssd_txt = f"{event_rmssd:.1f}" if pd.notna(event_rmssd) else "N/A"
    hr_txt = f"{event_hr:.0f}" if pd.notna(event_hr) else "N/A"
    ready_txt = f"{event_readiness:.0f}" if pd.notna(event_readiness) else "N/A"
    recovery_txt = f"{event_recovery:.0f}" if pd.notna(event_recovery) else "N/A"

    # Baseline RMSSD from the full window
    baseline_rmssd = (
        daily["mean_rmssd"].median() if "mean_rmssd" in daily.columns else np.nan
    )
    baseline_rmssd_txt = f"{baseline_rmssd:.1f}" if pd.notna(baseline_rmssd) else "~10"
    pct_below_median = (
        (1 - baseline_rmssd / POPULATION_RMSSD_MEDIAN) * 100
        if pd.notna(baseline_rmssd)
        else 80
    )

    # Verdict banner
    verdict_html = f"""
    <div class="cs-verdict">
      <div class="cs-verdict-dot"></div>
      <div class="cs-verdict-text">
        <strong>{n_methods_detecting}/{n_methods} algorithms detected the acute event</strong> —
        ranked #{feb9_rank if feb9_rank else "N/A"} in ensemble scoring across {len(daily)} days.
      </div>
    </div>"""

    # Event biometrics as stat cards
    stats_html = f"""
    <div class="cs-stats-row">
      <div class="cs-stat">
        <div class="cs-stat-number critical">{rmssd_txt}</div>
        <div class="cs-stat-label">RMSSD (ms) — event night</div>
      </div>
      <div class="cs-stat">
        <div class="cs-stat-number critical">{hr_txt}</div>
        <div class="cs-stat-label">Mean HR (bpm) — event night</div>
      </div>
      <div class="cs-stat">
        <div class="cs-stat-number warning">{ready_txt}</div>
        <div class="cs-stat-label">Readiness — event day</div>
      </div>
      <div class="cs-stat">
        <div class="cs-stat-number warning">{recovery_txt}</div>
        <div class="cs-stat-label">Recovery index — event day</div>
      </div>
    </div>"""

    # Top anomalies as findings cards
    top_cards = []
    for i, row in top5.iterrows():
        d = row["date"]
        score = row["ensemble_score"]
        is_event = d == KNOWN_EVENT_DATE
        sev = "critical" if score >= 0.7 else "severe" if score >= 0.5 else "moderate"
        sev_label = (
            "Critical" if score >= 0.7 else "Severe" if score >= 0.5 else "Moderate"
        )
        drow = daily[daily["date"] == d]
        r_val = (
            f"{drow['mean_rmssd'].iloc[0]:.1f} ms"
            if not drow.empty and pd.notna(drow["mean_rmssd"].iloc[0])
            else "-"
        )
        h_val = (
            f"{drow['mean_hr'].iloc[0]:.0f} bpm"
            if not drow.empty and pd.notna(drow["mean_hr"].iloc[0])
            else "-"
        )
        event_tag = (
            ' <span style="color:#FCA5A5;font-size:0.6875rem">(known event)</span>'
            if is_event
            else ""
        )
        top_cards.append(f"""
        <div class="cs-finding">
          <div class="cs-finding-header">
            <span class="cs-finding-title">{d}{event_tag}</span>
            <span class="cs-sev {sev}">{sev_label}</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Ensemble score</span>
            <span class="cs-metric-val{"  critical" if score >= 0.7 else ""}">{score:.3f}</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">RMSSD</span>
            <span class="cs-metric-val">{r_val}</span>
          </div>
          <div class="cs-metric">
            <span class="cs-metric-name">Heart rate</span>
            <span class="cs-metric-val">{h_val}</span>
          </div>
        </div>""")

    findings_html = f"""
    <div class="cs-findings-grid">
      {"".join(top_cards)}
    </div>"""

    # Clinical relevance as a finding card
    relevance_html = f"""
    <div class="cs-findings-grid">
      <div class="cs-finding">
        <div class="cs-finding-header">
          <span class="cs-finding-title">Baseline RMSSD</span>
          <span class="cs-sev critical">Deficient</span>
        </div>
        <div class="cs-metric">
          <span class="cs-metric-name">Median RMSSD</span>
          <span class="cs-metric-val critical">{baseline_rmssd_txt} ms</span>
        </div>
        <div class="cs-metric">
          <span class="cs-metric-name">vs. deficiency threshold</span>
          <span class="cs-metric-val">{"Below" if pd.notna(baseline_rmssd) and baseline_rmssd < ESC_RMSSD_DEFICIENCY else "At"} {ESC_RMSSD_DEFICIENCY} ms (ESC/NASPE 1996)</span>
        </div>
        <div class="cs-metric">
          <span class="cs-metric-name">vs. population median</span>
          <span class="cs-metric-val critical">{pct_below_median:.0f}% below ({POPULATION_RMSSD_MEDIAN} ms)</span>
        </div>
        <div class="cs-metric">
          <span class="cs-metric-name">Post-HSCT expected</span>
          <span class="cs-metric-val">{HSCT_TYPICAL_RMSSD[0]}-{HSCT_TYPICAL_RMSSD[1]} ms</span>
        </div>
      </div>
      <div class="cs-finding">
        <div class="cs-finding-header">
          <span class="cs-finding-title">Nocturnal Heart Rate</span>
          <span class="cs-sev critical">Elevated</span>
        </div>
        <div class="cs-metric">
          <span class="cs-metric-name">Concern threshold</span>
          <span class="cs-metric-val">{NOCTURNAL_HR_ELEVATED} bpm (nocturnal)</span>
        </div>
        <div class="cs-metric">
          <span class="cs-metric-name">Note</span>
          <span class="cs-metric-val" style="font-size:0.75rem">IST criterion (90 bpm) applies to 24-hour or resting awake HR, not sleep-only data</span>
        </div>
      </div>
    </div>"""

    # Limitations as conclusion block
    limitations_html = f"""
    <div class="cs-conclusion">
      <strong>Limitations</strong> —
      Single known positive event (no specificity/PPV calculation).
      LSTM autoencoder trained on {len(daily)} days (limited).
      Oura Ring PPG is not a medical device.
      tsfresh features depend on HR data points per night (variable coverage).
    </div>"""

    return verdict_html + stats_html + findings_html + relevance_html + limitations_html


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> int:
    """Run all anomaly detection methods and generate report."""
    print("=" * 70)
    print("  ANOMALY DETECTION ENGINE - Oura Ring Biometric Streams")
    print(f"  Data: {DATA_START.strftime('%b %Y')} onwards")
    print("=" * 70)
    t_total = time.time()

    # Load data
    data = load_data()
    daily = build_daily_features(data)

    # Run all 5 methods
    all_results: dict[str, dict] = {}

    # 1. Matrix Profile
    try:
        all_results["matrix_profile"] = run_matrix_profile(daily)
    except Exception as e:
        print(f"  [ERROR] Matrix Profile failed: {e}")
        traceback.print_exc()
        all_results["matrix_profile"] = {
            "method": "Matrix Profile",
            "feb9_detected": False,
            "error": str(e),
        }

    # 2. Isolation Forest
    try:
        all_results["isolation_forest"] = run_isolation_forest(daily)
    except Exception as e:
        print(f"  [ERROR] Isolation Forest failed: {e}")
        traceback.print_exc()
        all_results["isolation_forest"] = {
            "method": "Isolation Forest",
            "feb9_detected": False,
            "error": str(e),
        }

    # 3. LSTM Autoencoder
    try:
        all_results["lstm_autoencoder"] = run_lstm_autoencoder(daily)
    except Exception as e:
        print(f"  [ERROR] LSTM Autoencoder failed: {e}")
        traceback.print_exc()
        all_results["lstm_autoencoder"] = {
            "method": "LSTM Autoencoder",
            "feb9_detected": False,
            "error": str(e),
        }

    # 4. SPC
    try:
        all_results["spc"] = run_spc(daily)
    except Exception as e:
        print(f"  [ERROR] SPC failed: {e}")
        traceback.print_exc()
        all_results["spc"] = {"method": "SPC", "feb9_detected": False, "error": str(e)}

    # 5. tsfresh
    try:
        all_results["tsfresh"] = run_tsfresh_clustering(data)
    except Exception as e:
        print(f"  [ERROR] tsfresh failed: {e}")
        traceback.print_exc()
        all_results["tsfresh"] = {
            "method": "tsfresh",
            "feb9_detected": False,
            "error": str(e),
        }

    # Validate against known event
    validation = validate_feb9(all_results)

    # Compute ensemble scores
    ensemble = compute_ensemble_scores(daily, all_results)

    # Generate HTML report
    html_content = generate_html_report(daily, ensemble, all_results, validation)
    HTML_OUTPUT.write_text(html_content, encoding="utf-8")
    print(f"\n[REPORT] HTML report saved to: {HTML_OUTPUT}")

    # Generate JSON metrics
    json_metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_range": {
            "start": daily["date"].iloc[0],
            "end": daily["date"].iloc[-1],
            "n_days": len(daily),
        },
        "known_event": KNOWN_EVENT_DATE,
        "validation": validation,
        "methods": {},
        "ensemble": {
            "top_10_anomalies": ensemble.nlargest(10, "ensemble_score")[
                ["date", "ensemble_score"]
            ].to_dict("records"),
            "threshold_90pct": float(ensemble["ensemble_score"].quantile(0.9)),
            "n_anomaly_days": int(ensemble["is_anomaly"].sum()),
            "daily_scores": ensemble[["date", "ensemble_score"]]
            .dropna(subset=["ensemble_score"])
            .to_dict("records"),
        },
        "total_runtime_s": round(time.time() - t_total, 2),
    }

    for method_name, result in all_results.items():
        json_metrics["methods"][method_name] = {
            "feb9_detected": result.get("feb9_detected", False),
            "runtime_s": result.get("runtime_s", 0),
            "error": result.get("error"),
            "n_anomalies": (
                len(result.get("anomalies", []))
                if "anomalies" in result
                else result.get("n_anomalies", None)
            ),
        }

    JSON_OUTPUT.write_text(
        json.dumps(json_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[REPORT] JSON metrics saved to: {JSON_OUTPUT}")

    total_time = round(time.time() - t_total, 2)
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE - Total runtime: {total_time}s")
    print(
        f"  Methods detecting Feb 9: {validation['methods_detected']}/{validation['methods_tested']}"
    )
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
