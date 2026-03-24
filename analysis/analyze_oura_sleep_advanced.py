#!/usr/bin/env python3
"""
Advanced Sleep Architecture Analysis

Deep analysis exploiting Oura epoch-level (5-min) and movement-level (30-sec) data
to characterise sleep fragmentation, stage transitions, ultradian rhythm regularity,
and HRV-sleep coupling in a post-HSCT patient with chronic GVHD.

Analyses:
  1. Sleep Fragmentation Index (wake-to-sleep transitions / hour)
  2. Markov Transition Matrix between sleep stages
  3. REM Latency Tracking (time from sleep onset to first REM epoch)
  4. Sleep Cycle Detection (NREM-REM cycles per night)
  5. Movement Density Analysis (restlessness by stage and hour)
  6. Sleep Efficiency Trends (nightly + 7-day rolling average)
  7. Ultradian Rhythm Analysis (FFT spectral analysis of staging)
  8. Sleep-HRV Coupling (HRV per sleep stage, coupling coefficient)
  9. Pre/Post Treatment Comparison (Mann-Whitney U, effect sizes)

Data sources (from oura.db):
  oura_sleep_epochs (5-min resolution): phase 1=deep, 2=light, 3=REM, 4=awake
  oura_sleep_movement (30-sec resolution): 1=still, 2=restless, 3=toss/turn, 4=active
  oura_sleep_periods: nightly aggregates
  oura_sleep_hr_timeseries (~5-min): beat-level sleep HR
  oura_hrv (5-min): RMSSD epochs

See config.py for patient details and treatment dates.

Output:
  - reports/advanced_sleep_analysis.html (interactive Plotly dashboard)
  - reports/advanced_sleep_metrics.json (structured metrics)
  - Summary printed to stdout

Usage:
    python analysis/analyze_oura_sleep_advanced.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import signal, stats

# ---------------------------------------------------------------------------
# Paths & patient config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH,
    REPORTS_DIR,
    TREATMENT_START,
    TREATMENT_START_STR,
    PATIENT_LABEL,
)
from _theme import (
    wrap_html as _theme_wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    BG_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    BORDER_SUBTLE,
    C_SLEEP,
    C_CRITICAL,
    C_WARNING,
    C_GOOD,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_ORANGE,
)

pio.templates.default = "clinical_dark"

HTML_OUT = REPORTS_DIR / "advanced_sleep_analysis.html"
JSON_OUT = REPORTS_DIR / "advanced_sleep_metrics.json"

# Sleep stage mapping (Oura epoch encoding)
PHASE_MAP = {1: "deep", 2: "light", 3: "REM", 4: "awake"}
PHASE_LABELS = ["deep", "light", "REM", "awake"]
PHASE_COLORS = {
    "deep": ACCENT_BLUE,
    "light": C_SLEEP,
    "REM": ACCENT_PURPLE,
    "awake": ACCENT_RED,
}

# Movement classification mapping (Oura movement encoding)
MOVEMENT_MAP = {1: "still", 2: "restless", 3: "toss/turn", 4: "active"}

# Derived palette aliases (mapped from theme constants)
C_RUX = ACCENT_ORANGE  # Ruxolitinib marker colour
C_OK = C_GOOD  # Healthy norms
C_BLUE = C_SLEEP  # Primary data traces (sleep-associated indigo)

# Published norms for comparison
NORMS = {
    "fragmentation_index": {
        "healthy": 5.0,
        "clinical_concern": 10.0,
        "unit": "transitions/hr",
    },
    "rem_latency": {
        "normal_min": 70,
        "normal_max": 120,
        "elevated": 120,
        "unit": "min",
    },
    "cycles_per_night": {"healthy_min": 4, "healthy_max": 6},
    "cycle_duration": {"normal_min": 80, "normal_max": 100, "unit": "min"},
    "efficiency": {"healthy": 85, "poor": 75, "unit": "%"},
    "ultradian_period": {
        "expected": 90,
        "range_min": 80,
        "range_max": 110,
        "unit": "min",
    },
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_connection() -> sqlite3.Connection:
    """Open read-only connection to biometrics database."""
    if not DATABASE_PATH.exists():
        print(f"ERROR: Database not found at {DATABASE_PATH}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_read(sql: str, con: sqlite3.Connection) -> pd.DataFrame:
    """Wrap pd.read_sql_query; return empty DataFrame on missing table."""
    try:
        return pd.read_sql_query(sql, con)
    except Exception:
        logging.warning(
            "Query failed (table may be missing), returning empty DataFrame"
        )
        return pd.DataFrame()


def load_sleep_periods(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load sleep periods with basic filtering."""
    df = _safe_read(
        """
        SELECT period_id, day, type, average_hrv, average_heart_rate, average_breath,
               total_sleep_duration, rem_sleep_duration, deep_sleep_duration,
               light_sleep_duration, awake_time, efficiency, latency, restless_periods,
               lowest_heart_rate, bedtime_start, bedtime_end, time_in_bed
        FROM oura_sleep_periods
        ORDER BY day
        """,
        conn,
    )
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


def load_epochs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all epoch data joined with period date."""
    df = _safe_read(
        """
        SELECT e.period_id, e.epoch_index, e.phase, sp.day, sp.bedtime_start, sp.type,
               sp.time_in_bed
        FROM oura_sleep_epochs e
        JOIN oura_sleep_periods sp ON e.period_id = sp.period_id
        ORDER BY sp.day, e.epoch_index
        """,
        conn,
    )
    df["day"] = pd.to_datetime(df["day"]).dt.date
    df["stage"] = df["phase"].map(PHASE_MAP)
    return df


def load_movements(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all movement data joined with period date."""
    df = _safe_read(
        """
        SELECT m.period_id, m.movement_index, m.classification, sp.day, sp.type,
               sp.bedtime_start, sp.time_in_bed
        FROM oura_sleep_movement m
        JOIN oura_sleep_periods sp ON m.period_id = sp.period_id
        ORDER BY sp.day, m.movement_index
        """,
        conn,
    )
    df["day"] = pd.to_datetime(df["day"]).dt.date
    df["label"] = df["classification"].map(MOVEMENT_MAP)
    return df


def load_hrv(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load HRV data for sleep periods."""
    df = _safe_read(
        """
        SELECT id, timestamp, rmssd, source
        FROM oura_hrv
        WHERE source LIKE 'sleep_period:%'
        ORDER BY timestamp
        """,
        conn,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_hr_timeseries(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load heart rate timeseries during sleep."""
    df = _safe_read(
        """
        SELECT period_id, timestamp, bpm
        FROM oura_sleep_hr_timeseries
        ORDER BY timestamp
        """,
        conn,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Analysis 1: Sleep Fragmentation Index
# ---------------------------------------------------------------------------
def compute_fragmentation_index(epochs: pd.DataFrame) -> pd.DataFrame:
    """
    Count wake-to-sleep transitions per hour of sleep.

    A fragmentation event is any transition from awake (phase 4) to any sleep
    phase (1, 2, or 3). The index is normalised per hour of total sleep time.

    Returns DataFrame with columns: day, period_id, frag_count, sleep_hours,
    fragmentation_index.
    """
    results = []
    for pid, grp in epochs.groupby("period_id"):
        grp = grp.sort_values("epoch_index")
        phases = grp["phase"].values
        day = grp["day"].iloc[0]

        # Count wake->sleep transitions
        frag_count = 0
        for i in range(1, len(phases)):
            if phases[i - 1] == 4 and phases[i] in (1, 2, 3):
                frag_count += 1

        # Total sleep time in hours (count of non-awake epochs * 5 min / 60)
        sleep_epochs = np.sum(phases != 4)
        sleep_hours = sleep_epochs * 5.0 / 60.0

        fi = frag_count / sleep_hours if sleep_hours > 0 else np.nan
        results.append(
            {
                "day": day,
                "period_id": pid,
                "frag_count": frag_count,
                "sleep_epochs": int(sleep_epochs),
                "sleep_hours": round(sleep_hours, 2),
                "fragmentation_index": round(fi, 2) if not np.isnan(fi) else None,
            }
        )
    return pd.DataFrame(results).sort_values("day").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 2: Sleep Stage Transition Matrix
# ---------------------------------------------------------------------------
def compute_transition_matrix(epochs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 4x4 Markov transition probability matrix between sleep stages.

    Returns:
        counts: raw transition count matrix (4x4)
        probs: row-normalised probability matrix (4x4)
    Order: deep(0), light(1), REM(2), awake(3) matching PHASE_LABELS.
    """
    phase_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
    counts = np.zeros((4, 4), dtype=int)

    for _pid, grp in epochs.groupby("period_id"):
        phases = grp.sort_values("epoch_index")["phase"].values
        for i in range(1, len(phases)):
            fr = phase_to_idx[phases[i - 1]]
            to = phase_to_idx[phases[i]]
            counts[fr, to] += 1

    # Row-normalise to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    probs = counts / row_sums
    return counts, probs


# Published healthy norms (approximation from Kishi et al. 2011, Feinberg & Floyd 1979)
HEALTHY_TRANSITION_PROBS = np.array(
    [
        # deep   light   REM    awake
        [0.55, 0.40, 0.02, 0.03],  # from deep
        [0.10, 0.65, 0.15, 0.10],  # from light
        [0.02, 0.20, 0.65, 0.13],  # from REM
        [0.05, 0.50, 0.10, 0.35],  # from awake
    ]
)


# ---------------------------------------------------------------------------
# Analysis 3: REM Latency Tracking
# ---------------------------------------------------------------------------
def compute_rem_latency(epochs: pd.DataFrame) -> pd.DataFrame:
    """
    Time from sleep onset to first REM epoch.

    Sleep onset = first non-awake epoch.
    REM latency = (first REM epoch_index - sleep onset epoch_index) * 5 minutes.
    """
    results = []
    for pid, grp in epochs.groupby("period_id"):
        grp = grp.sort_values("epoch_index")
        day = grp["day"].iloc[0]
        phases = grp["phase"].values
        indices = grp["epoch_index"].values

        # Find sleep onset (first non-awake epoch)
        sleep_onset_idx = None
        for i, p in enumerate(phases):
            if p in (1, 2, 3):
                sleep_onset_idx = i
                break

        if sleep_onset_idx is None:
            results.append(
                {
                    "day": day,
                    "period_id": pid,
                    "rem_latency_min": None,
                    "has_rem": False,
                }
            )
            continue

        # Find first REM epoch
        first_rem_idx = None
        for i in range(sleep_onset_idx, len(phases)):
            if phases[i] == 3:
                first_rem_idx = i
                break

        if first_rem_idx is None:
            results.append(
                {
                    "day": day,
                    "period_id": pid,
                    "rem_latency_min": None,
                    "has_rem": False,
                }
            )
        else:
            latency_epochs = indices[first_rem_idx] - indices[sleep_onset_idx]
            latency_min = latency_epochs * 5
            results.append(
                {
                    "day": day,
                    "period_id": pid,
                    "rem_latency_min": int(latency_min),
                    "has_rem": True,
                }
            )

    return pd.DataFrame(results).sort_values("day").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 4: Sleep Cycle Detection
# ---------------------------------------------------------------------------
def detect_sleep_cycles(epochs: pd.DataFrame) -> pd.DataFrame:
    """
    Identify NREM-REM cycles within each night.

    A cycle = contiguous NREM block (deep/light) followed by a REM block.
    A cycle is considered complete if it contains at least 1 REM epoch.
    Cycle boundaries are drawn when we transition from REM back to NREM
    (or awake, which resets to next NREM-REM pair).

    Returns per-night: cycle_count, mean_cycle_duration_min, cycle_details.
    """
    results = []
    for pid, grp in epochs.groupby("period_id"):
        grp = grp.sort_values("epoch_index")
        day = grp["day"].iloc[0]
        phases = grp["phase"].values

        cycles = []
        current_cycle_start = None
        in_nrem = False
        seen_rem = False

        for i, p in enumerate(phases):
            if p in (1, 2):  # NREM
                if not in_nrem and not seen_rem:
                    # Start of new potential cycle
                    current_cycle_start = i
                    in_nrem = True
                elif seen_rem:
                    # Was in REM, now back to NREM = cycle boundary
                    if current_cycle_start is not None:
                        cycle_len = i - current_cycle_start
                        cycles.append(cycle_len * 5)  # minutes
                    current_cycle_start = i
                    in_nrem = True
                    seen_rem = False
                else:
                    in_nrem = True
            elif p == 3:  # REM
                if current_cycle_start is None:
                    current_cycle_start = i
                seen_rem = True
                in_nrem = False
            elif p == 4:  # Awake
                if seen_rem and current_cycle_start is not None:
                    cycle_len = i - current_cycle_start
                    cycles.append(cycle_len * 5)
                    current_cycle_start = None
                    in_nrem = False
                    seen_rem = False
                # If awake during NREM block, keep the cycle start
                # (brief awakenings within a cycle)

        # Close final cycle if it ended with REM
        if seen_rem and current_cycle_start is not None:
            cycle_len = len(phases) - current_cycle_start
            cycles.append(cycle_len * 5)

        mean_dur = np.mean(cycles) if cycles else None
        results.append(
            {
                "day": day,
                "period_id": pid,
                "cycle_count": len(cycles),
                "mean_cycle_duration_min": round(mean_dur, 1) if mean_dur else None,
                "cycle_durations_min": cycles,
            }
        )

    return pd.DataFrame(results).sort_values("day").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 5: Movement Density Analysis
# ---------------------------------------------------------------------------
def compute_movement_density(movements: pd.DataFrame, epochs: pd.DataFrame) -> dict:
    """
    Analyse movement patterns by sleep stage and time-of-night.

    Returns:
        - nightly_restlessness: restlessness index per night
        - stage_movement: movement classification distribution per sleep stage
        - hourly_density: movement intensity by hour of night (0-11)
    """
    # Nightly restlessness index
    nightly = []
    for pid, grp in movements.groupby("period_id"):
        day = grp["day"].iloc[0]
        total = len(grp)
        restless_plus = grp["classification"].isin([2, 3, 4]).sum()
        idx = restless_plus / total if total > 0 else np.nan
        nightly.append(
            {
                "day": day,
                "period_id": pid,
                "restlessness_index": round(idx, 4),
                "total_movements": total,
                "restless_count": int(restless_plus),
            }
        )
    nightly_df = pd.DataFrame(nightly).sort_values("day").reset_index(drop=True)

    # Movement by sleep stage (cross-reference via period_id and index mapping)
    # Each epoch covers 10 movement slots (5 min / 30 sec = 10)
    stage_movement: dict[str, Counter] = {s: Counter() for s in PHASE_LABELS}
    for pid in epochs["period_id"].unique():
        ep = epochs[epochs["period_id"] == pid].sort_values("epoch_index")
        mv = movements[movements["period_id"] == pid].sort_values("movement_index")
        if ep.empty or mv.empty:
            continue

        epoch_phases = ep["phase"].values
        mv_classes = mv["classification"].values
        n_epochs = len(epoch_phases)

        for mi, cls in enumerate(mv_classes):
            epoch_idx = min(mi // 10, n_epochs - 1)
            stage = PHASE_MAP.get(epoch_phases[epoch_idx], "unknown")
            if stage in stage_movement:
                stage_movement[stage][MOVEMENT_MAP.get(cls, "unknown")] += 1

    # Hourly density (divide night into 12 half-hour bins mapped to hour offsets)
    hourly_intensity: dict[int, list[int]] = defaultdict(list)
    for pid, grp in movements.groupby("period_id"):
        n_mov = len(grp)
        time_in_bed = (
            grp["time_in_bed"].iloc[0] if "time_in_bed" in grp.columns else n_mov * 30
        )
        total_hours = time_in_bed / 3600.0
        if total_hours <= 0:
            continue
        bin_size = max(1, n_mov // min(12, max(1, int(total_hours))))
        classes = grp.sort_values("movement_index")["classification"].values
        for hour_bin in range(min(12, int(total_hours) + 1)):
            start = hour_bin * bin_size
            end = min(start + bin_size, n_mov)
            if start >= n_mov:
                break
            chunk = classes[start:end]
            intensity = np.mean(chunk) if len(chunk) > 0 else 1.0
            hourly_intensity[hour_bin].append(float(intensity))

    hourly_avg = {
        h: round(np.mean(vals), 3) for h, vals in sorted(hourly_intensity.items())
    }

    return {
        "nightly": nightly_df,
        "stage_movement": {s: dict(c) for s, c in stage_movement.items()},
        "hourly_density": hourly_avg,
    }


# ---------------------------------------------------------------------------
# Analysis 6: Sleep Efficiency Trends
# ---------------------------------------------------------------------------
def compute_efficiency_trends(
    periods: pd.DataFrame, epochs: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute nightly sleep efficiency from both Oura aggregate and epoch data.

    Epoch-based efficiency = (non-awake epochs / total epochs) * 100.
    Includes 7-day rolling average.
    """
    # Oura-reported efficiency
    eff = periods[["day", "period_id", "efficiency", "type"]].copy()

    # Epoch-based efficiency
    epoch_eff = []
    for pid, grp in epochs.groupby("period_id"):
        total = len(grp)
        sleep = (grp["phase"] != 4).sum()
        day = grp["day"].iloc[0]
        epoch_eff.append(
            {
                "period_id": pid,
                "epoch_efficiency": round(sleep / total * 100, 1)
                if total > 0
                else None,
            }
        )
    epoch_df = pd.DataFrame(epoch_eff)

    result = eff.merge(epoch_df, on="period_id", how="left").sort_values("day")
    # 7-day rolling average (on long_sleep only for stability)
    long_mask = result["type"] == "long_sleep"
    result["rolling_7d_efficiency"] = np.nan
    if long_mask.sum() >= 7:
        vals = result.loc[long_mask, "efficiency"].values.astype(float)
        rolling = pd.Series(vals).rolling(7, min_periods=3).mean()
        result.loc[long_mask, "rolling_7d_efficiency"] = rolling.values

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 7: Ultradian Rhythm Analysis (FFT)
# ---------------------------------------------------------------------------
def compute_ultradian_rhythm(epochs: pd.DataFrame) -> dict:
    """
    Apply FFT to sleep staging time series to detect 90-min cycle regularity.

    Converts epoch phases to a numeric depth signal:
      deep=3, light=2, REM=1, awake=0
    Then runs scipy.signal.periodogram to find dominant periods.
    """
    # Only long_sleep nights with sufficient epochs (>= 60 = 5 hours)
    valid_periods = []
    for pid, grp in epochs.groupby("period_id"):
        if grp["type"].iloc[0] != "long_sleep":
            continue
        if len(grp) < 60:
            continue
        valid_periods.append(pid)

    if not valid_periods:
        return {"dominant_period_min": None, "spectral_peaks": [], "n_nights": 0}

    all_peaks = []
    all_dominant = []

    for pid in valid_periods:
        grp = epochs[epochs["period_id"] == pid].sort_values("epoch_index")
        phases = grp["phase"].values

        # Convert to depth signal: deep=3, light=2, REM=1, awake=0
        depth = np.zeros(len(phases))
        for i, p in enumerate(phases):
            if p == 1:
                depth[i] = 3.0  # deep
            elif p == 2:
                depth[i] = 2.0  # light
            elif p == 3:
                depth[i] = 1.0  # REM
            else:
                depth[i] = 0.0  # awake

        # Remove mean (DC component)
        depth = depth - depth.mean()

        # Sampling frequency: 1 sample per 5 minutes = 1/300 Hz
        fs = 1.0 / 300.0  # Hz

        freqs, power = signal.periodogram(depth, fs=fs, detrend="constant")

        # Convert to periods in minutes, filter to physiological range (30-180 min)
        with np.errstate(divide="ignore"):
            periods_min = np.where(freqs > 0, 1.0 / freqs / 60.0, np.inf)

        mask = (periods_min >= 30) & (periods_min <= 180) & (power > 0)
        if mask.sum() > 0:
            peak_idx = np.argmax(power[mask])
            filtered_periods = periods_min[mask]
            filtered_power = power[mask]
            dominant = filtered_periods[peak_idx]
            all_dominant.append(dominant)
            # Top 3 peaks
            top_k = min(3, len(filtered_power))
            top_indices = np.argsort(filtered_power)[-top_k:][::-1]
            for ti in top_indices:
                all_peaks.append(
                    {
                        "period_min": round(float(filtered_periods[ti]), 1),
                        "power": round(float(filtered_power[ti]), 4),
                    }
                )

    mean_dominant = round(float(np.mean(all_dominant)), 1) if all_dominant else None

    # Aggregate peaks by binning to nearest 5 min
    peak_bins: dict[int, list[float]] = defaultdict(list)
    for p in all_peaks:
        bin_center = round(p["period_min"] / 5) * 5
        peak_bins[bin_center].append(p["power"])

    spectral_summary = sorted(
        [
            {"period_min": k, "mean_power": round(np.mean(v), 4), "count": len(v)}
            for k, v in peak_bins.items()
        ],
        key=lambda x: x["mean_power"],
        reverse=True,
    )[:5]

    return {
        "dominant_period_min": mean_dominant,
        "spectral_peaks": spectral_summary,
        "n_nights": len(valid_periods),
    }


# ---------------------------------------------------------------------------
# Analysis 8: Sleep-HRV Coupling
# ---------------------------------------------------------------------------
def compute_hrv_coupling(
    epochs: pd.DataFrame, hrv: pd.DataFrame, periods: pd.DataFrame
) -> dict:
    """
    Correlate sleep stage with concurrent HRV (RMSSD).

    Match HRV timestamps to epoch windows within each sleep period.
    Expected: deep sleep should show highest parasympathetic tone (highest HRV).
    """
    stage_hrv: dict[str, list[float]] = {s: [] for s in PHASE_LABELS}
    matched_count = 0

    for pid in epochs["period_id"].unique():
        ep_grp = epochs[epochs["period_id"] == pid].sort_values("epoch_index")
        if ep_grp.empty:
            continue

        # Get bedtime_start for this period
        period_row = periods[periods["period_id"] == pid]
        if period_row.empty:
            continue
        bedtime_start_str = period_row["bedtime_start"].iloc[0]
        if not bedtime_start_str:
            continue

        try:
            bt_start = pd.to_datetime(bedtime_start_str, utc=True)
        except (ValueError, TypeError) as e:
            logging.debug(f"Skipping period with unparseable bedtime_start: {e}")
            continue

        # HRV readings within this period's timeframe
        time_in_bed = period_row["time_in_bed"].iloc[0]
        if not time_in_bed or time_in_bed <= 0:
            continue
        bt_end = bt_start + pd.Timedelta(seconds=int(time_in_bed))
        hrv_window = hrv[(hrv["timestamp"] >= bt_start) & (hrv["timestamp"] <= bt_end)]

        if hrv_window.empty:
            continue

        epoch_phases = ep_grp["phase"].values
        n_epochs = len(epoch_phases)

        for _, row in hrv_window.iterrows():
            ts = row["timestamp"]
            rmssd = row["rmssd"]
            if rmssd is None or rmssd <= 0:
                continue

            # Map timestamp to epoch index
            elapsed_sec = (ts - bt_start).total_seconds()
            epoch_idx = int(elapsed_sec // 300)
            if 0 <= epoch_idx < n_epochs:
                stage = PHASE_MAP.get(epoch_phases[epoch_idx])
                if stage:
                    stage_hrv[stage].append(float(rmssd))
                    matched_count += 1

    # Statistics per stage
    stage_stats = {}
    all_means = []
    for stage in PHASE_LABELS:
        vals = stage_hrv[stage]
        if vals:
            stage_stats[stage] = {
                "mean_rmssd": round(np.mean(vals), 2),
                "median_rmssd": round(np.median(vals), 2),
                "std_rmssd": round(np.std(vals), 2),
                "n": len(vals),
                "p25": round(np.percentile(vals, 25), 2),
                "p75": round(np.percentile(vals, 75), 2),
            }
            all_means.append(np.mean(vals))
        else:
            stage_stats[stage] = {"mean_rmssd": None, "n": 0}

    # Coupling assessment: is deep > light > REM > awake?
    coupling_correct = False
    if all(stage_stats[s]["mean_rmssd"] is not None for s in ["deep", "light", "REM"]):
        deep_m = stage_stats["deep"]["mean_rmssd"]
        light_m = stage_stats["light"]["mean_rmssd"]
        rem_m = stage_stats["REM"]["mean_rmssd"]
        # Expected: deep >= light, and (deep or light) > awake
        coupling_correct = deep_m >= light_m

    # Kruskal-Wallis test across stages
    groups = [np.array(stage_hrv[s]) for s in PHASE_LABELS if len(stage_hrv[s]) >= 5]
    kw_stat, kw_p = (None, None)
    if len(groups) >= 2:
        try:
            kw_stat, kw_p = stats.kruskal(*groups)
            kw_stat = round(float(kw_stat), 2)
            kw_p = float(kw_p)
        except Exception:
            pass

    return {
        "stage_stats": stage_stats,
        "matched_hrv_epochs": matched_count,
        "coupling_correct_deep_gt_light": coupling_correct,
        "kruskal_wallis": {"H": kw_stat, "p": kw_p},
        "raw_by_stage": stage_hrv,
    }


# ---------------------------------------------------------------------------
# Analysis 9: Pre/Post Ruxolitinib Comparison
# ---------------------------------------------------------------------------
def compare_pre_post_ruxolitinib(
    periods: pd.DataFrame,
    frag_df: pd.DataFrame,
    rem_latency_df: pd.DataFrame,
    cycles_df: pd.DataFrame,
    efficiency_df: pd.DataFrame,
    movement_nightly: pd.DataFrame,
) -> dict:
    """
    Compare all sleep metrics before vs after treatment start (TREATMENT_START).
    Mann-Whitney U test + Cliff's delta effect size.
    """
    cutoff = TREATMENT_START

    def split_pre_post(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
        pre = df[df["day"] < cutoff][col].dropna().values.astype(float)
        post = df[df["day"] >= cutoff][col].dropna().values.astype(float)
        return pre, post

    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float | None:
        """Compute Cliff's delta effect size."""
        if len(x) == 0 or len(y) == 0:
            return None
        n_x, n_y = len(x), len(y)
        more = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        return round((more - less) / (n_x * n_y), 3)

    def run_comparison(pre: np.ndarray, post: np.ndarray, name: str) -> dict:
        result: dict[str, Any] = {
            "metric": name,
            "pre_n": len(pre),
            "post_n": len(post),
        }
        if len(pre) >= 2 and len(post) >= 1:
            result["pre_mean"] = round(float(np.mean(pre)), 2)
            result["pre_median"] = round(float(np.median(pre)), 2)
            result["pre_std"] = round(float(np.std(pre)), 2)
            result["post_mean"] = (
                round(float(np.mean(post)), 2) if len(post) > 0 else None
            )
            result["post_median"] = (
                round(float(np.median(post)), 2) if len(post) > 0 else None
            )
            if len(post) >= 2:
                try:
                    u_stat, p_val = stats.mannwhitneyu(
                        pre, post, alternative="two-sided"
                    )
                    result["mann_whitney_U"] = float(u_stat)
                    result["p_value"] = float(p_val)
                except Exception:
                    result["mann_whitney_U"] = None
                    result["p_value"] = None
            else:
                # With very few post values, report percentile rank instead
                if len(post) > 0:
                    rank = stats.percentileofscore(pre, post[0])
                    result["post_percentile_in_pre"] = round(rank, 1)
                result["mann_whitney_U"] = None
                result["p_value"] = None
            result["cliffs_delta"] = cliffs_delta(pre, post)
        else:
            result["note"] = "Insufficient data for comparison"
        return result

    comparisons = []

    # Sleep efficiency (from periods, long_sleep only)
    lsp = periods[periods["type"] == "long_sleep"]
    pre_eff, post_eff = split_pre_post(lsp, "efficiency")
    comparisons.append(run_comparison(pre_eff, post_eff, "sleep_efficiency_%"))

    # Average HR
    pre_hr, post_hr = split_pre_post(lsp, "average_heart_rate")
    comparisons.append(run_comparison(pre_hr, post_hr, "average_heart_rate_bpm"))

    # Average HRV
    pre_hrv, post_hrv = split_pre_post(lsp, "average_hrv")
    comparisons.append(run_comparison(pre_hrv, post_hrv, "average_hrv_ms"))

    # Total sleep duration
    pre_dur, post_dur = split_pre_post(lsp, "total_sleep_duration")
    pre_dur_h = pre_dur / 3600.0
    post_dur_h = post_dur / 3600.0
    comparisons.append(run_comparison(pre_dur_h, post_dur_h, "total_sleep_hours"))

    # Fragmentation index
    frag_long = frag_df.merge(periods[["period_id", "type"]], on="period_id")
    frag_long = frag_long[frag_long["type"] == "long_sleep"]
    pre_fi, post_fi = split_pre_post(frag_long, "fragmentation_index")
    comparisons.append(run_comparison(pre_fi, post_fi, "fragmentation_index"))

    # REM latency
    rem_long = rem_latency_df.merge(periods[["period_id", "type"]], on="period_id")
    rem_long = rem_long[(rem_long["type"] == "long_sleep") & rem_long["has_rem"]]
    pre_rl, post_rl = split_pre_post(rem_long, "rem_latency_min")
    comparisons.append(run_comparison(pre_rl, post_rl, "rem_latency_min"))

    # Cycle count
    cyc_long = cycles_df.merge(periods[["period_id", "type"]], on="period_id")
    cyc_long = cyc_long[cyc_long["type"] == "long_sleep"]
    pre_cc, post_cc = split_pre_post(cyc_long, "cycle_count")
    comparisons.append(run_comparison(pre_cc, post_cc, "cycle_count"))

    # Restlessness
    rest_long = movement_nightly.merge(periods[["period_id", "type"]], on="period_id")
    rest_long = rest_long[rest_long["type"] == "long_sleep"]
    pre_ri, post_ri = split_pre_post(rest_long, "restlessness_index")
    comparisons.append(run_comparison(pre_ri, post_ri, "restlessness_index"))

    # Deep sleep %
    lsp_calc = lsp.copy()
    total_dur = lsp_calc["total_sleep_duration"].replace(0, np.nan)
    lsp_calc["deep_pct"] = (
        lsp_calc["deep_sleep_duration"] / total_dur * 100
    ).round(1)
    pre_dp, post_dp = split_pre_post(lsp_calc, "deep_pct")
    comparisons.append(run_comparison(pre_dp, post_dp, "deep_sleep_%"))

    # REM sleep %
    lsp_calc["rem_pct"] = (
        lsp_calc["rem_sleep_duration"] / total_dur * 100
    ).round(1)
    pre_rp, post_rp = split_pre_post(lsp_calc, "rem_pct")
    comparisons.append(run_comparison(pre_rp, post_rp, "rem_sleep_%"))

    return {
        "cutoff_date": str(cutoff),
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Dashboard builder (Plotly)
# ---------------------------------------------------------------------------
def build_dashboard(
    frag_df: pd.DataFrame,
    trans_counts: np.ndarray,
    trans_probs: np.ndarray,
    rem_latency_df: pd.DataFrame,
    cycles_df: pd.DataFrame,
    movement_data: dict,
    efficiency_df: pd.DataFrame,
    ultradian: dict,
    hrv_coupling: dict,
    rux_comparison: dict,
    periods: pd.DataFrame,
) -> str:
    """Build multi-panel interactive HTML dashboard."""

    # Filter to long_sleep for main trend plots
    long_periods = periods[periods["type"] == "long_sleep"]
    long_pids = set(long_periods["period_id"])

    frag_long = frag_df[frag_df["period_id"].isin(long_pids)].copy()
    rem_long = rem_latency_df[
        rem_latency_df["period_id"].isin(long_pids) & rem_latency_df["has_rem"]
    ].copy()
    cycles_long = cycles_df[cycles_df["period_id"].isin(long_pids)].copy()
    eff_long = efficiency_df[efficiency_df["period_id"].isin(long_pids)].copy()
    rest_long = movement_data["nightly"]
    rest_long = rest_long[rest_long["period_id"].isin(long_pids)].copy()

    # Convert days to strings for plotly
    def day_str(df: pd.DataFrame) -> list[str]:
        return [str(d) for d in df["day"]]

    fig = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=[
            "Sleep Fragmentation Index",
            "Stage Transition Probabilities",
            "REM Latency",
            "NREM-REM Cycles per Night",
            "Movement Density by Stage",
            "Sleep Efficiency",
            "Ultradian Rhythm (FFT)",
            "HRV-Sleep Stage Coupling",
            "Pre/Post Ruxolitinib: Fragmentation",
            "Pre/Post Ruxolitinib: Efficiency",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "scatter"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
    )

    # ---- Panel 1: Fragmentation Index ----
    if not frag_long.empty:
        days = day_str(frag_long)
        fi_vals = frag_long["fragmentation_index"].values

        # Subtle fill below the line
        fig.add_trace(
            go.Scatter(
                x=days,
                y=fi_vals,
                mode="lines+markers",
                name="Fragmentation Index",
                marker=dict(size=4, color=C_BLUE, line=dict(width=0)),
                line=dict(color=C_BLUE, width=2),
                fill="tozeroy",
                fillcolor="rgba(99, 102, 241, 0.1)",
                hovertemplate="<b>%{x}</b><br>Fragmentation: %{y:.1f} transitions/hour<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # Clinical threshold line
        fig.add_hline(
            y=NORMS["fragmentation_index"]["clinical_concern"],
            line_dash="dash",
            line_color=C_CRITICAL,
            annotation_text="Clinical concern",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=NORMS["fragmentation_index"]["healthy"],
            line_dash="dot",
            line_color=C_OK,
            annotation_text="Healthy norm",
            row=1,
            col=1,
        )
        # Ruxolitinib marker
        fig.add_shape(
            type="line",
            x0=str(TREATMENT_START),
            x1=str(TREATMENT_START),
            y0=0,
            y1=1,
            yref="y domain",
            line=dict(color=C_RUX, width=1.5, dash="dash"),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=str(TREATMENT_START),
            y=0.95,
            yref="y domain",
            text="Rux Start",
            showarrow=False,
            font=dict(color=C_RUX, size=9),
            row=1,
            col=1,
        )

    # ---- Panel 2: Transition Matrix Heatmap ----
    fig.add_trace(
        go.Heatmap(
            z=trans_probs,
            x=PHASE_LABELS,
            y=PHASE_LABELS,
            text=np.round(trans_probs, 3).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=12, color="#FFFFFF"),
            colorscale=[
                [0.0, BG_PRIMARY],
                [0.15, "#1a2744"],
                [0.35, "#1E3A5F"],
                [0.55, ACCENT_BLUE],
                [0.75, "#60A5FA"],
                [1.0, "#DBEAFE"],
            ],
            showscale=True,
            colorbar=dict(
                len=0.12,
                y=0.93,
                thickness=10,
                title=dict(text="P", font=dict(size=10)),
                tickfont=dict(size=9),
                outlinewidth=0,
            ),
            hovertemplate="<b>%{y} -> %{x}</b><br>Probability: %{z:.3f}<extra></extra>",
            xgap=2,
            ygap=2,
        ),
        row=1,
        col=2,
    )

    # ---- Panel 3: REM Latency ----
    if not rem_long.empty:
        days = day_str(rem_long)
        lat_vals = rem_long["rem_latency_min"].values

        fig.add_trace(
            go.Scatter(
                x=days,
                y=lat_vals,
                mode="lines+markers",
                name="REM Latency",
                marker=dict(size=4, color=ACCENT_PURPLE, line=dict(width=0)),
                line=dict(color=ACCENT_PURPLE, width=2),
                fill="tozeroy",
                fillcolor="rgba(139, 92, 246, 0.08)",
                hovertemplate="<b>%{x}</b><br>REM Latency: %{y} min<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=NORMS["rem_latency"]["elevated"],
            line_dash="dash",
            line_color=C_WARNING,
            annotation_text="Elevated (>120 min)",
            row=2,
            col=1,
        )
        fig.add_shape(
            type="line",
            x0=str(TREATMENT_START),
            x1=str(TREATMENT_START),
            y0=0,
            y1=1,
            yref="y2 domain",
            line=dict(color=C_RUX, width=1.5, dash="dash"),
            row=2,
            col=1,
        )

    # ---- Panel 4: Sleep Cycles per Night ----
    if not cycles_long.empty:
        days = day_str(cycles_long)
        cc_vals = cycles_long["cycle_count"].values

        fig.add_trace(
            go.Bar(
                x=days,
                y=cc_vals,
                name="Cycles/night",
                marker=dict(
                    color=ACCENT_PURPLE,
                    line=dict(color="rgba(139, 92, 246, 0.6)", width=1),
                    opacity=0.85,
                ),
                hovertemplate="<b>%{x}</b><br>Cycles: %{y}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        fig.add_hline(
            y=NORMS["cycles_per_night"]["healthy_min"],
            line_dash="dot",
            line_color=C_OK,
            annotation_text="Min. healthy (4)",
            row=2,
            col=2,
        )

    # ---- Panel 5: Movement by Sleep Stage ----
    stage_mv = movement_data["stage_movement"]
    mv_categories = ["still", "restless", "toss/turn", "active"]
    mv_colors = {
        "still": ACCENT_GREEN,
        "restless": ACCENT_AMBER,
        "toss/turn": ACCENT_ORANGE,
        "active": ACCENT_RED,
    }
    mv_edge_colors = {
        "still": "rgba(16, 185, 129, 0.5)",
        "restless": "rgba(245, 158, 11, 0.5)",
        "toss/turn": "rgba(249, 115, 22, 0.5)",
        "active": "rgba(239, 68, 68, 0.5)",
    }
    for cat in mv_categories:
        vals = [stage_mv.get(stage, {}).get(cat, 0) for stage in PHASE_LABELS]
        fig.add_trace(
            go.Bar(
                x=PHASE_LABELS,
                y=vals,
                name=cat,
                marker=dict(
                    color=mv_colors.get(cat, TEXT_SECONDARY),
                    line=dict(color=mv_edge_colors.get(cat, TEXT_TERTIARY), width=1),
                    opacity=0.9,
                ),
                hovertemplate="<b>%{x}</b><br>" + cat + ": %{y:,}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # ---- Panel 6: Sleep Efficiency ----
    if not eff_long.empty:
        days = day_str(eff_long)
        # Subtle scatter behind the trend line
        fig.add_trace(
            go.Scatter(
                x=days,
                y=eff_long["efficiency"].values,
                mode="markers",
                name="Nightly eff. (Oura)",
                marker=dict(size=3, color=C_BLUE, opacity=0.35),
                hovertemplate="<b>%{x}</b><br>Oura Efficiency: %{y:.0f}%<extra></extra>",
            ),
            row=3,
            col=2,
        )
        if "epoch_efficiency" in eff_long.columns:
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=eff_long["epoch_efficiency"].values,
                    mode="markers",
                    name="Epoch-based eff.",
                    marker=dict(
                        size=3, color=ACCENT_PURPLE, opacity=0.30, symbol="diamond"
                    ),
                    hovertemplate="<b>%{x}</b><br>Epoch Efficiency: %{y:.1f}%<extra></extra>",
                ),
                row=3,
                col=2,
            )
        # Prominent rolling average trend line
        if "rolling_7d_efficiency" in eff_long.columns:
            rolling = eff_long["rolling_7d_efficiency"].values
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=rolling,
                    mode="lines",
                    name="7d avg",
                    line=dict(color=ACCENT_BLUE, width=3),
                    hovertemplate="<b>%{x}</b><br>7-day Avg: %{y:.1f}%<extra></extra>",
                ),
                row=3,
                col=2,
            )
        fig.add_hline(
            y=NORMS["efficiency"]["healthy"],
            line_dash="dot",
            line_color=C_OK,
            annotation_text="Healthy (85%)",
            row=3,
            col=2,
        )
        fig.add_hline(
            y=NORMS["efficiency"]["poor"],
            line_dash="dash",
            line_color=C_WARNING,
            annotation_text="Poor (75%)",
            row=3,
            col=2,
        )
        fig.add_shape(
            type="line",
            x0=str(TREATMENT_START),
            x1=str(TREATMENT_START),
            y0=0,
            y1=1,
            yref="y6 domain",
            line=dict(color=C_RUX, width=1.5, dash="dash"),
            row=3,
            col=2,
        )

    # ---- Panel 7: Ultradian FFT Peaks ----
    if ultradian["spectral_peaks"]:
        peaks = ultradian["spectral_peaks"]
        fig.add_trace(
            go.Bar(
                x=[p["period_min"] for p in peaks],
                y=[p["mean_power"] for p in peaks],
                name="Spectral Power",
                marker=dict(
                    color=C_SLEEP,
                    line=dict(color="rgba(99, 102, 241, 0.5)", width=1),
                    opacity=0.9,
                ),
                text=[f"{p['count']}n" for p in peaks],
                textposition="outside",
                textfont=dict(size=10, color=TEXT_SECONDARY),
                hovertemplate="<b>%{x} min</b><br>Spectral Power: %{y:.4f}<br>Nights: %{text}<extra></extra>",
            ),
            row=4,
            col=1,
        )
        # Mark 90-min expected peak
        fig.add_vline(
            x=NORMS["ultradian_period"]["expected"],
            line_dash="dash",
            line_color=C_OK,
            annotation_text="Expected 90 min",
            row=4,
            col=1,
        )

    # ---- Panel 8: HRV per Sleep Stage ----
    coupling = hrv_coupling["stage_stats"]
    hrv_means = [coupling[s].get("mean_rmssd") or 0 for s in PHASE_LABELS]
    hrv_stds = [coupling[s].get("std_rmssd") or 0 for s in PHASE_LABELS]
    hrv_ns = [coupling[s].get("n") or 0 for s in PHASE_LABELS]
    fig.add_trace(
        go.Bar(
            x=PHASE_LABELS,
            y=hrv_means,
            name="Mean RMSSD",
            marker=dict(
                color=[PHASE_COLORS[s] for s in PHASE_LABELS],
                line=dict(
                    color=[PHASE_COLORS[s] for s in PHASE_LABELS],
                    width=1,
                ),
                opacity=0.85,
            ),
            error_y=dict(
                type="data",
                array=hrv_stds,
                visible=True,
                color=TEXT_SECONDARY,
                thickness=1.5,
            ),
            customdata=list(zip(hrv_ns, hrv_stds)),
            hovertemplate="<b>%{x}</b><br>RMSSD: %{y:.1f} ms<br>SD: %{customdata[1]:.1f} ms<br>n=%{customdata[0]}<extra></extra>",
        ),
        row=4,
        col=2,
    )

    # ---- Panels 9a/9b: Pre/Post Ruxolitinib ----
    comps = rux_comparison["comparisons"]
    frag_comp = next((c for c in comps if c["metric"] == "fragmentation_index"), None)
    eff_comp = next((c for c in comps if c["metric"] == "sleep_efficiency_%"), None)

    if frag_comp and frag_comp.get("pre_mean") is not None:
        labels = ["Pre-Ruxolitinib", "Post-Ruxolitinib"]
        vals = [frag_comp.get("pre_mean", 0), frag_comp.get("post_mean", 0)]
        colors = [C_BLUE, C_RUX]
        p_val = frag_comp.get("p_value")
        p_text = f"p={p_val:.4f}" if p_val is not None else ""
        fig.add_trace(
            go.Bar(
                x=labels,
                y=vals,
                marker=dict(
                    color=colors,
                    line=dict(color=[C_BLUE, C_RUX], width=1),
                    opacity=0.85,
                ),
                name="Fragmentation",
                hovertemplate="<b>%{x}</b><br>Fragmentation: %{y:.2f} transitions/hour<extra></extra>",
            ),
            row=5,
            col=1,
        )
        if frag_comp.get("pre_std"):
            post_std = frag_comp.get("post_std", 0) if frag_comp.get("post_std") else 0
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=vals,
                    error_y=dict(
                        type="data",
                        array=[frag_comp["pre_std"], post_std],
                        visible=True,
                        color=TEXT_SECONDARY,
                        thickness=1.5,
                    ),
                    mode="markers",
                    marker=dict(size=0.1, color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=5,
                col=1,
            )
        # Add p-value annotation if available
        if p_text:
            fig.add_annotation(
                x=0.5,
                y=1.0,
                xref="x9 domain",
                yref="y9 domain",
                text=p_text,
                showarrow=False,
                font=dict(size=10, color=TEXT_SECONDARY),
            )

    if eff_comp and eff_comp.get("pre_mean") is not None:
        labels = ["Pre-Ruxolitinib", "Post-Ruxolitinib"]
        vals = [eff_comp.get("pre_mean", 0), eff_comp.get("post_mean", 0)]
        colors = [C_BLUE, C_RUX]
        p_val = eff_comp.get("p_value")
        p_text = f"p={p_val:.4f}" if p_val is not None else ""
        fig.add_trace(
            go.Bar(
                x=labels,
                y=vals,
                marker=dict(
                    color=colors,
                    line=dict(color=[C_BLUE, C_RUX], width=1),
                    opacity=0.85,
                ),
                name="Efficiency",
                hovertemplate="<b>%{x}</b><br>Efficiency: %{y:.1f}%<extra></extra>",
            ),
            row=5,
            col=2,
        )
        if p_text:
            fig.add_annotation(
                x=0.5,
                y=1.0,
                xref="x10 domain",
                yref="y10 domain",
                text=p_text,
                showarrow=False,
                font=dict(size=10, color=TEXT_SECONDARY),
            )

    # ---- Layout ----
    fig.update_layout(
        height=2600,
        width=1400,
        margin=dict(l=60, r=30, t=120, b=40),
        showlegend=True,
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
            bgcolor="rgba(26, 29, 39, 0.8)",
            bordercolor=BORDER_SUBTLE,
            borderwidth=1,
        ),
    )

    # Global: subtle dotted gridlines, no zeroline where not meaningful
    fig.update_xaxes(
        showgrid=True,
        gridcolor=BORDER_SUBTLE,
        griddash="dot",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=BORDER_SUBTLE,
        linewidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=BORDER_SUBTLE,
        griddash="dot",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=BORDER_SUBTLE,
        linewidth=1,
    )

    # Crosshair spikes on time-series panels
    for row, col in [(1, 1), (2, 1), (3, 2)]:
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor=TEXT_TERTIARY,
            spikedash="dot",
            row=row,
            col=col,
        )
        fig.update_yaxes(
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor=TEXT_TERTIARY,
            spikedash="dot",
            row=row,
            col=col,
        )

    # Axis labels with color matching
    fig.update_yaxes(title_text="Transitions/hour", row=1, col=1)
    fig.update_yaxes(title_text="From stage", row=1, col=2)
    fig.update_xaxes(title_text="To stage", row=1, col=2)
    fig.update_yaxes(title_text="Minutes", title_font_color=ACCENT_PURPLE, row=2, col=1)
    fig.update_yaxes(title_text="Cycle count", row=2, col=2)
    fig.update_yaxes(title_text="Movement count", row=3, col=1)
    fig.update_yaxes(
        title_text="Efficiency (%)", title_font_color=ACCENT_BLUE, row=3, col=2
    )
    fig.update_yaxes(title_text="Spectral power", row=4, col=1)
    fig.update_xaxes(title_text="Period (minutes)", row=4, col=1)
    fig.update_yaxes(title_text="RMSSD (ms)", row=4, col=2)
    fig.update_yaxes(title_text="Fragmentation index", row=5, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=5, col=2)

    # Date axis formatting for time series panels -- consistent format
    for row, col in [(1, 1), (2, 1), (2, 2), (3, 2)]:
        fig.update_xaxes(
            tickformat="%d %b",
            tickangle=-30,
            row=row,
            col=col,
        )

    # Disable grid on heatmap and categorical panels where it adds noise
    fig.update_xaxes(showgrid=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, row=1, col=2)

    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Summary & clinical interpretation
# ---------------------------------------------------------------------------
def build_summary(
    frag_df: pd.DataFrame,
    trans_probs: np.ndarray,
    rem_latency_df: pd.DataFrame,
    cycles_df: pd.DataFrame,
    movement_data: dict,
    efficiency_df: pd.DataFrame,
    ultradian: dict,
    hrv_coupling: dict,
    rux_comparison: dict,
    periods: pd.DataFrame,
) -> dict:
    """Build structured summary metrics dict for JSON export and stdout."""
    long_periods = periods[periods["type"] == "long_sleep"]
    long_pids = set(long_periods["period_id"])

    frag_long = frag_df[frag_df["period_id"].isin(long_pids)]
    rem_long = rem_latency_df[
        rem_latency_df["period_id"].isin(long_pids) & rem_latency_df["has_rem"]
    ]
    cycles_long = cycles_df[cycles_df["period_id"].isin(long_pids)]
    eff_long = efficiency_df[efficiency_df["period_id"].isin(long_pids)]
    day_series = pd.to_datetime(periods["day"], errors="coerce").dropna()
    data_start = day_series.min().date().isoformat() if not day_series.empty else None
    data_end = day_series.max().date().isoformat() if not day_series.empty else None
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data_range_label = f"{data_start} to {data_end}" if data_start and data_end else ""

    summary: dict[str, Any] = {
        "generated_at": generated_at,
        "data_range": {
            "start": data_start,
            "end": data_end,
            "total_nights": int(len(periods)),
            "long_sleep_nights": int(len(long_periods)),
        },
        "meta": {
            "patient": PATIENT_LABEL,
            "data_range": data_range_label,
            "total_nights": len(periods),
            "long_sleep_nights": len(long_periods),
            "ruxolitinib_start": str(TREATMENT_START),
            "generated": generated_at,
            "generated_at": generated_at,
        },
    }

    # 1. Fragmentation
    if not frag_long.empty:
        fi_vals = frag_long["fragmentation_index"].dropna()
        summary["fragmentation"] = {
            "mean": round(float(fi_vals.mean()), 2),
            "median": round(float(fi_vals.median()), 2),
            "std": round(float(fi_vals.std()), 2),
            "min": round(float(fi_vals.min()), 2),
            "max": round(float(fi_vals.max()), 2),
            "pct_above_clinical": round(
                float(
                    (fi_vals > NORMS["fragmentation_index"]["clinical_concern"]).mean()
                    * 100
                ),
                1,
            ),
            "healthy_norm": NORMS["fragmentation_index"]["healthy"],
            "clinical_concern_threshold": NORMS["fragmentation_index"][
                "clinical_concern"
            ],
            "unit": "transitions/hour",
        }

    # 2. Transition matrix
    summary["transition_matrix"] = {
        "labels": PHASE_LABELS,
        "probabilities": trans_probs.round(4).tolist(),
        "healthy_norms": HEALTHY_TRANSITION_PROBS.round(4).tolist(),
        "key_deviations": [],
    }
    for i, fr_stage in enumerate(PHASE_LABELS):
        for j, to_stage in enumerate(PHASE_LABELS):
            diff = trans_probs[i, j] - HEALTHY_TRANSITION_PROBS[i, j]
            if abs(diff) >= 0.10:
                summary["transition_matrix"]["key_deviations"].append(
                    {
                        "from": fr_stage,
                        "to": to_stage,
                        "patient": round(float(trans_probs[i, j]), 3),
                        "healthy": round(float(HEALTHY_TRANSITION_PROBS[i, j]), 3),
                        "difference": round(float(diff), 3),
                    }
                )

    # 3. REM latency
    if not rem_long.empty:
        rl_vals = rem_long["rem_latency_min"].dropna()
        summary["rem_latency"] = {
            "mean_min": round(float(rl_vals.mean()), 1),
            "median_min": round(float(rl_vals.median()), 1),
            "std_min": round(float(rl_vals.std()), 1),
            "pct_elevated": round(
                float((rl_vals > NORMS["rem_latency"]["elevated"]).mean() * 100), 1
            ),
            "nights_without_rem": int(
                (
                    ~rem_latency_df[rem_latency_df["period_id"].isin(long_pids)][
                        "has_rem"
                    ]
                ).sum()
            ),
            "elevated_threshold_min": NORMS["rem_latency"]["elevated"],
        }

    # 4. Sleep cycles
    if not cycles_long.empty:
        cc_vals = cycles_long["cycle_count"]
        dur_vals = cycles_long["mean_cycle_duration_min"].dropna()
        summary["sleep_cycles"] = {
            "mean_cycles_per_night": round(float(cc_vals.mean()), 1),
            "median_cycles_per_night": round(float(cc_vals.median()), 1),
            "pct_below_4_cycles": round(float((cc_vals < 4).mean() * 100), 1),
            "mean_cycle_duration_min": round(float(dur_vals.mean()), 1)
            if not dur_vals.empty
            else None,
            "healthy_range": f"{NORMS['cycles_per_night']['healthy_min']}-{NORMS['cycles_per_night']['healthy_max']}",
        }

    # 5. Movement density
    rest_long = movement_data["nightly"][
        movement_data["nightly"]["period_id"].isin(long_pids)
    ]
    if not rest_long.empty:
        ri_vals = rest_long["restlessness_index"]
        summary["movement"] = {
            "mean_restlessness_index": round(float(ri_vals.mean()), 4),
            "median_restlessness_index": round(float(ri_vals.median()), 4),
            "stage_movement_totals": movement_data["stage_movement"],
            "hourly_density": movement_data["hourly_density"],
        }

    # 6. Efficiency
    if not eff_long.empty:
        oura_eff = eff_long["efficiency"].dropna()
        epoch_eff = eff_long["epoch_efficiency"].dropna()
        summary["efficiency"] = {
            "oura_mean": round(float(oura_eff.mean()), 1),
            "oura_median": round(float(oura_eff.median()), 1),
            "epoch_mean": round(float(epoch_eff.mean()), 1)
            if not epoch_eff.empty
            else None,
            "pct_below_75": round(float((oura_eff < 75).mean() * 100), 1),
            "pct_below_85": round(float((oura_eff < 85).mean() * 100), 1),
            "healthy_threshold": NORMS["efficiency"]["healthy"],
        }

    # 7. Ultradian
    summary["ultradian_rhythm"] = ultradian

    # 8. HRV coupling
    coupling_summary = {
        "stage_stats": {k: v for k, v in hrv_coupling["stage_stats"].items()},
        "coupling_correct_deep_gt_light": hrv_coupling[
            "coupling_correct_deep_gt_light"
        ],
        "kruskal_wallis": hrv_coupling["kruskal_wallis"],
        "matched_hrv_epochs": hrv_coupling["matched_hrv_epochs"],
    }
    summary["hrv_coupling"] = coupling_summary

    # 9. Pre/post ruxolitinib
    summary["ruxolitinib_comparison"] = rux_comparison

    # Clinical interpretation
    interpretations = []

    # Fragmentation
    if "fragmentation" in summary:
        fi_mean = summary["fragmentation"]["mean"]
        if fi_mean > NORMS["fragmentation_index"]["clinical_concern"]:
            interpretations.append(
                f"SEVERE: Mean fragmentation index {fi_mean}/hr exceeds clinical "
                f"concern threshold ({NORMS['fragmentation_index']['clinical_concern']}/hr). "
                "Indicates frequent awakenings disrupting sleep cycles."
            )
        elif fi_mean > NORMS["fragmentation_index"]["healthy"]:
            interpretations.append(
                f"ELEVATED: Mean fragmentation index {fi_mean}/hr is above "
                f"healthy norm ({NORMS['fragmentation_index']['healthy']}/hr)."
            )

    # REM latency
    if "rem_latency" in summary:
        rl_pct = summary["rem_latency"]["pct_elevated"]
        if rl_pct > 30:
            interpretations.append(
                f"PATHOLOGICAL: {rl_pct}% of nights have REM latency >120 minutes. "
                "Elevated REM latency may indicate autonomic dysfunction or medication effects."
            )

    # Sleep cycles
    if "sleep_cycles" in summary:
        pct_low = summary["sleep_cycles"]["pct_below_4_cycles"]
        if pct_low > 50:
            interpretations.append(
                f"REDUCED: {pct_low}% of nights have <4 cycles (expected 4-6). "
                "Incomplete sleep cycles result in insufficient restoration."
            )

    # Efficiency
    if "efficiency" in summary:
        pct_poor = summary["efficiency"]["pct_below_75"]
        if pct_poor > 30:
            interpretations.append(
                f"POOR: {pct_poor}% of nights have sleep efficiency <75%. "
                "Persistently low efficiency is associated with insomnia criteria."
            )

    # HRV coupling
    if not hrv_coupling["coupling_correct_deep_gt_light"]:
        interpretations.append(
            "ABNORMAL: HRV is NOT higher during deep sleep than during light sleep. "
            "Normal parasympathetic dominance during deep sleep is absent - "
            "suggests autonomic dysfunction."
        )

    # Ultradian
    if ultradian["dominant_period_min"]:
        dom = ultradian["dominant_period_min"]
        if abs(dom - 90) > 20:
            interpretations.append(
                f"IRREGULAR: Dominant ultradian period is {dom} min (expected ~90 min). "
                "Deviation from normal 90-minute cycle suggests fragmented sleep architecture."
            )

    summary["clinical_interpretation"] = interpretations

    return summary


# ---------------------------------------------------------------------------
# Print summary to stdout
# ---------------------------------------------------------------------------
def print_summary(summary: dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 80)
    print(f"  ADVANCED SLEEP ARCHITECTURE ANALYSIS - {PATIENT_LABEL.upper()}")
    print("=" * 80)
    print(f"  Data: {summary['meta']['data_range']}")
    print(
        f"  Nights: {summary['meta']['total_nights']} total, "
        f"{summary['meta']['long_sleep_nights']} long sleep"
    )
    print(f"  Ruxolitinib start: {summary['meta']['ruxolitinib_start']}")
    print("=" * 80)

    if "fragmentation" in summary:
        f = summary["fragmentation"]
        print("\n--- 1. SLEEP FRAGMENTATION ---")
        print(
            f"  Mean: {f['mean']}/hr (healthy: <{f['healthy_norm']}, "
            f"clinical: >{f['clinical_concern_threshold']})"
        )
        print(f"  Median: {f['median']}/hr | SD: {f['std']}")
        print(f"  Range: {f['min']} - {f['max']}/hr")
        print(f"  Nights above clinical threshold: {f['pct_above_clinical']}%")

    if "transition_matrix" in summary:
        tm = summary["transition_matrix"]
        print("\n--- 2. TRANSITION MATRIX (Markov) ---")
        print("  Patient vs healthy (probabilities):")
        for dev in tm["key_deviations"]:
            direction = "+" if dev["difference"] > 0 else ""
            print(
                f"    {dev['from']} -> {dev['to']}: {dev['patient']:.3f} "
                f"(healthy: {dev['healthy']:.3f}, diff: {direction}{dev['difference']:.3f})"
            )

    if "rem_latency" in summary:
        rl = summary["rem_latency"]
        print("\n--- 3. REM LATENCY ---")
        print(f"  Mean: {rl['mean_min']} min | Median: {rl['median_min']} min")
        print(f"  Elevated (>120 min): {rl['pct_elevated']}% of nights")
        print(f"  Nights without REM: {rl['nights_without_rem']}")

    if "sleep_cycles" in summary:
        sc = summary["sleep_cycles"]
        print("\n--- 4. SLEEP CYCLES ---")
        print(
            f"  Mean: {sc['mean_cycles_per_night']} cycles/night "
            f"(healthy: {sc['healthy_range']})"
        )
        print(f"  Below 4 cycles: {sc['pct_below_4_cycles']}% of nights")
        if sc["mean_cycle_duration_min"]:
            print(
                f"  Average cycle duration: {sc['mean_cycle_duration_min']} min "
                f"(expected 80-100)"
            )

    if "movement" in summary:
        mv = summary["movement"]
        print("\n--- 5. MOVEMENT ANALYSIS ---")
        print(f"  Mean restlessness index: {mv['mean_restlessness_index']:.4f}")

    if "efficiency" in summary:
        ef = summary["efficiency"]
        print("\n--- 6. SLEEP EFFICIENCY ---")
        print(
            f"  Oura mean: {ef['oura_mean']}% | Epoch-based: {ef.get('epoch_mean', 'N/A')}%"
        )
        print(f"  Below 75%: {ef['pct_below_75']}% | Below 85%: {ef['pct_below_85']}%")

    print("\n--- 7. ULTRADIAN RHYTHM ---")
    ul = summary["ultradian_rhythm"]
    if ul["dominant_period_min"]:
        print(
            f"  Dominant period: {ul['dominant_period_min']} min "
            f"(expected: ~90 min) over {ul['n_nights']} nights"
        )
        if ul["spectral_peaks"]:
            print("  Top spectral peaks:")
            for p in ul["spectral_peaks"][:3]:
                print(
                    f"    {p['period_min']} min (power: {p['mean_power']:.4f}, "
                    f"n={p['count']} nights)"
                )
    else:
        print("  Insufficient data for FFT analysis")

    print("\n--- 8. HRV-SLEEP COUPLING ---")
    hc = summary["hrv_coupling"]
    for stage in PHASE_LABELS:
        ss = hc["stage_stats"].get(stage, {})
        if ss.get("mean_rmssd"):
            print(
                f"  {stage}: RMSSD = {ss['mean_rmssd']:.1f} ms "
                f"(SD {ss.get('std_rmssd', 0):.1f}, n={ss['n']})"
            )
    kw = hc["kruskal_wallis"]
    if kw["H"]:
        sig = (
            "***"
            if kw["p"] < 0.001
            else "**"
            if kw["p"] < 0.01
            else "*"
            if kw["p"] < 0.05
            else "ns"
        )
        print(f"  Kruskal-Wallis: H={kw['H']}, p={kw['p']:.4e} ({sig})")
    print(
        f"  Coupling deep>light: {'YES' if hc['coupling_correct_deep_gt_light'] else 'NO (abnormal)'}"
    )

    print("\n--- 9. PRE/POST RUXOLITINIB ---")
    rux = summary["ruxolitinib_comparison"]
    print(f"  Cutoff: {rux['cutoff_date']}")
    for comp in rux["comparisons"]:
        pre_n = comp.get("pre_n", 0)
        post_n = comp.get("post_n", 0)
        pre_m = comp.get("pre_mean")
        post_m = comp.get("post_mean")
        if pre_m is not None:
            line = f"  {comp['metric']}: pre={pre_m} (n={pre_n})"
            if post_m is not None:
                line += f", post={post_m} (n={post_n})"
            if comp.get("post_percentile_in_pre") is not None:
                line += f" [percentile in pre: {comp['post_percentile_in_pre']}%]"
            if comp.get("p_value") is not None:
                line += f", p={comp['p_value']:.4f}"
            if comp.get("cliffs_delta") is not None:
                line += f", d={comp['cliffs_delta']}"
            print(line)

    if summary.get("clinical_interpretation"):
        print(f"\n{'=' * 80}")
        print("  CLINICAL INTERPRETATION")
        print("=" * 80)
        for interp in summary["clinical_interpretation"]:
            print(f"\n  * {interp}")

    print(f"\n{'=' * 80}")
    print("  Report: reports/advanced_sleep_analysis.html")
    print("  Metrics: reports/advanced_sleep_metrics.json")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# HTML assembly using _theme design system
# ---------------------------------------------------------------------------
def assemble_html(plotly_div: str, summary: dict) -> str:
    """Build full HTML page using the dark theme design system."""
    meta = summary.get("meta", {})
    interps = summary.get("clinical_interpretation", [])

    # --- KPI cards from summary ---
    kpi_cards = []

    if "fragmentation" in summary:
        f = summary["fragmentation"]
        frag_status = (
            "critical"
            if f["mean"] > NORMS["fragmentation_index"]["clinical_concern"]
            else (
                "warning"
                if f["mean"] > NORMS["fragmentation_index"]["healthy"]
                else "normal"
            )
        )
        frag_label = "Elevated" if frag_status in ("warning", "critical") else ""
        kpi_cards.append(
            make_kpi_card(
                "Fragmentation",
                f["mean"],
                "transitions/hour",
                status=frag_status,
                status_label=frag_label,
                detail=f"{f['pct_above_clinical']}% above clinical threshold",
            )
        )

    if "efficiency" in summary:
        ef = summary["efficiency"]
        eff_status = (
            "critical"
            if ef["oura_mean"] < 75
            else ("warning" if ef["oura_mean"] < 85 else "normal")
        )
        eff_label = "Low" if eff_status in ("warning", "critical") else ""
        kpi_cards.append(
            make_kpi_card(
                "Sleep Efficiency",
                ef["oura_mean"],
                "%",
                status=eff_status,
                status_label=eff_label,
                detail=f"{ef['pct_below_85']}% of nights below 85%",
            )
        )

    if "sleep_cycles" in summary:
        sc = summary["sleep_cycles"]
        cyc_status = "warning" if sc["pct_below_4_cycles"] > 50 else "normal"
        cyc_label = "Low" if cyc_status in ("warning", "critical") else ""
        kpi_cards.append(
            make_kpi_card(
                "Sleep Cycles",
                sc["mean_cycles_per_night"],
                "/night",
                status=cyc_status,
                status_label=cyc_label,
                detail=f"{sc['pct_below_4_cycles']}% below 4 cycles",
            )
        )

    if "rem_latency" in summary:
        rl = summary["rem_latency"]
        rl_status = (
            "critical"
            if rl["pct_elevated"] > 30
            else ("warning" if rl["pct_elevated"] > 15 else "normal")
        )
        rl_label = "Elevated" if rl_status in ("warning", "critical") else ""
        kpi_cards.append(
            make_kpi_card(
                "REM Latency",
                rl["mean_min"],
                "min",
                status=rl_status,
                status_label=rl_label,
                detail=f"{rl['pct_elevated']}% above 120 min",
            )
        )

    if "hrv_coupling" in summary:
        hc = summary["hrv_coupling"]
        coupling_ok = hc["coupling_correct_deep_gt_light"]
        coupling_label = "" if coupling_ok else "Abnormal"
        kpi_cards.append(
            make_kpi_card(
                "HRV Coupling",
                "Normal" if coupling_ok else "Abnormal",
                "",
                status="normal" if coupling_ok else "critical",
                status_label=coupling_label,
                detail="Deep > light parasympathetic"
                if coupling_ok
                else "Abnormal autonomic regulation",
                decimals=0,
            )
        )

    if (
        "ultradian_rhythm" in summary
        and summary["ultradian_rhythm"]["dominant_period_min"]
    ):
        ul = summary["ultradian_rhythm"]
        dom = ul["dominant_period_min"]
        ul_status = "normal" if abs(dom - 90) <= 20 else "warning"
        ul_label = "Abnormal" if ul_status in ("warning", "critical") else ""
        kpi_cards.append(
            make_kpi_card(
                "Ultradian Period",
                dom,
                "min",
                status=ul_status,
                status_label=ul_label,
                detail=f"Expected ~90 min ({ul['n_nights']} nights)",
            )
        )

    body = make_kpi_row(*kpi_cards)

    # --- Clinical interpretation callout ---
    if interps:
        interp_items = "\n".join(
            f'<li style="color: {ACCENT_RED}; margin-bottom: 6px;">{i}</li>'
            for i in interps
        )
        callout = (
            f'<div class="odt-narrative">'
            f"<strong>Clinical Interpretation</strong>"
            f'<ul style="margin-top: 8px; padding-left: 20px;">{interp_items}</ul>'
            f"</div>"
        )
        body += callout

    # --- Main dashboard chart ---
    body += make_section(
        "Sleep Architecture Analysis (9 panels)",
        plotly_div,
        section_id="dashboard",
    )

    # --- Data range subtitle ---
    subtitle = (
        f"{meta.get('long_sleep_nights', '?')} nights analyzed | "
        f"{meta.get('data_range', '')} | "
        f"Ruxolitinib start: {meta.get('ruxolitinib_start', TREATMENT_START_STR)}"
    )

    return _theme_wrap_html(
        title="Advanced Sleep Architecture Analysis",
        body_content=body,
        report_id="sleep",
        subtitle=subtitle,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Run all analyses and produce dashboard + JSON."""
    print("Connecting to database...")
    conn = get_connection()

    try:
        print("Loading data...")
        periods = load_sleep_periods(conn)
        epochs = load_epochs(conn)
        movements = load_movements(conn)
        hrv = load_hrv(conn)
        hr_ts = load_hr_timeseries(conn)

        print(
            f"  Periods: {len(periods)}, Epochs: {len(epochs)}, "
            f"Movements: {len(movements)}, HRV: {len(hrv)}, HR-ts: {len(hr_ts)}"
        )

        # 1. Fragmentation
        print("Analysis 1/9: Sleep fragmentation index...")
        frag_df = compute_fragmentation_index(epochs)

        # 2. Transition matrix
        print("Analysis 2/9: Transition matrix...")
        trans_counts, trans_probs = compute_transition_matrix(epochs)

        # 3. REM latency
        print("Analysis 3/9: REM latency...")
        rem_latency_df = compute_rem_latency(epochs)

        # 4. Sleep cycles
        print("Analysis 4/9: Sleep cycle detection...")
        cycles_df = detect_sleep_cycles(epochs)

        # 5. Movement density
        print("Analysis 5/9: Movement density...")
        movement_data = compute_movement_density(movements, epochs)

        # 6. Efficiency trends
        print("Analysis 6/9: Sleep efficiency trends...")
        efficiency_df = compute_efficiency_trends(periods, epochs)

        # 7. Ultradian rhythm
        print("Analysis 7/9: Ultradian rhythm analysis (FFT)...")
        ultradian = compute_ultradian_rhythm(epochs)

        # 8. HRV coupling
        print("Analysis 8/9: Sleep-HRV coupling...")
        hrv_coupling = compute_hrv_coupling(epochs, hrv, periods)

        # 9. Pre/post ruxolitinib
        print("Analysis 9/9: Pre/post ruxolitinib comparison...")
        rux_comparison = compare_pre_post_ruxolitinib(
            periods,
            frag_df,
            rem_latency_df,
            cycles_df,
            efficiency_df,
            movement_data["nightly"],
        )
    finally:
        conn.close()

    # Build summary
    summary = build_summary(
        frag_df,
        trans_probs,
        rem_latency_df,
        cycles_df,
        movement_data,
        efficiency_df,
        ultradian,
        hrv_coupling,
        rux_comparison,
        periods,
    )

    # Save JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON saved: {JSON_OUT}")

    # Build and save dashboard
    print("Building interactive dashboard...")
    plotly_div = build_dashboard(
        frag_df,
        trans_counts,
        trans_probs,
        rem_latency_df,
        cycles_df,
        movement_data,
        efficiency_df,
        ultradian,
        hrv_coupling,
        rux_comparison,
        periods,
    )
    html = assemble_html(plotly_div, summary)
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved: {HTML_OUT}")

    # Print summary
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
