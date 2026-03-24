#!/usr/bin/env python3
"""
Bayesian Cardiovascular Digital Twin from Oura Ring Data

Implements a state-space model for continuous physiological state estimation.
See config.py for patient details.

Models 5 latent physiological states from wearable sensor data:
  1. Autonomic tone (parasympathetic dominance)
  2. Cardiac reserve (cardiovascular fitness)
  3. Circadian phase (circadian rhythm alignment)
  4. Inflammation level (systemic inflammation proxy)
  5. Sleep quality (composite recovery)

Key analyses:
  - Linear Kalman filter/smoother with EM-learned parameters
  - Unscented Kalman filter (UKF) for nonlinear circadian dynamics
  - One-step-ahead prediction with innovation monitoring
  - Ruxolitinib (started 2026-03-16) response quantification
  - Multi-modal sensor fusion quality assessment

Output:
  - Interactive HTML report: reports/digital_twin_report.html
  - JSON metrics: reports/digital_twin_metrics.json

Usage:
    python analysis/analyze_oura_digital_twin.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import traceback
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Path resolution & patient config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATABASE_PATH, REPORTS_DIR, TRANSPLANT_DATE, TREATMENT_START, KNOWN_EVENT_DATE,
    HEV_DIAGNOSIS_DATE, PATIENT_AGE, PATIENT_LABEL,
    ESC_RMSSD_DEFICIENCY, NOCTURNAL_HR_ELEVATED, POPULATION_RMSSD_MEDIAN, HSCT_RMSSD_RANGE,
)
from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    BG_ELEVATED, BORDER_SUBTLE,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    ACCENT_PURPLE, ACCENT_CYAN, ACCENT_ORANGE,
    C_PRE_TX,
)

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "digital_twin_report.html"
JSON_OUTPUT = REPORTS_DIR / "digital_twin_metrics.json"

# Clinical reference values — imported from config.py
HSCT_TYPICAL_RMSSD = HSCT_RMSSD_RANGE  # local alias for backwards compat

# State-space model parameters
N_STATES = 5  # latent state dimension
N_OBS = 5  # observation dimension
EM_ITERATIONS = 15  # EM algorithm iterations for parameter learning
PREDICTION_HORIZON = 7  # days ahead for forecasting
ALERT_SIGMA = 2.0  # innovation threshold for alerts

# State labels
STATE_NAMES = [
    "Autonomic Tone",
    "Cardiac Reserve",
    "Circadian Phase",
    "Inflammation Level",
    "Sleep Quality",
]
OBS_NAMES = [
    "HRV (RMSSD)",
    "Heart Rate",
    "SpO2",
    "Temperature Dev.",
    "Sleep Efficiency",
]

# Visualization — mapped to dark theme palette
COLORS = {
    "pre": C_PRE_TX,
    "post": ACCENT_RED,
    "hrv": ACCENT_GREEN,
    "hr": ACCENT_PURPLE,
    "spo2": ACCENT_CYAN,
    "temp": ACCENT_ORANGE,
    "sleep": ACCENT_BLUE,
    "ci": "rgba(59, 130, 246, 0.12)",
    "predicted": ACCENT_GREEN,
    "actual": ACCENT_BLUE,
    "innovation": ACCENT_RED,
    "alert": ACCENT_RED,
}

TIMELINE_MARKERS = [
    (pd.Timestamp(KNOWN_EVENT_DATE), "Acute event", ACCENT_RED, "dot"),
    (pd.Timestamp(TREATMENT_START), "Ruxolitinib start", COLORS["post"], "dash"),
    (pd.Timestamp(HEV_DIAGNOSIS_DATE), "HEV diagnosis", ACCENT_AMBER, "longdashdot"),
]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string."""
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


def _add_timeline_vlines(
    fig: go.Figure,
    positions: list[tuple[int, int]],
    *,
    line_width: float = 1.25,
    opacity: float = 0.8,
) -> None:
    """Add shared timeline markers to a set of subplot positions."""
    for row, col in positions:
        for when, _, color, dash in TIMELINE_MARKERS:
            fig.add_vline(
                x=when,
                row=row,
                col=col,
                line_color=color,
                line_dash=dash,
                line_width=line_width,
                opacity=opacity,
            )


# ---------------------------------------------------------------------------
# Global metrics collector
# ---------------------------------------------------------------------------
metrics: dict[str, Any] = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "patient_age": PATIENT_AGE,
    "ruxolitinib_start": str(TREATMENT_START),
    "data_range": {},
    "kalman": {},
    "ukf": {},
    "prediction": {},
    "drug_response": {},
    "sensor_fusion": {},
}
figures: list[go.Figure] = []


def log(prefix: str, msg: str) -> None:
    """Print timestamped progress message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{prefix}] {ts} - {msg}")


# ===========================================================================
# Section 1: Data Loading & Preparation
# ===========================================================================
def load_oura_data() -> pd.DataFrame:
    """Load and aggregate Oura Ring data into daily observations.

    Returns a DataFrame indexed by date with columns:
        mean_rmssd, mean_hr, spo2_average, temperature_deviation, sleep_efficiency
    """
    log("DATA", "Loading Oura Ring data from database...")

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)

    # --- HRV: aggregate 5-minute samples to daily mean ---
    hrv_df = pd.read_sql_query(
        """
        SELECT substr(timestamp, 1, 10) AS day, AVG(rmssd) AS mean_rmssd
        FROM oura_hrv
        WHERE rmssd IS NOT NULL AND rmssd > 0
        GROUP BY substr(timestamp, 1, 10)
        ORDER BY day
        """,
        conn,
    )
    hrv_df["day"] = pd.to_datetime(hrv_df["day"])
    log("DATA", f"  HRV: {len(hrv_df)} daily aggregates")

    # --- Heart rate: aggregate to daily mean ---
    hr_df = pd.read_sql_query(
        """
        SELECT substr(timestamp, 1, 10) AS day, AVG(bpm) AS mean_hr
        FROM oura_heart_rate
        WHERE bpm IS NOT NULL AND bpm > 0
        GROUP BY substr(timestamp, 1, 10)
        ORDER BY day
        """,
        conn,
    )
    hr_df["day"] = pd.to_datetime(hr_df["day"])
    log("DATA", f"  HR: {len(hr_df)} daily aggregates")

    # --- SpO2: already daily ---
    spo2_df = pd.read_sql_query(
        """
        SELECT date AS day, spo2_average
        FROM oura_spo2
        WHERE spo2_average IS NOT NULL AND spo2_average > 0
        ORDER BY date
        """,
        conn,
    )
    spo2_df["day"] = pd.to_datetime(spo2_df["day"])
    log("DATA", f"  SpO2: {len(spo2_df)} valid days")

    # --- Temperature deviation from readiness ---
    temp_df = pd.read_sql_query(
        """
        SELECT date AS day, temperature_deviation
        FROM oura_readiness
        WHERE temperature_deviation IS NOT NULL
        ORDER BY date
        """,
        conn,
    )
    temp_df["day"] = pd.to_datetime(temp_df["day"])
    log("DATA", f"  Temp deviation: {len(temp_df)} days")

    # --- Sleep efficiency from long_sleep periods ---
    sleep_df = pd.read_sql_query(
        """
        SELECT day, efficiency AS sleep_efficiency,
               average_hrv, average_heart_rate, lowest_heart_rate,
               total_sleep_duration, rem_sleep_duration, deep_sleep_duration
        FROM oura_sleep_periods
        WHERE type = 'long_sleep'
        ORDER BY day
        """,
        conn,
    )
    sleep_df["day"] = pd.to_datetime(sleep_df["day"])
    # Handle duplicate days (multiple sleep periods): take the longest
    sleep_df = sleep_df.sort_values("total_sleep_duration", ascending=False)
    sleep_df = sleep_df.drop_duplicates(subset=["day"], keep="first")
    log("DATA", f"  Sleep: {len(sleep_df)} nights")

    conn.close()

    # --- Merge into daily panel ---
    # Create full date range
    all_dates = pd.date_range(
        start=min(
            hrv_df["day"].min(),
            hr_df["day"].min(),
            spo2_df["day"].min(),
            temp_df["day"].min(),
            sleep_df["day"].min(),
        ),
        end=max(
            hrv_df["day"].max(),
            hr_df["day"].max(),
            spo2_df["day"].max(),
            temp_df["day"].max(),
            sleep_df["day"].max(),
        ),
        freq="D",
    )
    daily = pd.DataFrame({"day": all_dates})

    # Merge each source
    daily = daily.merge(hrv_df[["day", "mean_rmssd"]], on="day", how="left")
    daily = daily.merge(hr_df[["day", "mean_hr"]], on="day", how="left")
    daily = daily.merge(spo2_df[["day", "spo2_average"]], on="day", how="left")
    daily = daily.merge(
        temp_df[["day", "temperature_deviation"]], on="day", how="left"
    )
    daily = daily.merge(
        sleep_df[["day", "sleep_efficiency", "average_hrv", "lowest_heart_rate",
                   "rem_sleep_duration", "deep_sleep_duration", "total_sleep_duration"]],
        on="day",
        how="left",
    )

    daily = daily.set_index("day").sort_index()

    # Report completeness
    n_total = len(daily)
    for col in ["mean_rmssd", "mean_hr", "spo2_average", "temperature_deviation", "sleep_efficiency"]:
        n_valid = daily[col].notna().sum()
        log("DATA", f"  {col}: {n_valid}/{n_total} days ({100*n_valid/n_total:.0f}%)")

    metrics["data_range"] = {
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "n_days": n_total,
        "completeness": {
            col: float(daily[col].notna().mean())
            for col in ["mean_rmssd", "mean_hr", "spo2_average",
                         "temperature_deviation", "sleep_efficiency"]
        },
    }

    return daily


# ===========================================================================
# Section 2: Observation Standardization
# ===========================================================================
def standardize_observations(
    daily: pd.DataFrame,
) -> tuple[np.ma.MaskedArray, dict[str, tuple[float, float]]]:
    """Standardize observations to zero mean, unit variance.

    Returns:
        obs_masked: (n_days, 5) masked array (NaN -> masked)
        scalers: dict mapping column -> (mean, std)
    """
    log("DATA", "Standardizing observations...")

    obs_cols = ["mean_rmssd", "mean_hr", "spo2_average",
                "temperature_deviation", "sleep_efficiency"]
    obs_raw = daily[obs_cols].values.astype(np.float64)

    scalers: dict[str, tuple[float, float]] = {}
    obs_std = np.full_like(obs_raw, np.nan)

    for i, col in enumerate(obs_cols):
        vals = obs_raw[:, i]
        valid = ~np.isnan(vals)
        if valid.sum() > 1:
            mu = np.nanmean(vals)
            sigma = np.nanstd(vals)
            if sigma < 1e-10:
                sigma = 1.0
            scalers[col] = (float(mu), float(sigma))
            obs_std[valid, i] = (vals[valid] - mu) / sigma
        else:
            scalers[col] = (0.0, 1.0)

    # Create masked array (mask where NaN)
    obs_masked = np.ma.masked_invalid(obs_std)
    n_masked = obs_masked.mask.sum() if obs_masked.mask is not np.bool_(False) else 0
    total = obs_masked.size
    log("DATA", f"  Observations: {obs_masked.shape}, "
         f"masked: {n_masked}/{total} ({100*n_masked/total:.1f}%)")

    return obs_masked, scalers


# ===========================================================================
# Section 3: Linear Kalman Filter / Smoother
# ===========================================================================
def run_kalman_filter(
    obs_masked: np.ma.MaskedArray, daily: pd.DataFrame
) -> dict[str, Any]:
    """Run linear Kalman filter with EM-learned parameters.

    State vector (5D):
        [autonomic_tone, cardiac_reserve, circadian_phase,
         inflammation_level, sleep_quality]

    The EM algorithm learns:
        - Transition covariance (Q)
        - Observation covariance (R)
        - Initial state mean and covariance

    Returns dict with filtered/smoothed states and covariances.
    """
    from pykalman import KalmanFilter

    log("KALMAN", "Initializing linear state-space model...")
    n_days = obs_masked.shape[0]

    # --- Define transition matrix A ---
    # Model slow physiological dynamics:
    # - autonomic_tone: slow mean-reversion (0.95 persistence)
    # - cardiac_reserve: very slow drift (0.98)
    # - circadian_phase: near-unit root with small coupling to autonomic tone
    # - inflammation: slow decay toward baseline (0.92)
    # - sleep_quality: moderate persistence (0.90) + coupling from autonomic tone
    A = np.array([
        [0.95, 0.00, 0.00, -0.05, 0.05],  # autonomic <- inflammation(-), sleep(+)
        [0.03, 0.98, 0.00, -0.02, 0.00],   # cardiac <- autonomic(+), inflammation(-)
        [0.00, 0.00, 0.97, 0.00, 0.00],     # circadian: slow drift
        [-0.02, 0.00, 0.00, 0.92, -0.03],   # inflammation <- autonomic(-), sleep(-)
        [0.05, 0.02, 0.03, -0.04, 0.90],    # sleep <- autonomic(+), cardiac(+), circadian(+)
    ])

    # --- Define observation matrix H ---
    # Maps latent states to observables:
    # mean_rmssd ~ autonomic_tone (+) + sleep_quality (+)
    # mean_hr ~ cardiac_reserve (-) + inflammation (+)
    # spo2 ~ cardiac_reserve (+) + inflammation (-)
    # temp_dev ~ inflammation (+) + circadian_phase (+)
    # sleep_eff ~ sleep_quality (+) + autonomic_tone (+)
    H = np.array([
        [0.70, 0.10, 0.00, -0.10, 0.30],   # rmssd
        [-0.20, -0.50, 0.00, 0.40, -0.10],  # hr (inverted: low state = high HR)
        [0.10, 0.40, 0.00, -0.50, 0.10],    # spo2
        [-0.10, 0.00, 0.30, 0.60, 0.00],    # temp deviation
        [0.30, 0.10, 0.10, -0.10, 0.60],    # sleep efficiency
    ])

    # --- Initial state ---
    initial_state_mean = np.zeros(N_STATES)
    initial_state_covariance = np.eye(N_STATES) * 1.0

    # --- Process noise (will be refined by EM) ---
    Q_init = np.eye(N_STATES) * 0.05

    # --- Observation noise (will be refined by EM) ---
    R_init = np.eye(N_OBS) * 0.3

    # --- Create Kalman filter ---
    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=H,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        transition_covariance=Q_init,
        observation_covariance=R_init,
        em_vars=[
            "transition_covariance",
            "observation_covariance",
            "initial_state_mean",
            "initial_state_covariance",
        ],
    )

    # --- EM parameter learning ---
    log("KALMAN", f"Running EM algorithm ({EM_ITERATIONS} iterations)...")
    t0 = time.time()
    kf = kf.em(obs_masked, n_iter=EM_ITERATIONS)
    em_time = time.time() - t0
    log("KALMAN", f"  EM converged in {em_time:.1f}s")

    # --- Kalman smoothing (forward-backward) ---
    log("KALMAN", "Running Kalman smoother (forward-backward)...")
    t0 = time.time()
    smoothed_means, smoothed_covs = kf.smooth(obs_masked)
    smooth_time = time.time() - t0
    log("KALMAN", f"  Smoothing complete in {smooth_time:.2f}s")

    # --- Also get filtered estimates (forward only) ---
    log("KALMAN", "Running Kalman filter (forward only)...")
    filtered_means, filtered_covs = kf.filter(obs_masked)
    log("KALMAN", "  Filtering complete")

    # --- Extract 95% credible intervals ---
    smoothed_stds = np.zeros_like(smoothed_means)
    filtered_stds = np.zeros_like(filtered_means)
    for t in range(n_days):
        smoothed_stds[t] = np.sqrt(np.diag(smoothed_covs[t]))
        filtered_stds[t] = np.sqrt(np.diag(filtered_covs[t]))

    # --- Log-likelihood ---
    ll = kf.loglikelihood(obs_masked)
    log("KALMAN", f"  Log-likelihood: {ll:.2f}")

    # --- Store learned parameters ---
    learned_Q = kf.transition_covariance
    learned_R = kf.observation_covariance

    log("KALMAN", f"  Learned Q diagonal: {np.diag(learned_Q).round(4)}")
    log("KALMAN", f"  Learned R diagonal: {np.diag(learned_R).round(4)}")

    # --- Metrics ---
    metrics["kalman"] = {
        "em_iterations": EM_ITERATIONS,
        "em_time_s": round(em_time, 2),
        "log_likelihood": round(float(ll), 2),
        "learned_Q_diag": np.diag(learned_Q).round(6).tolist(),
        "learned_R_diag": np.diag(learned_R).round(6).tolist(),
        "state_means_final": {
            STATE_NAMES[i]: round(float(smoothed_means[-1, i]), 4)
            for i in range(N_STATES)
        },
        "state_stds_final": {
            STATE_NAMES[i]: round(float(smoothed_stds[-1, i]), 4)
            for i in range(N_STATES)
        },
    }

    return {
        "kf": kf,
        "smoothed_means": smoothed_means,
        "smoothed_covs": smoothed_covs,
        "smoothed_stds": smoothed_stds,
        "filtered_means": filtered_means,
        "filtered_covs": filtered_covs,
        "filtered_stds": filtered_stds,
        "log_likelihood": ll,
        "transition_matrix": A,
        "observation_matrix": H,
        "learned_Q": learned_Q,
        "learned_R": learned_R,
    }


# ===========================================================================
# Section 4: Unscented Kalman Filter (Nonlinear Dynamics)
# ===========================================================================
def _ensure_positive_definite(P: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Force covariance matrix to be symmetric positive semi-definite.

    1. Symmetrize: P = (P + P.T) / 2
    2. Eigendecompose and clip negative eigenvalues to eps.
    """
    P = (P + P.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(P)
    if np.any(eigvals < 0):
        eigvals = np.clip(eigvals, eps, None)
        P = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return P


def _has_nan_inf(arr: np.ndarray) -> bool:
    """Check if array contains NaN or Inf."""
    return bool(np.any(~np.isfinite(arr)))


# Physiologically plausible state bounds (z-scored units)
_STATE_BOUNDS = [
    (-5.0, 5.0),           # Autonomic Tone
    (-5.0, 5.0),           # Cardiac Reserve
    (0.0, 2.0 * np.pi),    # Circadian Phase (wrap)
    (-5.0, 5.0),           # Inflammation Level
    (-5.0, 5.0),           # Sleep Quality
]


def _clip_state(x: np.ndarray) -> np.ndarray:
    """Clip UKF state vector to physiologically plausible ranges.

    Circadian phase is wrapped to [0, 2*pi]; others are clipped to [-5, 5].
    """
    x_clipped = x.copy()
    for i, (lo, hi) in enumerate(_STATE_BOUNDS):
        if i == 2:  # circadian phase: wrap
            x_clipped[i] = x_clipped[i] % (2.0 * np.pi)
        else:
            x_clipped[i] = np.clip(x_clipped[i], lo, hi)
    return x_clipped


# Minimum covariance diagonal to detect sigma-point collapse
_COV_COLLAPSE_THRESHOLD = 1e-8
# Re-inflation value when collapse detected
_COV_REINFLATE = 0.01


def run_ukf(
    obs_masked: np.ma.MaskedArray, daily: pd.DataFrame
) -> dict[str, Any]:
    """Unscented Kalman Filter with nonlinear circadian dynamics.

    Uses MerweScaledSigmaPoints to propagate uncertainty through the
    nonlinear state transition without requiring Jacobian computation.

    Nonlinear state transition:
        - circadian_phase: phase(t+1) = sin(omega + arcsin(phase)) (wraps)
        - autonomic_tone: exponential decay toward baseline
        - Others: linear as in standard KF

    Production hardening:
        - Covariance positive-definiteness enforcement after every step
        - NaN/Inf propagation guard with rollback to last known good state
        - Missing observation handling (all-NaN = predict only, partial = reduced update)
        - Sigma-point collapse detection with covariance re-inflation
        - State bounds clipping to physiologically plausible ranges

    Returns dict with UKF filtered states.
    """
    try:
        from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    except ImportError:
        log("UKF", "WARNING: filterpy not installed. Skipping UKF analysis.")
        log("UKF", "  Install with: pip install filterpy")
        n_days = obs_masked.shape[0]
        # Return zero arrays so downstream code still works
        ukf_states = np.zeros((n_days, N_STATES))
        ukf_covs = np.zeros((n_days, N_STATES, N_STATES))
        for t in range(n_days):
            ukf_covs[t] = np.eye(N_STATES)
        ukf_stds = np.ones((n_days, N_STATES))
        metrics["ukf"] = {
            "error": "filterpy not available",
            "time_s": 0.0,
            "state_means_final": {name: 0.0 for name in STATE_NAMES},
            "state_stds_final": {name: 1.0 for name in STATE_NAMES},
        }
        return {"ukf_states": ukf_states, "ukf_covs": ukf_covs, "ukf_stds": ukf_stds}

    log("UKF", "Initializing Unscented Kalman Filter...")
    n_days = obs_masked.shape[0]

    # Circadian period (in days)
    CIRCADIAN_PERIOD = 1.0  # one full cycle per day
    CIRCADIAN_OMEGA = 2 * np.pi / CIRCADIAN_PERIOD

    # Autonomic decay rate
    AUTONOMIC_TAU = 10.0  # days to decay to ~37% of perturbation
    AUTONOMIC_DECAY = np.exp(-1.0 / AUTONOMIC_TAU)

    # Observation matrix (same as linear KF)
    H_matrix = np.array([
        [0.70, 0.10, 0.00, -0.10, 0.30],
        [-0.20, -0.50, 0.00, 0.40, -0.10],
        [0.10, 0.40, 0.00, -0.50, 0.10],
        [-0.10, 0.00, 0.30, 0.60, 0.00],
        [0.30, 0.10, 0.10, -0.10, 0.60],
    ])

    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        """Nonlinear state transition function for UKF.

        Args:
            x: state vector (5,)
            dt: time step (unused, fixed at 1 day)

        Returns:
            Propagated state vector (5,)
        """
        x_new = np.zeros_like(x)
        # Autonomic tone: exponential decay + coupling
        x_new[0] = AUTONOMIC_DECAY * x[0] - 0.05 * x[3] + 0.05 * x[4]
        # Cardiac reserve: slow drift
        x_new[1] = 0.98 * x[1] + 0.03 * x[0] - 0.02 * x[3]
        # Circadian phase: nonlinear sinusoidal wrap
        phase = x[2]
        x_new[2] = np.sin(CIRCADIAN_OMEGA + np.arcsin(np.clip(phase, -0.99, 0.99)))
        # Inflammation: slow decay
        x_new[3] = 0.92 * x[3] - 0.02 * x[0] - 0.03 * x[4]
        # Sleep quality: moderate persistence
        x_new[4] = 0.90 * x[4] + 0.05 * x[0] + 0.02 * x[1] + 0.03 * x[2] - 0.04 * x[3]
        return x_new

    def hx(x: np.ndarray) -> np.ndarray:
        """Observation function (linear): maps state to observation space."""
        return H_matrix @ x

    # --- Sigma point generation ---
    # MerweScaledSigmaPoints: alpha controls spread, beta=2 optimal for Gaussian,
    # kappa=3-n is a common choice
    sigma_points = MerweScaledSigmaPoints(
        n=N_STATES, alpha=0.1, beta=2.0, kappa=3.0 - N_STATES,
    )

    # --- UKF setup ---
    ukf = UnscentedKalmanFilter(
        dim_x=N_STATES, dim_z=N_OBS, dt=1.0,
        fx=fx, hx=hx, points=sigma_points,
    )

    # Initial state
    ukf.x = np.zeros(N_STATES)
    ukf.P = np.eye(N_STATES) * 1.0
    ukf.R = np.eye(N_OBS) * 0.3  # observation noise
    ukf.Q = np.eye(N_STATES) * 0.05  # process noise

    # --- Run UKF sequentially ---
    log("UKF", "Running UKF forward pass...")
    t0 = time.time()

    ukf_states = np.zeros((n_days, N_STATES))
    ukf_covs = np.zeros((n_days, N_STATES, N_STATES))

    # Last known good state for NaN/Inf rollback
    last_good_x = ukf.x.copy()
    last_good_P = ukf.P.copy()
    n_resets = 0
    n_cov_fixes = 0
    n_reinflations = 0

    for t in range(n_days):
        # --- Predict step ---
        ukf.predict()

        # --- Covariance positive-definiteness check after predict ---
        P_fixed = _ensure_positive_definite(ukf.P)
        if not np.allclose(P_fixed, ukf.P, atol=1e-12):
            n_cov_fixes += 1
        ukf.P = P_fixed

        # --- NaN/Inf check after predict ---
        if _has_nan_inf(ukf.x) or _has_nan_inf(ukf.P):
            log("UKF", f"  WARNING: NaN/Inf at t={t} after predict. Resetting to last good state.")
            ukf.x = last_good_x.copy()
            ukf.P = last_good_P.copy()
            n_resets += 1
            ukf_states[t] = ukf.x.copy()
            ukf_covs[t] = ukf.P.copy()
            continue

        # --- Sigma-point collapse detection ---
        max_diag = np.max(np.diag(ukf.P))
        if max_diag < _COV_COLLAPSE_THRESHOLD:
            log("UKF", f"  WARNING: Sigma-point collapse at t={t} (max diag P = {max_diag:.2e}). Re-inflating.")
            ukf.P += np.eye(N_STATES) * _COV_REINFLATE
            n_reinflations += 1

        # --- Update step (if observation available) ---
        z = obs_masked[t]

        # Determine which observations are available
        z_arr = np.ma.filled(z, fill_value=np.nan)
        valid_mask = ~np.isnan(z_arr)

        if valid_mask.any():
            if valid_mask.all():
                # Full observation available - use standard UKF update
                ukf.update(z_arr)
            else:
                # Partial observation: manual reduced-dimension update
                z_obs = z_arr[valid_mask]
                H_obs = H_matrix[valid_mask, :]
                R_obs = ukf.R[np.ix_(valid_mask, valid_mask)]

                # Innovation with reduced observation
                z_pred = H_obs @ ukf.x
                y = z_obs - z_pred
                S = H_obs @ ukf.P @ H_obs.T + R_obs
                try:
                    K = ukf.P @ H_obs.T @ np.linalg.inv(S)
                    ukf.x = ukf.x + K @ y
                    ukf.P = (np.eye(N_STATES) - K @ H_obs) @ ukf.P
                except np.linalg.LinAlgError:
                    pass  # skip update if singular
        # else: all observations missing - predict only (no update)

        # --- Covariance positive-definiteness check after update ---
        P_fixed = _ensure_positive_definite(ukf.P)
        if not np.allclose(P_fixed, ukf.P, atol=1e-12):
            n_cov_fixes += 1
        ukf.P = P_fixed

        # --- NaN/Inf check after update ---
        if _has_nan_inf(ukf.x) or _has_nan_inf(ukf.P):
            log("UKF", f"  WARNING: NaN/Inf at t={t} after update. Resetting to last good state.")
            ukf.x = last_good_x.copy()
            ukf.P = last_good_P.copy()
            n_resets += 1
        else:
            # --- Clip state to physiological bounds ---
            ukf.x = _clip_state(ukf.x)
            # Save as last known good state
            last_good_x = ukf.x.copy()
            last_good_P = ukf.P.copy()

        ukf_states[t] = ukf.x.copy()
        ukf_covs[t] = ukf.P.copy()

    ukf_time = time.time() - t0
    log("UKF", f"  UKF complete in {ukf_time:.2f}s")
    if n_resets > 0:
        log("UKF", f"  WARNING: {n_resets} NaN/Inf resets during UKF run")
    if n_cov_fixes > 0:
        log("UKF", f"  Covariance PD fixes applied: {n_cov_fixes}")
    if n_reinflations > 0:
        log("UKF", f"  Sigma-point collapse re-inflations: {n_reinflations}")

    ukf_stds = np.zeros_like(ukf_states)
    for t in range(n_days):
        ukf_stds[t] = np.sqrt(np.diag(ukf_covs[t]))

    # --- Store metrics ---
    metrics["ukf"] = {
        "time_s": round(ukf_time, 2),
        "sigma_points": {
            "alpha": 0.1,
            "beta": 2.0,
            "kappa": 3.0 - N_STATES,
            "n_sigma": 2 * N_STATES + 1,
        },
        "n_nan_inf_resets": n_resets,
        "n_cov_pd_fixes": n_cov_fixes,
        "n_sigma_reinflations": n_reinflations,
        "state_means_final": {
            STATE_NAMES[i]: round(float(ukf_states[-1, i]), 4)
            for i in range(N_STATES)
        },
        "state_stds_final": {
            STATE_NAMES[i]: round(float(ukf_stds[-1, i]), 4)
            for i in range(N_STATES)
        },
    }

    return {
        "ukf_states": ukf_states,
        "ukf_covs": ukf_covs,
        "ukf_stds": ukf_stds,
    }


# ===========================================================================
# Section 5: Real-Time State Estimation & Prediction
# ===========================================================================
def run_prediction_analysis(
    kf_results: dict, obs_masked: np.ma.MaskedArray, daily: pd.DataFrame,
    scalers: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """One-step-ahead prediction and innovation monitoring.

    At each time step:
        1. Predict next observation from filtered state
        2. Compare with actual observation
        3. Track innovation (prediction error) and its variance
        4. Flag alerts when innovation > 2*sqrt(innovation_variance)

    Also generates 7-day forecast from the last filtered state.
    """
    log("PREDICT", "Running one-step-ahead prediction analysis...")

    kf = kf_results["kf"]
    A = kf_results["transition_matrix"]
    H = kf_results["observation_matrix"]
    Q = kf_results["learned_Q"]
    R = kf_results["learned_R"]
    filtered_means = kf_results["filtered_means"]
    filtered_covs = kf_results["filtered_covs"]

    n_days = obs_masked.shape[0]
    obs_cols = ["mean_rmssd", "mean_hr", "spo2_average",
                "temperature_deviation", "sleep_efficiency"]

    # --- One-step-ahead predictions ---
    predicted_obs = np.full((n_days, N_OBS), np.nan)
    innovation = np.full((n_days, N_OBS), np.nan)
    innovation_var = np.full((n_days, N_OBS), np.nan)
    alerts = []

    for t in range(1, n_days):
        # Predict state at time t from filtered state at t-1
        x_pred = A @ filtered_means[t - 1]
        P_pred = A @ filtered_covs[t - 1] @ A.T + Q

        # Predicted observation
        z_pred = H @ x_pred
        predicted_obs[t] = z_pred

        # Innovation
        z_actual = obs_masked[t]
        if not np.ma.is_masked(z_actual) or (hasattr(z_actual, 'mask') and not np.all(z_actual.mask)):
            z_arr = np.ma.filled(z_actual, fill_value=np.nan)
            valid = ~np.isnan(z_arr)
            if valid.any():
                innov = z_arr - z_pred
                S = H @ P_pred @ H.T + R  # innovation covariance

                innovation[t, valid] = innov[valid]
                innovation_var[t] = np.diag(S)

                # Check for alerts
                for j in np.where(valid)[0]:
                    std_innov = np.sqrt(S[j, j])
                    if std_innov > 0 and abs(innov[j]) > ALERT_SIGMA * std_innov:
                        alerts.append({
                            "day": str(daily.index[t].date()),
                            "variable": OBS_NAMES[j],
                            "innovation": round(float(innov[j]), 3),
                            "threshold": round(float(ALERT_SIGMA * std_innov), 3),
                            "sigma_ratio": round(float(abs(innov[j]) / std_innov), 2),
                        })

    log("PREDICT", f"  Found {len(alerts)} alert events (>{ALERT_SIGMA} sigma)")

    # --- Prediction performance metrics ---
    pred_metrics: dict[str, dict] = {}
    for j, col in enumerate(obs_cols):
        actual = obs_masked[:, j]
        pred = predicted_obs[:, j]
        # Find where both are valid
        if hasattr(actual, 'mask'):
            valid = ~actual.mask & ~np.isnan(pred)
        else:
            valid = ~np.isnan(np.array(actual)) & ~np.isnan(pred)

        if valid.sum() > 2:
            a = np.array(actual[valid], dtype=float)
            p = pred[valid]
            rmse = float(np.sqrt(np.mean((a - p) ** 2)))
            mae = float(np.mean(np.abs(a - p)))
            ss_res = np.sum((a - p) ** 2)
            ss_tot = np.sum((a - np.mean(a)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            # De-standardize for interpretable RMSE
            mu, sigma = scalers[col]
            rmse_orig = rmse * sigma
            mae_orig = mae * sigma

            pred_metrics[col] = {
                "rmse_std": round(rmse, 4),
                "mae_std": round(mae, 4),
                "r2": round(r2, 4),
                "rmse_original_units": round(rmse_orig, 4),
                "mae_original_units": round(mae_orig, 4),
                "n_valid": int(valid.sum()),
            }
        else:
            pred_metrics[col] = {"rmse_std": None, "mae_std": None, "r2": None, "n_valid": 0}

    # --- 7-day forecast from last filtered state ---
    log("PREDICT", f"Generating {PREDICTION_HORIZON}-day forecast...")
    last_state = filtered_means[-1]
    last_cov = filtered_covs[-1]

    forecast_means = np.zeros((PREDICTION_HORIZON, N_OBS))
    forecast_stds = np.zeros((PREDICTION_HORIZON, N_OBS))
    forecast_state_means = np.zeros((PREDICTION_HORIZON, N_STATES))
    forecast_state_stds = np.zeros((PREDICTION_HORIZON, N_STATES))

    x_fc = last_state.copy()
    P_fc = last_cov.copy()

    for h in range(PREDICTION_HORIZON):
        # Propagate state
        x_fc = A @ x_fc
        P_fc = A @ P_fc @ A.T + Q

        forecast_state_means[h] = x_fc
        forecast_state_stds[h] = np.sqrt(np.diag(P_fc))

        # Predicted observation
        z_fc = H @ x_fc
        S_fc = H @ P_fc @ H.T + R
        forecast_means[h] = z_fc
        forecast_stds[h] = np.sqrt(np.diag(S_fc))

    # De-standardize forecast
    forecast_orig = np.zeros_like(forecast_means)
    forecast_orig_std = np.zeros_like(forecast_stds)
    for j, col in enumerate(obs_cols):
        mu, sigma = scalers[col]
        forecast_orig[:, j] = forecast_means[:, j] * sigma + mu
        forecast_orig_std[:, j] = forecast_stds[:, j] * sigma

    last_date = daily.index[-1]
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(PREDICTION_HORIZON)]

    log("PREDICT", "  Forecast summary (original units):")
    for j, col in enumerate(obs_cols):
        log("PREDICT", f"    {col}: {forecast_orig[0, j]:.2f} +/- {forecast_orig_std[0, j]:.2f} (day+1) "
            f"-> {forecast_orig[-1, j]:.2f} +/- {forecast_orig_std[-1, j]:.2f} (day+{PREDICTION_HORIZON})")

    metrics["prediction"] = {
        "per_variable": pred_metrics,
        "n_alerts": len(alerts),
        "alerts_last_7d": [a for a in alerts if a["day"] >= str((daily.index[-1] - timedelta(days=7)).date())],
        "forecast": {
            "horizon_days": PREDICTION_HORIZON,
            "start_date": str(forecast_dates[0].date()),
            "end_date": str(forecast_dates[-1].date()),
            "day1_predictions": {
                obs_cols[j]: {
                    "mean": round(float(forecast_orig[0, j]), 2),
                    "std": round(float(forecast_orig_std[0, j]), 2),
                }
                for j in range(N_OBS)
            },
        },
    }

    return {
        "predicted_obs": predicted_obs,
        "innovation": innovation,
        "innovation_var": innovation_var,
        "alerts": alerts,
        "pred_metrics": pred_metrics,
        "forecast_means": forecast_orig,
        "forecast_stds": forecast_orig_std,
        "forecast_dates": forecast_dates,
        "forecast_state_means": forecast_state_means,
        "forecast_state_stds": forecast_state_stds,
    }


# ===========================================================================
# Section 6: Ruxolitinib Drug Response Analysis
# ===========================================================================
def analyze_drug_response(
    kf_results: dict,
    ukf_results: dict,
    daily: pd.DataFrame,
    scalers: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Quantify ruxolitinib's effect on latent states.

    Method 1: Compare filtered state estimates before vs after drug start.
    Method 2: Fit exponential response curve to post-drug state trajectory.

    Reports:
        - Shift in each latent variable (in SD units)
        - Time constant of autonomic recovery (tau)
        - Steady-state effect estimate
    """
    log("DRUG_RESPONSE", "Analyzing ruxolitinib response in state space...")

    drug_date = pd.Timestamp(TREATMENT_START)
    dates = daily.index

    # Find index of drug start
    drug_idx = None
    for i, d in enumerate(dates):
        if d >= drug_date:
            drug_idx = i
            break

    if drug_idx is None or drug_idx < 7:
        log("DRUG_RESPONSE", "  Insufficient data around drug start")
        return {"drug_idx": None}

    log("DRUG_RESPONSE", f"  Drug start: day {drug_idx} ({TREATMENT_START})")

    smoothed = kf_results["smoothed_means"]
    smoothed_stds = kf_results["smoothed_stds"]
    ukf_states = ukf_results["ukf_states"]

    n_post = len(dates) - drug_idx
    n_pre = drug_idx

    log("DRUG_RESPONSE", f"  Pre-drug: {n_pre} days, Post-drug: {n_post} days")

    # --- Method 1: Compare pre/post distributions ---
    response_stats: dict[str, dict] = {}
    for i, name in enumerate(STATE_NAMES):
        pre_vals = smoothed[:drug_idx, i]
        post_vals = smoothed[drug_idx:, i]

        pre_mean = float(np.mean(pre_vals))
        post_mean = float(np.mean(post_vals))
        pre_std = float(np.std(pre_vals))
        shift_sd = (post_mean - pre_mean) / pre_std if pre_std > 0 else 0.0

        # Mann-Whitney U test (non-parametric, appropriate for small post-drug n)
        if len(post_vals) >= 2:
            try:
                stat, pval = scipy_stats.mannwhitneyu(
                    pre_vals, post_vals, alternative="two-sided"
                )
            except ValueError:
                stat, pval = 0.0, 1.0
        else:
            stat, pval = 0.0, 1.0

        response_stats[name] = {
            "pre_mean": round(pre_mean, 4),
            "post_mean": round(post_mean, 4),
            "shift_sd": round(shift_sd, 3),
            "mann_whitney_p_value": round(float(pval), 4),
            "direction": "improved" if (
                (name in ["Autonomic Tone", "Cardiac Reserve", "Sleep Quality"] and shift_sd > 0.1)
                or (name == "Inflammation Level" and shift_sd < -0.1)
            ) else "worsened" if (
                (name in ["Autonomic Tone", "Cardiac Reserve", "Sleep Quality"] and shift_sd < -0.1)
                or (name == "Inflammation Level" and shift_sd > 0.1)
            ) else "stable",
        }

    # --- Method 2: Exponential response fit ---
    # Fit: state(t) = a * (1 - exp(-(t-t0)/tau)) + baseline
    # for autonomic tone (primary drug target)
    tau_estimates: dict[str, Optional[float]] = {}
    steady_state: dict[str, Optional[float]] = {}

    for i, name in enumerate(STATE_NAMES):
        post_vals = smoothed[drug_idx:, i]
        if len(post_vals) < 3:
            tau_estimates[name] = None
            steady_state[name] = None
            continue

        t_post = np.arange(len(post_vals), dtype=float)
        baseline = float(smoothed[drug_idx - 1, i])

        def exp_response(t, amplitude, tau):
            return amplitude * (1 - np.exp(-t / max(tau, 0.1))) + baseline

        try:
            popt, pcov = curve_fit(
                exp_response,
                t_post,
                post_vals,
                p0=[post_vals[-1] - baseline, 3.0],
                maxfev=5000,
                bounds=([-5, 0.1], [5, 30]),
            )
            tau_estimates[name] = round(float(popt[1]), 2)
            steady_state[name] = round(float(popt[0] + baseline), 4)
            log("DRUG_RESPONSE", f"  {name}: tau={popt[1]:.1f}d, "
                f"amplitude={popt[0]:.3f}, steady-state={popt[0]+baseline:.3f}")
        except (RuntimeError, ValueError):
            tau_estimates[name] = None
            steady_state[name] = None
            log("DRUG_RESPONSE", f"  {name}: exponential fit failed")

    # --- UKF comparison ---
    ukf_response: dict[str, dict] = {}
    for i, name in enumerate(STATE_NAMES):
        ukf_pre = ukf_states[:drug_idx, i]
        ukf_post = ukf_states[drug_idx:, i]
        ukf_shift = (np.mean(ukf_post) - np.mean(ukf_pre)) / (np.std(ukf_pre) if np.std(ukf_pre) > 0 else 1.0)
        ukf_response[name] = {
            "ukf_shift_sd": round(float(ukf_shift), 3),
            "kf_shift_sd": response_stats[name]["shift_sd"],
            "agreement": abs(ukf_shift - response_stats[name]["shift_sd"]) < 0.5,
        }

    metrics["drug_response"] = {
        "drug_start_date": TREATMENT_START,
        "drug_start_idx": drug_idx,
        "n_pre_days": n_pre,
        "n_post_days": n_post,
        "response_stats": response_stats,
        "tau_estimates": tau_estimates,
        "steady_state_estimates": steady_state,
        "ukf_comparison": ukf_response,
    }

    return {
        "drug_idx": drug_idx,
        "response_stats": response_stats,
        "tau_estimates": tau_estimates,
        "steady_state": steady_state,
        "ukf_response": ukf_response,
    }


# ===========================================================================
# Section 7: Multi-Modal Sensor Fusion Quality
# ===========================================================================
def analyze_sensor_fusion(
    kf_results: dict, obs_masked: np.ma.MaskedArray, daily: pd.DataFrame
) -> dict[str, Any]:
    """Assess multi-modal data fusion quality.

    Tracks:
        - Per-sensor observation likelihood
        - Sensor contribution to state estimation (Kalman gain norms)
        - Missing data handling quality
        - Information content per sensor
    """
    log("FUSION", "Analyzing multi-modal sensor fusion quality...")

    H = kf_results["observation_matrix"]
    R = kf_results["learned_R"]
    Q = kf_results["learned_Q"]
    A = kf_results["transition_matrix"]
    filtered_means = kf_results["filtered_means"]
    filtered_covs = kf_results["filtered_covs"]

    n_days = obs_masked.shape[0]
    obs_cols = ["mean_rmssd", "mean_hr", "spo2_average",
                "temperature_deviation", "sleep_efficiency"]

    # --- Observation availability ---
    availability = {}
    for j, col in enumerate(obs_cols):
        if hasattr(obs_masked[:, j], 'mask') and obs_masked[:, j].mask is not np.bool_(False):
            n_avail = int((~obs_masked[:, j].mask).sum())
        else:
            n_avail = int((~np.isnan(np.array(obs_masked[:, j]))).sum())
        availability[col] = {
            "n_available": n_avail,
            "fraction": round(n_avail / n_days, 3),
        }

    # --- Sensor information content (via R^-1 and H) ---
    # Information from sensor j = H_j^T R_jj^-1 H_j (Fisher information contribution)
    sensor_info: dict[str, float] = {}
    for j, col in enumerate(obs_cols):
        h_j = H[j, :]  # 1 x n_states
        r_jj = R[j, j]
        if r_jj > 0:
            info_norm = float(np.dot(h_j, h_j) / r_jj)
        else:
            info_norm = 0.0
        sensor_info[col] = round(info_norm, 4)

    total_info = sum(sensor_info.values())
    sensor_weight: dict[str, float] = {}
    for col in obs_cols:
        sensor_weight[col] = round(sensor_info[col] / total_info * 100, 1) if total_info > 0 else 0.0

    log("FUSION", "  Sensor contribution to state estimation:")
    for col in obs_cols:
        log("FUSION", f"    {col}: {sensor_weight[col]:.1f}% "
            f"(avail: {availability[col]['fraction']*100:.0f}%)")

    # --- Average Kalman gain norms per sensor ---
    avg_gain = np.zeros(N_OBS)
    n_updates = 0
    for t in range(1, n_days):
        P_pred = A @ filtered_covs[t - 1] @ A.T + Q
        S = H @ P_pred @ H.T + R
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
            for j in range(N_OBS):
                avg_gain[j] += np.linalg.norm(K[:, j])
            n_updates += 1
        except np.linalg.LinAlgError:
            continue

    if n_updates > 0:
        avg_gain /= n_updates

    gain_contribution: dict[str, float] = {}
    total_gain = avg_gain.sum()
    for j, col in enumerate(obs_cols):
        gain_contribution[col] = round(float(avg_gain[j] / total_gain * 100) if total_gain > 0 else 0.0, 1)

    # --- Residual analysis (should be white noise if model correct) ---
    residuals: dict[str, dict] = {}
    for j, col in enumerate(obs_cols):
        predicted = kf_results["kf"].observation_matrices @ kf_results["filtered_means"].T
        pred_j = predicted[j, :]

        actual_j = obs_masked[:, j]
        if hasattr(actual_j, 'mask') and actual_j.mask is not np.bool_(False):
            valid = ~actual_j.mask
        else:
            valid = ~np.isnan(np.array(actual_j))

        if valid.sum() > 10:
            resid = np.array(actual_j[valid]) - pred_j[valid]
            # Ljung-Box test for autocorrelation
            n_r = len(resid)
            lags = min(10, n_r // 5)
            if lags >= 1:
                acf_vals = [float(np.corrcoef(resid[:-k], resid[k:])[0, 1]) if k > 0 else 1.0
                            for k in range(1, lags + 1)]
                lb_stat = n_r * (n_r + 2) * sum(ac**2 / (n_r - k) for k, ac in enumerate(acf_vals, 1))
                lb_pval = float(1 - scipy_stats.chi2.cdf(lb_stat, df=lags))
            else:
                lb_pval = 1.0
                acf_vals = []

            residuals[col] = {
                "mean": round(float(np.mean(resid)), 4),
                "std": round(float(np.std(resid)), 4),
                "ljung_box_p_value": round(lb_pval, 4),
                "white_noise": lb_pval > 0.05,
                "acf_lag1": round(float(acf_vals[0]), 4) if acf_vals else None,
            }

    n_lb_pass = sum(1 for r in residuals.values() if r.get("white_noise"))
    n_lb_total = len(residuals)

    metrics["sensor_fusion"] = {
        "availability": availability,
        "information_content": sensor_info,
        "sensor_weight_pct": sensor_weight,
        "kalman_gain_contribution_pct": gain_contribution,
        "residual_diagnostics": residuals,
        "significance_threshold_p_value": 0.05,
        "ljung_box_pass_count": n_lb_pass,
        "ljung_box_total": n_lb_total,
    }

    return {
        "availability": availability,
        "sensor_info": sensor_info,
        "sensor_weight": sensor_weight,
        "gain_contribution": gain_contribution,
        "residuals": residuals,
    }


# ===========================================================================
# Section 8: Visualization
# ===========================================================================
def create_state_trajectory_figure(
    kf_results: dict,
    ukf_results: dict,
    daily: pd.DataFrame,
    drug_response: dict,
) -> go.Figure:
    """Main panel: 5 latent state trajectories with credible bands."""
    log("VIZ", "Creating state trajectory figure...")

    dates = daily.index
    smoothed = kf_results["smoothed_means"]
    smoothed_stds = kf_results["smoothed_stds"]
    ukf_states = ukf_results["ukf_states"]
    ukf_stds = ukf_results["ukf_stds"]

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=[f"{name}" for name in STATE_NAMES],
        vertical_spacing=0.06,
    )

    state_colors = [COLORS["hrv"], COLORS["hr"], COLORS["spo2"],
                    COLORS["temp"], COLORS["sleep"]]
    state_units = ["SD", "SD", "SD", "SD", "SD"]

    for i, name in enumerate(STATE_NAMES):
        row = i + 1
        color = state_colors[i]

        # 95% credible band (drawn FIRST so it's behind everything)
        upper = smoothed[:, i] + 2 * smoothed_stds[:, i]
        lower = smoothed[:, i] - 2 * smoothed_stds[:, i]
        fig.add_trace(
            go.Scatter(
                x=list(dates) + list(dates[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself", fillcolor=_hex_to_rgba(color, 0.07),
                line=dict(width=0),
                name="95% CI",
                legendgroup="ci",
                showlegend=(i == 0), hoverinfo="skip",
            ),
            row=row, col=1,
        )

        # UKF overlay (behind the KF line, subtle)
        fig.add_trace(
            go.Scatter(
                x=dates, y=ukf_states[:, i],
                name="UKF Filtered" if i == 0 else name,
                legendgroup="ukf",
                line=dict(color=color, width=1.2, dash="dot"),
                showlegend=(i == 0), opacity=0.45,
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{name} (UKF): %{{y:.3f}} {state_units[i]}<extra></extra>",
            ),
            row=row, col=1,
        )

        # KF smoothed mean (PROMINENT - drawn last for visual priority)
        fig.add_trace(
            go.Scatter(
                x=dates, y=smoothed[:, i],
                name="KF Smoothed" if i == 0 else name,
                legendgroup="kf",
                line=dict(color=color, width=2.5),
                showlegend=(i == 0),
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{name} (KF): %{{y:.3f}} {state_units[i]}<extra></extra>",
            ),
            row=row, col=1,
        )

        # Zero reference line
        fig.add_hline(y=0, row=row, col=1,
                       line=dict(color=TEXT_TERTIARY, width=0.5, dash="dash"))

    fig.update_layout(
        height=1520,
        margin=dict(l=64, r=34, t=132, b=68),
        font=dict(size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.12),
    )

    _add_timeline_vlines(fig, [(row, 1) for row in range(1, 6)])

    for i in range(5):
        ax_num = i + 1
        yax = f"yaxis{ax_num}" if ax_num > 1 else "yaxis"
        xax = f"xaxis{ax_num}" if ax_num > 1 else "xaxis"
        fig.update_layout(**{
            yax: dict(
                title="State (SD units)",
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                zeroline=False,
            ),
            xax: dict(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                spikemode="across", spikethickness=1,
                spikecolor=TEXT_TERTIARY, spikedash="dot",
            ),
        })

    return fig


def create_prediction_figure(
    pred_results: dict,
    obs_masked: np.ma.MaskedArray,
    daily: pd.DataFrame,
    scalers: dict[str, tuple[float, float]],
) -> go.Figure:
    """Prediction performance: actual vs predicted scatter + innovation time series."""
    log("VIZ", "Creating prediction performance figure...")

    dates = daily.index
    obs_cols = ["mean_rmssd", "mean_hr", "spo2_average",
                "temperature_deviation", "sleep_efficiency"]
    predicted_obs = pred_results["predicted_obs"]
    innovation = pred_results["innovation"]
    innovation_var = pred_results["innovation_var"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "HRV Prediction", "Heart Rate Prediction",
            "Innovation Sequence (HRV)", "Innovation Sequence (HR)",
            "Prediction R-sq Summary", "7-Day Forecast (HRV)",
        ],
        vertical_spacing=0.10, horizontal_spacing=0.10,
    )

    # --- Scatter: Actual vs Predicted for HRV and HR ---
    scatter_units = {"mean_rmssd": "ms", "mean_hr": "bpm"}
    for j, (col, title, row_col) in enumerate([
        ("mean_rmssd", "HRV", (1, 1)),
        ("mean_hr", "HR", (1, 2)),
    ]):
        actual = obs_masked[:, j]
        pred = predicted_obs[:, j]
        if hasattr(actual, 'mask') and actual.mask is not np.bool_(False):
            valid = ~actual.mask & ~np.isnan(pred)
        else:
            valid = ~np.isnan(np.array(actual)) & ~np.isnan(pred)

        if valid.any():
            a = np.array(actual[valid], dtype=float)
            p = pred[valid]

            # De-standardize
            mu, sigma = scalers[col]
            a_orig = a * sigma + mu
            p_orig = p * sigma + mu
            unit = scatter_units[col]

            # Subtle diagonal y=x reference line (drawn first, behind scatter)
            vmin = min(a_orig.min(), p_orig.min())
            vmax = max(a_orig.max(), p_orig.max())
            pad = (vmax - vmin) * 0.05
            fig.add_trace(
                go.Scatter(
                    x=[vmin - pad, vmax + pad], y=[vmin - pad, vmax + pad],
                    mode="lines",
                    line=dict(color=TEXT_TERTIARY, dash="dash", width=1),
                    showlegend=False, hoverinfo="skip",
                ),
                row=row_col[0], col=row_col[1],
            )

            # Scatter points with transparency for density
            fig.add_trace(
                go.Scatter(
                    x=a_orig, y=p_orig, mode="markers",
                    marker=dict(size=5, color=COLORS["actual"], opacity=0.45,
                                symbol="circle",
                                line=dict(width=0.5, color=_hex_to_rgba(COLORS["actual"], 0.7))),
                    name=f"{title} actual vs pred",
                    showlegend=False,
                    hovertemplate=f"<b>Actual:</b> %{{x:.1f}} {unit}<br><b>Predicted:</b> %{{y:.1f}} {unit}<extra></extra>",
                ),
                row=row_col[0], col=row_col[1],
            )

    # --- Innovation time series ---
    for j, (col, title, row_col) in enumerate([
        ("mean_rmssd", "HRV", (2, 1)),
        ("mean_hr", "HR", (2, 2)),
    ]):
        innov = innovation[:, j]
        innov_v = innovation_var[:, j]
        valid = ~np.isnan(innov)

        if valid.any():
            fig.add_trace(
                go.Scatter(
                    x=dates[valid], y=innov[valid],
                    mode="lines+markers",
                    marker=dict(size=2.5), line=dict(width=1.5, color=COLORS["innovation"]),
                    name=f"{title} innovation", showlegend=False,
                    hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{title} Innovation: %{{y:.3f}}<extra></extra>",
                ),
                row=row_col[0], col=row_col[1],
            )

            # Zero reference
            fig.add_hline(y=0, row=row_col[0], col=row_col[1],
                          line=dict(color=TEXT_TERTIARY, width=0.5, dash="dash"))

            # Alert thresholds
            valid_v = ~np.isnan(innov_v)
            combined = valid & valid_v
            if combined.any():
                thresh = ALERT_SIGMA * np.sqrt(innov_v[combined])
                fig.add_trace(
                    go.Scatter(
                        x=dates[combined], y=thresh,
                        mode="lines",
                        line=dict(color=COLORS["alert"], dash="dot", width=1),
                        name=f"+{ALERT_SIGMA:.0f}\u03c3", showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_col[0], col=row_col[1],
                )
                fig.add_trace(
                    go.Scatter(
                        x=dates[combined], y=-thresh,
                        mode="lines",
                        line=dict(color=COLORS["alert"], dash="dot", width=1),
                        name=f"-{ALERT_SIGMA:.0f}\u03c3", showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_col[0], col=row_col[1],
                )

    # --- R-squared summary bar chart ---
    pred_metrics = pred_results["pred_metrics"]
    r2_vals = []
    r2_labels = []
    r2_colors = []
    for col in obs_cols:
        r2 = pred_metrics[col].get("r2")
        if r2 is not None:
            r2_vals.append(r2)
            r2_labels.append(col.replace("mean_", "").replace("_", " ").title())
            r2_colors.append(ACCENT_GREEN if r2 >= 0 else ACCENT_RED)

    fig.add_trace(
        go.Bar(
            x=r2_labels, y=r2_vals,
            marker_color=r2_colors, showlegend=False,
            text=[f"{v:.3f}" for v in r2_vals], textposition="outside",
            hovertemplate="<b>%{x}</b><br>R\u00b2: %{y:.4f}<extra></extra>",
        ),
        row=3, col=1,
    )

    # --- 7-day forecast for HRV ---
    forecast = pred_results["forecast_means"]
    forecast_std = pred_results["forecast_stds"]
    forecast_dates = pred_results["forecast_dates"]

    # Historical HRV (last 14 days, de-standardized)
    mu_hrv, sigma_hrv = scalers["mean_rmssd"]
    n_hist = 14
    hist_dates = dates[-n_hist:]
    hist_hrv = obs_masked[-n_hist:, 0]
    if hasattr(hist_hrv, 'mask') and hist_hrv.mask is not np.bool_(False):
        hist_valid = ~hist_hrv.mask
    else:
        hist_valid = ~np.isnan(np.array(hist_hrv))

    if hist_valid.any():
        h = np.array(hist_hrv[hist_valid], dtype=float) * sigma_hrv + mu_hrv
        fig.add_trace(
            go.Scatter(
                x=hist_dates[hist_valid], y=h,
                mode="lines+markers",
                marker=dict(size=4, symbol="circle"),
                line=dict(color=COLORS["actual"], width=2),
                name="Historical HRV", showlegend=False,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Observed RMSSD: %{y:.1f} ms<extra></extra>",
            ),
            row=3, col=2,
        )

    # Forecast
    fc_hrv = forecast[:, 0]
    fc_std = forecast_std[:, 0]
    fc_dates_ts = [pd.Timestamp(d) for d in forecast_dates]

    # Forecast uncertainty band (drawn first, behind)
    upper = fc_hrv + 2 * fc_std
    lower = fc_hrv - 2 * fc_std
    fig.add_trace(
        go.Scatter(
            x=list(fc_dates_ts) + list(fc_dates_ts[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself", fillcolor=_hex_to_rgba(COLORS["predicted"], 0.12),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ),
        row=3, col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=fc_dates_ts, y=fc_hrv,
            mode="lines+markers",
            marker=dict(size=6, symbol="diamond", line=dict(width=1, color="white")),
            line=dict(color=COLORS["predicted"], width=2.5),
            name="Forecast HRV", showlegend=False,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast RMSSD: %{y:.1f} ms<extra></extra>",
        ),
        row=3, col=2,
    )

    fig.update_layout(
        height=1040,
        margin=dict(l=64, r=34, t=128, b=70),
        font=dict(size=12),
    )

    _add_timeline_vlines(fig, [(2, 1), (2, 2), (3, 2)])

    # Axis labels
    fig.update_xaxes(title_text="Actual (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Actual (bpm)", row=1, col=2)
    fig.update_yaxes(title_text="Predicted (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted (bpm)", row=1, col=2)
    fig.update_yaxes(title_text="Innovation (SD)", row=2, col=1)
    fig.update_yaxes(title_text="Innovation (SD)", row=2, col=2)
    fig.update_yaxes(title_text="R\u00b2", row=3, col=1, zeroline=False)
    fig.update_yaxes(title_text="RMSSD (ms)", row=3, col=2)

    # Refined gridlines on all subplots
    for row_i in range(1, 4):
        for col_i in range(1, 3):
            fig.update_xaxes(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                row=row_i, col=col_i,
            )
            fig.update_yaxes(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                row=row_i, col=col_i,
            )
    # Crosshair spikes on time-series panels (row 2 and 3,col2)
    for col_i in range(1, 3):
        fig.update_xaxes(
            spikemode="across", spikethickness=1,
            spikecolor=TEXT_TERTIARY, spikedash="dot",
            row=2, col=col_i,
        )
    fig.update_xaxes(
        spikemode="across", spikethickness=1,
        spikecolor=TEXT_TERTIARY, spikedash="dot",
        row=3, col=2,
    )

    return fig


def create_drug_response_figure(
    kf_results: dict,
    drug_response: dict,
    daily: pd.DataFrame,
) -> go.Figure:
    """Drug response visualization: state shifts + exponential fits."""
    log("VIZ", "Creating drug response figure...")

    dates = daily.index
    smoothed = kf_results["smoothed_means"]
    drug_idx = drug_response.get("drug_idx")

    if drug_idx is None:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient post-drug data", x=0.5, y=0.5,
                            xref="paper", yref="paper", showarrow=False)
        return fig

    state_colors = [COLORS["hrv"], COLORS["hr"], COLORS["spo2"],
                    COLORS["temp"], COLORS["sleep"]]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f"{name} Response" for name in STATE_NAMES
        ] + ["Response Summary"],
        vertical_spacing=0.15, horizontal_spacing=0.08,
    )

    row_col_map = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    response_stats = drug_response.get("response_stats", {})
    tau_estimates = drug_response.get("tau_estimates", {})

    for i, name in enumerate(STATE_NAMES):
        r, c = row_col_map[i]
        color = state_colors[i]

        # Pre-drug (muted baseline)
        fig.add_trace(
            go.Scatter(
                x=dates[:drug_idx], y=smoothed[:drug_idx, i],
                mode="lines", line=dict(color=color, width=1.5, dash="solid"),
                name="Pre-drug" if i == 0 else f"Pre {name}",
                legendgroup="pre",
                showlegend=(i == 0), opacity=0.4,
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{name} (pre): %{{y:.3f}} SD<extra></extra>",
            ),
            row=r, col=c,
        )

        # Post-drug (bold, prominent)
        fig.add_trace(
            go.Scatter(
                x=dates[drug_idx:], y=smoothed[drug_idx:, i],
                mode="lines",
                line=dict(color=color, width=2.5, shape="spline"),
                name="Post-drug" if i == 0 else f"Post {name}",
                legendgroup="post",
                showlegend=(i == 0),
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{name} (post): %{{y:.3f}} SD<extra></extra>",
            ),
            row=r, col=c,
        )

        # Prominent effect size annotation
        stats = response_stats.get(name, {})
        shift = stats.get("shift_sd", 0)
        direction = stats.get("direction", "stable")
        tau = tau_estimates.get(name)
        tau_str = f"  |  \u03c4 = {tau:.1f}d" if tau else ""
        # Color the annotation by direction
        ann_color = ACCENT_GREEN if direction == "improved" else ACCENT_RED if direction == "worsened" else TEXT_SECONDARY
        fig.add_annotation(
            x=0.5, y=1.02,
            xref=f"x{i + 1} domain" if i + 1 > 1 else "x domain",
            yref=f"y{i + 1} domain" if i + 1 > 1 else "y domain",
            text=f"<b>\u0394 {shift:+.2f} SD</b>{tau_str}",
            showarrow=False, font=dict(size=11, color=ann_color),
            bgcolor=BG_ELEVATED, bordercolor=BORDER_SUBTLE, borderwidth=1,
            borderpad=4,
        )

    # --- Summary bar chart ---
    names_short = ["Auton.", "Cardiac", "Circad.", "Inflam.", "Sleep"]
    shifts = [response_stats.get(n, {}).get("shift_sd", 0) for n in STATE_NAMES]
    bar_colors = [ACCENT_GREEN if s > 0 else ACCENT_RED for s in shifts]

    fig.add_trace(
        go.Bar(
            x=names_short, y=shifts,
            marker_color=bar_colors, showlegend=False,
            text=[f"{s:+.2f}" for s in shifts], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Shift: %{y:+.3f} SD<extra></extra>",
        ),
        row=2, col=3,
    )
    fig.update_yaxes(title_text="Shift (SD)", row=2, col=3, zeroline=False)

    fig.update_layout(
        height=780,
        margin=dict(l=64, r=34, t=118, b=66),
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.12),
    )

    _add_timeline_vlines(fig, row_col_map)

    # Refined gridlines on all subplots
    for r_i in range(1, 3):
        for c_i in range(1, 4):
            fig.update_xaxes(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                row=r_i, col=c_i,
            )
            fig.update_yaxes(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                zeroline=False,
                row=r_i, col=c_i,
            )
    # Crosshair spikes on time-series panels
    for r_i, c_i in row_col_map:
        fig.update_xaxes(
            spikemode="across", spikethickness=1,
            spikecolor=TEXT_TERTIARY, spikedash="dot",
            row=r_i, col=c_i,
        )

    return fig


def create_sensor_fusion_figure(
    fusion_results: dict,
    obs_masked: np.ma.MaskedArray,
    daily: pd.DataFrame,
) -> go.Figure:
    """Sensor fusion quality visualization."""
    log("VIZ", "Creating sensor fusion figure...")

    obs_cols = ["mean_rmssd", "mean_hr", "spo2_average",
                "temperature_deviation", "sleep_efficiency"]
    obs_labels = ["HRV", "Heart Rate", "SpO2", "Temp Dev.", "Sleep Eff."]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Information Content per Sensor (%)",
            "Kalman Gain Contribution (%)",
            "Data Availability Timeline",
            "Residual Diagnostics",
        ],
        vertical_spacing=0.15, horizontal_spacing=0.10,
    )

    sensor_colors = [COLORS["hrv"], COLORS["hr"], COLORS["spo2"],
                     COLORS["temp"], COLORS["sleep"]]

    # --- Bar 1: Information content ---
    info = fusion_results["sensor_info"]
    info_total = sum(info[c] for c in obs_cols)
    info_pct = [info[c] / info_total * 100 if info_total > 0 else 0 for c in obs_cols]
    fig.add_trace(
        go.Bar(
            x=obs_labels, y=info_pct,
            marker_color=sensor_colors, showlegend=False,
            text=[f"{v:.1f}%" for v in info_pct], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Information: %{y:.1f}%<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- Bar 2: Kalman gain contribution ---
    gain = fusion_results["gain_contribution"]
    gain_vals = [gain[c] for c in obs_cols]
    fig.add_trace(
        go.Bar(
            x=obs_labels, y=gain_vals,
            marker_color=sensor_colors, showlegend=False,
            text=[f"{v:.1f}%" for v in gain_vals], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Kalman Gain: %{y:.1f}%<extra></extra>",
        ),
        row=1, col=2,
    )

    # --- Data availability heatmap-like timeline ---
    dates = daily.index
    for j, (col, label) in enumerate(zip(obs_cols, obs_labels)):
        sensor_data = obs_masked[:, j]
        if hasattr(sensor_data, 'mask') and sensor_data.mask is not np.bool_(False):
            available = (~sensor_data.mask).astype(float)
        else:
            available = (~np.isnan(np.array(sensor_data))).astype(float)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[j] * len(dates),
                mode="markers",
                marker=dict(
                    size=5,
                    color=available,
                    colorscale=[[0, TEXT_TERTIARY], [1, sensor_colors[j]]],
                    cmin=0, cmax=1,
                    symbol=["circle" if a else "x" for a in available],
                ),
                name=label, showlegend=False,
                hovertemplate=[
                    f"<b>%{{x|%Y-%m-%d}}</b><br>{label}: {'Available' if a else 'Missing'}<extra></extra>"
                    for a in available
                ],
            ),
            row=2, col=1,
        )

    fig.update_yaxes(
        tickvals=list(range(len(obs_labels))),
        ticktext=obs_labels,
        row=2, col=1,
    )

    # --- Residual diagnostics bar chart ---
    residuals = fusion_results.get("residuals", {})
    res_labels = []
    res_lb_p = []
    res_colors = []
    for col in obs_cols:
        if col in residuals:
            r = residuals[col]
            res_labels.append(col.replace("mean_", "").replace("_", " ").title())
            p = r.get("ljung_box_p", 0)
            res_lb_p.append(p)
            res_colors.append(COLORS["hrv"] if r.get("white_noise", False) else COLORS["post"])

    if res_labels:
        fig.add_trace(
            go.Bar(
                x=res_labels, y=res_lb_p,
                marker_color=res_colors, showlegend=False,
                text=[f"p={p:.3f}" for p in res_lb_p], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Ljung-Box p: %{y:.4f}<br>%{text}<extra></extra>",
            ),
            row=2, col=2,
        )
        # Significance line
        fig.add_hline(y=0.05, row=2, col=2,
                       line=dict(color=ACCENT_RED, dash="dash", width=1),
                       annotation_text="p=0.05",
                       annotation_font_size=11,
                       annotation_font_color=ACCENT_RED)
    fig.update_yaxes(title_text="Ljung-Box p-value", row=2, col=2, zeroline=False)
    fig.update_yaxes(title_text="Contribution (%)", row=1, col=1, zeroline=False)
    fig.update_yaxes(title_text="Contribution (%)", row=1, col=2, zeroline=False)

    fig.update_layout(
        height=920,
        margin=dict(l=64, r=34, t=120, b=66),
        font=dict(size=12),
    )

    # Refined gridlines on all subplots
    for r_i in range(1, 3):
        for c_i in range(1, 3):
            fig.update_xaxes(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                row=r_i, col=c_i,
            )
            fig.update_yaxes(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                row=r_i, col=c_i,
            )

    return fig


def create_observation_overlay_figure(
    kf_results: dict,
    obs_masked: np.ma.MaskedArray,
    daily: pd.DataFrame,
    scalers: dict[str, tuple[float, float]],
) -> go.Figure:
    """Raw observations overlaid with filtered estimates (de-standardized)."""
    log("VIZ", "Creating observation overlay figure...")

    dates = daily.index
    obs_cols = ["mean_rmssd", "mean_hr", "spo2_average",
                "temperature_deviation", "sleep_efficiency"]
    obs_titles = ["HRV (RMSSD, ms)", "Heart Rate (bpm)", "SpO2 (%)",
                  "Temperature Deviation (°C)", "Sleep Efficiency (%)"]
    obs_colors = [COLORS["hrv"], COLORS["hr"], COLORS["spo2"],
                  COLORS["temp"], COLORS["sleep"]]

    H = kf_results["observation_matrix"]
    smoothed = kf_results["smoothed_means"]

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=obs_titles,
        vertical_spacing=0.08,
    )

    obs_unit_labels = ["ms", "bpm", "%", "\u00b0C", "%"]

    for j, (col, title, color) in enumerate(zip(obs_cols, obs_titles, obs_colors)):
        row = j + 1
        mu, sigma = scalers[col]
        unit = obs_unit_labels[j]

        # Raw observations (de-standardized) - small muted scatter
        raw = obs_masked[:, j]
        if hasattr(raw, 'mask') and raw.mask is not np.bool_(False):
            valid = ~raw.mask
        else:
            valid = ~np.isnan(np.array(raw))

        if valid.any():
            raw_orig = np.array(raw[valid], dtype=float) * sigma + mu
            fig.add_trace(
                go.Scatter(
                    x=dates[valid], y=raw_orig,
                    mode="markers",
                    marker=dict(size=3.5, color=color, opacity=0.35,
                                symbol="circle"),
                    name="Observed" if j == 0 else f"{col} raw",
                    legendgroup="observed",
                    showlegend=(j == 0),
                    hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Observed: %{{y:.1f}} {unit}<extra></extra>",
                ),
                row=row, col=1,
            )

        # Model estimate: H[j,:] @ smoothed_state for each time step
        model_est = (H[j, :] @ smoothed.T) * sigma + mu
        fig.add_trace(
            go.Scatter(
                x=dates, y=model_est,
                mode="lines", line=dict(color=color, width=2.5),
                name="Model Estimate" if j == 0 else f"{col} model",
                legendgroup="model",
                showlegend=(j == 0),
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Model: %{{y:.1f}} {unit}<extra></extra>",
            ),
            row=row, col=1,
        )

    fig.update_layout(
        height=1520,
        margin=dict(l=64, r=34, t=132, b=68),
        font=dict(size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.12),
    )

    _add_timeline_vlines(fig, [(row, 1) for row in range(1, 6)])

    # Refined gridlines and crosshair spikes for all subplots
    for i in range(5):
        ax_num = i + 1
        yax = f"yaxis{ax_num}" if ax_num > 1 else "yaxis"
        xax = f"xaxis{ax_num}" if ax_num > 1 else "xaxis"
        fig.update_layout(**{
            yax: dict(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                zeroline=False,
            ),
            xax: dict(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                spikemode="across", spikethickness=1,
                spikecolor=TEXT_TERTIARY, spikedash="dot",
            ),
        })

    return fig


def create_kf_vs_ukf_figure(
    kf_results: dict,
    ukf_results: dict,
    daily: pd.DataFrame,
) -> go.Figure:
    """Compare linear KF vs UKF state estimates."""
    log("VIZ", "Creating KF vs UKF comparison figure...")

    dates = daily.index
    smoothed = kf_results["smoothed_means"]
    ukf_states = ukf_results["ukf_states"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[f"{name}: KF vs UKF" for name in STATE_NAMES] + ["State Difference (UKF - KF)"],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    state_colors = [COLORS["hrv"], COLORS["hr"], COLORS["spo2"],
                    COLORS["temp"], COLORS["sleep"]]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]

    for i, name in enumerate(STATE_NAMES):
        r, c = positions[i]

        # KF smoothed - solid line, prominent
        fig.add_trace(
            go.Scatter(
                x=dates, y=smoothed[:, i],
                mode="lines",
                line=dict(color=state_colors[i], width=2.5),
                name="KF Smoothed" if i == 0 else f"KF {name}",
                legendgroup="kf_comp",
                showlegend=(i == 0),
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{name} (KF): %{{y:.3f}} SD<extra></extra>",
            ),
            row=r, col=c,
        )
        # UKF filtered - dashed line, slightly thinner
        fig.add_trace(
            go.Scatter(
                x=dates, y=ukf_states[:, i],
                mode="lines",
                line=dict(color=state_colors[i], width=1.5, dash="dashdot"),
                name="UKF Filtered" if i == 0 else f"UKF {name}",
                legendgroup="ukf_comp",
                showlegend=(i == 0), opacity=0.7,
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{name} (UKF): %{{y:.3f}} SD<extra></extra>",
            ),
            row=r, col=c,
        )

    # Difference panel
    for i, name in enumerate(STATE_NAMES):
        diff = ukf_states[:, i] - smoothed[:, i]
        fig.add_trace(
            go.Scatter(
                x=dates, y=diff,
                mode="lines", line=dict(color=state_colors[i], width=1.5),
                name=name, showlegend=False,
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{name} diff: %{{y:.4f}}<extra></extra>",
            ),
            row=3, col=2,
        )
    # Zero reference on difference panel
    fig.add_hline(y=0, row=3, col=2,
                  line=dict(color=TEXT_TERTIARY, width=0.5, dash="dash"))

    fig.update_layout(
        height=1020,
        margin=dict(l=64, r=34, t=128, b=66),
        font=dict(size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.12),
    )

    _add_timeline_vlines(fig, positions + [(3, 2)])

    # Refined gridlines and crosshair spikes on all subplots
    for r_i in range(1, 4):
        for c_i in range(1, 3):
            fig.update_xaxes(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                spikemode="across", spikethickness=1,
                spikecolor=TEXT_TERTIARY, spikedash="dot",
                row=r_i, col=c_i,
            )
            fig.update_yaxes(
                gridcolor="rgba(255,255,255,0.05)", griddash="dot",
                zeroline=False,
                row=r_i, col=c_i,
            )

    return fig


# ===========================================================================
# Section 9: HTML Report Generation
# ===========================================================================
def generate_html_report(figs: list[go.Figure], daily: pd.DataFrame) -> str:
    """Generate dark-themed HTML report using the shared design system."""
    log("REPORT", "Generating dark-themed HTML report...")

    # Convert figures to HTML divs (Plotly JS loaded once by wrap_html)
    fig_divs = []
    for fig in figs:
        div = fig.to_html(full_html=False, include_plotlyjs=False)
        fig_divs.append(div)

    # Build clinical summary data
    drug_resp = metrics.get("drug_response", {})
    resp_stats = drug_resp.get("response_stats", {})
    pred = metrics.get("prediction", {})
    fusion = metrics.get("sensor_fusion", {})
    data_range = metrics.get("data_range", {})
    pred_vars = pred.get("per_variable", {})
    sensor_weights = fusion.get("sensor_weight_pct", {})
    completeness = data_range.get("completeness", {})
    residuals = fusion.get("residual_diagnostics", {})
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    n_alerts = pred.get("n_alerts", 0)
    recent_alerts = pred.get("alerts_last_7d", [])
    recent_alert_count = len(recent_alerts)
    n_post = drug_resp.get("n_post_days", 0)

    metric_labels = {
        "mean_rmssd": "HRV (RMSSD)",
        "mean_hr": "Heart Rate",
        "spo2_average": "SpO2",
        "temperature_deviation": "Temperature Deviation",
        "sleep_efficiency": "Sleep Efficiency",
    }
    metric_units = {
        "mean_rmssd": "ms",
        "mean_hr": "bpm",
        "spo2_average": "%",
        "temperature_deviation": "°C",
        "sleep_efficiency": "%",
    }

    def _fmt_num(value: Any, decimals: int = 2) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if not np.isfinite(numeric):
                return "N/A"
            return f"{numeric:.{decimals}f}"
        return str(value)

    def _fmt_p(value: Any) -> str:
        if value is None:
            return "p=N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return f"p={value}"
        if not np.isfinite(numeric):
            return "p=N/A"
        if numeric < 0.001:
            return "p<0.001"
        return f"p={numeric:.3f}"

    sorted_predictions = [
        (var, values)
        for var, values in pred_vars.items()
        if values.get("r2") is not None and np.isfinite(float(values["r2"]))
    ]
    sorted_predictions.sort(key=lambda item: item[1].get("r2", float("-inf")), reverse=True)
    best_pred_var, best_pred_values = (
        sorted_predictions[0] if sorted_predictions else (None, {})
    )
    best_pred_label = metric_labels.get(best_pred_var, "No prediction signal")
    best_pred_r2 = best_pred_values.get("r2")
    best_pred_rmse = best_pred_values.get("rmse_original_units")
    best_fit_summary = (
        f"{best_pred_label} with R-sq {float(best_pred_r2):.3f} and RMSE "
        f"{_fmt_num(best_pred_rmse)} {metric_units.get(best_pred_var, '')}".strip()
        if isinstance(best_pred_r2, (int, float, np.floating))
        else "Prediction-quality summary unavailable"
    )

    lead_sensor, lead_sensor_pct = (
        max(sensor_weights.items(), key=lambda item: item[1])
        if sensor_weights else ("No dominant sensor", 0.0)
    )
    lead_sensor_label = metric_labels.get(lead_sensor, str(lead_sensor).replace("_", " ").title())

    avg_completeness_pct = (
        float(np.mean(list(completeness.values()))) * 100 if completeness else None
    )
    lowest_coverage_sensor, lowest_coverage_fraction = (
        min(completeness.items(), key=lambda item: item[1])
        if completeness else ("unknown", 0.0)
    )
    lowest_coverage_label = metric_labels.get(
        lowest_coverage_sensor,
        str(lowest_coverage_sensor).replace("_", " ").title(),
    )

    n_residual_pass = sum(1 for r in residuals.values() if r.get("white_noise"))
    n_residual_total = len(residuals)

    dominant_state_name = "No dominant state"
    dominant_state_stats: dict[str, Any] = {}
    if resp_stats:
        non_stable_states = [
            item for item in resp_stats.items()
            if str(item[1].get("direction", "stable")) != "stable"
        ]
        candidate_states = non_stable_states or list(resp_stats.items())
        dominant_state_name, dominant_state_stats = max(
            candidate_states,
            key=lambda item: abs(item[1].get("shift_sd", 0) or 0),
        )
    dominant_shift = float(dominant_state_stats.get("shift_sd", 0) or 0)
    dominant_direction = str(dominant_state_stats.get("direction", "stable"))
    dominant_p = dominant_state_stats.get("mann_whitney_p")

    valid_tau = [
        (name, tau)
        for name, tau in drug_resp.get("tau_estimates", {}).items()
        if tau is not None and np.isfinite(float(tau))
    ]
    fastest_tau_name, fastest_tau = (
        min(valid_tau, key=lambda item: float(item[1]))
        if valid_tau else ("No fitted response", None)
    )

    recent_alert_status = (
        "critical" if recent_alert_count >= 3
        else "warning" if recent_alert_count > 0
        else "normal"
    )
    shift_tone = (
        "good" if dominant_direction == "improved"
        else "critical" if dominant_direction == "worsened"
        else "neutral"
    )
    prediction_tone = (
        "good" if isinstance(best_pred_r2, (int, float, np.floating)) and best_pred_r2 >= 0.30
        else "warning" if isinstance(best_pred_r2, (int, float, np.floating)) and best_pred_r2 > 0
        else "neutral"
    )
    residual_tone = (
        "good" if n_residual_total > 0 and n_residual_pass == n_residual_total
        else "warning" if n_residual_pass > 0
        else "critical"
    )
    n_post = drug_resp.get("n_post_days", 0)
    post_window_tone = "warning" if n_post < 14 else "good"

    hero_cards = [
        (
            "Strongest modeled shift",
            f"{dominant_shift:+.2f} SD" if dominant_state_stats else "N/A",
            (
                f"{dominant_state_name} · {dominant_direction.title()} · {_fmt_p(dominant_p)} · exploratory"
                if dominant_state_stats else "Shift summary unavailable"
            ),
            shift_tone,
        ),
        (
            "Best short-term fit",
            (
                f"R-sq {float(best_pred_r2):.3f}"
                if isinstance(best_pred_r2, (int, float, np.floating)) else "N/A"
            ),
            (
                f"{best_pred_label} · RMSE {_fmt_num(best_pred_rmse)} {metric_units.get(best_pred_var, '')}".strip()
                if best_pred_var else "Prediction summary unavailable"
            ),
            prediction_tone,
        ),
        (
            "Residual checks",
            f"{n_residual_pass}/{n_residual_total}" if n_residual_total else "N/A",
            (
                "Ljung-Box p>0.05 across modeled sensors"
                if n_residual_total else "Residual diagnostics unavailable"
            ),
            residual_tone,
        ),
        (
            "Post-drug window",
            f"{n_post} days",
            f"Short window; HEV diagnosed {HEV_DIAGNOSIS_DATE}",
            post_window_tone,
        ),
    ]

    hero_cards_html = "".join(
        f'<div class="dt-hero-card" data-tone="{tone}">'
        f'<div class="dt-hero-card-label">{label}</div>'
        f'<div class="dt-hero-card-value">{value}</div>'
        f'<div class="dt-hero-card-detail">{detail}</div>'
        f'</div>'
        for label, value, detail, tone in hero_cards
    )

    hero_html = (
        '<section class="dt-hero">'
        '<div class="dt-hero-glow"></div>'
        '<div class="dt-hero-glow dt-hero-glow-2"></div>'
        '<div class="dt-hero-inner">'
        '<div class="dt-hero-badge">Exploratory N=1 Physiological State-Space Model</div>'
        '<h1 class="dt-hero-title">Bayesian Cardiovascular Digital Twin</h1>'
        '<p class="dt-hero-subtitle">'
        f'This page compresses {data_range.get("n_days", 0)} wearable days into five latent '
        'physiological states, combining a Kalman smoother with UKF nonlinear dynamics to '
        'track recovery, instability, and treatment response. Use it for relative trajectory '
        'tracking and model diagnostics, not for diagnosis or causal proof.'
        '</p>'
        '<div class="dt-hero-meta">'
        f'Generated {generated} · {data_range.get("start", "?")} to {data_range.get("end", "?")} '
        f'· Post-drug window: {drug_resp.get("n_post_days", 0)} days'
        '</div>'
        '<div class="dt-hero-chip-row">'
        f'<span class="dt-hero-chip">{N_STATES} latent states</span>'
        f'<span class="dt-hero-chip">{N_OBS} wearable inputs</span>'
        f'<span class="dt-hero-chip">Acute event {KNOWN_EVENT_DATE}</span>'
        f'<span class="dt-hero-chip">Ruxolitinib {TREATMENT_START}</span>'
        f'<span class="dt-hero-chip">HEV Dx {HEV_DIAGNOSIS_DATE}</span>'
        f'<span class="dt-hero-chip">{PATIENT_LABEL}</span>'
        '</div>'
        f'<div class="dt-hero-grid">{hero_cards_html}</div>'
        '</div>'
        '</section>'
    )

    kpi_row = make_kpi_row(
        make_kpi_card(
            "Days Modeled", data_range.get("n_days", 0), "",
            status="info", decimals=0,
            detail=f'{data_range.get("start", "?")} to {data_range.get("end", "?")}',
        ),
        make_kpi_card(
            "Post-Drug Days", n_post, "",
            status="warning" if n_post < 14 else "normal", decimals=0,
            detail=f"Since {TREATMENT_START}",
            status_label="Insufficient" if n_post < 14 else "",
        ),
        make_kpi_card(
            "Innovation Alerts", recent_alert_count, "",
            status=recent_alert_status, decimals=0,
            detail=f"{n_alerts} total across the modeled window",
            status_label="Elevated" if recent_alert_status == "warning" else "",
        ),
        make_kpi_card(
            "Avg Sensor Coverage",
            avg_completeness_pct if avg_completeness_pct is not None else "N/A",
            "%" if avg_completeness_pct is not None else "",
            status="normal" if avg_completeness_pct and avg_completeness_pct >= 85 else "warning",
            decimals=1 if avg_completeness_pct is not None else 0,
            detail=f"Lowest: {lowest_coverage_label} ({lowest_coverage_fraction * 100:.0f}%)",
            status_label="" if avg_completeness_pct and avg_completeness_pct >= 85 else "Low",
        ),
        make_kpi_card(
            "Residual Checks",
            f"{n_residual_pass}/{n_residual_total}" if n_residual_total else "N/A",
            "",
            status=residual_tone if n_residual_total else "neutral",
            detail=(
                "Ljung-Box p>0.05 across modeled sensors"
                if n_residual_total else "Residual diagnostics unavailable"
            ),
            status_label="Partial" if residual_tone == "warning" and n_residual_total else "",
        ),
    )

    summary_html = (
        '<div class="dt-summary-grid">'
        '<div class="dt-summary-card">'
        '<div class="dt-summary-title">What the model suggests</div>'
        '<ul class="dt-bullet-list">'
        f'<li>Strongest modeled post-drug shift: <strong>{dominant_state_name}</strong> {dominant_shift:+.2f} SD ({_fmt_p(dominant_p)}, exploratory).</li>'
        f'<li>Best one-step predictive stream: <strong>{best_fit_summary}</strong>.</li>'
        f'<li>Recent instability is limited: <strong>{recent_alert_count}</strong> innovation alerts in the last 7 days ({n_alerts} total across the modeled window).</li>'
        '</ul>'
        '</div>'
        '<div class="dt-summary-card">'
        '<div class="dt-summary-title">What supports the model</div>'
        '<ul class="dt-bullet-list">'
        f'<li>Residual diagnostics pass for <strong>{n_residual_pass}/{n_residual_total}</strong> modeled sensors.</li>'
        f'<li>Average sensor coverage is <strong>{_fmt_num(avg_completeness_pct, 1)}%</strong>; lowest coverage is <strong>{lowest_coverage_label}</strong> at {lowest_coverage_fraction * 100:.0f}%.</li>'
        f'<li><strong>{lead_sensor_label}</strong> contributes the largest information share at {lead_sensor_pct:.1f}%.</li>'
        '</ul>'
        '</div>'
        '<div class="dt-summary-card dt-summary-card--warn">'
        '<div class="dt-summary-title">Why caution is still needed</div>'
        '<ul class="dt-bullet-list">'
        '<li>This is a <strong>single-patient (N=1)</strong> exploratory model, not a validated clinical instrument.</li>'
        f'<li>The post-drug window is only <strong>{n_post} days</strong>, which is too short for strong treatment claims.</li>'
        f'<li><strong>HEV diagnosed {HEV_DIAGNOSIS_DATE}</strong> may confound late-March shifts after ruxolitinib started on {TREATMENT_START}.</li>'
        '</ul>'
        '</div>'
        '</div>'
    )

    # --- Build body from sections ---
    body = hero_html + kpi_row
    body += make_section(
        "Read This First",
        '<div class="dt-section-intro">If someone opens only one part of this page, it should be this block. It summarizes the signal, the diagnostics that support the model, and the reasons the interpretation remains exploratory.</div>'
        f'{summary_html}',
        section_id="read-this-first",
    )

    # Section 1: State Trajectories
    body += make_section(
        "Latent State Trajectories",
        '<div class="dt-section-intro">The Kalman smoother estimates 5 latent physiological states from noisy, '
        'intermittent sensor data. Solid lines show the smoothed posterior mean; '
        'shaded bands show 95% credible intervals. '
        'Dotted lines overlay the UKF estimates for comparison. Vertical markers indicate the Feb 9 acute event, '
        'Mar 16 ruxolitinib start, and Mar 18 HEV diagnosis.</div>'
        + fig_divs[0],
        section_id="state-trajectories",
    )

    tau_list = drug_resp.get("tau_estimates", {})
    drug_summary_html = (
        '<div class="dt-stat-band">'
        f'<div class="dt-band-item"><span>Largest shift</span><strong>{dominant_state_name}</strong>'
        f'<small>{dominant_shift:+.2f} SD · {dominant_direction.title()} · {_fmt_p(dominant_p)}</small></div>'
        f'<div class="dt-band-item"><span>Fastest fitted response</span><strong>{fastest_tau_name}</strong>'
        f'<small>{_fmt_num(fastest_tau, 1)} days to reach modeled equilibrium</small></div>'
        '</div>'
    )
    state_interpretation = {
        "Autonomic Tone": "Higher = stronger vagal/recovery signal",
        "Cardiac Reserve": "Higher = better modeled cardiovascular resilience",
        "Circadian Phase": "Direction is descriptive; interpret with timing context",
        "Inflammation Level": "Higher = worse modeled inflammatory load",
        "Sleep Quality": "Higher = better modeled sleep/recovery signal",
    }
    drug_table_rows = ''.join(
        f'<tr>'
        f'<td>{name}</td>'
        f'<td>{_fmt_num(resp_stats.get(name, {}).get("pre_mean"))}</td>'
        f'<td>{_fmt_num(resp_stats.get(name, {}).get("post_mean"))}</td>'
        f'<td>{float(resp_stats.get(name, {}).get("shift_sd", 0) or 0):+.3f}</td>'
        f'<td>{state_interpretation.get(name, "Interpret relative to model definition")}</td>'
        f'<td><span class="tag tag-{resp_stats.get(name, {}).get("direction", "stable")}">'
        f'{str(resp_stats.get(name, {}).get("direction", "stable")).title()} (pre/post drug)</span></td>'
        f'<td>{_fmt_p(resp_stats.get(name, {}).get("mann_whitney_p"))}</td>'
        f'<td>{f"{_fmt_num(tau_list.get(name), 1)} days" if tau_list.get(name) else "N/A"}</td>'
        f'</tr>'
        for name in STATE_NAMES
    )

    # Section 2: Drug Response
    body += make_section(
        f"Ruxolitinib Drug Response (started {TREATMENT_START})",
        '<div class="dt-section-intro">Latent-state shifts are standardized, so the pre/post comparison shows magnitude rather than raw clinical units. '
        'Use this block to gauge which modeled subsystems moved most after treatment began. The post-drug window is short, and HEV diagnosed on '
        f'{HEV_DIAGNOSIS_DATE} may confound late-March movement.</div>'
        f'{drug_summary_html}'
        '<div class="dt-table-shell"><div class="dt-table-caption">'
        'Positive shifts indicate a higher modeled state load after treatment start; time constants estimate how quickly the post-drug response stabilized. '
        'P-values here are unadjusted and should be treated as descriptive, not confirmatory.'
        '</div><table>'
        f'<tr><th>State</th><th>Pre-drug Mean</th><th>Post-drug Mean</th>'
        f'<th>Shift (SD)</th><th>How to read direction</th><th>Direction</th><th>p-value</th><th>Time Constant</th></tr>'
        f'{drug_table_rows}'
        f'</table></div>'
        + fig_divs[3],
        section_id="drug-response",
    )

    # Section 3: Observations vs Model
    body += make_section(
        "Observations vs Model Estimates",
        '<div class="dt-section-intro">Raw sensor observations (dots) overlaid with the model\'s filtered '
        'estimates (lines). Good model fit is indicated by the line tracking '
        'the dots closely. Vertical markers indicate the Feb 9 acute event, Mar 16 ruxolitinib start, and Mar 18 HEV diagnosis.</div>'
        + fig_divs[1],
        section_id="obs-vs-model",
    )

    # Section 4: Prediction Performance
    pred_cards_html = "".join(
        f'<div class="dt-metric-card">'
        f'<div class="dt-metric-label">{metric_labels.get(var, var)}</div>'
        f'<div class="dt-metric-value">R-sq {float(values["r2"]):.3f}</div>'
        f'<div class="dt-metric-detail">RMSE {_fmt_num(values.get("rmse_original_units"))} {metric_units.get(var, "")}</div>'
        f'</div>'
        for var, values in sorted_predictions
    )
    if not pred_cards_html:
        pred_cards_html = '<div class="dt-empty-state">Prediction diagnostics were not available for this run.</div>'

    alert_html = ''.join(
        f'<div class="alert-box"><strong>{a["day"]}</strong>: '
        f'{a["variable"]} deviation = {a["sigma_ratio"]:.1f} sigma</div>'
        for a in recent_alerts[:5]
    )
    if not alert_html:
        alert_html = '<div class="dt-empty-state">No recent innovation alerts in the last 7 days.</div>'

    body += make_section(
        "Prediction Performance",
        '<div class="dt-section-intro">One-step-ahead residuals show how quickly the model tracks changes in physiology. '
        'Higher R-sq and lower RMSE indicate the latent-state model is anticipating the next observation well. '
        'Vertical markers are shown on the time-series subplots for the acute event, treatment start, and HEV diagnosis.</div>'
        '<div class="dt-mini-heading">Per-sensor forecast quality</div>'
        f'<div class="dt-metric-grid">{pred_cards_html}</div>'
        f'<div class="dt-mini-heading">Innovation alerts ({n_alerts} total, {len(recent_alerts)} in last 7 days)</div>'
        f'<div class="dt-alert-stack">{alert_html}</div>'
        + fig_divs[2],
        section_id="prediction",
    )

    # Section 5: KF vs UKF
    body += make_section(
        "KF vs UKF Comparison",
        '<div class="dt-section-intro">The Unscented Kalman Filter uses nonlinear circadian dynamics and '
        'exponential autonomic decay with sigma-point propagation (no Jacobian needed). '
        'Large differences from the linear KF suggest significant nonlinear dynamics.</div>'
        + fig_divs[4],
        section_id="kf-vs-ukf",
    )

    # Section 6: Sensor Fusion
    sensor_cards_html = "".join(
        f'<div class="dt-sensor-card">'
        f'<div class="dt-sensor-label">{metric_labels.get(sensor, sensor.replace("_", " ").title())}</div>'
        f'<div class="dt-sensor-value">{pct:.1f}%</div>'
        f'<div class="dt-sensor-detail">share of the fused state-estimation signal</div>'
        f'</div>'
        for sensor, pct in sorted(sensor_weights.items(), key=lambda item: -item[1])
    )
    if not sensor_cards_html:
        sensor_cards_html = '<div class="dt-empty-state">Sensor weighting diagnostics were not available for this run.</div>'

    body += make_section(
        "Multi-Modal Sensor Fusion Quality",
        '<div class="dt-section-intro">This block shows which wearable streams are carrying the latent-state estimates and whether the residuals still contain structure the model failed to absorb.</div>'
        f'<div class="dt-mini-heading">Sensor contribution to state estimation</div>'
        f'<div class="dt-sensor-grid">{sensor_cards_html}</div>'
        f'<div class="dt-mini-heading">Residual diagnostics</div>'
        f'<p class="dt-note">White noise residuals (Ljung-Box p>0.05) indicate the model '
        f'captures temporal structure well. Significant autocorrelation suggests '
        f'model misspecification for that sensor.</p>'
        + fig_divs[5],
        section_id="sensor-fusion",
    )

    # Disclaimer
    body += (
        '<div class="disclaimer">'
        '<strong>Disclaimer:</strong> This is a computational model for research '
        'and self-monitoring purposes only. It is NOT a medical device and should '
        'NOT be used for clinical decision-making. The Oura Ring is a consumer '
        'wearable; its measurements have known limitations in accuracy. '
        'All state estimates are model-dependent and should be interpreted with '
        'appropriate uncertainty. Consult qualified healthcare professionals for '
        'medical decisions.'
        '</div>'
    )

    # Extra CSS for tags, alerts, and disclaimer adapted to dark theme
    extra_css = f"""
/* Hide standard header - Digital Twin uses a custom hero */
.odt-header {{ display: none; }}

/* === Hero === */
.dt-hero {{
    position: relative;
    margin: -18px -32px 28px;
    padding: 56px 40px 40px;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 34%),
        radial-gradient(circle at top right, rgba(139,92,246,0.14), transparent 30%),
        linear-gradient(135deg, #0D1120 0%, #12182B 55%, #0D1120 100%);
    border-bottom: 1px solid rgba(59,130,246,0.10);
    border-radius: 0 0 24px 24px;
}}
.dt-hero-glow {{
    position: absolute;
    width: 560px;
    height: 560px;
    border-radius: 50%;
    filter: blur(120px);
    background: radial-gradient(circle, rgba(16,185,129,0.18) 0%, transparent 70%);
    top: -220px;
    left: -120px;
    opacity: 0.35;
    pointer-events: none;
}}
.dt-hero-glow-2 {{
    background: radial-gradient(circle, rgba(139,92,246,0.20) 0%, transparent 70%);
    top: -160px;
    right: -180px;
    left: auto;
}}
.dt-hero-inner {{
    position: relative;
    z-index: 1;
    max-width: 1160px;
}}
.dt-hero-badge {{
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.6875rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT_CYAN};
    background: rgba(34,211,238,0.10);
    border: 1px solid rgba(34,211,238,0.22);
    margin-bottom: 18px;
}}
.dt-hero-title {{
    font-size: clamp(2.4rem, 4vw, 3.5rem);
    font-weight: 800;
    line-height: 1.04;
    letter-spacing: -0.04em;
    color: {TEXT_PRIMARY};
    max-width: 900px;
    margin-bottom: 16px;
}}
.dt-hero-subtitle {{
    max-width: 820px;
    font-size: 1.02rem;
    line-height: 1.75;
    color: {TEXT_SECONDARY};
    margin-bottom: 14px;
}}
.dt-hero-meta {{
    font-size: 0.8125rem;
    color: {TEXT_TERTIARY};
    letter-spacing: 0.02em;
    margin-bottom: 18px;
}}
.dt-hero-chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 24px;
}}
.dt-hero-chip {{
    display: inline-flex;
    align-items: center;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(26,29,39,0.58);
    border: 1px solid rgba(255,255,255,0.08);
    color: {TEXT_SECONDARY};
    font-size: 0.75rem;
    font-weight: 600;
}}
.dt-hero-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}}
.dt-hero-card {{
    background: rgba(17, 24, 39, 0.62);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px 18px 16px;
    min-height: 152px;
    box-shadow: 0 18px 36px rgba(0,0,0,0.22);
}}
.dt-hero-card[data-tone="critical"] {{ border-color: rgba(239,68,68,0.22); }}
.dt-hero-card[data-tone="warning"] {{ border-color: rgba(245,158,11,0.22); }}
.dt-hero-card[data-tone="good"] {{ border-color: rgba(16,185,129,0.22); }}
.dt-hero-card[data-tone="info"] {{ border-color: rgba(59,130,246,0.22); }}
.dt-hero-card-label {{
    font-size: 0.6875rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_TERTIARY};
    font-weight: 700;
    margin-bottom: 14px;
}}
.dt-hero-card-value {{
    font-size: clamp(1.7rem, 2.6vw, 2.2rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    color: {TEXT_PRIMARY};
    margin-bottom: 10px;
}}
.dt-hero-card-detail {{
    font-size: 0.8125rem;
    line-height: 1.55;
    color: {TEXT_SECONDARY};
}}

/* === Section helpers === */
.dt-section-intro {{
    font-size: 0.96875rem;
    line-height: 1.75;
    color: {TEXT_SECONDARY};
    max-width: 900px;
    margin-bottom: 18px;
}}
.dt-summary-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
}}
.dt-summary-card {{
    padding: 18px 20px;
    border-radius: 16px;
    background: rgba(17,24,39,0.52);
    border: 1px solid rgba(255,255,255,0.07);
}}
.dt-summary-card--warn {{
    border-color: rgba(245,158,11,0.22);
    background: rgba(245,158,11,0.05);
}}
.dt-summary-title {{
    font-size: 0.8125rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_TERTIARY};
    font-weight: 700;
    margin-bottom: 12px;
}}
.dt-bullet-list {{
    margin: 0;
    padding-left: 18px;
    color: {TEXT_SECONDARY};
}}
.dt-bullet-list li {{
    margin: 0 0 10px;
    line-height: 1.65;
}}
.dt-bullet-list li:last-child {{
    margin-bottom: 0;
}}
.dt-mini-heading {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {TEXT_TERTIARY};
    font-weight: 700;
    margin: 18px 0 12px;
}}
.dt-note {{
    font-size: 0.875rem;
    line-height: 1.7;
    color: {TEXT_SECONDARY};
    margin-bottom: 18px;
}}
.dt-empty-state {{
    padding: 16px 18px;
    border-radius: 12px;
    background: rgba(26,29,39,0.5);
    border: 1px solid rgba(255,255,255,0.06);
    color: {TEXT_TERTIARY};
    font-size: 0.875rem;
}}

/* === Summary bands === */
.dt-stat-band {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin: 16px 0 22px;
}}
.dt-band-item {{
    padding: 16px 18px;
    border-radius: 14px;
    background: rgba(26,29,39,0.55);
    border: 1px solid rgba(255,255,255,0.07);
}}
.dt-band-item span {{
    display: block;
    font-size: 0.6875rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_TERTIARY};
    font-weight: 700;
    margin-bottom: 8px;
}}
.dt-band-item strong {{
    display: block;
    font-size: 1.1rem;
    color: {TEXT_PRIMARY};
    margin-bottom: 6px;
}}
.dt-band-item small {{
    display: block;
    font-size: 0.8125rem;
    line-height: 1.55;
    color: {TEXT_SECONDARY};
}}

/* === Prediction quality cards === */
.dt-metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 0 0 20px;
}}
.dt-metric-card {{
    padding: 16px 18px;
    border-radius: 14px;
    background: rgba(26,29,39,0.52);
    border: 1px solid rgba(255,255,255,0.07);
}}
.dt-metric-label {{
    font-size: 0.75rem;
    color: {TEXT_SECONDARY};
    font-weight: 700;
    margin-bottom: 10px;
}}
.dt-metric-value {{
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: {TEXT_PRIMARY};
    margin-bottom: 6px;
}}
.dt-metric-detail {{
    font-size: 0.8125rem;
    color: {TEXT_TERTIARY};
}}

/* === Sensor cards === */
.dt-sensor-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 0 0 18px;
}}
.dt-sensor-card {{
    padding: 16px 18px;
    border-radius: 14px;
    background: rgba(26,29,39,0.52);
    border: 1px solid rgba(255,255,255,0.07);
}}
.dt-sensor-label {{
    font-size: 0.75rem;
    color: {TEXT_SECONDARY};
    font-weight: 700;
    margin-bottom: 10px;
}}
.dt-sensor-value {{
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: {TEXT_PRIMARY};
    margin-bottom: 6px;
}}
.dt-sensor-detail {{
    font-size: 0.8125rem;
    line-height: 1.55;
    color: {TEXT_TERTIARY};
}}

/* === Alert stack === */
.dt-alert-stack {{
    display: grid;
    gap: 10px;
    margin: 0 0 18px;
}}

/* === Status Tags (pill-shaped) === */
.tag {{
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: background 0.15s, transform 0.15s;
}}
.tag:hover {{
    transform: translateY(-1px);
}}
.tag-improved {{
    background: rgba(16, 185, 129, 0.12);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.2);
}}
.tag-worsened {{
    background: rgba(239, 68, 68, 0.12);
    color: #FCA5A5;
    border: 1px solid rgba(239, 68, 68, 0.2);
}}
.tag-stable {{
    background: rgba(107, 114, 128, 0.12);
    color: {TEXT_SECONDARY};
    border: 1px solid rgba(107, 114, 128, 0.15);
}}

/* === Alert Boxes (gradient left border) === */
.alert-box {{
    background: rgba(245, 158, 11, 0.06);
    border: 1px solid rgba(245, 158, 11, 0.15);
    border-left: 3px solid {ACCENT_AMBER};
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.875rem;
    color: {TEXT_PRIMARY};
    line-height: 1.5;
    transition: background 0.15s;
}}
.alert-box:hover {{
    background: rgba(245, 158, 11, 0.09);
}}
.alert-box strong {{
    color: {ACCENT_AMBER};
    font-weight: 700;
}}

/* === Disclaimer (glass morphism) === */
.disclaimer {{
    background: rgba(26, 29, 39, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 22px 24px;
    margin-top: 36px;
    font-size: 0.875rem;
    color: {TEXT_TERTIARY};
    line-height: 1.7;
    box-shadow: 0 18px 36px rgba(0,0,0,0.18);
}}
.disclaimer strong {{
    color: {TEXT_SECONDARY};
    font-weight: 600;
}}

/* === Table shell === */
.dt-table-shell {{
    margin: 16px 0 22px;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(17,24,39,0.42);
}}
.dt-table-caption {{
    padding: 14px 18px;
    font-size: 0.8125rem;
    line-height: 1.55;
    color: {TEXT_TERTIARY};
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
}}
.dt-table-shell table {{
    margin: 0;
}}

@media (max-width: 1024px) {{
    .dt-hero-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .dt-summary-grid {{
        grid-template-columns: 1fr;
    }}
}}

@media (max-width: 900px) {{
    .dt-hero {{
        margin: -20px -16px 24px;
        padding: 36px 20px 30px;
        border-radius: 0 0 20px 20px;
    }}
}}

@media (max-width: 640px) {{
    .dt-hero-grid,
    .dt-stat-band {{
        grid-template-columns: 1fr;
    }}
    .dt-hero-title {{
        font-size: 1.95rem;
    }}
    .dt-hero-subtitle {{
        font-size: 0.9375rem;
    }}
}}
"""

    return wrap_html(
        title="Bayesian Cardiovascular Digital Twin",
        body_content=body,
        report_id="digital_twin",
        subtitle="5-state Kalman filter/smoother with UKF nonlinear dynamics",
        extra_css=extra_css,
        data_start=data_range.get("start"),
        data_end=data_range.get("end"),
        post_days=n_post,
    )


# ===========================================================================
# Section 10: Main Orchestration
# ===========================================================================
MIN_DAYS_FOR_UKF = 7  # Minimum days of data required for UKF analysis


def main() -> None:
    """Run the complete digital twin pipeline."""
    print("=" * 70)
    print("  BAYESIAN CARDIOVASCULAR DIGITAL TWIN")
    print(f"  {PATIENT_LABEL} - Oura Ring Data")
    print("=" * 70)
    t_start = time.time()

    # --- Step 1: Load data ---
    daily = load_oura_data()

    # --- Empty data guard ---
    if len(daily) < MIN_DAYS_FOR_UKF:
        log("MAIN", f"INSUFFICIENT DATA: {len(daily)} days < {MIN_DAYS_FOR_UKF} minimum. "
            "Skipping Kalman/UKF analysis.")
        metrics["error"] = f"Only {len(daily)} days of data (need >= {MIN_DAYS_FOR_UKF})"
        # Generate minimal report with warning using theme
        warn_body = make_section(
            "Insufficient Data",
            f'<p>Only <strong>{len(daily)} days</strong> of Oura Ring data available. '
            f'The UKF and Kalman filter analyses require at least '
            f'<strong>{MIN_DAYS_FOR_UKF} days</strong>.</p>'
            f'<p>Continue collecting data and re-run the analysis.</p>',
        )
        html = wrap_html(
            title="Digital Twin - Insufficient Data",
            body_content=warn_body,
            report_id="digital_twin",
            subtitle="Not enough data for analysis",
        )
        HTML_OUTPUT.write_text(html, encoding="utf-8")
        log("MAIN", f"Minimal report written to {HTML_OUTPUT.name}")
        return

    # --- Step 2: Standardize observations ---
    obs_masked, scalers = standardize_observations(daily)
    metrics["scalers"] = {k: {"mean": v[0], "std": v[1]} for k, v in scalers.items()}

    # --- Step 3: Linear Kalman Filter ---
    print("\n--- Linear Kalman Filter ---")
    kf_results = run_kalman_filter(obs_masked, daily)

    # --- Step 4: Unscented Kalman Filter ---
    print("\n--- Unscented Kalman Filter ---")
    ukf_results = run_ukf(obs_masked, daily)

    # --- Step 5: Prediction analysis ---
    print("\n--- Prediction Analysis ---")
    pred_results = run_prediction_analysis(kf_results, obs_masked, daily, scalers)

    # --- Step 6: Drug response analysis ---
    print("\n--- Drug Response Analysis ---")
    drug_response = analyze_drug_response(kf_results, ukf_results, daily, scalers)

    # --- Step 7: Sensor fusion quality ---
    print("\n--- Sensor Fusion Analysis ---")
    fusion_results = analyze_sensor_fusion(kf_results, obs_masked, daily)

    # --- Step 8: Generate visualizations ---
    print("\n--- Generating Visualizations ---")
    fig1 = create_state_trajectory_figure(kf_results, ukf_results, daily, drug_response)
    fig2 = create_observation_overlay_figure(kf_results, obs_masked, daily, scalers)
    fig3 = create_prediction_figure(pred_results, obs_masked, daily, scalers)
    fig4 = create_drug_response_figure(kf_results, drug_response, daily)
    fig5 = create_kf_vs_ukf_figure(kf_results, ukf_results, daily)
    fig6 = create_sensor_fusion_figure(fusion_results, obs_masked, daily)

    all_figs = [fig1, fig2, fig3, fig4, fig5, fig6]

    # --- Step 9: Generate HTML report ---
    print("\n--- Generating Report ---")
    html = generate_html_report(all_figs, daily)
    HTML_OUTPUT.write_text(html, encoding="utf-8")
    log("REPORT", f"HTML report: {HTML_OUTPUT}")

    # --- Step 10: Save JSON metrics ---
    total_time = time.time() - t_start
    metrics["total_time_s"] = round(total_time, 2)

    # Clean up non-serializable values
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    metrics_clean = make_serializable(metrics)
    JSON_OUTPUT.write_text(
        json.dumps(metrics_clean, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log("REPORT", f"JSON metrics: {JSON_OUTPUT}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  DIGITAL TWIN COMPLETE")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Data: {metrics['data_range'].get('n_days', 0)} days")
    print(f"  Log-likelihood: {metrics['kalman'].get('log_likelihood', 'N/A')}")
    print(f"  Prediction alerts: {metrics['prediction'].get('n_alerts', 0)}")
    n_post = metrics['drug_response'].get('n_post_days', 0)
    print(f"  Post-drug days: {n_post}")
    if n_post > 0:
        for name in STATE_NAMES:
            s = metrics['drug_response'].get('response_stats', {}).get(name, {})
            print(f"    {name}: {s.get('shift_sd', 0):+.3f} SD ({s.get('direction', 'N/A')})")
    print(f"\n  Report: {HTML_OUTPUT}")
    print(f"  Metrics: {JSON_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n  FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
