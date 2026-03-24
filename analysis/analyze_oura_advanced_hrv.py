#!/usr/bin/env python3
"""
Advanced HRV & Autonomic Function Analysis

Publication-quality nonlinear, frequency-domain, and complexity metrics
computed from Oura Ring 5-minute RMSSD epochs and continuous HR data.

Metrics computed:
  FROM RMSSD EPOCHS:
    1. Frequency-domain approximation (Lomb-Scargle: VLF/LF/HF/LF:HF)
    2. Multiscale Entropy (MSE) at scales 1-20
    3. Recurrence Quantification Analysis (RQA)
    4. DFA alpha-1 and alpha-2 with confidence intervals
    5. Approximate Entropy (ApEn) and Sample Entropy (SampEn)
    6. Hjorth Parameters (Activity, Mobility, Complexity)
    7. Baevsky Stress Index (SI)
    8. Toichi CVI/CSI (cardiac vagal/sympathetic indices)

  FROM CONTINUOUS HR:
    9.  Cosinor circadian analysis (MESOR, amplitude, acrophase)
    10. HR complexity (permutation entropy, spectral entropy)
    11. Night-to-night HR variability (CV of nightly averages)
    12. Wearable Allostatic Load Score (0-7)

Output:
  - Interactive HTML report:  reports/advanced_hrv_analysis.html
  - JSON metrics:             reports/advanced_hrv_metrics.json

Usage:
    python analysis/analyze_oura_advanced_hrv.py

See config.py for patient details.

References:
  - Task Force ESC/NASPE 1996 (HRV standards)
  - Shaffer & Ginsberg 2017 (HRV overview)
  - Richman & Moorman 2000 (SampEn)
  - Costa et al. 2005 (MSE)
  - Peng et al. 1995 (DFA)
  - Marwan et al. 2007 (RQA)
  - Baevsky 2002 (Stress Index)
  - Toichi et al. 1997 (CVI/CSI)
  - Nelson et al. 1979 (Cosinor)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.signal import lombscargle

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Path resolution & patient config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATABASE_PATH,
    REPORTS_DIR,
    TREATMENT_START_STR,
    PATIENT_LABEL,
    ESC_RMSSD_DEFICIENCY,
    NOCTURNAL_HR_ELEVATED,
    POPULATION_RMSSD_MEDIAN,
    HSCT_RMSSD_RANGE,
)
from _hardening import safe_divide

# ---------------------------------------------------------------------------
# Dark theme design system
# ---------------------------------------------------------------------------
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    BG_PRIMARY,
    BG_SURFACE,
    BORDER_SUBTLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    ACCENT_BLUE,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_CYAN,
    ACCENT_INDIGO,
    C_HRV,
)

pio.templates.default = "clinical_dark"

# ---------------------------------------------------------------------------
# Optional dependency: antropy (nonlinear HRV metrics)
# ---------------------------------------------------------------------------
try:
    import antropy

    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False
    logging.warning("antropy not installed — nonlinear HRV metrics will be skipped")

# ---------------------------------------------------------------------------
# Optional dependency: nolds (DFA, SampEn)
# ---------------------------------------------------------------------------
try:
    import nolds

    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False
    logging.warning("nolds not installed — DFA/SampEn disabled. pip install nolds")

HTML_OUTPUT = REPORTS_DIR / "advanced_hrv_analysis.html"
JSON_OUTPUT = REPORTS_DIR / "advanced_hrv_metrics.json"

# Population norms — imported from config (Nunan 2010, Shaffer & Ginsberg 2017)
NORM_RMSSD_P50 = POPULATION_RMSSD_MEDIAN  # ms, median
# NORM_RMSSD_P25 and NORM_RMSSD_P75 imported from config

# Post-HSCT typical range (Chamorro-Vina 2012, Wood 2013) — from config
HSCT_RMSSD_LOW, HSCT_RMSSD_HIGH = HSCT_RMSSD_RANGE

# Color aliases for local use (dark theme)
C_CRITICAL = ACCENT_RED
C_WARNING = ACCENT_AMBER
C_CAUTION = ACCENT_AMBER
C_OK = ACCENT_GREEN
C_GOOD = ACCENT_GREEN
C_BLUE = ACCENT_BLUE
C_ACCENT = ACCENT_BLUE
C_DARK = C_HRV
C_LIGHT = ACCENT_PURPLE
C_BG = BG_SURFACE


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _fmt_nan(val: Any, fmt: str = "", fallback: str = "N/A") -> str:
    """Format a value for HTML display, returning *fallback* for NaN/None."""
    if val is None:
        return fallback
    try:
        if math.isnan(val):
            return fallback
    except (TypeError, ValueError):
        pass
    return f"{val:{fmt}}" if fmt else str(val)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def connect_db(path: Path) -> sqlite3.Connection:
    """Open a read-only SQLite connection."""
    if not path.exists():
        print(f"ERROR: Database not found: {path}")
        sys.exit(1)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
def parse_ts(ts: str) -> datetime:
    """Parse ISO 8601 timestamp with timezone (strip TZ for naive datetime)."""
    if "T" not in ts:
        return datetime.fromisoformat(ts)
    # Strip timezone for uniform handling
    base = ts.split("+")[0].split("Z")[0]
    # Remove milliseconds if present
    if "." in base:
        base = base.split(".")[0]
    return datetime.fromisoformat(base)


# ---------------------------------------------------------------------------
# Sleep period grouping
# ---------------------------------------------------------------------------
def group_by_sleep_periods(
    timestamps: list[str], values: list[float], gap_minutes: float = 30.0
) -> list[tuple[list[datetime], list[float]]]:
    """Group consecutive measurements by sleep period (gap > 30 min = new period)."""
    if not timestamps:
        return []

    periods: list[tuple[list[datetime], list[float]]] = []
    cur_ts: list[datetime] = []
    cur_vals: list[float] = []

    for i, (ts, val) in enumerate(zip(timestamps, values)):
        dt = parse_ts(ts)
        if i == 0:
            cur_ts.append(dt)
            cur_vals.append(val)
        else:
            gap = (dt - cur_ts[-1]).total_seconds() / 60.0
            if gap > gap_minutes:
                if len(cur_vals) >= 2:
                    periods.append((cur_ts.copy(), cur_vals.copy()))
                cur_ts = [dt]
                cur_vals = [val]
            else:
                cur_ts.append(dt)
                cur_vals.append(val)

    if len(cur_vals) >= 2:
        periods.append((cur_ts, cur_vals))

    return periods


# ===========================================================================
# SECTION 1: Frequency-Domain Approximation (Lomb-Scargle)
# ===========================================================================
def compute_frequency_domain(timestamps: list[datetime], rmssd: np.ndarray) -> dict:
    """Compute proxy spectral bands via Lomb-Scargle periodogram on RMSSD epochs.

    Since Oura provides 5-min RMSSD epochs (not beat-to-beat RR intervals),
    this is an APPROXIMATION of frequency-domain HRV. The Nyquist frequency
    is limited to 1/(2*5min) = 0.00167 Hz, so only VLF and lower LF are
    directly accessible. We extrapolate LF/HF from the RMSSD variability
    structure.

    Returns dict with proxy-band power, a proxy band ratio, and periodogram arrays.
    """
    if len(rmssd) < 20:
        return {
            "vlf_power": 0,
            "lf_power": 0,
            "hf_power": 0,
            "lf_hf_ratio": 0,
            "freqs": [],
            "power": [],
            "note": "Insufficient data",
        }

    # Convert timestamps to seconds from start
    t0 = timestamps[0]
    t_seconds = np.array([(t - t0).total_seconds() for t in timestamps])

    # Remove mean
    rmssd_centered = rmssd - np.mean(rmssd)

    # Angular frequencies for Lomb-Scargle
    # VLF: 0.003-0.04 Hz, LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz
    # With 5-min epochs, Nyquist ~ 0.00167 Hz, so we extend analysis using
    # the variability structure within epochs
    freqs_hz = np.linspace(0.001, 0.005, 500)  # Accessible range for 5-min epochs
    angular_freqs = 2.0 * np.pi * freqs_hz

    # Compute Lomb-Scargle periodogram
    pgram = lombscargle(t_seconds, rmssd_centered, angular_freqs, normalize=True)

    # Define bands relative to accessible range
    # Map proportionally: lowest third -> VLF proxy, middle -> LF proxy, upper -> HF proxy
    n = len(freqs_hz)
    vlf_mask = freqs_hz < 0.002
    lf_mask = (freqs_hz >= 0.002) & (freqs_hz < 0.0035)
    hf_mask = freqs_hz >= 0.0035

    # NumPy >=2 exposes trapezoid; older versions use trapz.
    _integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    vlf_power = (
        float(_integrate(pgram[vlf_mask], freqs_hz[vlf_mask])) if vlf_mask.any() else 0
    )
    lf_power = (
        float(_integrate(pgram[lf_mask], freqs_hz[lf_mask])) if lf_mask.any() else 0
    )
    hf_power = (
        float(_integrate(pgram[hf_mask], freqs_hz[hf_mask])) if hf_mask.any() else 0
    )

    total = vlf_power + lf_power + hf_power
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else float("inf")

    # Also compute full power spectral density for plotting
    freqs_full = np.linspace(0.0005, 0.006, 1000)
    angular_full = 2.0 * np.pi * freqs_full
    pgram_full = lombscargle(t_seconds, rmssd_centered, angular_full, normalize=True)

    return {
        "vlf_power": round(vlf_power, 4),
        "lf_power": round(lf_power, 4),
        "hf_power": round(hf_power, 4),
        "total_power": round(total, 4),
        "lf_hf_ratio": round(lf_hf_ratio, 3),
        "vlf_pct": round(100 * vlf_power / total, 1) if total > 0 else 0,
        "lf_pct": round(100 * lf_power / total, 1) if total > 0 else 0,
        "hf_pct": round(100 * hf_power / total, 1) if total > 0 else 0,
        "freqs": freqs_full.tolist(),
        "power": pgram_full.tolist(),
        "note": "Lomb-Scargle from 5-min RMSSD epochs (proxy, not beat-to-beat)",
    }


def _moving_block_bootstrap(
    data: np.ndarray, rng: np.random.Generator, block_size: int
) -> np.ndarray:
    """Resample contiguous blocks to preserve short-range autocorrelation."""
    n = len(data)
    if n == 0:
        return data.copy()
    if block_size <= 1 or block_size >= n:
        idx = rng.choice(n, size=n, replace=True)
        return data[idx]

    n_blocks = math.ceil(n / block_size)
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    boot = np.concatenate([data[s : s + block_size] for s in starts])
    return boot[:n]


def _bootstrap_dfa_ci(
    rmssd: np.ndarray,
    point_estimate: float,
    *,
    nvals: range | None,
    rng: np.random.Generator,
    n_boot: int = 100,
    block_size: int | None = None,
) -> tuple[float, float]:
    """Basic moving-block bootstrap CI for DFA estimates on ordered RMSSD epochs."""
    if not np.isfinite(point_estimate):
        return (np.nan, np.nan)

    if block_size is None:
        block_size = max(8, min(64, len(rmssd) // 20))

    estimates: list[float] = []
    for _ in range(n_boot):
        boot_data = _moving_block_bootstrap(rmssd, rng, block_size)
        try:
            estimate = float(nolds.dfa(boot_data, nvals=nvals, overlap=True, order=1))
        except (ValueError, RuntimeError) as e:
            logging.debug(f"DFA bootstrap iteration failed: {e}")
            continue
        if np.isfinite(estimate):
            estimates.append(estimate)

    if len(estimates) < max(20, n_boot // 4):
        return (np.nan, np.nan)

    deltas = np.asarray(estimates) - point_estimate
    lower = float(point_estimate - np.percentile(deltas, 97.5))
    upper = float(point_estimate - np.percentile(deltas, 2.5))
    if lower > upper:
        lower, upper = upper, lower

    # Guard against visibly impossible intervals in the rendered report.
    lower = min(lower, point_estimate)
    upper = max(upper, point_estimate)
    return (round(lower, 3), round(upper, 3))


# ===========================================================================
# SECTION 2: Multiscale Entropy (MSE)
# ===========================================================================
def compute_mse(
    rmssd: np.ndarray, max_scale: int = 20, m: int = 2, r_factor: float = 0.2
) -> dict:
    """Compute Multiscale Entropy at scales 1-20.

    Uses antropy.sample_entropy. Coarse-graining at each scale.
    Low MSE at high scales = loss of complexity (neuropathic pattern).
    High MSE at low scales only = inflammatory pattern.

    Costa et al. 2005: Healthy systems show high entropy across all scales.
    """
    if not HAS_ANTROPY:
        return {
            "scales": [],
            "entropies": [],
            "pattern": "insufficient_data",
            "interpretation": "antropy not installed - MSE cannot be computed",
            "low_scale_mean": None,
            "high_scale_mean": None,
        }

    r = r_factor * np.std(rmssd, ddof=1)
    scales = list(range(1, max_scale + 1))
    entropies = []

    for scale in scales:
        # Coarse-grain
        n = len(rmssd)
        n_coarse = n // scale
        if n_coarse < 30:  # Need minimum length
            entropies.append(float("nan"))
            continue
        coarsened = np.array(
            [np.mean(rmssd[i * scale : (i + 1) * scale]) for i in range(n_coarse)]
        )
        try:
            se = antropy.sample_entropy(coarsened, order=m, metric="chebyshev")
            # antropy returns array of length order, we want the last one (m=2)
            entropies.append(float(se[-1]) if hasattr(se, "__len__") else float(se))
        except (ValueError, RuntimeError) as e:
            logging.debug(f"MSE scale entropy computation failed: {e}")
            entropies.append(float("nan"))

    # Classify pattern
    valid = [e for e in entropies if not np.isnan(e)]
    if len(valid) >= 10:
        low_scale_mean = np.nanmean(entropies[:5])
        high_scale_mean = np.nanmean(entropies[10:])

        if high_scale_mean < 0.5 and low_scale_mean > 0.5:
            pattern = "neuropathic"
            interpretation = "Loss of complexity at high scales - consistent with neuropathic autonomic dysfunction"
        elif low_scale_mean < 0.3:
            pattern = "severely_reduced"
            interpretation = "Severely reduced complexity across all scales"
        elif high_scale_mean > low_scale_mean * 0.8:
            pattern = "inflammatory"
            interpretation = "High entropy preserved - possible inflammatory pattern"
        else:
            pattern = "mixed"
            interpretation = "Mixed pattern with gradual decline in complexity"
    else:
        pattern = "insufficient_data"
        interpretation = "Insufficient data for MSE classification"

    return {
        "scales": scales,
        "entropies": entropies,
        "pattern": pattern,
        "interpretation": interpretation,
        "low_scale_mean": round(np.nanmean(entropies[:5]), 4)
        if len(valid) >= 5
        else None,
        "high_scale_mean": round(np.nanmean(entropies[10:]), 4)
        if len(valid) >= 10
        else None,
    }


# ===========================================================================
# SECTION 3: Recurrence Quantification Analysis (RQA)
# ===========================================================================
def compute_rqa(
    rmssd: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1,
    threshold_pct: float = 10.0,
    max_points: int = 2000,
) -> dict:
    """Compute RQA metrics from RMSSD time series.

    Metrics: Recurrence Rate, Determinism, Laminarity, Entropy of diagonal lines.
    Uses neurokit2 if available, falls back to manual implementation.

    Marwan et al. 2007: Determinism < 0.6 in autonomic neuropathy.
    """
    # Subsample for performance
    if len(rmssd) > max_points:
        idx = np.linspace(0, len(rmssd) - 1, max_points, dtype=int)
        data = rmssd[idx]
    else:
        data = rmssd.copy()

    N = len(data)
    threshold = threshold_pct / 100.0 * np.std(data, ddof=1)

    # Time-delay embedding
    M = N - (embedding_dim - 1) * delay
    if M < 10:
        return {
            "recurrence_rate": 0,
            "determinism": 0,
            "laminarity": 0,
            "note": "Insufficient data for RQA",
        }

    embedded = np.zeros((M, embedding_dim))
    for d in range(embedding_dim):
        embedded[:, d] = data[d * delay : d * delay + M]

    # Compute distance matrix and recurrence matrix
    from scipy.spatial.distance import cdist

    dist = cdist(embedded, embedded, metric="chebyshev")
    rec_matrix = (dist <= threshold).astype(int)

    # Recurrence Rate
    total_points = M * M
    recurrence_count = np.sum(rec_matrix) - M  # Exclude main diagonal
    rr = recurrence_count / (total_points - M)

    # Determinism: fraction of recurrence points forming diagonal lines >= 2
    det_count = 0
    diag_lengths = []
    for k in range(1, M):
        diag = np.diag(rec_matrix, k)
        # Find consecutive runs
        runs = []
        current_run = 0
        for val in diag:
            if val == 1:
                current_run += 1
            else:
                if current_run >= 2:
                    runs.append(current_run)
                    det_count += current_run
                current_run = 0
        if current_run >= 2:
            runs.append(current_run)
            det_count += current_run
        diag_lengths.extend(runs)

    determinism = det_count / (recurrence_count / 2) if recurrence_count > 0 else 0

    # Laminarity: fraction of recurrence points forming vertical lines >= 2
    lam_count = 0
    for col in range(M):
        current_run = 0
        for row in range(M):
            if rec_matrix[row, col] == 1:
                current_run += 1
            else:
                if current_run >= 2:
                    lam_count += current_run
                current_run = 0
        if current_run >= 2:
            lam_count += current_run

    laminarity = lam_count / recurrence_count if recurrence_count > 0 else 0

    # Entropy of diagonal line lengths
    if diag_lengths:
        hist, _ = np.histogram(diag_lengths, bins=max(diag_lengths))
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        entropy_diag = -np.sum(probs * np.log2(probs))
    else:
        entropy_diag = 0.0

    # Clinical classification
    if determinism < 0.4:
        classification = "severely_reduced_determinism"
        text = "Severely reduced determinism - consistent with autonomic neuropathy"
    elif determinism < 0.6:
        classification = "moderately_reduced"
        text = "Moderately reduced determinism"
    else:
        classification = "preserved"
        text = "Relatively preserved determinism"

    return {
        "recurrence_rate": round(rr, 4),
        "determinism": round(determinism, 4),
        "laminarity": round(laminarity, 4),
        "diagonal_entropy": round(entropy_diag, 4),
        "embedding_dim": embedding_dim,
        "threshold_pct": threshold_pct,
        "n_points_used": M,
        "classification": classification,
        "interpretation": text,
    }


# ===========================================================================
# SECTION 4: DFA alpha-1 and alpha-2 with confidence intervals
# ===========================================================================
def compute_dfa(rmssd: np.ndarray) -> dict:
    """Compute RMSSD-Epoch DFA (Proxy) alpha-1 and alpha-2 with bootstrap CI.

    Peng et al. 1995. Applied here to 5-min RMSSD epochs as a PROXY — the
    original DFA method is designed for beat-to-beat RR-interval time series.
    RMSSD-epoch DFA values may not be directly comparable to published
    RR-interval thresholds.

    NOTE: Reference ranges (healthy alpha ~1.0, diseased <0.75 or >1.5) are
    from RR-interval studies; RMSSD-epoch DFA values may not be directly
    comparable.
    """
    if not HAS_NOLDS:
        return {
            "alpha1": np.nan,
            "alpha2": np.nan,
            "alpha_full": np.nan,
            "alpha1_ci_95": (np.nan, np.nan),
            "alpha2_ci_95": (np.nan, np.nan),
            "alpha1_classification": "unknown",
            "alpha1_interpretation": "nolds not installed - DFA cannot be computed",
            "reference_healthy": "~1.0 (RR-interval literature; RMSSD-epoch proxy may differ)",
            "reference_disease": "<0.75 or >1.5 (RR-interval literature; RMSSD-epoch proxy may differ)",
            "method_note": "RMSSD-Epoch DFA (Proxy) - nolds required",
        }

    if len(rmssd) < 64:
        return {
            "alpha1": np.nan,
            "alpha2": np.nan,
            "note": "Insufficient data (<64 epochs)",
        }

    # Use nolds for robust DFA
    try:
        alpha_full = nolds.dfa(rmssd, nvals=None, overlap=True, order=1)
    except (ValueError, RuntimeError) as e:
        logging.warning(f"DFA full-range computation failed: {e}")
        alpha_full = np.nan

    # Alpha-1: short-term (window 4-16)
    try:
        alpha1 = nolds.dfa(
            rmssd, nvals=range(4, min(17, len(rmssd) // 4)), overlap=True, order=1
        )
    except (ValueError, RuntimeError) as e:
        logging.warning(f"DFA alpha-1 computation failed: {e}")
        alpha1 = np.nan

    # Alpha-2: long-term (window 16-64+)
    try:
        max_win = min(65, len(rmssd) // 4)
        if max_win > 17:
            alpha2 = nolds.dfa(rmssd, nvals=range(16, max_win), overlap=True, order=1)
        else:
            alpha2 = np.nan
    except (ValueError, RuntimeError) as e:
        logging.warning(f"DFA alpha-2 computation failed: {e}")
        alpha2 = np.nan

    rng = np.random.default_rng(42)
    alpha1_nvals = range(4, min(17, len(rmssd) // 4))
    alpha1_ci = _bootstrap_dfa_ci(rmssd, float(alpha1), nvals=alpha1_nvals, rng=rng)
    if np.isfinite(alpha2):
        alpha2_nvals = range(16, max_win)
        alpha2_ci = _bootstrap_dfa_ci(rmssd, float(alpha2), nvals=alpha2_nvals, rng=rng)
    else:
        alpha2_ci = (np.nan, np.nan)

    # Classification
    # NOTE: Thresholds (<0.75, >1.5) are from RR-interval literature.
    # Not validated for RMSSD-epoch DFA — interpret with caution.
    if alpha1 < 0.75:
        a1_class = "reduced_correlation"
        a1_text = "Reduced short-term correlation (RMSSD-epoke-proxy; RR-intervall-terskel <0.75 — ikke validert for denne metoden)"
    elif alpha1 > 1.5:
        a1_class = "excessive_correlation"
        a1_text = "Excessive correlation (RMSSD-epoke-proxy; RR-intervall-terskel >1.5 — ikke validert for denne metoden)"
    else:
        a1_class = "normal"
        a1_text = "Within normal range (RMSSD-epoke-proxy; RR-intervall-terskel ~1.0 — ikke validert for denne metoden)"

    return {
        "alpha1": round(alpha1, 4),
        "alpha2": round(alpha2, 4),
        "alpha_full": round(alpha_full, 4),
        "alpha1_ci_95": alpha1_ci,
        "alpha2_ci_95": alpha2_ci,
        "alpha1_classification": a1_class,
        "alpha1_interpretation": a1_text,
        "reference_healthy": "~1.0 (RR-interval literature; RMSSD-epoch proxy may differ)",
        "reference_disease": "<0.75 or >1.5 (RR-interval literature; RMSSD-epoch proxy may differ)",
        "method_note": "RMSSD-Epoch DFA (Proxy) — not beat-to-beat RR-interval DFA",
    }


# ===========================================================================
# SECTION 5: Approximate Entropy (ApEn) and Sample Entropy (SampEn)
# ===========================================================================
def compute_entropy_measures(
    rmssd: np.ndarray, m: int = 2, r_factor: float = 0.2
) -> dict:
    """Compute ApEn and SampEn with m=2, r=0.2*SD.

    Richman & Moorman 2000: SampEn is less biased than ApEn.
    Low values = reduced complexity = pathological.
    Healthy RMSSD SampEn: ~1.5-2.5. Post-HSCT: often < 1.0.

    NOTE: Reference ranges are from RR-interval studies; RMSSD-epoch SampEn
    values may not be directly comparable.
    """
    if not HAS_ANTROPY:
        return {
            "apen": None,
            "sampen": None,
            "m": m,
            "r": 0,
            "r_factor": r_factor,
            "classification": "unknown",
            "interpretation": "antropy not installed",
            "reference_healthy": "SampEn ~1.5-2.5",
            "reference_post_hsct": "Often < 1.0",
        }

    r = r_factor * np.std(rmssd, ddof=1)

    # Subsample for ApEn performance (it's O(N^2))
    data_sub = rmssd[:2000] if len(rmssd) > 2000 else rmssd

    try:
        apen = float(antropy.app_entropy(data_sub, order=m))
    except Exception:
        apen = float("nan")

    try:
        sampen_arr = antropy.sample_entropy(data_sub, order=m, metric="chebyshev")
        sampen = (
            float(sampen_arr[-1])
            if hasattr(sampen_arr, "__len__")
            else float(sampen_arr)
        )
    except Exception:
        sampen = float("nan")

    # Classification
    if not np.isnan(sampen):
        if sampen < 0.5:
            classification = "severely_reduced"
            text = "Severely reduced entropy - highly regular, low complexity"
        elif sampen < 1.0:
            classification = "moderately_reduced"
            text = "Moderately reduced entropy"
        elif sampen < 1.5:
            classification = "mildly_reduced"
            text = "Mildly reduced entropy"
        else:
            classification = "normal"
            text = "Normal complexity"
    else:
        classification = "unknown"
        text = "Could not be computed"

    return {
        "apen": round(apen, 4) if not np.isnan(apen) else None,
        "sampen": round(sampen, 4) if not np.isnan(sampen) else None,
        "m": m,
        "r": round(r, 4),
        "r_factor": r_factor,
        "classification": classification,
        "interpretation": text,
        "reference_healthy": "SampEn ~1.5-2.5 (RR-interval literature; RMSSD-epoch SampEn values may not be directly comparable)",
        "reference_post_hsct": "Often < 1.0",
    }


# ===========================================================================
# SECTION 6: Hjorth Parameters
# ===========================================================================
def compute_hjorth(rmssd: np.ndarray) -> dict:
    """Compute Hjorth Activity, Mobility, Complexity.

    Activity: variance of signal (power).
    Mobility: sqrt(var(d/dt) / var(signal)) - mean frequency.
    Complexity: mobility of first derivative / mobility of signal.

    Low mobility + low complexity = monotonous signal = autonomic failure.
    """
    if not HAS_ANTROPY:
        return {
            "activity": np.nan,
            "mobility": np.nan,
            "complexity": np.nan,
            "note": "antropy not installed",
            "interpretation": "Hjorth parameters cannot be computed",
        }

    try:
        params = antropy.hjorth_params(rmssd)
        if hasattr(params, "__len__") and len(params) == 3:
            activity, mobility, complexity = params
        elif hasattr(params, "__len__") and len(params) == 2:
            mobility, complexity = params
            activity = float(np.var(rmssd, ddof=1))
        else:
            raise ValueError(f"Unexpected hjorth_params output: {params!r}")
    except Exception as e:
        logging.warning(f"Hjorth params failed: {e}")
        return {
            "activity": np.nan,
            "mobility": np.nan,
            "complexity": np.nan,
            "note": "Computation failed",
        }

    return {
        "activity": round(float(activity), 4),
        "mobility": round(float(mobility), 4),
        "complexity": round(float(complexity), 4),
        "interpretation": (
            f"Activity (variance): {activity:.2f} ms^2. "
            f"Mobility: {mobility:.4f} (low = monotonous signal). "
            f"Complexity: {complexity:.4f}."
        ),
    }


# ===========================================================================
# SECTION 7: Baevsky Stress Index (SI)
# ===========================================================================
def compute_baevsky_si(rmssd: np.ndarray) -> dict:
    """Compute Baevsky Stress Index from RMSSD distribution.

    SI = AMo / (2 * Mo * MxDMn)
    where:
      AMo = mode amplitude (% of values in mode bin)
      Mo  = mode value (ms)
      MxDMn = max - min (ms)

    Baevsky 2002 (Russian space medicine):
      Normal: 50-150
      Moderate stress: 150-500
      High stress: > 500
      Pathological: > 1000

    Note: Classically uses RR intervals, here adapted for RMSSD epochs.
    """
    if len(rmssd) < 10:
        return {"si": 0, "note": "Insufficient data"}

    # Mode (most frequent value, using 0.5ms bins for RMSSD)
    bin_width = 0.5  # ms
    bins = np.arange(
        np.min(rmssd) - bin_width, np.max(rmssd) + bin_width * 2, bin_width
    )
    hist, edges = np.histogram(rmssd, bins=bins)

    mode_idx = np.argmax(hist)
    mo = (edges[mode_idx] + edges[mode_idx + 1]) / 2.0  # Mode value
    amo = safe_divide(hist[mode_idx], len(rmssd)) * 100  # Mode amplitude (%)
    mxdmn = np.max(rmssd) - np.min(rmssd)  # Range

    if mo > 0 and mxdmn > 0:
        si = amo / (2.0 * mo * mxdmn)
    else:
        si = 0.0

    # Scale to conventional units (multiply by 1000 for RR-equivalent range)
    # Since RMSSD values are much smaller than RR intervals, adjust scaling
    si_scaled = si * 1000  # Approximate scaling for interpretation

    if si_scaled < 150:
        classification = "normal"
        text = "Normal stress level"
    elif si_scaled < 500:
        classification = "moderate_stress"
        text = "Moderately elevated stress level"
    elif si_scaled < 1000:
        classification = "high_stress"
        text = "High stress load"
    else:
        classification = "pathological"
        text = "Pathologically elevated stress index"

    return {
        "si_raw": round(si, 4),
        "si_scaled": round(si_scaled, 2),
        "amo_pct": round(amo, 2),
        "mode_ms": round(mo, 2),
        "range_ms": round(mxdmn, 2),
        "classification": classification,
        "interpretation": text,
        "reference": "Normal < 150, Moderate 150-500, High > 500, Pathological > 1000",
    }


# ===========================================================================
# SECTION 8: Toichi CVI/CSI
# ===========================================================================
def compute_toichi(rmssd: np.ndarray) -> dict:
    """Compute Toichi Cardiac Vagal Index (CVI) and Cardiac Sympathetic Index (CSI).

    From Poincare decomposition:
      SD1 = std(diff) / sqrt(2)  [short-term, vagal]
      SD2 = std(sum) / sqrt(2)   [long-term, sympathetic]
      CVI = log10(SD1 * SD2)     [vagal tone]
      CSI = SD2 / SD1            [sympathovagal balance]

    Toichi et al. 1997:
      CVI reflects parasympathetic (vagal) modulation.
      CSI reflects sympathetic modulation.
      Low CVI = reduced vagal tone. High CSI = sympathetic dominance.
    """
    if len(rmssd) < 10:
        return {"cvi": 0, "csi": 0, "sd1": 0, "sd2": 0, "note": "Insufficient data"}

    # Consecutive pairs
    x_n = rmssd[:-1]
    x_n1 = rmssd[1:]

    diff = x_n1 - x_n
    summ = x_n1 + x_n

    sd1 = np.std(diff, ddof=1) / np.sqrt(2)
    sd2 = np.std(summ, ddof=1) / np.sqrt(2)

    # CVI and CSI
    if sd1 > 0 and sd2 > 0:
        cvi = np.log10(sd1 * sd2)
        csi = sd2 / sd1
    else:
        cvi = 0.0
        csi = 0.0

    # Reference values for healthy adult male
    sd1_healthy = 30.0  # ms
    sd2_healthy = 60.0  # ms
    cvi_healthy = np.log10(sd1_healthy * sd2_healthy)  # ~3.26
    csi_healthy = sd2_healthy / sd1_healthy  # ~2.0

    return {
        "sd1": round(sd1, 3),
        "sd2": round(sd2, 3),
        "sd1_sd2_ratio": round(sd1 / sd2, 4) if sd2 > 0 else 0,
        "cvi": round(cvi, 4),
        "csi": round(csi, 4),
        "cvi_healthy_ref": round(cvi_healthy, 3),
        "csi_healthy_ref": round(csi_healthy, 2),
        "sd1_pct_of_normal": round(100 * sd1 / sd1_healthy, 1),
        "sd2_pct_of_normal": round(100 * sd2 / sd2_healthy, 1),
        "interpretation": (
            f"CVI={cvi:.2f} (healthy ref: {cvi_healthy:.2f}) - "
            f"{'severely reduced' if cvi < cvi_healthy - 1 else 'moderately reduced' if cvi < cvi_healthy - 0.5 else 'normal'} vagal tone. "
            f"CSI={csi:.2f} (healthy ref: {csi_healthy:.1f}) - "
            f"{'sympathetic dominance' if csi > csi_healthy * 1.5 else 'normal balance'}."
        ),
    }


# ===========================================================================
# SECTION 9: HR Circadian Analysis (Cosinor)
# ===========================================================================
def compute_cosinor(hr_timestamps: list[datetime], hr_bpm: np.ndarray) -> dict:
    """Cosinor analysis: fit 24h cosine to HR for MESOR, amplitude, acrophase.

    y(t) = MESOR + A*cos(2*pi*t/24 - phi)

    Nelson et al. 1979.
    Low amplitude = flattened circadian rhythm = circadian disruption.
    Shifted acrophase = chronodisruption.
    """
    if len(hr_bpm) < 48:
        return {
            "mesor": 0,
            "amplitude": 0,
            "acrophase_hours": 0,
            "note": "Insufficient data for cosinor analysis",
        }

    # Convert timestamps to decimal hours from midnight of first day
    day0 = hr_timestamps[0].date()
    t_hours = np.array(
        [
            (ts - datetime.combine(day0, datetime.min.time())).total_seconds() / 3600.0
            for ts in hr_timestamps
        ]
    )

    # Cosinor model: y = M + A*cos(2*pi*t/P - phi)
    # Linearized: y = M + beta*cos(2*pi*t/P) + gamma*sin(2*pi*t/P)
    # where A = sqrt(beta^2 + gamma^2), phi = arctan2(gamma, beta)
    period = 24.0  # hours
    omega = 2 * np.pi / period

    cos_t = np.cos(omega * t_hours)
    sin_t = np.sin(omega * t_hours)

    # OLS fit
    X = np.column_stack([np.ones(len(t_hours)), cos_t, sin_t])
    try:
        params, residuals, rank, sv = np.linalg.lstsq(X, hr_bpm, rcond=None)
        mesor = params[0]
        beta = params[1]
        gamma = params[2]
        amplitude = np.sqrt(beta**2 + gamma**2)
        acrophase_rad = np.arctan2(gamma, beta)
        acrophase_hours = (-acrophase_rad * 12 / np.pi) % 24  # Convert to hours

        # R-squared
        y_pred = X @ params
        ss_res = np.sum((hr_bpm - y_pred) ** 2)
        ss_tot = np.sum((hr_bpm - np.mean(hr_bpm)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    except Exception:
        return {
            "mesor": float(np.mean(hr_bpm)),
            "amplitude": 0,
            "acrophase_hours": 0,
            "r_squared": 0,
            "note": "Cosinor fitting failed",
        }

    # Generate fitted curve for plotting (one full 24h cycle)
    t_plot = np.linspace(0, 24, 200)
    y_plot = mesor + amplitude * np.cos(omega * t_plot - (-acrophase_rad))

    # Clinical interpretation
    # Normal HR amplitude: 15-25 bpm. Low amplitude (<10): circadian disruption.
    if amplitude < 5:
        rhythm_status = "severely_flattened"
        rhythm_text = "Severely flattened circadian rhythm - serious chronodisruption"
    elif amplitude < 10:
        rhythm_status = "flattened"
        rhythm_text = "Flattened circadian rhythm"
    elif amplitude < 15:
        rhythm_status = "mildly_reduced"
        rhythm_text = "Mildly reduced circadian rhythm amplitude"
    else:
        rhythm_status = "normal"
        rhythm_text = "Normal circadian rhythm"

    return {
        "mesor": round(mesor, 2),
        "amplitude": round(amplitude, 2),
        "acrophase_hours": round(acrophase_hours, 2),
        "acrophase_hhmm": f"{int(acrophase_hours):02d}:{int((acrophase_hours % 1) * 60):02d}",
        "r_squared": round(r_squared, 4),
        "rhythm_status": rhythm_status,
        "interpretation": rhythm_text,
        "t_plot": t_plot.tolist(),
        "y_plot": y_plot.tolist(),
        "reference_amplitude": "15-25 bpm (healthy adult)",
        "reference_acrophase": "14:00-17:00 (normal)",
    }


# ===========================================================================
# SECTION 10: HR Complexity (Permutation & Spectral Entropy)
# ===========================================================================
def compute_hr_complexity(hr_bpm: np.ndarray) -> dict:
    """Compute permutation entropy and spectral entropy on HR time series.

    Permutation entropy (Bandt & Pompe 2002): Measures ordinal pattern regularity.
    Spectral entropy (Inouye et al. 1991): Flatness of power spectrum.
    Both normalized 0-1. Low values = reduced complexity.
    """
    if not HAS_ANTROPY:
        return {
            "permutation_entropy": None,
            "spectral_entropy": None,
            "pe_classification": "unknown",
            "pe_interpretation": "antropy not installed",
            "reference": "Healthy: PE > 0.85, SE > 0.80",
        }

    # Subsample for performance
    data = hr_bpm[:5000] if len(hr_bpm) > 5000 else hr_bpm.copy()

    try:
        perm_ent = float(antropy.perm_entropy(data, order=3, normalize=True))
    except Exception:
        perm_ent = float("nan")

    try:
        spec_ent = float(
            antropy.spectral_entropy(data, sf=1.0, method="welch", normalize=True)
        )
    except Exception:
        spec_ent = float("nan")

    # Classification
    if not np.isnan(perm_ent):
        if perm_ent < 0.6:
            pe_class = "severely_reduced"
            pe_text = "Severely reduced permutation entropy"
        elif perm_ent < 0.8:
            pe_class = "moderately_reduced"
            pe_text = "Moderately reduced complexity"
        else:
            pe_class = "normal"
            pe_text = "Normal ordinal pattern complexity"
    else:
        pe_class = "unknown"
        pe_text = "Could not be computed"

    return {
        "permutation_entropy": round(perm_ent, 4) if not np.isnan(perm_ent) else None,
        "spectral_entropy": round(spec_ent, 4) if not np.isnan(spec_ent) else None,
        "pe_classification": pe_class,
        "pe_interpretation": pe_text,
        "reference": "Healthy: PE > 0.85, SE > 0.80",
    }


# ===========================================================================
# SECTION 11: Night-to-Night HR Variability
# ===========================================================================
def compute_nightly_hr_variability(sleep_periods: pd.DataFrame) -> dict:
    """Compute CV of nightly HR averages as a prognostic marker.

    High night-to-night variability with high baseline HR = poor autonomic reserve.
    CV > 10%: clinically significant instability.
    """
    if sleep_periods.empty or "average_heart_rate" not in sleep_periods.columns:
        return {"cv_pct": 0, "note": "No nocturnal HR data"}

    nightly_hr = sleep_periods["average_heart_rate"].dropna().values
    if len(nightly_hr) < 5:
        return {"cv_pct": 0, "n_nights": len(nightly_hr), "note": "Too few nights"}

    mean_hr = np.mean(nightly_hr)
    std_hr = np.std(nightly_hr, ddof=1)
    cv = (std_hr / mean_hr) * 100 if mean_hr > 0 else 0

    # Trend analysis
    nights = np.arange(len(nightly_hr))
    if len(nightly_hr) >= 5:
        slope = np.polyfit(nights, nightly_hr, 1)[0]
        trend_text = f"{'rising' if slope > 0.1 else 'declining' if slope < -0.1 else 'stable'} trend ({slope:+.2f} bpm/night)"
    else:
        slope = 0
        trend_text = "Insufficient for trend analysis"

    if cv > 10:
        classification = "high_variability"
        text = "High night-to-night variability - unstable autonomic regulation"
    elif cv > 5:
        classification = "moderate_variability"
        text = "Moderate night-to-night variability"
    else:
        classification = "stable"
        text = "Stable nocturnal heart rate"

    return {
        "n_nights": len(nightly_hr),
        "mean_hr": round(mean_hr, 1),
        "std_hr": round(std_hr, 2),
        "cv_pct": round(cv, 2),
        "min_hr": round(float(np.min(nightly_hr)), 1),
        "max_hr": round(float(np.max(nightly_hr)), 1),
        "slope_bpm_per_night": round(slope, 3),
        "trend": trend_text,
        "classification": classification,
        "interpretation": text,
        "nightly_values": nightly_hr.tolist(),
    }


# ===========================================================================
# SECTION 12: Wearable Allostatic Load Score
# ===========================================================================
def compute_allostatic_load(conn: sqlite3.Connection) -> dict:
    """Compute Wearable Allostatic Load Score (0-7).

    Count biomarkers exceeding clinically significant thresholds:
      1. HR > 90th percentile for age (>90 bpm resting)
      2. HRV < 10th percentile for age (RMSSD < 15 ms)
      3. Sleep efficiency < 85%
      4. Temperature deviation > 0.5 °C
      5. SpO2 < 95%
      6. Deep sleep < 10% of total
      7. REM sleep < 15% of total

    Higher score = higher allostatic load = worse adaptation.
    """
    score = 0
    details = {}

    # 1. HR > IST threshold (average sleep HR)
    rows = conn.execute(
        "SELECT average_heart_rate FROM oura_sleep_periods WHERE average_heart_rate IS NOT NULL"
    ).fetchall()
    if rows:
        avg_hr = np.mean([r[0] for r in rows])
        hr_flag = avg_hr > NOCTURNAL_HR_ELEVATED
        score += int(hr_flag)
        details["hr_avg_sleep"] = {
            "value": round(avg_hr, 1),
            "threshold": NOCTURNAL_HR_ELEVATED,
            "exceeded": hr_flag,
            "unit": "bpm",
        }

    # 2. HRV < clinical deficiency threshold (ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017)
    rows = conn.execute(
        "SELECT AVG(rmssd) FROM oura_hrv WHERE rmssd IS NOT NULL"
    ).fetchall()
    if rows and rows[0][0] is not None:
        avg_rmssd = rows[0][0]
        hrv_flag = avg_rmssd < ESC_RMSSD_DEFICIENCY
        score += int(hrv_flag)
        details["hrv_rmssd"] = {
            "value": round(avg_rmssd, 1),
            "threshold": ESC_RMSSD_DEFICIENCY,
            "exceeded": hrv_flag,
            "unit": "ms",
        }

    # 3. Sleep efficiency < 85%
    rows = conn.execute(
        "SELECT efficiency FROM oura_sleep_periods WHERE efficiency IS NOT NULL"
    ).fetchall()
    if rows:
        avg_eff = np.mean([r[0] for r in rows])
        eff_flag = avg_eff < 85
        score += int(eff_flag)
        details["sleep_efficiency"] = {
            "value": round(avg_eff, 1),
            "threshold": 85,
            "exceeded": eff_flag,
            "unit": "%",
        }

    # 4. Temperature deviation > 0.5 °C
    rows = conn.execute(
        "SELECT temperature_deviation FROM oura_readiness WHERE temperature_deviation IS NOT NULL"
    ).fetchall()
    if rows:
        avg_temp_dev = np.mean([abs(r[0]) for r in rows])
        temp_flag = avg_temp_dev > 0.5
        score += int(temp_flag)
        details["temp_deviation"] = {
            "value": round(avg_temp_dev, 3),
            "threshold": 0.5,
            "exceeded": temp_flag,
            "unit": "°C",
        }

    # 5. SpO2 < 95%
    rows = conn.execute(
        "SELECT spo2_average FROM oura_spo2 WHERE spo2_average > 0"
    ).fetchall()
    if rows:
        avg_spo2 = np.mean([r[0] for r in rows])
        spo2_flag = avg_spo2 < 95
        score += int(spo2_flag)
        details["spo2"] = {
            "value": round(avg_spo2, 2),
            "threshold": 95,
            "exceeded": spo2_flag,
            "unit": "%",
        }

    # 6. Deep sleep < 10% of total
    rows = conn.execute(
        "SELECT deep_sleep_duration, total_sleep_duration FROM oura_sleep_periods "
        "WHERE deep_sleep_duration IS NOT NULL AND total_sleep_duration > 0"
    ).fetchall()
    if rows:
        deep_pcts = [r[0] / r[1] * 100 for r in rows if r[1] > 0]
        avg_deep = np.mean(deep_pcts) if deep_pcts else 0
        deep_flag = avg_deep < 10
        score += int(deep_flag)
        details["deep_sleep_pct"] = {
            "value": round(avg_deep, 1),
            "threshold": 10,
            "exceeded": deep_flag,
            "unit": "%",
        }

    # 7. REM sleep < 15% of total
    rows = conn.execute(
        "SELECT rem_sleep_duration, total_sleep_duration FROM oura_sleep_periods "
        "WHERE rem_sleep_duration IS NOT NULL AND total_sleep_duration > 0"
    ).fetchall()
    if rows:
        rem_pcts = [r[0] / r[1] * 100 for r in rows if r[1] > 0]
        avg_rem = np.mean(rem_pcts) if rem_pcts else 0
        rem_flag = avg_rem < 15
        score += int(rem_flag)
        details["rem_sleep_pct"] = {
            "value": round(avg_rem, 1),
            "threshold": 15,
            "exceeded": rem_flag,
            "unit": "%",
        }

    # Severity
    if score >= 5:
        severity = "severe"
        text = f"Severe allostatic load ({score}/7 biomarkers exceeded)"
    elif score >= 3:
        severity = "moderate"
        text = f"Moderate allostatic load ({score}/7)"
    elif score >= 1:
        severity = "mild"
        text = f"Mild allostatic load ({score}/7)"
    else:
        severity = "normal"
        text = "No biomarkers exceeded"

    return {
        "score": score,
        "max_score": 7,
        "severity": severity,
        "interpretation": text,
        "details": details,
    }


# ===========================================================================
# HTML REPORT GENERATION
# ===========================================================================
def generate_html_report(metrics: dict, figures: dict) -> str:
    """Generate dark-themed interactive HTML report with all analyses."""

    # Convert plotly figures to HTML divs (no plotlyjs, not full pages)
    fig_htmls = {}
    for key, fig in figures.items():
        fig_htmls[key] = fig.to_html(full_html=False, include_plotlyjs=False)

    # Extract key metrics for summary
    dfa = metrics.get("dfa", {})
    entropy = metrics.get("entropy", {})
    toichi = metrics.get("toichi", {})
    baevsky = metrics.get("baevsky", {})
    cosinor = metrics.get("cosinor", {})
    allostatic = metrics.get("allostatic_load", {})
    rqa = metrics.get("rqa", {})
    mse = metrics.get("mse", {})
    hjorth = metrics.get("hjorth", {})
    hr_complex = metrics.get("hr_complexity", {})
    nightly = metrics.get("nightly_hr_variability", {})
    freq = metrics.get("frequency_domain", {})

    # --- Extra CSS for script-specific components ---
    extra_css = f"""
.allostatic-bar {{
    height: 24px; border-radius: 12px; background: var(--bg-elevated, {BG_SURFACE});
    position: relative; margin: 8px 0;
    border: 1px solid var(--border-subtle);
}}
.allostatic-fill {{
    height: 100%; border-radius: 12px;
    background: linear-gradient(90deg, {ACCENT_GREEN}, {ACCENT_AMBER}, {ACCENT_RED});
}}
.badge {{
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 0.75rem; font-weight: 600;
}}
.badge-critical {{ background: rgba(239,68,68,0.15); color: {ACCENT_RED}; }}
.badge-warning {{ background: rgba(245,158,11,0.15); color: {ACCENT_AMBER}; }}
.badge-ok {{ background: rgba(16,185,129,0.15); color: {ACCENT_GREEN}; }}
.interpretation {{
    background: rgba(245,158,11,0.08); border-left: 3px solid {ACCENT_AMBER};
    padding: 12px 16px; margin: 12px 0; border-radius: 0 8px 8px 0;
    font-size: 0.875rem; color: var(--text-primary);
}}
.interpretation.critical {{
    background: rgba(239,68,68,0.08); border-left-color: {ACCENT_RED};
}}
.interpretation.info {{
    background: rgba(59,130,246,0.08); border-left-color: {ACCENT_BLUE};
}}
.interpretation ul {{ margin-top: 8px; padding-left: 20px; }}
.interpretation li {{ margin-bottom: 4px; }}
.interpretation em {{ color: var(--text-secondary); }}
"""

    # --- KPI Cards (Executive Summary) ---
    def _dfa_status() -> str:
        a1 = dfa.get("alpha1", 1.0)
        try:
            if float(a1) < 0.75 or float(a1) > 1.5:
                return "critical"
        except (TypeError, ValueError):
            pass
        return "warning"

    def _entropy_status() -> str:
        c = entropy.get("classification", "")
        if c in ("severely_reduced", "moderately_reduced"):
            return "critical"
        return "warning"

    def _cvi_status() -> str:
        try:
            if float(toichi.get("cvi", 99)) < 2.5:
                return "critical"
        except (TypeError, ValueError):
            pass
        return "warning"

    def _allo_status() -> str:
        s = allostatic.get("score", 0)
        return "critical" if s >= 5 else "warning" if s >= 3 else "normal"

    def _baevsky_status() -> str:
        si = baevsky.get("si_scaled", 0)
        return "critical" if si > 500 else "warning" if si > 150 else "normal"

    dfa_st = _dfa_status()
    dfa_lbl = "Abnormal" if dfa_st in ("critical", "warning") else ""
    ent_st = _entropy_status()
    ent_lbl = "Low" if ent_st in ("critical", "warning") else ""
    cvi_st = _cvi_status()
    cvi_lbl = "Low" if cvi_st in ("critical", "warning") else ""
    allo_st = _allo_status()
    allo_lbl = "Elevated" if allo_st in ("critical", "warning") else ""
    baev_st = _baevsky_status()
    baev_lbl = "Elevated" if baev_st in ("critical", "warning") else ""
    cosinor_st = "warning" if cosinor.get("amplitude", 99) < 10 else "info"
    cosinor_lbl = "Low" if cosinor_st in ("critical", "warning") else ""
    rqa_st = "critical" if rqa.get("determinism", 1) < 0.6 else "normal"
    rqa_lbl = "Low" if rqa_st in ("critical", "warning") else ""
    nightly_st = "warning" if nightly.get("cv_pct", 0) > 10 else "normal"
    nightly_lbl = "Elevated" if nightly_st in ("critical", "warning") else ""

    kpi_row = make_kpi_row(
        make_kpi_card(
            "DFA alpha-1 (RMSSD-Epoch Proxy)",
            dfa.get("alpha1", "N/A"),
            "",
            status=dfa_st,
            decimals=4,
            status_label=dfa_lbl,
            detail=f"Applied to RMSSD epochs (not RR intervals) — ref ~1.0 (not directly comparable) | 95% KI: {dfa.get('alpha1_ci_95', ('?', '?'))}",
        ),
        make_kpi_card(
            "Sample Entropy",
            entropy.get("sampen", "N/A"),
            "",
            status=ent_st,
            decimals=4,
            status_label=ent_lbl,
            detail=f"Healthy: 1.5-2.5 | {entropy.get('classification', '')}",
        ),
        make_kpi_card(
            "Toichi CVI (vagal)",
            toichi.get("cvi", "N/A"),
            "",
            status=cvi_st,
            decimals=4,
            status_label=cvi_lbl,
            detail=f"Healthy: {toichi.get('cvi_healthy_ref', '~3.26')}",
        ),
        make_kpi_card(
            "Allostatic Load",
            f"{allostatic.get('score', 0)}/7",
            "",
            status=allo_st,
            status_label=allo_lbl,
            detail=allostatic.get("severity", ""),
        ),
        make_kpi_card(
            "Baevsky SI",
            baevsky.get("si_scaled", "N/A"),
            "",
            status=baev_st,
            decimals=2,
            status_label=baev_lbl,
            detail=f"Normal: <150 | {baevsky.get('classification', '')}",
        ),
        make_kpi_card(
            "Cosinor Amplitude",
            cosinor.get("amplitude", "N/A"),
            "bpm",
            status=cosinor_st,
            decimals=2,
            status_label=cosinor_lbl,
            detail=f"Healthy: 15-25 bpm | Acrophase: {cosinor.get('acrophase_hhmm', '?')}",
        ),
        make_kpi_card(
            "RQA Determinism",
            rqa.get("determinism", "N/A"),
            "",
            status=rqa_st,
            decimals=4,
            status_label=rqa_lbl,
            detail=f"Healthy: >0.6 | {rqa.get('classification', '')}",
        ),
        make_kpi_card(
            "Nightly HR CV",
            nightly.get("cv_pct", "N/A"),
            "%",
            status=nightly_st,
            decimals=2,
            status_label=nightly_lbl,
            detail=f"Mean: {nightly.get('mean_hr', '?')} bpm | {nightly.get('classification', '')}",
        ),
    )

    # --- Sections ---
    body = kpi_row

    # Section 1: Frequency Domain
    sec1_content = f"""
    <div class="interpretation info">
        <strong>Note:</strong> Analysis based on 5-minute RMSSD epochs (not beat-to-beat RR intervals).
        Nyquist frequency is limited to ~0.00167 Hz, so the table below uses proxy bands across the
        accessible spectrum only. These labels are heuristic and do not support standard LF/HF physiology claims.
    </div>
    {fig_htmls.get("frequency_domain", "")}
    <table>
        <tr><th>Band</th><th>Power</th><th>Proportion</th><th>Interpretation</th></tr>
        <tr><td>Lower proxy band</td><td>{freq.get("vlf_power", "N/A")}</td><td>{freq.get("vlf_pct", "?")}%</td><td>Lowest third of accessible RMSSD-epoch spectrum</td></tr>
        <tr><td>Mid proxy band</td><td>{freq.get("lf_power", "N/A")}</td><td>{freq.get("lf_pct", "?")}%</td><td>Middle third of accessible RMSSD-epoch spectrum</td></tr>
        <tr><td>Upper proxy band</td><td>{freq.get("hf_power", "N/A")}</td><td>{freq.get("hf_pct", "?")}%</td><td>Upper third of accessible RMSSD-epoch spectrum</td></tr>
        <tr><td><strong>Upper/lower proxy ratio</strong></td><td colspan="2"><strong>{freq.get("lf_hf_ratio", "N/A")}</strong></td><td>Heuristic spectral balance only; not a standard sympathovagal index</td></tr>
    </table>"""
    body += make_section("1. Proxy Frequency Spectrum (Lomb-Scargle)", sec1_content)

    # Section 2: MSE
    sec2_content = f"""
    {fig_htmls.get("mse", "")}
    <div class="interpretation critical">
        <strong>Pattern:</strong> {mse.get("interpretation", "Unknown")}
    </div>
    <p>Low-scale mean: {mse.get("low_scale_mean", "?")} | High-scale mean: {mse.get("high_scale_mean", "?")}</p>
    <p><em>Costa et al. 2005: Healthy systems show high entropy across all scales. Neuropathic disease shows decline at high scales.</em></p>"""
    body += make_section("2. Multiscale Entropy (MSE)", sec2_content)

    # Section 3: RQA
    sec3_content = f"""
    {fig_htmls.get("rqa", "")}
    <table>
        <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
        <tr><td>Recurrence Rate (RR)</td><td>{rqa.get("recurrence_rate", "N/A")}</td><td>Proportion of recurrent states</td></tr>
        <tr><td>Determinism (DET)</td><td>{rqa.get("determinism", "N/A")}</td><td>{"Reduced" if rqa.get("determinism", 1) < 0.6 else "Preserved"} predictability</td></tr>
        <tr><td>Laminarity (LAM)</td><td>{rqa.get("laminarity", "N/A")}</td><td>Degree of intermittency</td></tr>
        <tr><td>Diagonal Entropy</td><td>{rqa.get("diagonal_entropy", "N/A")}</td><td>Complexity of diagonal lines</td></tr>
    </table>
    <div class="interpretation {"critical" if rqa.get("determinism", 1) < 0.6 else ""}">
        {rqa.get("interpretation", "")}
    </div>"""
    body += make_section("3. Recurrence Quantification Analysis (RQA)", sec3_content)

    # Section 4: DFA
    sec4_content = f"""
    <div class="interpretation info">
        <strong>Note:</strong> DFA is originally designed for beat-to-beat RR-interval time series.
        Applied here to 5-minute RMSSD epochs as a proxy. Reference values from RR-interval studies;
        RMSSD-epoch DFA values are not directly comparable.
    </div>
    {fig_htmls.get("dfa", "")}
    <table>
        <tr><th>Parameter</th><th>Value</th><th>95% CI</th><th>Reference (RR-interval literature)</th></tr>
        <tr><td>alpha-1 (short-term)</td><td><strong>{dfa.get("alpha1", "N/A")}</strong></td><td>{dfa.get("alpha1_ci_95", "")}</td><td>~1.0 (RR-interval ref; proxy may differ)</td></tr>
        <tr><td>alpha-2 (long-term)</td><td>{dfa.get("alpha2", "N/A")}</td><td>{dfa.get("alpha2_ci_95", "")}</td><td>~1.0 (RR-interval ref; proxy may differ)</td></tr>
        <tr><td>alpha (full)</td><td>{dfa.get("alpha_full", "N/A")}</td><td>-</td><td>~1.0 (RR-interval ref)</td></tr>
    </table>
    <div class="interpretation {"critical" if dfa.get("alpha1", 1) < 0.75 or dfa.get("alpha1", 1) > 1.5 else ""}">
        {dfa.get("alpha1_interpretation", "")}
    </div>"""
    body += make_section("4. RMSSD-Epoch DFA (Proxy)", sec4_content)

    # Section 5: Entropy
    sec5_content = f"""
    {fig_htmls.get("entropy", "")}
    <table>
        <tr><th>Metric</th><th>Value</th><th>Parameters</th><th>Healthy Reference</th></tr>
        <tr><td>ApEn</td><td>{entropy.get("apen", "N/A")}</td><td>m={entropy.get("m", 2)}, r={entropy.get("r", "?")}</td><td>~1.0-1.5</td></tr>
        <tr><td>SampEn</td><td><strong>{entropy.get("sampen", "N/A")}</strong></td><td>m={entropy.get("m", 2)}, r={entropy.get("r", "?")}</td><td>~1.5-2.5</td></tr>
    </table>
    <div class="interpretation {"critical" if entropy.get("classification", "") in ("severely_reduced", "moderately_reduced") else ""}">
        {entropy.get("interpretation", "")}
    </div>"""
    body += make_section(
        "5. Approximate Entropy (ApEn) and Sample Entropy (SampEn)", sec5_content
    )

    # Section 6: Hjorth
    sec6_content = f"""
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
        <tr><td>Activity</td><td>{_fmt_nan(hjorth.get("activity"), ".4f")} ms<sup>2</sup></td><td>Signal variance (power)</td></tr>
        <tr><td>Mobility</td><td>{_fmt_nan(hjorth.get("mobility"), ".4f")}</td><td>Mean frequency (low = monotonous signal)</td></tr>
        <tr><td>Complexity</td><td>{_fmt_nan(hjorth.get("complexity"), ".4f")}</td><td>Rate of frequency change</td></tr>
    </table>"""
    body += make_section("6. Hjorth Parameters", sec6_content)

    # Section 7: Baevsky SI
    sec7_content = f"""
    <table>
        <tr><th>Component</th><th>Value</th></tr>
        <tr><td>AMo (mode amplitude)</td><td>{baevsky.get("amo_pct", "N/A")}%</td></tr>
        <tr><td>Mo (mode)</td><td>{baevsky.get("mode_ms", "N/A")} ms</td></tr>
        <tr><td>MxDMn (range)</td><td>{baevsky.get("range_ms", "N/A")} ms</td></tr>
        <tr><td><strong>SI (scaled)</strong></td><td><strong>{baevsky.get("si_scaled", "N/A")}</strong></td></tr>
    </table>
    <div class="interpretation {"critical" if baevsky.get("si_scaled", 0) > 500 else "warning" if baevsky.get("si_scaled", 0) > 150 else ""}">
        {baevsky.get("interpretation", "")} <br>
        <em>Reference (Baevsky 2002): Normal &lt;150, Moderate 150-500, High &gt;500, Pathological &gt;1000</em>
    </div>"""
    body += make_section("7. Baevsky Stress Index (SI)", sec7_content)

    # Section 8: Toichi
    sec8_content = f"""
    {fig_htmls.get("poincare", "")}
    <table>
        <tr><th>Parameter</th><th>Patient</th><th>Healthy Reference</th><th>% of Normal</th></tr>
        <tr><td>SD1 (vagal)</td><td>{toichi.get("sd1", "N/A")} ms</td><td>30 ms</td><td>{toichi.get("sd1_pct_of_normal", "?")}%</td></tr>
        <tr><td>SD2 (sympathetic)</td><td>{toichi.get("sd2", "N/A")} ms</td><td>60 ms</td><td>{toichi.get("sd2_pct_of_normal", "?")}%</td></tr>
        <tr><td><strong>CVI</strong></td><td><strong>{toichi.get("cvi", "N/A")}</strong></td><td>{toichi.get("cvi_healthy_ref", "?")}</td><td>-</td></tr>
        <tr><td><strong>CSI</strong></td><td><strong>{toichi.get("csi", "N/A")}</strong></td><td>{toichi.get("csi_healthy_ref", "?")}</td><td>-</td></tr>
    </table>
    <div class="interpretation critical">
        {toichi.get("interpretation", "")}
    </div>"""
    body += make_section("8. Toichi CVI/CSI (Vagal/Sympathetic Index)", sec8_content)

    # Section 9: Cosinor
    sec9_content = f"""
    {fig_htmls.get("cosinor", "")}
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Reference</th></tr>
        <tr><td>MESOR</td><td>{cosinor.get("mesor", "N/A")} bpm</td><td>60-80 bpm (healthy)</td></tr>
        <tr><td>Amplitude</td><td>{cosinor.get("amplitude", "N/A")} bpm</td><td>15-25 bpm</td></tr>
        <tr><td>Acrophase</td><td>{cosinor.get("acrophase_hhmm", "N/A")}</td><td>14:00-17:00</td></tr>
        <tr><td>R<sup>2</sup></td><td>{cosinor.get("r_squared", "N/A")}</td><td>&gt;0.3 for significant rhythm</td></tr>
    </table>
    <div class="interpretation {"critical" if cosinor.get("amplitude", 99) < 10 else ""}">
        {cosinor.get("interpretation", "")}
    </div>"""
    body += make_section("9. Cosinor Circadian Analysis (Heart Rate)", sec9_content)

    # Section 10: HR Complexity
    sec10_content = f"""
    <table>
        <tr><th>Metric</th><th>Value</th><th>Healthy Reference</th></tr>
        <tr><td>Permutation Entropy (PE)</td><td>{hr_complex.get("permutation_entropy", "N/A")}</td><td>&gt;0.85</td></tr>
        <tr><td>Spectral Entropy (SE)</td><td>{hr_complex.get("spectral_entropy", "N/A")}</td><td>&gt;0.80</td></tr>
    </table>
    <div class="interpretation">
        {hr_complex.get("pe_interpretation", "")}
    </div>"""
    body += make_section(
        "10. HR Complexity (Permutation and Spectral Entropy)", sec10_content
    )

    # Section 11: Night-to-night
    sec11_content = f"""
    {fig_htmls.get("nightly", "")}
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Number of Nights</td><td>{nightly.get("n_nights", "?")}</td></tr>
        <tr><td>Mean Nocturnal HR</td><td>{nightly.get("mean_hr", "?")} bpm</td></tr>
        <tr><td>SD Nocturnal HR</td><td>{nightly.get("std_hr", "?")} bpm</td></tr>
        <tr><td>CV</td><td><strong>{nightly.get("cv_pct", "?")}%</strong></td></tr>
        <tr><td>Min-Max</td><td>{nightly.get("min_hr", "?")} - {nightly.get("max_hr", "?")} bpm</td></tr>
        <tr><td>Trend</td><td>{nightly.get("trend", "?")}</td></tr>
    </table>"""
    body += make_section("11. Night-to-Night HR Variability", sec11_content)

    # Section 12: Allostatic Load
    allo_score = allostatic.get("score", 0)
    allo_badge_cls = (
        "badge-critical"
        if allo_score >= 5
        else "badge-warning"
        if allo_score >= 3
        else "badge-ok"
    )
    sec12_content = f"""
    <div class="allostatic-bar">
        <div class="allostatic-fill" style="width: {allo_score / 7 * 100:.0f}%"></div>
    </div>
    <p style="text-align:center; font-size:1.2em; font-weight:700; margin:8px 0; color:var(--text-primary);">
        {allo_score} / 7
        <span class="badge {allo_badge_cls}">
            {allostatic.get("severity", "").upper()}
        </span>
    </p>
    <table>
        <tr><th>Biomarker</th><th>Value</th><th>Threshold</th><th>Status</th></tr>"""

    for key, detail in allostatic.get("details", {}).items():
        exceeded = detail.get("exceeded", False)
        badge = "badge-critical" if exceeded else "badge-ok"
        status = "EXCEEDED" if exceeded else "OK"
        sec12_content += f"""
        <tr>
            <td>{key.replace("_", " ").title()}</td>
            <td>{detail.get("value", "?")} {detail.get("unit", "")}</td>
            <td>{"<" if key in ("sleep_efficiency", "deep_sleep_pct", "rem_sleep_pct", "spo2") else ">"} {detail.get("threshold", "?")} {detail.get("unit", "")}</td>
            <td><span class="badge {badge}">{status}</span></td>
        </tr>"""

    sec12_content += f"""
    </table>
    <div class="interpretation {"critical" if allo_score >= 5 else ""}">
        {allostatic.get("interpretation", "")}
    </div>"""
    body += make_section("12. Wearable Allostatic Load Score", sec12_content)

    # Section 13: Clinical Summary
    summary_content = f"""
    <div class="interpretation critical">
        <strong>Key Findings:</strong><br>
        Advanced analysis of {metrics.get("n_rmssd", 0):,} RMSSD epochs and {metrics.get("n_hr", 0):,}
        heart rate measurements confirms severe autonomic dysfunction in this {PATIENT_LABEL.lower()}.
        <br><br>
        <strong>Key Metrics:</strong>
        <ul>
            <li>DFA alpha-1 (RMSSD-Epoch Proxy) = {dfa.get("alpha1", "?")} (RR-interval ref ~1.0; proxy may differ): {dfa.get("alpha1_interpretation", "")}</li>
            <li>SampEn = {entropy.get("sampen", "?")} (RR-interval ref 1.5-2.5; RMSSD-epoch may differ): {entropy.get("interpretation", "")}</li>
            <li>Toichi CVI = {toichi.get("cvi", "?")} (reference {toichi.get("cvi_healthy_ref", "?")}): Severely reduced vagal tone</li>
            <li>Cosinor amplitude = {cosinor.get("amplitude", "?")} bpm (reference 15-25): {cosinor.get("interpretation", "")}</li>
            <li>Allostatic load score = {allostatic.get("score", 0)}/7: {allostatic.get("severity", "")}</li>
            <li>RQA determinism = {rqa.get("determinism", "?")}: {rqa.get("interpretation", "")}</li>
        </ul>
        <br>
        <strong>Clinical Interpretation:</strong><br>
        Combined analyses show a pattern consistent with severe autonomic neuropathy with both
        parasympathetic failure (low CVI, low SampEn, low SD1) and chronodisruption (low cosinor amplitude).
        MSE pattern ({mse.get("pattern", "?")}) and DFA findings support loss of fractal dynamics.
        The high allostatic load ({allostatic.get("score", 0)}/7) indicates systemic
        physiological stress exceeding adaptive capacity.
        <br><br>
        <em>Analysis based on Oura Ring Gen 4 data ({metrics.get("data_range", {}).get("start", "?")} to {metrics.get("data_range", {}).get("end", "?")}). RMSSD epochs are 5-minute intervals
        during sleep. DFA and SampEn are computed from RMSSD epochs (proxy) - not beat-to-beat RR intervals.
        Reference values from RR-interval studies are not directly comparable.
        Frequency domain values are approximations based on Lomb-Scargle periodogram, not
        beat-to-beat analysis, and should be interpreted as relative indicators.
        Population norms for RMSSD are from controlled clinical 5-minute recordings;
        consumer wearable nocturnal values may differ.</em>
    </div>"""
    body += make_section("Clinical Summary", summary_content)

    # --- Assemble with wrap_html ---
    subtitle = (
        f"RMSSD epochs: {metrics.get('n_rmssd', 0):,} | "
        f"HR measurements: {metrics.get('n_hr', 0):,}"
    )
    return wrap_html(
        title="Advanced HRV Analysis",
        body_content=body,
        report_id="hrv",
        subtitle=subtitle,
        extra_css=extra_css,
    )


# ===========================================================================
# FIGURE GENERATION
# ===========================================================================
def create_frequency_domain_fig(freq_data: dict) -> go.Figure:
    """Lomb-Scargle periodogram plot with layered spectral band fills."""
    fig = go.Figure()
    freqs = freq_data.get("freqs", [])
    power = freq_data.get("power", [])
    if freqs and power:
        freqs_arr = np.array(freqs)
        power_arr = np.array(power)

        # Layered band fills (bottom layer first)
        band_defs = [
            (
                freqs_arr < 0.002,
                "Lower band",
                "rgba(99, 102, 241, 0.12)",
                ACCENT_INDIGO,
            ),
            (
                (freqs_arr >= 0.002) & (freqs_arr < 0.0035),
                "Mid band",
                "rgba(139, 92, 246, 0.18)",
                C_HRV,
            ),
            (freqs_arr >= 0.0035, "Upper band", "rgba(6, 182, 212, 0.12)", ACCENT_CYAN),
        ]
        for mask, label, fill_col, line_col in band_defs:
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=freqs_arr[mask].tolist(),
                        y=power_arr[mask].tolist(),
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color=line_col, width=0.5),
                        fillcolor=fill_col,
                        name=label,
                        hovertemplate=(
                            f"<b>{label}</b><br>"
                            "Freq: %{x:.5f} Hz<br>"
                            "Power: %{y:.4f}<extra></extra>"
                        ),
                    )
                )

        # Main PSD line on top
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=power,
                mode="lines",
                line=dict(color=C_HRV, width=2),
                name="PSD",
                hovertemplate=(
                    "<b>Spectral Power</b><br>"
                    "Freq: %{x:.5f} Hz<br>"
                    "Power: %{y:.4f}<extra></extra>"
                ),
            )
        )

        # Band boundary lines with subtle labels
        peak_power = float(max(power))
        for f_val, label in [(0.002, "Lower / Mid"), (0.0035, "Mid / Upper")]:
            fig.add_shape(
                type="line",
                x0=f_val,
                x1=f_val,
                y0=0,
                y1=peak_power * 1.05,
                line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dash"),
            )
            fig.add_annotation(
                x=f_val,
                y=peak_power * 1.02,
                text=label,
                showarrow=False,
                font=dict(size=9, color=TEXT_SECONDARY),
                bgcolor="rgba(15,17,23,0.7)",
                borderpad=2,
            )

    fig.update_layout(
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Normalized Spectral Power",
        height=350,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.2)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
        ),
    )
    return fig


def create_mse_fig(mse_data: dict) -> go.Figure:
    """Multiscale Entropy plot with patient vs healthy reference comparison."""
    fig = go.Figure()
    scales = mse_data.get("scales", [])
    entropies = mse_data.get("entropies", [])

    # Reference band: healthy MSE typically 1.5-2.5 across scales
    if scales:
        ref_values = [2.0 - 0.02 * (s - 1) for s in scales]  # Slight decline with scale
        ref_upper = [r + 0.3 for r in ref_values]
        ref_lower = [r - 0.3 for r in ref_values]

        # Healthy reference shaded band
        fig.add_trace(
            go.Scatter(
                x=list(scales) + list(reversed(scales)),
                y=ref_upper + list(reversed(ref_lower)),
                fill="toself",
                fillcolor="rgba(16,185,129,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Healthy range",
                showlegend=True,
                hoverinfo="skip",
            )
        )

        # Healthy reference center line
        fig.add_trace(
            go.Scatter(
                x=scales,
                y=ref_values,
                mode="lines+markers",
                marker=dict(
                    size=6,
                    color=ACCENT_GREEN,
                    symbol="diamond",
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                line=dict(color=ACCENT_GREEN, width=1.5, dash="dash"),
                name="Healthy reference",
                hovertemplate="<b>Healthy Ref</b><br>Scale %{x}<br>SampEn: %{y:.3f}<extra></extra>",
            )
        )

        # Connecting lines between patient and reference at each scale
        for i, s in enumerate(scales):
            if i < len(entropies) and not np.isnan(entropies[i]):
                fig.add_trace(
                    go.Scatter(
                        x=[s, s],
                        y=[entropies[i], ref_values[i]],
                        mode="lines",
                        line=dict(color="rgba(255,255,255,0.08)", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # Patient MSE - on top
    fig.add_trace(
        go.Scatter(
            x=scales,
            y=entropies,
            mode="lines+markers",
            marker=dict(
                size=8,
                color=C_HRV,
                symbol="circle",
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            line=dict(color=C_HRV, width=2.5),
            name="Patient MSE",
            hovertemplate="<b>Patient MSE</b><br>Scale %{x}<br>SampEn: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis_title="Scale Factor",
        yaxis_title="Sample Entropy",
        height=380,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
            dtick=2,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
        ),
    )
    return fig


def create_rqa_fig(rmssd: np.ndarray, rqa_data: dict) -> go.Figure:
    """Recurrence plot with refined dark-aware colorscale."""
    from scipy.spatial.distance import cdist

    max_pts = 500
    if len(rmssd) > max_pts:
        idx = np.linspace(0, len(rmssd) - 1, max_pts, dtype=int)
        data = rmssd[idx]
    else:
        data = rmssd.copy()

    N = len(data)
    threshold = 0.10 * np.std(data, ddof=1)

    # Time-delay embedding (dim=3, delay=1)
    M = N - 2
    embedded = np.column_stack([data[:-2], data[1:-1], data[2:]])
    dist = cdist(embedded, embedded, metric="chebyshev")

    # Use distance matrix for richer colorscale (inverted: close = bright)
    max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
    proximity = 1.0 - (dist / max_dist)
    # Zero out non-recurrent points to keep background dark
    rec_mask = dist <= threshold
    display_matrix = np.where(rec_mask, proximity, 0.0)

    # Custom dark-aware sequential colorscale
    rqa_colorscale = [
        [0.0, BG_PRIMARY],
        [0.15, "#1a1040"],
        [0.3, "#2d1b69"],
        [0.5, "#5b2da8"],
        [0.7, "#8B5CF6"],
        [0.85, "#a78bfa"],
        [1.0, "#ddd6fe"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=display_matrix,
            colorscale=rqa_colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text="Proximity", font=dict(size=11, color=TEXT_SECONDARY)),
                tickfont=dict(size=10, color=TEXT_SECONDARY),
                thickness=12,
                len=0.7,
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ),
            hovertemplate=(
                "<b>Recurrence Plot</b><br>"
                "Point i: %{x}<br>"
                "Point j: %{y}<br>"
                "Proximity: %{z:.3f}<extra></extra>"
            ),
        )
    )

    # Add RQA metrics annotation
    det = rqa_data.get("determinism", 0)
    lam = rqa_data.get("laminarity", 0)
    rr = rqa_data.get("recurrence_rate", 0)
    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text=f"DET={det:.3f}  LAM={lam:.3f}  RR={rr:.4f}",
        showarrow=False,
        align="right",
        bgcolor="rgba(15,17,23,0.85)",
        bordercolor=BORDER_SUBTLE,
        borderwidth=1,
        font=dict(size=10, color=TEXT_PRIMARY),
        borderpad=4,
    )

    fig.update_layout(
        margin=dict(l=50, r=80, t=50, b=40),
        xaxis_title="Time point i",
        yaxis_title="Time point j",
        height=420,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(zeroline=False),
    )
    return fig


def create_dfa_fig(rmssd: np.ndarray, dfa_data: dict) -> go.Figure:
    """RMSSD-Epoch DFA (Proxy) log-log plot with labeled scaling regions."""
    if not HAS_NOLDS or len(rmssd) < 20:
        return go.Figure()

    # Compute fluctuation function manually for plotting
    mean_val = np.mean(rmssd)
    y = np.cumsum(rmssd - mean_val)
    N = len(y)

    n_values = np.unique(
        np.logspace(np.log10(4), np.log10(min(N // 4, 200)), 30).astype(int)
    )
    F_n = []

    for n in n_values:
        n_windows = N // n
        if n_windows < 2:
            continue
        residuals = []
        for i in range(n_windows):
            segment = y[i * n : (i + 1) * n]
            x_fit = np.arange(n)
            coeffs = np.polyfit(x_fit, segment, 1)
            trend = np.polyval(coeffs, x_fit)
            residuals.extend((segment - trend) ** 2)
        F_n.append(np.sqrt(np.mean(residuals)))

    F_n = np.array(F_n)
    n_values = n_values[: len(F_n)]

    fig = go.Figure()
    log_n = np.log10(n_values)
    log_F = np.log10(F_n)

    # Shaded scaling regions
    log_16 = np.log10(16)
    y_min, y_max = float(log_F.min()) - 0.1, float(log_F.max()) + 0.1

    # Alpha-1 region shade
    fig.add_shape(
        type="rect",
        x0=float(log_n.min()),
        x1=log_16,
        y0=y_min,
        y1=y_max,
        fillcolor="rgba(239,68,68,0.05)",
        line=dict(width=0),
        layer="below",
    )
    # Alpha-2 region shade
    fig.add_shape(
        type="rect",
        x0=log_16,
        x1=float(log_n.max()),
        y0=y_min,
        y1=y_max,
        fillcolor="rgba(16,185,129,0.05)",
        line=dict(width=0),
        layer="below",
    )

    # Data points
    fig.add_trace(
        go.Scatter(
            x=log_n,
            y=log_F,
            mode="markers",
            marker=dict(
                size=8, color=C_HRV, line=dict(width=1, color="rgba(255,255,255,0.2)")
            ),
            name="F(n) points",
            hovertemplate="<b>DFA</b><br>log(n): %{x:.2f}<br>log(F): %{y:.3f}<extra></extra>",
        )
    )

    # Alpha-1 fit (short scales, n=4-16)
    alpha1_val = dfa_data.get("alpha1", "?")
    alpha1_ci = dfa_data.get("alpha1_ci_95", ("?", "?"))
    mask1 = n_values <= 16
    if mask1.sum() >= 3:
        c1 = np.polyfit(log_n[mask1], log_F[mask1], 1)
        fit1 = np.polyval(c1, log_n[mask1])
        fig.add_trace(
            go.Scatter(
                x=log_n[mask1],
                y=fit1,
                mode="lines",
                line=dict(color=ACCENT_RED, width=2.5),
                name=f"alpha-1 = {alpha1_val}",
                hovertemplate=f"<b>alpha-1 = {alpha1_val}</b><extra></extra>",
            )
        )
        # Alpha-1 annotation box
        mid_x1 = float(np.mean(log_n[mask1]))
        mid_y1 = float(np.mean(fit1))
        fig.add_annotation(
            x=mid_x1,
            y=mid_y1 + 0.12,
            text=f"<b>alpha-1 = {alpha1_val}</b><br><span style='font-size:9px'>CI: [{alpha1_ci[0]}, {alpha1_ci[1]}]</span>",
            showarrow=True,
            arrowhead=2,
            arrowcolor=ACCENT_RED,
            arrowwidth=1,
            bgcolor="rgba(239,68,68,0.15)",
            bordercolor=ACCENT_RED,
            borderwidth=1,
            font=dict(size=11, color=TEXT_PRIMARY),
            borderpad=5,
        )

    # Alpha-2 fit (long scales, n>=16)
    alpha2_val = dfa_data.get("alpha2", "?")
    alpha2_ci = dfa_data.get("alpha2_ci_95", ("?", "?"))
    mask2 = n_values >= 16
    if mask2.sum() >= 3:
        c2 = np.polyfit(log_n[mask2], log_F[mask2], 1)
        fit2 = np.polyval(c2, log_n[mask2])
        fig.add_trace(
            go.Scatter(
                x=log_n[mask2],
                y=fit2,
                mode="lines",
                line=dict(color=ACCENT_GREEN, width=2.5, dash="dash"),
                name=f"alpha-2 = {alpha2_val}",
                hovertemplate=f"<b>alpha-2 = {alpha2_val}</b><extra></extra>",
            )
        )
        # Alpha-2 annotation box
        mid_x2 = float(np.mean(log_n[mask2]))
        mid_y2 = float(np.mean(fit2))
        fig.add_annotation(
            x=mid_x2,
            y=mid_y2 - 0.12,
            text=f"<b>alpha-2 = {alpha2_val}</b><br><span style='font-size:9px'>CI: [{alpha2_ci[0]}, {alpha2_ci[1]}]</span>",
            showarrow=True,
            arrowhead=2,
            arrowcolor=ACCENT_GREEN,
            arrowwidth=1,
            bgcolor="rgba(16,185,129,0.15)",
            bordercolor=ACCENT_GREEN,
            borderwidth=1,
            font=dict(size=11, color=TEXT_PRIMARY),
            borderpad=5,
        )

    # Reference line for alpha=1.0
    x_ref = np.array([log_n.min(), log_n.max()])
    y_ref = log_F.mean() + 1.0 * (x_ref - log_n.mean())
    fig.add_trace(
        go.Scatter(
            x=x_ref,
            y=y_ref,
            mode="lines",
            line=dict(color=TEXT_SECONDARY, width=1, dash="dot"),
            name="alpha=1.0 (RR-interval ref)",
            hovertemplate="<b>Reference slope (alpha=1.0)</b><extra></extra>",
        )
    )

    fig.update_layout(
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis_title="log10(n)",
        yaxis_title="log10(F(n))",
        height=380,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
        ),
    )
    return fig


def create_entropy_fig(rmssd: np.ndarray, entropy_data: dict) -> go.Figure:
    """Bar chart comparing ApEn, SampEn with reference values."""
    categories = [
        "ApEn<br>(patient)",
        "SampEn<br>(patient)",
        "SampEn<br>(healthy ref)",
        "SampEn<br>(post-HSCT ref)",
    ]
    values = [
        entropy_data.get("apen", 0) or 0,
        entropy_data.get("sampen", 0) or 0,
        2.0,  # Healthy reference
        0.8,  # Post-HSCT typical
    ]
    colors = [C_HRV, C_HRV, ACCENT_GREEN, TEXT_SECONDARY]
    border_colors = [
        "rgba(139,92,246,0.6)",
        "rgba(139,92,246,0.6)",
        "rgba(16,185,129,0.6)",
        "rgba(156,163,175,0.4)",
    ]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=colors,
                line=dict(width=1.5, color=border_colors),
                opacity=0.9,
            ),
            text=[f"<b>{v:.3f}</b>" for v in values],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=13),
            hovertemplate="<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>",
        )
    )

    # Add a threshold line at healthy lower bound
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=3.5,
        y0=1.5,
        y1=1.5,
        line=dict(color=ACCENT_GREEN, width=1, dash="dash"),
    )
    fig.add_annotation(
        x=3.5,
        y=1.5,
        text="Healthy lower bound",
        showarrow=False,
        xanchor="right",
        font=dict(size=9, color=ACCENT_GREEN),
    )

    fig.update_layout(
        margin=dict(t=50, b=70, l=60, r=30),
        yaxis_title="Entropy Value",
        height=370,
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
            range=[0, max(values) * 1.25],
        ),
        xaxis=dict(zeroline=False),
        bargap=0.3,
    )
    return fig


def create_poincare_fig(rmssd: np.ndarray, toichi_data: dict) -> go.Figure:
    """Poincare plot with SD1/SD2 ellipse glow effect and density-based sizing."""
    x_n = rmssd[:-1]
    x_n1 = rmssd[1:]
    mean_val = np.mean(rmssd)

    sd1 = toichi_data.get("sd1", 0)
    sd2 = toichi_data.get("sd2", 0)

    fig = go.Figure()

    # Compute local density for point sizing (2D histogram-based)
    from scipy.stats import gaussian_kde

    try:
        xy = np.vstack([x_n, x_n1])
        # Subsample for KDE if too many points
        if len(x_n) > 5000:
            kde_idx = np.random.default_rng(42).choice(len(x_n), 5000, replace=False)
            kde = gaussian_kde(xy[:, kde_idx])
        else:
            kde = gaussian_kde(xy)
        density = kde(xy)
        # Normalize density to size range 2-6
        d_min, d_max = density.min(), density.max()
        if d_max > d_min:
            sizes = 2 + 4 * (
                1 - (density - d_min) / (d_max - d_min)
            )  # Invert: sparse = larger
        else:
            sizes = np.full(len(density), 3.0)
    except Exception:
        sizes = np.full(len(x_n), 3.0)
        density = np.ones(len(x_n))

    # Scatter with density-based sizing
    fig.add_trace(
        go.Scatter(
            x=x_n,
            y=x_n1,
            mode="markers",
            marker=dict(
                size=sizes,
                color=x_n,
                colorscale="RdYlBu_r",
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="RMSSD (ms)", font=dict(size=11, color=TEXT_SECONDARY)
                    ),
                    tickfont=dict(size=10, color=TEXT_SECONDARY),
                    thickness=12,
                    len=0.6,
                    outlinewidth=0,
                ),
                opacity=0.5,
                line=dict(width=0),
            ),
            name="RMSSD(n) vs RMSSD(n+1)",
            hovertemplate=(
                "<b>Poincare</b><br>"
                "RMSSD(n): %{x:.1f} ms<br>"
                "RMSSD(n+1): %{y:.1f} ms<extra></extra>"
            ),
        )
    )

    # Identity line
    max_val = max(float(np.max(x_n)), float(np.max(x_n1)))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color=TEXT_SECONDARY, dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # SD1/SD2 ellipse - glow effect (outer glow layers + main)
    theta = np.linspace(0, 2 * np.pi, 100)
    x_ell = sd2 * np.cos(theta)
    y_ell = sd1 * np.sin(theta)
    angle = np.pi / 4
    x_rot = np.cos(angle) * x_ell - np.sin(angle) * y_ell + mean_val
    y_rot = np.sin(angle) * x_ell + np.cos(angle) * y_ell + mean_val

    # Glow layers (3 progressively wider/more transparent)
    for glow_scale, glow_alpha, glow_width in [
        (1.15, 0.04, 6),
        (1.08, 0.08, 4),
        (1.03, 0.15, 3),
    ]:
        x_glow = (
            np.cos(angle) * (sd2 * glow_scale * np.cos(theta))
            - np.sin(angle) * (sd1 * glow_scale * np.sin(theta))
            + mean_val
        )
        y_glow = (
            np.sin(angle) * (sd2 * glow_scale * np.cos(theta))
            + np.cos(angle) * (sd1 * glow_scale * np.sin(theta))
            + mean_val
        )
        fig.add_trace(
            go.Scatter(
                x=x_glow,
                y=y_glow,
                mode="lines",
                line=dict(color=f"rgba(239,68,68,{glow_alpha})", width=glow_width),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Main ellipse
    fig.add_trace(
        go.Scatter(
            x=x_rot,
            y=y_rot,
            mode="lines",
            line=dict(color=ACCENT_RED, width=2),
            name=f"SD1={sd1:.1f}, SD2={sd2:.1f}",
            hovertemplate=f"<b>Ellipse</b><br>SD1={sd1:.1f} ms, SD2={sd2:.1f} ms<extra></extra>",
        )
    )

    # Annotation
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Toichi:</b> CVI={toichi_data.get('cvi', '?')}, CSI={toichi_data.get('csi', '?')}<br>"
            f"SD1={sd1:.1f} ms ({toichi_data.get('sd1_pct_of_normal', '?')}% of healthy)<br>"
            f"SD2={sd2:.1f} ms ({toichi_data.get('sd2_pct_of_normal', '?')}% of healthy)"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(26, 29, 39, 0.92)",
        bordercolor=BORDER_SUBTLE,
        borderwidth=1,
        font=dict(size=10, color=TEXT_PRIMARY),
        borderpad=6,
    )

    fig.update_layout(
        margin=dict(l=50, r=80, t=50, b=40),
        xaxis_title="RMSSD(n) [ms]",
        yaxis_title="RMSSD(n+1) [ms]",
        height=450,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", griddash="dot", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", griddash="dot", zeroline=False),
    )
    return fig


def create_cosinor_fig(
    cosinor_data: dict, hr_timestamps: list[datetime], hr_bpm: np.ndarray
) -> go.Figure:
    """Cosinor fit overlaid on hourly HR averages with subtle fill and prominent acrophase."""
    fig = go.Figure()

    # Hourly averages
    hourly_hr = {}
    for ts, bpm in zip(hr_timestamps, hr_bpm):
        h = ts.hour
        hourly_hr.setdefault(h, []).append(bpm)

    hours_sorted = sorted(hourly_hr.keys())
    avg_by_hour = [np.mean(hourly_hr[h]) for h in hours_sorted]
    std_by_hour = [
        np.std(hourly_hr[h], ddof=1) if len(hourly_hr[h]) > 1 else 0
        for h in hours_sorted
    ]

    # Cosinor fit curve with fill below
    t_plot = cosinor_data.get("t_plot", [])
    y_plot = cosinor_data.get("y_plot", [])
    mesor = cosinor_data.get("mesor", 0)
    if t_plot and y_plot:
        # Subtle gradient fill between MESOR and curve
        fig.add_trace(
            go.Scatter(
                x=t_plot,
                y=y_plot,
                mode="lines",
                fill="tozeroy",
                line=dict(color="rgba(239,68,68,0)", width=0),
                fillcolor="rgba(239,68,68,0.06)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Fitted curve (smooth, solid)
        fig.add_trace(
            go.Scatter(
                x=t_plot,
                y=y_plot,
                mode="lines",
                line=dict(color=ACCENT_RED, width=2.5, shape="spline"),
                name=f"Cosinor fit (A={cosinor_data.get('amplitude', '?')} bpm)",
                hovertemplate=(
                    "<b>Cosinor Fit</b><br>"
                    "Hour: %{x:.1f}<br>"
                    "HR: %{y:.1f} bpm<extra></extra>"
                ),
            )
        )

    # Data points with error bars
    fig.add_trace(
        go.Scatter(
            x=hours_sorted,
            y=avg_by_hour,
            mode="markers+lines",
            marker=dict(
                size=8,
                color=ACCENT_BLUE,
                line=dict(width=1, color="rgba(255,255,255,0.2)"),
            ),
            line=dict(color=ACCENT_BLUE, width=1.5),
            error_y=dict(
                type="data",
                array=std_by_hour,
                visible=True,
                color="rgba(59,130,246,0.25)",
                thickness=1.5,
            ),
            name="Mean HR per hour",
            hovertemplate=(
                "<b>Hourly Mean</b><br>Hour: %{x}:00<br>HR: %{y:.1f} bpm<extra></extra>"
            ),
        )
    )

    # MESOR line
    fig.add_shape(
        type="line",
        x0=0,
        x1=24,
        y0=mesor,
        y1=mesor,
        line=dict(color=TEXT_SECONDARY, width=1, dash="dot"),
    )
    fig.add_annotation(
        x=0.5,
        y=mesor,
        text=f"MESOR = {mesor:.1f} bpm",
        showarrow=False,
        yshift=-12,
        font=dict(size=10, color=TEXT_SECONDARY),
    )

    # Acrophase marker with prominent annotation
    acro = cosinor_data.get("acrophase_hours", 0)
    amp = cosinor_data.get("amplitude", 0)
    acro_hhmm = cosinor_data.get("acrophase_hhmm", "?")

    # Glow effect for acrophase
    fig.add_trace(
        go.Scatter(
            x=[acro],
            y=[mesor + amp],
            mode="markers",
            marker=dict(size=20, color="rgba(239,68,68,0.15)", line=dict(width=0)),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[acro],
            y=[mesor + amp],
            mode="markers",
            marker=dict(
                size=12,
                color=ACCENT_RED,
                symbol="star",
                line=dict(width=1, color="rgba(255,255,255,0.4)"),
            ),
            name=f"Acrophase ({acro_hhmm})",
            hovertemplate=f"<b>Acrophase</b><br>Time: {acro_hhmm}<br>Peak HR: {mesor + amp:.1f} bpm<extra></extra>",
        )
    )
    fig.add_annotation(
        x=acro,
        y=mesor + amp,
        text=f"<b>Acrophase: {acro_hhmm}</b><br>Amplitude: {amp:.1f} bpm",
        showarrow=True,
        arrowhead=2,
        arrowcolor=ACCENT_RED,
        arrowwidth=1.5,
        ay=-40,
        ax=40,
        bgcolor="rgba(239,68,68,0.12)",
        bordercolor=ACCENT_RED,
        borderwidth=1,
        font=dict(size=11, color=TEXT_PRIMARY),
        borderpad=5,
    )

    fig.update_layout(
        margin=dict(t=50, b=50, l=60, r=30),
        xaxis_title="Time of Day (hours)",
        yaxis_title="Heart Rate (bpm)",
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=2,
            range=[0, 24],
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
        ),
        height=420,
    )
    return fig


def create_nightly_fig(nightly_data: dict, sleep_periods: pd.DataFrame) -> go.Figure:
    """Night-to-night HR variability trend with gradient fill and refined thresholds."""
    fig = go.Figure()

    if not sleep_periods.empty and "average_heart_rate" in sleep_periods.columns:
        df = sleep_periods.dropna(subset=["average_heart_rate"]).sort_values("day")
        days = pd.to_datetime(df["day"])
        hrs = df["average_heart_rate"].values

        # Subtle gradient fill below the data line
        fig.add_trace(
            go.Scatter(
                x=days,
                y=hrs,
                mode="lines",
                fill="tozeroy",
                line=dict(color="rgba(59,130,246,0)", width=0),
                fillcolor="rgba(59,130,246,0.06)",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Main data trace
        fig.add_trace(
            go.Scatter(
                x=days,
                y=hrs,
                mode="lines+markers",
                marker=dict(
                    size=5,
                    color=ACCENT_BLUE,
                    line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
                ),
                line=dict(color=ACCENT_BLUE, width=1.5),
                name="Nocturnal Mean HR",
                hovertemplate=(
                    "<b>Nocturnal HR</b><br>"
                    "%{x|%d %b %Y}<br>"
                    "HR: %{y:.1f} bpm<extra></extra>"
                ),
            )
        )

        # Trend line
        if len(hrs) >= 5:
            x_num = np.arange(len(hrs))
            slope, intercept = np.polyfit(x_num, hrs, 1)
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=intercept + slope * x_num,
                    mode="lines",
                    line=dict(color=ACCENT_RED, width=1.5, dash="dash"),
                    name=f"Trend ({slope:+.2f} bpm/night)",
                    hovertemplate=f"<b>Trend</b><br>Slope: {slope:+.2f} bpm/night<extra></extra>",
                )
            )

        # CV annotation
        cv = nightly_data.get("cv_pct", 0)
        fig.add_annotation(
            x=0.02,
            y=0.92,
            xref="paper",
            yref="paper",
            text=f"<b>CV = {cv:.1f}%</b> | Mean = {nightly_data.get('mean_hr', '?')} bpm | n={nightly_data.get('n_nights', '?')} nights",
            showarrow=False,
            align="left",
            bgcolor="rgba(26, 29, 39, 0.92)",
            bordercolor=BORDER_SUBTLE,
            borderwidth=1,
            font=dict(size=11, color=TEXT_PRIMARY),
            borderpad=5,
        )

    # Threshold lines with distinct dash patterns
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=NOCTURNAL_HR_ELEVATED,
        y1=NOCTURNAL_HR_ELEVATED,
        line=dict(color=ACCENT_RED, width=1.5, dash="dashdot"),
    )
    fig.add_annotation(
        x=0.98,
        y=NOCTURNAL_HR_ELEVATED,
        xref="paper",
        yshift=10,
        text=f"Nocturnal concern ({NOCTURNAL_HR_ELEVATED} bpm)",
        showarrow=False,
        font=dict(size=9, color=ACCENT_RED),
        xanchor="right",
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=65,
        y1=65,
        line=dict(color=ACCENT_GREEN, width=1, dash="dot"),
    )
    fig.add_annotation(
        x=0.98,
        y=65,
        xref="paper",
        yshift=-10,
        text="Healthy nocturnal mean (65 bpm)",
        showarrow=False,
        font=dict(size=9, color=ACCENT_GREEN),
        xanchor="right",
    )

    # Ruxolitinib start annotation
    fig.add_shape(
        type="line",
        x0=TREATMENT_START_STR,
        x1=TREATMENT_START_STR,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=ACCENT_BLUE, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=TREATMENT_START_STR,
        y=0.92,
        yref="paper",
        text="Ruxolitinib start",
        showarrow=False,
        font=dict(size=10, color=ACCENT_BLUE),
        bgcolor="rgba(15,17,23,0.7)",
        borderpad=2,
    )

    fig.update_layout(
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis_title="Date",
        yaxis_title="Mean Nocturnal HR (bpm)",
        height=400,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(255,255,255,0.2)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            griddash="dot",
            zeroline=False,
        ),
    )
    return fig


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> int:
    """Run all analyses and generate report."""
    print("=" * 70)
    print("  ADVANCED HRV & AUTONOMIC FUNCTION ANALYSIS")
    print(f"  {PATIENT_LABEL}")
    print("=" * 70)
    print()

    conn = connect_db(DATABASE_PATH)
    try:
        _run_analysis(conn)
    finally:
        conn.close()

    return 0


def _run_analysis(conn: sqlite3.Connection) -> None:
    """Core analysis logic, separated for DB connection safety."""

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("[1/12] Loading RMSSD data...")
    hrv_rows = conn.execute(
        "SELECT timestamp, rmssd FROM oura_hrv WHERE rmssd IS NOT NULL ORDER BY timestamp"
    ).fetchall()
    hrv_timestamps_raw = [r["timestamp"] for r in hrv_rows]
    rmssd_all = np.array([r["rmssd"] for r in hrv_rows], dtype=float)

    # Parse timestamps
    hrv_timestamps = [parse_ts(ts) for ts in hrv_timestamps_raw]

    print(f"  -> {len(rmssd_all):,} RMSSD epochs loaded")
    print(
        f"  -> Mean: {np.mean(rmssd_all):.1f} ms, Median: {np.median(rmssd_all):.1f} ms"
    )
    print(f"  -> Range: {np.min(rmssd_all):.0f}-{np.max(rmssd_all):.0f} ms")

    print("\n[2/12] Loading HR data...")
    hr_rows = conn.execute(
        "SELECT timestamp, bpm FROM oura_heart_rate ORDER BY timestamp"
    ).fetchall()
    hr_timestamps_raw = [r["timestamp"] for r in hr_rows]
    hr_bpm = np.array([r["bpm"] for r in hr_rows], dtype=float)
    hr_timestamps = [parse_ts(ts) for ts in hr_timestamps_raw]

    print(f"  -> {len(hr_bpm):,} HR measurements loaded")
    print(f"  -> Mean: {np.mean(hr_bpm):.1f} bpm")

    print("\n[3/12] Loading sleep periods...")
    sleep_df = pd.read_sql_query("SELECT * FROM oura_sleep_periods ORDER BY day", conn)
    print(f"  -> {len(sleep_df)} sleep periods")

    sleep_days = pd.to_datetime(sleep_df.get("day"), errors="coerce").dropna()
    data_start = sleep_days.min().date().isoformat() if not sleep_days.empty else None
    data_end = sleep_days.max().date().isoformat() if not sleep_days.empty else None
    generated_at = datetime.now(timezone.utc).isoformat()

    # Collect all metrics
    metrics: dict[str, Any] = {
        "generated": generated_at,
        "generated_at": generated_at,
        "data_range": {
            "start": data_start,
            "end": data_end,
            "n_sleep_periods": int(len(sleep_df)),
            "n_rmssd_epochs": int(len(rmssd_all)),
            "n_hr_samples": int(len(hr_bpm)),
        },
        "n_rmssd": len(rmssd_all),
        "n_hr": len(hr_bpm),
        "rmssd_mean": round(float(np.mean(rmssd_all)), 2),
        "rmssd_median": round(float(np.median(rmssd_all)), 2),
        "rmssd_std": round(float(np.std(rmssd_all, ddof=1)), 2),
        "rmssd_min": round(float(np.min(rmssd_all)), 1),
        "rmssd_max": round(float(np.max(rmssd_all)), 1),
        "hr_mean": round(float(np.mean(hr_bpm)), 1),
        "hr_std": round(float(np.std(hr_bpm, ddof=1)), 1),
    }

    figures: dict[str, go.Figure] = {}

    # -----------------------------------------------------------------------
    # Compute all metrics
    # -----------------------------------------------------------------------
    print("\n[4/12] Frequency domain (Lomb-Scargle)...")
    freq_data = compute_frequency_domain(hrv_timestamps, rmssd_all)
    metrics["frequency_domain"] = {
        k: v for k, v in freq_data.items() if k not in ("freqs", "power")
    }
    figures["frequency_domain"] = create_frequency_domain_fig(freq_data)
    print(
        f"  -> VLF={freq_data['vlf_pct']}%, LF={freq_data['lf_pct']}%, HF={freq_data['hf_pct']}%, LF/HF={freq_data['lf_hf_ratio']}"
    )

    print("\n[5/12] Multiscale entropy (MSE)...")
    mse_data = compute_mse(rmssd_all, max_scale=20)
    metrics["mse"] = {k: v for k, v in mse_data.items() if k != "entropies"}
    metrics["mse"]["entropies"] = [
        round(e, 4) if not np.isnan(e) else None for e in mse_data["entropies"]
    ]
    figures["mse"] = create_mse_fig(mse_data)
    print(f"  -> Pattern: {mse_data['pattern']}")

    print("\n[6/12] Recurrence quantification (RQA)...")
    rqa_data = compute_rqa(rmssd_all)
    metrics["rqa"] = rqa_data
    figures["rqa"] = create_rqa_fig(rmssd_all, rqa_data)
    print(
        f"  -> DET={rqa_data['determinism']}, LAM={rqa_data['laminarity']}, RR={rqa_data['recurrence_rate']}"
    )

    print("\n[7/12] DFA alpha-1/alpha-2...")
    dfa_data = compute_dfa(rmssd_all)
    metrics["dfa"] = dfa_data
    figures["dfa"] = create_dfa_fig(rmssd_all, dfa_data)
    print(
        f"  -> alpha-1={dfa_data['alpha1']} ({dfa_data['alpha1_ci_95']}), alpha-2={dfa_data['alpha2']}"
    )

    print("\n[8/12] ApEn/SampEn...")
    entropy_data = compute_entropy_measures(rmssd_all)
    metrics["entropy"] = entropy_data
    figures["entropy"] = create_entropy_fig(rmssd_all, entropy_data)
    print(f"  -> ApEn={entropy_data['apen']}, SampEn={entropy_data['sampen']}")

    print("\n[9/12] Hjorth parameters...")
    hjorth_data = compute_hjorth(rmssd_all)
    metrics["hjorth"] = hjorth_data
    print(
        f"  -> Activity={hjorth_data['activity']}, Mobility={hjorth_data['mobility']}, Complexity={hjorth_data['complexity']}"
    )

    print("\n[10/12] Baevsky stress index...")
    baevsky_data = compute_baevsky_si(rmssd_all)
    metrics["baevsky"] = baevsky_data
    print(f"  -> SI={baevsky_data['si_scaled']} ({baevsky_data['classification']})")

    print("\n[11/12] Toichi CVI/CSI + Poincare...")
    toichi_data = compute_toichi(rmssd_all)
    metrics["toichi"] = toichi_data
    figures["poincare"] = create_poincare_fig(rmssd_all, toichi_data)
    print(
        f"  -> CVI={toichi_data['cvi']}, CSI={toichi_data['csi']}, SD1={toichi_data['sd1']}, SD2={toichi_data['sd2']}"
    )

    print("\n[12/12] HR-based analyses...")
    # Cosinor
    cosinor_data = compute_cosinor(hr_timestamps, hr_bpm)
    metrics["cosinor"] = {
        k: v for k, v in cosinor_data.items() if k not in ("t_plot", "y_plot")
    }
    figures["cosinor"] = create_cosinor_fig(cosinor_data, hr_timestamps, hr_bpm)
    print(
        f"  -> MESOR={cosinor_data['mesor']} bpm, Amplitude={cosinor_data['amplitude']} bpm, Acrophase={cosinor_data.get('acrophase_hhmm', '?')}"
    )

    # HR complexity
    hr_complex = compute_hr_complexity(hr_bpm)
    metrics["hr_complexity"] = hr_complex
    print(
        f"  -> PermEnt={hr_complex.get('permutation_entropy', '?')}, SpectralEnt={hr_complex.get('spectral_entropy', '?')}"
    )

    # Nightly HR variability
    nightly_data = compute_nightly_hr_variability(sleep_df)
    metrics["nightly_hr_variability"] = {
        k: v for k, v in nightly_data.items() if k != "nightly_values"
    }
    metrics["nightly_hr_variability"]["nightly_values"] = nightly_data.get(
        "nightly_values", []
    )
    figures["nightly"] = create_nightly_fig(nightly_data, sleep_df)
    print(
        f"  -> CV={nightly_data.get('cv_pct', '?')}%, Mean={nightly_data.get('mean_hr', '?')} bpm"
    )

    # Allostatic load
    allostatic_data = compute_allostatic_load(conn)
    metrics["allostatic_load"] = allostatic_data
    print(
        f"  -> Allostatic load: {allostatic_data['score']}/7 ({allostatic_data['severity']})"
    )

    # -----------------------------------------------------------------------
    # Generate outputs
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Generating report...")

    # JSON metrics
    # Convert numpy types for JSON serialization
    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    def _sanitize_nan(obj: Any) -> Any:
        """Recursively replace float NaN/Inf with None for valid JSON."""
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize_nan(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize_nan(v) for v in obj]
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            return _json_safe(obj)

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_nan(metrics), f, indent=2, ensure_ascii=False, cls=NumpyEncoder
        )
    print(f"  -> JSON: {JSON_OUTPUT}")

    # HTML report
    html = generate_html_report(metrics, figures)
    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> HTML: {HTML_OUTPUT}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"""
  RMSSD:          {metrics["rmssd_mean"]:.1f} +/- {metrics["rmssd_std"]:.1f} ms (healthy ref: ~{POPULATION_RMSSD_MEDIAN} ms)
  DFA alpha-1:    {dfa_data["alpha1"]} (95% CI: {dfa_data["alpha1_ci_95"]}) [healthy: ~1.0]
  DFA alpha-2:    {dfa_data["alpha2"]} [healthy: ~1.0]
  SampEn:         {entropy_data.get("sampen", "?")} [healthy: 1.5-2.5]
  ApEn:           {entropy_data.get("apen", "?")}
  Toichi CVI:     {toichi_data["cvi"]} [healthy: {toichi_data["cvi_healthy_ref"]}]
  Toichi CSI:     {toichi_data["csi"]} [healthy: {toichi_data["csi_healthy_ref"]}]
  SD1/SD2:        {toichi_data["sd1"]}/{toichi_data["sd2"]} ms [healthy: 30/60 ms]
  Baevsky SI:     {baevsky_data["si_scaled"]} [{baevsky_data["classification"]}]
  RQA DET:        {rqa_data["determinism"]} [healthy: >0.6]
  RQA LAM:        {rqa_data["laminarity"]}
  MSE pattern:    {mse_data["pattern"]}
  Hjorth:         Act={hjorth_data["activity"]}, Mob={hjorth_data["mobility"]}, Compl={hjorth_data["complexity"]}
  Proxy ratio:    {freq_data["lf_hf_ratio"]}
  Cosinor MESOR:  {cosinor_data["mesor"]} bpm [healthy: 60-80]
  Cosinor Amp:    {cosinor_data["amplitude"]} bpm [healthy: 15-25]
  HR PermEnt:     {hr_complex.get("permutation_entropy", "?")} [healthy: >0.85]
  Nightly CV:     {nightly_data.get("cv_pct", "?")}%
  Allostatic:     {allostatic_data["score"]}/7 ({allostatic_data["severity"]})

  Report: {HTML_OUTPUT}
  Metrics: {JSON_OUTPUT}
""")


if __name__ == "__main__":
    sys.exit(main())
