#!/usr/bin/env python3
"""Module 5: Anomaly Pattern Comparison.

Compares how "bad days" manifest differently between Patient 1 (post-HSCT) and
Patient 2 (post-Stroke).  What is each patient's anomaly "signature"?

Three anomaly detection methods (z-score threshold, Isolation Forest,
percentile-based) are ensembled.  Anomaly fingerprints, co-deviation matrices,
PCA biplots, and DBSCAN clustering reveal each patient's dominant failure modes.

Outputs:
  - Interactive HTML dashboard: reports/comparative_anomaly_report.html
  - JSON metrics:               reports/comparative_anomaly_metrics.json

Usage:
    python analysis/analyze_comparative_anomalies.py
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from scipy.spatial.distance import cosine as cosine_distance

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution & config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import REPORTS_DIR, FONT_FAMILY, KNOWN_EVENT_DATE
from _comparative_utils import (
    PatientConfig,
    default_patients,
    load_patient_data,
    PATIENT_COLORS,
)
from _theme import (
    wrap_html,
    make_kpi_card,
    make_kpi_row,
    make_section,
    disclaimer_banner,
    format_p_value,
    COLORWAY,
    BG_PRIMARY,
    BG_SURFACE,
    BG_ELEVATED,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_AMBER,
    ACCENT_PURPLE,
    ACCENT_CYAN,
    ACCENT_ORANGE,
    ACCENT_PINK,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    BORDER_SUBTLE,
)
from _hardening import safe_connect, safe_read_sql, section_html_or_placeholder

pio.templates.default = "clinical_dark"

HTML_OUTPUT = REPORTS_DIR / "comparative_anomaly_report.html"
JSON_OUTPUT = REPORTS_DIR / "comparative_anomaly_metrics.json"

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# (key, table, column, display_name, unit, higher_is_better, duration_to_hours)
METRIC_DEFS: list[tuple[str, str, str, str, str, bool | None, bool]] = [
    ("sleep_score", "oura_sleep", "score", "Sleep Score", "pts", True, False),
    ("total_sleep_hrs", "oura_sleep", "total_sleep_duration", "Total Sleep", "hrs", True, True),
    ("deep_sleep_hrs", "oura_sleep", "deep_sleep_duration", "Deep Sleep", "hrs", True, True),
    ("rem_sleep_hrs", "oura_sleep", "rem_sleep_duration", "REM Sleep", "hrs", True, True),
    ("efficiency", "oura_sleep", "efficiency", "Efficiency", "%", True, False),
    ("hr_lowest", "oura_sleep", "hr_lowest", "HR Lowest", "bpm", False, False),
    ("hr_average", "oura_sleep", "hr_average", "HR Average", "bpm", False, False),
    ("hrv_average", "oura_sleep", "hrv_average", "HRV Average", "ms", True, False),
    ("breath_average", "oura_sleep", "breath_average", "Breath Rate", "brpm", False, False),
    ("temperature_delta", "oura_sleep", "temperature_delta", "Temp Delta", "\u00b0C", None, False),
    ("readiness_score", "oura_readiness", "score", "Readiness", "pts", True, False),
    ("recovery_index", "oura_readiness", "recovery_index", "Recovery Index", "pts", True, False),
    ("hrv_balance", "oura_readiness", "hrv_balance", "HRV Balance", "pts", True, False),
    ("sleep_balance", "oura_readiness", "sleep_balance", "Sleep Balance", "pts", True, False),
    ("activity_score", "oura_activity", "score", "Activity Score", "pts", True, False),
    ("steps", "oura_activity", "steps", "Steps", "steps", True, False),
    ("active_calories", "oura_activity", "active_calories", "Active Cal", "kcal", True, False),
]

METRIC_KEYS = [m[0] for m in METRIC_DEFS]
METRIC_DISPLAY = {m[0]: m[3] for m in METRIC_DEFS}

# Cluster labelling by dominant metric groups
_CLUSTER_GROUPS = {
    "Cardiac stress": {"hr_lowest", "hr_average"},
    "Autonomic withdrawal": {"hrv_average", "hrv_balance", "recovery_index"},
    "Sleep disruption": {"sleep_score", "total_sleep_hrs", "deep_sleep_hrs", "rem_sleep_hrs",
                         "efficiency", "sleep_balance"},
    "Inflammatory": {"temperature_delta", "breath_average"},
}


def _embed(fig: go.Figure) -> str:
    """Embed a Plotly figure as inline HTML (no JS bundle)."""
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert hex color (#RRGGBB) to rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def _load_table(patient: PatientConfig, table: str) -> pd.DataFrame:
    """Load a table for one patient, indexed by date."""
    return load_patient_data(patient, table)


def _fallback_hr_from_sleep_periods(patient: PatientConfig, df_sleep: pd.DataFrame) -> pd.DataFrame:
    """Fill NULL hr_average/hr_lowest from oura_sleep_periods if available."""
    need_hr_avg = df_sleep["hr_average"].isna().any() if "hr_average" in df_sleep.columns else False
    need_hr_low = df_sleep["hr_lowest"].isna().any() if "hr_lowest" in df_sleep.columns else False

    if not need_hr_avg and not need_hr_low:
        return df_sleep

    try:
        sp = load_patient_data(patient, "oura_sleep_periods",
                               columns="day, lowest_heart_rate, average_heart_rate")
        if sp.empty:
            return df_sleep

        if need_hr_low and "lowest_heart_rate" in sp.columns:
            fill = sp["lowest_heart_rate"].reindex(df_sleep.index)
            df_sleep["hr_lowest"] = df_sleep["hr_lowest"].fillna(fill)
        if need_hr_avg and "average_heart_rate" in sp.columns:
            fill = sp["average_heart_rate"].reindex(df_sleep.index)
            df_sleep["hr_average"] = df_sleep["hr_average"].fillna(fill)
    except Exception as exc:
        logger.warning("HR fallback from sleep_periods failed for %s: %s", patient.patient_id, exc)

    return df_sleep


def load_all_metrics(patients: tuple[PatientConfig, PatientConfig]) -> dict[str, pd.DataFrame]:
    """Load all metrics for both patients into per-patient DataFrames aligned by date.

    Returns dict[patient_id -> DataFrame] where columns are metric keys and index is date.
    """
    result: dict[str, pd.DataFrame] = {}

    for p in patients:
        tables_needed = {m[1] for m in METRIC_DEFS}
        raw: dict[str, pd.DataFrame] = {}
        for tbl in tables_needed:
            raw[tbl] = _load_table(p, tbl)

        # P1 fallback for hr
        if p.patient_id == "henrik":
            if "oura_sleep" in raw and not raw["oura_sleep"].empty:
                raw["oura_sleep"] = _fallback_hr_from_sleep_periods(p, raw["oura_sleep"])

        # Collect each metric as a Series keyed by date
        series_list: list[pd.Series] = []
        for key, table, column, _disp, _unit, _hib, to_hours in METRIC_DEFS:
            df = raw.get(table, pd.DataFrame())
            if df.empty or column not in df.columns:
                series_list.append(pd.Series(dtype=float, name=key))
                continue
            s = df[column].rename(key)
            if to_hours:
                s = s / 3600.0
            series_list.append(s)

        combined = pd.concat(series_list, axis=1, sort=True)
        # Union of all dates across tables
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        result[p.patient_id] = combined

    return result


# ---------------------------------------------------------------------------
# 2. Per-Patient Z-Score Normalization
# ---------------------------------------------------------------------------

def compute_zscores(data: dict[str, pd.DataFrame]) -> tuple[
    dict[str, pd.DataFrame],  # z-score matrices
    dict[str, dict[str, dict[str, float]]],  # normalization params
]:
    """Z-score each metric per patient. If std ~ 0, set z=0."""
    z_data: dict[str, pd.DataFrame] = {}
    norm_params: dict[str, dict[str, dict[str, float]]] = {}

    for pid, df in data.items():
        z_df = pd.DataFrame(index=df.index)
        norm_params[pid] = {}

        for col in METRIC_KEYS:
            if col not in df.columns:
                z_df[col] = 0.0
                norm_params[pid][col] = {"mean": 0.0, "std": 0.0}
                continue

            s = df[col].astype(float)
            mean_val = s.mean()
            std_val = s.std()

            if pd.isna(mean_val):
                mean_val = 0.0
            if pd.isna(std_val) or std_val < 1e-9:
                std_val = 1.0
                z_df[col] = 0.0
            else:
                z_df[col] = (s - mean_val) / std_val

            norm_params[pid][col] = {"mean": float(mean_val), "std": float(std_val)}

        # Fill remaining NaN with 0 (missing data => no deviation)
        z_df = z_df.fillna(0.0)
        z_data[pid] = z_df

    return z_data, norm_params


# ---------------------------------------------------------------------------
# 3. Anomaly Detection
# ---------------------------------------------------------------------------

def _zscore_method(z_df: pd.DataFrame) -> pd.DataFrame:
    """Z-Score Threshold: anomalous if 3+ metrics |z|>2.0 OR composite>2.0."""
    abs_z = z_df.abs()
    n_extreme = (abs_z > 2.0).sum(axis=1)
    composite = abs_z.mean(axis=1)
    is_anomaly = (n_extreme >= 3) | (composite > 2.0)
    return pd.DataFrame({
        "zscore_anomaly": is_anomaly.astype(int),
        "zscore_n_extreme": n_extreme,
        "zscore_composite": composite,
    }, index=z_df.index)


def _iforest_method(z_df: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest: contamination=0.1, n_estimators=200."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    mat = z_df.values.copy()
    # Impute any remaining NaN with 0
    mat = np.nan_to_num(mat, nan=0.0)

    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(mat)

    clf = IsolationForest(contamination=0.1, n_estimators=200, random_state=42)
    labels = clf.fit_predict(mat_scaled)
    raw_scores = clf.decision_function(mat_scaled)

    # Normalize: higher = more anomalous (raw scores are lower = more anomalous)
    scores = -raw_scores
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros_like(scores)

    return pd.DataFrame({
        "iforest_anomaly": (labels == -1).astype(int),
        "iforest_score": scores,
    }, index=z_df.index)


def _percentile_method(z_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Percentile-Based: flag outside 5th/95th, anomalous if 3+ extreme."""
    extreme_count = pd.Series(0, index=raw_df.index, dtype=int)

    for col in METRIC_KEYS:
        if col not in raw_df.columns:
            continue
        s = raw_df[col].dropna()
        if len(s) < 5:
            continue
        p5, p95 = s.quantile(0.05), s.quantile(0.95)
        is_extreme = (raw_df[col] < p5) | (raw_df[col] > p95)
        extreme_count = extreme_count.add(is_extreme.astype(int), fill_value=0)

    return pd.DataFrame({
        "pctile_anomaly": (extreme_count >= 3).astype(int),
        "pctile_n_extreme": extreme_count,
    }, index=raw_df.index)


def detect_anomalies(
    z_data: dict[str, pd.DataFrame],
    raw_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Run all 3 methods and produce ensemble for each patient.

    Returns dict[patient_id -> DataFrame] with anomaly columns and ensemble score.
    """
    results: dict[str, pd.DataFrame] = {}

    for pid in z_data:
        z_df = z_data[pid]
        raw_df = raw_data[pid]

        # Align indices
        common_idx = z_df.index.intersection(raw_df.index)
        z_df = z_df.loc[common_idx]
        raw_df = raw_df.loc[common_idx]

        zs = _zscore_method(z_df)
        ifo = _iforest_method(z_df)
        pct = _percentile_method(z_df, raw_df)

        combined = pd.concat([zs, ifo, pct], axis=1)

        # Ensemble: anomaly if >=2 of 3 agree
        votes = (
            combined["zscore_anomaly"]
            + combined["iforest_anomaly"]
            + combined["pctile_anomaly"]
        )
        combined["ensemble_anomaly"] = (votes >= 2).astype(int)
        combined["method_agreement"] = votes

        # Weighted ensemble score
        combined["ensemble_score"] = (
            0.35 * combined["zscore_composite"]
            + 0.40 * combined["iforest_score"]
            + 0.25 * (combined["pctile_n_extreme"] / max(len(METRIC_KEYS), 1))
        )

        results[pid] = combined

    return results


# ---------------------------------------------------------------------------
# 4. Anomaly Fingerprinting
# ---------------------------------------------------------------------------

def compute_fingerprints(
    z_data: dict[str, pd.DataFrame],
    anomaly_data: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """For each patient: mean anomaly fingerprint, co-occurrence matrix, PCA."""
    from sklearn.decomposition import PCA

    fingerprints: dict[str, dict[str, Any]] = {}

    for pid in z_data:
        z_df = z_data[pid]
        anom = anomaly_data[pid]
        anomaly_mask = anom["ensemble_anomaly"] == 1
        anomaly_days = z_df.loc[z_df.index.isin(anom.index[anomaly_mask])]

        n_anomaly = len(anomaly_days)
        fp: dict[str, Any] = {"n_anomaly_days": n_anomaly}

        if n_anomaly < 2:
            fp["mean_profile"] = {k: 0.0 for k in METRIC_KEYS}
            fp["co_occurrence"] = {}
            fp["pca_pc1_loadings"] = {k: 0.0 for k in METRIC_KEYS}
            fp["pca_pc2_loadings"] = {k: 0.0 for k in METRIC_KEYS}
            fp["pca_variance_explained"] = [0.0, 0.0]
            fp["top_co_deviating_pairs"] = []
            fp["anomaly_z_matrix"] = anomaly_days
            fingerprints[pid] = fp
            continue

        # Mean fingerprint
        mean_profile = anomaly_days[METRIC_KEYS].mean()
        fp["mean_profile"] = {k: float(mean_profile.get(k, 0.0)) for k in METRIC_KEYS}

        # Co-occurrence matrix (correlation among anomaly days)
        co_occ = anomaly_days[METRIC_KEYS].corr()
        fp["co_occurrence"] = co_occ

        # Binary co-occurrence: when |z|>1.5 for metric A, how often is B also |z|>1.5?
        binary_extreme = (anomaly_days[METRIC_KEYS].abs() > 1.5).astype(float)
        if binary_extreme.sum().sum() > 0:
            binary_co = binary_extreme.T.dot(binary_extreme) / max(n_anomaly, 1)
        else:
            binary_co = pd.DataFrame(0.0, index=METRIC_KEYS, columns=METRIC_KEYS)
        fp["binary_co_occurrence"] = binary_co

        # Top co-deviating pairs
        pairs = []
        for i, m1 in enumerate(METRIC_KEYS):
            for m2 in METRIC_KEYS[i + 1:]:
                val = float(binary_co.loc[m1, m2]) if m1 in binary_co.index and m2 in binary_co.columns else 0.0
                pairs.append((m1, m2, val))
        pairs.sort(key=lambda x: x[2], reverse=True)
        fp["top_co_deviating_pairs"] = [(a, b, round(v, 3)) for a, b, v in pairs[:10]]

        # PCA on anomaly z-vectors
        mat = anomaly_days[METRIC_KEYS].values
        mat = np.nan_to_num(mat, nan=0.0)
        n_components = min(2, mat.shape[0], mat.shape[1])
        if n_components >= 1:
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(mat)
            fp["pca_pc1_loadings"] = {k: float(pca.components_[0, i]) for i, k in enumerate(METRIC_KEYS)}
            if n_components >= 2:
                fp["pca_pc2_loadings"] = {k: float(pca.components_[1, i]) for i, k in enumerate(METRIC_KEYS)}
                fp["pca_coords"] = pca.transform(mat)
            else:
                fp["pca_pc2_loadings"] = {k: 0.0 for k in METRIC_KEYS}
                coords_1d = pca.transform(mat)
                fp["pca_coords"] = np.column_stack([coords_1d, np.zeros(len(coords_1d))])
            fp["pca_variance_explained"] = [float(v) for v in pca.explained_variance_ratio_]
        else:
            fp["pca_pc1_loadings"] = {k: 0.0 for k in METRIC_KEYS}
            fp["pca_pc2_loadings"] = {k: 0.0 for k in METRIC_KEYS}
            fp["pca_variance_explained"] = [0.0, 0.0]
            fp["pca_coords"] = np.zeros((n_anomaly, 2))

        fp["anomaly_z_matrix"] = anomaly_days
        fingerprints[pid] = fp

    return fingerprints


# ---------------------------------------------------------------------------
# 5. Cluster Analysis of Bad Days
# ---------------------------------------------------------------------------

def _label_cluster(centroid: pd.Series) -> str:
    """Label a cluster by its dominant metric group."""
    abs_c = centroid.abs()
    best_label = "Multi-system decompensation"
    best_score = 0.0

    for label, metrics in _CLUSTER_GROUPS.items():
        present = [m for m in metrics if m in abs_c.index]
        if not present:
            continue
        score = abs_c[present].mean()
        if score > best_score:
            best_score = score
            best_label = label

    # Only label specifically if the top group stands out
    group_scores = {}
    for label, metrics in _CLUSTER_GROUPS.items():
        present = [m for m in metrics if m in abs_c.index]
        if present:
            group_scores[label] = abs_c[present].mean()

    if group_scores:
        sorted_scores = sorted(group_scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] > 1.5 * sorted_scores[1]:
            return best_label

    return "Multi-system decompensation"


def cluster_anomalies(
    z_data: dict[str, pd.DataFrame],
    anomaly_data: dict[str, pd.DataFrame],
    fingerprints: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """DBSCAN clustering on anomaly z-vectors per patient."""
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    clusters: dict[str, dict[str, Any]] = {}

    for pid in z_data:
        z_df = z_data[pid]
        anom = anomaly_data[pid]
        anomaly_mask = anom["ensemble_anomaly"] == 1
        anomaly_idx = anom.index[anomaly_mask]
        anomaly_days = z_df.loc[z_df.index.isin(anomaly_idx)]
        n_anom = len(anomaly_days)

        cl: dict[str, Any] = {"n_anomaly_days": n_anom}

        if n_anom < 5:
            cl["n_clusters"] = 0
            cl["silhouette_score"] = None
            cl["cluster_labels"] = pd.Series(dtype=int)
            cl["cluster_descriptions"] = []
            cl["skipped_reason"] = f"Only {n_anom} anomaly days (need >= 5)"
            clusters[pid] = cl
            continue

        mat = anomaly_days[METRIC_KEYS].values
        mat = np.nan_to_num(mat, nan=0.0)

        db = DBSCAN(eps=2.0, min_samples=2)
        labels = db.fit_predict(mat)
        label_series = pd.Series(labels, index=anomaly_days.index, name="cluster")

        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_cl = len(unique_labels)

        cl["n_clusters"] = n_cl
        cl["cluster_labels"] = label_series

        # Silhouette score
        if n_cl >= 2 and n_cl < n_anom:
            try:
                sil = silhouette_score(mat, labels)
                cl["silhouette_score"] = float(sil)
            except Exception:
                cl["silhouette_score"] = None
        else:
            cl["silhouette_score"] = None

        # Cluster descriptions
        descriptions = []
        for cl_id in sorted(unique_labels):
            mask_cl = labels == cl_id
            centroid = pd.Series(mat[mask_cl].mean(axis=0), index=METRIC_KEYS)
            label_name = _label_cluster(centroid)
            n_days = int(mask_cl.sum())
            top_metrics = centroid.abs().nlargest(3)
            descriptions.append({
                "cluster_id": int(cl_id),
                "label": label_name,
                "n_days": n_days,
                "top_metrics": {k: round(float(centroid[k]), 2) for k in top_metrics.index},
                "centroid": {k: round(float(v), 3) for k, v in centroid.items()},
            })

        # Noise cluster
        n_noise = int((labels == -1).sum())
        if n_noise > 0:
            descriptions.append({
                "cluster_id": -1,
                "label": "Unclustered (noise)",
                "n_days": n_noise,
                "top_metrics": {},
                "centroid": {},
            })

        cl["cluster_descriptions"] = descriptions
        clusters[pid] = cl

    return clusters


# ---------------------------------------------------------------------------
# 6. Cross-Patient Comparison
# ---------------------------------------------------------------------------

def cross_patient_comparison(
    fingerprints: dict[str, dict[str, Any]],
    anomaly_data: dict[str, pd.DataFrame],
    z_data: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Compute cross-patient similarity and comparison metrics."""
    pids = list(fingerprints.keys())
    comp: dict[str, Any] = {}

    if len(pids) < 2:
        return {"error": "Need 2 patients for comparison"}

    pid_a, pid_b = pids[0], pids[1]
    fp_a, fp_b = fingerprints[pid_a], fingerprints[pid_b]

    # Cosine similarity of mean fingerprint vectors
    vec_a = np.array([fp_a["mean_profile"].get(k, 0.0) for k in METRIC_KEYS])
    vec_b = np.array([fp_b["mean_profile"].get(k, 0.0) for k in METRIC_KEYS])
    if np.linalg.norm(vec_a) > 0 and np.linalg.norm(vec_b) > 0:
        comp["fingerprint_cosine_similarity"] = float(1.0 - cosine_distance(vec_a, vec_b))
    else:
        comp["fingerprint_cosine_similarity"] = 0.0

    # Cosine similarity of PC1 loading vectors
    pc1_a = np.array([fp_a["pca_pc1_loadings"].get(k, 0.0) for k in METRIC_KEYS])
    pc1_b = np.array([fp_b["pca_pc1_loadings"].get(k, 0.0) for k in METRIC_KEYS])
    if np.linalg.norm(pc1_a) > 0 and np.linalg.norm(pc1_b) > 0:
        comp["pca_axis_similarity"] = float(1.0 - cosine_distance(pc1_a, pc1_b))
    else:
        comp["pca_axis_similarity"] = 0.0

    # Metric rank comparison: rank by mean |z| on anomaly days
    rank_comp: dict[str, dict[str, int]] = {}
    for pid in pids:
        anom = anomaly_data[pid]
        z_df = z_data[pid]
        anomaly_mask = anom["ensemble_anomaly"] == 1
        anomaly_z = z_df.loc[z_df.index.isin(anom.index[anomaly_mask])]
        if anomaly_z.empty:
            rank_comp[pid] = {k: 0 for k in METRIC_KEYS}
            continue
        mean_abs_z = anomaly_z[METRIC_KEYS].abs().mean()
        ranks = mean_abs_z.rank(ascending=False, method="min").astype(int)
        rank_comp[pid] = {k: int(ranks.get(k, 0)) for k in METRIC_KEYS}
    comp["metric_rank_comparison"] = rank_comp

    # Mean |z| on anomaly days (for bar chart)
    mean_abs_z_vals: dict[str, dict[str, float]] = {}
    for pid in pids:
        anom = anomaly_data[pid]
        z_df = z_data[pid]
        anomaly_mask = anom["ensemble_anomaly"] == 1
        anomaly_z = z_df.loc[z_df.index.isin(anom.index[anomaly_mask])]
        if anomaly_z.empty:
            mean_abs_z_vals[pid] = {k: 0.0 for k in METRIC_KEYS}
        else:
            mean_abs_z_vals[pid] = {k: float(anomaly_z[k].abs().mean()) for k in METRIC_KEYS}
    comp["mean_abs_z_anomaly"] = mean_abs_z_vals

    # Temporal concentration: Gini coefficient of inter-anomaly gaps
    for pid in pids:
        anom = anomaly_data[pid]
        anomaly_dates = anom.index[anom["ensemble_anomaly"] == 1].sort_values()
        if len(anomaly_dates) >= 3:
            gaps = np.diff(anomaly_dates.values).astype("timedelta64[D]").astype(float)
            if gaps.sum() > 0:
                sorted_gaps = np.sort(gaps)
                n = len(sorted_gaps)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_gaps) - (n + 1) * np.sum(sorted_gaps)) / (n * np.sum(sorted_gaps))
                comp[f"temporal_gini_{pid}"] = float(gini)
            else:
                comp[f"temporal_gini_{pid}"] = 0.0
        else:
            comp[f"temporal_gini_{pid}"] = None

    # Severity distribution comparison (Mann-Whitney U)
    scores_a = anomaly_data[pid_a].loc[anomaly_data[pid_a]["ensemble_anomaly"] == 1, "ensemble_score"]
    scores_b = anomaly_data[pid_b].loc[anomaly_data[pid_b]["ensemble_anomaly"] == 1, "ensemble_score"]
    if len(scores_a) >= 3 and len(scores_b) >= 3:
        stat, p = scipy_stats.mannwhitneyu(scores_a.dropna(), scores_b.dropna(), alternative="two-sided")
        comp["severity_mannwhitney_stat"] = float(stat)
        comp["severity_mannwhitney_p"] = float(p)
    else:
        comp["severity_mannwhitney_stat"] = None
        comp["severity_mannwhitney_p"] = None

    # Narrative summary
    sim = comp["fingerprint_cosine_similarity"]
    if sim > 0.7:
        similarity_text = "highly similar anomaly signatures"
    elif sim > 0.3:
        similarity_text = "moderately similar anomaly signatures"
    else:
        similarity_text = "distinct anomaly signatures"

    n_a = fp_a["n_anomaly_days"]
    n_b = fp_b["n_anomaly_days"]
    comp["narrative_summary"] = (
        f"Patient 1 and Patient 2 show {similarity_text} "
        f"(cosine={sim:.2f}). Patient 1 has {n_a} anomaly days vs P2's {n_b}. "
        f"PCA axis similarity is {comp['pca_axis_similarity']:.2f}, suggesting "
        f"{'shared' if comp['pca_axis_similarity'] > 0.5 else 'different'} "
        f"dominant failure modes."
    )

    return comp


# ---------------------------------------------------------------------------
# 7. Top-10 Anomaly Tables
# ---------------------------------------------------------------------------

def _top_anomaly_table(
    anomaly_df: pd.DataFrame,
    z_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    cluster_labels: pd.Series | None,
    n: int = 10,
) -> list[dict[str, Any]]:
    """Extract top-N anomaly days by ensemble score."""
    anom_only = anomaly_df[anomaly_df["ensemble_anomaly"] == 1].copy()
    if anom_only.empty:
        return []
    anom_only = anom_only.sort_values("ensemble_score", ascending=False).head(n)

    rows = []
    for dt in anom_only.index:
        row: dict[str, Any] = {
            "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
            "ensemble_score": round(float(anom_only.loc[dt, "ensemble_score"]), 3),
            "methods_agreeing": int(anom_only.loc[dt, "method_agreement"]),
        }
        # Top deviating metrics
        if dt in z_df.index:
            z_row = z_df.loc[dt, METRIC_KEYS]
            top3 = z_row.abs().nlargest(3)
            row["top_deviating"] = [
                {"metric": METRIC_DISPLAY.get(k, k), "z": round(float(z_row[k]), 2)}
                for k in top3.index
            ]
        if cluster_labels is not None and dt in cluster_labels.index:
            row["cluster"] = int(cluster_labels.loc[dt])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

# --- Figure 1: Anomaly Timeline ---

def _fig_anomaly_timeline(
    anomaly_data: dict[str, pd.DataFrame],
    cluster_data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Two-row timeline: ensemble score over time, anomaly days colored by cluster."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            f"{patients[0].display_name} \u2014 Anomaly Timeline",
            f"{patients[1].display_name} \u2014 Anomaly Timeline",
        ],
    )

    cluster_colors = [ACCENT_RED, ACCENT_AMBER, ACCENT_PURPLE, ACCENT_CYAN,
                      ACCENT_ORANGE, ACCENT_PINK, ACCENT_BLUE, ACCENT_GREEN]
    noise_color = TEXT_TERTIARY

    for row_i, p in enumerate(patients, 1):
        pid = p.patient_id
        anom = anomaly_data.get(pid)
        if anom is None or anom.empty:
            continue

        # Background: all days ensemble score
        fig.add_trace(go.Scatter(
            x=anom.index, y=anom["ensemble_score"],
            mode="lines",
            line=dict(color=PATIENT_COLORS.get(pid, ACCENT_BLUE), width=1, shape="spline"),
            opacity=0.4,
            name=f"{p.display_name} score",
            legendgroup=pid,
            showlegend=(row_i == 1),
        ), row=row_i, col=1)

        # Anomaly days as markers
        anom_mask = anom["ensemble_anomaly"] == 1
        anom_days = anom[anom_mask]

        cl_info = cluster_data.get(pid, {})
        cl_labels = cl_info.get("cluster_labels", pd.Series(dtype=int))

        if not anom_days.empty:
            # Color by cluster
            colors = []
            for dt in anom_days.index:
                if dt in cl_labels.index:
                    cl_id = cl_labels.loc[dt]
                    if cl_id == -1:
                        colors.append(noise_color)
                    else:
                        colors.append(cluster_colors[cl_id % len(cluster_colors)])
                else:
                    colors.append(ACCENT_RED)

            fig.add_trace(go.Scatter(
                x=anom_days.index, y=anom_days["ensemble_score"],
                mode="markers",
                marker=dict(size=8, color=colors, line=dict(width=1, color=TEXT_PRIMARY)),
                name=f"{p.display_name} anomaly days",
                legendgroup=f"{pid}_anom",
                showlegend=(row_i == 1),
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.3f}<extra></extra>",
            ), row=row_i, col=1)

        # P1's Feb 9 event
        if pid == "henrik":
            event_ts = pd.Timestamp(KNOWN_EVENT_DATE)
            # plotly yref: row 1 = "y domain", row 2 = "y2 domain"
            y_ref = "y domain" if row_i == 1 else f"y{row_i} domain"
            fig.add_shape(
                type="line",
                x0=event_ts, x1=event_ts,
                y0=0, y1=1, yref=y_ref,
                line=dict(color=ACCENT_RED, width=2, dash="dash"),
                opacity=0.7,
                row=row_i, col=1,
            )
            fig.add_annotation(
                x=event_ts, y=1.05, yref=y_ref,
                text="Acute Event (Feb 9)",
                showarrow=False,
                font=dict(size=10, color=ACCENT_RED),
                row=row_i, col=1,
            )

    fig.update_yaxes(title_text="Ensemble Score", row=1, col=1)
    fig.update_yaxes(title_text="Ensemble Score", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_layout(
        height=600,
        title=dict(text="Anomaly Detection Timeline", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=60, r=20, t=70, b=50),
        hovermode="x unified",
    )
    return fig


# --- Figure 2: Radar Charts ---

def _fig_radar(
    fingerprints: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Side-by-side Scatterpolar: mean anomaly fingerprint."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=[p.display_name for p in patients],
    )

    # Use absolute z-scores for radar (direction irrelevant for "severity")
    categories = [METRIC_DISPLAY.get(k, k) for k in METRIC_KEYS]
    categories_closed = categories + [categories[0]]

    for col_i, p in enumerate(patients, 1):
        pid = p.patient_id
        fp = fingerprints.get(pid, {})
        profile = fp.get("mean_profile", {})
        vals = [abs(profile.get(k, 0.0)) for k in METRIC_KEYS]
        vals_closed = vals + [vals[0]]

        color = PATIENT_COLORS.get(pid, ACCENT_BLUE)
        fill_c = _hex_to_rgba(color, 0.15)

        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor=fill_c,
            line=dict(color=color, width=2),
            name=p.display_name,
            opacity=0.8,
        ), row=1, col=col_i)

    fig.update_layout(
        height=500,
        title=dict(text="Mean Anomaly Fingerprint (|z-score|)", font=dict(size=16)),
        margin=dict(l=80, r=80, t=80, b=40),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, None]),
            bgcolor=BG_SURFACE,
        ),
        polar2=dict(
            radialaxis=dict(visible=True, range=[0, None]),
            bgcolor=BG_SURFACE,
        ),
    )
    return fig


# --- Figure 3: Co-Deviation Heatmaps ---

def _fig_co_deviation(
    fingerprints: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Side-by-side annotated heatmaps of metric co-occurrence on anomaly days."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[p.display_name for p in patients],
        horizontal_spacing=0.12,
    )

    display_labels = [METRIC_DISPLAY.get(k, k) for k in METRIC_KEYS]

    for col_i, p in enumerate(patients, 1):
        pid = p.patient_id
        fp = fingerprints.get(pid, {})
        co_occ = fp.get("binary_co_occurrence", pd.DataFrame())

        if isinstance(co_occ, pd.DataFrame) and not co_occ.empty:
            mat = co_occ.reindex(index=METRIC_KEYS, columns=METRIC_KEYS, fill_value=0.0).values
        else:
            mat = np.zeros((len(METRIC_KEYS), len(METRIC_KEYS)))

        # Annotations
        annotations = []
        for i in range(len(METRIC_KEYS)):
            for j in range(len(METRIC_KEYS)):
                val = mat[i, j]
                if val > 0.01:
                    annotations.append(
                        dict(
                            text=f"{val:.2f}",
                            x=display_labels[j],
                            y=display_labels[i],
                            xref=f"x{col_i}" if col_i > 1 else "x",
                            yref=f"y{col_i}" if col_i > 1 else "y",
                            showarrow=False,
                            font=dict(size=7, color=TEXT_PRIMARY if val < 0.5 else BG_PRIMARY),
                        )
                    )

        fig.add_trace(go.Heatmap(
            z=mat,
            x=display_labels,
            y=display_labels,
            colorscale="Viridis",
            showscale=(col_i == 2),
            zmin=0, zmax=1,
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        ), row=1, col=col_i)

        fig.update_layout(annotations=fig.layout.annotations + tuple(annotations))

    fig.update_layout(
        height=650,
        title=dict(text="Co-Deviation Heatmaps (binary co-occurrence)", font=dict(size=16)),
        margin=dict(l=120, r=40, t=80, b=120),
    )
    return fig


# --- Figure 4: PCA Biplot ---

def _fig_pca_biplot(
    fingerprints: dict[str, dict[str, Any]],
    cluster_data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Anomaly days in PC1-PC2 space, colored by patient, shaped by cluster. Loading arrows."""
    from sklearn.decomposition import PCA

    fig = go.Figure()

    # Combine all anomaly z-vectors for a joint PCA
    all_mats = []
    all_pids = []
    all_cluster_ids = []

    for p in patients:
        pid = p.patient_id
        fp = fingerprints.get(pid, {})
        anom_z = fp.get("anomaly_z_matrix", pd.DataFrame())
        if isinstance(anom_z, pd.DataFrame) and not anom_z.empty:
            mat = anom_z[METRIC_KEYS].values
            mat = np.nan_to_num(mat, nan=0.0)
            all_mats.append(mat)
            all_pids.extend([pid] * len(mat))

            cl_info = cluster_data.get(pid, {})
            cl_labels = cl_info.get("cluster_labels", pd.Series(dtype=int))
            for dt in anom_z.index:
                if dt in cl_labels.index:
                    all_cluster_ids.append(int(cl_labels.loc[dt]))
                else:
                    all_cluster_ids.append(-1)

    if not all_mats:
        fig.update_layout(
            title="PCA Biplot (insufficient data)",
            height=500,
        )
        return fig

    combined_mat = np.vstack(all_mats)
    n_components = min(2, combined_mat.shape[0], combined_mat.shape[1])
    if n_components < 2:
        fig.update_layout(title="PCA Biplot (insufficient dimensions)", height=500)
        return fig

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(combined_mat)

    # Cluster shapes
    shape_map = {-1: "x", 0: "circle", 1: "square", 2: "diamond",
                 3: "triangle-up", 4: "star", 5: "hexagon"}

    for p in patients:
        pid = p.patient_id
        mask = [i for i, x in enumerate(all_pids) if x == pid]
        if not mask:
            continue
        px = coords[mask, 0]
        py = coords[mask, 1]
        cl_ids = [all_cluster_ids[i] for i in mask]
        symbols = [shape_map.get(c, "circle") for c in cl_ids]

        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode="markers",
            marker=dict(
                size=10,
                color=PATIENT_COLORS.get(pid, ACCENT_BLUE),
                symbol=symbols,
                line=dict(width=1, color=TEXT_PRIMARY),
                opacity=0.8,
            ),
            name=p.display_name,
            hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
        ))

    # Loading arrows (top 5 by magnitude)
    loadings = pca.components_
    loading_mag = np.sqrt(loadings[0] ** 2 + loadings[1] ** 2)
    top_idx = np.argsort(loading_mag)[-5:]
    arrow_scale = max(abs(coords).max(), 1) * 0.8

    for idx in top_idx:
        lx = loadings[0, idx] * arrow_scale
        ly = loadings[1, idx] * arrow_scale
        fig.add_annotation(
            x=lx, y=ly,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5, arrowwidth=1.5,
            arrowcolor=TEXT_SECONDARY,
        )
        fig.add_annotation(
            x=lx * 1.12, y=ly * 1.12,
            text=METRIC_DISPLAY.get(METRIC_KEYS[idx], METRIC_KEYS[idx]),
            showarrow=False,
            font=dict(size=9, color=TEXT_SECONDARY),
        )

    var_exp = pca.explained_variance_ratio_
    fig.update_layout(
        height=550,
        title=dict(text="PCA Biplot of Anomaly Days", font=dict(size=16)),
        xaxis_title=f"PC1 ({var_exp[0]:.1%} variance)",
        yaxis_title=f"PC2 ({var_exp[1]:.1%} variance)",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# --- Figure 5: Metric Rank Bars ---

def _fig_metric_rank_bars(
    comparison: dict[str, Any],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Grouped horizontal bar chart: mean |z| on anomaly days."""
    mean_abs = comparison.get("mean_abs_z_anomaly", {})
    if not mean_abs:
        fig = go.Figure()
        fig.update_layout(title="Metric Comparison (no data)", height=400)
        return fig

    pids = [p.patient_id for p in patients]
    # Sort by largest absolute difference between patients
    diffs = {}
    for k in METRIC_KEYS:
        vals = [mean_abs.get(pid, {}).get(k, 0.0) for pid in pids]
        diffs[k] = abs(vals[0] - vals[1]) if len(vals) >= 2 else 0.0
    sorted_keys = sorted(METRIC_KEYS, key=lambda k: diffs[k], reverse=True)
    sorted_labels = [METRIC_DISPLAY.get(k, k) for k in sorted_keys]

    fig = go.Figure()
    for p in patients:
        pid = p.patient_id
        vals = [mean_abs.get(pid, {}).get(k, 0.0) for k in sorted_keys]
        fig.add_trace(go.Bar(
            y=sorted_labels,
            x=vals,
            orientation="h",
            name=p.display_name,
            marker_color=PATIENT_COLORS.get(pid, ACCENT_BLUE),
            opacity=0.85,
        ))

    fig.update_layout(
        height=550,
        title=dict(text="Mean |z-score| on Anomaly Days by Metric", font=dict(size=16)),
        xaxis_title="Mean |z-score|",
        barmode="group",
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=140, r=20, t=60, b=50),
    )
    return fig


# --- Figure 6: Cluster Profiles ---

def _fig_cluster_profiles(
    cluster_data: dict[str, dict[str, Any]],
    patients: tuple[PatientConfig, PatientConfig],
) -> go.Figure:
    """Per cluster per patient: bar chart of centroid z-scores across metrics."""
    # Collect all clusters
    all_clusters = []
    for p in patients:
        pid = p.patient_id
        cl = cluster_data.get(pid, {})
        for desc in cl.get("cluster_descriptions", []):
            if desc["cluster_id"] == -1:
                continue
            all_clusters.append((p, desc))

    if not all_clusters:
        fig = go.Figure()
        fig.update_layout(title="Cluster Profiles (no clusters found)", height=400)
        return fig

    n_panels = len(all_clusters)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols

    subplot_titles = [
        f"{p.display_name}: {desc['label']} (n={desc['n_days']})"
        for p, desc in all_clusters
    ]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.15 / max(n_rows, 1),
    )

    display_labels = [METRIC_DISPLAY.get(k, k) for k in METRIC_KEYS]

    for i, (p, desc) in enumerate(all_clusters):
        r = (i // n_cols) + 1
        c = (i % n_cols) + 1
        centroid = desc.get("centroid", {})
        vals = [centroid.get(k, 0.0) for k in METRIC_KEYS]
        colors = [ACCENT_RED if v < -1 else ACCENT_AMBER if v < 0 else ACCENT_GREEN if v < 1 else ACCENT_CYAN
                  for v in vals]

        fig.add_trace(go.Bar(
            y=display_labels,
            x=vals,
            orientation="h",
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{y}: z=%{x:.2f}<extra></extra>",
        ), row=r, col=c)

    fig.update_layout(
        height=max(350, 250 * n_rows),
        title=dict(text="Cluster Centroid Profiles (z-scores)", font=dict(size=16)),
        margin=dict(l=140, r=20, t=70, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML Assembly
# ---------------------------------------------------------------------------

def _build_top10_html(top10: list[dict[str, Any]], patient_name: str) -> str:
    """Render a top-10 anomaly table as HTML."""
    if not top10:
        return f"<p>No anomaly days detected for {patient_name}.</p>"

    rows_html = ""
    for row in top10:
        deviating = row.get("top_deviating", [])
        dev_str = ", ".join(
            f"{d['metric']} (z={d['z']:+.1f})" for d in deviating
        )
        cl = row.get("cluster", "N/A")
        rows_html += (
            f"<tr>"
            f"<td>{row['date']}</td>"
            f"<td>{row['ensemble_score']:.3f}</td>"
            f"<td>{row['methods_agreeing']}/3</td>"
            f"<td>{dev_str}</td>"
            f"<td>{cl}</td>"
            f"</tr>"
        )

    return (
        f'<table class="odt-table">'
        f'<thead><tr>'
        f'<th>Date</th><th>Score</th><th>Agreement</th>'
        f'<th>Top Deviating Metrics</th><th>Cluster</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )


def _build_executive_summary(
    anomaly_data: dict[str, pd.DataFrame],
    fingerprints: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Build executive summary section."""
    cards = []
    for p in patients:
        pid = p.patient_id
        anom = anomaly_data.get(pid, pd.DataFrame())
        n_anom = int((anom.get("ensemble_anomaly", pd.Series()) == 1).sum()) if not anom.empty else 0
        n_total = len(anom)
        pct = (n_anom / n_total * 100) if n_total > 0 else 0.0
        status = "warning" if pct > 15 else "normal"
        cards.append(make_kpi_card(
            f"{p.display_name.split('(')[0].strip()} ANOMALY DAYS",
            n_anom,
            unit=f"/ {n_total}",
            status=status,
            detail=f"{pct:.1f}% of days flagged",
            decimals=0,
        ))

    sim = comparison.get("fingerprint_cosine_similarity", 0.0)
    sim_status = "info" if sim > 0.5 else "warning"
    cards.append(make_kpi_card(
        "FINGERPRINT SIMILARITY",
        sim,
        unit="cosine",
        status=sim_status,
        detail="1.0 = identical signatures",
        decimals=2,
    ))

    pca_sim = comparison.get("pca_axis_similarity", 0.0)
    cards.append(make_kpi_card(
        "PCA AXIS SIMILARITY",
        pca_sim,
        unit="cosine",
        status="info",
        detail="Dominant failure mode alignment",
        decimals=2,
    ))

    narrative = comparison.get("narrative_summary", "")
    return make_kpi_row(*cards) + f'<p style="margin-top:16px;color:{TEXT_SECONDARY};">{narrative}</p>'


def _build_patient_detection_section(
    pid: str,
    patient: PatientConfig,
    anomaly_df: pd.DataFrame,
    z_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    cluster_labels: pd.Series | None,
) -> str:
    """Build anomaly detection results section for one patient."""
    n_total = len(anomaly_df)
    n_anom = int((anomaly_df["ensemble_anomaly"] == 1).sum())
    n_zscore = int(anomaly_df["zscore_anomaly"].sum())
    n_iforest = int(anomaly_df["iforest_anomaly"].sum())
    n_pctile = int(anomaly_df["pctile_anomaly"].sum())

    cards = [
        make_kpi_card("TOTAL DAYS", n_total, decimals=0),
        make_kpi_card("ENSEMBLE ANOMALIES", n_anom, decimals=0,
                      status="warning" if n_anom > n_total * 0.15 else "normal"),
        make_kpi_card("Z-SCORE METHOD", n_zscore, decimals=0,
                      detail=f"{n_zscore / n_total * 100:.1f}%" if n_total > 0 else "N/A"),
        make_kpi_card("ISOLATION FOREST", n_iforest, decimals=0,
                      detail=f"{n_iforest / n_total * 100:.1f}%" if n_total > 0 else "N/A"),
        make_kpi_card("PERCENTILE METHOD", n_pctile, decimals=0,
                      detail=f"{n_pctile / n_total * 100:.1f}%" if n_total > 0 else "N/A"),
    ]

    top10 = _top_anomaly_table(anomaly_df, z_df, raw_df, cluster_labels)
    table_html = _build_top10_html(top10, patient.display_name)

    return make_kpi_row(*cards) + "<h3>Top 10 Anomaly Days</h3>" + table_html


def _build_clinical_implications(
    fingerprints: dict[str, dict[str, Any]],
    cluster_data: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Generate clinical implications narrative."""
    parts = []

    parts.append("<h3>Anomaly Signature Differences</h3>")
    for p in patients:
        pid = p.patient_id
        fp = fingerprints.get(pid, {})
        profile = fp.get("mean_profile", {})
        if profile:
            sorted_metrics = sorted(profile.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            items = ", ".join(
                f"{METRIC_DISPLAY.get(k, k)} (z={v:+.2f})" for k, v in sorted_metrics
            )
            parts.append(f"<p><strong>{p.display_name}</strong> typical bad day: {items}</p>")

    parts.append("<h3>Cluster Patterns</h3>")
    for p in patients:
        pid = p.patient_id
        cl = cluster_data.get(pid, {})
        descs = cl.get("cluster_descriptions", [])
        if descs:
            items = "; ".join(
                f"{d['label']} ({d['n_days']} days)" for d in descs if d["cluster_id"] != -1
            )
            if items:
                parts.append(f"<p><strong>{p.display_name}</strong>: {items}</p>")
        else:
            parts.append(f"<p><strong>{p.display_name}</strong>: Insufficient anomaly days for clustering.</p>")

    p_val = comparison.get("severity_mannwhitney_p")
    if p_val is not None:
        parts.append("<h3>Severity Comparison</h3>")
        sig = "significantly different" if p_val < 0.05 else "not significantly different"
        parts.append(
            f"<p>Anomaly severity distributions are {sig} "
            f"(Mann-Whitney U, p={format_p_value(p_val)}).</p>"
        )

    parts.append("<h3>Key Takeaways</h3>")
    parts.append("<ul>")
    sim = comparison.get("fingerprint_cosine_similarity", 0.0)
    if sim > 0.5:
        parts.append(
            "<li>Both patients show similar anomaly signatures, suggesting shared "
            "physiological stress pathways despite different conditions.</li>"
        )
    else:
        parts.append(
            "<li>The patients have distinct anomaly signatures, reflecting their different "
            "underlying conditions (HSCT vs stroke recovery).</li>"
        )
    parts.append(
        "<li>Z-score normalization removes raw-value bias, enabling "
        "meaningful comparison of 'how bad is bad' relative to each patient's baseline.</li>"
    )
    parts.append(
        "<li>Ensemble methods reduce false positives: only days flagged by "
        "multiple methods are considered truly anomalous.</li>"
    )
    parts.append("</ul>")

    return "\n".join(parts)


def _build_methodology() -> str:
    """Build methodology section content."""
    return (
        "<h3>Anomaly Detection Pipeline</h3>"
        "<ol>"
        "<li><strong>Data Collection:</strong> 17 metrics from Oura Ring across sleep, readiness, and activity domains.</li>"
        "<li><strong>Z-Score Normalization:</strong> Each metric normalized per patient (z = (x - mean) / std). "
        "This removes raw-value differences (e.g., HRV 9ms vs 43ms) and focuses on relative deviation.</li>"
        "<li><strong>Three Detection Methods:</strong>"
        "<ul>"
        "<li><em>Z-Score Threshold:</em> Day anomalous if 3+ metrics exceed |z| > 2.0 or composite magnitude > 2.0</li>"
        "<li><em>Isolation Forest:</em> Unsupervised tree-based method (contamination=0.1, n_estimators=200)</li>"
        "<li><em>Percentile-Based:</em> Flag values outside 5th/95th percentile, anomalous if 3+ metrics extreme</li>"
        "</ul></li>"
        "<li><strong>Ensemble:</strong> Day is anomaly if 2+ of 3 methods agree. "
        "Weighted score: z-score (0.35) + Isolation Forest (0.40) + percentile (0.25).</li>"
        "<li><strong>Fingerprinting:</strong> Mean z-vector of anomaly days reveals the 'typical bad day' signature.</li>"
        "<li><strong>PCA:</strong> Reduces dimensionality; PC1 loadings reveal the dominant failure mode.</li>"
        "<li><strong>DBSCAN Clustering:</strong> Groups anomaly days by similarity (eps=2.0, min_samples=2). "
        "Clusters labeled by dominant metric group.</li>"
        "</ol>"
        "<h3>Limitations</h3>"
        "<ul>"
        "<li>Patient 1 has ~81 days of data vs P2's ~518 days. Small sample size limits clustering reliability.</li>"
        "<li>Oura Ring consumer-grade sensors; not clinical-grade measurement.</li>"
        "<li>Temperature delta may show seasonal drift in P2's longer dataset.</li>"
        "<li>Isolation Forest contamination rate (10%) is assumed, not empirically derived.</li>"
        "<li>Cosine similarity treats all metrics equally; clinical weighting may differ.</li>"
        "</ul>"
    )


def build_html(
    raw_data: dict[str, pd.DataFrame],
    z_data: dict[str, pd.DataFrame],
    anomaly_data: dict[str, pd.DataFrame],
    fingerprints: dict[str, dict[str, Any]],
    cluster_data: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    patients: tuple[PatientConfig, PatientConfig],
) -> str:
    """Assemble full HTML report."""
    sections: list[str] = []

    # 1. Executive Summary
    sections.append(section_html_or_placeholder(
        "Executive Summary",
        lambda: make_section(
            "Executive Summary",
            _build_executive_summary(anomaly_data, fingerprints, comparison, patients),
            section_id="executive-summary",
        ),
    ))

    # 2-3. Per-patient detection results
    for p in patients:
        pid = p.patient_id
        anom_df = anomaly_data.get(pid, pd.DataFrame())
        z_df = z_data.get(pid, pd.DataFrame())
        raw_df = raw_data.get(pid, pd.DataFrame())
        cl_labels = cluster_data.get(pid, {}).get("cluster_labels", None)

        sections.append(section_html_or_placeholder(
            f"{p.display_name} Detection",
            lambda _p=p, _pid=pid, _anom=anom_df, _z=z_df, _raw=raw_df, _cl=cl_labels: make_section(
                f"{_p.display_name}: Anomaly Detection Results",
                _build_patient_detection_section(_pid, _p, _anom, _z, _raw, _cl),
                section_id=f"detection-{_pid}",
            ),
        ))

    # Anomaly Timeline (Fig 1)
    sections.append(section_html_or_placeholder(
        "Anomaly Timeline",
        lambda: make_section(
            "Anomaly Timeline",
            _embed(_fig_anomaly_timeline(anomaly_data, cluster_data, patients)),
            section_id="anomaly-timeline",
        ),
    ))

    # 4. Fingerprint Comparison (Fig 2 + Fig 3)
    sections.append(section_html_or_placeholder(
        "Fingerprint Comparison",
        lambda: make_section(
            "Anomaly Fingerprint Comparison",
            _embed(_fig_radar(fingerprints, patients))
            + _embed(_fig_co_deviation(fingerprints, patients)),
            section_id="fingerprint-comparison",
        ),
    ))

    # 5. Cluster Analysis (Fig 4 + Fig 6)
    sections.append(section_html_or_placeholder(
        "Cluster Analysis",
        lambda: make_section(
            "Cluster Analysis",
            _embed(_fig_pca_biplot(fingerprints, cluster_data, patients))
            + _embed(_fig_cluster_profiles(cluster_data, patients)),
            section_id="cluster-analysis",
        ),
    ))

    # Metric Rank Bars (Fig 5)
    sections.append(section_html_or_placeholder(
        "Metric Comparison",
        lambda: make_section(
            "Metric Severity Comparison",
            _embed(_fig_metric_rank_bars(comparison, patients)),
            section_id="metric-comparison",
        ),
    ))

    # 6. Clinical Implications
    sections.append(section_html_or_placeholder(
        "Clinical Implications",
        lambda: make_section(
            "Clinical Implications",
            _build_clinical_implications(fingerprints, cluster_data, comparison, patients),
            section_id="clinical-implications",
        ),
    ))

    # 7. Methodology
    sections.append(section_html_or_placeholder(
        "Methodology",
        lambda: make_section(
            "Methodology & Limitations",
            _build_methodology(),
            section_id="methodology",
        ),
    ))

    body = "\n".join(sections)

    return wrap_html(
        title="Anomaly Pattern Comparison",
        body_content=body,
        report_id="comp_anomalies",
        subtitle="Module 5: Anomaly Signature Analysis",
        header_meta="Patient 1 (post-HSCT) vs Patient 2 (post-Stroke)",
    )


# ---------------------------------------------------------------------------
# JSON Export
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return None  # Skip raw DataFrames
    if isinstance(obj, pd.Series):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (date, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def export_json(
    norm_params: dict[str, dict[str, dict[str, float]]],
    anomaly_data: dict[str, pd.DataFrame],
    z_data: dict[str, pd.DataFrame],
    raw_data: dict[str, pd.DataFrame],
    fingerprints: dict[str, dict[str, Any]],
    cluster_data: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    patients: tuple[PatientConfig, PatientConfig],
) -> None:
    """Write structured metrics JSON."""
    output: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "patients": {},
        "comparison": {},
    }

    for p in patients:
        pid = p.patient_id
        anom = anomaly_data.get(pid, pd.DataFrame())
        fp = fingerprints.get(pid, {})
        cl = cluster_data.get(pid, {})

        n_anom = int((anom.get("ensemble_anomaly", pd.Series()) == 1).sum()) if not anom.empty else 0
        top10 = _top_anomaly_table(
            anom, z_data.get(pid, pd.DataFrame()),
            raw_data.get(pid, pd.DataFrame()),
            cl.get("cluster_labels"),
        )

        # Methods agreement distribution
        if not anom.empty and "method_agreement" in anom.columns:
            anom_only = anom[anom["ensemble_anomaly"] == 1]
            agree_dist = anom_only["method_agreement"].value_counts().to_dict()
        else:
            agree_dist = {}

        patient_json: dict[str, Any] = {
            "normalization_params": norm_params.get(pid, {}),
            "anomaly_detection": {
                "n_anomaly_days": n_anom,
                "top_10_anomalies": top10,
                "methods_agreement": {str(k): int(v) for k, v in agree_dist.items()},
            },
            "fingerprint": {
                "mean_profile": fp.get("mean_profile", {}),
                "pca_pc1_loadings": fp.get("pca_pc1_loadings", {}),
                "pca_variance_explained": fp.get("pca_variance_explained", []),
                "top_co_deviating_pairs": fp.get("top_co_deviating_pairs", []),
            },
            "clusters": {
                "n_clusters": cl.get("n_clusters", 0),
                "silhouette_score": cl.get("silhouette_score"),
                "cluster_descriptions": cl.get("cluster_descriptions", []),
            },
        }
        output["patients"][pid] = patient_json

    output["comparison"] = {
        "fingerprint_cosine_similarity": comparison.get("fingerprint_cosine_similarity"),
        "pca_axis_similarity": comparison.get("pca_axis_similarity"),
        "metric_rank_comparison": comparison.get("metric_rank_comparison"),
        "severity_mannwhitney_p": comparison.get("severity_mannwhitney_p"),
        "narrative_summary": comparison.get("narrative_summary"),
    }

    output = _sanitize(output)

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("JSON metrics written to %s", JSON_OUTPUT)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    """Run comparative anomaly analysis pipeline."""
    logger.info("[1/8] Loading patient data...")
    patients = default_patients()
    if patients[1] is None:
        print("Skipping: mitch.db not found (second patient data not available)")
        return 0
    raw_data = load_all_metrics(patients)

    for pid, df in raw_data.items():
        logger.info("  %s: %d days, %d metrics", pid, len(df), len(df.columns))

    logger.info("[2/8] Computing z-scores...")
    z_data, norm_params = compute_zscores(raw_data)

    logger.info("[3/8] Running anomaly detection...")
    anomaly_data = detect_anomalies(z_data, raw_data)

    for pid, anom in anomaly_data.items():
        n = int((anom["ensemble_anomaly"] == 1).sum())
        logger.info("  %s: %d anomaly days", pid, n)

    logger.info("[4/8] Computing fingerprints...")
    fingerprints = compute_fingerprints(z_data, anomaly_data)

    logger.info("[5/8] Clustering anomaly days...")
    cluster_data = cluster_anomalies(z_data, anomaly_data, fingerprints)

    logger.info("[6/8] Cross-patient comparison...")
    comparison = cross_patient_comparison(fingerprints, anomaly_data, z_data)

    logger.info("[7/8] Generating HTML report...")
    html = build_html(
        raw_data, z_data, anomaly_data,
        fingerprints, cluster_data, comparison, patients,
    )
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", HTML_OUTPUT)

    logger.info("[8/8] Exporting JSON metrics...")
    export_json(
        norm_params, anomaly_data, z_data, raw_data,
        fingerprints, cluster_data, comparison, patients,
    )

    logger.info("Comparative anomaly analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
