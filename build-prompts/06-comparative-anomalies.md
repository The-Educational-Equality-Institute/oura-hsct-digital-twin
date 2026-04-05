# Build Task: analysis/analyze_comparative_anomalies.py

## What to build

Create `analysis/analyze_comparative_anomalies.py` — Module 5: Anomaly Pattern Comparison. Compare how "bad days" manifest differently between Henrik and Mitchell. What is each patient's anomaly "signature"?

## Project location & output

- Script: `analysis/analyze_comparative_anomalies.py`
- HTML: `reports/comparative_anomaly_report.html`
- JSON: `reports/comparative_anomaly_metrics.json`
- Report ID: `comp_anomalies`

## Context

- Henrik had known acute event on 2026-02-09
- Mitchell's worst days concentrated in late 2021 (Nov-Dec)
- Raw comparison meaningless (HRV 9ms vs 43ms) — everything must be z-scored per patient

## Metrics to include (from all 3 tables)

From oura_sleep: score, total_sleep_duration (as hours), deep_sleep_duration (hours), rem_sleep_duration (hours), efficiency, hr_lowest, hr_average, hrv_average, breath_average, temperature_delta
From oura_readiness: score (as readiness_score), recovery_index, hrv_balance, sleep_balance
From oura_activity: score (as activity_score), steps, active_calories

Note: Henrik's oura_sleep may have NULL hr_average/hr_lowest — use oura_sleep_periods fallback.
Note: All oura_readiness fields are contributor scores 0-100.

## Core analyses

### 1. Per-Patient Z-Score Normalization
- For each metric: `z = (x - patient_mean) / patient_std`
- If std ≈ 0, set z-scores to 0
- Store normalization params (mean, std) for JSON output

### 2. Three Anomaly Detection Methods

**Method A: Z-Score Threshold**
- Day is anomalous if 3+ metrics exceed |z| > 2.0 OR composite magnitude (mean |z|) > 2.0

**Method B: Isolation Forest**
- `IsolationForest(contamination=0.1, random_state=42, n_estimators=200)`
- StandardScaler on z-score matrix
- Normalize scores so higher = more anomalous
- Defer sklearn imports inside function body

**Method C: Percentile-Based**
- Per metric: flag values outside 5th/95th percentiles
- Day anomalous if 3+ metrics in extreme percentiles

**Ensemble**: `is_anomaly` if ≥2 of 3 methods agree. Weighted ensemble score (z:0.35, iforest:0.40, percentile:0.25).

### 3. Anomaly Fingerprinting (the novel analysis)

For each patient:
- Extract z-score vector for each anomaly day = the "fingerprint"
- Mean fingerprint = "typical bad day signature"
- Co-occurrence matrix: correlation of z-scores among anomaly days only
- Binary co-occurrence: when metric A is extreme, how often is metric B?
- PCA on anomaly day z-vectors: PC1 loadings reveal the dominant "mode of failure"

### 4. Cluster Analysis of Bad Days

- DBSCAN (eps=2.0, min_samples=2) on anomaly day z-vectors
- If <5 anomaly days, skip clustering
- Label clusters by dominant metrics:
  - HR-dominated → "Cardiac stress type"
  - HRV-dominated → "Autonomic withdrawal type"
  - Sleep-dominated → "Sleep disruption type"
  - Temperature-dominated → "Inflammatory type"
  - Multi-metric → "Multi-system decompensation"

### 5. Cross-Patient Comparison

- Cosine similarity of mean fingerprint vectors
- Cosine similarity of PC1 loading vectors
- Metric rank comparison: rank metrics by mean |z| on anomaly days — show side by side
- Temporal concentration: do bad days bunch together?
- Anomaly severity distribution comparison (Mann-Whitney U on scores)

## Visualizations (6 figures)

1. **Anomaly Timeline** — Two rows (separate date ranges). Ensemble score over time, anomaly days colored by cluster type. Henrik's Feb 9 event marked with `add_shape` + `add_annotation`.
2. **Radar Charts** — Side-by-side `go.Scatterpolar`: mean anomaly fingerprint for each patient. Same axes, allowing visual comparison.
3. **Co-Deviation Heatmaps** — Side-by-side annotated heatmaps of metric co-occurrence.
4. **PCA Biplot** — Anomaly days in PC1-PC2 space, colored by patient, shaped by cluster. Loading arrows.
5. **Metric Rank Bars** — Grouped horizontal bar chart: mean |z| on anomaly days for each metric, Henrik vs Mitchell side by side, sorted by largest difference.
6. **Cluster Profiles** — Per cluster per patient: bar chart of centroid z-scores across metrics.

## HTML sections

1. Executive Summary
2. Henrik: Anomaly Detection Results + top 10 days table
3. Mitchell: Anomaly Detection Results + top 10 days table
4. Fingerprint Comparison (radar + heatmaps)
5. Cluster Analysis (PCA biplot + cluster profiles)
6. Clinical Implications
7. Methodology

## JSON structure

```json
{
  "generated_at": "...",
  "patients": {
    "henrik": {
      "normalization_params": { "metric": { "mean", "std" } },
      "anomaly_detection": { "n_anomaly_days", "top_10_anomalies", "methods_agreement" },
      "fingerprint": { "mean_profile": {...}, "pca_pc1_loadings": {...}, "top_co_deviating_pairs": [...] },
      "clusters": { "n_clusters", "silhouette_score", "cluster_descriptions": [...] }
    },
    "mitchell": { ... }
  },
  "comparison": {
    "fingerprint_cosine_similarity", "pca_axis_similarity", 
    "metric_rank_comparison", "severity_mannwhitney_p",
    "narrative_summary"
  }
}
```

## Notes

- Defer sklearn imports (IsolationForest, StandardScaler, DBSCAN, PCA, silhouette_score) inside function bodies
- Each major section wrapped in try/except with traceback.print_exc() for resilience
- Mitchell's 518 days spanning seasons — temperature_delta may have seasonal drift
- Henrik has only ~81 days — clustering may produce very few clusters, handle gracefully
