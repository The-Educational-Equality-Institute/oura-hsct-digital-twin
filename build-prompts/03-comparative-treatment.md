# Build Task: analysis/analyze_comparative_treatment.py

## What to build

Create `analysis/analyze_comparative_treatment.py` — Module 2: Treatment Response Detection. Detect inflection points and changepoints in both patients' biometric data. For Henrik we know the treatment date (Rux Mar 16); for Mitchell we want to discover any natural changepoints.

## Project location & output

- Script: `/home/henrik/projects/teei/oura-hsct-digital-twin/analysis/analyze_comparative_treatment.py`
- HTML: `reports/comparative_treatment_response.html`
- JSON: `reports/comparative_treatment_response.json`
- Report ID: `comp_treatment`

## Patient context

- **Henrik**: Known events — Acute episode 2026-02-09, Ruxolitinib start 2026-03-16, HEV diagnosis 2026-03-18. 81 days of data.
- **Mitchell**: No known treatment dates — discover changepoints automatically. 518 days spanning 5 years (with gaps).

## Imports

Same as Module 1, plus: `from config import TREATMENT_START, KNOWN_EVENT_DATE, HEV_DIAGNOSIS_DATE`

Also needs `ruptures` for PELT changepoint detection (already in the venv — used by existing `analyze_ruxolitinib_response.py` in helseoversikt).

## Core analyses

### 1. Four Changepoint Detection Methods

Apply to 6 metrics each: HRV, lowest HR, average HR, sleep efficiency, deep sleep hours, steps.

**Method A: PELT (Penalized Exact Linear Time)**
- Use `ruptures.Pelt(model="rbf")` 
- Henrik: test multiple penalties, select best that captures Rux transition
- Mitchell: use BIC-derived penalty (`2 * np.log(n) * variance`) for parsimony
- Interpolate NaN before fitting, standardize signal

**Method B: CUSUM (Cumulative Sum)**
- `cusum = np.cumsum(signal - np.mean(signal))`
- Detect inflections via second-derivative sign changes
- Filter by magnitude (≥0.5 SD)

**Method C: Bayesian Online Change Point Detection (BOCPD)**
- Normal-Gamma conjugate prior (Adams & MacKay 2007)
- Hazard rate: 1/30 for Henrik, 1/50 for Mitchell
- Threshold changepoint probability > 0.3

**Method D: Rolling Window Comparison**
- Adjacent 14-day windows
- Two-sample t-test + Cohen's d for each pair
- Flag where p < 0.01 AND |d| > 0.5

### 2. Henrik: Pre/Post Treatment Analysis

For each metric, split at TREATMENT_START (Mar 16):
- Pre/post means, medians, SDs, sample sizes
- Mann-Whitney U test (two-sided)
- Cohen's d with pooled SD
- Bootstrap 95% CI for difference in means (1000 iterations)
- Percentage change
- Three-period comparison: pre-acute vs post-acute-pre-rux vs post-rux
- Bonferroni correction for multiple comparisons

### 3. Mitchell: Automatic Changepoint Discovery

- Run all 4 methods across all metrics
- Build consensus map: for each date, count how many (method × metric) pairs flagged it within 3-day tolerance
- High-confidence changepoints: consensus ≥ 3
- Characterize each: which metrics shifted, direction, magnitude, sustained?
- Generate candidate event timeline with descriptions

### 4. Multi-Metric Convergence

- Per-metric z-scores relative to each patient's baseline
- Daily convergence score: count of metrics deviating beyond 1.5σ simultaneously
- Dates with convergence ≥ 3 = systemic shift events

## Visualizations

1. **Henrik annotated timelines** — Per metric: rolling mean + 95% CI + changepoint markers + vertical event annotations (Rux start, acute event, HEV). Use `add_shape` + `add_annotation`, NOT `add_vline`.
2. **Henrik pre/post stat cards** — Visual comparison boxes
3. **BOCPD probability charts** — Per metric, for both patients
4. **Mitchell discovered changepoints** — Timeline with candidate events highlighted
5. **Comparative distributions** — Side-by-side violin plots for overlap period
6. **Multi-metric convergence heatmap** — Days × metrics, colored by z-score deviation

## HTML sections

1. Executive Summary with key findings
2. Henrik: Treatment Response (annotated timelines, pre/post cards, PELT overlay, BOCPD)
3. Mitchell: Discovered Events (timeline, consensus table, yearly trends)
4. Comparative (distributions, overlapping timelines)
5. Methods Appendix

## JSON structure

```json
{
  "generated_at": "...",
  "patients": {
    "henrik": {
      "known_events": [...],
      "metrics": {
        "hrv_average": {
          "pre_treatment": { "mean", "median", "std", "n" },
          "post_treatment": { ... },
          "comparison": { "pct_change", "mann_whitney_p", "cohens_d", "bootstrap_ci_95", "bonferroni_p" },
          "changepoints": { "pelt": [...], "cusum": [...], "bocpd": [...], "rolling_window": [...] }
        }
      },
      "multi_metric_convergence": [...]
    },
    "mitchell": {
      "discovered_events": [{ "date", "consensus_score", "methods_detecting", "metrics_affected", "sustained" }],
      "metrics": { ... }
    }
  },
  "methods": { "pelt": {...}, "cusum": {...}, "bocpd": {...}, "rolling_window": {...} }
}
```

## Notes

- This is the most complex module (~1,500 lines estimated)
- NaN handling: interpolate before changepoint detection, use `min_periods` for rolling
- Mitchell's 5-year data has large gaps — handle gracefully
- Statistical rigor: report both corrected and uncorrected p-values
- If `ruptures` is not installed, gracefully skip PELT and log a warning
