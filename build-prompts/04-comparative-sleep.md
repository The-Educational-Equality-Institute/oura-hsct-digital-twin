# Build Task: analysis/analyze_comparative_sleep.py

## What to build

Create `analysis/analyze_comparative_sleep.py` — Module 3: Sleep Architecture as Health Signal. Compare deep/REM/light/awake percentages, efficiency, and timing between both patients as markers of recovery quality.

## Project location & output

- Script: `analysis/analyze_comparative_sleep.py`
- HTML: `reports/comparative_sleep_analysis.html`
- JSON: `reports/comparative_sleep_metrics.json`
- Report ID: `comp_sleep`

## Patient context

- **Henrik** (36yo): Post-HSCT. Avg sleep 5.75 hrs, efficiency ~79%, HRV 9ms. Expected: disrupted architecture (low REM%, high awake%), consistent with autonomic dysfunction.
- **Mitchell** (33yo): Post-stroke. Avg sleep 6.7 hrs, efficiency ~83%, HRV 43ms. Better overall but HRV declining.

## Database notes

- Sleep durations in `oura_sleep` are in SECONDS (divide by 3600 for hours)
- Architecture percentages: compute as `stage_duration / (total_sleep_duration + awake_time) * 100`
- Bedtime/wake times: `bedtime_start` and `bedtime_end` are ISO 8601 datetime strings
- `oura_readiness.sleep_balance` is a 0-100 SCORE, not a duration

## 6 Core Analyses

### 1. Sleep Architecture Breakdown
- Nightly deep%, REM%, light%, awake% for each patient
- Summary stats: mean, median, std, IQR per stage
- 7-day and 14-day rolling averages
- Weekday vs weekend breakdown
- Temporal evolution (first half vs second half of observation period)

### 2. Sleep Efficiency Trends
- Nightly efficiency + 7-day rolling
- Distribution: histogram, skewness
- % of nights below 75% (poor), below 85% (recommended), above 90% (good)
- Linear regression trend + significance

### 3. Sleep Timing Analysis
- Bedtime consistency: SD of bedtime hour (handle midnight crossing: 23:30=23.5, 00:30=24.5)
- Wake time consistency
- Social jet lag proxy: weekday vs weekend sleep midpoint difference
- Circadian regularity: CV of sleep midpoint

### 4. Inter-Patient Comparison
For each metric (deep%, REM%, efficiency, total hours, timing variability):
- Mann-Whitney U test
- Cohen's d effect size
- Cliff's delta (non-parametric effect size)
- Bootstrap 95% CI for median difference

### 5. Benchmark Comparison
Compare each patient against:
- **Population norms**: Deep 13-23%, REM 20-25%, Efficiency ≥85%, Total 7-9hrs (age 30-39)
- **Post-HSCT norms**: Deep 8-15%, REM 10-18%, Efficiency 70-82%
- **Post-stroke norms**: Deep 10-18%, REM 12-20%, Efficiency 72-85%
- Z-scores relative to each reference

### 6. Recovery Quality Indicators
- REM rebound: nights where REM% > mean+1SD following low-REM nights
- Deep sleep adequacy: % of nights meeting 13% minimum
- Sleep debt: cumulative deviation from 7h target
- Spearman correlation of efficiency with time (positive = improving)
- Architecture stability: CV per stage % (lower = more stable)

## Visualizations (6 figures)

1. **Stacked Area** — Two panels (one per patient): nightly deep/light/REM/awake % stacked to 100%. Population norm bands.
2. **Architecture Distributions** — 2×2 grid: one panel per stage. Overlapping violin plots for both patients. Norm ranges as shaded bands.
3. **Efficiency Comparison** — Top: dual line charts (nightly + 7d rolling). Bottom: overlapping KDE. Reference lines at 75% and 85%.
4. **Timing Analysis** — Scatter of bedtime hour over time for both patients. Box plots of sleep midpoint variability.
5. **Benchmark Radar** — 6 axes (deep%, REM%, efficiency, total hours, regularity, sleep balance score). Two traces. Outer ring = population norms.
6. **Recovery Trajectory** — Efficiency trend with linear fit, annotated slope/R²/Spearman. Henrik: vertical line at Rux start.

## HTML report sections

1. Executive Summary (key metric cards)
2. Sleep Architecture (fig 1 + 2)
3. Sleep Efficiency (fig 3)
4. Sleep Timing (fig 4)
5. Benchmark Comparison (fig 5 + table)
6. Recovery Trajectory (fig 6)
7. Statistical Comparison Table (all tests, effect sizes)
8. Clinical Interpretation (auto-generated bullet points)
9. Methodology & Limitations

## JSON structure

```json
{
  "meta": { "generated", "patients": { "henrik": {...}, "mitchell": {...} } },
  "architecture": { "henrik": { "deep_pct": { "mean", "median", "std" }, ... }, "mitchell": {...} },
  "efficiency": { "henrik": { "mean", "pct_below_75", "trend_slope" }, "mitchell": {...} },
  "timing": { "henrik": { "bedtime_mean_hour", "bedtime_sd_min", "social_jetlag_min" }, ... },
  "comparison": { "deep_pct": { "mann_whitney_U", "p_value", "cohens_d", "significant" }, ... },
  "benchmarks": { "henrik": { "deep_pct_vs_norm_zscore", ... }, "mitchell": {...} },
  "recovery_indicators": { "henrik": { "efficiency_trend_rho", "deep_adequacy_pct" }, ... }
}
```
