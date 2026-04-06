# Build Task: analysis/analyze_comparative_coupling.py

## What to build

Create `analysis/analyze_comparative_coupling.py` — Module 4: Activity-Recovery Coupling. Analyze whether more activity on day N predicts better or worse sleep/HRV on day N+1, and whether this relationship differs between patients.

## Project location & output

- Script: `analysis/analyze_comparative_coupling.py`
- HTML: `reports/comparative_activity_recovery_coupling.html`
- JSON: `reports/comparative_activity_recovery_coupling.json`
- Report ID: `comp_coupling`

## Clinical hypothesis

Henrik's compromised recovery physiology (post-HSCT, HRV 9ms) may show negative or absent coupling (activity hurts next-day recovery), while Mitchell's healthier system (post-stroke but HRV 43ms) shows positive or neutral coupling.

## Key data facts

- Henrik: ~2700 steps/day, ~6 min active/day, `high_activity_time` nearly always 0
- Mitchell: ~10000 steps/day, ~30 min active/day, much more variable
- Henrik's `oura_sleep` has NULL for hr_average/hr_lowest — must use `oura_sleep_periods` (type='long_sleep')
- `oura_readiness.resting_heart_rate` is a SCORE (0-100), not bpm
- Henrik has ~77 usable lag-1 pairs (small N — use non-parametric tests, relaxed p<0.10)

## Activity predictor metrics (day N)

steps, active_calories, medium_activity_time, high_activity_time, activity score, composite_activity (z-scored mean of steps + active_cal + medium_time)

## Recovery outcome metrics (day N+1)

hrv_average, total_sleep_duration (hrs), efficiency, deep_sleep_duration, rem_sleep_duration, average_heart_rate (from sleep_periods), lowest_heart_rate, sleep score, readiness score, recovery_index

## Core analyses

### 1. Lag Correlation (primary)
- Spearman correlation between activity[N] and recovery[N+1] for all metric pairs
- Also compute at lag 0 (same day) and lag 2 for context
- Fisher z-transform for CI around Spearman r

### 2. Cross-Correlation Functions
- For key pairs (steps→HRV, steps→readiness, composite→HRV): compute at lags -3 to +3
- Negative lag = recovery leads activity (reverse causality check)
- Positive lag = activity leads recovery (causal direction of interest)

### 3. Dose-Response Analysis
- Bin activity days into quantiles (quartiles if N≥40, tertiles if N≥20, median split if N<20)
- Compare next-day recovery metric means across bins
- Kruskal-Wallis H test (non-parametric ANOVA)
- Jonckheere-Terpstra or linear trend test
- Bootstrap 95% CI for each bin's mean

### 4. Regression
- OLS: recovery[N+1] ~ activity[N] for primary pairs
- Report: slope, intercept, R², p-value, standardized beta
- Standardized beta enables cross-patient comparison

### 5. Full Correlation Matrix
- All activity metrics × all recovery metrics at lag 1
- Return (r_matrix, p_matrix) DataFrames

### 6. Clinical Assessment
- Per-patient: coupling_direction (positive/negative/absent), strength (strong/moderate/weak), recovery_capacity_rating
- Logic: positive if r > 0.15 and p < 0.10; negative if r < -0.15 and p < 0.10; absent otherwise

### 7. Cross-Patient Comparison
- Fisher r-to-z test: is Henrik's coupling significantly different from Mitchell's?
- Compare dose-response slopes
- Narrative interpretation

## Visualizations (6 figures)

1. **Dual Scatter** — `make_subplots(1, 2)`: activity[N] vs recovery[N+1] for both patients with OLS line and confidence band. Key pairs: steps→HRV, steps→readiness.
2. **Cross-Correlation Bars** — Grouped bar chart, x=lag, y=r. Two groups per lag (Henrik, Mitchell). Stars for significance. Dashed line at r=0.
3. **Dose-Response** — Grouped bar chart: activity bins on x, mean next-day recovery on y, with error bars. Two groups per bin.
4. **Correlation Heatmap (Henrik)** — Activity rows × recovery columns, diverging RdBu, annotated with r and * for significance.
5. **Correlation Heatmap (Mitchell)** — Same layout for comparison.
6. **Lagged Heatmap** — Side-by-side: lags on y-axis, recovery metrics on x-axis. Shows how correlation evolves across lags.

## HTML sections

1. KPI row (Henrik coupling r, Mitch coupling r, difference p-value, recovery capacity ratings)
2. Executive Summary
3. Scatter Plots
4. Cross-Correlation Functions
5. Dose-Response Analysis
6. Correlation Heatmaps
7. Clinical Interpretation (per-patient + comparative narrative + caveats about small N)

## JSON structure — see detailed plan (lag1_correlations, cross_correlations, dose_response, regression, coupling_assessment per patient, plus comparison with Fisher z-tests)

## Notes

- Use Spearman throughout (robust to small N and non-normality)
- Adaptive binning based on sample size
- Henrik's small N (~77 pairs) means many correlations won't reach p<0.05 — report effect sizes alongside p-values
- Guard against division by zero in composite activity z-scoring
