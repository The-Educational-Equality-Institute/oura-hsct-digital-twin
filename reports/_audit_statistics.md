# Statistical Audit: analyze_oura_gvhd_predict.py

**Date:** 2026-03-24
**Auditor:** Claude Opus 4.6
**Script:** `/home/ovehe/projects/helseoversikt/oura-digital-twin/analysis/analyze_oura_gvhd_predict.py`
**Lines:** 2681

---

## Executive Summary

This script implements a GVHD flare prediction system using Oura Ring wearable data for a single post-HSCT patient (N=1 retrospective case study). The statistical methods are generally appropriate for the use case, with several strengths and some issues. The script itself is self-aware about its N=1 limitation, which is commendable.

**Overall Rating: GOOD with minor issues**

- No Mann-Whitney U tests are used (none needed)
- No paired vs unpaired test confusion (no hypothesis tests between groups)
- No multiple comparison correction needed (no family of hypothesis tests)
- Cohen's d has a minor formula issue
- CUSUM implementation is correct
- HMM/rSLDS are properly initialized with clinical priors and validated
- Rolling windows are correctly configured
- p-hacking risk is LOW (the script is explicitly descriptive, not inferential)
- Baseline/intervention periods are correctly defined

---

## 1. Mann-Whitney U Tests

**Finding: NOT USED - N/A**

The script does not use Mann-Whitney U tests anywhere. This is appropriate because the script is a single-patient longitudinal analysis, not a between-group comparison. The only statistical test used is `scipy_stats.pointbiserialr` in the feature importance section (line 1972), which is appropriate for correlating a binary variable with a continuous variable.

**Verdict: PASS**

---

## 2. Paired vs Unpaired Tests

**Finding: NOT APPLICABLE**

The script does not perform any paired or unpaired hypothesis tests. The pre-event vs post-event composite scores (lines 877-884) are compared descriptively (mean values only), not with a formal test. This is correct for an N=1 study where formal hypothesis testing would be inappropriate.

The ruxolitinib response assessment (lines 2197-2209) similarly reports pre/post means and delta without a formal test. Again, appropriate for N=1.

**Verdict: PASS**

---

## 3. Multiple Comparison Correction

**Finding: NOT NEEDED**

The script computes p-values only in one place: point-biserial correlation in `compute_feature_importance()` (line 1972). These p-values are computed for 17 features against a binary target.

**Issue:** The p-values ARE computed for 17 features (lines 1930-1948), but they are not used for any decision-making, thresholding, or significance claims. They are stored in the JSON output but not filtered or reported as "significant." The combined importance score (lines 1997-2002) uses absolute correlation magnitude, mutual information, composite correlation, and Cohen's d - it does NOT use p-values for ranking.

Since the p-values are not used for inference or significance claims, multiple comparison correction is not strictly needed. However, if these p-values were to be interpreted, Bonferroni or BH correction would be required for 17 comparisons.

**Recommendation:** Add a note to the JSON output that p-values are uncorrected for multiple comparisons, or apply BH correction since the infrastructure is already there.

**Verdict: PASS (marginal) - no correction needed because p-values are not used for inference**

---

## 4. Cohen's d Calculation

**Finding: MINOR FORMULA ISSUE**

Location: Lines 1989-1993

```python
cohens_d = (high_vals.mean() - low_vals.mean()) / (
    np.sqrt((low_vals.std() ** 2 + high_vals.std() ** 2) / 2) + 1e-6
)
```

**Analysis:** This uses the pooled standard deviation formula for Cohen's d, which is:

d = (M1 - M2) / sqrt((s1^2 + s2^2) / 2)

This is actually **Cohen's d with the average variance** pooling method, sometimes called Cohen's d_s. The classic pooled SD formula (for unequal group sizes) should weight by degrees of freedom:

s_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))

**However**, the formula used here is a common and acceptable simplification when group sizes are roughly similar. Since the binary target is the 65th percentile (approximately 35%/65% split), the group sizes are moderately unequal.

**Additional note:** `np.std()` and pandas `.std()` use different defaults. `numpy` defaults to ddof=0 (population SD), while pandas defaults to ddof=1 (sample SD). Here, `low_vals` and `high_vals` are numpy arrays extracted from pandas, so `.std()` uses numpy's ddof=0. For Cohen's d, ddof=1 (sample SD) is conventional. This introduces a slight bias that shrinks the denominator and inflates d, but the effect is small for the sample sizes involved (~25-50 days).

**Verdict: MINOR ISSUE - formula is an acceptable simplification but uses population SD (ddof=0) instead of sample SD (ddof=1)**

---

## 5. Confidence Intervals

**Finding: NOT COMPUTED**

The script does not compute confidence intervals for any estimates. For an N=1 descriptive case study, this is not necessarily wrong, but CIs around key estimates (composite score mean, pre/post differences) would strengthen the report.

**Verdict: ACCEPTABLE for N=1 case study, but CIs would add rigor**

---

## 6. Time Series Analysis (CUSUM Change-Point Detection)

**Finding: CORRECTLY IMPLEMENTED**

Location: Lines 521-544

```python
mu = np.mean(temp_vals[:14])  # Baseline from first 2 weeks
sigma = np.std(temp_vals[:14]) if np.std(temp_vals[:14]) > 0.01 else 0.1
k = 0.5 * sigma  # Allowance (slack)
h = 4.0 * sigma  # Decision threshold
```

**Analysis of CUSUM implementation:**

1. **Baseline estimation** (mu from first 14 days): Appropriate. Two weeks provides a reasonable in-control reference.

2. **Sigma estimation** (from first 14 days): Appropriate. Using the same reference period as the mean ensures consistency. The floor of 0.01 prevents division issues.

3. **Slack parameter k = 0.5*sigma**: Standard choice. Page (1954) recommends k = delta/2 where delta is the minimum shift to detect. Setting k = 0.5*sigma means the CUSUM is tuned to detect shifts of ~1 sigma, which is reasonable for temperature deviations.

4. **Threshold h = 4.0*sigma**: Conservative but reasonable. Higher h reduces false alarms at the cost of slower detection. In clinical settings, conservatism is appropriate. Standard values range from 3*sigma to 5*sigma.

5. **Reset on alarm** (lines 536-537): Correct. After a regime change is detected, both CUSUM statistics are reset to zero. This is the standard "Western Electric" approach for repeated change detection.

6. **Bilateral CUSUM**: Both positive (upward shift) and negative (downward shift) CUSUMs are tracked (lines 532-533). This is correct for detecting both fever and hypothermia.

**One consideration:** The CUSUM uses the first 14 days as baseline, which is the same period used for the composite score baseline. If the patient was already symptomatic during these 14 days, the baseline may be biased. This is a fundamental limitation of within-patient normalization, not a code bug.

**Verdict: PASS - correctly implemented bilateral CUSUM with standard parameters**

---

## 7. Hidden Markov Models / rSLDS

**Finding: WELL IMPLEMENTED WITH PROPER VALIDATION**

### 7a. rSLDS (Primary Model) - Lines 1071-1206

**Initialization:**
- Uses `ssm.SLDS` with K=4 states, D=3 observations, recurrent transitions, Gaussian dynamics, identity emissions
- Multiple random restarts (n_restarts=3) with different seeds, keeping best ELBO
- Seeds are deterministic (RSLDS_BASE_SEED + restart) for reproducibility

**Fitting:**
- Laplace-EM with 50 iterations (capped at 200)
- ELBO convergence validated: non-finite values cause restart to be discarded (line 1139)
- Emission variance floor (0.01) prevents singular covariance (lines 1162-1173)

**Validation:**
- Degenerate solution detection (all observations in one state) with automatic fallback (lines 1408-1422)
- State index range validation (lines 1424-1428)
- State reordering by severity (mean composite score) ensures consistent labeling (lines 1438-1454)
- Known event validation (Feb 9) against Viterbi path (lines 1486-1504)

### 7b. HMM Fallback - Lines 1209-1272

**Initialization:**
- Clinically-informed transition matrix (line 1240): Strong remission self-loop (0.80), high pre-flare-to-flare transition (0.25), reasonable recovery dynamics
- Start probabilities favor remission (0.6) which is clinically appropriate
- `init_params = "mc"` (line 1246): Only means and covariances learned from data; transition and start probabilities use the clinical priors. This is a good design choice.

**Fitting:**
- 200 iterations, tol=1e-5, full covariance
- Convergence monitoring via `model.monitor_`

**Issue:** The HMM uses `init_params="stmc"` first (line 1235) then overrides to `"mc"` (line 1246). The initial `init_params="stmc"` is immediately overridden, so it has no effect. The explicit setting of `startprob_` and `transmat_` followed by `init_params = "mc"` is correct: it means "initialize s and t from my values, learn m and c from data."

### 7c. Data Preprocessing for Models

- ffill/bfill imputation for missing values (line 1334): Reasonable for small gaps in daily data
- NaN threshold of 30% to skip model entirely (line 1276): Appropriate guard
- Standardization before fitting (lines 1355-1358): Correct practice for multi-feature models
- Minimum 21 observations required (line 1275): Reasonable for 4-state model with 3 features

**Verdict: PASS - well-designed model pipeline with appropriate guards and fallbacks**

---

## 8. Rolling Window Calculations

**Finding: CORRECTLY CONFIGURED**

Location: Lines 411-418

```python
daily["temp_var_7d"] = daily["temp_dev"].rolling(7, min_periods=3).std()
daily["hrv_7d"] = daily["hrv_median"].rolling(7, min_periods=3).mean()
daily["hr_7d"] = daily["sleep_hr"].rolling(7, min_periods=3).mean()
```

**Analysis:**
- **Window size = 7 days**: Appropriate for weekly pattern detection in daily biometric data. Smooths day-to-day noise while preserving multi-day trends.
- **min_periods = 3**: Allows computation with as few as 3 of 7 days present. This is a reasonable balance between data availability and reliability. The first 2 days of data will show NaN, and days 3-6 will use partial windows.
- **std() for temperature variability**: Correct choice for measuring dispersion.
- **mean() for HRV and HR**: Correct choice for trend smoothing.

The 7-day rolling composite (line 873) also uses min_periods=3:
```python
composite_7d = composite.rolling(7, min_periods=3).mean()
```

**Verdict: PASS**

---

## 9. P-Hacking Risk Assessment

**Finding: LOW RISK**

The script is explicitly designed as a descriptive N=1 case study, not an inferential analysis. Key safeguards:

1. **No significance claims**: The script never claims any result is "statistically significant" (the word does not appear in the code).

2. **Multiple p-values not filtered**: The 17 point-biserial p-values (feature importance) are stored but not thresholded or used for selection.

3. **Known event validation is retrospective**: The script honestly labels all validation as "retrospective" and notes that sensitivity/specificity cannot be computed from N=1 (lines 1720-1737).

4. **Composite score threshold is percentile-based**: The binary target for feature importance uses the 65th percentile (line 1954), which is data-adaptive but clearly stated. This is a form of "optimistic" threshold selection since it maximizes separation in the observed data.

5. **Alert thresholds are pre-specified**: YELLOW_PREFLARE_PROB=0.30, RED_PREFLARE_PROB=0.50, RED_FLARE_PROB=0.20 are fixed constants (lines 153-157), not tuned post-hoc.

6. **Disclaimers are prominent**: Lines 2497-2506 explicitly state "N=1 retrospective case study - all detection metrics are descriptive, not inferential."

**Potential concern:** The composite score weights (line 138-145) are manually chosen, not empirically derived. If these were tuned to optimize detection of the Feb 9 event, that would be circular. However, the weights appear to be based on clinical reasoning (temperature highest at 25% for GVHD inflammatory signature), not post-hoc optimization. The code does not iterate over weight combinations.

**Verdict: PASS - explicit N=1 framing and no significance claims**

---

## 10. Baseline vs Intervention Period Definitions

**Finding: CORRECTLY DEFINED**

From config.py:
- `TREATMENT_START = date(2026, 3, 16)` (Ruxolitinib start)
- `KNOWN_EVENT_DATE = "2026-02-09"` (acute episode)
- `DATA_START = date(2026, 1, 8)` (analysis window start)

**Composite score baseline** (lines 782-783):
```python
baseline_end = daily.index.min() + timedelta(days=14)
baseline = daily.loc[daily.index <= baseline_end]
```
This uses the first 14 days of the analysis window (Jan 8-22, 2026) as baseline. This is ~18 days before the Feb 9 acute event and ~67 days before ruxolitinib start. The baseline period is appropriately before both events.

**CUSUM baseline** (line 523): First 14 days, consistent with composite baseline.

**Pre/post ruxolitinib comparison** (lines 2197-2199):
```python
rux_date = pd.Timestamp(RUXOLITINIB_START)
pre_rux = composite[composite.index < rux_date]
post_rux = composite[composite.index >= rux_date]
```
Correctly splits on treatment date. The code appropriately flags when post_rux has <7 days as "Too early to assess" (line 2206).

**Pre-event window** (lines 548-553): 7 days before Feb 9, correctly defined.

**Alert validation window** (lines 1723-1726): +/-3 days around Feb 9, correctly defined for event-centered validation.

**Verdict: PASS - all temporal windows correctly anchored to clinical dates**

---

## Additional Findings

### A. Composite Score Normalization (Lines 788-864)

The z-scoring formula used is:
```python
components["temperature"] = np.clip((temp_abs - temp_bl_mean) / temp_bl_std * 25 + 50, 0, 100)
```

This transforms z-scores to a 0-100 scale where 50 = baseline mean. Each SD deviation adds/subtracts 25 points. The clipping to [0, 100] creates ceiling and floor effects. This is a reasonable ad-hoc normalization for visualization purposes, but it means that extreme values beyond ~2 SD are censored. For the composite scoring purpose (weighted sum), this is acceptable.

**Note:** Missing SpO2 days are filled with 50 (neutral score, line 812). This prevents SpO2 gaps from pulling the composite toward 0, which is correct behavior since missing data should not be interpreted as either good or bad.

### B. Sleep Fragmentation Index (Lines 426-463)

The fragmentation index = transitions / hours is a well-established metric. The implementation correctly:
- Counts phase transitions using `np.diff(phases) != 0`
- Converts epochs to hours (5-minute epochs)
- Takes worst-case (highest fragmentation) when multiple sleep periods exist per day

### C. Feature Importance Combined Score (Lines 1997-2002)

The combined importance score mixes different scales:
- abs(correlation): 0-1 range, weighted 30%
- MI (mutual information): 0-log(2) range (~0-0.69), weighted 30%
- abs(composite_corr): 0-1 range, weighted 20%
- min(abs(cohens_d)/2, 1): capped at 1, weighted 20%

The MI component is on a different scale than the others. For 2 classes and 5 bins, MI ranges from 0 to log(2) = 0.693. This means MI contributes at most 0.3 * 0.693 = 0.208, while correlation can contribute 0.3 * 1.0 = 0.3. This creates an implicit down-weighting of MI relative to correlation. This may be intentional (correlation is more interpretable) but should be documented.

### D. Mutual Information Implementation (Lines 2079-2103)

The custom MI calculation discretizes the continuous feature into 5 equal-width bins. This is a simple but valid approach. The formula is correct:

MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))

using natural log (nats). The smoothing constant (1e-10) prevents log(0). The `max(mi, 0.0)` at line 2103 ensures non-negative MI (rounding errors could produce tiny negatives).

**Note:** Equal-width binning (line 2083) is sensitive to outliers. Quantile-based binning would be more robust, but for an importance ranking (not precise estimation), this is acceptable.

### E. Point-Biserial Correlation (Line 1972)

`scipy_stats.pointbiserialr(t_vals, f_vals)` is used correctly: binary variable first, continuous second. This returns the same value as Pearson r between a binary and continuous variable but with an exact p-value for the point-biserial case.

### F. SpO2 Trend (Line 2138)

```python
"trend_slope": round(float(np.polyfit(range(len(spo2)), spo2.values, 1)[0]), 4),
```

This fits a linear trend using ordinary least squares. **Issue:** `np.polyfit` with `range(len(spo2))` uses integer indices, not actual dates. If there are gaps in the SpO2 data (missing days), the trend assumes evenly-spaced observations. Since the query filters `spo2_average > 0` (line 229), missing days are simply dropped, and the trend slope represents change-per-observation, not change-per-day. This could misrepresent the actual temporal trend.

**Recommendation:** Use actual date indices (days since start) instead of sequential integers for polyfit.

---

## Summary Table

| # | Check | Status | Severity |
|---|-------|--------|----------|
| 1 | Mann-Whitney U appropriateness | N/A (not used) | - |
| 2 | Paired vs unpaired tests | N/A (no hypothesis tests) | - |
| 3 | Multiple comparison correction | PASS (marginal) | Low |
| 4 | Cohen's d formula | MINOR ISSUE (ddof=0 vs ddof=1) | Low |
| 5 | Confidence intervals | NOT COMPUTED | Low |
| 6 | CUSUM change-point detection | PASS | - |
| 7 | HMM/rSLDS initialization & validation | PASS | - |
| 8 | Rolling window calculations | PASS | - |
| 9 | P-hacking risk | LOW | - |
| 10 | Baseline vs intervention periods | PASS | - |
| A | Composite score normalization | PASS | - |
| B | Sleep fragmentation index | PASS | - |
| C | Feature importance scaling | NOTE (MI scale mismatch) | Low |
| D | Mutual information implementation | PASS | - |
| E | Point-biserial correlation | PASS | - |
| F | SpO2 trend slope | MINOR ISSUE (index vs date) | Low |

---

## Recommended Fixes (Priority Order)

1. **Cohen's d (Low):** Change `low_vals.std()` and `high_vals.std()` to use `ddof=1` for sample standard deviation, or use the full pooled formula weighted by degrees of freedom.

2. **SpO2 trend slope (Low):** Replace `range(len(spo2))` with actual date offsets (days since first reading) to handle gaps correctly.

3. **Multiple comparison note (Low):** Add a field to the JSON output noting that feature importance p-values are uncorrected for 17 comparisons.

4. **MI scaling (Low):** Consider normalizing MI to [0,1] by dividing by log(2) (theoretical maximum for binary target), so all four components of the importance score operate on comparable scales.

None of these issues affect the validity of the conclusions, which are explicitly framed as descriptive N=1 observations rather than inferential claims.
