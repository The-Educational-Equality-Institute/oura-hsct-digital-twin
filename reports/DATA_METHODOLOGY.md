# Oura Digital Twin - Data Methodology and Limitations

**Generated:** 2026-03-23
**Patient:** Post-HSCT, 36M, MDS-AML, 26+ months post-transplant
**Device:** Oura Ring Gen 4
**Intervention:** Ruxolitinib 10 mg BID, started 2026-03-16

---

## Data Counting: Days vs Sleep Periods

The Oura ring records individual **sleep periods**, not calendar days. One calendar day can produce multiple sleep periods:

- A main overnight sleep session
- Daytime naps
- Interrupted sleep recorded as separate periods

**Current data (as of 2026-03-23):**

| Phase | Calendar days | Sleep periods | Difference |
|-------|--------------|---------------|------------|
| Pre-Ruxolitinib | 66 | 73 | +7 extra periods |
| Post-Ruxolitinib | 7 | 11 | +4 extra periods |
| **Total** | **73** | **84** | **+11 extra periods** |

The dashboard reports **both** numbers for transparency. Calendar days is the clinically relevant count. Sleep periods is what the statistical analyses use as sample size.

### Why it matters

When a report says "n=11" for the post-ruxolitinib period, that is 11 sleep periods across 7 calendar days. This inflates apparent sample size. All p-values in the dashboard are computed on sleep period counts, not unique days, because each period produces independent physiological measurements (HR, HRV, SpO2, temperature).

Whether multiple sleep periods within the same day should be treated as independent observations is debatable. For short naps, the autonomic profile differs from overnight sleep. For split overnight sessions, they may be correlated. We report both counts and leave interpretation to the clinician.

---

## Post-Intervention Sample Size

Ruxolitinib started 2026-03-16. As of 2026-03-23, we have **7 calendar days** of post-intervention data. This is a severe limitation:

- **CausalImpact (BSTS):** Typically requires 2-4 weeks post-intervention for reliable estimates. Our 7-day window means the posterior interval is wide. The temperature signal (p=0.014) passes despite this because the effect size is large relative to pre-intervention variance.
- **UKF state shifts:** The Mann-Whitney U test compares 71 pre-treatment observations to 8 post-treatment observations. Small post-treatment n limits statistical power.
- **rSLDS disease states:** The Hidden Markov model has limited data for state transition estimation in the post-intervention window.
- **Composite biomarkers:** Pre/post deltas are computed but should be treated as preliminary trends, not confirmed effects.

**Recommendation:** Re-run all analyses at day 14, 28, and 90 post-intervention for progressively more reliable estimates.

---

## Data Completeness

Not every sensor produces data every night. Completeness varies by metric:

| Metric | Available nights | Fraction | Reason for gaps |
|--------|-----------------|----------|-----------------|
| HRV (RMSSD) | 73/79 | 92.4% | Ring not worn or poor skin contact |
| Heart rate | 68/79 | 86.1% | Ring not worn |
| SpO2 | 59/79 | 74.7% | SpO2 sensor requires tight fit; more gaps |
| Temperature | 73/79 | 92.4% | Ring not worn |
| Sleep efficiency | 69/79 | 87.3% | Sleep detection failures |

The Kalman filter handles missing data by increasing uncertainty on days without observations. This is methodologically appropriate but means that estimates on gap days are predictions, not measurements.

---

## Statistical Methods

### CausalImpact (Bayesian Structural Time Series)
- Pre-intervention: 71 days (Jan 4 - Mar 15)
- Post-intervention: 8 days (Mar 16 - Mar 23)
- Counterfactual constructed from pre-intervention covariate relationships
- 95% credible intervals from posterior MCMC sampling
- P-values are one-sided tail probabilities

### Unscented Kalman Filter (UKF)
- 5 latent states estimated from 5 observed variables
- Process noise (Q) and observation noise (R) learned via EM algorithm (15 iterations)
- State shifts tested with Mann-Whitney U (non-parametric, appropriate for small post-treatment n)

### rSLDS (Recurrent Switching Linear Dynamical System)
- 4 disease states: Remission, Pre-flare, Active Flare, Recovery
- Viterbi path gives most likely state sequence
- State probabilities give uncertainty per day
- Model trained on full time series; post-intervention states are in-sample, not forecasted

### Anomaly Detection Ensemble
- 5 algorithms: Matrix Profile, Isolation Forest, LSTM Autoencoder, Statistical Process Control, tsfresh
- Ensemble score is normalized mean of individual scores
- Threshold set at 90th percentile of ensemble scores
- Feb 9 acute event score (0.959) used as validation

### Composite Biomarkers
- 6 indices computed daily from raw Oura metrics
- Each is a weighted combination of 3-5 raw variables
- Weights derived from clinical literature, not from this patient's data
- Pre/post comparison uses means and Mann-Whitney U

---

## Known Biases and Limitations

1. **Single-subject design.** All analyses are n=1. Population-level inference is not possible.

2. **No control group.** The CausalImpact counterfactual is the best available substitute, but it assumes the pre-intervention model would have continued unchanged. Any concurrent event (e.g., HEV diagnosis on March 18) could confound the ruxolitinib signal.

3. **HEV confound.** Hepatitis E was diagnosed 2026-03-18, two days after ruxolitinib start. The temperature response attributed to ruxolitinib could be partially or wholly due to HEV-related immune activation. These two signals cannot be cleanly separated with 7 days of data.

4. **Ring position and wear compliance.** The Oura ring is worn on one finger. Peripheral temperature, SpO2, and heart rate can vary with ring fit, hand position during sleep, and ambient temperature. We have no objective measure of wear compliance.

5. **Sleep period independence.** Multiple sleep periods on the same day are treated as independent observations. For naps vs overnight sleep, this may be reasonable. For split overnight sessions, autonomic measurements may be correlated.

6. **Circular analysis risk.** The composite biomarker weights were chosen partly based on early observations of this patient's data. The indices are therefore descriptive, not predictive. They should not be interpreted as validated clinical scores.

7. **Software versions.** Results depend on specific versions of scipy, statsmodels, hmmlearn, and plotly. Different versions may produce slightly different p-values due to numerical precision differences.

---

## Reproducibility

All analyses can be re-run from the raw database:

```bash
# Activate your Python environment, then:
cd oura-digital-twin
python run_all.py
```

Raw data source: `data/oura.db` (SQLite, symlink or copy)
Analysis scripts: `analysis/` (11 Python scripts)
JSON outputs: `reports/*.json` (machine-readable results)
HTML reports: `reports/*.html` (interactive visualizations)

The database is populated from the Oura API via `api/import_oura.py`. Re-importing will update the data and may change results.
