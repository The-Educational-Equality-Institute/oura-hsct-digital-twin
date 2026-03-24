# Audit: Deployed GvHD Prediction Report Content

**URL:** https://digital-twin.theeducationalequalityinstitute.org/reports/gvhd_prediction_report
**Fetched:** 2026-03-24
**File size:** 180,025 bytes (HTML)
**Page title:** GvHD Prediction Model - Oura Digital Twin
**Robots meta:** `noindex, nofollow`

---

## 1. Patient Identifiers & Labels

| Field | Value on deployed page |
|-------|----------------------|
| Patient label | "Post-HSCT Patient" (in metadata line) |
| Named patient identifiers | NONE - no name, DOB, or personal identifiers found |
| References to "patient" | 94 occurrences (generic term only, e.g., "single-patient case study") |
| Organization credit | "The Educational Equality Institute" |
| GitHub link | https://github.com/theeducationalequalityinstitute/oura-digital-twin |
| License | "Open source under MIT License" |
| Copyright | "(c) 2026 The Educational Equality Institute" |

**Assessment:** No PHI leakage detected. Patient is identified only as "Post-HSCT Patient."

---

## 2. Clinical Constants & Dates

### Treatment Dates Displayed

| Date | Label | Where shown |
|------|-------|-------------|
| 2026-01-08 | Data start date | Subtitle, footer |
| 2026-03-23 | Data end date | Subtitle, footer |
| 2026-02-09 | Acute decompensation / validation event | Chart annotations ("Acute Event"), alert table, narrative text |
| 2026-03-16 | Ruxolitinib start | Subtitle, chart annotations ("Ruxolitinib"), footer ("March 16") |
| 2026-03-18 | HEV diagnosis | Subtitle, chart annotations ("HEV Dx"), context strip |
| 2026-03-24 15:56 | Report generation timestamp | Metadata line, footer |

**Note:** HSCT date (2023-11-23) is NOT explicitly shown on this page. The page says "Post-HSCT Patient" but does not display the transplant date.

### Date Range

- Total analysis window: 75 days (2026-01-08 to 2026-03-23)
- Post-ruxolitinib window: 8 days (stated in context strip and footer)
- Data contains 76 unique date strings spanning the full range

### Clinical Event Annotations on Charts

All 5 Plotly charts contain vertical reference lines for:
- **2026-02-09** (red line, #EF4444) - labeled "Acute Event"
- **2026-03-16** (green line, #10B981) - labeled "Ruxolitinib"
- **2026-03-18** (amber line, #F59E0B) - labeled "HEV Dx"

### CUSUM Change-Point Dates

Regime changes detected: **2026-01-31, 2026-02-11, 2026-03-02**

Also highlighted as rectangular shapes on temperature chart:
- 2026-01-29, 2026-01-31, 2026-02-09, 2026-03-02, 2026-03-14

---

## 3. Top-Level KPI Metrics

| KPI | Value | Status/Label | Detail |
|-----|-------|--------------|--------|
| Peak GVHD Composite | **72.9** | Critical | on 2026-03-02 |
| rSLDS Feb 9 State | **Active Flare** | Flare | rSLDS (Linderman ssm) |
| RED Alerts in +/-3d Window | **1/9** | Detected | "8 outside +/-3d event window, retrospective only" |
| Combined GVHD + BOS | **34.2** | Normal / LOW | -- |

---

## 4. All Metric Values Displayed

### 4.1 rSLDS State Distribution Table

| State | Days | Percentage |
|-------|------|-----------|
| Remission | 35 | 46.7% |
| Pre-flare | 1 | 1.3% |
| Active Flare | 8 | 10.7% |
| Recovery | 31 | 41.3% |

### 4.2 Alert History Table (9 total RED alerts)

| Date | Level | P(Pre-flare) | P(Flare) | Composite |
|------|-------|-------------|----------|-----------|
| 2026-01-08 | RED | 0.971 | 0.029 | 45.6 |
| 2026-01-09 | RED | 0.0 | 1.0 | 65.4 |
| 2026-01-10 | RED | 0.0 | 1.0 | 59.4 |
| 2026-01-11 | RED | 0.0 | 1.0 | 58.8 |
| 2026-01-12 | RED | 0.0 | 1.0 | 42.1 |
| 2026-01-13 | RED | 0.0 | 1.0 | 66.7 |
| 2026-01-14 | RED | 0.0 | 1.0 | 40.0 |
| 2026-01-22 | RED | 0.0 | 1.0 | 52.2 |
| 2026-02-09 | RED | 0.0 | 1.0 | 69.6 |

### 4.3 Feature Importance Table

| Feature | Importance | Correlation | Mutual Info | Cohen's d |
|---------|-----------|-------------|-------------|-----------|
| HRV Median (RMSSD) | 0.520 | -0.548 | 0.266 | -1.45 |
| Sleep Heart Rate | 0.503 | +0.594 | 0.124 | +1.46 |
| Lowest HR | 0.502 | +0.571 | 0.170 | +1.41 |
| Readiness Score | 0.353 | -0.410 | 0.079 | -0.89 |
| Temperature Deviation | 0.282 | +0.326 | 0.127 | +0.65 |
| Temperature Gradient | 0.209 | +0.255 | 0.064 | +0.53 |
| HRV Variability | 0.200 | -0.244 | 0.044 | -0.57 |

### 4.4 BOS Risk Integration Metrics

| Metric | Value |
|--------|-------|
| BOS Risk Score | 16.9 (LOW) |
| Combined Risk | 34.2 (LOW) |
| SpO2 Mean | 96.07% |
| SpO2 Trend | -0.0047 %/day |

### 4.5 Composite Score Weights

| Component | Weight |
|-----------|--------|
| Temperature | 25% |
| HRV | 20% |
| Resting HR | 15% |
| Sleep Fragmentation | 15% |
| SpO2 | 15% |
| Activity | 10% |

### 4.6 Feature Importance Scoring Weights

| Method | Weight |
|--------|--------|
| Point-biserial correlation | 30% |
| Mutual information | 30% |
| Composite correlation | 20% |
| Cohen's d effect size | 20% |

Binary target: composite score > 65th percentile.

---

## 5. Chart Descriptions & Data

### Chart 1: Temperature Fluctuation Analysis (3 subplots)

**Subplot 1: Temperature Deviation (Daily)**
- Traces: "Temp Deviation" (daily values)
- Reference lines: Fever threshold (+0.5 degC), Hypothermia (-0.5 degC)
- Baseline reference line (gray, #9CA3AF) at y=0
- Red shaded zone above +0.5, Blue shaded zone below -0.5
- Rectangular highlights at CUSUM change-point dates

**Subplot 2: 7-Day Temperature Variability (Rolling SD)**
- Trace: "7d Variability"

**Subplot 3: Night-to-Night Temperature Gradient**
- Trace: "Nightly Gradient"

All three subplots share event annotation lines for Acute Event, Ruxolitinib, HEV Dx.

### Chart 2: Multi-Stream GVHD Composite Score (2 subplots)

**Subplot 1: GVHD Composite Score (Daily + 7-Day Rolling)**
- Traces: "Daily Score", "7-Day Rolling Avg"
- Alert threshold line at y=65 (labeled "Alert (65)")
- Red shaded zone above 65

**Subplot 2: Component Breakdown (Stacked Weighted Contribution)**
- 6 stacked area traces:
  - "Temperature (25%)"
  - "HRV (20%)"
  - "Resting HR (15%)"
  - "Sleep Frag (15%)"
  - "SpO2 (15%)"
  - "Activity (10%)"

### Chart 3: rSLDS State Model (3 subplots)

**Subplot 1: rSLDS State Probabilities (4 states x 75 days)**
- 4 area traces: "Remission", "Pre-flare", "Active Flare", "Recovery"
- Shows probability distributions over time

**Subplot 2: Most Likely State (Viterbi Path)**
- 4 traces: "Viterbi: Remission", "Viterbi: Pre-flare", "Viterbi: Active Flare", "Viterbi: Recovery"

**Subplot 3: ELBO Convergence (Laplace-EM)**
- Trace: "ELBO" - shows optimization convergence

### Chart 4: Retrospective Alert Burden (2 subplots)

**Subplot 1: Early Warning Alert Timeline**
- Traces: "Composite Score", "RED Alert"
- Shows composite score with RED alert markers

**Subplot 2: Pre-flare Probability with Alert Thresholds**
- Traces: "P(Pre-flare)", "P(Active Flare)"
- Threshold lines: YELLOW at 0.3, RED at 0.5

### Chart 5: Predictive Feature Importance

- Bar chart (data corresponds to the feature importance table)
- 7 features ranked by combined importance score

---

## 6. Statistical Claims & Methodology Descriptions

### 6.1 Model Description
- **Model:** 4-state rSLDS (recurrent Switching Linear Dynamical System)
- **States:** Remission, Pre-flare, Active Flare, Recovery
- **Fitting method:** Laplace-EM (Linderman et al. 2017)
- **Observations:** composite score + temperature deviation + HRV (3 features)
- **Claim:** "Each discrete state governs linear dynamics in a continuous latent space, with recurrent transitions that depend on the latent state"

### 6.2 Alert Definitions
- **YELLOW alert:** P(pre-flare) > 0.3 for 3+ consecutive days
- **RED alert:** P(pre-flare) > 0.5 OR P(active flare) > 0.2

### 6.3 Validation Claim
- "Retrospective validation against Feb 9, 2026 acute decompensation"
- Result: "1/9 RED alerts fell within +/-3d of the event and 8 occurred outside that window"
- Context strip note: "retrospective only"

### 6.4 Feature Importance Methodology
- Binary target: composite score > 65th percentile
- Combination: point-biserial correlation (30%), mutual information (30%), composite correlation (20%), Cohen's d effect size (20%)
- Top feature ranking claim: "HRV Median (RMSSD), Sleep Heart Rate, Lowest HR"

### 6.5 Cohen's d Values (Effect Sizes)
- HRV Median: d = -1.45 (large effect)
- Sleep Heart Rate: d = +1.46 (large effect)
- Lowest HR: d = +1.41 (large effect)
- Readiness Score: d = -0.89 (large effect)
- Temperature Deviation: d = +0.65 (medium effect)
- Temperature Gradient: d = +0.53 (medium effect)
- HRV Variability: d = -0.57 (medium effect)

### 6.6 CUSUM Change-Point Detection
- Method: CUSUM (Cumulative Sum) change-point detection
- Result: "Regime changes detected: 2026-01-31, 2026-02-11, 2026-03-02"

### 6.7 Composite Score Normalization
- Each component z-scored against first 14 days baseline
- Normalized to 0-100 scale (higher = more GVHD-like)

---

## 7. Context Warnings & Disclaimers

### Context Strip (top of page, 3 items)
1. "Oura Ring Gen 4 sensor data - not clinical measurements"
2. "N=1 case study - not validated for clinical decisions"
3. **[AMBER/WARNING]** "HEV diagnosed Mar 18; interpret findings cautiously in this Day 8 post-ruxolitinib window"

### Clinical Context Narrative
> "Chronic GVHD affecting skin, liver, mouth (NIH 2014: moderate). Known acute decompensation on Feb 9, 2026 (validation target). rSLDS classified Feb 9 as Active Flare, while RED alerts were mostly off-window (8/9 outside +/-3d), so treat this as a retrospective state-classification signal. Top ranked features in this run: HRV Median (RMSSD), Sleep Heart Rate, Lowest HR. Temperature deviation contributes inflammatory context but is not the leading feature."

### Disclaimer (bottom of page)
> "This analysis is for research purposes only and should not be used as a sole basis for clinical decisions. N=1 retrospective case study - all detection metrics are descriptive, not inferential. Sensitivity/specificity cannot be computed from a single patient. Validation requires an external multi-patient cohort. Temperature deviation from consumer wearables has limited precision compared to clinical thermometry. All clinical decisions should be made in consultation with the treating hematologist."

### Footer Disclaimers
1. "All metrics derived from Oura Ring Gen 4 consumer wearable data. Not clinical-grade measurements."
2. "Single-patient case study (N=1). Not validated for clinical decision-making. Not a medical device."
3. "Data: January 8 - March 23, 2026. Post-intervention: 8 days (ruxolitinib, March 16)"
4. "Updated daily at 06:15 CET. Last generated: 2026-03-24 15:56"
5. "This project is not affiliated with, endorsed by, or sponsored by Oura Health Oy. Oura(R) is a registered trademark of Oura Health Oy."

---

## 8. Navigation / Cross-Links

The page includes a full navigation sidebar linking to all 11 reports:

| Report Name | Filename |
|-------------|----------|
| Full Analysis | oura_full_analysis.html |
| Biomarker Trends | composite_biomarkers.html |
| Sleep Analysis | advanced_sleep_analysis.html |
| Causal: Ruxolitinib | causal_inference_report.html |
| GvHD Prediction | gvhd_prediction_report.html |
| SpO2 & BOS | spo2_bos_screening.html |
| Advanced HRV | advanced_hrv_analysis.html |
| Digital Twin | digital_twin_report.html |
| Foundation Model | foundation_model_report.html |
| Anomaly Detection | anomaly_detection_report.html |
| 3D Dashboard | oura_3d_dashboard.html |
| About | roadmap.html#honest |
| Next Steps | roadmap.html#roadmap |

---

## 9. Technical Details

- **Plotly charts:** 5 interactive Plotly.js plots
- **Chart div IDs:** UUID-based (e.g., `4fccf25d-64de-4281-824e-da921a03b83d`)
- **Theme:** Dark mode (#0F1117 background)
- **Fonts:** Inter (loaded as WOFF2 web fonts, multiple weights)
- **CSS:** Inline, comprehensive design system with CSS custom properties
- **Print styles:** Included (hides nav, switches to white background)
- **Animations:** Fade-in, scale-in, gradient border spin on verdict banner
- **Accessibility:** Reduced-motion media query disables all animations

---

## 10. Summary of Key Values for Source Code Comparison

These are the critical numbers that should match the source script output:

| Metric | Deployed Value |
|--------|---------------|
| Peak GVHD Composite | 72.9 |
| Peak date | 2026-03-02 |
| Feb 9 state classification | Active Flare |
| Total RED alerts | 9 |
| RED alerts in +/-3d window | 1 |
| RED alerts outside window | 8 |
| Combined GVHD+BOS risk | 34.2 |
| BOS Risk Score | 16.9 |
| SpO2 Mean | 96.07% |
| SpO2 Trend | -0.0047 %/day |
| Days in Remission | 35 (46.7%) |
| Days in Pre-flare | 1 (1.3%) |
| Days in Active Flare | 8 (10.7%) |
| Days in Recovery | 31 (41.3%) |
| Top feature | HRV Median (RMSSD), importance 0.520 |
| Top Cohen's d | Sleep Heart Rate, d = +1.46 |
| CUSUM changes | 2026-01-31, 2026-02-11, 2026-03-02 |
| Composite alert threshold | 65 |
| Baseline window | First 14 days |
| Total analysis days | 75 |
| Data range | 2026-01-08 to 2026-03-23 |
| Ruxolitinib start | 2026-03-16 |
| HEV diagnosis | 2026-03-18 |
| Acute event (validation) | 2026-02-09 |
| Report generated | 2026-03-24 15:56 |
