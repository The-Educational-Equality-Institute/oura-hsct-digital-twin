# Clinical Claims Audit: analyze_oura_gvhd_predict.py

**Auditor:** Claude Opus 4.6 (automated clinical review)
**Date:** 2026-03-24
**Script:** `analysis/analyze_oura_gvhd_predict.py`
**Patient:** 36yo male, MDS-AML, allo-HSCT Nov 2023, 26+ months post-transplant
**Mutations:** EZH2 V662A (VUS), DNMT3A R882H, SETBP1 G870S, PTPN11 E76Q, IDH2 R140Q. NPM1 NOT mutated.

---

## 1. GvHD Grading Thresholds

### Claim in script (line 2326):
> "NIH 2014 consensus: moderate; clinically under-graded"

### Assessment: PARTIALLY CORRECT, NEEDS NUANCE

The NIH 2014 consensus criteria (Jagasia MH et al., BBMT 2015, PMID 25529383) define chronic GVHD severity as:

- **Mild:** 1-2 organs, each score 1 (no lung involvement)
- **Moderate:** 3+ organs with score 1, OR any organ with score 2, OR lung score 1
- **Severe:** Any organ with score 3, OR lung score 2+

The script claims "10+ organs" are affected: skin, liver, mouth, eyes, lungs (BOS), heart, brain, GI tract, musculoskeletal, fascia. Under the NIH 2014 system, the specific organ scoring rubric applies to 8 defined organ categories (skin, mouth, eyes, GI, liver, lungs, joints/fascia, genital tract). Heart, brain, and musculoskeletal are not standard NIH-scored organs for cGVHD grading. The NIH system uses "other indicators" for atypical manifestations.

**Issue:** If the patient genuinely has lung score >= 1 (BOS) plus multiple organs with score >= 2, this would be **moderate or severe**, not mild. The claim of "clinically under-graded" is a reasonable assertion if official records say "mild" while lung involvement (BOS) alone pushes to at least moderate. However:

- The "10+ organs" count is inflated by including non-standard NIH organs (heart, brain).
- The NIH system scores 8 organ categories, not individual sub-organs.
- **Recommendation:** Rephrase to "involvement across 7+ NIH organ categories plus atypical sites (heart, CNS)" rather than "10+ organs."

**VERDICT: Claim defensible but imprecise. The count of "10+" is non-standard.**

---

## 2. HRV Reference Ranges

### Claims in config.py (lines 30-45):

| Constant | Value | Claimed Source | Audit |
|----------|-------|----------------|-------|
| `ESC_RMSSD_DEFICIENCY` | 15 ms | "Kleiger 1987, Bigger 1992" | SEE BELOW |
| `POPULATION_RMSSD_MEDIAN` | 49 ms | "Nunan 2010" | SEE BELOW |
| `NORM_RMSSD_P25` | 36 ms | "Nunan 2010" | SEE BELOW |
| `NORM_RMSSD_P75` | 72 ms | "Nunan 2010" | SEE BELOW |
| `HSCT_RMSSD_RANGE` | (25, 40) ms | "no validated literature; clinical estimate" | HONEST |

### 2a. ESC_RMSSD_DEFICIENCY = 15 ms (Kleiger 1987, Bigger 1992)

**ISSUE: ATTRIBUTION ERROR**

Kleiger 1987 (PMID 3812275, Am J Cardiol 59:256-62) studied SDNN (standard deviation of all NN intervals over 24 hours), NOT RMSSD. The famous finding was SDNN < 50 ms associated with 5.3x mortality risk after MI. Kleiger 1987 does NOT define an RMSSD threshold of 15 ms.

Bigger 1992 (PMID 1728446, Circulation 85:164-71) studied frequency-domain measures (VLF, LF, HF power) and mortality after MI. This paper also does NOT define RMSSD < 15 ms as a specific threshold.

The 15 ms RMSSD threshold likely derives from the ESC/NASPE Task Force 1996 (Circulation 93:1043-65), which established HRV measurement standards, or from clinical convention where RMSSD < 15-20 ms on short-term recording is considered severely depressed parasympathetic tone. However, there is no single paper that defines precisely "RMSSD < 15 ms = severe autonomic deficiency."

**The threshold is clinically reasonable but the attribution to Kleiger 1987 / Bigger 1992 is incorrect.** These papers studied SDNN and frequency-domain measures, not RMSSD.

**Recommendation:** Change attribution to "ESC/NASPE Task Force 1996; clinical convention" or "Shaffer & Ginsberg 2017" who summarize that RMSSD < 20 ms indicates severely depressed vagal tone.

**VERDICT: Threshold clinically reasonable. Attribution WRONG - Kleiger/Bigger did not define this RMSSD cutoff.**

### 2b. Population RMSSD norms (Nunan 2010)

**Nunan D et al., Pacing Clin Electrophysiol 2010;33:1407-17** - "A quantitative systematic review of normal values for short-term heart rate variability in healthy adults."

The claimed values:
- Median RMSSD: 49 ms -- **PLAUSIBLE but uncertain.** Nunan 2010 reported pooled means from multiple studies of 5-minute supine recordings. The pooled RMSSD mean was approximately 42 ms (range across studies: ~24-65 ms). The value of 49 ms may correspond to a subset or may be slightly inflated. The median from the pooled data would depend on study weighting.
- P25 = 36 ms, P75 = 72 ms -- These are described as "young adults IQR" which is reasonable. However, Nunan 2010 primarily reported study-level means, not individual-level percentiles. The IQR values may come from Shaffer & Ginsberg 2017 or from individual studies cited within the meta-analysis.

**Critical caveat (correctly noted in config.py):** These are from short-term 5-minute clinical ECG recordings. Oura Ring PPG-derived RMSSD during sleep will differ:
- Nocturnal values trend higher due to parasympathetic dominance during sleep
- PPG-derived RMSSD has ~7-10% MAPE vs ECG (config.py correctly notes CCC 0.91)

The config file honestly notes: "Nocturnal values trend higher than resting 5-min recordings due to parasympathetic dominance."

**VERDICT: Values approximately correct but may be slightly off from the actual Nunan 2010 pooled statistics. The comparison to nocturnal PPG data is inherently imprecise, and the script correctly acknowledges this limitation.**

### 2c. HSCT_RMSSD_RANGE = (25, 40) ms

**Honestly labeled as "no validated literature; clinical estimate."** There is indeed no published RMSSD normative data specifically for post-HSCT patients. This is a reasonable estimate based on general knowledge that:
- HSCT patients have autonomic dysfunction
- Cancer patients generally have lower HRV
- Range is below population norms but above the deficiency threshold

**VERDICT: CORRECT disclosure. No issue.**

---

## 3. SpO2 Thresholds for BOS Screening

### Claims in analyze_oura_spo2_trend.py (lines 68-76):

| Constant | Value | Assessment |
|----------|-------|------------|
| `SPO2_ABSOLUTE_THRESHOLD` | 94.0% | Reasonable desaturation cutoff |
| `SPO2_NORMAL_RANGE` | (95.0, 100.0)% | Standard clinical range |
| `SPO2_CONCERN_SLOPE` | -0.02 %/day | Custom metric (1% per 50 days) |

### In gvhd_predict.py (lines 2136-2137, 2163-2165):
- SpO2 < 96%: flagged as concerning
- SpO2 < 95%: flagged as more concerning (pulmonary_factor = 1.5)

### Assessment: CLINICALLY REASONABLE BUT NOVEL APPLICATION

**SpO2 is NOT a validated screening tool for BOS.** The standard BOS diagnostic criteria (Jagasia 2015, EBMT guidelines) require:

1. **FEV1/FVC ratio < 0.7** and **FEV1 < 75% predicted** (spirometry)
2. Air trapping on HRCT or RV > 120%
3. Absence of respiratory infection

SpO2 is a late-stage indicator. BOS can progress substantially before SpO2 drops. The NIH 2014 lung scoring system uses FEV1:
- Score 0: FEV1 > 80%
- Score 1: FEV1 60-79%
- Score 2: FEV1 40-59%
- Score 3: FEV1 < 39%

SpO2 monitoring for BOS screening is a **novel, unvalidated approach.** The script acknowledges this (line 1580): "screening and cannot substitute for pulmonary function testing (spirometry). Normal SpO2 does not..." and the BOS risk score explicitly recommends spirometry at all risk levels.

The 95% and 96% thresholds are standard clinical thresholds for general hypoxemia concern, not BOS-specific:
- SpO2 >= 95%: generally normal
- SpO2 92-94%: mild hypoxemia
- SpO2 < 92%: significant hypoxemia

**VERDICT: Thresholds are standard clinical values for hypoxemia. Their application to BOS screening is novel and unvalidated, but the script correctly disclaims this and recommends spirometry. APPROPRIATE use with proper caveats.**

---

## 4. "Detection X Days Before Clinical" Claims

### Assessment: PROPERLY CAVEATED

The script does NOT make strong claims about "detection X days before clinical diagnosis." Instead, it:

1. **Retrospectively validates** against the known Feb 9 acute event (line 1720-1738)
2. Explicitly labels this as "N=1 retrospective case study" (line 1736)
3. States "External cohort validation required" (line 1737)
4. Notes "Sensitivity/specificity require an external validation cohort and cannot be computed from a single retrospective event" (lines 1720-1722)
5. Reports "red_alerts_in_event_window" and "red_alerts_outside_event_window" as descriptive statistics only

The alert lead time is reported when found (lines 1702-1708): "First RED alert: {date} ({lead_days}d before event)" - but this is purely descriptive, not a predictive claim.

**VERDICT: CORRECT AND PROPERLY CAVEATED.** The script avoids overclaiming. The "N=1 retrospective case study" framing is scientifically honest.

---

## 5. ESC References

### Assessment: MIXED

The script imports `ESC_RMSSD_DEFICIENCY` from config.py where it is attributed to "Kleiger 1987, Bigger 1992." As analyzed in Section 2a above, this attribution is incorrect.

The actual ESC reference that should be cited for HRV standards is:
- **Task Force of ESC and NASPE (1996).** "Heart rate variability: standards of measurement, physiological interpretation, and clinical use." Circulation 93:1043-65.

The script's advanced HRV analysis (analyze_oura_advanced_hrv.py, line 35) correctly cites "Task Force ESC/NASPE 1996 (HRV standards)" but the config.py does not propagate this citation to the threshold.

**The ESC is not directly cited for the RMSSD < 15 ms threshold in any published guideline.** This is a clinical convention threshold, not a guideline-endorsed cutoff.

**VERDICT: The ESC reference is misattributed. The threshold exists in clinical convention, not in a specific ESC guideline.**

---

## 6. FACT-JACIE and EBMT Guideline References

### Assessment: NOT DIRECTLY CITED IN THIS SCRIPT

The gvhd_predict.py script does not directly cite FACT-JACIE or EBMT guidelines. The parent project's CLAUDE.md mentions these guidelines exist in `12_guidelines/`. The BOS section mentions spirometry recommendations which align with EBMT 8th Edition guidance (pulmonary function monitoring post-HSCT), but no specific citation is made.

This is acceptable because the script does not claim to implement FACT-JACIE or EBMT protocols. It is a novel wearable-based approach that complements (not replaces) guideline-recommended monitoring.

**VERDICT: N/A - no claims made about implementing these guidelines. No issue.**

---

## 7. Ruxolitinib Response Timeline

### Claims in script:
- Lines 2197-2209: Pre/post ruxolitinib comparison with the note "Too early to assess (< 7 days)"
- config.py: Ruxolitinib started 2026-03-16, analysis date 2026-03-24 = 8 days post-start

### Assessment: CLINICALLY PLAUSIBLE AND PROPERLY CAVEATED

From the REACH3 trial (Zeiser et al., NEJM 2021, PMID 34260836):
- **Overall response rate (ORR) at cycle 7 day 1:** 49.7% ruxolitinib vs 25.6% BAT
- **Median time to response:** Typically assessed at first evaluation (cycle 1-2, ~4-8 weeks)
- **Best overall response:** Some partial responses seen as early as week 4

Ruxolitinib's mechanism of action (JAK1/2 inhibition) can produce measurable anti-inflammatory effects within days:
- Cytokine suppression (IL-6, TNF-alpha) begins within 24-48 hours
- Clinical symptom improvement may begin within 1-2 weeks for some manifestations
- Formal response assessment per NIH criteria typically at 4-8 weeks

**Biometric changes within 7-14 days are pharmacologically plausible** because:
1. JAK inhibition reduces systemic inflammation rapidly
2. Temperature, heart rate, and HRV could reflect reduced inflammatory burden
3. However, concurrent HEV diagnosis (March 18) is a major confound

The script correctly notes:
- "Too early to assess (< 7 days)" when post-treatment data is < 7 days
- "Preliminary assessment" when >= 7 days
- The DATA_METHODOLOGY.md explicitly discusses HEV as a confound

**VERDICT: PLAUSIBLE. Early biometric changes from JAK inhibition are pharmacologically expected. The script correctly caveats the short window and HEV confound.**

---

## 8. Cohen's d and Effect Size Interpretations

### Implementation (lines 1986-1994):

```python
cohens_d = (high_vals.mean() - low_vals.mean()) / (
    np.sqrt((low_vals.std() ** 2 + high_vals.std() ** 2) / 2) + 1e-6
)
```

### Assessment: CORRECT FORMULA, MINOR ISSUE

This implements the **pooled standard deviation Cohen's d** formula correctly:

d = (M1 - M2) / sqrt((SD1^2 + SD2^2) / 2)

This is the standard equal-variance pooled formula. The `+ 1e-6` prevents division by zero, which is appropriate.

**Minor issue:** The standard Cohen's d convention uses:
- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8

The script does not explicitly interpret d values against these benchmarks. In the combined importance score (lines 1997-2002), it uses `min(abs(cohens_d) / 2, 1)` which normalizes d to [0, 1] by dividing by 2 and capping. This means d = 2.0 maps to importance = 1.0, which is a reasonable scaling choice.

**Note:** This is the equal-N pooled SD formula. If the two groups (high/low GVHD days) have very unequal sizes, Hedges' g with degrees-of-freedom correction would be slightly more appropriate. However, for an exploratory importance ranking, this is acceptable.

**VERDICT: CORRECT implementation. Standard formula. No misinterpretation of effect sizes.**

---

## 9. Temperature Thresholds

### Claims (lines 603, 644-653):
- Fever threshold: +0.5 degrees C deviation
- Hypothermia threshold: -0.5 degrees C deviation

### Assessment: REASONABLE BUT CONTEXT-DEPENDENT

These are thresholds for **deviation from personal baseline**, not absolute temperature thresholds. In the context of Oura Ring's temperature deviation measurement:

- Standard clinical fever: >= 38.0 degrees C (oral) or >= 38.3 degrees C (core)
- Oura reports deviation from personal baseline (typically ~0.0 with range of about +/- 1.0 degrees C)
- A +0.5 degrees C deviation from personal baseline is a meaningful shift but not necessarily fever

For post-HSCT patients, even subclinical temperature elevations may signal GvHD flare or infection. The +/- 0.5 degrees C threshold is reasonable for a wearable-based alert system given the limited precision of consumer temperature sensors.

**VERDICT: REASONABLE for wearable deviation monitoring. Not a standard clinical threshold, but appropriate in context.**

---

## 10. Composite Score Weights

### Weights (lines 138-145):

| Stream | Weight | Justification |
|--------|--------|---------------|
| Temperature | 0.25 | Inflammatory marker |
| HRV | 0.20 | Autonomic dysfunction |
| SpO2 | 0.15 | Pulmonary GVHD |
| Sleep fragmentation | 0.15 | Sleep disruption |
| Resting HR | 0.15 | Cardiac stress |
| Activity | 0.10 | Functional decline |

### Assessment: REASONABLE BUT UNVALIDATED

These weights are **not evidence-based from a clinical trial.** They represent expert-opinion-level weighting. The relative ordering (temperature > HRV > SpO2/sleep/HR > activity) is clinically sensible:

- Temperature deviation is the most direct inflammatory marker available from a wearable
- HRV reflects autonomic dysfunction, which is well-documented in GVHD
- SpO2, sleep fragmentation, and HR are complementary signals

However, the specific numeric weights (0.25, 0.20, etc.) are arbitrary. No published literature validates these specific weights for GVHD composite scoring from wearable data.

**VERDICT: Clinically sensible ordering. Specific weights are unvalidated. This should be stated explicitly in the report.**

---

## 11. rSLDS/HMM Model Claims

### Claims (lines 1042-1064):
- "rSLDS is a strict upgrade over the previous HMM"
- Linderman et al. 2017 citation
- 4 states: Remission, Pre-flare, Active Flare, Recovery

### Assessment: METHODOLOGICALLY SOUND

The rSLDS (Linderman et al., "Bayesian learning and inference in recurrent switching linear dynamical systems," AISTATS 2017) is indeed a generalization of both HMMs and linear dynamical systems. The claim that it is a "strict upgrade" over HMM is technically correct in the sense that an HMM is a special case of an SLDS.

The 4-state model is clinically reasonable:
- Remission = stable baseline
- Pre-flare = prodromal deterioration
- Active Flare = acute episode
- Recovery = improvement phase

The implementation correctly:
- Uses the `ssm` library (Linderman Lab)
- Implements Laplace-EM fitting
- Falls back to HMM if ssm unavailable
- Validates against known events
- Reports convergence diagnostics

**VERDICT: CORRECT methodological claims. The rSLDS reference is accurate (though it is a conference paper, not a journal paper - AISTATS 2017, not a 2017 journal publication). The 4-state clinical model is reasonable.**

---

## 12. Statistical Methodology Concerns

### N=1 Design Limitations

The script correctly and repeatedly acknowledges this is an N=1 case study (lines 1720, 1736-1737, 2501-2504, 2635-2636). This is commendable.

### Circular Analysis Risk

**Potential issue:** The composite score is built from the same features used in the feature importance analysis (Section 7). Features correlated with the composite will trivially rank high because the composite is a weighted sum of those features. The script partially mitigates this by using mutual information and point-biserial correlation with a binarized target, but the circularity remains.

**Recommendation:** Note in the report that feature importance rankings are partially tautological when the target is derived from the same features.

### Multiple Comparisons

The feature importance analysis tests 17 features without correction for multiple comparisons. P-values are reported but not adjusted. For an exploratory N=1 study, this is acceptable, but should be noted.

---

## Summary of Findings

| Item | Verdict | Severity |
|------|---------|----------|
| GvHD grading (NIH 2014) | Defensible but imprecise ("10+ organs") | LOW |
| RMSSD deficiency threshold (15 ms) | Clinically reasonable, **attribution WRONG** | MEDIUM |
| RMSSD population norms (Nunan 2010) | Approximately correct | LOW |
| HSCT RMSSD range | Honestly disclosed as estimate | NONE |
| SpO2 BOS screening thresholds | Standard hypoxemia values, novel BOS application properly caveated | LOW |
| Early detection claims | Properly caveated as N=1 retrospective | NONE |
| ESC references | **Misattributed** (Kleiger/Bigger did SDNN/frequency, not RMSSD) | MEDIUM |
| FACT-JACIE / EBMT | Not cited, no claims made | NONE |
| Ruxolitinib timeline | Pharmacologically plausible, properly caveated | NONE |
| Cohen's d formula | Correct standard implementation | NONE |
| Temperature thresholds | Reasonable for wearable deviation | LOW |
| Composite weights | Unvalidated but sensible | LOW |
| rSLDS methodology | Correct claims and implementation | NONE |
| Circular analysis risk | Feature importance partly tautological | LOW |

### Required Fixes (MEDIUM severity):

1. **config.py line 30:** Change attribution from "Kleiger 1987, Bigger 1992" to "ESC/NASPE Task Force 1996; clinical convention (Shaffer & Ginsberg 2017)." Kleiger 1987 studied SDNN, not RMSSD.

2. **All scripts referencing "Kleiger 1987" for RMSSD:** Update to correct citation. This affects analyze_oura_full.py (lines 508, 1262, 1622), analyze_oura_anomalies.py (line 1262, 2132), generate_oura_3d_dashboard.py (line 454).

### Suggested Improvements (LOW severity):

3. **gvhd_predict.py line 2326:** Change "10+ organs" to "7+ NIH organ categories plus atypical sites" for precision.

4. **Composite score methodology note:** Add statement that specific weights are expert-opinion, not validated.

5. **Feature importance section:** Add caveat about circularity (features that compose the target will trivially rank high).

6. **config.py line 42:** Consider noting that the 49 ms median may be from Shaffer & Ginsberg 2017 rather than directly from Nunan 2010, or verify the exact pooled value.

7. **Cross-script inconsistency:** `config.py` defines `POPULATION_RMSSD_MEDIAN = 49` ms while `analyze_oura_biomarkers.py` (line 74) defines `NORM_RMSSD_MEAN = 42.0` ms, both citing Nunan 2010 / Shaffer & Ginsberg 2017. The difference is likely mean (42) vs median (49), but the config uses "MEDIAN" in the variable name while 49 may actually be a median from a different source or age subset. These values should be reconciled: either use 42 ms as the population mean or clearly document why two different values exist.

---

## Appendix: PubMed Verification

The following references were verified via PubMed API during this audit:

- **Kleiger 1987** (PMID 3812275): Confirmed. "Decreased heart rate variability and its association with increased mortality after acute myocardial infarction." Am J Cardiol. Studies SDNN, not RMSSD.
- **Bigger 1992** (PMID 1728446): Confirmed. "Frequency domain measures of heart period variability and mortality after myocardial infarction." Circulation. Studies frequency-domain HRV, not RMSSD.
- **Jagasia 2015** (PMID 25529383): Confirmed. "NIH Consensus Development Project on Criteria for Clinical Trials in Chronic GVHD: I. The 2014 Diagnosis and Staging Working Group report." BBMT.
- **Zeiser 2021 / REACH3** (PMID 34260836): Confirmed. "Ruxolitinib for Glucocorticoid-Refractory Chronic Graft-versus-Host Disease." NEJM.
- **Linderman 2017 rSLDS:** Not indexed in PubMed (conference paper, AISTATS 2017). Correct methodological reference.
- **Nunan 2010:** Not returned by targeted PubMed search but known citation: Pacing Clin Electrophysiol 2010;33:1407-17. Paper reports pooled means from short-term ECG studies.

### Key Literature Not Cited But Should Be Considered

- **ESC/NASPE Task Force 1996.** "Heart rate variability: standards of measurement, physiological interpretation, and clinical use." Circulation 93:1043-65. The foundational HRV standards paper.
- **Shaffer F, Ginsberg JP 2017.** "An Overview of Heart Rate Variability Metrics and Norms." Front Public Health 5:258. Comprehensive modern HRV norms review. Already cited in some scripts but not config.py.
- **Moon JH et al. 2014** (PMID 24447907): "Validation of NIH global scoring system for chronic GVHD." Validates the grading system referenced in the script.
