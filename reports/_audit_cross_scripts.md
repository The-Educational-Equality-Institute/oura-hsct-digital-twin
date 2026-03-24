# Cross-Script Consistency Audit

**Generated:** 2026-03-24
**Scripts audited:** 8 analysis scripts + GVHD reference script
**Config reference:** config.py (project root)

---

## 1. Config Import Verification

All 9 scripts correctly import from `config.py` via `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`.

| Script | Import mechanism | Correct? |
|--------|-----------------|----------|
| analyze_oura_full.py | `from config import DATABASE_PATH, REPORTS_DIR, TRANSPLANT_DATE, PATIENT_AGE, PATIENT_LABEL, ESC_RMSSD_DEFICIENCY, NOCTURNAL_HR_ELEVATED, IST_HR_THRESHOLD, POPULATION_RMSSD_MEDIAN, NORM_RMSSD_P25, NORM_RMSSD_P75, HSCT_RMSSD_RANGE, TREATMENT_START` | YES |
| analyze_oura_biomarkers.py | `from config import DATABASE_PATH, REPORTS_DIR, TRANSPLANT_DATE, TREATMENT_START, TREATMENT_START_STR, PATIENT_AGE, PATIENT_LABEL, ESC_RMSSD_DEFICIENCY` | YES |
| analyze_oura_causal.py | `from config import DATABASE_PATH, REPORTS_DIR, TRANSPLANT_DATE, TREATMENT_START, PATIENT_AGE, PATIENT_LABEL, DATA_START, ESC_RMSSD_DEFICIENCY, NOCTURNAL_HR_ELEVATED, POPULATION_RMSSD_MEDIAN` | YES |
| analyze_oura_digital_twin.py | `from config import DATABASE_PATH, REPORTS_DIR, TRANSPLANT_DATE, TREATMENT_START, KNOWN_EVENT_DATE, HEV_DIAGNOSIS_DATE, PATIENT_AGE, PATIENT_LABEL, ESC_RMSSD_DEFICIENCY, NOCTURNAL_HR_ELEVATED, POPULATION_RMSSD_MEDIAN, HSCT_RMSSD_RANGE` | YES |
| analyze_oura_advanced_hrv.py | `from config import DATABASE_PATH, REPORTS_DIR, TREATMENT_START_STR, PATIENT_LABEL, ESC_RMSSD_DEFICIENCY, NOCTURNAL_HR_ELEVATED, POPULATION_RMSSD_MEDIAN, NORM_RMSSD_P25, NORM_RMSSD_P75, HSCT_RMSSD_RANGE` | YES |
| analyze_oura_sleep_advanced.py | `from config import DATABASE_PATH, REPORTS_DIR, TREATMENT_START, TREATMENT_START_STR, PATIENT_LABEL` | YES |
| analyze_oura_spo2_trend.py | `from config import DATABASE_PATH, REPORTS_DIR, TREATMENT_START, PATIENT_LABEL, BASELINE_DAYS` | YES |
| generate_oura_3d_dashboard.py | `from config import DATABASE_PATH, REPORTS_DIR, TREATMENT_START_STR, KNOWN_EVENT_DATE, DATA_START, PATIENT_LABEL, PATIENT_TIMEZONE, FONT_FAMILY, PLOTLY_CDN_URL, C_CRITICAL, C_ACCENT, C_PRE_TX, C_POST_TX` | YES |
| analyze_oura_gvhd_predict.py (ref) | `from config import DATABASE_PATH, REPORTS_DIR, TRANSPLANT_DATE, TREATMENT_START, KNOWN_EVENT_DATE, HEV_DIAGNOSIS_DATE, PATIENT_AGE, PATIENT_LABEL, DATA_START, ESC_RMSSD_DEFICIENCY, NOCTURNAL_HR_ELEVATED` | YES |

**Minor note:** GVHD script uses `Path(__file__).parent.parent` (not `.resolve()`) for path insertion. All others use `.resolve()`. Functionally equivalent but inconsistent.

All scripts use `pio.templates.default = "clinical_dark"` and import from `_theme.py`.

---

## 2. Clinical Constants Consistency

### 2.1 Treatment/Event Dates - CONSISTENT

All scripts import dates from config.py. No hardcoded overrides found.

| Constant | config.py value | Scripts using it |
|----------|----------------|-----------------|
| TRANSPLANT_DATE | date(2023, 11, 23) | full, biomarkers, causal, digital_twin, gvhd |
| TREATMENT_START | date(2026, 3, 16) | full, biomarkers, causal, digital_twin, sleep, spo2, gvhd |
| KNOWN_EVENT_DATE | "2026-02-09" | digital_twin, 3d_dashboard, gvhd |
| HEV_DIAGNOSIS_DATE | "2026-03-18" | digital_twin, gvhd |
| DATA_START | date(2026, 1, 8) | causal, 3d_dashboard, gvhd |

### 2.2 Clinical Thresholds - CONSISTENT

All shared thresholds are imported from config.py without local overrides.

### 2.3 Local Constants That Differ

**FINDING 1 - NORM_RMSSD_MEAN vs POPULATION_RMSSD_MEDIAN (LOW SEVERITY)**

- `config.py`: `POPULATION_RMSSD_MEDIAN = 49` (Nunan 2010 meta-analysis)
- `analyze_oura_biomarkers.py` line 74: `NORM_RMSSD_MEAN = 42.0` (local, refs Nunan 2010 + Shaffer & Ginsberg 2017)
- These are technically different statistics (median vs mean of a right-skewed distribution)
- Impact: ADSI z-score in biomarkers uses 42 as reference; other scripts use 49 for contextual display
- **Recommendation:** Add `POPULATION_RMSSD_MEAN = 42` and `POPULATION_RMSSD_SD = 15.0` to config.py for transparency

**FINDING 2 - NORM_RMSSD_SD only in biomarkers (LOW SEVERITY)**

- `analyze_oura_biomarkers.py` line 75: `NORM_RMSSD_SD = 15.0` (local)
- No other script uses this value, but it is the SD paired with NORM_RMSSD_MEAN
- Should be in config.py alongside the mean

---

## 3. SQL Query Consistency

### 3.1 HRV Table (oura_hrv) - INCONSISTENT NULL FILTERING

| Script | WHERE clause | Issue |
|--------|-------------|-------|
| full | `WHERE rmssd IS NOT NULL` | Includes zero values |
| biomarkers | `WHERE rmssd IS NOT NULL AND rmssd > 0` | Excludes zeros |
| causal | No WHERE clause (bare SELECT) | Includes NULL and zero |
| digital_twin | No WHERE clause (bare SELECT from oura_hrv) | Includes NULL and zero |
| advanced_hrv | `WHERE rmssd IS NOT NULL` | Includes zero values |
| sleep_advanced | `WHERE source LIKE 'sleep_period:%'` | Source-filtered, includes zero |
| 3d_dashboard | `WHERE rmssd IS NOT NULL` | Includes zero values |
| gvhd | No WHERE clause on rmssd | Includes NULL and zero |

**IMPACT:** Scripts that include NULL/zero RMSSD values will produce different daily means than those that exclude them. A zero RMSSD is physiologically impossible and represents a sensor failure. The causal and digital_twin scripts are the most affected because they use raw `AVG(rmssd)` in SQL without filtering NULL/zero.

- `analyze_oura_digital_twin.py` line 192: `SELECT ... AVG(rmssd) AS mean_rmssd FROM oura_hrv GROUP BY ...` - no null/zero filter
- `analyze_oura_causal.py` line 274: `SELECT timestamp, rmssd FROM oura_hrv ORDER BY timestamp` - no filter at all

**Recommendation:** Standardize all HRV queries to use `WHERE rmssd IS NOT NULL AND rmssd > 0`.

### 3.2 Sleep Period Type Filtering - INCONSISTENT

| Script | Filter | Consequence |
|--------|--------|------------|
| biomarkers | `WHERE type = 'long_sleep'` | Primary sleep only |
| causal | `WHERE type = 'long_sleep'` | Primary sleep only |
| digital_twin | `WHERE type = 'long_sleep'` | Primary sleep only |
| full | Post-hoc: `sp[sp["type"] == "long_sleep"]` | Primary sleep only |
| sleep_advanced | All types loaded, then filtered in Python: `type == "long_sleep"` per analysis | Mixed |
| spo2 | No type filter | All sleep types |
| 3d_dashboard | `WHERE type IN ('long_sleep', 'sleep')` | Includes short naps |
| gvhd | No type filter in SQL | All sleep types |

**IMPACT:** The 3d_dashboard and gvhd scripts include nap/rest periods that other scripts exclude. This can skew sleep HR means, efficiency, and duration metrics lower (naps tend to be shorter, less efficient).

- GVHD predict includes all sleep periods but de-duplicates by taking the longest per day
- 3d_dashboard includes both 'long_sleep' and 'sleep' types

**Recommendation:** Standardize to `type = 'long_sleep'` for primary analyses, or explicitly document when naps are included.

### 3.3 Heart Rate Source Filtering - INCONSISTENT

| Script | Filter | Data subset |
|--------|--------|------------|
| full | No source filter | All sources (rest, awake, etc.) |
| biomarkers | `WHERE source = 'awake'` (for awake HR only) | Awake readings |
| causal | No source filter | All sources |
| digital_twin | No source filter | All sources |
| spo2 | `WHERE source='rest'` (for nightly HR) | Rest readings only |
| 3d_dashboard | No source filter | All sources |
| gvhd | No source filter | All sources |

**IMPACT:** When computing "daily mean HR", scripts without source filtering mix rest and awake readings. The SpO2 script's nightly HR load uses source='rest', which is correct for nocturnal analysis. The biomarkers script correctly separates awake HR for dipping calculation.

**Note:** Most scripts that compute daily aggregates use sleep_periods.average_heart_rate for nightly HR instead of aggregating oura_heart_rate, which is the correct approach. The oura_heart_rate aggregations are used for trend analysis where mixed sources are acceptable.

### 3.4 oura_sleep vs oura_sleep_periods

- `analyze_oura_full.py` uniquely queries the `oura_sleep` table (line 129): `FROM oura_sleep WHERE score IS NOT NULL`
- All other scripts use `oura_sleep_periods`
- These are different tables: `oura_sleep` contains daily sleep summaries; `oura_sleep_periods` contains per-period detail
- The full script loads both and uses `oura_sleep` for score-based analysis and `oura_sleep_periods` for duration/HR analysis. This is correct.

### 3.5 Activity Table Columns

- `analyze_oura_causal.py` line 321 queries `daily_movement FROM oura_activity`
- This column may not exist in all database versions
- No other script queries `daily_movement`
- If the column is missing, `pd.to_numeric(... errors="coerce")` will handle it, but it is still a latent risk

---

## 4. Metric Computation Consistency

### 4.1 Daily HRV Aggregation - INCONSISTENT ACROSS SCRIPTS

| Script | Method | Column name |
|--------|--------|------------|
| full | Python: `hrv.groupby("date")["rmssd"].mean()` | rmssd_daily_mean |
| biomarkers | SQL: `AVG(rmssd) AS rmssd_mean` | rmssd_mean |
| causal | Python: `groupby("date").agg(mean_rmssd=("rmssd", "mean"))` | mean_rmssd |
| digital_twin | SQL: `AVG(rmssd) AS mean_rmssd` | mean_rmssd |
| gvhd | Python: `groupby("date").agg(hrv_median=("rmssd", "median"), hrv_mean=("rmssd", "mean"))` | hrv_median (primary), hrv_mean |
| advanced_hrv | Uses raw 5-min epochs, not daily aggregates | N/A |

**FINDING 3 - GVHD uses MEDIAN, all others use MEAN (MEDIUM SEVERITY)**

The GVHD prediction script uses `hrv_median` as its primary HRV metric for the composite score (line 824: `daily["hrv_median"]`), while all other scripts use mean RMSSD. For a right-skewed distribution with outliers (common in nocturnal HRV), median is more robust but produces consistently lower values than mean.

This means the GVHD composite score's HRV component is computed on a different central tendency than the biomarkers ADSI, causal analysis, and digital twin. If the GVHD script were switched to mean, its baseline and z-scores would change.

**Not necessarily wrong** - the GVHD script computes its own baseline z-scores relative to itself, so the internal logic is self-consistent. But cross-script comparisons of "HRV status" will differ.

### 4.2 Sleep Duration Computation - CONSISTENT

All scripts that compute sleep duration use `total_sleep_duration` (seconds) from `oura_sleep_periods` and convert to hours via `/ 3600`. No discrepancies.

### 4.3 Sleep Stage Percentage Computation - CONSISTENT

All scripts compute REM%, deep%, etc. the same way:
- `rem_pct = rem_sleep_duration / total_sleep_duration * 100`
- Use `total_sleep_duration.replace(0, np.nan)` to avoid division by zero

### 4.4 Sleep Fragmentation - TWO DIFFERENT DEFINITIONS

**FINDING 4 - Fragmentation computed differently (MEDIUM SEVERITY)**

| Script | Definition | Normalization |
|--------|-----------|---------------|
| sleep_advanced | Wake-to-sleep transitions only (phase 4 -> 1,2,3) per hour | transitions/hr of sleep |
| gvhd | ALL phase transitions (any diff(phases) != 0) per hour | transitions/hr of total time in bed |

The sleep_advanced script counts only **wake-to-sleep reentry** events (a clinically standard fragmentation index). The GVHD script counts **all phase transitions** (deep->light, light->REM, REM->awake, etc.), which includes normal stage cycling. This produces much higher values.

Additionally:
- sleep_advanced normalizes by sleep hours (non-awake epoch count * 5 / 60)
- GVHD normalizes by total time in bed (total epoch count * 5 / 60)

**Impact:** The GVHD composite score's sleep fragmentation component is inflated relative to what sleep_advanced reports. Published norms (fragmentation_index healthy < 5/hr) only apply to the sleep_advanced definition (wake-to-sleep).

### 4.5 Allostatic Load - TWO IMPLEMENTATIONS, SAME THRESHOLDS

**FINDING 5 - Allostatic load: same 7 indicators, different computation scope (LOW SEVERITY)**

| Aspect | biomarkers | advanced_hrv |
|--------|-----------|-------------|
| Scale | 0-7 | 0-7 |
| Computation | Daily (per-row of daily DataFrame) | Whole-dataset average |
| HR threshold | sleep_hr_mean > 80 bpm | average_heart_rate > 80 bpm (NOCTURNAL_HR_ELEVATED) |
| HRV threshold | rmssd_mean < 15 ms | AVG(rmssd) < 15 ms |
| Sleep eff | < 85% | < 85% |
| Temp dev | abs() > 0.5 C | abs(mean) > 0.5 C |
| SpO2 | < 95% | < 95% |
| Deep sleep | < 10% | < 10% |
| REM sleep | < 15% | < 15% |

The biomarkers script computes allostatic load per day (a daily time series), while advanced_hrv computes a single summary score across the full dataset. Both use the same 7 indicators and thresholds. The daily version is more useful for trend analysis.

### 4.6 GVHD Activity Score - TWO DIFFERENT IMPLEMENTATIONS

**FINDING 6 - Two distinct GVHD scores with different methodologies (MEDIUM SEVERITY)**

| Aspect | biomarkers (GVHD Activity Score) | gvhd_predict (GVHD Composite) |
|--------|----------------------------------|-------------------------------|
| Components | 5 (temp slope, SpO2 slope, HRV slope, frag slope, stress ratio) | 6 (temp z, SpO2 z, HRV z, frag z, resting HR z, activity z) |
| Methodology | 7-day rolling OLS slopes (rate of change) | Z-scores vs 14-day baseline (absolute deviation) |
| Weights | temp=0.25, SpO2=0.20, HRV=0.25, frag=0.15, recovery=0.15 | temp=0.25, SpO2=0.15, HRV=0.20, frag=0.15, HR=0.15, activity=0.10 |
| Scale | 0-100 | 0-100 |
| Name | "GVHD Activity Score" | "GVHD Composite Score" |

The biomarkers version measures **rate of deterioration** (slopes). The GVHD predict version measures **absolute deviation from baseline** (z-scores). These will produce different temporal patterns: the slope-based version detects acceleration, while the z-score version detects sustained elevation.

**Not a bug** - they intentionally measure different aspects. But the 3d_dashboard reads both and displays them in the same context. The dashboard correctly labels them differently (GVHD Score from biomarkers, composite from GVHD module).

---

## 5. Data Interpretation Consistency

### 5.1 resting_heart_rate in oura_readiness - CORRECTLY HANDLED

Multiple scripts note that `oura_readiness.resting_heart_rate` is a **contributor score (0-100)**, NOT an actual heart rate in bpm.

- `analyze_oura_full.py` lines 290-296: CRITICAL comment + uses it as `rhr_contributor_score_mean`
- `analyze_oura_causal.py` line 311: Renames to `rhr_score` to avoid confusion
- `analyze_oura_gvhd_predict.py` line 202: Loads it as `resting_heart_rate` but uses `body_temperature` and `hrv_balance` instead for analysis

All scripts that need actual resting HR correctly use `oura_sleep_periods.average_heart_rate` or `oura_sleep_periods.lowest_heart_rate`.

### 5.2 Temperature Deviation Interpretation - CONSISTENT

All scripts treat `oura_readiness.temperature_deviation` as a real physiological value (degrees C from personal baseline). Positive = warmer than baseline. Used consistently across biomarkers, digital_twin, gvhd, and spo2.

### 5.3 SpO2 Sentinel Filtering - CONSISTENT

All scripts that load SpO2 data filter out zero values: `WHERE spo2_average > 0`. Oura reports 0 when no valid reading was obtained.

---

## 6. Summary of Findings

### Issues Requiring Attention

| # | Severity | Finding | Scripts affected |
|---|----------|---------|-----------------|
| 1 | HIGH | HRV null/zero filtering inconsistent: digital_twin and causal include NULL/zero RMSSD | digital_twin, causal |
| 2 | MEDIUM | Sleep period type filtering: gvhd and 3d_dashboard include naps | gvhd, 3d_dashboard |
| 3 | MEDIUM | HRV central tendency: gvhd uses median, all others use mean | gvhd vs all others |
| 4 | MEDIUM | Fragmentation index: two different definitions (wake-to-sleep vs all transitions) | sleep_advanced vs gvhd |
| 5 | MEDIUM | Two GVHD scores with different methodologies (slopes vs z-scores) | biomarkers vs gvhd |
| 6 | LOW | NORM_RMSSD_MEAN (42) in biomarkers vs POPULATION_RMSSD_MEDIAN (49) in config | biomarkers |
| 7 | LOW | Allostatic load: daily vs summary implementations | biomarkers vs advanced_hrv |
| 8 | LOW | Path resolution: `.parent.parent` vs `.resolve().parent.parent` | gvhd |

### Items That Are Correct and Consistent

- All treatment dates from config.py - no hardcoded overrides
- All clinical thresholds from config.py
- Sleep duration and stage percentage computations
- Temperature deviation interpretation
- SpO2 sentinel value filtering
- resting_heart_rate field correctly identified as contributor score
- Database path and connection pattern (read-only mode)
- Plotly template and theme system
- Report output directory

### Recommended Actions

1. **Fix HRV null/zero filtering** in `analyze_oura_digital_twin.py` (line 192) and `analyze_oura_causal.py` (line 274). Add `WHERE rmssd IS NOT NULL AND rmssd > 0`.
2. **Document the two fragmentation definitions** in comments or a methodology note. They measure different things (clinical fragmentation vs total transitions).
3. **Standardize sleep period type filtering** or document why gvhd includes all types.
4. **Move NORM_RMSSD_MEAN and NORM_RMSSD_SD to config.py** alongside the existing POPULATION_RMSSD_MEDIAN for full transparency.
5. **Consider adding `daily_movement` column check** in causal script to avoid errors if the column is absent.
