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

**Minor note:** GVHD script uses `Path(__file__).parent.parent` (not `.resolve()`) for path insertion. All others use `.resolve()`. This is functionally equivalent in practice (no symlinks in the analysis dir), but inconsistent.

All scripts use `pio.templates.default = "clinical_dark"` and import from `_theme.py`.

---

## 2. Clinical Constants Consistency

### 2.1 Treatment/Event Dates

| Constant | config.py value | Scripts using it |
|----------|----------------|-----------------|
| TRANSPLANT_DATE | date(2023, 11, 23) | full, biomarkers, causal, digital_twin, gvhd |
| TREATMENT_START | date(2026, 3, 16) | full, biomarkers, causal, digital_twin, sleep_advanced, spo2, gvhd |
| KNOWN_EVENT_DATE | "2026-02-09" (str) | digital_twin, 3d_dashboard, gvhd |
| HEV_DIAGNOSIS_DATE | "2026-03-18" (str) | digital_twin, gvhd |
| DATA_START | date(2026, 1, 8) | causal, 3d_dashboard, gvhd |

All scripts that use these dates import them from config.py. No hardcoded overrides found. CONSISTENT.

### 2.2 Clinical Thresholds

| Constant | config.py value | Usage |
|----------|----------------|-------|
| ESC_RMSSD_DEFICIENCY | 15 ms | full, biomarkers (allostatic threshold), causal, digital_twin, advanced_hrv, gvhd |
| NOCTURNAL_HR_ELEVATED | 80 bpm | full, causal, digital_twin, gvhd |
| IST_HR_THRESHOLD | 90 bpm | full only |
| POPULATION_RMSSD_MEDIAN | 49 ms | full, causal, digital_twin, advanced_hrv |
| NORM_RMSSD_P25 | 36 ms | full, advanced_hrv |
| NORM_RMSSD_P75 | 72 ms | full, advanced_hrv |
| HSCT_RMSSD_RANGE | (25, 40) ms | full, digital_twin, advanced_hrv |
| BASELINE_DAYS | 14 | spo2 |

All scripts import these from config.py without local overrides. CONSISTENT.

### 2.3 Local Constants That Should Be Checked

**FINDING - NORM_RMSSD_MEAN discrepancy:**
- `config.py` defines `POPULATION_RMSSD_MEDIAN = 49` (Nunan 2010 meta-analysis)
- `analyze_oura_biomarkers.py` defines locally `NORM_RMSSD_MEAN = 42.0` (Nunan 2010, Shaffer & Ginsberg 2017)
- These are different statistics (median vs mean), but both reference "Nunan 2010"
- The biomarkers script uses NORM_RMSSD_MEAN=42 for z-score computation in ADSI, while other scripts use POPULATION_RMSSD_MEDIAN=49
- **IMPACT:** The ADSI z-score in biomarkers uses a lower reference (42), making patient RMSSD look closer to normal than if using 49. This is technically correct (mean vs median of a right-skewed distribution), but the two values should be documented together in config.py for transparency.

**FINDING - NORM_RMSSD_SD only in biomarkers:**
- `analyze_oura_biomarkers.py` defines `NORM_RMSSD_SD = 15.0` locally
- No other script defines or uses this value
- Should be centralized in config.py if ADSI computation is reused

---

## 3. SQL Query Consistency

*Analysis in progress...*
