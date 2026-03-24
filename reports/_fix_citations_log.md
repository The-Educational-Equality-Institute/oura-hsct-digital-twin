# Fix H3 - Wrong RMSSD Citations (Kleiger 1987 / Bigger 1992)

**Date:** 2026-03-24
**Issue:** Kleiger 1987 studied SDNN, Bigger 1992 studied frequency-domain measures. Neither defined RMSSD thresholds. Citations were incorrectly applied to RMSSD deficiency threshold (< 15 ms).
**Correct citation:** ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017

## Files Fixed

### 1. config.py (line 30)
- **Before:** `# Clinical concern threshold (Kleiger 1987, Bigger 1992): RMSSD < 15 ms = severe autonomic deficiency`
- **After:** `# Clinical concern threshold (ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017): RMSSD < 15 ms = severe autonomic deficiency`

### 2. config.example.py (line 33)
- **Before:** `# RMSSD < 15 ms = severe autonomic deficiency (Kleiger 1987)`
- **After:** `# RMSSD < 15 ms = severe autonomic deficiency (ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017)`

### 3. analyze_oura_advanced_hrv.py (line 1077)
- **Before:** `# 2. HRV < clinical deficiency threshold (Kleiger 1987, Bigger 1992)`
- **After:** `# 2. HRV < clinical deficiency threshold (ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017)`

### 4. analyze_oura_full.py (3 locations)
- **Line 508:** Annotation text `"Deficiency (Kleiger 1987)"` -> `"Deficiency (ESC/NASPE 1996)"`
- **Line 1323:** Category label `"Deficiency Threshold (Kleiger 1987)"` -> `"Deficiency Threshold (ESC/NASPE 1996)"`
- **Lines 1622-1623:** Removed two incorrect reference list entries (Kleiger RE et al. Am J Cardiol 1987, Bigger JT et al. Circulation 1992). Replaced with single correct entry: `Task Force of ESC/NASPE. Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation 1996;93:1043-65`
- **Line 1619:** Updated reference count from 7 to 6

### 5. analyze_oura_anomalies.py (2 locations)
- **Line 1262:** Annotation text `"Parasympathetic deficiency (Kleiger 1987 / Bigger 1992)"` -> `"Parasympathetic deficiency (ESC/NASPE 1996; Shaffer & Ginsberg 2017)"`
- **Line 2132:** HTML metric text `"(Kleiger 1987)"` -> `"(ESC/NASPE 1996)"`

### 6. generate_oura_3d_dashboard.py (line 454)
- **Before:** `f"vs healthy 42-49 ms | Kleiger 1987 / Bigger 1992"`
- **After:** `f"vs healthy 42-49 ms | ESC/NASPE 1996; Shaffer & Ginsberg 2017"`

## Files Already Correct (no changes needed)
- **analyze_oura_biomarkers.py** (line 102) - already cited `ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017`
- **analyze_oura_spo2_trend.py** - no Kleiger/Bigger references found
- **analyze_oura_sleep_advanced.py** - no Kleiger/Bigger references found

## NORM_RMSSD Check
All RMSSD norms (`ESC_RMSSD_DEFICIENCY`, `NORM_RMSSD_P25`, `NORM_RMSSD_P75`, `POPULATION_RMSSD_MEDIAN`, `HSCT_RMSSD_RANGE`) are defined in `config.py` and imported by scripts. No hardcoded RMSSD norms found outside config.

## Verification
Post-fix grep for `Kleiger|Bigger` across all `*.py` files: **0 matches**. All Python source files are clean.
