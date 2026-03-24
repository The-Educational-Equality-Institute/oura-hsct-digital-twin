# Fix H2 - HRV null/zero filtering in analyze_oura_digital_twin.py

**Date:** 2026-03-24
**File:** `analysis/analyze_oura_digital_twin.py`

## Fixes Applied

### 1. HRV query (line 192-194) - PRIMARY FIX
- **Before:** No WHERE clause, AVG included NULL/zero RMSSD values
- **After:** Added `WHERE rmssd IS NOT NULL AND rmssd > 0`
- **Impact:** Prevents zero/null RMSSD samples from dragging down daily averages, which would corrupt the Kalman filter state estimation for Autonomic Tone

### 2. Heart rate query (line 206-208)
- **Before:** No WHERE clause, AVG included NULL/zero BPM values
- **After:** Added `WHERE bpm IS NOT NULL AND bpm > 0`
- **Impact:** Prevents zero/null BPM from corrupting daily HR averages used for Cardiac Reserve estimation

### 3. SpO2 query (line 220-222)
- **Before:** `WHERE spo2_average > 0` (missing NULL check)
- **After:** `WHERE spo2_average IS NOT NULL AND spo2_average > 0`
- **Impact:** Minor - SQL `> 0` typically excludes NULL anyway, but explicit NULL check is defensive and consistent with project conventions

### Already correct (no changes needed)
- **Temperature query (line 233-235):** Already had `WHERE temperature_deviation IS NOT NULL`
- **Sleep query (line 246-250):** Filtered by `type = 'long_sleep'`, efficiency can legitimately be 0

## Pattern consistency
Matches the established pattern in other scripts:
- `analyze_oura_biomarkers.py` lines 158, 182: `WHERE rmssd IS NOT NULL AND rmssd > 0`
- `analyze_oura_full.py` line 111: `WHERE rmssd IS NOT NULL`
- `analyze_oura_full.py` line 118: `WHERE bpm IS NOT NULL`
- `analyze_oura_advanced_hrv.py` line 2311: `WHERE rmssd IS NOT NULL`
