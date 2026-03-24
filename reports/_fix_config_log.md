# config.py Fix Log

**Date:** 2026-03-24
**File:** `/home/ovehe/projects/helseoversikt/oura-digital-twin/config.py`

## FIX H3 - Wrong RMSSD citations

**Problem:** Line 30 references "Kleiger 1987, Bigger 1992" for RMSSD threshold. Kleiger studied SDNN, Bigger studied frequency-domain HRV. Neither defines an RMSSD cutoff.

**Fix:** Changed citation to "ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017"

**Scope:** config.py line 30 only. The analysis scripts also contain Kleiger/Bigger references (analyze_oura_full.py, analyze_oura_anomalies.py, analyze_oura_advanced_hrv.py, generate_oura_3d_dashboard.py, analyze_oura_biomarkers.py) but those are NOT in scope for this fix.

---

## FIX M5 - Add RMSSD norms

**Problem:** `POPULATION_RMSSD_MEAN` and `POPULATION_RMSSD_SD` do not exist in config.py, but `analyze_oura_biomarkers.py` already imports them (line 58). This means biomarkers currently fails on import.

**Fix:** Added after POPULATION_RMSSD_MEDIAN:
- `POPULATION_RMSSD_MEAN = 42.0` (Shaffer & Ginsberg 2017)
- `POPULATION_RMSSD_SD = 15.0` (Shaffer & Ginsberg 2017)
- Added comment explaining median vs mean difference (right-skewed distribution)

---

## FIX L8 - Remove unused light-theme palette

**Analysis:** Grepped all `from config import` blocks across 14 analysis/*.py files.

**Only script importing color constants from config.py:** `generate_oura_3d_dashboard.py` imports:
- `C_CRITICAL` (semantic)
- `C_ACCENT` (primary palette)
- `C_PRE_TX` (period/series)
- `C_POST_TX` (period/series)

**Also imported by _theme.py from config:** `FONT_FAMILY`, `PLOTLY_CDN_URL` (visual identity, not palette)

**Constants REMOVED (unused by any analysis script):**
- Primary palette: `C_PRIMARY`, `C_SECONDARY`, `C_MUTED`, `C_LIGHT`, `C_DARK`
- Semantic: `C_GOOD`, `C_WARNING`
- Layout: `C_BG`, `C_CARD`, `C_TEXT`, `C_GRID`, `C_TEXT_MUTED`
- Aliases: `C_NEUTRAL`, `C_BG_LIGHT`, `C_CAUTION`
- Biometric: `C_HR`, `C_SPO2`, `C_HRV`, `C_SLEEP`, `C_TEMP`
- Period: `C_BASELINE`, `C_COUNTERFACTUAL`, `C_FORECAST`, `C_RUX_LINE`, `C_EFFECT`
- `PLOTLY_TEMPLATE` (only used internally within config by PLOTLY_LAYOUT, being removed)
- `PLOTLY_COLORWAY` (only used internally within config by PLOTLY_LAYOUT, being removed)
- `PLOTLY_LAYOUT` dict (not imported by any script)

**Constants KEPT:**
- `C_CRITICAL` - imported by generate_oura_3d_dashboard.py
- `C_ACCENT` - imported by generate_oura_3d_dashboard.py
- `C_PRE_TX` - imported by generate_oura_3d_dashboard.py
- `C_POST_TX` - imported by generate_oura_3d_dashboard.py
- `FONT_FAMILY` - imported by _theme.py and generate_oura_3d_dashboard.py
- `PLOTLY_CDN_URL` - imported by _theme.py and generate_oura_3d_dashboard.py

---

## FIX type consistency - dates

**Problem:** `KNOWN_EVENT_DATE` and `HEV_DIAGNOSIS_DATE` are strings while `TRANSPLANT_DATE`, `TREATMENT_START`, `DATA_START` are `date` objects.

**Fix:** Converted to date objects:
- `KNOWN_EVENT_DATE = date(2026, 2, 9)`
- `HEV_DIAGNOSIS_DATE = date(2026, 3, 18)`

**Impact on analysis scripts (NOT changed, just documented):**

Scripts using `KNOWN_EVENT_DATE` as string (would need `str()` wrapper):
1. `analyze_oura_gvhd_predict.py` - uses `pd.Timestamp(KNOWN_EVENT_DATE)` (works with date), string comparisons at lines 176, 1494, 1689, 1695, 1699, 2597
2. `generate_oura_3d_dashboard.py` - assigns to ACUTE_EVENT (line 69), used as string
3. `analyze_oura_foundation_models.py` - `datetime.strptime(KNOWN_EVENT_DATE, "%Y-%m-%d")` (line 963, BREAKS), string comparisons (lines 687, 925, 938, 1013, 1223, 2006, 2016)
4. `analyze_oura_anomalies.py` - `datetime.strptime(KNOWN_EVENT_DATE, "%Y-%m-%d")` (BREAKS at lines 371, 474, 589, 694, 867, 1027), string comparisons throughout
5. `analyze_oura_digital_twin.py` - `pd.Timestamp(KNOWN_EVENT_DATE)` (works), f-strings (lines 2297, 2331, 2333, 2404, 2467)

Scripts using `HEV_DIAGNOSIS_DATE` as string:
1. `analyze_oura_gvhd_predict.py` - assigns to HEV_DIAGNOSIS (line 102)
2. `analyze_oura_digital_twin.py` - `pd.Timestamp(HEV_DIAGNOSIS_DATE)` (works), f-strings (lines 2297, 2333, 2404, 2467)
3. `analyze_oura_causal.py` - imports but unclear usage

**CRITICAL BREAKAGE:** `datetime.strptime()` calls in anomalies and foundation_models will fail with TypeError since date objects don't have the same interface as strings. These scripts need `str(KNOWN_EVENT_DATE)` or conversion to use `date` object methods directly.

**pd.Timestamp()** calls will work fine with date objects.
**f-string** usage will work (date.__str__() returns "YYYY-MM-DD").
**String equality** comparisons (e.g., `== KNOWN_EVENT_DATE`) will fail where comparing against string dates from DataFrames.
