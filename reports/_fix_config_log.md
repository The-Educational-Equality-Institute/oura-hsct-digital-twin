# config.py Fix Log

**Date:** 2026-03-24
**File:** `/home/ovehe/projects/helseoversikt/oura-digital-twin/config.py`

## FIX H3 - Wrong RMSSD citations

**Status: Already correct from prior session.**

Line 30 already reads:
```
ESC_RMSSD_DEFICIENCY = 15  # Clinical concern threshold (ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017)
```
No "Kleiger 1987" or "Bigger 1992" references remain in config.py.

---

## FIX M5 - Add RMSSD norms

**Status: Already correct from prior session.**

Lines 42-45 contain:
```python
POPULATION_RMSSD_MEDIAN = 49  # General population median RMSSD (ms) - Nunan 2010
POPULATION_RMSSD_MEAN = 42.0  # Shaffer & Ginsberg 2017, healthy adults
POPULATION_RMSSD_SD = 15.0    # Shaffer & Ginsberg 2017
# NOTE: Median (49) and mean (42) differ because RMSSD is right-skewed in the population.
```

---

## FIX L8 - Remove unused light-theme palette

**Status: FIXED this session.**

**Removed from config.py:**
```python
C_ACCENT = "#3399FF"        # Accent blue - highlights
C_CRITICAL = "#DC3545"      # Red - clinical threshold violations
C_PRE_TX = "#6C757D"        # Pre-treatment (muted)
C_POST_TX = "#0056B3"       # Post-treatment (primary blue)
```
Also removed the comment block above them.

**Verification:**
- `grep` confirmed 0 Python files import C_ACCENT/C_CRITICAL/C_PRE_TX/C_POST_TX from `config`
- All 13 analysis scripts import these from `_theme.py` instead
- `_theme.py` defines its own dark-theme versions independently (does not re-export from config)
- `C_PRIMARY`, `C_SECONDARY`, `C_BACKGROUND`, `C_SURFACE`, `C_TEXT`, `C_SUBTEXT` never existed in config.py
- `PLOTLY_LAYOUT` dict never existed in config.py
- `FONT_FAMILY` and `PLOTLY_CDN_URL` are still used by `_theme.py` and kept

---

## Previous session notes (preserved)

### FIX type consistency - dates (from prior session)

`KNOWN_EVENT_DATE` and `HEV_DIAGNOSIS_DATE` were converted from strings to `date` objects for type consistency with `TRANSPLANT_DATE`, `TREATMENT_START`, `DATA_START`.
