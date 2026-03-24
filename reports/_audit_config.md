# Configuration Consistency Audit

**Date:** 2026-03-24
**Scope:** `config.py`, `analysis/_config.py`, `analysis/_theme.py`, and all 13 `analysis/*.py` scripts

---

## 1. Import Hierarchy: CLEAN

All 13 analysis scripts import from `config.py` (top-level) directly. None import from `_config.py`.

| Script | Imports from `config` | Imports from `_theme` |
|--------|----------------------|----------------------|
| analyze_oura_biomarkers.py | Yes | Yes |
| analyze_oura_full.py | Yes | Yes |
| analyze_oura_causal.py | Yes | Yes |
| analyze_oura_advanced_hrv.py | Yes | Yes |
| analyze_oura_spo2_trend.py | Yes | Yes |
| analyze_oura_digital_twin.py | Yes | Yes |
| analyze_oura_gvhd_predict.py | Yes | Yes |
| analyze_oura_anomalies.py | Yes | Yes |
| analyze_oura_sleep_advanced.py | Yes | Yes |
| analyze_oura_foundation_models.py | Yes | Yes |
| generate_oura_3d_dashboard.py | Yes | Yes |
| generate_roadmap.py | Yes | Yes |
| _theme.py | Yes | N/A (self) |

**`_config.py` status:** Clean re-export layer. Imports only `DATABASE_PATH`, `REPORTS_DIR`, `validate_config` from `config`. Adds `BIOMETRICS_DB` alias and `INVESTIGATION_DB` resolution. No overrides, no date constants. No scripts import from it.

---

## 2. Date Consistency

### Canonical dates in `config.py`

| Constant | Value | Type |
|----------|-------|------|
| `TRANSPLANT_DATE` | `2023-11-23` | `date` |
| `TREATMENT_START` | `2026-03-16` | `date` |
| `KNOWN_EVENT_DATE` | `2026-02-09` | `str` |
| `HEV_DIAGNOSIS_DATE` | `2026-03-18` | `str` |
| `DATA_START` | `2026-01-08` | `date` |

### Type inconsistency: KNOWN_EVENT_DATE and HEV_DIAGNOSIS_DATE are `str`, others are `date`

`TRANSPLANT_DATE`, `TREATMENT_START`, and `DATA_START` are `datetime.date` objects.
`KNOWN_EVENT_DATE` and `HEV_DIAGNOSIS_DATE` are plain strings (`"2026-02-09"`, `"2026-03-18"`).

The config comment says "str for compat" but this forces every consumer to do `pd.Timestamp(KNOWN_EVENT_DATE)` or `datetime.strptime(KNOWN_EVENT_DATE, "%Y-%m-%d")` conversions. 15+ occurrences of `pd.Timestamp(KNOWN_EVENT_DATE)` across 5 scripts.

**Severity: LOW** - Works correctly, but inconsistent with the other date types.

### Hardcoded date strings in comments/HTML (not code logic)

These are date literals embedded in docstrings or HTML strings, not in computation logic:

| File | Line | Hardcoded date | Context |
|------|------|---------------|---------|
| analyze_oura_biomarkers.py | 521-522 | `2026-03-16` | Docstring comment |
| analyze_oura_digital_twin.py | 19 | `2026-03-16` | Module docstring |
| analyze_oura_causal.py | 3374 | `2026-03-18` | HTML template literal |
| generate_oura_3d_dashboard.py | 336 | `2026-03-18` | HTML template literal |

**Severity: LOW** - These are display text in comments and HTML, not logic. However, if dates change, these will be stale. The causal and 3D dashboard scripts hardcode HEV date in HTML strings rather than using `HEV_DIAGNOSIS_DATE`.

### Hardcoded medical record dates (not config dates)

| File | Lines | Dates | Purpose |
|------|-------|-------|---------|
| analyze_oura_spo2_trend.py | 90-92 | `2024-03-21`, `2025-03-20`, `2025-12-17` | DLCO measurements from medical records |
| analyze_oura_full.py | 356 | `2023-08-01`, `2023-09-01` | Samsung Health pre-diagnosis data filter |

**Severity: NONE** - These are historical medical record dates, not configurable treatment dates. They correctly belong in their scripts as data constants.

---

## 3. Color Palette: DUAL SYSTEM (config.py light theme vs _theme.py dark theme)

### Critical finding: `config.py` defines a LIGHT theme, `_theme.py` defines a DARK theme

`config.py` defines 25+ color constants and a full `PLOTLY_LAYOUT` dict for a light/white theme.
`_theme.py` defines a completely different dark theme palette and overrides all the same named constants.

**Example conflicts (same name, different values):**

| Constant | `config.py` (light) | `_theme.py` (dark) |
|----------|---------------------|-------------------|
| `C_PRIMARY` | `#0056B3` | `#3B82F6` (ACCENT_BLUE) |
| `C_SECONDARY` | `#007BFF` | `#06B6D4` (ACCENT_CYAN) |
| `C_CRITICAL` | `#DC3545` | `#EF4444` (ACCENT_RED) |
| `C_GOOD` | `#28A745` | `#10B981` (ACCENT_GREEN) |
| `C_WARNING` | `#FFC107` | `#F59E0B` (ACCENT_AMBER) |
| `C_MUTED` | `#6C757D` | `#9CA3AF` (TEXT_SECONDARY) |
| `C_BG` | `#F8F9FA` (light gray) | `#0F1117` (near-black) |
| `C_CARD` | `#FFFFFF` | `#1A1D27` |
| `C_TEXT` | `#343A40` | `#E8E8ED` |
| `C_GRID` | `#E9ECEF` | `#2D3348` |
| `C_BASELINE` | `#003366` | `#60A5FA` |
| `C_COUNTERFACTUAL` | `#A7D9F7` | `#93C5FD` |
| `C_PRE_TX` | `#6C757D` | `#9CA3AF` (TEXT_SECONDARY) |
| `C_POST_TX` | `#0056B3` | `#3B82F6` (ACCENT_BLUE) |

### The conflict in practice

**12 of 13 scripts** import colors from `_theme.py` (dark theme). This is correct since all reports use `pio.templates.default = "clinical_dark"`.

**1 script** (`generate_oura_3d_dashboard.py`) imports `C_CRITICAL` and `C_ACCENT` from `config.py` (light theme) AND also imports colors from `_theme.py` (dark theme):

```python
from config import (
    ...
    C_CRITICAL, C_ACCENT,     # Light theme: #DC3545, #3399FF
    C_PRE_TX, C_POST_TX,      # Light theme: #6C757D, #0056B3
)
from _theme import (
    ACCENT_RED, ...            # Dark theme: #EF4444
)
```

This means `generate_oura_3d_dashboard.py` uses light-theme `C_CRITICAL` (#DC3545) while all other scripts use dark-theme `ACCENT_RED` (#EF4444) for critical indicators. Similarly, `C_PRE_TX` and `C_POST_TX` come from the light palette.

**Severity: MEDIUM** - Visual inconsistency. The reds and blues are similar enough to not cause obvious breakage, but the pre/post treatment colors and critical red differ between 3D dashboard and all other reports.

### `config.py` PLOTLY_LAYOUT is unused

`config.py` defines a full `PLOTLY_LAYOUT` dict (lines 96-108) for the light theme. No analysis script imports or references it. All scripts use `pio.templates.default = "clinical_dark"` from `_theme.py`. This is dead code.

**Severity: LOW** - Dead code, no functional impact.

---

## 4. Clinical Constants: CLEAN with minor local additions

All clinical thresholds are correctly imported from `config.py`:
- `ESC_RMSSD_DEFICIENCY` (15 ms) - used in 8 scripts, no hardcoded override
- `NOCTURNAL_HR_ELEVATED` (80 bpm) - used in 5 scripts, no hardcoded override
- `IST_HR_THRESHOLD` (90 bpm) - used in 1 script, no hardcoded override
- `POPULATION_RMSSD_MEDIAN` (49 ms) - used in 5 scripts
- `NORM_RMSSD_P25` (36 ms) / `NORM_RMSSD_P75` (72 ms) - used in 3 scripts
- `HSCT_RMSSD_RANGE` (25, 40) - used in 4 scripts
- `BASELINE_DAYS` (14) - used in 3 scripts

### Local constants NOT in config.py (acceptable)

| Script | Constant | Value | Assessment |
|--------|----------|-------|------------|
| analyze_oura_biomarkers.py | `NORM_RMSSD_MEAN` | 42.0 ms | Different stat (mean vs median in config). Config has median=49. This is the biomarker composite Z-score mean. |
| analyze_oura_biomarkers.py | `NORM_RMSSD_SD` | 15.0 ms | Used only for Z-score normalization. Not in config. |
| analyze_oura_full.py | `NOCTURNAL_HR_DIP_NORMAL_LOW/HIGH` | 45/55 bpm | Nocturnal dip range. Script-specific. |
| analyze_oura_spo2_trend.py | `SPO2_ABSOLUTE_THRESHOLD` | 94.0% | SpO2-specific constant. |
| analyze_oura_spo2_trend.py | DLCO_MEASUREMENTS | 3 dates | Medical record data points. |

**Note:** `NORM_RMSSD_MEAN = 42.0` in biomarkers vs `POPULATION_RMSSD_MEDIAN = 49` in config are intentionally different statistics (mean vs median) from the same Nunan 2010 reference, used for different purposes (Z-score vs threshold). Not a conflict.

---

## 5. Summary of Issues

### Issues requiring action

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **MEDIUM** | `generate_oura_3d_dashboard.py` imports `C_CRITICAL`, `C_ACCENT`, `C_PRE_TX`, `C_POST_TX` from `config.py` (light theme) instead of `_theme.py` (dark theme) | Change imports to use `_theme.py` equivalents |
| 2 | **LOW** | `analyze_oura_causal.py:3374` and `generate_oura_3d_dashboard.py:336` hardcode `2026-03-18` in HTML strings instead of using `HEV_DIAGNOSIS_DATE` | Use f-string with `HEV_DIAGNOSIS_DATE` |
| 3 | **LOW** | `KNOWN_EVENT_DATE` and `HEV_DIAGNOSIS_DATE` are `str` while other dates are `date` objects | Consider making all dates `date` type |
| 4 | **LOW** | `config.py` `PLOTLY_LAYOUT` dict (lines 96-108) and all light-theme color constants are dead code | Remove or mark as deprecated |
| 5 | **INFO** | `analyze_oura_biomarkers.py:521-522` and `analyze_oura_digital_twin.py:19` have `2026-03-16` in docstring comments | Update if date changes |

### Clean findings (no action needed)

- All 13 scripts import dates from `config.py` - zero hardcoded date logic
- `_config.py` is a clean re-export layer with no overrides
- All clinical thresholds (ESC, nocturnal HR, IST, RMSSD norms) imported from config
- No script uses `_config.py` directly
- `PATIENT_AGE` not hardcoded in any script (all import from config)
- No light-theme config.py hex colors (#0056B3 etc.) found in any analysis script
- `TRANSPLANT_DATE` (2023-11-23) used correctly via config import everywhere
