# GVHD Predict Fix Log

Date: 2026-03-24

## Status: ALL FIXES ALREADY APPLIED

Every fix requested was verified against the current file state. All 10 fixes are already present in `analyze_oura_gvhd_predict.py`.

| Fix | Description | Status | Evidence |
|-----|------------|--------|----------|
| C1 | Missing-data days classified as Remission | ALREADY APPLIED | Line 1467: `np.full(len(full_viterbi), -1, dtype=int)`, lines 1491-1496: No Data handling, line 1510: event validation guard |
| H1 | Sleep fragmentation period alignment | ALREADY APPLIED | Line 464: sorts by `total_sleep_duration` descending, lines 458-460: merge sleep duration before dedup |
| I1 | breath_corr NaN crash | ALREADY APPLIED | Line 2176: `pd.notna(breath_corr)` guard |
| I2 | CUSUM dual-counter reset | ALREADY APPLIED | Lines 539-547: independent resets with `reset_logged` flag |
| I3 | Cohen's d pooled SD | ALREADY APPLIED | Lines 2008-2013: weighted pooled SD formula with `n_low`, `n_high`, `ddof=1` |
| I5 | WCAG AA contrast | ALREADY APPLIED | No instances of `rgba(255,255,255,0.4)` or `rgba(255,255,255,0.35)` found |
| L1 | Pre-event window off-by-one | ALREADY APPLIED | Lines 560, 888: both use `timedelta(days=6)` |
| L2 | Dead imports | ALREADY APPLIED | Lines 81-91: clean imports, no TRANSPLANT_DATE/PATIENT_AGE/ESC_RMSSD_DEFICIENCY/NOCTURNAL_HR_ELEVATED/BG_SURFACE/COLORWAY/THEME_STATUS_COLORS |
| L4 | Feature importance labels missing units | ALREADY APPLIED | Lines 1948-1966: all labels include units |
| L6 | html.escape for fallback_reason | ALREADY APPLIED | Line 35: `import html as _html_escape_mod`, line 2404: `_html_escape_mod.escape()` |

## No changes made - file is already in the target state.
