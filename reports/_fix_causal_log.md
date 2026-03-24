# Fix Log: analyze_oura_causal.py

Date: 2026-03-24

## FIX H2 - HRV null/zero filtering (line 279)

**Problem:** The SQL query loading HRV data from `oura_hrv` did not filter out null or zero RMSSD values. These invalid readings could corrupt downstream causal inference calculations.

**Fix:** Added pandas filter after loading and coercing to numeric:
```python
hrv = hrv[hrv["rmssd"].notna() & (hrv["rmssd"] > 0)]
```
This is consistent with the script's existing style (load from SQL, coerce with `pd.to_numeric`, then filter in pandas).

## FIX M2 - Hardcoded HEV date in HTML (line 3376)

**Problem:** The string `2026-03-18` was hardcoded in the HTML limitations section instead of using the config variable.

**Fix:**
1. Added `HEV_DIAGNOSIS_DATE` to the import block from `config` (line 101).
2. Replaced the hardcoded `2026-03-18` with `{HEV_DIAGNOSIS_DATE}` inside the existing f-string.

## FIX L5 - Date tick format inconsistency (6 occurrences)

**Problem:** Date formatting used `%b %d` (month-first, e.g., "Mar 24") instead of the project-standard `%d %b` (day-first, e.g., "24 Mar").

**Fix:** Changed all 6 occurrences of `%b %d` to `%d %b`:
- 4 hover templates (lines 1046, 1066, 1088, 1117)
- 2 x-axis tickformat settings (lines 1159, 1165)

## Verification

- Zero remaining `%b %d` patterns in the file.
- Zero remaining `2026-03-18` hardcoded strings in the file.
- `HEV_DIAGNOSIS_DATE` correctly imported and used inside an f-string context.
- HRV null/zero filter placed immediately after `pd.to_numeric` coercion, before any aggregation.
