# Fix Log: analyze_oura_full.py

Date: 2026-03-24

## FIX C2 — oura_sleep table empty, data in oura_sleep_periods

### Diagnosis
- `oura_sleep` has 73 rows. Only `date`, `score` (73/73), and `hrv_average` (70/73) have data.
- All other columns (total_sleep_duration, rem_sleep_duration, deep_sleep_duration, light_sleep_duration, efficiency, hr_lowest, hr_average, breath_average, temperature_delta) are 100% NULL.
- `oura_sleep_periods` has 89 rows with all biometric data populated.
- `oura_sleep_periods` uses `day` instead of `date`, and has different column names (e.g. `average_hrv` vs `hrv_average`, `lowest_heart_rate` vs `hr_lowest`).
- `oura_sleep_periods` does NOT have `score` or `temperature_delta`.

### Column mapping
| oura_sleep | oura_sleep_periods |
|---|---|
| score | (not available - keep from oura_sleep JOIN) |
| total_sleep_duration | total_sleep_duration |
| rem_sleep_duration | rem_sleep_duration |
| deep_sleep_duration | deep_sleep_duration |
| light_sleep_duration | light_sleep_duration |
| efficiency | efficiency |
| hr_lowest | lowest_heart_rate |
| hr_average | average_heart_rate |
| hrv_average | average_hrv |
| breath_average | average_breath |
| temperature_delta | (not available - dropped) |

### Fix applied
Changed lines 125-132: Query now JOINs oura_sleep (for score) with oura_sleep_periods (for biometrics), using a subquery to select only the longest sleep period per day.

## FIX M6 — SpO2 axis label
- Line 1160: Changed `"SpO2 %"` to `"SpO2 (%)"` with parenthesized unit.

## Status: APPLIED
