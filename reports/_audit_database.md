# Database Schema & Data Integrity Audit

**Date:** 2026-03-24
**Database:** `database/helseoversikt.db`
**Symlink:** `oura-digital-twin/data/oura.db` -> `../../database/helseoversikt.db`

---

## 1. Table Inventory (21 oura_* tables)

| Table | Rows | Date Column | Range Start | Range End | Days Span |
|-------|------|-------------|-------------|-----------|-----------|
| oura_hrv | 5,986 | timestamp | 2026-01-08 | 2026-03-23 | 74 |
| oura_heart_rate | 29,263 | timestamp | 2026-01-04 | 2026-03-22 | 77 |
| oura_sleep | 73 | date | 2026-01-05 | 2026-03-23 | 77 |
| oura_sleep_periods | 89 | day | 2026-01-08 | 2026-03-23 | 74 |
| oura_sleep_epochs | 6,436 | (via period_id) | - | - | - |
| oura_sleep_movement | 64,057 | (via period_id) | - | - | - |
| oura_sleep_hr_timeseries | 5,896 | timestamp | 2026-01-09 | 2026-03-23 | 73 |
| oura_readiness | 73 | date | 2026-01-05 | 2026-03-23 | 77 |
| oura_spo2 | 69 | date | 2026-01-08 | 2026-03-23 | 74 |
| oura_activity | 78 | date | 2026-01-04 | 2026-03-22 | 77 |
| oura_stress | 74 | date | 2026-01-09 | 2026-03-23 | 73 |
| oura_cardiovascular_age | 60 | date | 2026-01-08 | 2026-03-23 | 74 |
| oura_resilience | 62 | date | 2026-01-15 | 2026-03-23 | 67 |
| oura_personal_info | 1 | (fetched_at) | - | - | - |
| oura_ring_config | 1 | (set_up_at) | - | - | - |
| oura_workouts | 1 | day | 2026-01-26 | 2026-01-26 | 0 |
| oura_sessions | 0 | day | - | - | - |
| oura_sleep_time | 0 | date | - | - | - |
| oura_rest_mode | 0 | - | - | - | - |
| oura_tags | 0 | timestamp | - | - | - |
| oura_vo2_max | 0 | date | - | - | - |

---

## 2. CRITICAL: Date Range Coverage Gap

**Expected analysis period:** 2023-11-23 (HSCT) through 2026-03-24 (today)
**Actual data range:** 2026-01-04 through 2026-03-23

**ZERO rows exist before 2026-01-01 in any table.** The Oura ring data covers ~80 days only. There is no pre-transplant, early-recovery, or pre-2026 data in the database. All analysis scripts correctly constrain queries to `DATA_START` (config: `2026-01-08`) through dynamically resolved `DATA_END`.

This is NOT a bug -- the ring was acquired in January 2026. But any analysis claiming to cover the full post-HSCT period should note this limitation.

---

## 3. CRITICAL: oura_sleep Table is Effectively Empty

The `oura_sleep` table has 73 rows but only TWO columns contain data:
- `score`: 73/73 populated
- `hrv_average`: 70/73 populated

**ALL other columns are 100% NULL:**
- total_sleep_duration, rem_sleep_duration, deep_sleep_duration, light_sleep_duration
- awake_time, efficiency, latency, restless_periods
- bedtime_start, bedtime_end, hr_lowest, hr_average, breath_average, temperature_delta

**Root cause:** The Oura API v2 Daily Sleep endpoint returns score and HRV summary only. Detailed sleep data comes from the Sleep Periods endpoint (stored in `oura_sleep_periods`).

**Impact on scripts:**
- `analyze_oura_full.py` queries `oura_sleep` for 12 columns -- gets score + NULLs for everything else. It also loads `oura_sleep_periods` separately. Depending on how the script handles NULLs, some sleep charts in the full report may be empty.
- `analyze_oura_anomalies.py` queries `oura_sleep` for `score` and `hrv_average` only -- this works correctly (70/73 usable rows).
- All other scripts use `oura_sleep_periods` correctly.

**Recommendation:** `analyze_oura_full.py` should be reviewed. Its `oura_sleep` DataFrame will have 11 NULL columns out of 12.

---

## 4. NULL Values in Critical Columns

### SEVERE (functional impact on analysis)
| Table.Column | NULL Count | Total | % NULL | Notes |
|-------------|-----------|-------|--------|-------|
| oura_sleep.total_sleep_duration | 73 | 73 | 100% | Table effectively empty (see above) |
| oura_sleep.hr_lowest | 73 | 73 | 100% | Same |
| oura_sleep.hr_average | 73 | 73 | 100% | Same |
| oura_sleep.breath_average | 73 | 73 | 100% | Same |
| oura_sleep.temperature_delta | 73 | 73 | 100% | Same |

### MODERATE (partial data loss)
| Table.Column | NULL Count | Total | % NULL | Notes |
|-------------|-----------|-------|--------|-------|
| oura_sleep_periods.average_hrv | 12 | 89 | 13.5% | Short naps/rest sessions lack HRV |
| oura_sleep_periods.average_heart_rate | 12 | 89 | 13.5% | Same pattern |
| oura_sleep_periods.average_breath | 13 | 89 | 14.6% | Same pattern |
| oura_sleep_periods.lowest_heart_rate | 12 | 89 | 13.5% | Same pattern |

**NULL sleep_periods pattern:** All 12 NULL-HRV rows are short sessions (type: `sleep`, `late_nap`, or `rest`) with durations 180-1440 seconds (3-24 minutes). These are too short for reliable biometric averages. The GVHD script does not filter by `type = 'long_sleep'`, so these NULLs flow into composite scores.

### MINOR
| Table.Column | NULL Count | Total | % NULL | Notes |
|-------------|-----------|-------|--------|-------|
| oura_sleep.hrv_average | 3 | 73 | 4.1% | First 3 days before ring calibrated |
| oura_readiness.resting_heart_rate | 1 | 73 | 1.4% | Single missing day |
| oura_spo2.breathing_disturbance_index | 1 | 69 | 1.4% | Single missing day |

---

## 5. Date Gaps in Daily Tables

Five date gaps found, consistent across oura_sleep, oura_readiness, and oura_spo2:

| Gap | Missing Date | Notes |
|-----|-------------|-------|
| 2026-01-24 | 1 day | Ring not worn? |
| 2026-01-28 | 1 day | Ring not worn? |
| 2026-02-10 | 1 day | Day after KNOWN_EVENT_DATE (2026-02-09) |
| 2026-03-16 | 1 day | Ruxolitinib start date |
| 2026-03-19 | 1 day | Day after HEV diagnosis (2026-03-18) |

The gap on 2026-02-10 (day after the known acute event) and 2026-03-16 (ruxolitinib start) are clinically significant -- these are precisely the days where data would be most valuable for causal/anomaly analysis.

oura_activity has NO gaps (78 contiguous days from Jan 4).

---

## 6. SpO2 Data Quality Issue

10 out of 69 SpO2 rows have `spo2_average = 0.0` (14.5%). All scripts correctly filter with `WHERE spo2_average > 0`, leaving 59 usable rows. However, no SpO2 values between 0 and 90 exist -- the distribution jumps from 0 to 90+. The 0 values are sentinel/missing indicators from the Oura API, not actual measurements.

After filtering zeros: min = 90.024, max = 97.338, mean = 96.0.

---

## 7. Timestamp Format Inconsistency

| Table | Format | Example |
|-------|--------|---------|
| oura_heart_rate | UTC (Z suffix) | `2026-01-04T18:08:18.551Z` |
| oura_hrv | Local TZ offset | `2026-01-08T02:15:57+01:00` |
| oura_sleep_hr_timeseries | Local TZ offset | `2026-03-17T02:35:30+01:00` |

**All 29,263 heart_rate rows use UTC.** All 5,986 HRV rows and 5,896 sleep_hr_timeseries rows use `+01:00` local offset.

Scripts handle this correctly via `pd.to_datetime(..., utc=True)` which normalizes both formats. But `substr(timestamp, 1, 10)` for date extraction returns different date strings depending on whether the timestamp is UTC or local (+01:00). For a measurement at `2026-01-09T00:30:00+01:00`, the date is correctly `2026-01-09`, but the same moment in UTC is `2026-01-08T23:30:00Z` which would extract as `2026-01-08`.

**Impact:** The GVHD and other scripts use `substr(timestamp, 1, 10)` for date filtering on HR data. Since HR is stored in UTC, a heart rate reading at 00:30 local time would be filtered against the *previous* day's date. This may cause minor date-boundary misalignment (~1 hour window).

---

## 8. Schema-Query Compatibility (All Scripts)

**All queries match the actual schema.** No missing column errors found.

| Script | Tables Queried | Status |
|--------|---------------|--------|
| analyze_oura_gvhd_predict.py | readiness, sleep_periods, spo2, hrv, heart_rate, activity, stress, sleep_epochs | ALL OK |
| analyze_oura_causal.py | hrv, heart_rate, sleep_periods, spo2, activity | ALL OK |
| analyze_oura_anomalies.py | hrv, heart_rate, sleep_periods, sleep (legacy), spo2, readiness | ALL OK |
| analyze_oura_biomarkers.py | sleep_periods, spo2, cardiovascular_age, readiness, stress | ALL OK |
| analyze_oura_full.py | hrv, heart_rate, sleep (legacy!), sleep_periods, readiness, spo2, stress, cardiovascular_age | ALL OK (but sleep is mostly NULL) |
| analyze_oura_advanced_hrv.py | sleep_periods, hrv, heart_rate, readiness, spo2 | ALL OK |
| generate_oura_3d_dashboard.py | heart_rate, hrv, spo2, readiness, activity | ALL OK |
| analyze_oura_foundation_models.py | hrv, heart_rate, spo2, readiness | ALL OK |
| _theme.py | heart_rate, sleep_periods, readiness (for DATA_END) | ALL OK |

---

## 9. Duplicate Dates in sleep_periods

14 dates have multiple sleep sessions (89 total rows for ~75 unique days). This is expected -- Oura records separate sessions for:
- `long_sleep` (primary overnight sleep)
- `sleep` (short sleep)
- `late_nap`
- `rest`

Scripts that need one row per day should filter `WHERE type = 'long_sleep'`. Currently:
- `analyze_oura_anomalies.py`: Correctly filters `WHERE type = 'long_sleep'`
- `analyze_oura_gvhd_predict.py`: Does NOT filter by type -- includes naps/rest in composite scores
- `analyze_oura_causal.py`: Does NOT filter by type

---

## 10. JOIN Integrity

| Relationship | Status |
|-------------|--------|
| sleep_epochs.period_id -> sleep_periods.period_id | 0 orphans (88/89 periods have epochs) |
| sleep_movement.period_id -> sleep_periods.period_id | 0 orphans |
| sleep_hr_timeseries.period_id -> sleep_periods.period_id | 0 orphans |

One sleep_period (`2026-01-08`, the earliest) has no epoch data. This is the first night's recording and may have been incomplete.

---

## 11. Data Freshness

All tables last imported: **2026-03-23 11:56:28-30**. Data is ~1 day old. The `oura_heart_rate` table ends at 2026-03-22 (slightly behind other tables due to API sync timing).

---

## Summary of Issues by Severity

### CRITICAL
1. **oura_sleep table 100% NULL** for 11/13 data columns. `analyze_oura_full.py` queries all of them.

### HIGH
2. **No pre-2026 data exists.** Analyses cannot cover the HSCT recovery period (Nov 2023 - Dec 2025).
3. **Timestamp format mismatch** between UTC (heart_rate) and local TZ (hrv, sleep_hr) -- may cause 1-hour date-boundary shift when using `substr()` extraction.

### MEDIUM
4. **GVHD script includes naps/rest** in sleep metrics (no `type` filter), introducing NULL values and short-session noise.
5. **5 date gaps** in daily tables, 2 at clinically significant moments (post-acute-event, treatment start).
6. **14.5% SpO2 sentinel zeros** -- correctly filtered by all scripts.

### LOW
7. 13.5% NULL biometrics in sleep_periods (short sessions only -- expected behavior).
8. One sleep_period with no epoch data (first recording day).
9. `oura_sleep.hrv_average` NULL for first 3 days (calibration period).
