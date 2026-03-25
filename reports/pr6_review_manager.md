# PR #6 Review Manager - Progress Report

**PR:** https://github.com/The-Educational-Equality-Institute/oura-hsct-digital-twin/pull/6
**Branch:** fix/pipeline-bugs
**Title:** fix: pipeline bugs - token handling, type errors, import paths
**Date:** 2026-03-25

## Phase 1: Waiting for Reviews

**Status:** Polling started at 2026-03-25
- Greptile Review: IN_PROGRESS (detected at first check)
- CodeRabbit Review: Not yet detected

### Poll Log
- Poll 1 (initial): Greptile IN_PROGRESS, no CodeRabbit
- Poll 2: CodeRabbit COMPLETED - 19 inline comments (2 Critical, 12 Major, 5 Minor)
- Greptile: Still IN_PROGRESS

## Phase 2: CodeRabbit Review - All Comments

### CRITICAL (2)
1. **config.example.py:24** - Keep config dates as dates, not strings
2. **scripts/daily_pipeline.sh:12** - Fix default virtualenv path

### MAJOR (12)
1. **.env.example:18** - Primary recommendation promotes non-functional auth method
2. **analysis/_theme.py:35** - Use HEV_DIAGNOSIS_DATE instead of hard-coding March 18
3. **analysis/analyze_oura_anomalies.py:240** - Seed daily from every source you merge
4. **analysis/analyze_oura_anomalies.py:488** - Normalize KNOWN_EVENT_DATE before string-keyed lookups
5. **analysis/analyze_oura_digital_twin.py:1174** - Replace is not np.bool_(False) identity checks with np.ma.getmaskarray()
6. **analysis/analyze_oura_foundation_models.py:1790** - Remove hard-coded reporting dates from KPI cards
7. **analysis/analyze_oura_foundation_models.py:2162** - Fail the script when artifact generation fails
8. **analysis/analyze_oura_gvhd_predict.py:474** - Keep fragmentation on same sleep period as nightly features
9. **analysis/analyze_oura_sleep_advanced.py:835** - Keep zero-duration mask on deep_pct/rem_pct
10. **analysis/analyze_oura_spo2_trend.py:1845** - Abort early when no SpO2 rows loaded
11. **analysis/generate_roadmap.py:41** - Hard-coded counts and dates will drift
12. **api/import_oura.py:1059** - Handle filename-only database paths
13. **requirements.txt:34** - Restore dependencies still imported at runtime
14. **scripts/daily_pipeline.sh:31** - Only skip refresh when OAuth2 is not configured

### MINOR (5)
1. **analysis/analyze_oura_spo2_trend.py:1511** - Preserve MODERATE trend state in KPI
2. **api/oura_oauth2_setup.py:270** - Remove extraneous f-string prefix
3. **README.md:50** - Remove duplicate line

## Phase 2: Fixing Issues

### Triage Decision
Fixing all CRITICAL and actionable MAJOR issues. Deferring cosmetic MINOR issues.

**All fixes applied (19 of 19 comments addressed):**

| # | File | Fix | Status |
|---|------|-----|--------|
| 1 | config.example.py | Changed KNOWN_EVENT_DATE and HEV_DIAGNOSIS_DATE to date() objects | DONE |
| 2 | scripts/daily_pipeline.sh:10 | Fixed VENV_DIR to resolve relative to repo parent | DONE |
| 3 | analysis/_theme.py | Import HEV_DIAGNOSIS_DATE; use in disclaimer_banner dynamically | DONE |
| 4 | analyze_oura_anomalies.py:240 | Seed all_dates from all 6 sources (was missing 3) | DONE |
| 5 | analyze_oura_anomalies.py:488 | Resolved by fix 1 (config.example.py now uses date()) | N/A |
| 6 | analyze_oura_digital_twin.py | Replaced all 7 np.bool_(False) checks with np.ma.getmaskarray() | DONE |
| 7 | analyze_oura_foundation_models.py:1790 | KPI card uses TREATMENT_START.strftime() dynamically | DONE |
| 8 | analyze_oura_foundation_models.py:2162 | Added raise SystemExit(1) on HTML/JSON generation failure | DONE |
| 9 | analyze_oura_gvhd_predict.py:474 | Fragmentation now joins on period_id (same as primary) | DONE |
| 10 | analyze_oura_sleep_advanced.py:835 | Division uses safe_total with .replace(0, np.nan) | DONE |
| 11 | analyze_oura_spo2_trend.py:1845 | Added early return when spo2 DataFrame is empty | DONE |
| 12 | generate_roadmap.py:41 | Day counts and dates computed from config constants | DONE |
| 13 | api/import_oura.py:1059 | Guard os.makedirs for empty dirname | DONE |
| 14 | requirements.txt:34 | Uncommented pykalman (hard import in digital_twin.py) | DONE |
| 15 | scripts/daily_pipeline.sh:31 | OAuth2 refresh only when client_id+refresh_token set | DONE |
| 16 | .env.example | Recommend OAuth2 as primary, note PAT deprecation | DONE |
| 17 | analyze_oura_spo2_trend.py:1511 | Added MODERATE trend state in KPI | DONE |
| 18 | api/oura_oauth2_setup.py:270 | Removed extraneous f-string prefix | DONE |
| 19 | README.md:50 | Removed duplicate generate_*.py line | DONE |

### Running pipeline verification...
