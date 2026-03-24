# Unit Label & Formatting Audit

**Date:** 2026-03-24
**Scope:** All 11 analysis scripts + _theme.py, _bos_risk.py, _config.py, generate_roadmap.py in `analysis/`
**Commit reviewed:** 59f2fa3 ("Fix unit labels: degree symbols, events/hour, breaths/min")

---

## 1. Commit 59f2fa3 Review

The commit fixed issues in 3 files:
- `analyze_oura_digital_twin.py` - spacing in "p = 0.05" -> "p=0.05" (cosmetic, not unit-related)
- `analyze_oura_gvhd_predict.py` - Fixed "C" -> "°C" in hover templates, annotations, and y-axis titles; fixed "Breathing Rate" -> "Breathing Rate (breaths/min)"
- `analyze_oura_spo2_trend.py` - Fixed "events/hr" -> "events/hour" in hovertemplates; added "events/hour" unit to BDI Mean KPI card

**Verdict:** The commit addressed its stated scope correctly. Several lower-priority issues remain in other scripts (detailed below).

---

## 2. ISSUES FOUND (sorted by severity)

### ISSUE-01: analyze_oura_full.py — "SpO2 %" y-axis title inconsistent [MEDIUM]
- **File:** `analyze_oura_full.py`, line 1160
- **Current:** `fig.update_yaxes(title_text="SpO2 %", ...)`
- **Expected:** `"SpO2 (%)"` — parenthesized unit, consistent with all other SpO2 axis titles
- **Compare:** `analyze_oura_spo2_trend.py` lines 1148, 1486-1487 all use `"SpO2 (%)"`

### ISSUE-02: analyze_oura_foundation_models.py — "Dato" x-axis title (Norwegian) [MEDIUM]
- **File:** `analyze_oura_foundation_models.py`, lines 1363 and 1597
- **Current:** `xaxis_title="Dato"` (Norwegian)
- **Expected:** `"Date"` — all other scripts use English axis labels
- **This is the ONLY script using Norwegian for axis titles.**

### ISSUE-03: analyze_oura_gvhd_predict.py — Feature importance labels missing units [LOW]
- **File:** `analyze_oura_gvhd_predict.py`, lines 1931-1946
- Several feature labels lack unit suffixes:
  - "Temperature Deviation" -> should be "Temperature Deviation (°C)"
  - "Temperature Variability (7d)" -> should be "Temperature Variability (°C)"
  - "Temperature Gradient" -> should be "Temperature Gradient (°C/night)"
  - "HRV Variability" -> should be "HRV Variability (ms)"
  - "Sleep Heart Rate" -> should be "Sleep Heart Rate (bpm)"
  - "Lowest HR" -> should be "Lowest HR (bpm)"
  - "Sleep Efficiency" -> should be "Sleep Efficiency (%)"
  - "SpO2 Average" -> should be "SpO2 Average (%)"
- Note: Some already have units: "Stress High (sec)", "Respiratory rate (breaths/min)", "HRV Median (RMSSD)"
- **These appear in the feature importance bar chart (x-axis labels).**

### ISSUE-04: Date format inconsistency in tick formats [LOW]
- Two competing date formats for axis tick labels:
  - `"%d %b"` (day-first, e.g., "24 Mar") — used by **28 instances** across 4 files:
    - `analyze_oura_full.py` (18), `generate_oura_3d_dashboard.py` (6), `analyze_oura_biomarkers.py` (2), `analyze_oura_anomalies.py` (2)
  - `"%b %d"` (month-first, e.g., "Mar 24") — used by **3 instances** across 2 files:
    - `analyze_oura_causal.py` (2), `analyze_oura_sleep_advanced.py` (1)
- Hover templates are more consistent, predominantly using `"%b %d"` or `"%b %d, %Y"`.
- **Recommendation:** Standardize all tick formats to `"%d %b"` (the majority) or define in `_theme.py`.

### ISSUE-05: Hover date format variation [LOW]
- **Primary patterns found:**
  - `%b %d` (no year) — 85 instances (most common for short hovers)
  - `%b %d, %Y` (with year) — 14 instances (used in spo2_trend, foundation_models, gvhd_predict)
  - `%Y-%m-%d` (ISO format) — 14 instances (used exclusively in digital_twin)
  - `%d %b %Y` — 1 instance (analyze_oura_advanced_hrv.py line 2207)
  - `%b %d %H:%M` — 3 instances (hourly data in foundation_models)
- **Inconsistency:** `analyze_oura_digital_twin.py` uses ISO `%Y-%m-%d` for all hovers while all other scripts use `%b %d` or `%b %d, %Y`.
- **Recommendation:** This is a stylistic choice. The digital_twin script's ISO format is intentional for precision. Not a bug.

### ISSUE-06: analyze_oura_gvhd_predict.py — "Breathing Rate" label inconsistency [FIXED in 59f2fa3]
- Line 1947 now correctly shows "Respiratory rate (breaths/min)".
- However, elsewhere in the composite config the label "Breathing Rate" is not used — the stream_config at line 979-986 uses descriptive names with percentages. No remaining issue.

---

## 3. Temperature Units — Comprehensive Check

All temperature references now use °C correctly:
- `analyze_oura_gvhd_predict.py` lines 489-499, 563, 602, 635, 646, 651, 668, 697, 727-729: ALL °C
- `analyze_oura_full.py` line 990, 1013: °C
- `analyze_oura_causal.py` line 524: °C
- `analyze_oura_digital_twin.py` line 1923, 2147: °C
- `analyze_oura_advanced_hrv.py` lines 1056, 1097, 1106: °C
- `analyze_oura_spo2_trend.py` line 1284: Unicode \u00b0C (°C)
- `generate_oura_3d_dashboard.py` line 557, 567, 576, 682, 1770, 1797: °C/\u00b0C
- `generate_roadmap.py` lines 74-75: °C

**PASS: No remaining "deg" or plain-text degree approximations. All use proper °C.**

---

## 4. BDI / Breathing Disturbance Units — Comprehensive Check

All BDI references use "events/hour":
- `analyze_oura_spo2_trend.py` lines 73, 1169, 1172, 1184, 1212, 1561, 1700

**PASS: No remaining "events/hr" abbreviations.**

---

## 5. Respiratory Rate Units — Comprehensive Check

All respiratory rate references use "breaths/min":
- `analyze_oura_full.py` line 844 (hover), 874 (y-axis)
- `analyze_oura_causal.py` line 520 (label + unit)
- `analyze_oura_gvhd_predict.py` line 1947 (feature label)

**PASS: All consistent.**

---

## 6. HRV Units (ms) — Comprehensive Check

Axis titles with HRV/RMSSD units:
- `analyze_oura_full.py`: "RMSSD (ms)" lines 563, 564, 566, 570, 572, 874, 1355
- `analyze_oura_anomalies.py`: "RMSSD (ms)" lines 1462, 1705; "EWMA (ms)" line 1707
- `analyze_oura_advanced_hrv.py`: "RMSSD(n) [ms]" line 2066; "RMSSD(n+1) [ms]" line 2067; "RMSSD (ms)" line 1309
- `analyze_oura_digital_twin.py`: "RMSSD (ms)" line 1608; "Predicted (ms)" line 1603; "Actual (ms)" line 1601
- `analyze_oura_causal.py`: label "HRV mean RMSSD (ms)" line 512; "HRV max RMSSD (ms)" line 514
- `analyze_oura_biomarkers.py`: name "RMSSD (ms)" line 957; hover "RMSSD: %{y:.1f} ms" line 959
- `generate_oura_3d_dashboard.py`: "ms" line 453, 1443, 1731

Hover templates all correctly show "ms" for RMSSD values.

**PASS: All HRV labels consistently use "ms".**

---

## 7. Heart Rate Units (bpm) — Comprehensive Check

Axis titles:
- `analyze_oura_full.py`: "BPM" lines 708, 710, 872, 1011; "Heart Rate (bpm)" line 706
- `analyze_oura_anomalies.py`: "HR (bpm)" line 1463
- `analyze_oura_advanced_hrv.py`: "Heart Rate (bpm)" line 2169; "Mean Nocturnal HR (bpm)" line 2269
- `analyze_oura_digital_twin.py`: "Predicted (bpm)" line 1604; "Actual (bpm)" line 1602
- `analyze_oura_foundation_models.py`: "Heart Rate (bpm)" line 1691
- `analyze_oura_causal.py`: unit "bpm" lines 516, 518
- `generate_oura_3d_dashboard.py`: "bpm" line 1797

Note: `analyze_oura_full.py` uses short "BPM" (lines 708, 710, 872, 1011) while others use "bpm" or "Heart Rate (bpm)".

**PASS with note: "BPM" vs "bpm" capitalization varies. Not a unit error, but inconsistent casing.**

---

## 8. SpO2 Units (%) — Comprehensive Check

Axis titles:
- `analyze_oura_spo2_trend.py`: "SpO2 (%)" lines 905, 1148, 1284, 1486, 1487
- `analyze_oura_full.py`: **"SpO2 %" line 1160** (ISSUE-01: missing parentheses)
- `analyze_oura_causal.py`: unit "%" line 522
- `generate_oura_3d_dashboard.py`: "%" line 1797

Hover templates all use either `%{y:.1f}%` or `SpO2: %{y:.1f}%`.

**One issue found (ISSUE-01). Otherwise consistent.**

---

## 9. Date Format Consistency — Summary

### Tick formats (axis labels):
| Format | Count | Files |
|--------|-------|-------|
| `%d %b` | 28 | full, 3d_dashboard, biomarkers, anomalies |
| `%b %d` | 3 | causal, sleep_advanced |

**The majority use `%d %b` (day-first). Two scripts deviate.**

### Hover formats:
| Format | Count | Context |
|--------|-------|---------|
| `%b %d` | ~85 | Short hover, no year |
| `%b %d, %Y` | 14 | Long hover, with year |
| `%Y-%m-%d` | 14 | digital_twin only (ISO) |
| `%d %b %Y` | 1 | advanced_hrv (single instance) |

**The hover format split is: all scripts use `%b %d` except digital_twin (ISO) and one instance in advanced_hrv.**

---

## 10. Number Formatting

Decimal places are generally consistent within each metric type:
- RMSSD values: `.1f` (1 decimal) — consistent across all scripts
- Heart rate: `.0f` (integer) — consistent
- SpO2: `.1f` (1 decimal) — consistent
- Temperature: `.2f` or `.3f` depending on context (deviation vs variability) — appropriate
- Composite scores: `.1f` — consistent
- P-values: `.4f` or `.6f` — appropriate for significance
- Percentages: `.1f` or `.0f` — appropriate

**No thousands separators are used for display numbers (correct for medical values).**
**One exception: `analyze_oura_full.py` line 1211 uses `%{y:,.0f}` for steps (with comma separator) — appropriate.**

**PASS: Number formatting is consistent within metric types.**

---

## 11. Legend Label Consistency

### Trace naming conventions:
Most trace names are descriptive and consistent. Notable patterns:
- "RMSSD" used consistently (not "HRV" in trace names where RMSSD is the actual metric)
- Temperature traces: "Temp Deviation" (full.py, gvhd, 3d_dashboard) — consistent
- SpO2 traces: "SpO2" or "Nightly SpO2" — consistent
- HR traces: "Daily Mean HR", "Avg HR (sleep)", "Lowest HR" — clear
- Sleep: "Sleep Score", "Sleep Hours", "Efficiency %" — clear

**PASS: Legend labels are clear and reasonably consistent.**

---

## 12. ASCII Approximations of Special Characters

Searched for: "deg", "degrees", raw "C" where °C should be, ASCII arrows, etc.

- No "deg " or "degrees" found in any unit context (only `np.degrees()` math function calls and CSS `deg` values, both correct)
- No plain "C" used where °C should be (all fixed in commit 59f2fa3)
- Unicode \u00b0 used correctly in spo2_trend and 3d_dashboard for degree symbol
- Delta symbol (Δ) used correctly in gvhd_predict line 729: "Δ°C/night"

**PASS: No remaining ASCII approximations of special characters.**

---

## Summary

| Category | Status | Issues |
|----------|--------|--------|
| Temperature (°C) | PASS | None |
| BDI (events/hour) | PASS | None |
| Respiratory (breaths/min) | PASS | None |
| HRV (ms) | PASS | None |
| Heart Rate (bpm) | PASS | Minor BPM/bpm casing |
| SpO2 (%) | **1 FIX** | ISSUE-01: "SpO2 %" should be "SpO2 (%)" in full.py |
| Degree symbols | PASS | None |
| ASCII approximations | PASS | None |
| Date tick formats | **INCONSISTENT** | ISSUE-04: 2 scripts use %b %d, 4 use %d %b |
| Hover date formats | Acceptable | digital_twin intentionally uses ISO |
| Language | **1 FIX** | ISSUE-02: "Dato" in foundation_models.py |
| Feature labels | **LOW** | ISSUE-03: some labels missing units in gvhd feature importance |
| Number formatting | PASS | Consistent within types |
| Legend labels | PASS | Clear and consistent |

### Fixes recommended:
1. **ISSUE-01** (MEDIUM): `analyze_oura_full.py` line 1160 — change `"SpO2 %"` to `"SpO2 (%)"`
2. **ISSUE-02** (MEDIUM): `analyze_oura_foundation_models.py` lines 1363, 1597 — change `"Dato"` to `"Date"`
3. **ISSUE-03** (LOW): `analyze_oura_gvhd_predict.py` lines 1931-1942 — add units to feature importance labels
4. **ISSUE-04** (LOW): Standardize tick date format to `"%d %b"` in `analyze_oura_causal.py` and `analyze_oura_sleep_advanced.py`
