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

**Verdict:** The commit addressed its stated scope. But several issues remain unaddressed in other scripts.

---

## 2. FINDINGS — Issues Requiring Fixes

### ISSUE-01: analyze_oura_biomarkers.py — Hover templates missing units for composite scores
- **Line 859:** `hovertemplate=f"<b>%{{x|%b %d}}</b><br>{label}: %{{y:.1f}}<extra></extra>"`
  - Used generically for biomarker lines. The `{label}` already contains context but no unit suffix in the hover value itself.
  - **Severity:** LOW — the label provides context, and composite scores are unitless (0-100).
- **Line 890:** Same pattern for 7d averages.

### ISSUE-02: analyze_oura_biomarkers.py — Missing units on y-axis for biomarker time series
- The multi-panel biomarker figure uses shared y-axis but no explicit title on the panels.
- **Severity:** LOW — panels are labeled in the legend.

*(Investigating further...)*

---

## 3. Temperature Units — Comprehensive Check

All temperature references now use °C correctly:
- `analyze_oura_gvhd_predict.py` lines 489-499, 563, 602, 635, 646, 651, 668, 697, 727-729: ALL °C (CORRECT)
- `analyze_oura_full.py` line 990, 1013: °C (CORRECT)
- `analyze_oura_causal.py` line 524: °C (CORRECT)
- `analyze_oura_digital_twin.py` line 1923, 2147: °C (CORRECT)
- `analyze_oura_advanced_hrv.py` lines 1056, 1097, 1106: °C (CORRECT)
- `analyze_oura_spo2_trend.py` line 1284: Unicode \u00b0C (CORRECT)
- `generate_oura_3d_dashboard.py` line 557, 567, 576, 682, 1770: °C/\u00b0C (CORRECT)
- `generate_roadmap.py` lines 74-75: °C (CORRECT)

**Verdict: NO remaining "deg" or plain-text degree approximations found. All temperature labels use proper °C.**

---

## 4. BDI / Breathing Disturbance Units — Comprehensive Check

- `analyze_oura_spo2_trend.py` line 73: Comment says "events/hour" (CORRECT)
- `analyze_oura_spo2_trend.py` line 1169: `name="BDI (events/hour)"` (CORRECT)
- `analyze_oura_spo2_trend.py` line 1172: hover `events/hour` (CORRECT)
- `analyze_oura_spo2_trend.py` line 1184: hover `events/hour` (CORRECT)
- `analyze_oura_spo2_trend.py` line 1212: y-axis `BDI (events/hour)` (CORRECT)
- `analyze_oura_spo2_trend.py` line 1561: KPI `events/hour` (CORRECT)
- `analyze_oura_spo2_trend.py` line 1700: text `events/hour` (CORRECT)

**Verdict: NO remaining "events/hr" abbreviations. All use "events/hour".**

---

## 5. Respiratory Rate Units — Comprehensive Check

- `analyze_oura_full.py` line 844: hover `breaths/min` (CORRECT)
- `analyze_oura_full.py` line 874: y-axis "RMSSD (ms) / Respiratory rate (breaths/min)" (CORRECT)
- `analyze_oura_causal.py` line 520: label "Respiratory rate (breaths/min)" (CORRECT)
- `analyze_oura_gvhd_predict.py` line 1947: "Respiratory rate (breaths/min)" (CORRECT)

**Verdict: All respiratory rate labels use "breaths/min". No "breath/min" or "resp/min" found.**

---

## 6. HRV Units (ms) — Comprehensive Check

(Auditing all RMSSD and HRV labels...)

---

## 7. Heart Rate Units (bpm) — Comprehensive Check

(Auditing all HR labels...)

---

## 8. SpO2 Units (%) — Comprehensive Check

(Auditing all SpO2 labels...)

---

## 9. Date Format Consistency

(Auditing all date formats...)

---

## 10. Number Formatting

(Auditing decimal places and separators...)

---

## 11. Legend Label Consistency

(Auditing trace names...)

---

*Audit in progress — sections 6-11 being populated...*
