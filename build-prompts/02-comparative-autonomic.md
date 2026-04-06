# Build Task: analysis/analyze_comparative_autonomic.py

## What to build

Create `analysis/analyze_comparative_autonomic.py` — Module 1: Autonomic Recovery Trajectories. Compares HRV and resting HR recovery patterns between Henrik (post-HSCT) and Mitchell (post-stroke), normalized to days-since-their-major-event.

## Project location

`/home/henrik/projects/teei/oura-hsct-digital-twin/`

## Output files

- `reports/comparative_autonomic_report.html` (interactive Plotly dashboard)
- `reports/comparative_autonomic_metrics.json` (structured metrics)

## Patient context

- **Henrik**: Post-HSCT (Nov 23, 2023). Data: Jan 8 – Apr 4, 2026 (days ~777-857 post-HSCT). HRV ~9ms (severe autonomic dysfunction — ESC threshold is 15ms). Ruxolitinib started Mar 16, 2026. Ring Gen 4.
- **Mitchell**: Post-stroke (~Dec 2024). Data: Feb 2021 – Apr 2026 (518 days). HRV ~43ms declining to ~43ms recently. Resting HR improved from 54→49 bpm over 5 years. Ring Gen 3.

## Database schema

Same as described in the _comparative_utils prompt. Key tables: `oura_sleep` (hrv_average, hr_lowest, hr_average), `oura_sleep_periods` (average_hrv, average_heart_rate, lowest_heart_rate — use type='long_sleep'), `oura_readiness` (score, recovery_index).

**CRITICAL**: Henrik's `oura_sleep` has NULL for `hr_average` and `hr_lowest`. Must use `oura_sleep_periods` (type='long_sleep') as primary HR source for Henrik. Mitchell's `oura_sleep` is fully populated.

**CRITICAL**: `oura_readiness.resting_heart_rate` is a 0-100 SCORE, not bpm. Never use it as actual heart rate.

## Imports available

```python
from _comparative_utils import (
    PatientConfig, default_patients, load_patient_data,
    zscore_normalize, zscore_both, percentile_of_self,
    find_date_overlap, align_by_event, days_since_event,
    compare_distributions, dual_patient_timeseries,
    dual_patient_distribution, event_aligned_comparison,
)
from _theme import (
    wrap_html, make_kpi_card, make_kpi_row, make_section,
    disclaimer_banner, format_p_value, COLORWAY, STATUS_COLORS,
    BG_PRIMARY, BG_SURFACE, ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED,
    ACCENT_AMBER, ACCENT_PURPLE, ACCENT_CYAN, TEXT_PRIMARY, TEXT_SECONDARY,
)
from _hardening import safe_connect, safe_read_sql, section_html_or_placeholder
```

Also import from `config.py`: `REPORTS_DIR`, `ESC_RMSSD_DEFICIENCY` (15), `POPULATION_RMSSD_MEDIAN` (49), `POPULATION_RMSSD_MEAN` (42.0), `POPULATION_RMSSD_SD` (15.0)

## Core analyses to implement

### 1. Data Loading
- Load both patients' HRV + HR data, handling Henrik's NULL HR in oura_sleep by falling back to oura_sleep_periods
- Merge sleep + readiness data per patient

### 2. Timeline Normalization
- Add `days_since_event` column for each patient
- Henrik: days since HSCT (2023-11-23) → his data is at days 777-857
- Mitchell: days since stroke (~2024-12-15) → his data spans days 68-1572

### 3. Rolling Metrics
- 7-day and 14-day rolling means for HRV and HR
- HRV coefficient of variation (7-day rolling std/mean) — measures autonomic stability
- 30-day trailing linear regression slope for HRV (ms/day trend)

### 4. Three normalization approaches (to handle 9ms vs 43ms gap)
- **Z-score**: `(value - patient_mean) / patient_std` — both center on 0
- **Percent-of-baseline**: `(value / first_14d_mean) * 100` — shows improvement/decline from start
- **Population percentile**: Map HRV to log-normal population distribution (mean=42, SD=15)

### 5. Trend Statistics
- Linear regression slope + p-value for last 30 days
- Mann-Kendall trend test for monotonic trend
- Direction label: improving/stable/declining

### 6. Autonomic Severity Comparison
- Mean HRV ratio (Mitchell/Henrik)
- HR gap (mean sleep HR difference)
- Days below ESC threshold (<15ms) — Henrik nearly all, Mitchell essentially zero
- Convergence analysis: are they getting closer or further apart?

## Visualizations (6 Plotly figures)

1. **HRV Trajectory** — Dual panel: top = raw values (use LOG y-axis since 9ms vs 43ms), bottom = z-score normalized. Reference lines at ESC threshold (15ms) and population median (49ms). Use `add_shape` + `add_annotation` for event markers, NOT `add_vline`.
2. **HR Trajectory** — Same structure for resting/sleep heart rate (linear scale is fine here — 79 vs 62 bpm)
3. **Percent-of-Baseline** — Both patients' HRV as % of their first 14 days. X-axis = days into observation. Shows recovery/decline shape.
4. **HRV Distribution** — Overlapping violin plots with population reference distribution as shaded band
5. **Long-Term Context** — Mitchell's 5-year HRV trajectory with Henrik's 81-day window overlaid at the corresponding days-since-event position
6. **Autonomic Coupling (HR vs HRV)** — Scatter plot, one point per day per patient. Shows inverse relationship (higher HR = lower HRV). Compare slope positions.

## HTML Report Structure

Use `wrap_html` from `_theme.py` with `report_id="comp_autonomic"`.

Sections:
1. KPI row: Henrik Mean HRV (critical), Mitchell Mean HRV (info), Henrik Sleep HR (warning), Mitchell Sleep HR (normal), HRV Gap Ratio, Trajectory Status
2. Section: Autonomic Recovery Trajectories (fig 1)
3. Section: Heart Rate Comparison (fig 2)
4. Section: Relative Recovery (fig 3)
5. Section: HRV Distribution (fig 4)
6. Section: Long-Term Context (fig 5)
7. Section: Autonomic Coupling (fig 6)
8. Section: Clinical Context — disclaimer about different conditions

## JSON Output Structure

```json
{
  "report": "comparative_autonomic",
  "generated_at": "...",
  "patients": {
    "henrik": {
      "label": "...", "event": "HSCT", "event_date": "2023-11-23",
      "data_days": 81, "days_since_event_range": [777, 857],
      "hrv": { "mean", "median", "std", "min", "max", "trend_slope_per_week", "trend_p_value", "trend_direction", "pct_below_esc_threshold", "population_percentile" },
      "heart_rate": { "mean_sleep_hr", "mean_lowest_hr", "trend_slope_per_week", "trend_direction" }
    },
    "mitchell": { ... same structure ... }
  },
  "comparison": {
    "hrv_ratio", "hrv_gap_ms", "hr_gap_bpm", "trajectories_converging",
    "severity_classification": { "henrik": "severe_autonomic_dysfunction", "mitchell": "mild_autonomic_impairment" }
  }
}
```

## main() structure

Follow the existing scripts' pattern:
1. [1/7] Load data
2. [2/7] Normalize timelines
3. [3/7] Compute rolling metrics
4. [4/7] Compute normalized metrics
5. [5/7] Compute trends and comparison
6. [6/7] Generate HTML report
7. [7/7] Export JSON

Return 0 on success.

## Plotly rules

- Use `pio.templates.default = "clinical_dark"` at module level
- Font: use FONT_FAMILY from config.py or fall back to "Inter, Source Sans 3, sans-serif"
- NEVER use `add_vline` with `annotation_text` on datetime axes — always `add_shape` + `add_annotation`
- Embed figures: `fig.to_html(include_plotlyjs=False, full_html=False)` — the CDN script is included by `wrap_html`
