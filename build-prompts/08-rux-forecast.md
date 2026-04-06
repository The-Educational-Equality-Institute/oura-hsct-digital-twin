# Build Task: analysis/analyze_rux_forecast.py

## What to build
Ruxolitinib dose-response timeline with forward projection. At the current rate of improvement, when would HRV cross 15ms (ESC threshold)? When would HR normalize? Give the doctor a forecast.

## Output
- `reports/rux_forecast.html`
- `reports/rux_forecast.json`

## Core analyses

### 1. Current trajectory computation
For each metric (HRV, lowest HR, avg HR, efficiency):
- Compute linear regression over post-Rux period (Mar 16 onwards)
- Compute slope (units per day, units per week)
- Extrapolate forward: at this rate, when does metric hit target?
  - HRV target: 15ms (ESC threshold), 25ms (HSCT range low), 36ms (population 25th percentile)
  - HR target: 70 bpm (normal), 65 bpm (good), 60 bpm (excellent)
  - Efficiency target: 85% (healthy threshold)

### 2. Confidence intervals on forecast
- Bootstrap the regression 1000 times
- Get 50th/75th/95th percentile crossing dates
- Report as range: "HRV expected to reach 15ms between May 12 and July 3 (95% CI)"

### 3. Three scenarios
- **Optimistic**: Use upper CI of slope (fastest improvement rate)
- **Expected**: Use median slope
- **Conservative**: Use lower CI of slope (slowest improvement)

### 4. Non-linear modeling
- Also fit exponential recovery curve: y = a * (1 - exp(-t/tau)) + baseline
- Common in physiological recovery — fast initial gains, slowing over time
- Compare linear vs exponential fit (AIC/BIC)
- Use the better model for the forecast

### 5. Historical context
- Show pre-acute baseline, post-acute recovery, post-Rux trajectory as distinct phases
- Three-period visualization with forecast extension

## Data notes
- Use oura_sleep_periods (type='long_sleep') for HR data (Henrik's oura_sleep has NULL hr)
- TREATMENT_START from config.py = date(2026, 3, 16)
- Current post-Rux days: ~21
- Small N warning: only ~20 post-Rux data points — forecasts have wide CIs, state this clearly

## Visualizations
1. **Metric trajectory + forecast** — For HRV and HR: actual data as scatter, regression line extending into future, CI bands, horizontal target lines at ESC threshold / normal range. Use add_shape for target lines, NOT add_vline.
2. **Milestone timeline** — Gantt-style: when each target is expected to be reached (with CI bars)
3. **Recovery curve comparison** — Linear vs exponential fit overlay
4. **Doctor-ready summary card** — "At current rate: HRV reaches ESC threshold by [date]. HR normalizes by [date]."

## JSON structure
```json
{
  "generated_at": "...",
  "days_on_ruxolitinib": 21,
  "forecasts": {
    "hrv_average": {
      "current_mean": 10.6,
      "slope_per_week": 0.3,
      "model": "linear",
      "targets": {
        "esc_15ms": { "expected_date": "2026-06-15", "ci_95": ["2026-05-20", "2026-08-01"] },
        "hsct_25ms": { ... },
        "population_p25_36ms": { ... }
      }
    },
    ...
  }
}
```

## Register in _theme.py
{"id": "forecast", "file": "rux_forecast.html", "title": "Rux Forecast", "group": "Clinical"}
