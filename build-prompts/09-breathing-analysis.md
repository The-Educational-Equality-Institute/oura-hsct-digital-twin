# Build Task: analysis/analyze_comparative_breathing.py

## What to build
Breathing rate analysis for both patients. Elevated breathing rate is an early marker for respiratory complications post-HSCT (BOS — Bronchiolitis Obliterans Syndrome).

## Output
- `reports/comparative_breathing_analysis.html`
- `reports/comparative_breathing_metrics.json`

## Data sources
- oura_sleep.breath_average (both patients have this)
- oura_sleep_periods.average_breath (more detailed, per-period)
- Existing SpO2/BOS analysis in analyze_oura_spo2_trend.py (reference for BOS thresholds)
- config.py has BDI thresholds: BDI_NORMAL=5, BDI_MILD=15, BDI_MODERATE=30

## Core analyses

### 1. Breathing rate trends (both patients)
- Daily breathing rate over time
- 7-day and 30-day rolling averages
- Monthly averages for long-term trend
- Normal range: 12-20 breaths/min during sleep

### 2. Henrik-specific: BOS screening
- Post-HSCT patients at risk for BOS (respiratory GVHD)
- Track: is breathing rate trending upward?
- Combine with SpO2 data if available
- Flag if breath_average > 18 (elevated) or trending up > 0.02/day

### 3. Pre/post Ruxolitinib comparison
- Mann-Whitney U test for breathing rate pre vs post Rux
- Ruxolitinib can reduce inflammation — does it improve breathing?

### 4. Henrik vs Mitchell comparison
- Breathing rate distributions
- Mitchell as healthy-ish reference (post-stroke, no respiratory issues)
- Z-score normalization for fair comparison

### 5. Breathing-HRV coupling
- Correlation between breath rate and HRV
- Respiratory sinus arrhythmia: higher HRV often correlates with lower breath rate
- Is this coupling intact in Henrik? (autonomic dysfunction may disrupt it)

### 6. Anomaly detection
- Flag nights with breath_average > 2 SD above personal mean
- Cross-reference with other anomaly days

## Visualizations
1. Dual timeline (both patients)
2. Distribution comparison (violin plots)
3. Pre/post Rux comparison for Henrik
4. Breathing-HRV scatter plot
5. BOS risk indicator panel

## Notes
- Use _comparative_utils for loading, normalization, comparison
- NEVER use add_vline with annotation_text
- oura_readiness.resting_heart_rate is a SCORE, not bpm
- Henrik oura_sleep hr fields are NULL — use sleep_periods for HR
