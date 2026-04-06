# Build Task: analysis/analyze_comparative_temperature.py

## What to build
Systematic temperature deviation analysis for both patients. Validated by Mitchell's +13C flight day. Can flag illness before symptoms appear.

## Output
- `reports/comparative_temperature_analysis.html`
- `reports/comparative_temperature_metrics.json`

## Data sources
- oura_sleep.temperature_delta (per-night skin temp deviation from baseline)
- oura_readiness.temperature_deviation (daily readiness temp component)
- Both patients have this data

## Core analyses

### 1. Temperature baseline and variability
- Per patient: mean, median, SD, IQR of temperature_delta
- Normal range: typically -0.5 to +0.5 C
- Rolling 7-day and 30-day means

### 2. Anomaly detection
- Flag days with |temp_delta| > 2 SD
- Classify: fever (>+1.0C), mild elevation (+0.5 to +1.0), normal (-0.5 to +0.5), hypothermia (<-1.0C)
- Cross-reference with other metrics on anomaly days (was HR also elevated? Was sleep disrupted?)

### 3. Known event validation
- Mitchell May 4, 2022: +13C (confirmed: 24hr international flight with mask)
- How many other temp spikes exist? What caused them?
- Henrik: any temp spikes correlating with acute event (Feb 9)?

### 4. Early illness detection model
- Look at 48-72 hours BEFORE known anomaly days
- Does temperature start rising 1-2 days before other metrics crash?
- Compute lead/lag correlation: temp[N] vs readiness[N+1], temp[N] vs HR[N+1]
- If temp rises predict next-day decline, this is a clinically useful early warning

### 5. Pre/post Ruxolitinib (Henrik)
- Rux is anti-inflammatory — does it reduce temperature variability?
- Compare temp SD pre vs post Rux

### 6. Henrik vs Mitchell comparison
- Distribution comparison
- Variability comparison (who runs hotter/colder?)
- Response patterns to stress

## Visualizations
1. Dual timeline with anomaly markers
2. Temperature distribution (violin plots + population norm band)
3. Pre/post Rux temp comparison
4. Predictive value: temp deviation vs next-day readiness scatter
5. Anomaly calendar heatmap (like GitHub contribution graph)

## Notes
- Use _comparative_utils for loading, normalization, comparison
- Temperature_delta can be NULL for some days — handle gracefully
- Mitchell has many data gaps — temperature baseline resets after long gaps
